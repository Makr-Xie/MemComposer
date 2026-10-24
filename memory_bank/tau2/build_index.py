#!/usr/bin/env python
"""Build the query-keyed embedding index over a memory bank.

The key is the customer's opening request only. The document side is embedded raw, with no
instruction prefix; only queries get QUERY_INSTRUCTION (see retrieve.py). Vectors are
L2-normalised, so cosine is a dot product.

    <domain>/
      index.npz              embeddings (N,4096) float32 + folders + sim_ids, row-aligned
      index_manifest.json    model, instructions, measured separability, limitations

Rows follow folder sort order, matching `memory.json.retrieval_key.index_row` (set by render.py).

Usage: build_index.py --agent ministral retail|telecom|airline [--url http://127.0.0.1:8012/v1]
"""
import argparse, datetime, json, urllib.request
from pathlib import Path
import numpy as np

from agents import AGENTS, add_agent_arg

MODEL = 'Qwen3-Embedding-8B'
QUERY_INSTRUCTION = ('Given a new {domain} customer-service request as the customer first states '
                     'it, retrieve prior agent trajectories that began from a similar request.')


def embed(texts, url, batch=16):
    out = []
    for i in range(0, len(texts), batch):
        req = urllib.request.Request(
            url + '/embeddings', headers={'Content-Type': 'application/json'},
            data=json.dumps({'model': MODEL, 'input': texts[i:i + batch],
                             'encoding_format': 'float'}).encode())
        r = json.loads(urllib.request.urlopen(req, timeout=600).read())
        out.extend(d['embedding'] for d in sorted(r['data'], key=lambda d: d['index']))
    v = np.asarray(out, dtype=np.float32)
    assert np.isfinite(v).all(), 'non-finite embedding'
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    add_agent_arg(ap)
    ap.add_argument('domain', choices=['airline', 'retail', 'telecom'])
    ap.add_argument('--url', default='http://127.0.0.1:8012/v1')
    a = ap.parse_args()
    out = AGENTS[a.agent].base / a.domain

    keys = [json.loads(l) for l in open(out / 'query_keys.jsonl')]
    folders = sorted(d.name for d in out.iterdir() if d.is_dir() and d.name.startswith('task'))
    assert [k['folder'] for k in keys] == folders, 'query_keys.jsonl is not in folder order'
    for row, k in enumerate(keys):
        m = json.loads((out / k['folder'] / 'memory.json').read_text())
        assert m['retrieval_key']['index_row'] == row, f"{k['folder']} row mismatch"
        assert m['retrieval_key']['text'] == k['key_text'], f"{k['folder']} key mismatch"

    V = embed([k['key_text'] for k in keys], a.url)
    assert V.shape[0] == len(keys)

    # Separability, measured rather than assumed. Same-task pairs are near-identical by
    # construction; what matters is how far cross-task pairs sit from them.
    tasks = np.array([k['task_id'] for k in keys])
    S = V @ V.T
    iu = np.triu_indices(len(keys), 1)
    same = tasks[iu[0]] == tasks[iu[1]]
    sim_same, sim_cross = S[iu][same], S[iu][~same]
    # "Is the nearest neighbour from the same task" is only answerable for records whose task
    # has a second memory; a task with a single memory cannot match by construction.
    import collections
    cnt = collections.Counter(tasks.tolist())
    np.fill_diagonal(S, -np.inf)
    eligible = [i for i in range(len(keys)) if cnt[tasks[i]] > 1]
    nn_same = int(sum(tasks[int(np.argmax(S[i]))] == tasks[i] for i in eligible))
    by_text = collections.defaultdict(set)
    for t, k in zip(tasks, keys):
        by_text[k['key_text']].add(t)
    shared = {tx: sorted(ts) for tx, ts in by_text.items() if len(ts) > 1}
    tasks_colliding = sorted({t for ts in shared.values() for t in ts})

    np.savez_compressed(out / 'index.npz', embeddings=V.astype(np.float32),
                        folders=np.array(folders), sim_ids=np.array([k['sim_id'] for k in keys]))
    manifest = {
        'domain': a.domain,
        'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'corpus': str(out / 'query_keys.jsonl'),
        'count': len(keys),
        'tasks': int(len(set(tasks))),
        'key_definition': 'the customer\'s opening request only; no execution, outcome or label',
        'document_instruction': None,
        'query_instruction': QUERY_INSTRUCTION.format(domain=a.domain),
        'embedding_model': MODEL,
        'embedding_url': a.url,
        'dimension': int(V.shape[1]),
        'dtype': 'float32',
        'normalization': 'L2',
        'similarity': 'cosine (dot product of unit vectors)',
        'row_order': 'sorted folder name; equals memory.json.retrieval_key.index_row',
        'key_chars': {'min': int(min(len(k['key_text']) for k in keys)),
                      'max': int(max(len(k['key_text']) for k in keys)),
                      'mean': int(np.mean([len(k['key_text']) for k in keys]))},
        'separability': {'same_task_mean': round(float(sim_same.mean()), 4),
                         'same_task_min': round(float(sim_same.min()), 4),
                         'cross_task_mean': round(float(sim_cross.mean()), 4),
                         'cross_task_max': round(float(sim_cross.max()), 4)},
        'norm_min': round(float(np.linalg.norm(V, axis=1).min()), 7),
        'norm_max': round(float(np.linalg.norm(V, axis=1).max()), 7),
        'nearest_neighbour_same_task': {'matched': nn_same, 'of': len(eligible),
                                        'note': 'records whose task has a second memory; the '
                                                'other records cannot match by construction'},
        'identical_openings_across_tasks': {
            'lines': len(shared), 'tasks': len(tasks_colliding),
            'examples': [{'tasks': ts, 'text': tx[:160]}
                         for tx, ts in sorted(shared.items(), key=lambda kv: -len(kv[1]))[:3]]},
        'limitations': [
            'The key carries nothing that only appears later in the conversation, so it cannot '
            'separate tasks whose opening requests match but whose correct behaviour differs.',
            f'{len(shared)} opening lines are shared by more than one task, covering '
            f'{len(tasks_colliding)} of {len(set(tasks))} tasks; within such a group the key '
            'carries no information at all and ordering is arbitrary.',
            'Memories from the same task have near-identical keys by construction; ordering '
            'within a task is not meaningful. Discrimination is left to the selector.',
        ],
    }
    (out / 'index_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'{a.domain}: {len(keys)} vectors {V.shape} -> {out / "index.npz"}')
    print(f'  same-task   mean {sim_same.mean():.4f}  min {sim_same.min():.4f}')
    print(f'  cross-task  mean {sim_cross.mean():.4f}  max {sim_cross.max():.4f}')
    print(f'  nearest neighbour same task: {nn_same}/{len(eligible)} '
          f'(of the records whose task has a sibling)')
    print(f'  opening lines shared across tasks: {len(shared)}, covering '
          f'{len(tasks_colliding)}/{len(set(tasks))} tasks')


if __name__ == '__main__':
    main()
