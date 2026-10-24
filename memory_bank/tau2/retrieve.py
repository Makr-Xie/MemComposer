#!/usr/bin/env python
"""Embed sampled openings and retrieve top-k against a memory bank.

    python retrieve.py --agent ministral --domain retail --openings openings.json \
        --out retrieval.json --top-k 10 --cap 2

Retrieval rules for building training pools:
  * `--exclude-self` (default on)  drop every memory whose task_id is the query's own task.
    A task's own memories carry its `purpose` as the digest's `Task:` line and, for failures,
    an explicit `SHOULD HAVE:` -- the answer, verbatim.
  * `--twin-threshold`  also drop tasks whose retrieval key is near-identical to the query
    task's (two tasks can share an opening yet have opposite correct answers). 0 disables it.
  * `--cap N`  at most N memories from any one source task, applied in retrieval order.

Vectors must come from the same model and pooling as the bank's `index.npz`
(Qwen3-Embedding-8B, LAST pooling, L2-normalised), so call the same embedding server the
index was built with.
"""
import argparse, json, urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

from agents import AGENTS, add_agent_arg

# retail ids are numeric strings, telecom ids are scenario strings; sort by whichever applies so
# one ordering works for both.
def tkey(t):
    return (0, int(t), '') if str(t).isdigit() else (1, 0, str(t))


def embed(texts, url, model, instruction, batch=16):
    """Query-side embedding: `Instruct: ...\\nQuery: ...` with the manifest's query instruction."""
    out = []
    for i in range(0, len(texts), batch):
        chunk = [f'Instruct: {instruction}\nQuery: {t}' for t in texts[i:i + batch]]
        req = urllib.request.Request(
            url + '/embeddings', headers={'Content-Type': 'application/json'},
            data=json.dumps({'model': model, 'input': chunk, 'encoding_format': 'float'}).encode())
        r = json.loads(urllib.request.urlopen(req, timeout=600).read())
        out.extend(d['embedding'] for d in sorted(r['data'], key=lambda d: d['index']))
    v = np.asarray(out, dtype=np.float32)
    assert np.isfinite(v).all()
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def load_base(base):
    """Memory base rows, aligned to index.npz."""
    z = np.load(base / 'index.npz', allow_pickle=True)
    rows = []
    for i, (folder, sim_id) in enumerate(zip(z['folders'], z['sim_ids'])):
        m = json.loads((base / str(folder) / 'memory.json').read_text())
        assert m['retrieval_key']['index_row'] == i
        rows.append({'row': i, 'folder': str(folder), 'sim_id': str(sim_id),
                     'task_id': str(m['task_id']), 'trial': m['trial'],
                     'label_kind': m['label_kind'], 'key_text': m['retrieval_key']['text']})
    return z['embeddings'].astype(np.float32), rows


def main():
    p = argparse.ArgumentParser()
    add_agent_arg(p)
    p.add_argument('--domain', default='airline', choices=['airline', 'retail', 'telecom'])
    p.add_argument('--openings', required=True)
    p.add_argument('--top-k', type=int, default=10)
    p.add_argument('--cap', type=int, default=2, help='max memories per source task; 0 = off')
    p.add_argument('--exclude-self', action='store_true', default=True)
    p.add_argument('--no-exclude-self', dest='exclude_self', action='store_false')
    p.add_argument('--twin-threshold', type=float, default=0.0,
                   help='also exclude tasks whose key is this similar to the query task; 0 = off '
                        '(the released pools). Only affects training queries: a test task is not '
                        'in the memory bank, so nothing is excluded for it.')
    p.add_argument('--url', default='http://127.0.0.1:8012/v1')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    base = AGENTS[args.agent].base / args.domain
    man = json.loads((base / 'index_manifest.json').read_text())
    doc_vecs, rows = load_base(base)

    # Task-level twin detection, on the document side: one representative key per task.
    rep = {}
    for r in rows:
        rep.setdefault(r['task_id'], r['row'])
    tasks = sorted(rep, key=tkey)
    tsim = doc_vecs[[rep[t] for t in tasks]] @ doc_vecs[[rep[t] for t in tasks]].T
    twins = {t: {u for j, u in enumerate(tasks)
                 if u != t and tsim[i, j] >= args.twin_threshold}
             for i, t in enumerate(tasks)} if args.twin_threshold else {t: set() for t in tasks}

    data = json.loads(Path(args.openings).read_text())
    flat = [(tid, i, txt) for tid, lst in data['openings'].items() for i, txt in enumerate(lst)]
    qv = embed([t for _, _, t in flat], args.url, man['embedding_model'], man['query_instruction'])

    results = []
    for (tid, i, txt), v in zip(flat, qv):
        banned = ({tid} | twins.get(tid, set())) if args.exclude_self else set()
        scores = doc_vecs @ v
        hits, seen = [], Counter()
        for j in np.argsort(-scores):
            r = rows[int(j)]
            if r['task_id'] in banned:
                continue
            if args.cap and seen[r['task_id']] >= args.cap:
                continue
            seen[r['task_id']] += 1
            hits.append({'rank': len(hits) + 1, 'score': round(float(scores[j]), 4),
                         'sim_id': r['sim_id'], 'folder': r['folder'],
                         'task_id': r['task_id'], 'trial': r['trial'],
                         'label_kind': r['label_kind']})
            if len(hits) >= args.top_k:
                break
        results.append({'task_id': tid, 'sample': i, 'query': txt,
                        'excluded_tasks': sorted(banned, key=tkey), 'hits': hits})

    Path(args.out).write_text(json.dumps(
        {'embedding_model': man['embedding_model'], 'query_instruction': man['query_instruction'],
         'top_k': args.top_k, 'cap': args.cap, 'exclude_self': args.exclude_self,
         'twin_threshold': args.twin_threshold,
         'domain': args.domain,
         'twins': {t: sorted(v, key=tkey) for t, v in twins.items() if v},
         'source': str(args.openings), 'results': results}, indent=1, ensure_ascii=False))
    print(f'{len(results)} queries over {len(data["openings"])} tasks -> {args.out}')
    tw = {t: sorted(v, key=tkey) for t, v in twins.items() if v}
    print(f'twin pairs excluded: {len(tw)} tasks' + ('' if len(tw) > 6 else f' {tw}'))
    import statistics as st
    print(f'hits per query: {st.mean(len(r["hits"]) for r in results):.1f} '
          f'(min {min(len(r["hits"]) for r in results)}); '
          f'distinct source tasks per pool: '
          f'{st.mean(len({h["task_id"] for h in r["hits"]}) for r in results):.1f}')


if __name__ == '__main__':
    main()
