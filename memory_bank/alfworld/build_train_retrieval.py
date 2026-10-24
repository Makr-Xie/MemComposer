#!/usr/bin/env python
"""Sample the training queries and FREEZE their retrieval pools.

    python build_train_retrieval.py --model ministral8b --n 256
        -> retrieval/<model>/train_q<n>_top20.json  (+ .sha256, + train_q<n>_embeddings.npz)

Mirrors the tau2 training data: every query gets `--windows` rank windows of `--window-size`
(window 0 = ranks 1-10, window 1 = ranks 11-20), and the query's own task -- here its taskdir
(group_id) -- is excluded from its pool. Nothing else is excluded (as in tau2: no floorplan
exclusion, no difficulty filter).

Retrieval is build_retrieval.py's, unchanged: the key vectors are read from the valid_unseen npz it
wrote (after checking the memory base is byte-identical), so run it first; same query instruction,
cosine, rounded to 6 places, ties broken by sha256(seed|query_task_id|memory_id), at most --cap per
source taskdir.

Queries: --n train games, at most one per taskdir, allocated to task types in proportion to the
train split (largest remainder), then picked by seeded hash.
"""
import argparse
import collections
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from build_retrieval import INSTRUCTION, MODEL, embed, goal_of, tiebreak

import paths



def h(*parts):
    return hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()


def sample_queries(tasks, n, seed):
    train = [r for r in tasks if r['split'] == 'train']
    by_type = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in train:
        by_type[r['task_type']][r['group_id']].append(r)
    total = len(train)
    quota = {t: n * sum(map(len, g.values())) / total for t, g in by_type.items()}
    alloc = {t: int(q) for t, q in quota.items()}
    for t in sorted(quota, key=lambda t: (-(quota[t] - alloc[t]), t))[:n - sum(alloc.values())]:
        alloc[t] += 1
    out = []
    for t in sorted(by_type):
        groups = sorted(by_type[t], key=lambda g: h(seed, 'group', g))
        assert len(groups) >= alloc[t], (t, len(groups), alloc[t])
        for g in groups[:alloc[t]]:
            out.append(min(by_type[t][g], key=lambda r: h(seed, 'trial', r['task_id'])))
    return sorted(out, key=lambda r: r['task_id']), alloc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='memory base: ministral8b | qwen3-4b')
    ap.add_argument('--n', type=int, default=256, help='number of train queries')
    ap.add_argument('--windows', type=int, default=2)
    ap.add_argument('--window-size', type=int, default=10)
    ap.add_argument('--cap', type=int, default=2, help='max memories per source taskdir')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--url', default='http://127.0.0.1:8012/v1')
    a = ap.parse_args()
    K = a.windows * a.window_size

    base = paths.bank(a.model)
    mems = sorted((json.loads(p.read_text()) for p in base.glob('*/memory.json')), key=lambda m: m['memory_id'])
    fp = hashlib.sha256()
    for p in sorted(base.glob('*/memory.json')):
        fp.update(p.read_bytes())
    eval_pool = json.loads((paths.retrieval(a.model) / 'valid_unseen_top10.json').read_bytes())
    assert eval_pool['memory_base_sha256'] == fp.hexdigest(), 'memory base changed since the eval pool was frozen'

    npz = np.load(paths.retrieval(a.model) / 'valid_unseen_embeddings.npz')
    keys = [str(k) for k in npz['keys']]
    assert keys == sorted({m['key'] for m in mems}), 'key set differs from the frozen eval run'
    row = {k: i for i, k in enumerate(keys)}
    M = npz['key_vecs'][[row[m['key']] for m in mems]]

    tasks = [json.loads(l) for l in open(paths.task_index())]
    queries, alloc = sample_queries(tasks, a.n, a.seed)
    for q in queries:
        q['goal'] = goal_of(q['gamefile'])
    qv = embed([f'Instruct: {INSTRUCTION}\nQuery: {q["goal"]}' for q in queries], a.url)

    # sanity: the server still embeds like the frozen run (re-embed a few keys, compare)
    probe = keys[:: max(1, len(keys) // 8)][:8]
    drift = float(np.max(np.abs(embed(probe, a.url) - npz['key_vecs'][[row[k] for k in probe]])))
    assert drift < 2e-3, f'embedding drift {drift}'

    S = qv @ M.T
    out, stats = [], collections.Counter()
    for qi, q in enumerate(queries):
        order = sorted(range(len(mems)), key=lambda j: (-round(float(S[qi, j]), 6),
                                                        tiebreak(a.seed, q['task_id'], mems[j]['memory_id'])))
        per, hits = collections.Counter(), []
        for j in order:
            m = mems[j]
            if m['group_id'] == q['group_id'] or per[m['group_id']] >= a.cap:
                continue
            per[m['group_id']] += 1
            hits.append({'memory_id': m['memory_id'], 'score': round(float(S[qi, j]), 6), 'key': m['key'],
                         'config': m['config'], 'won': m['outcome']['won'], 'group_id': m['group_id'],
                         'floor_plan': m['floor_plan']})
            if len(hits) == K:
                break
        assert len(hits) == K
        qcfg = q['group_id'].rsplit('-', 1)[0]
        stats['queries'] += 1
        for w in range(a.windows):
            ws = hits[w * a.window_size:(w + 1) * a.window_size]
            stats[f'w{w}_same_goal'] += sum(x['key'] == q['goal'] for x in ws)
            stats[f'w{w}_same_config'] += sum(x['config'] == qcfg for x in ws)
            stats[f'w{w}_same_floorplan'] += sum(x['floor_plan'] == q['floor_plan'] for x in ws)
            stats[f'w{w}_won'] += sum(x['won'] for x in ws)
        out.append({'task_id': q['task_id'], 'goal': q['goal'], 'config': qcfg, 'group_id': q['group_id'],
                    'floor_plan': q['floor_plan'], 'task_type': q['task_type'],
                    'excluded_group_ids': [q['group_id']], 'hits': hits})

    n = stats['queries']
    doc = {
        'version': f'{a.model}-train-q{a.n}-top{K}',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'embedding_model': MODEL, 'query_instruction': INSTRUCTION, 'document_instruction': None,
        'key': 'memory goal text', 'query': 'train game goal text',
        'key_vectors_from': f'retrieval/{a.model}/valid_unseen_embeddings.npz', 'embedding_drift_check': drift,
        'memory_base': f'memory_banks/{a.model}', 'memory_base_sha256': fp.hexdigest(),
        'n_memories': len(mems), 'n_distinct_keys': len(keys),
        'tie_break': f'sha256(seed|query_task_id|memory_id), seed={a.seed}', 'cap_per_taskdir': a.cap,
        'exclude': 'own taskdir (group_id) only', 'k': K, 'windows': a.windows, 'window_size': a.window_size,
        'query_sampling': {'n': a.n, 'seed': a.seed, 'per_taskdir': 1, 'per_task_type': alloc},
        'stats': {'queries': n, **{k: round(v / n, 2) for k, v in sorted(stats.items()) if k != 'queries'}},
        'tasks': out,
    }
    dst = paths.retrieval(a.model) / f'train_q{a.n}_top{K}.json'
    if dst.exists():
        raise SystemExit(f'{dst} exists: the pool is frozen. Write a new version instead of overwriting.')
    blob = json.dumps(doc, ensure_ascii=False, indent=1).encode()
    dst.write_bytes(blob)
    (dst.parent / (dst.name + '.sha256')).write_text(hashlib.sha256(blob).hexdigest() + f'  {dst.name}\n')
    np.savez(dst.parent / f'train_q{a.n}_embeddings.npz',
             query_ids=np.array([q['task_id'] for q in queries]), query_vecs=qv)
    print(json.dumps({k: doc[k] for k in ('version', 'n_memories', 'query_sampling', 'stats')}, indent=1))
    print(f'-> {dst}  sha256 {hashlib.sha256(blob).hexdigest()[:16]}')


if __name__ == '__main__':
    main()
