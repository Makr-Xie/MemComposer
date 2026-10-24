#!/usr/bin/env python
"""Compute and FREEZE the top-10 retrieval pool for every eval task.

    python build_retrieval.py --model ministral8b --split valid_unseen
        -> retrieval/<model>/<split>_top10.json  (+ .sha256)

Run once. Every evaluation arm -- top-k, random-k, reordering, the selector -- reads this file and
chooses only among these 10; nothing re-retrieves at eval time. A rebuilt memory base gets a new
version of the file, never an in-place update.

Retrieval, tau2-style (memory_bank/tau2/build_index.py):
- key   = the memory's goal text, embedded raw (no instruction);
- query = the eval game's goal, wrapped `Instruct: <instruction>\\nQuery: <goal>`;
- Qwen3-Embedding-8B, L2-normalised, cosine.
ALFWorld goals are templates, so many memories share a key byte for byte and tie exactly. Ties are
broken by a seeded hash of (seed, query task_id, memory_id): reproducible, and unrelated to file
order. At most `--cap` memories per source taskdir (one room's trials), as tau2's `--cap 2`.
"""
import argparse
import collections
import datetime
import hashlib
import json
import re
import urllib.request
from pathlib import Path

import numpy as np

import paths

MODEL = 'Qwen3-Embedding-8B'
INSTRUCTION = 'Given a household task instruction, retrieve past attempts at similar household tasks'
GOAL_RE = re.compile(r'"rhs":\s*"Your task is to:\s*(.+?)"')


def embed(texts, url, batch=32):
    out = []
    for i in range(0, len(texts), batch):
        req = urllib.request.Request(url + '/embeddings', headers={'Content-Type': 'application/json'},
                                     data=json.dumps({'model': MODEL, 'input': texts[i:i + batch],
                                                      'encoding_format': 'float'}).encode())
        r = json.loads(urllib.request.urlopen(req, timeout=600).read())
        out.extend(d['embedding'] for d in sorted(r['data'], key=lambda d: d['index']))
    v = np.asarray(out, dtype=np.float32)
    assert np.isfinite(v).all()
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def goal_of(gamefile):
    gr = json.load(open(gamefile))['grammar']
    m = GOAL_RE.search(gr[gr.find('"task"'):])
    return m.group(1).strip()


def tiebreak(seed, qid, mid):
    return hashlib.sha256(f'{seed}|{qid}|{mid}'.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='memory base: ministral8b | qwen3-4b')
    ap.add_argument('--split', default='valid_unseen')
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--cap', type=int, default=2, help='max memories per source taskdir')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--url', default='http://127.0.0.1:8012/v1')
    a = ap.parse_args()

    base = paths.bank(a.model)
    mems = sorted((json.loads(p.read_text()) for p in base.glob('*/memory.json')), key=lambda m: m['memory_id'])
    fp = hashlib.sha256()
    for p in sorted(base.glob('*/memory.json')):
        fp.update(p.read_bytes())
    tasks = [json.loads(l) for l in open(paths.task_index())]
    queries = sorted((r for r in tasks if r['split'] == a.split), key=lambda r: r['task_id'])
    for q in queries:
        q['goal'] = goal_of(q['gamefile'])

    keys = sorted({m['key'] for m in mems})
    kv = embed(keys, a.url)
    row = {k: i for i, k in enumerate(keys)}
    M = kv[[row[m['key']] for m in mems]]
    qv = embed([f'Instruct: {INSTRUCTION}\nQuery: {q["goal"]}' for q in queries], a.url)
    S = qv @ M.T

    out, stats = [], collections.Counter()
    for qi, q in enumerate(queries):
        order = sorted(range(len(mems)), key=lambda j: (-round(float(S[qi, j]), 6),
                                                        tiebreak(a.seed, q['task_id'], mems[j]['memory_id'])))
        per, hits = collections.Counter(), []
        for j in order:
            m = mems[j]
            if per[m['group_id']] >= a.cap:
                continue
            assert m['floor_plan'] != q['floor_plan'] or a.split != 'valid_unseen', 'room leak'
            per[m['group_id']] += 1
            hits.append({'memory_id': m['memory_id'], 'score': round(float(S[qi, j]), 6), 'key': m['key'],
                         'config': m['config'], 'won': m['outcome']['won'], 'group_id': m['group_id']})
            if len(hits) == a.k:
                break
        top = hits[0]['score']
        tied_total = int(np.sum(np.round(S[qi], 6) == top))
        qcfg = q['group_id'].rsplit('-', 1)[0]
        stats['queries'] += 1
        stats['exact_goal_top1'] += hits[0]['key'] == q['goal']
        stats['same_goal_slots'] += sum(h['key'] == q['goal'] for h in hits)
        stats['same_config_slots'] += sum(h['config'] == qcfg for h in hits)
        stats['won_slots'] += sum(h['won'] for h in hits)
        stats['top_tie_gt_k'] += tied_total > a.k
        stats['no_same_config'] += not any(h['config'] == qcfg for h in hits)
        out.append({'task_id': q['task_id'], 'goal': q['goal'], 'config': qcfg, 'hits': hits,
                    'top_score_tied_memories': tied_total})

    n = stats['queries']
    doc = {
        'version': f'{a.model}-{a.split}-top{a.k}',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'embedding_model': MODEL, 'query_instruction': INSTRUCTION, 'document_instruction': None,
        'key': 'memory goal text', 'query': 'eval game goal text',
        'memory_base': f'memory_banks/{a.model}', 'memory_base_sha256': fp.hexdigest(),
        'n_memories': len(mems), 'n_distinct_keys': len(keys),
        'tie_break': f'sha256(seed|query_task_id|memory_id), seed={a.seed}', 'cap_per_taskdir': a.cap,
        'k': a.k,
        'stats': {'queries': n,
                  'top1_is_exact_goal': stats['exact_goal_top1'],
                  'mean_same_goal_in_topk': round(stats['same_goal_slots'] / n, 2),
                  'mean_same_config_in_topk': round(stats['same_config_slots'] / n, 2),
                  'mean_won_in_topk': round(stats['won_slots'] / n, 2),
                  'queries_where_top_score_ties_exceed_k': stats['top_tie_gt_k'],
                  'queries_with_no_same_config_hit': stats['no_same_config']},
        'tasks': out,
    }
    dst = paths.retrieval(a.model) / f'{a.split}_top{a.k}.json'
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise SystemExit(f'{dst} exists: the pool is frozen. Write a new version instead of overwriting.')
    blob = json.dumps(doc, ensure_ascii=False, indent=1).encode()
    dst.write_bytes(blob)
    (dst.parent / (dst.name + '.sha256')).write_text(hashlib.sha256(blob).hexdigest() + f'  {dst.name}\n')
    np.savez(dst.parent / f'{a.split}_embeddings.npz', keys=np.array(keys), key_vecs=kv,
             query_ids=np.array([q['task_id'] for q in queries]), query_vecs=qv)
    print(json.dumps({k: doc[k] for k in ('version', 'n_memories', 'n_distinct_keys', 'stats')}, indent=1))
    print(f'-> {dst}  sha256 {hashlib.sha256(blob).hexdigest()[:16]}')


if __name__ == '__main__':
    main()
