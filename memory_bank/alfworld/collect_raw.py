#!/usr/bin/env python
"""Collect raw no-memory trajectories over a split, one attempt per game. No review, no labels.

    python collect_raw.py --split train --run-id train_full_s20 \
        --out <data>/alfworld/rollouts/ministral8b/train_full_s20.jsonl

This is the raw material the memory base is built from later. Games are played exactly as in a
no-memory evaluation (`runner.run_one` with no memories); only what is kept differs.

- Every row carries a `run` block (model, backend, temperature, prompt hash, ...), so a file is
  self-describing and two runs can never be silently mixed.
- Games run in a seeded shuffle, so a run stopped early is still spread evenly over task types.
- Resumable: task_ids already in `--out` are skipped. Episodes that raised go to
  `<out>.errors.jsonl` instead of `--out`, so a resume retries them.
"""
import argparse
import collections
import datetime
import hashlib
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import paths                                                            # noqa: E402
sys.path.insert(0, str(paths.RUNTIME))

from llm import default_chat                                            # noqa: E402
from runner import index, run_one                                       # noqa: E402

PROMPT_FILE = paths.RUNTIME / 'alfworld_prompt.py'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train')
    ap.add_argument('--task-ids-file', default=None,
                    help='file with one task_id per line; overrides --split')
    ap.add_argument('--out', required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--model', default=None)
    ap.add_argument('--temp', type=float, default=0.4)
    ap.add_argument('--max-steps', type=int, default=20)
    ap.add_argument('--max-tokens', type=int, default=2048)
    ap.add_argument('--history-length', type=int, default=10)
    ap.add_argument('--concurrency', type=int, default=8)
    ap.add_argument('--shuffle-seed', type=int, default=0)
    a = ap.parse_args()

    idx = index()
    if a.task_ids_file:
        ids = [l.strip() for l in Path(a.task_ids_file).read_text().splitlines() if l.strip()]
    else:
        ids = sorted(t for t, r in idx.items() if r['split'] == a.split)
    random.Random(a.shuffle_seed).shuffle(ids)

    out = Path(a.out)
    err_path = out.with_name(out.stem + '.errors.jsonl')
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with out.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)['task_id'])
                except Exception:
                    pass
    todo = [t for t in ids if t not in done]

    chat = default_chat(a.model, max_tokens=a.max_tokens, temperature=a.temp)
    run_meta = {
        'run_id': a.run_id,
        'model': chat.model,
        'backend': 'local' if os.environ.get('ALFW_AGENT_BASE') else 'bedrock',
        'temperature': a.temp,
        'max_steps': a.max_steps,
        'max_tokens': a.max_tokens,
        'history_length': a.history_length,
        'max_format_retries': 2,
        'prompt_sha': hashlib.sha256(PROMPT_FILE.read_bytes()).hexdigest()[:16],
        'shuffle_seed': a.shuffle_seed,
        'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
    }
    print(f'games {len(ids)}; already done {len(done)}, to run {len(todo)}')
    print(f"executor {chat.model} ({run_meta['backend']})  temp {a.temp}  "
          f"max_steps {a.max_steps}  concurrency {a.concurrency}  prompt {run_meta['prompt_sha']}",
          flush=True)
    if not todo:
        print('nothing to do')
        return

    lock = threading.Lock()
    fh, eh = out.open('a'), err_path.open('a')
    t0 = time.time()
    c = {'n': 0, 'won': 0, 'err': 0}

    def work(tid):
        r = run_one(tid, chat, memories=None, max_steps=a.max_steps,
                    history_length=a.history_length, temperature=a.temp)
        r['attempt'] = 0
        r['run'] = run_meta
        with lock:
            (eh if r['error'] else fh).write(json.dumps(r, ensure_ascii=False) + '\n')
            (eh if r['error'] else fh).flush()
            c['n'] += 1
            c['won'] += r['won'] if not r['error'] else 0
            c['err'] += bool(r['error'])
            if c['n'] % 25 == 0 or c['n'] == len(todo):
                el = time.time() - t0
                ok = c['n'] - c['err']
                print(f"  {c['n']:4d}/{len(todo)}  won {c['won']}/{ok} = "
                      f"{c['won']/max(1, ok):.3f}  errors {c['err']}  "
                      f"[{el/60:.1f} min, eta {el/c['n']*(len(todo)-c['n'])/60:.0f} min]",
                      flush=True)

    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        list(ex.map(work, todo))
    fh.close()
    eh.close()

    rows = [json.loads(l) for l in out.open()]
    n = len(rows)
    won = sum(r['won'] for r in rows)
    print(f'\n=== {a.run_id}  {n} games  ({c["err"]} errors this session -> {err_path.name}) ===')
    print(f'success {won}/{n} = {won/max(1, n):.4f}')
    by = collections.defaultdict(list)
    for r in rows:
        by[r['task_type']].append(r['won'])
    for k in sorted(by):
        print(f'  {k:32s} {sum(by[k]):4d}/{len(by[k]):4d} = {sum(by[k])/len(by[k]):.3f}')
    print(f'LLM calls {chat.calls}  tokens in/out {chat.prompt_tokens}/{chat.completion_tokens}'
          f'  wall {(time.time()-t0)/60:.1f} min  -> {out}')


if __name__ == '__main__':
    main()
