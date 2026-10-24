#!/usr/bin/env python
"""Collect no-memory TRAIN rollouts for one opening sample: every task, every trial seed, no memories.

    python collect_rollouts.py --agent qwen32b --domain retail --opening 0 --url http://127.0.0.1:8225

Each task of `<data>/tau2/openings/<domain>/train_op<j>.parquet` is simulated once per trial seed
through a running tau2_service (memcomposer/tau2/tau2_service.py) with an empty selection, so the
agent sees tau2's stock prompt. The service writes full transcripts to its MEMSEL_SIM_DUMP file
(`sims.jsonl`); this script writes one summary row per (task, seed) to `k0.op<j>.jsonl` in the
rollout directory, including the simulations that failed (`error` set), which select_memories.py
counts separately. Every request is sent with dedup off, so every simulation actually runs.

Re-running continues: (task, seed) cells already on disk with a reward are skipped; failed cells are
retried. collect.sh starts the service and runs the four openings side by side.
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import time
import urllib.request
from pathlib import Path

import pandas as pd

import paths
from agents import AGENTS

# tau2's per-trial seeds, drawn exactly as memcomposer/reward.py draws them.
TAU2_SEED = int(os.environ.get('MEMSEL_TAU2_SEED', '300'))


def trial_seeds(n):
    r = random.Random(TAU2_SEED)
    return [r.randint(0, 1000000) for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--agent', required=True, choices=sorted(AGENTS))
    ap.add_argument('--domain', required=True, choices=paths.DOMAINS)
    ap.add_argument('--opening', type=int, required=True, help='opening sample j: train_op<j>.parquet')
    ap.add_argument('--url', required=True, help='tau2_service base URL, serving --domain')
    ap.add_argument('--out-dir', default=None, help='default: <data>/tau2/rollouts/<agent>/<domain>')
    ap.add_argument('--trials', type=int, default=2, help='trial seeds per task')
    ap.add_argument('--limit', type=int, default=0, help='first N tasks only (smoke test); 0 = all')
    ap.add_argument('--concurrency', type=int, default=40)
    ap.add_argument('--timeout', type=int, default=1800)
    a = ap.parse_args()

    d = pd.read_parquet(paths.openings(a.domain, a.opening))
    tasks = [(str(e['task_id']), e['opening']) for e in d['extra_info']]
    assert tasks and len({t for t, _ in tasks}) == len(tasks), 'tasks must be non-empty and unique'
    if a.limit:
        tasks = tasks[:a.limit]
    seeds = trial_seeds(a.trials)

    # The service is bound to one domain at start-up; grading against another domain's task set
    # would silently produce zeros, so refuse to start.
    h = json.loads(urllib.request.urlopen(f'{a.url}/health', timeout=30).read())
    assert h.get('domain') == a.domain, f'service at {a.url} serves {h.get("domain")!r}, not {a.domain!r}'
    print(f'{a.agent}/{a.domain} opening {a.opening}: {len(tasks)} tasks x seeds {seeds}; '
          f'service {a.url} ({h["tasks"]} tasks)', flush=True)

    out_dir = Path(a.out_dir) if a.out_dir else paths.rollouts(a.agent, a.domain)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f'k0.op{a.opening}.jsonl'
    done = set()
    if out.exists():
        for line in open(out):
            try:
                j = json.loads(line)
                if j.get('reward') is not None:          # a failed cell is not done; a re-run redoes it
                    done.add((j['task_id'], j['seed']))
            except Exception:
                pass

    def run(task_id, opening, seed):
        payload = {'task_id': task_id, 'opening': opening, 'folders': [], 'seed': seed, 'dedup': False}
        req = urllib.request.Request(f'{a.url}/run', data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
        t0 = time.time()
        for _ in range(2):
            try:
                r = json.loads(urllib.request.urlopen(req, timeout=a.timeout).read())
                if r.get('reward') is not None:
                    return r
                err = r.get('error') or 'no reward'
            except Exception as e:
                err = f'{type(e).__name__}: {e}'
            time.sleep(5)
        return {'reward': None, 'error': str(err)[:300], 'secs': round(time.time() - t0, 1)}

    jobs = [(t, op, s) for t, op in tasks for s in seeds if (t, s) not in done]
    print(f'{len(jobs)} simulations to run ({len(done)} already on disk) -> {out}', flush=True)
    t_start, n_ok, n_err, passed = time.time(), 0, 0, 0
    with open(out, 'a') as fh, cf.ThreadPoolExecutor(a.concurrency) as ex:
        futs = {ex.submit(run, t, op, s): (t, s) for t, op, s in jobs}
        for i, fu in enumerate(cf.as_completed(futs), 1):
            t, s = futs[fu]
            r = fu.result()
            row = {'k': 0, 'task_id': t, 'seed': s, 'folders': [], 'reward': r.get('reward'),
                   'termination': r.get('termination'), 'n_messages': r.get('n_messages'),
                   'secs': r.get('secs'), 'agent_base': r.get('agent_base'), 'served': r.get('dedup'),
                   'error': r.get('error')}
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
            fh.flush()
            if r.get('reward') is None:
                n_err += 1
            else:
                n_ok += 1
                passed += int(r['reward'] >= 1)
            print(f'  [{i}/{len(jobs)}] task {t} seed {s} reward {r.get("reward")} {r.get("secs")}s'
                  + (f'  ERROR {r.get("error")}' if r.get('reward') is None else ''), flush=True)
    print(f'done in {(time.time() - t_start) / 60:.1f} min: ok {n_ok}, failed {n_err}, passed {passed}', flush=True)


if __name__ == '__main__':
    main()
