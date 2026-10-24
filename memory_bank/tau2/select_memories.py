#!/usr/bin/env python
"""Pick two memories per task from an agent's no-memory train rollouts, by pass rate.

The rule, per task:

    pass rate >= 7/8  ->  two PASSING trajectories
    pass rate <= 1/8  ->  two FAILING trajectories
    in between        ->  one passing and one failing

Pass rate is computed over the task's VALID simulations only. A simulation that tau2 aborted
(e.g. because the agent emitted tool-call arguments that are not valid JSON) has no transcript
and no grade, so it can be neither counted nor selected. Counting it as a failure could put a
task in the middle bucket with no failing transcript to select. Both bucketings are recorded
per task (`bucket_valid_only`, `bucket_crash_as_fail`) so the choice stays auditable.

Within a bucket the pick is a random sample seeded by (seed, domain). A plain random pick can
take two near-identical trajectories; `--distinct-writes` prefers two picks whose write calls
differ.

The records carry no `label`. Review (label.sh) and the audit top-up (topup.py) are separate
steps; this script only decides which trajectories go in.

Usage:
    select_memories.py --agent ministral                      # all three domains, seed 0
    select_memories.py --agent ministral --domain retail --seed 1 --distinct-writes
"""
import argparse
import collections
import json
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
import paths
from agents import AGENTS, add_agent_arg

ROLLOUTS = None   # set from --agent in main()
OUT = None
DOMAINS = paths.DOMAINS
HIGH, LOW = 7 / 8, 1 / 8


def opening_index(domain):
    """(task_id, opening text) -> which of the four sampled openings it is."""
    import pandas as pd
    idx = {}
    for j in range(4):
        for e in pd.read_parquet(paths.openings(domain, j))['extra_info']:
            idx[(str(e['task_id']), e['opening'])] = j
    return idx


def load(domain):
    """Every simulation that produced a transcript, grouped by task."""
    by = collections.defaultdict(list)
    for line in open(ROLLOUTS.as_posix().format(domain) + '/sims.jsonl'):
        s = json.loads(line)
        by[str(s['task_id'])].append(s)
    return by


def crash_counts(domain):
    """Per task: total attempted sims and how many tau2 aborted (no transcript)."""
    total, crashed = collections.Counter(), collections.Counter()
    for j in range(4):
        for line in open(ROLLOUTS.as_posix().format(domain) + f'/k0.op{j}.jsonl'):
            r = json.loads(line)
            total[str(r['task_id'])] += 1
            crashed[str(r['task_id'])] += bool(r.get('error'))
    return total, crashed


def write_signature(messages):
    """The state-changing calls a trajectory made, arguments normalised -- a lesson fingerprint."""
    out = []
    for m in messages:
        for tc in (m.get('tool_calls') or []):
            fn = tc.get('function') or tc
            name = fn.get('name') or ''
            args = fn.get('arguments')
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            out.append(name + json.dumps(args, sort_keys=True, default=str))
    return tuple(out)


def pick(cands, k, rng, distinct):
    """k records from cands. With `distinct`, prefer pairs with different write signatures."""
    if len(cands) <= k:
        return list(cands)
    pool = list(cands)
    rng.shuffle(pool)
    if not distinct:
        return pool[:k]
    chosen, seen = [], set()
    for c in pool:
        sig = write_signature(c['messages'])
        if sig not in seen:
            chosen.append(c)
            seen.add(sig)
        if len(chosen) == k:
            return chosen
    for c in pool:                       # not enough distinct ones; fill from the rest
        if c not in chosen:
            chosen.append(c)
        if len(chosen) == k:
            break
    return chosen


def run(domain, seed, distinct):
    sims = load(domain)
    total, crashed = crash_counts(domain)
    opidx = opening_index(domain)
    rng = random.Random(f'{seed}|{domain}')
    records, stats = [], collections.Counter()

    for task_id in sorted(sims, key=lambda t: (len(t), t)):
        valid = sims[task_id]
        passed = [s for s in valid if (s.get('reward_info') or {}).get('reward', 0) >= 1]
        failed = [s for s in valid if (s.get('reward_info') or {}).get('reward', 0) < 1]
        rate = len(passed) / len(valid)
        rate_crash = len(passed) / total[task_id]
        bucket = 'high' if rate >= HIGH else ('low' if rate <= LOW else 'mid')
        bucket_crash = ('high' if rate_crash >= HIGH else 'low' if rate_crash <= LOW else 'mid')
        want = {'high': (2, 0), 'low': (0, 2), 'mid': (1, 1)}[bucket]
        got = pick(passed, want[0], rng, distinct) + pick(failed, want[1], rng, distinct)
        stats[bucket] += 1
        if len(got) < sum(want):
            stats['short'] += 1

        why = {'high': f'rule: pass rate {len(passed)}/{len(valid)} >= 7/8, two passing trajectories',
               'low': f'rule: pass rate {len(passed)}/{len(valid)} <= 1/8, two failing trajectories',
               'mid': f'rule: pass rate {len(passed)}/{len(valid)} in between, one passing and one failing'}[bucket]
        for s in got:
            ri = s.get('reward_info') or {}
            records.append({
                'sim_id': s['id'],
                'task_id': task_id,
                'domain': domain,
                'outcome': 'pass' if ri.get('reward', 0) >= 1 else 'fail',
                'reward': ri.get('reward'),
                'termination_reason': s.get('termination_reason'),
                'seed': s['_meta']['seed'],
                'opening_sample': opidx.get((task_id, s['_meta']['opening'])),
                'opening': s['_meta']['opening'],
                'task_pass_rate': {'valid': f'{len(passed)}/{len(valid)}',
                                   'attempted': f'{len(passed)}/{total[task_id]}',
                                   'crashed': crashed[task_id]},
                'bucket_valid_only': bucket,
                'bucket_crash_as_fail': bucket_crash,
                'selected_by': why,
                'selection': {'script': 'memory_bank/tau2/select_memories.py',
                              'seed': seed, 'distinct_writes': distinct},
                'reward_info': ri,
                'task': s['_meta'].get('task'),
                'messages': s['messages'],
            })
    return records, stats, sims, total, crashed


def main():
    ap = argparse.ArgumentParser()
    add_agent_arg(ap)
    ap.add_argument('--domain', choices=DOMAINS, action='append')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--distinct-writes', action='store_true',
                    help='prefer two picks whose write calls differ')
    ap.add_argument('--dry-run', action='store_true', help='print the counts, write nothing')
    a = ap.parse_args()
    global ROLLOUTS, OUT
    ag = AGENTS[a.agent]
    ROLLOUTS = Path(ag.rollouts)
    OUT = ag.base / '{}'

    for domain in (a.domain or DOMAINS):
        records, stats, sims, total, crashed = run(domain, a.seed, a.distinct_writes)
        n_pass = sum(r['outcome'] == 'pass' for r in records)
        moved = len({r['task_id'] for r in records
                     if r['bucket_valid_only'] != r['bucket_crash_as_fail']})
        print(f'{domain}: {len(sims)} tasks -> {len(records)} memories '
              f'({n_pass} pass, {len(records) - n_pass} fail); '
              f"buckets high {stats['high']} mid {stats['mid']} low {stats['low']}"
              + (f", SHORT {stats['short']}" if stats['short'] else '')
              + f'; {moved} tasks bucket differently if crashes count as failures')
        if a.dry_run:
            continue
        out = Path(str(OUT).format(domain))
        out.mkdir(parents=True, exist_ok=True)
        (out / 'selection.json').write_text(json.dumps(records, ensure_ascii=False))
        (out / 'selection_index.json').write_text(json.dumps(
            [{k: v for k, v in r.items() if k not in ('messages', 'task', 'reward_info')}
             for r in records], ensure_ascii=False, indent=1))
        print(f'  -> {out}/selection.json  '
              f'({(out / "selection.json").stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
