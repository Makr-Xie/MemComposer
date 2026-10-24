#!/usr/bin/env python
"""Top a memory selection back up to two usable memories per task.

`select_memories.py` picks two trajectories per task by pass rate, but a pick is only
usable once it carries a label that survives the mechanical audit, and some do not: a critique
that fails to parse, a `too_many_errors` termination that no single turn explains, a root cause
pointing at a turn where the named call was not made. This script replaces those, keeping the
rule intact -- a replacement always has the SAME outcome as the record it stands in for, so a
high-pass task still contributes two passes and a low-pass task two failures.

Each round: find the tasks below two usable memories, draw replacements from the unused
transcripts of the same outcome, label them (`self_critique.py` for failures, `pass_audit.py`
for passes, the agent itself as the critic), and re-check. It stops when every task is at two
or the candidate pool is exhausted; a task with no further transcript of the right outcome
stays short.

    topup.py --agent ministral --domain airline [--max-rounds 4] [--dry-run]

Writes `<base>/selection_final.json` (the records that will be rendered, each with `label`) and
`<base>/labels.json` (sim_id -> label, every round merged). Rounds are kept in `<base>/_rounds/`.
"""
import argparse
import collections
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from agents import AGENTS, add_agent_arg, creds

BASE = None
SIMS = None
SCRIPTS = Path(__file__).resolve().parent / 'review'
PY = os.environ.get('PYTHON', sys.executable)    # must import tau2 (the review scripts need it)
CRITIC = None
AG = None


def critic_env(agent):
    """The critic's credentials into this process's children only; never printed."""
    env = dict(os.environ)
    env.update(creds(agent))
    return env


def usable(label, outcome):
    """A label good enough to render. Failures must also pass the critique audit."""
    if not label or 'parse_error' in label:
        return False
    return bool(label.get('usable')) if outcome == 'fail' else True


def results_file(sims, domain, path):
    """The tau2 results.json shape the labelling scripts read."""
    tasks = {}
    for s in sims:
        t = (s.get('_meta') or {}).get('task')
        if t:
            tasks[t['id']] = t
    path.write_text(json.dumps({'info': {'environment_info': {'domain_name': domain}},
                                'tasks': list(tasks.values()), 'simulations': sims},
                               ensure_ascii=False))


def label_batch(domain, sims, rnd_dir, rnd, conc, siblings):
    """Run both labellers over one batch and return {sim_id: label}."""
    rnd_dir.mkdir(parents=True, exist_ok=True)
    src = rnd_dir / f'round{rnd}_in.json'
    results_file(sims, domain, src)
    env = critic_env(AG)
    out = {}
    # Each script picks its own targets out of the batch -- self_critique.py the failures,
    # pass_audit.py the passes -- so one file feeds both and a batch may be all of one kind.
    for script, key in (('self_critique.py', 'self_critique'), ('pass_audit.py', 'pass_audit')):
        dst = rnd_dir / f'round{rnd}_{key}.json'
        cmd = [PY, str(SCRIPTS / script), str(src), '-o', str(dst),
               '--model', CRITIC, '--concurrency', str(conc)]
        if script == 'self_critique.py':
            cmd += ['--siblings', str(siblings)]
        log = (rnd_dir / f'round{rnd}_{key}.log').open('w')
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        for s in json.loads(dst.read_text())['simulations']:
            if key in s:
                out[s['id']] = s[key]
    return out


def main():
    ap = argparse.ArgumentParser()
    add_agent_arg(ap)
    ap.add_argument('--domain', required=True, choices=['airline', 'retail', 'telecom'])
    ap.add_argument('--max-rounds', type=int, default=4)
    ap.add_argument('--concurrency', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true', help='report the gap, label nothing')
    a = ap.parse_args()
    global BASE, SIMS, CRITIC, AG
    AG = AGENTS[a.agent]
    BASE, CRITIC = AG.base, AG.model
    SIMS = AG.rollouts + '/sims.jsonl'
    base = BASE / a.domain
    rng = random.Random(f'topup|{a.seed}|{a.domain}')

    sims = {}
    for line in open(str(SIMS).format(a.domain)):
        s = json.loads(line)
        sims[s['id']] = s
    by_task = collections.defaultdict(list)
    for s in sims.values():
        by_task[str(s['task_id'])].append(s)

    selection = json.load(open(base / 'selection.json'))
    chosen = {r['task_id']: [] for r in selection}
    for r in selection:
        chosen[r['task_id']].append(r)

    # Round 0's labels are the ones already produced by label.sh.
    labels = {}
    store = base / 'labels.json'
    if store.exists():
        labels = json.load(open(store))
    else:
        for f, key in (('labelled_sc.json', 'self_critique'), ('labelled_pa.json', 'pass_audit')):
            for s in json.load(open(base / f))['simulations']:
                if key in s:
                    labels[s['id']] = s[key]

    work = BASE / '_work' / a.domain
    work.mkdir(parents=True, exist_ok=True)
    siblings = work / 'all.json'
    if not siblings.exists():
        results_file(list(sims.values()), a.domain, siblings)

    for rnd in range(1, a.max_rounds + 1):
        need = []
        for task, recs in chosen.items():
            good = [r for r in recs if usable(labels.get(r['sim_id']), r['outcome'])]
            if len(good) >= 2:
                continue
            short = [r for r in recs if not usable(labels.get(r['sim_id']), r['outcome'])]
            used = {r['sim_id'] for r in recs}
            for r in short[:2 - len(good)]:
                pool = [s for s in by_task[task] if s['id'] not in used
                        and (((s.get('reward_info') or {}).get('reward', 0) >= 1)
                             == (r['outcome'] == 'pass'))
                        and (s['id'] not in labels
                             or usable(labels[s['id']], r['outcome']))]
                if pool:
                    pick = rng.choice(pool)
                    used.add(pick['id'])
                    need.append((task, r, pick))
        if not need:
            print(f'round {rnd}: nothing left to top up')
            break
        print(f'round {rnd}: {len(need)} replacement(s) over '
              f'{len({t for t, _, _ in need})} task(s)', flush=True)
        if a.dry_run:
            for t, r, p in need[:5]:
                print(f'    task {t[:40]}: {r["sim_id"][:8]} ({r["outcome"]}) -> {p["id"][:8]}')
            break
        todo = [p for _, _, p in need if p['id'] not in labels]
        new = label_batch(a.domain, todo, work / 'rounds', rnd,
                          a.concurrency, siblings) if todo else {}
        labels.update(new)
        for task, old, pick in need:
            ri = pick.get('reward_info') or {}
            rec = dict(old)
            rec.update({'sim_id': pick['id'], 'reward': ri.get('reward'),
                        'termination_reason': pick.get('termination_reason'),
                        'seed': pick['_meta']['seed'], 'opening': pick['_meta']['opening'],
                        'opening_sample': None, 'reward_info': ri, 'messages': pick['messages'],
                        'selected_by': old['selected_by'] + f'; replacement for {old["sim_id"]}, '
                                       f'whose label did not survive the audit (round {rnd})'})
            chosen[task].append(rec)
        ok = sum(1 for t, recs in chosen.items()
                 if sum(usable(labels.get(r['sim_id']), r['outcome']) for r in recs) >= 2)
        print(f'  after round {rnd}: {ok}/{len(chosen)} tasks at two usable memories', flush=True)

    store.write_text(json.dumps(labels, ensure_ascii=False))
    # Finalise over EVERY labelled transcript of the task, not just the ones first drawn: a
    # replacement that was labelled in an earlier invocation is as good as one drawn now, and
    # rebuilding from `selection.json` alone would silently drop it.
    bucket = {r['task_id']: r['bucket_valid_only'] for r in selection}
    tmpl = {r['task_id']: r for r in selection}
    final, short = [], collections.Counter()
    for task, recs in chosen.items():
        want = {'high': ['pass', 'pass'], 'low': ['fail', 'fail'],
                'mid': ['pass', 'fail']}[bucket[task]]
        seen = {r['sim_id']: r for r in recs}
        for s in by_task[task]:
            if s['id'] in seen or s['id'] not in labels:
                continue
            ri = s.get('reward_info') or {}
            out = 'pass' if ri.get('reward', 0) >= 1 else 'fail'
            if not usable(labels[s['id']], out):
                continue
            cand = dict(tmpl[task])
            cand.update({'sim_id': s['id'], 'outcome': out, 'reward': ri.get('reward'),
                         'termination_reason': s.get('termination_reason'),
                         'seed': s['_meta']['seed'], 'opening': s['_meta']['opening'],
                         'opening_sample': None, 'reward_info': ri, 'messages': s['messages'],
                         'selected_by': tmpl[task]['selected_by'] + '; labelled replacement'})
            seen[s['id']] = cand
        pool = [r for r in seen.values() if usable(labels.get(r['sim_id']), r['outcome'])]
        good = []
        for o in want:
            pick = next((r for r in pool if r['outcome'] == o and r not in good), None)
            if pick:
                good.append(pick)
        short[len(good)] += 1
        for r in good:
            r = dict(r)
            r['label_source'] = labels[r['sim_id']]
            final.append(r)
    (base / 'selection_final.json').write_text(json.dumps(final, ensure_ascii=False))
    print(f'{a.domain}: {len(final)} memories over {len(chosen)} tasks; '
          f'per-task count {dict(sorted(short.items()))} -> {base}/selection_final.json')


if __name__ == '__main__':
    main()
