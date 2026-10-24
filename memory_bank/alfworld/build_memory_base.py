#!/usr/bin/env python
"""Assemble the ALFWorld memory base from raw trajectories, facts and self-reviews.

    python build_memory_base.py --model ministral8b --reviews reviews_full_final

One folder per memory, memory_banks/<model>/<group_id>__<trial>/:
    memory.json   everything: key, outcome, trajectory, facts, review, audit flags, provenance
    full.txt      what the executor reads (think <=300 chars, every step, all annotations)
    digest.txt    what the selector reads (no think, obs cut to 160, notes = what happened + should have)

Both texts are deterministic slices of memory.json; no model is involved. A trajectory is left out
when its review did not pass the audit, or when it won through the double-inReceptacle quirk
(facts `env_quirk_early_win`).
"""
import argparse
import collections
import json
import re
import shutil
from pathlib import Path

import paths

THINK, DIGEST_OBS, DIGEST_PROSE = 300, 160, 200
TX = {'heat': 'heated', 'cool': 'cooled', 'clean': 'cleaned', 'look': 'lamp on while held'}


def cut(s, n):
    s = ' '.join(str(s or '').split())
    return s if n is None or len(s) <= n else s[:n] + ' ...'


def config_of(group_id):
    """group_id = <task_type>-<Object>-<MRecep>-<Parent|Toggle>-<scene>; the config drops the scene."""
    return group_id.rsplit('-', 1)[0]


def facts_lines(f):
    """FACTS block for both renders. Receptacle TYPES only: instance names from the answer key stay out."""
    t, p = f['target'], f['progress']
    out = []
    if t['transform'] == 'look':
        out.append(f"target: hold a {t['object']} with the desklamp on")
    elif t['transform']:
        out.append(f"target: {t['object']} -> {t['transform']} with {t['appliance']} -> put in {t['receptacle']}")
    else:
        out.append(f"target: {'two ' if t['count'] == 2 else ''}{t['object']} -> put in {t['receptacle']}"
                   + (' (both in the same one)' if t['count'] == 2 else ''))
    out.append(f"where {t['object']}s were at the start: {', '.join(f['target_location']) or 'unknown'}")
    prog = [f"seen @{p['seen']}" if p['seen'] else 'never seen', f"taken @{p['taken']}" if p['taken'] else 'never taken']
    if t['transform']:
        prog.append(f"{TX[t['transform']]} @{p['transformed']}" if p['transformed'] else f"{TX[t['transform']]}: never")
    if 'placed_count' in p:
        prog.append(f"most placed together: {p['placed_count']} of 2")
    prog.append(f"goal reached @{p['placed']}" if p['placed'] else 'goal never reached')
    out.append('progress: ' + ', '.join(prog))
    ev = [e for e in f['events'] if e['type'] not in ('picked_non_target',)]
    if ev:
        out.append('events: ' + '; '.join(f"@{e['step']} {e['type'].replace('_', ' ')} ({e['object']})" for e in ev))
    if f['transform_available_unused']:
        verb = 'use desklamp' if t['transform'] == 'look' else t['transform']
        out.append(f"the {verb} action was available but unused @"
                   + ', '.join(map(str, f['transform_available_unused'])))
    if f['revisits']:
        out.append('revisits (nothing new learned): @' + ', '.join(map(str, f['revisits'])))
    if f['invalid_replies']:
        out.append(f"invalid replies: {len(f['invalid_replies'])}")
    if not f['won']:
        s = f['search']
        out.append(f"search: visited {len(s['visited'])} of {s['receptacles']} receptacles; never visited: "
                   + (', '.join(s['never_visited_types']) or 'none'))
    return out


def notes_by_step(review, won, full):
    """step -> list of annotation lines placed under that step."""
    notes = collections.defaultdict(list)
    prose = None if full else DIGEST_PROSE
    if won:
        for k in review['key_steps']:
            notes[k['step']].append(f"** KEY: {cut(k.get('why_it_mattered'), prose)}")
        for w in review['wasted_steps']:
            a = int(str(w['steps']).split('-')[0])
            notes[a].append(f"** WASTED (#{w['steps']}): {cut(w.get('what_happened'), prose)}")
            notes[a].append(f"   SHOULD HAVE: {w['should_have']}")
    else:
        rc = review['root_cause']
        notes[rc['step']].append(f"** ROOT CAUSE: {cut(rc.get('what_happened'), prose)}")
        if full:
            notes[rc['step']].append(f"   WHY: {rc.get('why_it_caused_the_failure') or ''}")
        if rc.get('should_have'):
            notes[rc['step']].append(f"   SHOULD HAVE: {rc['should_have']}")
        for c in review['contributing_errors']:
            notes[c['step']].append(f"** CONTRIBUTING: {cut(c.get('what_happened'), prose)}")
            if c.get('should_have'):
                notes[c['step']].append(f"   SHOULD HAVE: {c['should_have']}")
    return notes


def render(mem, full):
    f, rv, rec = mem['facts'], mem['review'], mem['trajectory']
    won = mem['outcome']['won']
    o = mem['outcome']
    lines = [f"Goal: {mem['key']}", f"Task type: {mem['task_type']}"]
    lines.append(f"OUTCOME: SUCCEEDED at step {o['won_step']} of {o['max_steps']} (reference solution: {o['reference_len']} steps)"
                 if won else
                 f"OUTCOME: FAILED, ran out of steps ({o['steps']} of {o['max_steps']}, reference solution: {o['reference_len']} steps)")
    lines.append('FACTS (computed from the log):')
    lines += ['  ' + x for x in facts_lines(f)]
    if won:
        lines.append(f"SUMMARY: {rv['summary']}")
        lines.append(f"STRATEGY: {rv['reusable_strategy']}")
    else:
        lines.append(f"FAILED CONDITION: {rv.get('failed_condition') or ''}")
        lines.append(f"SUMMARY: {rv['summary']}")
    lines.append('Trajectory:')
    notes = notes_by_step(rv, won, full)
    pending = 0
    for t in rec:
        if t['status'] != 'ok':
            pending += 1
            continue
        lines.append(f" #{t['step']} {t['action']}")
        if full and pending:
            lines.append(f"    ({pending} invalid repl{'y' if pending == 1 else 'ies'} before this action)")
        pending = 0
        if full and t.get('think'):
            lines.append(f"    think: {cut(t['think'], THINK)}")
        lines.append(f"    -> {t['obs'] if full else cut(t['obs'], DIGEST_OBS)}")
        for n in notes.get(t['step'], []):
            lines.append('    ' + n)
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--reviews', default='reviews_full_final', help='file stem under memory_banks/<model>/_work/')
    ap.add_argument('--out', default=None, help='default memory_banks/<model>/')
    a = ap.parse_args()
    work = paths.work(a.model)
    out = Path(a.out or paths.bank(a.model))
    raw = {json.loads(l)['task_id']: json.loads(l) for l in open(paths.rollouts(a.model))}
    facts = {json.loads(l)['task_id']: json.loads(l) for l in open(work / 'facts.jsonl')}
    reviews = {json.loads(l)['task_id']: json.loads(l) for l in open(work / f'{a.reviews}.jsonl')}

    # rebuild from scratch, leaving _work alone
    for d in out.iterdir():
        if d.is_dir() and d.name != '_work':
            shutil.rmtree(d)
    kept, why = collections.Counter(), collections.Counter()
    by_type = collections.defaultdict(lambda: collections.Counter())
    flag_count = collections.Counter()
    sizes = {'full': [], 'digest': []}
    for t, rec in raw.items():
        f = facts[t]
        tt = rec['task_type']
        if f['env_quirk_early_win']:
            why['env_quirk_early_win'] += 1
            by_type[tt]['quirk'] += 1
            continue
        r = reviews.get(t)
        if r is None:
            why['no_review'] += 1
            by_type[tt]['no_review'] += 1
            continue
        if not r['usable']:
            why['review_failed_audit'] += 1
            by_type[tt]['dropped'] += 1
            continue
        game = json.load(open(paths.gamefile(t)))
        trial = t.rsplit('/', 1)[1]
        mem = {
            'memory_id': f"{rec['group_id']}__{trial}",
            'key': rec['task'],
            'task_type': tt,
            'config': config_of(rec['group_id']),
            'group_id': rec['group_id'],
            'floor_plan': rec['floor_plan'],
            'task_id': t,
            'outcome': {'won': int(rec['won']), 'won_step': rec['won_step'], 'steps': rec['steps'],
                        'env_actions': rec['env_actions'], 'max_steps': rec['max_steps'],
                        'reference_len': len(game['walkthrough'])},
            'trajectory': [{k: x.get(k) for k in ('step', 'attempt', 'status', 'action', 'obs', 'think')}
                           for x in rec['trajectory']],
            'facts': f,
            'review': r['review'],
            'audit': {'flags': r['flags'], 'attempts': len(r['attempts'])},
            'meta': {'executor': rec['run']['model'], 'executor_run': rec['run']['run_id'],
                     'temperature': rec['run']['temperature'], 'prompt_sha': rec['run']['prompt_sha'],
                     'critic': r['critic'], 'critic_version': r['critic_version']},
        }
        d = out / mem['memory_id']
        d.mkdir(parents=True)
        (d / 'memory.json').write_text(json.dumps(mem, ensure_ascii=False, indent=1))
        full, dig = render(mem, True), render(mem, False)
        (d / 'full.txt').write_text(full)
        (d / 'digest.txt').write_text(dig)
        sizes['full'].append(len(full))
        sizes['digest'].append(len(dig))
        kept['won' if rec['won'] else 'lost'] += 1
        by_type[tt]['kept_' + ('won' if rec['won'] else 'lost')] += 1
        for x in r['flags']:
            flag_count[x.split(':')[0]] += 1

    n = sum(kept.values())
    lines = [f'# ALFWorld memory base: {a.model}', '',
             f"Built by `memory_bank/alfworld/build_memory_base.py` from `rollouts/{a.model}/train_full_s20.jsonl`, "
             f"`_work/facts.jsonl` and `_work/{a.reviews}.jsonl`.", '',
             f'- memories: **{n}** ({kept["won"]} won, {kept["lost"]} lost) of {len(raw)} trajectories',
             f'- left out: ' + ', '.join(f'{k} {v}' for k, v in why.items()),
             f'- full.txt chars: mean {sum(sizes["full"]) // max(1, n)}, max {max(sizes["full"], default=0)}; '
             f'digest.txt chars: mean {sum(sizes["digest"]) // max(1, n)}, max {max(sizes["digest"], default=0)}',
             '', '| task type | kept won | kept lost | dropped (audit) | quirk |', '|---|---|---|---|---|']
    for tt in sorted(by_type):
        c = by_type[tt]
        lines.append(f"| {tt} | {c['kept_won']} | {c['kept_lost']} | {c['dropped']} | {c['quirk']} |")
    lines += ['', 'Audit flags on kept memories (a memory can carry several): '
              + ', '.join(f'{k} {v}' for k, v in sorted(flag_count.items())),
              '(3r/4r repaired action, 4s should_have real but not typeable at that step, 9 short wasted run '
              'dropped, 10r instance numbers stripped, 11 should_have points at the answer-key spot)']
    (out / 'README.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
