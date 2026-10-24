#!/usr/bin/env python
"""Build a stable task index over the ALFWorld game files.

    python build_task_index.py --splits train valid_unseen valid_seen valid_train

Every playable game is one directory

    $ALFWORLD_DATA/json_2.1.1/<split>/<taskdir>/<trial_T.../>

holding `game.tw-pddl` (the TextWorld game) and `traj_data.json` (the ALFRED annotation).
`task_id` is `<split>/<taskdir>/<trial>` -- stable across processes and machines, which is what
the evaluator addresses games by, the same role `task_id` plays for tau2.

`group_id` is the taskdir alone. It MUST be the unit for any train/eval split: valid_unseen's 133
games come from only 47 taskdirs, so splitting by game leaks near-duplicates across the boundary
(the trials of one taskdir differ only in object placement seed).

A game whose `game.tw-pddl` is missing the solvable key is skipped, exactly as AlfredTWEnv skips
it -- that is why valid_unseen yields 133 and not the 134 quoted in the literature.
"""
import argparse
import collections
import json
from pathlib import Path

import paths

DATA = paths.ALFWORLD_DATA


def scan(split):
    root = DATA / 'json_2.1.1' / split
    rows, skipped = [], []
    for game in sorted(root.glob('*/*/game.tw-pddl')):
        trial_dir = game.parent
        taskdir = trial_dir.parent.name
        try:
            gd = json.loads(game.read_text())
        except Exception as e:
            skipped.append((str(game), f'unreadable: {e}'))
            continue
        # Mirror AlfredTWEnv: a game with no `solvable` key, or with solvable False, never
        # reaches the env, so it must not reach the index either.
        if 'solvable' not in gd:
            skipped.append((str(game), 'missing solvable key'))
            continue
        if not gd['solvable']:
            skipped.append((str(game), 'solvable=False'))
            continue
        t = json.loads((trial_dir / 'traj_data.json').read_text())
        p = t.get('pddl_params', {})
        anns = [a['task_desc'].strip() for a in t.get('turk_annotations', {}).get('anns', [])]
        rows.append({
            'task_id': f'{split}/{taskdir}/{trial_dir.name}',
            'group_id': taskdir,
            'split': split,
            'gamefile': str(game),
            'task_type': t.get('task_type'),
            'floor_plan': t.get('scene', {}).get('floor_plan'),
            'object_target': p.get('object_target'),
            'parent_target': p.get('parent_target'),
            'mrecep_target': p.get('mrecep_target'),
            'toggle_target': p.get('toggle_target'),
            'object_sliced': p.get('object_sliced'),
            'expert_plan_len': len(t.get('plan', {}).get('high_pddl', [])),
            'human_goals': anns,
        })
    return rows, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', nargs='+',
                    default=['train', 'valid_train', 'valid_seen', 'valid_unseen'])
    ap.add_argument('-o', '--out', default=str(paths.task_index()))
    a = ap.parse_args()

    allrows, allskip = [], []
    for split in a.splits:
        rows, skipped = scan(split)
        allrows += rows
        allskip += skipped
        groups = len({r['group_id'] for r in rows})
        plans = len({r['floor_plan'] for r in rows})
        types = collections.Counter(r['task_type'] for r in rows)
        print(f'{split:14s} games {len(rows):5d}  taskdirs {groups:5d}  floor_plans {plans:4d}'
              f'  skipped {len(skipped)}')
        for k, v in types.most_common():
            print(f'    {k:32s} {v}')

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as f:
        for r in allrows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'\n{len(allrows)} games -> {out}')
    if allskip:
        print(f'skipped {len(allskip)}:')
        for g, why in allskip[:10]:
            print(f'  {why}: {g}')


if __name__ == '__main__':
    main()
