#!/usr/bin/env python
"""Turn a rollout `sims.jsonl` into the tau2 `results.json` that the review scripts
(`review/self_critique.py`, `review/pass_audit.py`) expect.

Those scripts read one object with three keys -- `info.environment_info.domain_name`, `tasks`
and `simulations` -- and write a copy of it with `self_critique` / `pass_audit` added to each
simulation. `sims.jsonl` holds the same simulations one per line, each with a `_meta` block
that carries the domain and the full task, so the conversion is a regroup, not a rewrite:
every simulation dict is passed through untouched (`_meta` included; pydantic ignores it).

    # everything, for --siblings and for labelling a whole run
    sims_to_tau2.py --agent ministral retail -o /tmp/retail_all.json

    # only the trajectories picked by select_memories.py
    sims_to_tau2.py --agent ministral retail --selection -o /tmp/retail_selected.json

`self_critique.py` labels only the FAILING simulations in its input and `pass_audit.py` only
the PASSING ones, so one converted file feeds both.
"""
import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from agents import AGENTS, add_agent_arg

SIMS = SELECTION = None


def main():
    ap = argparse.ArgumentParser()
    add_agent_arg(ap)
    ap.add_argument('domain', choices=['airline', 'retail', 'telecom'])
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('--sims', help="override the agent's default train-nomem sims.jsonl")
    ap.add_argument('--selection', action='store_true',
                    help='keep only the sim_ids chosen by select_memories.py')
    a = ap.parse_args()
    global SIMS, SELECTION
    ag = AGENTS[a.agent]
    SIMS = ag.rollouts + '/sims.jsonl'
    SELECTION = str(ag.base / '{}' / 'selection.json')

    sims = [json.loads(l) for l in open(a.sims or SIMS.format(a.domain))]
    if a.selection:
        keep = {r['sim_id'] for r in json.load(open(SELECTION.format(a.domain)))}
        sims = [s for s in sims if s['id'] in keep]
        if len(sims) != len(keep):
            raise SystemExit(f'{len(keep) - len(sims)} selected sim_ids are missing from the dump')

    tasks, domains = {}, set()
    for s in sims:
        meta = s.get('_meta') or {}
        domains.add(meta.get('domain'))
        t = meta.get('task')
        if t:
            tasks[t['id']] = t
    if domains != {a.domain}:
        raise SystemExit(f'dump mixes domains: {domains}')
    missing = [s['id'] for s in sims if not (s.get('_meta') or {}).get('task')]
    if missing:
        raise SystemExit(f'{len(missing)} simulations carry no task in _meta, e.g. {missing[0]}')

    out = {'info': {'environment_info': {'domain_name': a.domain},
                    'source': 'sims.jsonl via memory_bank/tau2/sims_to_tau2.py',
                    'agent_info': {'llm': (sims[0]['_meta'] or {}).get('agent_llm'),
                                   'llm_args': {'temperature': sims[0]['_meta'].get('agent_temperature'),
                                                'max_tokens': sims[0]['_meta'].get('agent_max_tokens')}},
                    'user_info': {'llm': sims[0]['_meta'].get('user_llm')},
                    'num_trials': None},
           'tasks': list(tasks.values()),
           'simulations': sims}
    Path(a.output).write_text(json.dumps(out, ensure_ascii=False))
    n_pass = sum(1 for s in sims if ((s.get('reward_info') or {}).get('reward') or 0) >= 1)
    print(f'{a.domain}: {len(sims)} simulations ({n_pass} pass, {len(sims) - n_pass} fail) '
          f'over {len(tasks)} tasks -> {a.output} '
          f'({Path(a.output).stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
