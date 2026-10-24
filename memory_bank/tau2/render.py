#!/usr/bin/env python
"""Render a labelled selection into a memory bank the agent can read.

    render.py --agent ministral airline [--dry-run]

Input  `<bank>/<domain>/selection_final.json`
       (written by topup.py: the picks whose label survived the audit)
Output `<domain>/task<NN>_trial<t>/{memory.json,digest.txt,full.txt}`
       plus `query_keys.jsonl` (for build_index.py) and `curation.jsonl`.

The text itself is produced by digest_memory.render; this script translates the review labels
into the `label` object that renderer reads:

    self_critique            -> kind 'failure_attribution'  (root_cause, contributing_errors,
                                failed_condition, summary)
    pass_audit, final.fake_pass true  -> kind 'fake_pass'   (verified_reason, issues)
    pass_audit, otherwise             -> kind 'clean_pass'  (summary)

`final.fake_pass` decides, never the model's own `clean` / `pass_class`: those can contradict
the model's own issue list, while `final` is derived mechanically.

`note` and `minor_flags` hold a human reviewer's words. They are left empty: no record has been
through a human review, and machine flags in their place would overstate what the record is.
"""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from digest_memory import DROP_FIELDS as AIRLINE_DROP, PRIORITY as AIRLINE_PRIORITY, render
import paths
from agents import AGENTS, add_agent_arg

BASE = None

# Character budgets for the digest and full renderings (see digest_memory.render).
TEXT_LIMIT, PROSE_LIMIT, TOOL_LIMIT = 100, 200, 250
ELIDE, HEAD, TAIL, WINDOW, ANCHOR_CALLS = True, 2, 2, 1, True
FULL_TEXT_LIMIT, FULL_TOOL_LIMIT = 200, 100

DROP = {'airline': AIRLINE_DROP,
        'retail': {'user_id', 'email', 'dob'},
        'telecom': {'full_name', 'date_of_birth', 'email', 'address', 'account_number_last_4'}}
PRIORITY = {
    'airline': AIRLINE_PRIORITY,
    'retail': ('order_id', 'status', 'items', 'item_id', 'product_id', 'name', 'options',
               'price', 'available', 'quantity', 'variants', 'exchange_items',
               'exchange_new_items', 'exchange_price_difference', 'cancel_reason', 'address',
               'tracking_id', 'fulfillments', 'id', 'source', 'balance', 'payment_history'),
    'telecom': ('line_id', 'phone_number', 'status', 'roaming_enabled', 'contract_end_date',
                'suspension_start_date', 'data_used_gb', 'data_limit_gb', 'data_refueling_gb',
                'plan_id', 'customer_id', 'line_ids', 'bill_id', 'bill_ids', 'amount_due',
                'due_date', 'total_due', 'name', 'price_per_month',
                'data_refueling_price_per_gb')}

PROVENANCE = None
AG = None


def to_label(rec):
    """This pipeline's critique/audit output -> the `label` object digest_memory.render reads."""
    L = rec['label_source']
    if rec['outcome'] == 'fail':
        return {'kind': 'failure_attribution',
                'failed_condition': L.get('failed_condition'),
                'summary': L.get('summary'),
                'root_cause': L.get('root_cause') or {},
                'contributing_errors': L.get('contributing_errors') or [],
                'note': None, 'minor_flags': [], 'provenance': PROVENANCE}
    fake = bool((L.get('final') or {}).get('fake_pass'))
    label = {'kind': 'fake_pass' if fake else 'clean_pass',
             'summary': L.get('summary'),
             'note': None, 'minor_flags': [], 'provenance': PROVENANCE}
    if fake:
        label['verified_reason'] = L.get('fake_pass_reason')
        # Only the issues the mechanical resolution kept as real, and only critical ones:
        # a `minor` issue on a passing run is a lead, not a lesson.
        label['issues'] = [i for i in (L.get('issues') or [])
                           if i.get('severity') == 'critical' and i.get('turn_idx') is not None]
        # A fake_pass verdict can come from a mechanical check alone (the model may not overrule
        # `claims_action_not_performed`), and then the model, having called the run genuine, wrote
        # no reason and no critical issue. Fall back to the checks that decided it, which carry a
        # turn and a detail, so the record never renders with a blank "FAKE PASS BECAUSE".
        decided = [f for f in (L.get('auto_findings') or [])
                   if f.get('check') in ((L.get('final') or {}).get('reasons') or [])
                   and f.get('turn_idx') is not None]
        if not (label['verified_reason'] or '').strip() or label['verified_reason'] == 'None':
            detail = '; '.join(f"{f['check']} at turn {f['turn_idx']}: {f.get('detail')}"
                               for f in decided)
            label['verified_reason'] = (
                detail or 'mechanical check: '
                + ', '.join((L.get('final') or {}).get('reasons') or ['unspecified']))
        if not label['issues']:
            label['issues'] = [{'turn_idx': f['turn_idx'], 'category': f['check'],
                                'severity': 'critical', 'what_happened': f.get('detail'),
                                'should_have': None, 'evidence': None} for f in decided]
    return label


def to_grader(rec):
    """The grader block the renderer prints: what the grader concluded, plus the write diff."""
    ri = rec.get('reward_info') or {}
    L = rec['label_source']
    return {'db_match': (ri.get('db_check') or {}).get('db_match'),
            'db_diff': L.get('db_diff') or [],
            'unmet_communicate': [c.get('info') for c in (ri.get('communicate_checks') or [])
                                  if not c.get('met')] or None,
            'unmet_nl_assertions': [n.get('nl_assertion') for n in (ri.get('nl_assertions') or [])
                                    if not n.get('met')]}


def main():
    ap = argparse.ArgumentParser()
    add_agent_arg(ap)
    ap.add_argument('domain', choices=['airline', 'retail', 'telecom'])
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    global BASE, PROVENANCE, AG
    AG = AGENTS[a.agent]
    BASE = AG.base
    PROVENANCE = (f'{AG.label} self_critique / pass_audit + mechanical audit; '
                  'selection by memory_bank/tau2/select_memories.py, no human review')
    out = BASE / a.domain
    recs = json.load(open(out / 'selection_final.json'))

    # Folder names: numeric task ids keep their number (3 digits for retail, 2 elsewhere),
    # scenario-string ids are numbered by sort order.
    tids = sorted({r['task_id'] for r in recs},
                  key=lambda t: (0, int(t), '') if str(t).isdigit() else (1, 0, str(t)))
    rank = {t: (int(t) if str(t).isdigit() else i) for i, t in enumerate(tids)}
    width = 3 if a.domain == 'retail' else 2
    seeds = sorted({r['seed'] for r in recs})
    by_task = collections.defaultdict(list)
    for r in recs:
        by_task[r['task_id']].append(r)

    # The trial number is the record's position among ALL of its task's transcripts ordered by
    # (opening, seed, sim_id), so it is unique and stable across top-up rounds. It is NOT the
    # tau2 trial index: a task's sampled openings can repeat the same string, so (opening, seed)
    # does not identify a run, and a crashed sim leaves a gap. memory.json carries the real
    # seed and opening.
    import pandas as pd
    opidx = {}
    for j in range(4):
        for e in pd.read_parquet(paths.openings(a.domain, j))['extra_info']:
            opidx.setdefault((str(e['task_id']), e['opening']), j)
    trial = {}
    sims_path = AG.rollouts.format(a.domain) + '/sims.jsonl'
    per_task = collections.defaultdict(list)
    for line in open(sims_path):
        s = json.loads(line)
        per_task[str(s['task_id'])].append(s)
    for t, ss in per_task.items():
        ss.sort(key=lambda s: (opidx.get((t, s['_meta']['opening']), 9),
                               seeds.index(s['_meta']['seed']) if s['_meta']['seed'] in seeds else 9,
                               s['id']))
        for i, s in enumerate(ss):
            trial[s['id']] = i

    names = {r['sim_id']: f'task{rank[r["task_id"]]:0{width}d}_trial{trial[r["sim_id"]]}'
             for r in recs}
    assert len(set(names.values())) == len(recs), 'folder names collide'

    ordered = sorted(recs, key=lambda r: names[r['sim_id']])
    kinds, sizes, full_sizes, keys, curation = collections.Counter(), [], [], [], []
    for row, r in enumerate(ordered):
        name = names[r['sim_id']]
        record = {'sim_id': r['sim_id'], 'task_id': r['task_id'], 'domain': a.domain,
                  'outcome': r['outcome'], 'reward': r['reward'],
                  'termination_reason': r['termination_reason'],
                  'task': r['task'], 'messages': r['messages'],
                  'label': to_label(r), 'grader': to_grader(r),
                  'selected_by': r['selected_by']}
        user_turns = [m['content'] for m in r['messages']
                      if m['role'] == 'user' and (m.get('content') or '').strip()]
        key_text = user_turns[0] if user_turns else ''
        kinds[record['label']['kind']] += 1

        digest = render(record, 'digest', with_id=True, text_limit=TEXT_LIMIT,
                        prose_limit=PROSE_LIMIT, tool_limit=TOOL_LIMIT, elide=ELIDE, head=HEAD,
                        tail=TAIL, window=WINDOW, anchor_calls=ANCHOR_CALLS,
                        drop_fields=DROP[a.domain], priority=PRIORITY[a.domain], drop_nulls=True)
        full = render(record, 'full', with_id=True, window=WINDOW,
                      full_text_limit=FULL_TEXT_LIMIT, full_tool_limit=FULL_TOOL_LIMIT,
                      drop_fields=DROP[a.domain], priority=PRIORITY[a.domain], drop_nulls=True)
        sizes.append(len(digest)); full_sizes.append(len(full))
        keys.append({'sim_id': r['sim_id'], 'task_id': r['task_id'],
                     'trial': name.split('_trial')[1], 'label_kind': record['label']['kind'],
                     'folder': name, 'key_text': key_text,
                     'key_sha256': hashlib.sha256(key_text.encode()).hexdigest(),
                     'user_turns': user_turns})
        curation.append({'folder': name, 'sim_id': r['sim_id'], 'task_id': r['task_id'],
                         'label_kind': record['label']['kind'],
                         'kept_because': r['selected_by']})
        if a.dry_run:
            continue
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        (d / 'memory.json').write_text(json.dumps(
            {'folder': name, 'sim_id': r['sim_id'], 'task_id': r['task_id'],
             'trial': name.split('_trial')[1], 'label_kind': record['label']['kind'],
             'kept_because': r['selected_by'],
             'retrieval_key': {'text': key_text,
                               'sha256': hashlib.sha256(key_text.encode()).hexdigest(),
                               'index_row': row, 'index_file': 'index.npz'},
             'user_turns': user_turns, 'source_record': record},
            ensure_ascii=False, indent=2))
        (d / 'digest.txt').write_text(digest + '\n')
        (d / 'full.txt').write_text(full + '\n')

    per = collections.Counter(len(v) for v in by_task.values())
    print(f'{a.domain}: {len(ordered)} memories over {len(by_task)} tasks '
          f'(per-task {dict(sorted(per.items()))}); kinds {dict(kinds)}')
    print(f'  digest {min(sizes)}-{max(sizes)} chars (mean {sum(sizes) // len(sizes)}), '
          f'full {min(full_sizes)}-{max(full_sizes)} (mean {sum(full_sizes) // len(full_sizes)})')
    if a.dry_run:
        print('  dry run; nothing written')
        return
    (out / 'query_keys.jsonl').write_text(''.join(json.dumps(k, ensure_ascii=False) + '\n'
                                                  for k in keys))
    (out / 'curation.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n'
                                                for c in curation))
    print(f'  -> {out}/task*/  + query_keys.jsonl, curation.jsonl')


if __name__ == '__main__':
    main()
