#!/usr/bin/env python
"""Re-run the (current) audit over every stored critic answer; no model calls.

    python reaudit_reviews.py --model qwen3-4b --reviews reviews_full      # -> reviews_full_final.jsonl

Each review row was audited by whatever version of the audit in self_review.py was running when it
was written. This replays every row's saved attempts through the current audit (which, for example,
repairs look-task near misses such as `use desklamp N` for a lamp the agent never saw, or
`go to desklamp N`) and keeps the first attempt that passes. Rows are never re-sampled: the answers
are the ones the critic gave. The `_final` file is what build_memory_base.py reads by default.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import paths                                                            # noqa: E402

from self_review import audit, parse                                    # noqa: E402

AUDIT_VERSION = 'a3'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--reviews', default='reviews_full')
    a = ap.parse_args()
    work = paths.work(a.model)
    raw = {json.loads(l)['task_id']: json.loads(l) for l in open(paths.rollouts(a.model))}
    facts = {json.loads(l)['task_id']: json.loads(l) for l in open(work / 'facts.jsonl')}
    rows = [json.loads(l) for l in open(work / f'{a.reviews}.jsonl')]
    out = work / f'{a.reviews}_final.jsonl'
    c = collections.Counter()
    with out.open('w') as fh:
        for r in rows:
            before = r['usable']
            new = dict(r, usable=False, review=None, flags=[], audit_version=AUDIT_VERSION, usable_at_run=before)
            for i, at in enumerate(r['attempts']):
                try:
                    ok, reasons, flags, clean = audit(raw[r['task_id']], facts[r['task_id']], parse(at['raw']))
                except Exception as e:
                    ok, reasons, flags, clean = False, [f'1: the answer is not valid JSON ({type(e).__name__})'], [], None
                at['reaudit'] = {'ok': ok, 'reasons': reasons, 'flags': flags}
                if ok:
                    new.update(usable=True, review=clean, flags=flags, used_attempt=i)
                    break
            c['rows'] += 1
            c['usable'] += new['usable']
            c['recovered'] += new['usable'] and not before
            c['lost'] += before and not new['usable']
            fh.write(json.dumps(new, ensure_ascii=False) + '\n')
    print(f"{a.model}: {c['rows']} rows, usable {c['usable']} "
          f"(recovered {c['recovered']}, newly rejected {c['lost']}) -> {out}")


if __name__ == '__main__':
    main()
