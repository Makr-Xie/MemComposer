#!/usr/bin/env python
"""Render memory records as text, in two profiles.

    python digest_memory.py <memory_base.json|.jsonl> -o <out.txt> --profile digest|full

Deterministic: no model is involved. Every line is a slice of the source record. render.py
imports `render` from here and passes its own budgets; the table shows the CLI defaults.

profile  for           free text            tool results           label notes
digest   the selector  cut to 160 chars     compacted, cut to 400  what_happened + should_have
full     the agent     whole                whole                  all label fields

Blocks are ordered by (int(task_id), trial) and separated by a blank line, 96 '=', a blank line.
"""
import argparse, json, re, sys
from pathlib import Path

SEP = '\n' + '=' * 96 + '\n\n'
TRUNC = ' …'
DROP_FIELDS = {'user_id', 'id', 'email', 'address', 'dob', 'first_name', 'last_name',
               'name', 'last_four', 'brand', 'saved_passengers'}
# Fields that decide eligibility come first, so a truncated tool result still carries them.
# In the raw JSON created_at and insurance sit behind flights[], which eats the whole budget.
PRIORITY = ('reservation_id', 'membership', 'cabin', 'insurance', 'created_at', 'status',
            'flight_type', 'passenger_count', 'total_baggages', 'nonfree_baggages',
            'origin', 'destination')
ROLE_NAMES = {'assistant': 'agent'}

NOTE_HEAD = {'root_cause': 'ROOT CAUSE', 'contributing': 'CONTRIBUTING'}
DIFF_PATTERNS = [
    (re.compile(r'\[(?P<tag>[A-Z ]+)\]\s*(?P<path>[^:]+):\s*the target leaves this UNCHANGED at '
                r'(?P<a>.*?), but the agent changed it to (?P<b>.*?)(?: —.*)?$', re.S), 'a', 'b'),
    (re.compile(r'\[(?P<tag>[A-Z ]+)\]\s*(?P<path>[^:]+):\s*originally (?P<a>.*?), the target '
                r'changes it to (?P<b>.*?), the agent never changed it\.?$', re.S), 'a', 'b'),
    (re.compile(r'\[(?P<tag>[A-Z ]+)\]\s*(?P<path>[^:]+):\s*originally .*?, target = (?P<a>.*?), '
                r'agent left = (?P<b>.*?)(?: —.*)?$', re.S), 'a', 'b'),
]


def collapse(text):
    return ' '.join(str(text or '').split())


def cut(text, limit):
    text = collapse(text)
    return text if limit is None or len(text) <= limit else text[:limit] + TRUNC


def compact_tool(content, drop_fields=None, priority=None, drop_nulls=False):
    """Keep the business fields, drop identity and contact fields, elide long lists.

    Truncating a tool result as free text would cut away cabin / insurance / created_at, which is
    the part that explains the case. Compacting first keeps those inside a small budget.

    `drop_fields` and `priority` default to the airline sets; another domain passes its own,
    because which keys are PII and which decide eligibility are domain facts, not general ones.
    """
    drop_fields = DROP_FIELDS if drop_fields is None else drop_fields
    priority = PRIORITY if priority is None else priority
    # `drop_nulls` is opt-in so the default output stays unchanged. Without it, a retail order's
    # always-null exchange/cancel fields spend the cap before `items` is reached.
    try:
        data = json.loads(content)
    except (ValueError, TypeError):
        return None

    def leg(d):
        """One flight leg as a string. Same fields, far fewer characters: the per-leg dicts
        repeat the same keys and would fill much of a capped tool line."""
        parts = [d.get('flight_number') or '?']
        if d.get('origin') or d.get('destination'):
            parts.append(f"{d.get('origin','?')}-{d.get('destination','?')}")
        for k, pre in (('date', ''), ('price', '$')):
            if d.get(k) is not None:
                parts.append(f'{pre}{d[k]}')
        for k in ('status', 'available_seats'):
            if d.get(k) is not None:
                parts.append(f'{k}={d[k]}')
        return ' '.join(str(x) for x in parts)

    def is_legs(value):
        return (isinstance(value, list) and value
                and all(isinstance(v, dict) and 'flight_number' in v for v in value))

    def variant(d):
        """One catalogue variant as a string."""
        parts = [str(d.get('item_id') or '?')]
        if isinstance(d.get('options'), dict):
            parts.append('/'.join(str(v) for v in d['options'].values()))
        if d.get('price') is not None:
            parts.append(f"${d['price']}")
        if d.get('available') is not None:
            parts.append('available' if d['available'] else 'UNAVAILABLE')
        if d.get('name'):
            parts.insert(0, str(d['name']))
        if d.get('quantity') is not None:
            parts.append(f"x{d['quantity']}")
        return ' '.join(parts)

    def is_variants(value):
        return (isinstance(value, dict) and value
                and all(isinstance(v, dict) and 'item_id' in v for v in value.values()))

    def is_items(value):
        return (isinstance(value, list) and value
                and all(isinstance(v, dict) and 'item_id' in v for v in value))

    def reduce(value):
        if isinstance(value, dict):
            out = {}
            for key, val in value.items():
                if key in drop_fields or (drop_nulls and val is None):
                    continue
                if key == 'passengers':
                    out['passenger_count'] = len(val)
                elif key == 'reservations':
                    out['reservation_count'] = len(val)
                elif key == 'payment_methods' and isinstance(val, dict):
                    out[key] = [reduce(v) for v in val.values()]
                elif is_legs(val):
                    out[key] = [leg(v) for v in val]
                elif is_variants(val):
                    out[key] = [variant(v) for v in val.values()]
                elif is_items(val):
                    out[key] = [variant(v) for v in val]
                else:
                    out[key] = reduce(val)
            order = {k: i for i, k in enumerate(priority)}
            return {k: out[k] for k in sorted(out, key=lambda k: (order.get(k, len(order)), k))}
        if isinstance(value, list):
            if len(value) > 8:
                return {'result_count': len(value), 'first': [reduce(v) for v in value[:4]],
                        'note': 'remaining omitted'}
            return [reduce(v) for v in value]
        return value
    return json.dumps(reduce(data), ensure_ascii=False, separators=(',', ':'))


def compress_diff_line(line):
    line = collapse(line)
    for pattern, a, b in DIFF_PATTERNS:
        m = pattern.match(line)
        if m:
            return f"[{m.group('tag').strip()}] {m.group('path').strip()}: " \
                   f"{cut(m.group(a), 60)} -> {cut(m.group(b), 60)}"
    return line


def db_lines(record, full):
    """Grader evidence. Mechanically computed, unlike every label field."""
    grader = record.get('grader') or {}
    if not grader:
        return []
    out = []
    flags = [f'db_match={grader.get("db_match")}']
    for key in ('target_requires_write', 'passive_task'):
        if key in grader:
            flags.append(f'{key}={grader[key]}')
    out.append('DB: ' + ', '.join(flags))
    for key in ('unmet_communicate', 'unmet_nl_assertions'):
        if grader.get(key):
            out.append(f'  {key.upper()}: ' + ' | '.join(cut(x, None if full else 160)
                                                         for x in grader[key]))
    for line in grader.get('db_diff') or []:
        text = collapse(line)
        if not text or text == '[agent-side DB]':
            continue
        if text.startswith('=>'):
            if full:
                out.append('  ' + text)
            continue
        out.append('  ' + (text if full else compress_diff_line(text)))
    return out


def kind_lines(label, full, prose):
    kind = label['kind']
    if kind == 'failure_attribution':
        return [f'FAILED CONDITION: {cut(label.get("failed_condition"), prose)}']
    if kind == 'fake_pass':
        reason = re.sub(r'^\s*(4B\+auto|reviewer)\s*:\s*', '', collapse(label.get('verified_reason')))
        out = [f'FAKE PASS BECAUSE: {cut(reason, prose)}']
        if label.get('deterministic_flags'):
            out.append('FLAGS: ' + ', '.join(label['deterministic_flags']))
        return out
    out = ['MINOR FLAGS: ' + (', '.join(label.get('minor_flags') or []) or 'none')]
    if label.get('note'):
        out.append(f'NOTE: {cut(label["note"], prose)}')
    return out


def collect_notes(label, full, prose):
    """turn_idx -> list of rendered note blocks."""
    notes = {}

    def add(turn, head, body):
        notes.setdefault(turn, []).append((head, body))

    kind = label['kind']
    if kind == 'failure_attribution':
        rc = label.get('root_cause') or {}
        if rc.get('turn_idx') is not None:
            body = [('SHOULD HAVE', rc.get('should_have'))]
            if full:
                body += [('WHY', rc.get('why_it_caused_the_failure')),
                         ('POLICY', rc.get('policy_quote'))]
            add(rc['turn_idx'], NOTE_HEAD['root_cause'], [(cut(rc.get('what_happened'), prose), body)])
        for err in label.get('contributing_errors') or []:
            if err.get('turn_idx') is None:
                continue
            add(err['turn_idx'], NOTE_HEAD['contributing'],
                [(cut(err.get('what_happened'), prose), [('SHOULD HAVE', err.get('should_have'))])])
    elif kind == 'fake_pass':
        for issue in label.get('issues') or []:
            if issue.get('turn_idx') is None:
                continue
            head = f'ISSUE ({issue.get("category")})'
            if full and issue.get('severity'):
                head += f' · {issue["severity"]}'
            body = [('SHOULD HAVE', issue.get('should_have'))]
            if full:
                body.append(('EVIDENCE', issue.get('evidence')))
            add(issue['turn_idx'], head, [(cut(issue.get('what_happened'), prose), body)])
    return notes


def render_notes(entries, prose):
    out = []
    for head, items in entries:
        for what, body in items:
            out.append(f'     ** {head}: {what}')
            for tag, value in body:
                if value:
                    out.append(f'        {tag}: {cut(value, prose)}')
    return out


def keep_turns(messages, notes, head=2, tail=2, window=1, anchor_calls=False):
    """Which turn_idx survive elision: the first `head`, the last `tail`, and `window` either
    side of every turn carrying a note. fake_pass records have no root_cause - their anchors are
    `issues` - so anchoring on notes rather than on root_cause alone is what keeps them intact."""
    turns = [m['turn_idx'] for m in messages]
    order = {t: i for i, t in enumerate(turns)}
    keep = set(turns[:head]) | set(turns[-tail:] if tail else [])
    for anchor in notes:
        i = order.get(anchor)
        if i is None:
            continue
        keep |= set(turns[max(0, i - window): i + window + 1])
    if anchor_calls and not notes:
        # Only as a fallback, for a record with no notes to anchor on (typically a clean_pass).
        # Otherwise such a record keeps head and tail only, which for a correct trajectory
        # deletes the lesson: the sequence of checks it ran. Applied to records WITH notes it
        # would pull in most of the conversation, since the agent calls tools so often.
        for m in messages:
            if not m.get('tool_calls'):
                continue
            i = order[m['turn_idx']]
            keep |= set(turns[i: i + 2])
    # A tool result without the call that produced it reads as data appearing from nowhere, and
    # a call whose result is missing hides what it returned. Keep such pairs together.
    for i, m in enumerate(messages):
        t = m['turn_idx']
        if m.get('tool_calls') and t in keep and i + 1 < len(messages) \
                and messages[i + 1]['role'] == 'tool':
            keep.add(messages[i + 1]['turn_idx'])
        if m['role'] == 'tool' and t in keep and i and messages[i - 1].get('tool_calls'):
            keep.add(messages[i - 1]['turn_idx'])
    return keep


def render(record, profile, with_id, text_limit=160, tool_limit=400, prose_limit=200,
           elide=False, head=2, tail=2, window=1, anchor_calls=False,
           full_text_limit=None, full_tool_limit=None, drop_fields=None, priority=None,
           drop_nulls=False):
    full = profile == 'full'
    limit = full_text_limit if full else text_limit
    prose = None if full else prose_limit
    label = record['label']
    task = record.get('task') or {}
    # retail tasks carry no `purpose`; the customer's brief is the next best one-line statement
    # of what the task is, and it is what the Task line is for.
    purpose = task.get('purpose') or ((task.get('user_scenario') or {}).get('instructions')
                                      or {}).get('reason_for_call')
    lines = [f'Task: {cut(purpose, limit)}',
             f'KIND: {label["kind"]}']
    lines += kind_lines(label, full, prose)
    lines.append(f'SUMMARY: {cut(label.get("summary"), prose)}')
    lines += db_lines(record, full)
    lines += ['', 'Chat History:']

    notes = collect_notes(label, full, prose)
    keep = (keep_turns(record['messages'], notes, head, tail, window, anchor_calls)
            if elide else None)
    # In `full`, a tool result away from the annotated turns is reduced to a stub: the evidence
    # the label talks about is worth its characters, the rest is not.
    turns_seq = [m['turn_idx'] for m in record['messages']]
    pos = {t: i for i, t in enumerate(turns_seq)}
    near = set()
    for anchor in notes:
        i = pos.get(anchor)
        if i is not None:
            near |= set(turns_seq[max(0, i - window): i + window + 1])
    printed = set()
    gap = 0
    for msg in record['messages']:
        turn = msg['turn_idx']
        if keep is not None and turn not in keep:
            gap += 1
            continue
        if gap:
            lines.append(f'… {gap} turn{"s" if gap > 1 else ""} omitted …')
            gap = 0
        role = msg['role']
        role = ROLE_NAMES.get(role, role)
        if role == 'tool':
            if full:
                body = collapse(msg.get('content'))
                budget = None if (full_tool_limit is None or turn in near) else full_tool_limit
            else:
                body = (compact_tool(msg.get('content') or '', drop_fields, priority, drop_nulls)
                        or collapse(msg.get('content')))
                budget = tool_limit
            lines.append(f'#{turn} tool: {cut(body, budget)}')
        else:
            if msg.get('content'):
                lines.append(f'#{turn} {role}: {cut(msg["content"], limit)}')
            for call in (msg.get('tool_calls') or []):
                name = call.get('name') or (call.get('function') or {}).get('name')
                args = call.get('arguments') or (call.get('function') or {}).get('arguments') or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        pass
                lines.append(f'#{turn} {role} -> {name}('
                             f'{json.dumps(args, ensure_ascii=False, separators=(",", ":"))})')
        if turn in notes and turn not in printed:
            lines += render_notes(notes[turn], prose)
            printed.add(turn)
    if gap:
        lines.append(f'… {gap} turn{"s" if gap > 1 else ""} omitted …')
    # A note whose turn was elided is appended rather than dropped.
    for turn in sorted(set(notes) - printed):
        lines += render_notes(notes[turn], prose)
    if with_id:
        lines.append(f'ID: {record.get("sim_id")}')
    return '\n'.join(lines)


def load(path):
    path = Path(path)
    if path.suffix == '.jsonl':
        rows = [json.loads(l) for l in path.open()]
    else:
        blob = json.loads(path.read_text())
        rows = blob['records'] if isinstance(blob, dict) and 'records' in blob else blob
    out = []
    for r in rows:
        rec = r.get('source_record', r)
        rec.setdefault('sim_id', r.get('sim_id'))
        rec.setdefault('task_id', r.get('task_id'))
        rec.setdefault('trial', r.get('trial'))
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source')
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--profile', choices=['digest', 'full'], default='digest')
    ap.add_argument('--no-id', action='store_true')
    ap.add_argument('--text-limit', type=int, default=160,
                    help='digest only: char budget for utterances and the Task line')
    ap.add_argument('--elide', action='store_true',
                    help='digest only: keep the first/last turns and a window around each note')
    ap.add_argument('--head', type=int, default=2)
    ap.add_argument('--tail', type=int, default=2)
    ap.add_argument('--window', type=int, default=1)
    ap.add_argument('--anchor-calls', action='store_true',
                    help='also anchor on turns that make a tool call')
    ap.add_argument('--prose-limit', type=int, default=200,
                    help='digest only: char budget for model-written label prose')
    ap.add_argument('--tool-limit', type=int, default=400,
                    help='digest only: char budget for a compacted tool result')
    ap.add_argument('--full-text-limit', type=int, default=None,
                    help='full only: char budget for utterances')
    ap.add_argument('--full-tool-limit', type=int, default=None,
                    help='full only: char budget for a tool result away from an annotated turn')
    args = ap.parse_args()

    records = load(args.source)
    records.sort(key=lambda r: (int(r.get('task_id', 0)), r.get('trial', 0)))
    blocks = [render(r, args.profile, not args.no_id, args.text_limit, args.tool_limit,
                     args.prose_limit, args.elide, args.head, args.tail, args.window,
                     args.anchor_calls, args.full_text_limit, args.full_tool_limit)
              for r in records]
    text = SEP.join(blocks) + '\n'
    Path(args.out).write_text(text)

    sizes = [len(b) for b in blocks]
    print(f'{len(blocks)} records -> {args.out}')
    print(f'  profile {args.profile} | {len(text):,} chars ≈ {len(text)/4:,.0f} tokens | '
          f'per record mean {sum(sizes)/len(sizes):,.0f} chars ≈ {sum(sizes)/len(sizes)/4:,.0f} tokens '
          f'(max {max(sizes):,})')


if __name__ == '__main__':
    main()
