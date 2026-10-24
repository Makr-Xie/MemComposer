#!/usr/bin/env python
"""Self-review: each executor model critiques its own ALFWorld trajectories.

    python self_review.py --model ministral8b --pilot 48 --tag pilot      # 8 per task type
    python self_review.py --model qwen3-4b --all --tag full
    python self_review.py --model ministral8b --pilot 48 --dry-run 6      # write prompts, no calls

One call per trajectory at temperature 0. The answer goes through a mechanical audit; if it is
rejected (or does not parse) the critic gets the reasons and one retry at temperature 0.3. A
trajectory whose review still fails is DROPPED from the memory base. Wins caused by the
double-inReceptacle quirk (facts `env_quirk_early_win`) are not reviewed at all.

Output: memory_banks/<model>/_work/reviews_<tag>.jsonl, one row per trajectory, resumable.

Critic endpoints: Bedrock (BEDROCK_BASE_URL / BEDROCK_API_KEY) for ministral8b; for qwen3-4b a
local vLLM, ALFW_CRITIC_BASE (comma-separated for several servers).
"""
import argparse
import collections
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import paths                                                            # noqa: E402
sys.path.insert(0, str(paths.RUNTIME))

import self_review_prompt as P                                          # noqa: E402

MODELS = {
    'ministral8b': {'model': 'mistral.ministral-3-8b-instruct', 'base': None, 'concurrency': 16},
    'qwen3-4b': {'model': 'Qwen3-4B-Instruct-2507',
                 'base': os.environ.get('ALFW_CRITIC_BASE', 'http://127.0.0.1:8040/v1,http://127.0.0.1:8042/v1'),
                 'concurrency': 24},
}
CRITIC_VERSION = 'self_review'

FAIL_KEYS = {'failed_condition', 'failed_stage', 'blame', 'root_cause', 'contributing_errors', 'summary'}
SUCC_KEYS = {'key_steps', 'reusable_strategy', 'wasted_steps', 'summary'}
MAX_CONTRIB, MAX_KEY = 3, 4

# Gate 6: advice that contradicts the rules. Checked only in the fields that give advice
# (summary, strategy, why_it_mattered); describing what the agent wrongly did is not advice.
# A sentence carrying a negation ("not", "instead of", ...) is skipped: it is warning against it.
# The verb must act on the device directly ("heat <obj> on the stove"): a participle used as an
# adjective ("put the heated apple in the fridge") describes a correct placement and is skipped.
_OBJ = r'(?:(?:the|a|an|it|them|this|that)\s+)?(?:[a-z]+\s+){0,2}?'
RULE_VIOLATIONS = [
    ('heat_not_microwave', re.compile(r'\b(heat|heats|heating|heated)\s+' + _OBJ + r'(on|with|using|in|inside|at)\s+'
                                      r'(?:the |a )?(stove ?burner|stove|oven|toaster|pan|pot|fridge|sink ?basin)\b')),
    ('cool_not_fridge', re.compile(r'\b(cool|cools|cooling|cooled|chill|chilling)\s+' + _OBJ + r'(with|using|in|inside|at|under)\s+'
                                   r'(?:the |a )?(sink ?basin|sink|water|microwave|stove ?burner|freezer)\b')),
    ('clean_not_sink', re.compile(r'\b(clean|cleans|cleaning|cleaned|wash|washing|rinse|rinsing)\s+' + _OBJ +
                                  r'(with|using|in|inside|at)\s+(?:the |a )?(fridge|microwave|stove ?burner|dishwasher)\b')),
    ('microwave_controls', re.compile(r'\b(turn on|start|switch on|run|press)\s+(?:the )?microwave\b')),
    ('avoid_fridge', re.compile(r'\bavoid\w*\s+(?:the )?fridge\b')),
    ('container_heats', re.compile(r'\b(put|place|move|leave)\w*\s+' + _OBJ + r'(in|into|inside)\s+(?:the )?'
                                   r'(microwave|fridge)\s+(to|and|so it)\s+(heat|cool|warm|chill)')),
]
ADJ_BEFORE = re.compile(r'\b(the|a|an)\s+$')
NEGATION = re.compile(r"\b(not|never|instead|rather than|n't|no longer|without|does nothing|cannot|can't)\b", re.I)
QUOTE = re.compile(r'["“]([^"”]{8,})["”]')


def norm(s):
    return ' '.join(str(s or '').lower().strip().rstrip('.').split())


def step_table(rec):
    """step -> (executed action, admissible list shown at that executed call)."""
    return {t['step']: (t['action'], t.get('admissible') or [])
            for t in rec['trajectory'] if t['status'] == 'ok'}


def corpus(rec):
    parts = [rec['task'], rec['init_obs']]
    for t in rec['trajectory']:
        parts += [t.get('obs') or '', t.get('think') or '', t.get('action') or '']
    return norm(' '.join(parts))


def stage_consistent(stage, f, rec=None):
    """Gate 7: the critic's stage need not equal the mechanical one, but must fit the facts.

    pick_two counts instances: "never found" there means the SECOND object was never seen, which
    is true whenever fewer than two distinct targets ever appeared."""
    p, t = f['progress'], f['target']
    ev = {e['type'] for e in f['events']}
    need = t['transform'] not in (None, 'look')
    seen_n = taken_n = None
    if rec is not None and t['count'] == 2:
        obs = ' '.join(x.get('obs') or '' for x in rec['trajectory'] if x['status'] == 'ok')
        seen_n = len(set(re.findall(rf"\b{t['object']} \d+\b", obs)))
        taken_n = len(set(re.findall(rf"You pick up the ({t['object']} \d+)", obs)))
    two = seen_n is not None
    return {
        'never_found': p['seen'] is None or (two and seen_n < 2),
        'found_not_taken': (p['seen'] is not None and p['taken'] is None) or (two and seen_n >= 2 and taken_n < 2),
        'wrong_object': bool(ev & {'wrong_object', 'picked_non_target'}),
        'not_transformed': need and (p['transformed'] is None or bool(ev & {'placed_untransformed', 'wrong_transform'})),
        'undid_transform': 'undid_transform' in ev,
        'appliance_as_container': 'appliance_as_container' in ev,
        'transformed_not_placed': need and p['transformed'] is not None and p['placed'] is None,
        'taken_not_placed': (not need) and t['transform'] != 'look' and p['taken'] is not None and p['placed'] is None,
        'wrong_receptacle': bool(ev & {'wrong_receptacle', 'split_pair', 'put_down'}),
        'pick_two_one_placed': t['count'] == 2 and p.get('placed_count', 0) >= 1,
        'look_lamp_not_on': t['transform'] == 'look' and p['taken'] is not None,
    }.get(stage, False)


def audit(rec, f, out):
    """Return (ok, reasons, flags, cleaned). Hard failures go in reasons; soft ones in flags."""
    reasons, flags = [], []
    table = step_table(rec)
    text = corpus(rec)
    vocab = {m.split(' ')[0] for m in re.findall(r'\b[a-z]+ \d+\b', text)} - {'step', 'steps'}
    won = bool(f['won'])
    keys = SUCC_KEYS if won else FAIL_KEYS
    if not isinstance(out, dict):
        return False, ['1: the answer is not a JSON object'], flags, out
    missing = keys - set(out)
    if missing:
        return False, [f'1: missing fields {sorted(missing)}'], flags, out
    out = {k: out[k] for k in sorted(keys)}     # sorted: set order varies with PYTHONHASHSEED

    every_action = {norm(x) for _, adm in table.values() for x in adm}
    room = {norm(r) for r in re.findall(r'\b[a-z]+ \d+\b', rec['init_obs'].split('Your task is to:')[0])}
    strip_num = lambda x: re.sub(r'\s+\d+\b', '', norm(x))
    known_inst = set(re.findall(r'\b[a-z]+ \d+\b', text))
    look_task = f['target']['transform'] == 'look'

    def lamp_home(lamp):
        """Receptacle a desklamp was listed on, if the agent ever saw it."""
        for t in rec['trajectory']:
            mm = re.search(rf'On the ([a-z]+ \d+), you see [^.]*\b{lamp}\b', t.get('obs') or '')
            if mm:
                return mm.group(1)
        return None

    def well_formed(x):
        """An ALFWorld command over instances that exist in this game, e.g. `use desklamp 1` when
        the agent never reached the lamp: never typeable in this run, but a real move."""
        x = norm(x)
        pats = [r'take ([a-z]+ \d+) from ([a-z]+ \d+)', r'move ([a-z]+ \d+) to ([a-z]+ \d+)',
                r'(?:heat|cool|clean) ([a-z]+ \d+) with ((?:microwave|fridge|sinkbasin) \d+)',
                r'use (desklamp \d+)', r'(?:open|close) ([a-z]+ \d+)', r'examine ([a-z]+ \d+)']
        for pat in pats:
            mm = re.fullmatch(pat, x)
            if mm:
                return all(g in known_inst for g in mm.groups())
        return False

    def check_step(where, d, step_key='step', should_key='should_have', require_should=True, step=None):
        """Validate one cited step in place. Near misses are repaired (and flagged), not rejected:
        the review only has to be roughly right."""
        step = d.get(step_key) if step is None else step
        if not isinstance(step, int) or step not in table:
            reasons.append(f'2: {where} cites step {step!r}, which is not an executed [STEP n]')
            return
        act, adm = table[step]
        if 'action' in d or step_key == 'step':
            action = d.get('action')
            if action is None or norm(action) != norm(act):
                near = [x for x in sorted(table, key=lambda x: abs(x - step))
                        if abs(x - step) <= 2 and action and norm(table[x][0]) == norm(action)]
                if action is None or strip_num(action) == strip_num(act):
                    flags.append(f'3r: {where} action restored to "{act}"')
                    d['action'] = act
                elif near and step_key == 'step':
                    # right action, step number off by one or two: point at where it happened
                    flags.append(f'2r: {where} step {step} -> {near[0]} (where "{action}" was done)')
                    step = d['step'] = near[0]
                    act, adm = table[step]
                else:
                    reasons.append(f'3: {where} says the action at step {step} was "{action}", but it was "{act}"')
        should = d.get(should_key) if should_key else None
        if should_key is None:
            return
        if not should:
            if require_should:
                reasons.append(f'4: {where} has no should_have')
            return
        here = {norm(x): x for x in adm}
        if norm(should) == norm(act):
            reasons.append(f'4: {where} should_have "{should}" is what the agent actually did at step {step}')
        elif norm(should) in here:
            pass
        elif strip_num(should) in {strip_num(x) for x in adm} and \
                sum(strip_num(x) == strip_num(should) for x in adm) == 1:
            fixed = next(x for x in adm if strip_num(x) == strip_num(should))
            flags.append(f'4r: {where} should_have "{should}" -> "{fixed}"')
            d[should_key] = fixed
        elif re.fullmatch(r'go to (desklamp \d+)', norm(should)) and lamp_home(norm(should)[6:]):
            # the lamp is an object, not a place: go to where it was seen standing
            fixed = f'go to {lamp_home(norm(should)[6:])}'
            flags.append(f'4r: {where} should_have "{should}" -> "{fixed}"')
            d[should_key] = fixed
        elif norm(should) in every_action or (norm(should).startswith('go to ') and norm(should)[6:] in room) \
                or well_formed(should) or (look_task and re.fullmatch(r'(use|go to) desklamp \d+', norm(should))):
            # a real action in this game, just not typeable at that step (it names the goal,
            # not the next move): keep it, flag it
            flags.append(f'4s: {where} should_have "{should}" is a real action but not available at step {step}')
        else:
            avail = [x for x in adm if not x.startswith(('go to ', 'help', 'inventory', 'look'))][:15]
            reasons.append(f'4: {where} should_have "{should}" is not an action this game accepts. Available at '
                           f'step {step}: {"; ".join(avail) or "(nothing but go to)"}; or `go to <receptacle>`')
        inst = f.get('target_location_instances') or []
        if any(i in norm(d.get(should_key))  for i in inst):
            seen_before = any(i in norm(t.get('obs')) for t in rec['trajectory']
                              if t['status'] == 'ok' and t['step'] < step for i in inst)
            if not seen_before:
                flags.append(f'11: {where} should_have points at the answer-key location before the agent had seen it')

    advice_fields = []
    if won:
        ks = out['key_steps']
        if not isinstance(ks, list) or not ks:
            reasons.append('1: key_steps must be a non-empty list')
            ks = []
        if len(ks) > MAX_KEY:
            flags.append(f'key_steps cut from {len(ks)} to {MAX_KEY}')
            ks = ks[:MAX_KEY]
        out['key_steps'] = ks
        for i, k in enumerate(ks):
            if not isinstance(k, dict):
                reasons.append(f'1: key_steps[{i}] is not an object')
                continue
            check_step(f'key_steps[{i}]', k, should_key=None)
            advice_fields.append(k.get('why_it_mattered'))
        cited = {k.get('step') for k in ks if isinstance(k, dict)}
        need = f['target']['transform'] not in (None, 'look')
        # A missing winning (or processing) step is the commonest near miss -- pick_two needs four
        # key moves and the critic stops after the third. Add it mechanically and flag it.
        must = [(f['won_step'], 'the final step that completed the goal')]
        if need and f['progress']['transformed']:
            must.append((f['progress']['transformed'], f'the required {f["target"]["transform"]} step'))
        for st_, why in must:
            if st_ not in cited and st_ in table:
                ks.append({'step': st_, 'action': table[st_][0], 'why_it_mattered': why[0].upper() + why[1:] + '.'})
                cited.add(st_)
                flags.append(f'8r: key step {st_} added')
        ks.sort(key=lambda k: k.get('step') if isinstance(k, dict) and isinstance(k.get('step'), int) else 0)
        kept = []
        for i, w in enumerate(out['wasted_steps'] if isinstance(out['wasted_steps'], list) else []):
            m = re.fullmatch(r'\s*(\d+)\s*-\s*(\d+)\s*', str((w or {}).get('steps', '')))
            if not m or int(m.group(2)) - int(m.group(1)) < 2:
                flags.append(f'9: wasted_steps[{i}] "{(w or {}).get("steps")}" is shorter than 3 steps; dropped')
                continue
            check_step(f'wasted_steps[{i}]', w, step_key='steps', step=int(m.group(1)))
            kept.append(w)
        out['wasted_steps'] = kept
        advice_fields += [out['reusable_strategy']]
        plain = ['summary', 'reusable_strategy']
    else:
        if out['failed_stage'] not in P.STAGES:
            reasons.append(f'1: failed_stage "{out["failed_stage"]}" is not one of the listed stages')
        elif not stage_consistent(out['failed_stage'], f, rec):
            reasons.append(f'7: failed_stage "{out["failed_stage"]}" contradicts the facts '
                           f'(mechanical stage: {f["failed_stage"]})')
        if out['blame'] not in ('agent', 'step_budget'):
            reasons.append(f'1: blame "{out["blame"]}" must be "agent" or "step_budget"')
        elif out['blame'] == 'step_budget' and not f['ran_out_after_progress']:
            flags.append('blame=step_budget although the facts show no progress near the end')
        rc = out['root_cause']
        if not isinstance(rc, dict):
            reasons.append('1: root_cause is not an object')
        else:
            check_step('root_cause', rc, require_should=out['blame'] != 'step_budget')
        ce = out['contributing_errors'] if isinstance(out['contributing_errors'], list) else []
        if len(ce) > MAX_CONTRIB:
            flags.append(f'contributing_errors cut from {len(ce)} to {MAX_CONTRIB}')
            ce = ce[:MAX_CONTRIB]
        out['contributing_errors'] = ce
        for i, c in enumerate(ce):
            if not isinstance(c, dict):
                reasons.append(f'1: contributing_errors[{i}] is not an object')
                continue
            check_step(f'contributing_errors[{i}]', c)
        plain = ['summary']
    advice_fields.append(out['summary'])

    # gate 5: every quoted observation must be in the log (quotes INSIDE the string values)
    def strings(x):
        if isinstance(x, str):
            yield x
        elif isinstance(x, dict):
            for v in x.values():
                yield from strings(v)
        elif isinstance(x, list):
            for v in x:
                yield from strings(v)
    for sv in strings(out):
        for q in QUOTE.findall(sv):
            if norm(q) not in text:
                reasons.append(f'5: the quote "{q[:80]}" does not appear in the trajectory')
    # gate 6: advice that contradicts the rules
    for fld in advice_fields:
        for sent in re.split(r'(?<=[.;!?])\s+', str(fld or '')):
            if NEGATION.search(sent):
                continue
            low = sent.lower()
            for name, rx in RULE_VIOLATIONS:
                for mm in rx.finditer(low):
                    if ADJ_BEFORE.search(low[:mm.start()]):
                        continue            # "the heated apple ...": an adjective, not an instruction
                    reasons.append(f'6: "{sent.strip()[:100]}" contradicts the rules ({name})')
                    break
    # gate 10: no instance numbers in the general-purpose text -- stripped mechanically
    for key in plain:
        txt = str(out.get(key) or '')
        fixed = re.sub(r'\b([a-z]+) \d+\b', lambda m: m.group(1) if m.group(1) in vocab else m.group(0), txt)
        if fixed != txt:
            flags.append(f'10r: instance numbers removed from {key}')
            out[key] = fixed
    return not reasons, reasons, flags, out


def _uncomment(s):
    """Drop `// ...` comments that sit outside strings (some models annotate their own JSON) and
    trailing commas."""
    out, i, n, in_str = [], 0, len(s), False
    while i < n:
        ch = s[i]
        if in_str:
            out.append(ch)
            if ch == '\\' and i + 1 < n:
                out.append(s[i + 1])
                i += 1
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
            out.append(ch)
        elif s.startswith('//', i):
            while i < n and s[i] != '\n':
                i += 1
            continue
        else:
            out.append(ch)
        i += 1
    return re.sub(r',(\s*[}\]])', r'\1', ''.join(out))


def parse(txt):
    """The answer's JSON object. Tolerant: several fenced blocks (a "corrected JSON" after the
    first) are tried last-first, and comments / trailing commas are removed if strict parsing fails."""
    txt = (txt or '').strip()
    blocks = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', txt, re.S)[::-1]
    m = re.search(r'\{.*\}', txt, re.S)
    cands = blocks + ([m.group(0)] if m else [txt])
    err = None
    for c in cands:
        for variant in (c, _uncomment(c)):
            try:
                return json.loads(variant, strict=False)
            except Exception as e:
                err = e
    raise err


def review_one(chat, rec, f, walkthrough):
    system, user = P.build(rec, f, walkthrough)
    msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
    attempts = []
    for attempt in range(2):
        temp = 0.0 if attempt == 0 else 0.3
        raw = chat(msgs, temperature=temp)
        usage = getattr(chat, 'last_usage', {}) or {}
        try:
            out = parse(raw)
            ok, reasons, flags, clean = audit(rec, f, out)
        except Exception as e:
            ok, reasons, flags, clean = False, [f'1: the answer is not valid JSON ({type(e).__name__})'], [], None
        attempts.append({'temperature': temp, 'raw': raw, 'ok': ok, 'reasons': reasons, 'flags': flags,
                         'prompt_tokens': usage.get('prompt_tokens', 0),
                         'completion_tokens': usage.get('completion_tokens', 0)})
        if ok:
            return {'review': clean, 'usable': True, 'flags': flags, 'attempts': attempts}
        msgs = msgs[:2] + [{'role': 'assistant', 'content': raw},
                           {'role': 'user', 'content': P.RETRY.format(
                               reasons='\n'.join(f'- {r}' for r in reasons), previous=raw.strip()[:6000])}]
    return {'review': None, 'usable': False, 'flags': attempts[-1]['flags'], 'attempts': attempts}


def pick_pilot(facts, n, seed):
    """n trajectories, spread evenly over task types, half won and half lost where possible."""
    by = collections.defaultdict(lambda: {0: [], 1: []})
    for t, f in facts.items():
        by[f['task_type']][f['won']].append(t)
    per = n // len(by)
    rnd = random.Random(seed)
    out = []
    for tt in sorted(by):
        w, l = sorted(by[tt][1]), sorted(by[tt][0])
        rnd.shuffle(w)
        rnd.shuffle(l)
        k = min(len(w), per // 2)
        out += w[:k] + l[:per - k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=sorted(MODELS))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--pilot', type=int, help='stratified sample of this many trajectories')
    g.add_argument('--all', action='store_true')
    g.add_argument('--task-ids-file')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--concurrency', type=int, default=None)
    ap.add_argument('--dry-run', type=int, default=0, help='write this many prompts and exit, no model calls')
    a = ap.parse_args()

    work = paths.work(a.model)
    raw = {json.loads(l)['task_id']: json.loads(l)
           for l in open(paths.rollouts(a.model))}
    facts = {json.loads(l)['task_id']: json.loads(l) for l in open(work / 'facts.jsonl')}
    quirk = {t for t, f in facts.items() if f['env_quirk_early_win']}
    eligible = {t: f for t, f in facts.items() if t not in quirk}
    if a.pilot:
        ids = pick_pilot(eligible, a.pilot, a.seed)
    elif a.task_ids_file:
        ids = [l.strip() for l in open(a.task_ids_file) if l.strip() and l.strip() not in quirk]
    else:
        ids = sorted(eligible)

    if a.dry_run:
        path = work / f'prompts_{a.tag}_dryrun.txt'
        with path.open('w') as fh:
            for t in ids[:a.dry_run]:
                wt = json.load(open(paths.gamefile(t)))['walkthrough']
                system, user = P.build(raw[t], facts[t], wt)
                fh.write(f"{'=' * 100}\nTASK {t}  won={facts[t]['won']}\n{'-' * 40} SYSTEM\n{system}\n"
                         f"{'-' * 40} USER\n{user}\n\n")
        print(f'{min(a.dry_run, len(ids))} prompts -> {path}  (of {len(ids)} selected, {len(quirk)} quirk wins skipped)')
        return

    from llm import Chat
    cfg = MODELS[a.model]
    chat = Chat(cfg['model'], base_url=cfg['base'], api_key='local' if cfg['base'] else None,
                temperature=0.0, max_tokens=3500)
    out = work / f'reviews_{a.tag}.jsonl'
    done = set()
    if out.exists():
        done = {json.loads(l)['task_id'] for l in open(out)}
    todo = [t for t in ids if t not in done]
    conc = a.concurrency or cfg['concurrency']
    print(f'{a.model}: {len(ids)} selected ({len(quirk)} quirk wins skipped), {len(done)} done, '
          f'{len(todo)} to run; critic {cfg["model"]} {CRITIC_VERSION} concurrency {conc}', flush=True)

    lock = threading.Lock()
    fh = out.open('a')
    c = collections.Counter()
    t0 = time.time()

    def work_one(t):
        rec, f = raw[t], facts[t]
        wt = json.load(open(paths.gamefile(t)))['walkthrough']
        try:
            r = review_one(chat, rec, f, wt)
        except Exception as e:
            r = {'review': None, 'usable': False, 'flags': [], 'attempts': [], 'error': f'{type(e).__name__}: {e}'}
        row = {'task_id': t, 'won': f['won'], 'task_type': f['task_type'], 'critic': cfg['model'],
               'critic_version': CRITIC_VERSION, **r}
        with lock:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
            fh.flush()
            c['n'] += 1
            c['usable'] += r['usable']
            c['retried'] += len(r['attempts']) > 1
            if c['n'] % 10 == 0 or c['n'] == len(todo):
                print(f"  {c['n']}/{len(todo)} usable {c['usable']} retried {c['retried']} "
                      f"[{(time.time() - t0) / 60:.1f} min]", flush=True)

    with ThreadPoolExecutor(max_workers=conc) as ex:
        list(ex.map(work_one, todo))
    fh.close()
    print(f'-> {out}  LLM calls {chat.calls} tokens in/out {chat.prompt_tokens}/{chat.completion_tokens}')


if __name__ == '__main__':
    main()
