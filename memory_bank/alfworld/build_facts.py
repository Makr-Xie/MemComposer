#!/usr/bin/env python
"""Mechanical facts for every raw trajectory: no model involved, every field is computed.

    python build_facts.py --model ministral8b          # -> memory_banks/ministral8b/_work/facts.jsonl

The facts serve twice: as evidence handed to the self-review critic (the ALFWorld stand-in for
tau2's db_diff) and, rendered, as the FACTS block the executor reads.

The state tracker replays each trajectory from the observation text alone (ALFWorld's feedback is
templated, alfworld/data/alfred.twl2) and decides after every step whether the goal holds. That
verdict is compared with the environment's own `won` / `won_step`: any disagreement means a rule
here is wrong, which is the check that keeps the facts honest.

Object names are matched as whole words plus instance number ("pot 1" never matches "potato 1",
"knife 2" never matches "butterknife 2").

Two models of the room are kept apart: `State` is what is true (as far as the observations
reveal), `known` is what the agent has already seen of each receptacle, updated by its own takes
and moves. A return to a receptacle is a revisit only when it reveals nothing the agent did not
already know -- the listing text alone is not enough, because the agent's own pick-up changes it.
"""
import argparse
import collections
import json
import re
from pathlib import Path

import paths

TRANSFORM = {                       # task_type -> (verb, appliance type, state it sets)
    'pick_heat_then_place_in_recep': ('heat', 'microwave', 'hot'),
    'pick_cool_then_place_in_recep': ('cool', 'fridge', 'cold'),
    'pick_clean_then_place_in_recep': ('clean', 'sinkbasin', 'clean'),
    'look_at_obj_in_light': ('use', 'desklamp', None),
}
STATE_OF = {'heat': 'hot', 'cool': 'cold', 'clean': 'clean'}
UNDO = {'hot': 'cold', 'cold': 'hot'}           # heating clears cold and vice versa; clean sticks
INFO_VERBS = ('examine', 'inventory', 'look', 'help')
# Choosing one of these instead of an offered transform is not "declining" it: opening the
# microwave before heating is the normal sequence, the heat action is merely already listed.
NEUTRAL_VERBS = ('open', 'close', 'examine', 'inventory', 'look', 'help')

BLOCKING = {                        # event type -> failed_stage it maps to
    'wrong_object': 'wrong_object',
    'placed_untransformed': 'not_transformed',
    'wrong_transform': 'not_transformed',
    'appliance_as_container': 'appliance_as_container',
    'undid_transform': 'undid_transform',
    'wrong_receptacle': 'wrong_receptacle',
}

INST = r'[a-z]+ \d+'
RE_ARRIVE = re.compile(rf'^You arrive at ({INST})\.\s*(.*)$', re.S)
RE_OPEN = re.compile(rf'^You open the ({INST})\.\s*(.*)$', re.S)
RE_CLOSE = re.compile(rf'^You close the ({INST})\.')
RE_TAKE = re.compile(rf'^You pick up the ({INST}) from the ({INST})\.')
RE_MOVE = re.compile(rf'^You move the ({INST}) to the ({INST})\.')
RE_XFORM = re.compile(rf'^You (heat|cool|clean) the ({INST}) using the ({INST})\.')
RE_TOGGLE = re.compile(rf'^You turn (on|off) the ({INST})\.')
RE_LIST = re.compile(rf'(?:On the ({INST}), you see|The ({INST}) is open\. In it, you see'
                     rf'|This is a normal ({INST})\. In it, you see) (.+?)\.\s*$', re.S)
RE_CLOSED = re.compile(rf'The ({INST}) is closed\.')
RE_FACING = re.compile(r'^You are facing the (.+?)\. Next to it, you see (.+?)\.\s*$', re.S)
RE_ITEMS = re.compile(rf'\ban? ({INST})')
RE_INIT_IN = re.compile(r'\(inReceptacle (\S+) (\S+)\)')


def typ(inst):
    return inst.rsplit(' ', 1)[0] if inst else None


def items(listing):
    return [] if listing.strip() == 'nothing' else RE_ITEMS.findall(listing)


def listing_of(text):
    """(receptacle, frozenset of contents) if the text lists a receptacle's contents, else None."""
    lm = RE_LIST.search(text)
    if not lm:
        return None
    return (lm.group(1) or lm.group(2) or lm.group(3)), frozenset(items(lm.group(4)))


def pddl_type(name):
    """PDDL id -> ALFWorld receptacle type. Basins carry their real type as the last segment:
    `Sink_bar__minus_00_dot_1_bar_..._bar_SinkBasin` is a sinkbasin, not a "sink"."""
    last = name.rsplit('_bar_', 1)[-1]
    return (last if last.isalpha() else name.split('_', 1)[0]).lower()


def initial_locations(pddl_problem, obj):
    """Receptacle types that hold a target-type object at the start, from the PDDL :init.

    The walkthrough names only the one instance the expert used; the agent may find another
    (the potato the expert takes from the countertop, the agent finds in the fridge).
    """
    init = pddl_problem[:pddl_problem.find(':goal')]
    return sorted({pddl_type(r) for o, r in RE_INIT_IN.findall(init)
                   if o.split('_', 1)[0].lower() == obj})


class State:
    """What is true in the room, as far as the observations reveal it."""

    def __init__(self):
        self.loc = None
        self.held = None
        self.where = {}                     # object instance -> receptacle instance | 'hand'
        self.states = collections.defaultdict(set)   # 'hot'/'cold'/'clean' -> instances
        self.lamps_on = set()
        self.lamp_at = {}

    def see(self, recep, objs):
        for o in objs:
            # The double-inReceptacle quirk lists a held object at its old receptacle; the hand
            # wins, or the tracker would "agree" with a win that never had a placement.
            if o == self.held:
                continue
            self.where[o] = recep
            if typ(o) == 'desklamp':
                self.lamp_at[o] = recep


def goal_met(st, task_type, obj, recep):
    if task_type == 'look_at_obj_in_light':
        if not st.held or typ(st.held) != obj:
            return False
        return any(st.lamp_at.get(l) == st.loc for l in st.lamps_on)
    if task_type == 'pick_two_obj_and_place':
        return placed_count(st, obj, recep) >= 2
    need = TRANSFORM.get(task_type, (None, None, None))[2]
    return any(typ(o) == obj and typ(r) == recep and (need is None or o in st.states[need])
               for o, r in st.where.items())


def placed_count(st, obj, recep):
    per = collections.Counter(r for o, r in st.where.items() if typ(o) == obj and typ(r) == recep)
    return max(per.values(), default=0)


def env_steps(rec):
    """The executed steps in order, each with the invalid replies that preceded it."""
    out, pending = [], []
    for t in rec['trajectory']:
        if t['status'] != 'ok':
            pending.append({'step': t['step'], 'action': t['action'], 'status': t['status']})
            continue
        out.append(dict(t, invalid_before=pending))
        pending = []
    return out, pending


def facts_for(rec, walkthrough, pddl_problem=''):
    tt = rec['task_type']
    obj = rec['object_target'].lower()
    recep = rec['parent_target'].lower() or None
    verb, appliance, need = TRANSFORM.get(tt, (None, None, None))
    is_look = tt == 'look_at_obj_in_light'
    is_two = tt == 'pick_two_obj_and_place'
    max_steps = rec['max_steps']
    steps, trailing_invalid = env_steps(rec)
    invalid = [x for t in steps for x in t['invalid_before']] + trailing_invalid
    executed = {t['step'] for t in steps}
    # A step whose every reply was unusable still spends budget (runner burns it after the
    # format retries); it has no executed action but is waste all the same.
    burned = sorted({x['step'] for x in invalid if x['step'] not in executed and x['step'] <= max_steps})

    takes = [a for a in walkthrough if a.startswith('take ')]
    loc_inst = [a.split(' from ', 1)[1] for a in takes]

    st = State()
    known = {}                              # receptacle -> frozenset the agent believes it holds
    visited, opened, examined = set(), set(), set()
    progress = {'seen': None, 'taken': None, 'transformed': None, 'placed': None}
    taken_targets = set()
    max_placed = 0
    events, revisits, info_actions, avail_unused, place_unused = [], [], [], [], []
    steps_by_type = collections.Counter()
    timeline = {}                           # step -> wasted?
    progress_at = {}                        # step -> progress happened
    arrivals = []                           # (step, receptacle, exempted) for the purpose check
    goal_step = None
    last_progress = None

    def ready(o):
        return need is None or o in st.states[need]

    def held_needs_elsewhere(r):
        """Holding the target while it still has to go to a different type of place."""
        h = st.held
        if not h or typ(h) != obj or is_look or is_two:
            return False
        nxt = appliance if (need and h not in st.states[need]) else recep
        return nxt is not None and typ(r) != nxt

    room = set(re.findall(rf'\b({INST})\b', rec['init_obs'].split('Your task is to:')[0]))
    pick_src = {}                           # non-target instance -> receptacle it was taken from
    opened_at = {}                          # step -> receptacle opened at that step

    def learn(text):
        lv = listing_of(text)
        # "This is a normal mug 1. In it, ..." about an object in hand is not a receptacle search
        if lv and (lv[0] == st.held or (room and lv[0] not in room and not RE_ARRIVE.match(text)
                                        and not RE_OPEN.match(text))):
            return None, False
        if lv:
            r, contents = lv
            new = known.get(r) != contents
            known[r] = contents
            return r, new
        return None, False

    for t in steps:
        s, a, obs = t['step'], t['action'], t['obs']
        adm = t.get('admissible') or []
        h = st.held
        target_held = h is not None and typ(h) == obj
        averb = a.split(' ', 1)[0]
        prog, wasted = False, False

        # offers declined: the transform (or lamp) for the held target, and its placement
        if target_held and verb and averb not in NEUTRAL_VERBS:
            if is_look:
                offer = [c for c in adm if c.startswith('use desklamp')]
            else:
                offer = [] if h in st.states[need] else \
                    [c for c in adm if c.startswith(f'{verb} {h} with {appliance}')]
            if offer and a not in offer:
                avail_unused.append(s)
        if target_held and recep and ready(h) and not is_look and averb not in NEUTRAL_VERBS:
            offer = [c for c in adm if c.startswith(f'move {h} to {recep} ')]
            if offer and a not in offer:
                place_unused.append(s)

        before = {k: set(v) for k, v in st.states.items()}
        pc_before = placed_count(st, obj, recep) if is_two else 0

        m = RE_ARRIVE.match(obs)
        if m:
            r, rest = m.group(1), m.group(2)
            first = r not in visited
            lv = listing_of(rest)
            if lv:
                st.see(r, lv[1])
            st.loc = r
            _, new = learn(rest)
            steps_by_type[typ(r)] += 1
            # purposeful: carrying the target to where it must go next, or going for a target
            # the agent already saw here; confirmed later only if progress happens before it leaves
            # (a target the agent itself set down there counts too, unless it already sits in the
            # goal receptacle ready)
            purposeful = (target_held and not is_look and not held_needs_elsewhere(r)) or \
                (not target_held and any(typ(o) == obj and not (typ(r) == recep and ready(o))
                                         for o in known.get(r, ())))
            closed_unknown = bool(RE_CLOSED.search(rest)) and known.get(r) is None
            kind = None
            if first or new:
                wasted = held_needs_elsewhere(r)
            elif purposeful:
                kind = 'purpose'
            elif closed_unknown:
                kind = 'closed'             # seen shut before, never opened: must open it now
                wasted = held_needs_elsewhere(r)
            else:
                revisits.append(s)
                wasted = True
            arrivals.append((s, r, kind))
            visited.add(r)
        elif RE_OPEN.match(obs):
            r = RE_OPEN.match(obs).group(1)
            lv = listing_of(obs)
            if lv:
                st.see(r, lv[1])
            _, new = learn(obs)
            steps_by_type[typ(r)] += 1
            wasted = (not new) or held_needs_elsewhere(r)
            opened.add(r)
            visited.add(r)
            opened_at[s] = r
        elif RE_CLOSE.match(obs):
            steps_by_type[typ(RE_CLOSE.match(obs).group(1))] += 1
            wasted = True
        elif RE_TAKE.match(obs):
            o, src = RE_TAKE.match(obs).groups()
            st.held = o
            st.where[o] = 'hand'
            if src in known:
                known[src] = known[src] - {o}
            if typ(o) == obj:
                if is_two and typ(src) == recep and placed_count(st, obj, recep) < pc_before:
                    events.append({'step': s, 'type': 'removed_from_goal', 'object': o, 'repaired': None})
                    wasted = True
                elif o in taken_targets:
                    wasted = True           # taking it back after setting it down somewhere
                elif progress['taken'] is not None and not is_two:
                    taken_targets.add(o)    # swapping one instance for another: no progress
                    wasted = True
                else:
                    taken_targets.add(o)
                    if progress['taken'] is None:
                        progress['taken'] = s
                    prog = True
            else:
                events.append({'step': s, 'type': 'picked_non_target', 'object': o, 'source': src,
                               'repaired': None})
                pick_src[o] = (src, s)
                wasted = True
        elif RE_MOVE.match(obs):
            o, dst = RE_MOVE.match(obs).groups()
            st.held = None
            st.where[o] = dst
            if dst in known:
                known[dst] = known[dst] | {o}
            if typ(o) == obj:
                if is_look:
                    events.append({'step': s, 'type': 'put_down', 'object': o, 'receptacle': dst, 'repaired': None})
                    wasted = True
                elif typ(dst) == recep:
                    if is_two and ready(o) and placed_count(st, obj, recep) <= pc_before:
                        # the second object went into a different instance than the first
                        events.append({'step': s, 'type': 'split_pair', 'object': o, 'receptacle': dst,
                                       'repaired': None})
                        wasted = True
                    elif ready(o):
                        prog = not is_two   # pick_two progress is the placed-count increase below
                    else:
                        events.append({'step': s, 'type': 'placed_untransformed', 'object': o,
                                       'repaired': False})
                        wasted = True
                elif need and typ(dst) == appliance and o not in before.get(need, set()):
                    events.append({'step': s, 'type': 'appliance_as_container', 'object': o, 'repaired': False})
                    wasted = True
                else:
                    # resolved after the loop: put_down if taken again, else a wrong placement
                    events.append({'step': s, 'type': 'set_down', 'object': o, 'receptacle': dst,
                                   'ready': ready(o), 'repaired': None})
                    wasted = True
            else:
                # a non-target handled the way the target should be: into the goal receptacle,
                # into the task's appliance, or (heat tasks) onto the stove
                if recep and typ(dst) == recep and not is_look:
                    events.append({'step': s, 'type': 'non_target_on_goal', 'object': o, 'receptacle': dst,
                                   'source': pick_src.get(o, (None, None))[0],
                                   'picked_at': pick_src.get(o, (None, None))[1], 'repaired': None})
                elif need and (typ(dst) == appliance or (verb == 'heat' and typ(dst) == 'stoveburner')):
                    events.append({'step': s, 'type': 'wrong_object', 'object': o,
                                   'how': f'put in {typ(dst)}', 'picked_at': pick_src.get(o, (None, None))[1],
                                   'repaired': False})
                wasted = True
        elif RE_XFORM.match(obs):
            v, o, app = RE_XFORM.match(obs).groups()
            stt = STATE_OF[v]
            st.states[stt].add(o)
            if stt in UNDO:
                st.states[UNDO[stt]].discard(o)
            if typ(o) == obj and need:
                if v == verb:
                    if o not in before.get(need, set()):
                        if progress['transformed'] is None:
                            progress['transformed'] = s
                        prog = True
                        for e in events:
                            if e['object'] == o and e['type'] in ('placed_untransformed', 'appliance_as_container',
                                                                 'undid_transform', 'wrong_transform'):
                                e['repaired'] = True
                    else:
                        wasted = True
                elif need in UNDO and UNDO[need] == stt and o in before.get(need, set()):
                    events.append({'step': s, 'type': 'undid_transform', 'object': o, 'repaired': False})
                    wasted = True
                else:
                    events.append({'step': s, 'type': 'wrong_transform', 'object': o, 'how': f'{v}ed',
                                   'repaired': False})
                    wasted = True
            elif typ(o) != obj and need:
                # any transform of a non-target on a transform task is handling it as the target
                # (e.g. "cooling" a pot by cleaning it)
                events.append({'step': s, 'type': 'wrong_object', 'object': o, 'how': f'{v}ed',
                               'picked_at': pick_src.get(o, (None, None))[1], 'repaired': False})
                wasted = True
            else:
                wasted = True
        elif RE_TOGGLE.match(obs):
            on, lamp = RE_TOGGLE.match(obs).groups()
            was_on = lamp in st.lamps_on
            (st.lamps_on.add if on == 'on' else st.lamps_on.discard)(lamp)
            if typ(lamp) == 'desklamp' and st.loc:
                st.lamp_at.setdefault(lamp, st.loc)
            if was_on and on == 'on':
                wasted = True               # re-lighting a lamp that is already on
            elif is_look and on == 'on' and st.held and typ(st.held) == obj:
                if progress['transformed'] is None:
                    progress['transformed'] = s
                prog = True
            elif is_look and on == 'on' and st.held:
                events.append({'step': s, 'type': 'wrong_object', 'object': st.held, 'how': 'used with lamp',
                               'repaired': False})
                wasted = True
            elif is_look and on == 'on':
                pass                        # lamp first, object second also wins
            else:
                wasted = True
        else:
            # examine / look / inventory / help. Examining an unseen receptacle reveals its
            # contents and counts as search, not waste.
            info_actions.append(s)
            r, new = learn(obs)
            if r:
                st.see(r, known[r])
                if new:
                    examined.add(r)
            fm = RE_FACING.match(obs)
            if fm and st.loc:
                st.see(st.loc, items(fm.group(2)))
            wasted = (not new) or (r is not None and held_needs_elsewhere(r))

        if progress['seen'] is None and re.search(rf'\b{obj} \d+\b', obs):
            progress['seen'] = s
            prog = True

        if is_two:
            n = placed_count(st, obj, recep)
            if n > max_placed:
                prog = True
                max_placed = n
        if goal_met(st, tt, obj, recep) and goal_step is None:
            goal_step = s
            progress['placed'] = s
            if is_look and progress['transformed'] is None:
                progress['transformed'] = s
            prog = True
        if prog:
            last_progress = s
            wasted = False
        timeline[s] = wasted
        progress_at[s] = prog

    for s in burned:
        timeline[s] = True

    # an exempted ("purposeful") arrival that leads to no progress before the next arrival
    # was not purposeful after all
    # (and a return to a shut, never-opened receptacle has to open it before leaving)
    order = sorted(progress_at)
    arr_steps = [x[0] for x in arrivals]
    last_exec = max(executed) if executed else 0
    for i, (s, r, kind) in enumerate(arrivals):
        if kind is None or s == last_exec:  # an arrival on the final step had no time to act
            continue
        end = arr_steps[i + 1] if i + 1 < len(arr_steps) else max_steps + 1
        window = [x for x in order if s <= x < end]
        ok = any(progress_at.get(x) for x in window) if kind == 'purpose' else \
            any(opened_at.get(x) == r for x in window)
        if not ok:
            timeline[s] = True
            revisits.append(s)
    revisits.sort()

    # resolve set-downs and non-targets left on the goal receptacle
    def taken_again(o, step):
        return any(t['step'] > step and t['action'].startswith(f'take {o} from ') for t in steps)

    def next_is_target_take(step):
        """The set-down freed the hand: an attempt to take a target sits right next to it --
        refused just before (hand full) or tried just after (executed or not)."""
        nxt = [t for t in steps if t['step'] > step][:1]
        tries = [x['action'] for x in invalid if x['step'] in (step, step + 1) and x['action']]
        if nxt:
            tries += [x['action'] for x in nxt[0]['invalid_before'] if x['action']] + [nxt[0]['action']]
        return any(re.match(rf'take {obj} \d+', a) for a in tries)

    later_target_take = lambda step: any(t['step'] > step and RE_TAKE.match(t['obs'])
                                         and typ(RE_TAKE.match(t['obs']).group(1)) == obj for t in steps)
    for e in events:
        if e['type'] == 'set_down':
            if taken_again(e['object'], e['step']) or not e['ready'] or next_is_target_take(e['step']):
                e['type'] = 'put_down'
            else:
                e['type'] = 'wrong_receptacle'
                e['repaired'] = goal_step is not None
            e.pop('ready', None)
        elif e['type'] == 'non_target_on_goal':
            # Putting a non-target into the goal receptacle is a substitution -- even if it is
            # taken back later (the agent then "tries again" with the same wrong object) --
            # unless it only went back where it came from, or was set down to free the hand.
            put_back = e['receptacle'] == e.get('source') and e.get('picked_at') is not None \
                and e['step'] - e['picked_at'] <= 2
            if put_back or next_is_target_take(e['step']) or later_target_take(e['step']):
                e['type'] = 'put_down'
            else:
                e['type'], e['how'] = 'wrong_object', 'placed'
                e['repaired'] = False
        e.pop('source', None)
        if goal_step is not None and e.get('repaired') is False:
            e['repaired'] = True            # the run won; nothing left blocking

    won = bool(rec['won'])
    failed_stage, failed_step = None, None
    if not won:
        blocking = [e for e in events if e['type'] in BLOCKING and e['repaired'] is not True]
        # Substitution that never shows as a placement: a non-target still in hand at the end
        # while the target was never taken, and either the target had been seen or the agent
        # kept trying to put the non-target into the goal receptacle.
        if progress['taken'] is None and st.held and typ(st.held) != obj:
            # anchored on the FIRST pick-up of that object: that is where the substitution began
            pick = min((e for e in events if e['type'] == 'picked_non_target' and e['object'] == st.held),
                       key=lambda e: e['step'], default=None)
            tried = any(x['action'] and st.held in x['action'] and recep
                        and re.search(rf'\b{recep} \d+\b', x['action']) for x in invalid)
            dup = any(e['type'] == 'wrong_object' and e['object'] == st.held for e in events)
            if pick and not dup and (progress['seen'] is not None or tried):
                blocking.append({'step': pick['step'], 'type': 'wrong_object', 'object': st.held,
                                 'how': 'held instead of target', 'repaired': False})
                events.append(blocking[-1])
        if blocking:
            e = min(blocking, key=lambda e: e['step'])
            # a substitution's root cause is where the wrong object was picked up
            failed_stage, failed_step = BLOCKING[e['type']], e.get('picked_at') or e['step']
        elif progress['seen'] is None:
            failed_stage = 'never_found'
        elif progress['taken'] is None:
            failed_stage = 'found_not_taken'
        elif is_look:
            failed_stage = 'look_lamp_not_on'
        elif need and progress['transformed'] is None:
            failed_stage = 'not_transformed'
        elif is_two and max_placed == 1:
            failed_stage = 'pick_two_one_placed'
        elif need:
            failed_stage = 'transformed_not_placed'
        else:
            failed_stage = 'taken_not_placed'
    events.sort(key=lambda e: e['step'])
    if is_two:
        progress['placed_count'] = max_placed

    # maximal runs of >= 3 consecutive wasted steps (burned steps included)
    runs, cur = [], []
    for s in sorted(timeline):
        if timeline[s] and cur and s == cur[-1] + 1:
            cur.append(s)
            continue
        if len(cur) >= 3:
            runs.append([cur[0], cur[-1]])
        cur = [s] if timeline[s] else []
    if len(cur) >= 3:
        runs.append([cur[0], cur[-1]])

    room = sorted(set(re.findall(rf'\b({INST})\b', rec['init_obs'].split('Your task is to:')[0])))
    searched = visited | examined
    never = sorted({typ(r) for r in room} - {typ(v) for v in searched})
    return {
        'task_id': rec['task_id'],
        'group_id': rec['group_id'],
        'task_type': tt,
        'goal': rec['task'],
        'won': int(won),
        'won_step': rec['won_step'],
        'tracker_goal_step': goal_step,
        # ALFWorld quirk: an object listed in two receptacles at the start (a mug in the
        # coffeemachine on the countertop) keeps the second inReceptacle fact after it is
        # picked up from the first, so transforming it in hand already satisfies the goal.
        'env_quirk_early_win': bool(won) and goal_step != rec['won_step'],
        'env_actions': rec['env_actions'],
        'steps': rec['steps'],
        'max_steps': max_steps,
        'expert_plan_len': len(walkthrough),
        'target': {'object': obj, 'transform': 'look' if is_look else verb,
                   'appliance': appliance, 'receptacle': recep, 'count': 2 if is_two else 1},
        # all receptacle types holding a target-type object at the start (PDDL :init); the
        # reference-solution pick-up spot is kept separately and only the critic sees instances
        'target_location': initial_locations(pddl_problem, obj) if pddl_problem else
                           sorted({typ(x) for x in loc_inst}),
        'target_location_reference': sorted({typ(x) for x in loc_inst}),
        'target_location_instances': loc_inst,
        'progress': progress,
        'events': events,
        'transform_available_unused': avail_unused,
        'place_available_unused': place_unused,
        'search': {'receptacles': len(room), 'visited': sorted(visited), 'opened': sorted(opened),
                   'examined_from_afar': sorted(examined - visited),
                   'never_visited_types': never, 'steps_by_type': dict(steps_by_type)},
        'revisits': revisits,
        'info_actions': info_actions,
        'invalid_replies': invalid,
        'burned_steps': burned,
        'wasted_runs': runs,
        'last_progress_step': last_progress,
        'ran_out_after_progress': (not won) and last_progress is not None and last_progress >= max_steps - 6,
        'failed_stage': failed_stage,
        'failed_stage_step': failed_step,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='ministral8b | qwen3-4b')
    ap.add_argument('--raw', default=None, help='default rollouts/<model>/train_full_s20.jsonl')
    ap.add_argument('--out', default=None, help='default memory_banks/<model>/_work/facts.jsonl')
    a = ap.parse_args()
    raw = Path(a.raw or paths.rollouts(a.model))
    out = Path(a.out or paths.work(a.model) / 'facts.jsonl')
    out.parent.mkdir(parents=True, exist_ok=True)

    n, agree, quirk = 0, 0, 0
    with raw.open() as f, out.open('w') as g:
        for line in f:
            rec = json.loads(line)
            game = json.load(open(paths.gamefile(rec['task_id'])))
            fx = facts_for(rec, game['walkthrough'], game['pddl_problem'])
            g.write(json.dumps(fx, ensure_ascii=False) + '\n')
            n += 1
            agree += fx['tracker_goal_step'] == fx['won_step']
            quirk += fx['env_quirk_early_win']
    print(f'{n} trajectories -> {out}')
    print(f'tracker goal step == env won_step: {agree}/{n}  (the rest are env_quirk_early_win: {quirk})')


if __name__ == '__main__':
    main()
