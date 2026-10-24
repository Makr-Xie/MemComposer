"""Prompts for the ALFWorld self-review critic.

The critic is the executor model itself, reviewing its own trajectory. As in the tau2 counterpart
(memory_bank/tau2/review/self_critique.py), the verdict is ground truth and the critic only
attributes; it is handed an answer key and mechanically computed facts so a small model does not
have to guess.

Two prompts, one per outcome. Both share RULES and the USER layout; the *_text helpers and build()
turn a raw record and its facts (build_facts.py) into the sections.
"""
import re

RULES = """
ALFWorld rules (facts about this environment, not advice):
- An object is heated ONLY by the action `heat <object> with microwave <n>`, done while holding
  it at the microwave. Putting it inside the microwave, opening or closing the door, or using a
  stoveburner, oven or toaster does nothing.
- An object is cooled ONLY by `cool <object> with fridge <n>`. Putting it inside the fridge does
  nothing.
- An object is cleaned ONLY by `clean <object> with sinkbasin <n>`.
- Look tasks ("look at X under the desklamp", "examine X with the desklamp"): hold X and
  `use desklamp <n>` while standing where the desklamp is. Turning the lamp on first and picking
  X up afterwards also works. Examining X does nothing.
- An object is placed with `move <object> to <receptacle>`. The goal receptacle may itself be a
  fridge, microwave, coffeemachine or stoveburner; then moving the processed object into it IS
  the placement, not a mistake.
- The agent holds one object at a time; it must put one down to take another.
- Closed receptacles (drawer, cabinet, fridge, microwave, safe, ...) show nothing until opened.
  Surfaces (table, countertop, desk, bed, sofa, shelf, ...) show their contents on arrival.
- Two-object tasks ("put two X in Y") need both objects in the SAME receptacle instance.
- Only the exact object type counts: a pan is not a pot, a mug is not a cup, a knife is not a
  butterknife, a cloth is not a handtowel, a spraybottle is not a soapbottle.
- `examine`, `inventory`, `look` and `help` never change the room.
- At every step the agent may also `go to` any receptacle in the room; the per-step lists below
  leave those out.
""".strip()

STAGES = ('never_found', 'found_not_taken', 'wrong_object', 'not_transformed', 'undid_transform',
          'appliance_as_container', 'transformed_not_placed', 'taken_not_placed', 'wrong_receptacle',
          'pick_two_one_placed', 'look_lamp_not_on')

STAGE_HELP = """
failed_stage, pick exactly one:
  never_found            the target object was never seen
  found_not_taken        seen but never picked up
  wrong_object           a different object was handled as if it were the target
  not_transformed        the target was never heated/cooled/cleaned (or placed before that)
  undid_transform        processed, then undone (heated, then cooled)
  appliance_as_container put inside the microwave/fridge/sink instead of using heat/cool/clean
  transformed_not_placed processed but not put in the goal receptacle
  taken_not_placed       (tasks with no processing) taken but not put in the goal receptacle
  wrong_receptacle       the ready target was put in a different receptacle and left there
  pick_two_one_placed    only one of the two objects ended in the goal receptacle
  look_lamp_not_on       held the target but never had the desklamp on where it stood
""".strip()

SYSTEM_FAIL = f"""
You are reviewing your own earlier attempt at an ALFWorld household task, to write the lesson a
future attempt at a similar task will read.

You are given:
- <Rules>: how this environment works.
- <Task>: the goal and what it requires.
- <Answer Key>: a reference solution written by someone who already knew where every object was.
  The agent did not know; it had to search.
- <Verdict>: the outcome of this attempt.
- <Facts>: computed mechanically from the log. They are correct; build on them.
- <Trajectory>: every step as [STEP n] with the agent's own reasoning (cut short), its action,
  the observation, and the other actions that were available at that step.

The verdict is ground truth: this attempt FAILED. Do not re-litigate it.
Your job is attribution: find the step in THIS trajectory that caused THAT failure.

How to work:
1. State which goal condition was not met (failed_condition) and pick failed_stage. <Facts> gives
   the mechanical stage; keep it unless the events in <Facts> clearly support another stage.
2. Find the FIRST step where the agent's own choice made the failure inevitable or threw away the
   decisive opportunity: took the wrong kind of object, put the object into the microwave instead
   of heating it, placed it before cleaning it, kept going back to places it had already searched,
   walked away from the appliance while the right action was available. A step that merely
   revealed information is never the root cause.
3. Say what the agent should have done at that step instead.
4. List up to 3 further mistakes that contributed, each at its own step.
5. If the agent was making steady progress and only ran out of steps, set "blame" to
   "step_budget" and say what slowed it down. Do not manufacture an error.

Rules for your answer:
- Every step you cite must be a [STEP n] in <Trajectory>. Copy that step's action exactly into
  "action".
- "should_have" is the very next action the agent should have typed AT THAT STEP, from where it
  stood: ONE action copied exactly from the list under that step, or `go to <receptacle>` for a
  receptacle in the room. If the right move needs the agent somewhere else first (the object is
  on the bed, the lamp is on the sidetable, the sinkbasin is across the room), should_have is the
  `go to` that heads there, not the action it would do on arrival. It must differ from what the
  agent did.
- Do not use knowledge only the answer key has. Never write "should have gone to cabinet 4 because
  the mug was there", and never say where an object "actually was" or "turned out to be": the
  agent could not know that at the time. Justify the better action by what the agent could see or
  reason: receptacles it had not searched yet, where such objects usually are, the rules above.
- Use the agent's own reasoning when it shows the mistaken belief ("assumed the fork was already
  clean", "thought the stoveburner would heat it").
- Quote observations exactly, inside double quotes, or not at all. Never invent an observation.
- "summary": two sentences: what went wrong, then the fix as a rule a future attempt can follow.
  No instance numbers in the summary (write "the mug", not "mug 1").

{STAGE_HELP}

Output ONLY this JSON object:
{{
  "failed_condition": "<which goal condition was not met, in plain words>",
  "failed_stage": "<one of the stages above>",
  "blame": "agent" | "step_budget",
  "root_cause": {{
    "step": <int>,
    "action": "<the agent's action at that step, copied exactly>",
    "what_happened": "<what the agent did and believed at this step>",
    "why_it_caused_the_failure": "<the causal link to the failed condition>",
    "should_have": "<one available action at that step, copied exactly>"
  }},
  "contributing_errors": [
    {{"step": <int>, "action": "<copied exactly>", "what_happened": "<...>", "should_have": "<one available action>"}}
  ],
  "summary": "<two sentences, no instance numbers>"
}}
""".strip()

SYSTEM_SUCCESS = """
You are reviewing your own earlier attempt at an ALFWorld household task, to write the lesson a
future attempt at a similar task will read.

You are given:
- <Rules>: how this environment works.
- <Task>: the goal and what it requires.
- <Answer Key>: a reference solution written by someone who already knew where every object was.
  The agent did not know; it had to search.
- <Verdict>: the outcome of this attempt.
- <Facts>: computed mechanically from the log. They are correct; build on them.
- <Trajectory>: every step as [STEP n] with the agent's own reasoning (cut short), its action,
  the observation, and the other actions that were available at that step.

The verdict is ground truth: this attempt SUCCEEDED. Do not re-litigate it.
Your job: explain why it worked, so a future attempt can repeat it, and point out wasted effort.

How to work:
1. key_steps (1 to 4): the steps that made it succeed: where the target was found or taken, the
   processing step (heat/cool/clean/lamp) if the task has one, and the final placement. The
   winning step must be included, and for heat/cool/clean tasks the processing step too. For each,
   say why it mattered.
2. reusable_strategy: one sentence a future attempt in a DIFFERENT room could follow. Object and
   receptacle types only: no instance numbers, nothing that is true only of this room.
3. wasted_steps: only stretches of 3 or more consecutive steps that did not help (<Facts> lists
   candidate wasted runs). For each, what happened, and what the agent should have done at the
   FIRST step of the stretch. Use an empty list if there is no such stretch.
4. summary: two sentences, no instance numbers.

Rules for your answer:
- Every step you cite must be a [STEP n] in <Trajectory>. Copy that step's action exactly into
  "action".
- "should_have" is the very next action the agent should have typed AT THAT STEP, from where it
  stood: ONE action copied exactly from the list under that step, or `go to <receptacle>` for a
  receptacle in the room. If the right move needs the agent somewhere else first, should_have is
  the `go to` that heads there. It must differ from what the agent did.
- "steps" in wasted_steps is written "a-b" with a and b step numbers, b - a >= 2.
- Do not use knowledge only the answer key has, and never say where an object "actually was":
  justify by what the agent could see or reason.
- Quote observations exactly, inside double quotes, or not at all. Never invent an observation.

Output ONLY this JSON object:
{
  "key_steps": [
    {"step": <int>, "action": "<copied exactly>", "why_it_mattered": "<...>"}
  ],
  "reusable_strategy": "<one sentence, types only>",
  "wasted_steps": [
    {"steps": "<a-b>", "what_happened": "<...>", "should_have": "<one available action at step a>"}
  ],
  "summary": "<two sentences, no instance numbers>"
}
""".strip()

USER = """<Rules>
{rules}
</Rules>

<Task>
{task}
</Task>

<Answer Key>
Reference solution ({n_ref} steps; its author knew where every object was):
{walkthrough}
</Answer Key>

<Verdict>
{verdict}
</Verdict>

<Facts>
{facts}
</Facts>

<Trajectory>
{trajectory}
</Trajectory>"""

RETRY = """Your previous answer was rejected by an automatic check:
{reasons}

Remember: should_have is an action the agent could type at that very step (listed under the step,
or `go to <receptacle>`); "action" is copied exactly, instance numbers included.

Your previous answer was:
{previous}

Fix every listed problem and output the corrected JSON object only."""

THINK_CHARS = 300
_HIDE = re.compile(r'^(go to |help$|inventory$|look$)')


def task_text(f):
    t = f['target']
    obj, rec = t['object'], t['receptacle']
    if t['transform'] == 'look':
        need = f'hold a {obj} and turn on the desklamp where it stands'
    elif t['transform']:
        need = f"{t['transform']} a {obj} with the {t['appliance']}, then move it to a {rec}"
    elif t['count'] == 2:
        need = f'move two {obj}s into the same {rec}'
    else:
        need = f'move a {obj} to a {rec}'
    return f"Goal: {f['goal']}\nTask type: {f['task_type']}\nWhat it requires: {need}"


def verdict_text(f):
    if f['won']:
        return (f"SUCCEEDED: the goal was reached at step {f['won_step']} of {f['max_steps']} "
                f"(reference solution: {f['expert_plan_len']} steps).")
    return (f"FAILED: the goal was not reached within {f['max_steps']} steps "
            f"(reference solution: {f['expert_plan_len']} steps).")


def _steps(xs):
    return ', '.join(str(x) for x in xs) if xs else 'none'


def facts_text(f):
    t, p = f['target'], f['progress']
    lines = []
    if t['transform'] == 'look':
        lines.append(f"Target: hold a {t['object']} with the desklamp on")
    elif t['transform']:
        lines.append(f"Target: {t['object']} -> {t['transform']} with {t['appliance']} -> put in {t['receptacle']}")
    else:
        lines.append(f"Target: {t['count']} x {t['object']} -> put in {t['receptacle']}")
    lines.append(f"Where {t['object']}s were at the start: {', '.join(f['target_location']) or 'unknown'}"
                 f" (the reference solution took one from {', '.join(f['target_location_instances'])})")
    xform = {'heat': 'heated', 'cool': 'cooled', 'clean': 'cleaned', 'look': 'lamp on while held'}.get(t['transform'])
    prog = [f"first seen @{p['seen']}" if p['seen'] else 'target never seen',
            f"taken @{p['taken']}" if p['taken'] else 'never taken']
    if xform:
        prog.append(f"{xform} @{p['transformed']}" if p['transformed'] else f'{xform}: never')
    if 'placed_count' in p:
        prog.append(f"most placed together: {p['placed_count']} of 2")
    prog.append(f"goal reached @{p['placed']}" if p['placed'] else 'goal never reached')
    lines.append('Progress: ' + ', '.join(prog))
    ev = [e for e in f['events'] if e['type'] != 'picked_non_target' or not f['won']]
    if ev:
        lines.append('Events:')
        for e in ev:
            extra = ''
            if e.get('how'):
                extra += f" ({e['how']})"
            if e.get('receptacle'):
                extra += f" -> {e['receptacle']}"
            if e.get('picked_at'):
                extra += f", picked up @{e['picked_at']}"
            rep = {True: ' [later repaired]', False: ' [not repaired]'}.get(e.get('repaired'), '')
            lines.append(f"  @{e['step']} {e['type']}: {e['object']}{extra}{rep}")
    if f['transform_available_unused']:
        lines.append(f"The {t['transform'] if t['transform'] != 'look' else 'lamp'} action was available "
                     f"while holding the target but not used at steps: {_steps(f['transform_available_unused'])}")
    if f['place_available_unused']:
        lines.append(f"Placing the ready target was available but not used at steps: {_steps(f['place_available_unused'])}")
    lines.append(f"Revisits (returned, learned nothing new): {_steps(f['revisits'])}")
    lines.append(f"Info-only actions (examine/inventory/look): {_steps(f['info_actions'])}")
    if f['burned_steps']:
        lines.append(f"Steps lost entirely to unusable replies: {_steps(f['burned_steps'])}")
    lines.append('Wasted runs (3+ steps in a row without progress): '
                 + (', '.join(f'{a}-{b}' for a, b in f['wasted_runs']) or 'none'))
    s = f['search']
    lines.append(f"Search: visited {len(s['visited'])} of {s['receptacles']} receptacles; "
                 f"receptacle types never visited: {', '.join(s['never_visited_types']) or 'none'}")
    if not f['won']:
        st = f['failed_stage'] + (f" @{f['failed_stage_step']}" if f['failed_stage_step'] else '')
        lines.append(f"Mechanical failure stage: {st}")
        lines.append(f"Still making progress near the end (last progress @{f['last_progress_step']}): "
                     f"{'yes' if f['ran_out_after_progress'] else 'no'}")
    return '\n'.join(lines)


def trajectory_text(rec):
    out = []
    pending = []
    for t in rec['trajectory']:
        if t['status'] != 'ok':
            pending.append(t)
            continue
        think = ' '.join((t.get('think') or '').split())
        if len(think) > THINK_CHARS:
            think = think[:THINK_CHARS] + ' ...'
        avail = [a for a in (t.get('admissible') or []) if not _HIDE.match(a)]
        out.append(f"[STEP {t['step']}]")
        for x in pending:
            out.append(f"  (unusable reply first: `{x.get('action')}` -- {x['status']})")
        pending = []
        out.append(f"  think: {think or '(none)'}")
        out.append(f"  action: {t['action']}")
        out.append(f"  obs: {t['obs']}")
        out.append(f"  other actions available: {'; '.join(avail) if avail else '(only go to / look / inventory)'}")
    for x in pending:
        out.append(f"(step {x['step']} lost: unusable reply `{x.get('action')}` -- {x['status']})")
    return '\n'.join(out)


def build(rec, facts, walkthrough):
    """(system, user) for one trajectory."""
    user = USER.format(rules=RULES, task=task_text(facts), n_ref=len(walkthrough),
                       walkthrough='\n'.join(f'  {i}. {a}' for i, a in enumerate(walkthrough, 1)),
                       verdict=verdict_text(facts), facts=facts_text(facts),
                       trajectory=trajectory_text(rec))
    return (SYSTEM_SUCCESS if facts['won'] else SYSTEM_FAIL), user
