#!/usr/bin/env python
"""Prompt and reply parsing for the ALFWorld executor.

Shape follows verl-agent / SkillRL rather than ExpeL: the admissible action list is shown and the
model is told to pick from it, reasoning goes in <reasoning></reasoning> and the chosen action in
<action></action>.

The reasoning tag is <reasoning>, not SkillRL's <think>: `<think>` is a special token in Qwen3's
tokenizer (its thinking mode uses it), and a non-thinking Qwen3 variant answers a prompt
containing it with an immediate EOS and empty content. `<action>` is unaffected. Any executor
whose tokenizer reserves a tag used here fails this way silently, so probe a new model with one
prompt before a full run.

Showing the admissible list is also deliberate. The alternative (ExpeL's ReAct: no action list,
free generation) makes a small model spend much of its budget on syntactically invalid actions,
which turns the score into a measure of formatting rather than of memory quality.

`{retrieved_memories}` is the slot the selected memories render into. With no memories the whole
section is dropped, so the no-memory baseline is the same prompt minus that block -- nothing else
changes between arms.
"""
import re

SYSTEM = (
    "You are an expert agent operating in the ALFRED Embodied Environment. "
    "You complete household tasks by issuing one admissible action at a time."
)

MEMORY_BLOCK = """## Retrieved relevant experience

{retrieved_memories}

"""

TEMPLATE = """You are an expert agent operating in the ALFRED Embodied Environment.
Your task is to: {task}

{memory_block}## Current progress

You have taken {step_count} step(s) so far.{history_block}
You are now at step {current_step} and your current observation is:
{observation}

Your admissible actions of the current situation are: [{admissible}]

Now it's your turn to take an action.
First reason step-by-step about the current situation, enclosed in <reasoning> </reasoning> tags.
Then choose ONE action from the admissible list above and present it inside <action> </action> tags.
The text inside <action> </action> must match one admissible action exactly, with no extra words.
"""

HISTORY_LINE = "\n  step {i}: you took `{action}` and observed: {obs}"


def build_prompt(task, observation, admissible, history, memories=None, history_length=10,
                 hint=None):
    """history: list of (action, observation) in order. memories: list of rendered strings.

    `hint` carries a one-shot correction after an unusable reply. It is deliberately NOT written
    into `history`: history is the record of what actually happened in the environment, and a
    formatting slip did not happen in the environment. Keeping it out also leaves `history` in
    the same shape as a stored memory trajectory. The hint still has to exist, because at
    temperature 0 an unchanged prompt reproduces the same unusable reply forever.
    """
    recent = history[-history_length:]
    if recent:
        start = len(history) - len(recent) + 1
        lines = ''.join(HISTORY_LINE.format(i=start + n, action=a, obs=o)
                        for n, (a, o) in enumerate(recent))
        history_block = (f"\nBelow are the most recent {len(recent)} action(s) you took "
                         f"and what you observed:{lines}")
    else:
        history_block = ''
    memory_block = MEMORY_BLOCK.format(retrieved_memories='\n\n'.join(memories)) if memories else ''
    out = TEMPLATE.format(
        task=task or '(not stated)',
        memory_block=memory_block,
        step_count=len(history),
        current_step=len(history) + 1,
        observation=observation,
        admissible=', '.join(f'`{a}`' for a in admissible),
        history_block=history_block,
    )
    if hint:
        out += f'\nIMPORTANT: {hint}\nKeep your reasoning to at most three short sentences, ' \
               f'then give the <action> tag.\n'
    return out


_ACTION = re.compile(r'<action>\s*(.*?)\s*</action>', re.S | re.I)
_THINK = re.compile(r'<reasoning>\s*(.*?)\s*</reasoning>', re.S | re.I)


def parse_action(reply, admissible):
    """Return (action, status, reasoning). status is one of ok / no_tag / not_admissible.

    Only an exact match against the admissible list counts. A near miss is reported rather than
    silently repaired: how often the model produces an invalid action is a number we want to see,
    not one we want to hide inside a fuzzy matcher.
    """
    think = _THINK.search(reply)
    m = _ACTION.search(reply)
    if not m:
        return None, 'no_tag', (think.group(1).strip() if think else '')
    act = ' '.join(m.group(1).split()).strip().strip('`').strip()
    if act in admissible:
        return act, 'ok', (think.group(1).strip() if think else '')
    low = {a.lower(): a for a in admissible}
    if act.lower() in low:                      # case-only difference is not a real failure
        return low[act.lower()], 'ok', (think.group(1).strip() if think else '')
    return act, 'not_admissible', (think.group(1).strip() if think else '')
