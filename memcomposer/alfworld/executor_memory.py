"""The memory block the executor receives -- one function for every arm.

The top-k baselines and the selector's reward (via alfworld_runner.py) both call render_block(),
so a selection evaluated during training is shown to the executor byte for byte the way the
baselines showed it. alfworld_prompt.MEMORY_BLOCK puts the returned string under
"## Retrieved relevant experience".
"""
from pathlib import Path

HEADER = (
    "Below are past attempts at similar household tasks, made in other rooms. Some succeeded and "
    "some failed. Each case starts with its OUTCOME, FACTS computed from its log, and a SUMMARY; "
    "notes marked ** sit under the step they refer to (ROOT CAUSE / CONTRIBUTING for failures, "
    "KEY / WASTED for successes, with SHOULD HAVE). Object numbers and where things were belong to "
    "that other room, not to yours: read your own room from your own observations. Use a case only "
    "where it is relevant to your task.")


def render_block(memory_ids, base):
    """HEADER + '-- case i of k --' + full.txt per memory, in the given order. '' for none."""
    if not memory_ids:
        return ''
    base = Path(base)
    parts = [HEADER]
    for i, mid in enumerate(memory_ids, 1):
        parts.append(f'-- case {i} of {len(memory_ids)} --\n' + (base / mid / 'full.txt').read_text().rstrip())
    return '\n\n'.join(parts)
