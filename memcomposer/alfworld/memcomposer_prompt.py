"""MemComposer prompt for ALFWorld.

Adapted from the tau2-bench Ministral prompt (memcomposer/tau2/memcomposer_prompt.py, TEMPLATE with
OUTPUT_MINISTRAL). Differences, and nothing else:
- body sentences that spoke of customers, opening requests and conversations now speak of a
  household task, the room it starts in, and past attempts in other rooms;
- the "Select only cases ... Stop once additional cases would not materially improve the agent's
  decision." pair is replaced by the single sentence "Select cases that provide meaningful
  additional guidance for the current task.": on ALFWorld more memories keep helping up to
  k=7, and the stop-early wording pushes the selector towards small k, which tends to collapse;
- the query carries the opening room inventory as well as the goal;
- each case ends with a short `ID:` line (opaque hash of the memory id) for offline analysis.

The OUTPUT block must stay byte-identical to the tau2-bench OUTPUT_MINISTRAL (asserted against
that source in check_output_block()): the selector model misreads rewordings of it.
"""
import hashlib
import json
import os
import re
from pathlib import Path

DATA = Path(os.environ.get('MEMCOMPOSER_DATA', Path(__file__).resolve().parents[2] / 'data')) / 'alfworld'
TAU2_PROMPT = Path(__file__).resolve().parents[1] / 'tau2' / 'memcomposer_prompt.py'

TEMPLATE = """\
You are the memory selector for a household-task agent in ALFWorld.
You are given the agent's task instruction, the room it starts in, and a set of retrieved cases \
from past attempts at similar tasks in other rooms, numbered [1], [2], and so on.
Your task is to select and rank the cases that are useful for helping the agent complete the new \
task.
The room layout, object numbers and locations in a past case belong to that other room and will \
differ from the current one. Focus on whether the underlying lesson transfers.
Each case includes its outcome and an after-the-fact review. Use this feedback to distinguish \
useful successful behavior from mistakes that should not be repeated. A failed case can still be \
useful if it clearly demonstrates a relevant failure mode.
Select cases that provide meaningful additional guidance for the current task.
Return the selected case indices in descending order of usefulness.
If no retrieved case provides useful transferable guidance, return an empty list.

{output}

Retrieved cases:
{cases}

Query:
{query}"""

# The tau2-bench Ministral output block, verbatim.
OUTPUT = """\
First, inside <reasoning> </reasoning>, briefly reason about which cases you keep and why. \
Keep it to two or three sentences; do not walk through the cases one by one.
Then output only a JSON list of case numbers inside <answer> </answer>, e.g. \
<answer>[3, 1, 5]</answer>.
The answer block must match that shape exactly: an opening [, then the case numbers as plain \
digits separated by commas, then a closing ]. No quotation marks anywhere. Anything else \
scores zero.
Output nothing after that."""

_ROOM = re.compile(r'(You are in the middle of a room\..*?)\n\s*\nYour task is to:', re.S)


def check_output_block():
    """Fail loudly if this copy of the output block ever drifts from the tau2-bench one."""
    src = TAU2_PROMPT.read_text()
    m = re.search(r'OUTPUT_MINISTRAL = """\\\n(.*?)"""', src, re.S)
    assert m, 'OUTPUT_MINISTRAL not found in tau2 memcomposer_prompt.py'
    tau2 = m.group(1).replace('\\\n', '')
    assert tau2 == OUTPUT, 'ALFWorld output block differs from the tau2-bench Ministral one'


def short_id(memory_id):
    return hashlib.sha1(memory_id.encode()).hexdigest()[:8]


def room_of(init_obs):
    """The opening room inventory, as the agent first sees it."""
    m = _ROOM.search(init_obs)
    assert m, f'no room sentence in: {init_obs[:120]!r}'
    return ' '.join(m.group(1).split())


def query_text(goal, init_obs):
    return f'Task: {goal}\nRoom: {room_of(init_obs)}'


def case_text(i, memory_dir):
    digest = (Path(memory_dir) / 'digest.txt').read_text().rstrip()
    assert not any(l.startswith('[') for l in digest.split('\n')), f'{memory_dir}: a digest line starts with ['
    return f'[{i}]: {digest}\nID: {short_id(Path(memory_dir).name)}'


def build_prompt(goal, init_obs, memory_dirs):
    """The single user message the selector reads."""
    cases = '\n\n'.join(case_text(i, d) for i, d in enumerate(memory_dirs, 1))
    return TEMPLATE.format(output=OUTPUT, cases=cases, query=query_text(goal, init_obs))


def init_obs_index(path=DATA / 'init_obs.jsonl'):
    return {json.loads(l)['task_id']: json.loads(l)['init_obs'] for l in open(path)}
