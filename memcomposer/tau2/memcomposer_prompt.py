"""MemComposer prompt for tau2-bench: pick and rank the useful cases out of the retrieved top-k.

    from memcomposer_prompt import load_base, hits_to_cases, build_prompt, parse_selection

    base   = load_base('retail')                           # <memory bank>/retail, row-aligned
    cases  = hits_to_cases(result['hits'], base)[:10]      # one retrieval result's hits
    prompt = build_prompt(result['query'], cases, domain='retail', template='ministral')
    picked = [cases[i - 1] for i in parse_selection(reply, valid=range(1, len(cases) + 1))]

Cases come from `<memory bank>/<domain>/task*/digest.txt` unchanged - this module renders none
of the case text itself. Each is labelled by its place in retrieval order, 1..k (`index_by='rank'`,
the default), so the selector's output is a list of ranks. `index_by='row'` labels each case with
its memory-bank row instead, naming memories directly.

Two templates share one body:
  'qwen'       the prompt the Qwen3-4B selector was trained and evaluated with; its output block
               comes last, after the query
  'ministral'  the Ministral-3-8B prompt: the output block before the cases, with the answer
               format spelled out
The training data ships with the prompts already rendered; this module documents and rebuilds them.
The memory bank root is MEMSEL_MEMORY_BASE (default: the Qwen3-4B bank under MEMCOMPOSER_DATA).
"""
import os
import json
import re
from pathlib import Path

import numpy as np

MEMORY_BASE = Path(os.environ.get('MEMSEL_MEMORY_BASE',
                    Path(os.environ.get('MEMCOMPOSER_DATA', Path(__file__).resolve().parents[2] / 'data'))
                    / 'tau2' / 'memory_banks' / 'qwen'))

# The query sits AFTER the cases (the model reads the pool, then what it is selecting for), and the
# reasoning is capped at two or three sentences, because walking the pool case by case can exhaust
# the response budget before the answer.
TEMPLATE = """\
You are the memory selector for the {domain} agent.
You are given the customer's opening request and a set of retrieved cases from past \
conversations in the same domain, numbered [1], [2], and so on.
Your task is to select and rank the cases that are useful for helping the agent act correctly \
in the new conversation.
The concrete entities, wording, or itinerary in a past case may differ from the current \
request. Focus on whether the underlying lesson transfers.
Each case includes its subsequent grading or outcome. Use this feedback to distinguish useful \
successful behavior from mistakes that should not be repeated. A negatively graded case can \
still be useful if it clearly demonstrates a relevant failure mode.
Select only cases that provide meaningful additional guidance for the current request. \
Stop once additional cases would not materially improve the agent's decision.
Return the selected case indices in descending order of usefulness.
If no retrieved case provides useful transferable guidance, return an empty list.

{output}

Retrieved cases:
{cases}

Query:
{query}"""

OUTPUT_QWEN = """\
First, inside <reasoning> </reasoning>, briefly reason about which cases you keep and why. \
Keep it to two or three sentences; do not walk through the cases one by one.
Then output only a JSON list of case numbers inside <answer> </answer>, e.g. \
<answer>[...]</answer>. Output nothing after that."""

# The answer FORMAT spelled out. With the Qwen example `<answer>[...]</answer>`, Ministral-3-8B
# sometimes answers `["1", "2", "4"]` or `["[1]", "[2]", "[3]"]`, which read_answer in
# memcomposer/reward.py treats as a format failure.
#
# The brackets are described ONLY in the positive ("an opening [ ... a closing ]"); the single
# negative is about quotation marks. A wording that said "do not wrap each number in its own
# square brackets" was read as "no square brackets" and the outer brackets disappeared. Probe
# any change to this block on the full set of real prompts (generation only, greedy, as in eval)
# before adopting it; short synthetic prompts do not catch such failures.
OUTPUT_MINISTRAL = """\
First, inside <reasoning> </reasoning>, briefly reason about which cases you keep and why. \
Keep it to two or three sentences; do not walk through the cases one by one.
Then output only a JSON list of case numbers inside <answer> </answer>, e.g. \
<answer>[3, 1, 5]</answer>.
The answer block must match that shape exactly: an opening [, then the case numbers as plain \
digits separated by commas, then a closing ]. No quotation marks anywhere. Anything else \
scores zero.
Output nothing after that."""

# The Qwen prompt moves the output block to the very end, right after the query: placed before
# ~25k characters of cases, the format instruction was far from the point of generation and
# replies that never emitted <answer> paraphrased the header instead.
TEMPLATE_QWEN = TEMPLATE.replace('\n\n{output}\n\nRetrieved cases:', '\n\nRetrieved cases:') + '\n\n{output}'

# Both ask for a capped <reasoning> and an <answer> block; read_answer() in memcomposer/reward.py
# gates on that block, so the template and the reward must stay paired.
TEMPLATES = {'qwen': (TEMPLATE_QWEN, OUTPUT_QWEN), 'ministral': (TEMPLATE, OUTPUT_MINISTRAL)}


def load_base(domain='airline', body='digest', root=MEMORY_BASE):
    """The whole memory base, one dict per row of `index.npz`, ordered by that file's rows.

    Each entry: `index` (the row, == `memory.json` `retrieval_key.index_row`), `folder`,
    `sim_id`, `task_id`, `trial`, `label_kind`, and `text` (the `body`.txt rendering verbatim).
    `body` is 'digest' (selector-sized) or 'full' (elides nothing).
    """
    base_dir = Path(root) / domain
    z = np.load(base_dir / 'index.npz', allow_pickle=True)
    out = []
    for row, (folder, sim_id) in enumerate(zip(z['folders'], z['sim_ids'])):
        folder = str(folder)
        meta = json.loads((base_dir / folder / 'memory.json').read_text())
        assert meta['retrieval_key']['index_row'] == row, f'{folder}: index_row != npz row'
        out.append({'index': row, 'folder': folder, 'sim_id': str(sim_id),
                    'task_id': meta['task_id'], 'trial': meta['trial'],
                    'label_kind': meta['label_kind'],
                    'text': (base_dir / folder / f'{body}.txt').read_text().strip()})
    return out


def hits_to_cases(hits, base):
    """Retrieval hits -> base entries, in retrieval order. Joins on `sim_id`."""
    by_sim = {c['sim_id']: c for c in base}
    return [by_sim[h['sim_id']] for h in hits]


def format_cases(cases, index_by='rank'):
    """index_by='rank': label = position in retrieval order, 1..k. 'row': the memory-base row."""
    return '\n\n'.join(f"[{rank if index_by == 'rank' else c['index']}]: {c['text']}"
                       for rank, c in enumerate(cases, 1))


def build_prompt(query, cases, domain='airline', template='ministral', top_k=None, index_by='rank'):
    """The single user message the selector reads. `top_k` keeps the first k retrieved cases."""
    if top_k is not None:
        cases = cases[:top_k]
    body, output = TEMPLATES[template]
    return body.format(domain=domain, query=query.strip(), cases=format_cases(cases, index_by),
                       output=output)


def build_messages(query, cases, domain='airline', template='ministral', top_k=None, index_by='rank'):
    """Single user message; the template is self-contained, no system prompt."""
    return [{'role': 'user', 'content': build_prompt(query, cases, domain, template, top_k, index_by)}]


_PLAIN = re.compile(r'\[\s*(?:\d+\s*(?:,\s*\d+\s*)*)?\]')
_OBJ = re.compile(r'\{[^{}]*?"index"\s*:\s*\[?\s*(\d+)\s*\]?[^{}]*?\}', re.S)
_REASONING = re.compile(r'<reasoning>(.*?)</reasoning>', re.S)
_ANSWER = re.compile(r'<answer>(.*?)</answer>', re.S)


def extract_reasoning(text):
    """The <reasoning> block, or '' when the model produced none."""
    m = _REASONING.search(text or '')
    return m.group(1).strip() if m else ''


def extract_answer(text):
    """Contents of the last CLOSED <answer> </answer> block, or None if there is no closed one.

    Closed is the point. A reply that derails into a repetition loop is cut off by the response
    limit mid-ramble and never emits </answer>, so it returns None and the caller can treat it
    as the format failure it is. Searching for a bracketed list anywhere cannot tell the two
    apart: a loop printing "Case [142] is a good example." supplies one that looks valid.
    """
    blocks = _ANSWER.findall(text or '')
    return blocks[-1].strip() if blocks else None


def parse_selection(text, valid=None):
    """Case numbers from a reply, deduplicated, in the order given.

    `valid` is the set of numbers that were actually shown; anything outside it is dropped.
    Everything up to and including a </reasoning> tag is ignored, so numbers the model mentions
    while thinking are not mistaken for its answer. Objects of the form {"index": N, ...} are
    accepted as well as a bare [N, ...], since a model asked for one sometimes emits the other.
    Returns [] for an empty list, 'None', or unparseable output.
    """
    text = text or ''
    if '</reasoning>' in text:
        text = text.rsplit('</reasoning>', 1)[1]
    out, seen = [], set()
    for m in _OBJ.finditer(text):
        i = int(m.group(1))
        if i not in seen and (valid is None or i in valid):
            seen.add(i)
            out.append(i)
    if out:
        return out
    found = _PLAIN.findall(text)
    if not found:
        return []
    for i in json.loads(found[-1]):
        if i not in seen and (valid is None or i in valid):
            seen.add(i)
            out.append(i)
    return out
