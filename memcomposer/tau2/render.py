"""Render retrieved memories into the <experience> block for the agent's system prompt.

The block is a third sibling of <instructions> and <policy>, never nested inside <policy>: many
memories describe behaviour that must not be copied, and inside <policy> they would inherit the
policy's authority.

Each case is the memory's `full.txt` verbatim, minus its trailing `ID:` line. Env vars (read at
import): MEMSEL_DOMAIN, MEMSEL_MEMORY_BASE.
"""
import json
import os
from pathlib import Path

# The domain is process-wide: one service instance serves one domain, which is how
# tau2_service is launched. Set MEMSEL_DOMAIN before importing this module.
DOMAIN = os.environ.get('MEMSEL_DOMAIN', 'airline')
# MEMSEL_MEMORY_BASE: the bank root (one folder per domain); default the Qwen3-4B bank. Read at
# import, like DOMAIN: the FOLDER map below is built once per process. AIRLINE is the current
# domain's folder, whatever the domain.
BASE_ROOT = Path(os.environ.get('MEMSEL_MEMORY_BASE',
                    Path(os.environ.get('MEMCOMPOSER_DATA', Path(__file__).resolve().parents[2] / 'data'))
                    / 'tau2' / 'memory_banks' / 'qwen'))
AIRLINE = BASE_ROOT / DOMAIN

HEADER = """\
Below are cases from past conversations in this domain, which may be helpful here. Use a case \
when you feel it is relevant. Before you act, first briefly say which of these cases applies to \
your situation and which does not.

These are whole past conversations, not rules, and some of them are mistakes. Each case opens \
with how it was judged afterwards - KIND, the condition it failed, a summary, and the database \
difference the grader measured - and carries ROOT CAUSE / CONTRIBUTING / ISSUE notes inline at \
the turn they refer to. Specific values inside a case - cabin, insurance, membership, dates, \
reservation ids, balances - belong to that past customer, not to yours; read your own from your \
own tool results. Where a case and <policy> disagree, the policy is right."""


def load_full(sim_id, folder_by_sim=None):
    folder = (folder_by_sim or FOLDER)[sim_id]
    text = (AIRLINE / folder / 'full.txt').read_text().rstrip()
    return '\n'.join(l for l in text.split('\n') if not l.startswith('ID: '))


FOLDER = {json.loads((f).read_text())['sim_id']: f.parent.name
          for f in AIRLINE.glob('task*/memory.json')}


def render(hits):
    """hits: retrieval hits carrying 'sim_id' and 'label_kind'."""
    if not hits:
        return ''
    # Cases are kept in the selector's order: it ranks them by usefulness, and regrouping them
    # (e.g. by outcome) would discard that ranking. Each case opens with its own KIND line, so
    # whether it went right or wrong is still marked per case.
    parts = [HEADER, '']
    for i, h in enumerate(hits, 1):
        parts.append(f'\n-- case {i} of {len(hits)} --\n{load_full(h["sim_id"])}')
    return '<experience>\n' + '\n'.join(parts).rstrip() + '\n</experience>'

# The per-source-task memory cap is applied at retrieval (memory_bank/tau2/retrieve.py --cap),
# not here.
