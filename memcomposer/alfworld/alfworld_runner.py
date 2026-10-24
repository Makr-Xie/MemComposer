"""ALFWorld counterpart of memcomposer/tau2/tau2_runner.py -- the only module alfworld_service
imports, with the same six names: run_one, experience_from, _tasks, _meta, MAX_STEPS, DOMAIN.

One simulation = one ALFWorld game played by the executor with the selected memories, in the
same setup as the top-k baselines:
  executor   ALFW_AGENT_MODEL (default Bedrock mistral.ministral-3-8b-instruct, see llm.py),
             temperature ALFW_AGENT_TEMP (0.4), max_tokens ALFW_AGENT_MAX_TOKENS (2048),
             history 10, max_steps ALFW_MAX_STEPS (20)
  memories   executor_memory.render_block(folders, MEMSEL_MEMORY_BASE) -- the same function the
             baselines use, so the block is byte-identical. MEMSEL_MEMORY_BASE defaults to
             $MEMCOMPOSER_DATA/alfworld/memory_banks/ministral8b.

An infrastructure failure (LLM call failed after retries, environment exception) raises. The
service turns that into {"reward": null, "error": ...} and the reward function drops the trial
from the denominator. Returning 0 instead would teach GRPO that the selection lost.

The `seed` the reward function sends is not used: the executor samples at temperature 0.4 and
Bedrock takes no seed, so every trial is an independent draw. It still separates trials in the
dedup key, which is all it has to do.

Full per-simulation records (with trajectory) are appended to ALFW_SIM_LOG if set; the HTTP
response stays small.
"""
import json
import os
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from executor_memory import render_block                                 # noqa: E402
from llm import default_chat                                             # noqa: E402
import runner as _runner                                                 # noqa: E402

DOMAIN = 'alfworld'
MAX_STEPS = int(os.environ.get('ALFW_MAX_STEPS', '20'))
TEMPERATURE = float(os.environ.get('ALFW_AGENT_TEMP', '0.4'))
MAX_TOKENS = int(os.environ.get('ALFW_AGENT_MAX_TOKENS', '2048'))
BASE = Path(os.environ.get('MEMSEL_MEMORY_BASE',
                           Path(os.environ.get('MEMCOMPOSER_DATA', HERE.parents[1] / 'data'))
                           / 'alfworld' / 'memory_banks' / 'ministral8b'))
SIM_LOG = os.environ.get('ALFW_SIM_LOG')

_CHAT = None
_CHAT_LOCK = threading.Lock()
_LOG_LOCK = threading.Lock()
_META = None


def _chat():
    global _CHAT
    with _CHAT_LOCK:
        if _CHAT is None:
            _CHAT = default_chat(max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
        return _CHAT


def _tasks():
    return _runner.index()


def _meta():
    """memory_id -> folder, for every memory in the base."""
    global _META
    if _META is None:
        _META = {p.parent.name: p.parent for p in BASE.glob('*/memory.json')}
        assert _META, f'no memories under {BASE}'
    return _META


def experience_from(folders):
    """The executor's memory block for this selection, '' for an empty one. Deterministic: its
    hash is the service's dedup key."""
    meta = _meta()
    missing = [f for f in folders if f not in meta]
    if missing:
        raise KeyError(f'memories not in {BASE}: {missing[:3]}')
    return render_block(list(folders), BASE)


def run_one(task_id, opening, folders, seed=0, agent_base=None, max_steps=MAX_STEPS):
    if task_id not in _tasks():
        raise KeyError(f'unknown task_id {task_id}')
    block = experience_from(folders)
    rec = _runner.run_one(task_id, _chat(), memories=[block] if block else None,
                          max_steps=int(max_steps or MAX_STEPS), temperature=TEMPERATURE,
                          keep_trajectory=bool(SIM_LOG))
    if rec.get('error'):
        raise RuntimeError(f"simulation failed: {rec['error']}")
    if SIM_LOG:
        with _LOG_LOCK, open(SIM_LOG, 'a') as f:
            f.write(json.dumps({**rec, 'folders': list(folders), 'seed': seed, 'opening': opening},
                               ensure_ascii=False) + '\n')
    return {
        'task_id': task_id, 'folders': list(folders), 'seed': seed,
        'reward': float(rec['won']),
        'termination': rec['end_reason'],
        'n_messages': int(rec['steps']),
        'won': int(rec['won']), 'won_step': rec['won_step'], 'steps': rec['steps'],
        'env_actions': rec['env_actions'], 'n_memories': len(folders),
        'llm_calls': rec['llm_calls'], 'prompt_tokens': rec['prompt_tokens'],
        'completion_tokens': rec['completion_tokens'],
        'valid_action_ratio': rec['valid_action_ratio'],
    }
