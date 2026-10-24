"""Where the ALFWorld memory-construction pipeline reads and writes.

Everything hangs off `<MEMCOMPOSER_DATA>/alfworld` (`MEMCOMPOSER_DATA` defaults to `<repo>/data`):

    task_index.jsonl                          one row per playable game        build_task_index.py
    rollouts/<model>/train_full_s20.jsonl     raw no-memory trajectories       collect_raw.py
    memory_banks/<model>/_work/               facts.jsonl, reviews_*.jsonl     build_facts.py, self_review.py
    memory_banks/<model>/<group_id>__<trial>/ memory.json, full.txt, digest.txt  build_memory_base.py
    retrieval/<model>/                        frozen top-k candidate pools     build_*retrieval.py

Game files are read from `ALFWORLD_DATA` (the directory `alfworld-download` fills), addressed by
task_id, so records written on another machine resolve here.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get('MEMCOMPOSER_DATA', REPO / 'data')) / 'alfworld'
ALFWORLD_DATA = Path(os.environ.get('ALFWORLD_DATA', Path.home() / '.cache' / 'alfworld'))
RUNTIME = REPO / 'memcomposer' / 'alfworld'      # executor: env, runner, LLM client, prompts


def task_index():
    return DATA / 'task_index.jsonl'


def rollouts(model):
    return DATA / 'rollouts' / model / 'train_full_s20.jsonl'


def bank(model):
    return DATA / 'memory_banks' / model


def work(model):
    return bank(model) / '_work'


def retrieval(model):
    return DATA / 'retrieval' / model


def gamefile(task_id):
    """`<split>/<taskdir>/<trial>` -> its game.tw-pddl under ALFWORLD_DATA."""
    return ALFWORLD_DATA / 'json_2.1.1' / task_id / 'game.tw-pddl'
