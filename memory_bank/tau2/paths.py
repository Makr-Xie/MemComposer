"""Where the memory-construction pipeline reads and writes.

Everything hangs off `<MEMCOMPOSER_DATA>/tau2` (`MEMCOMPOSER_DATA` defaults to `<repo>/data`):

    rollouts/<agent>/<domain>/              sims.jsonl, k0.op{0..3}.jsonl (written by collect.sh)
    openings/<domain>/train_op{j}.parquet   the four sampled customer openings per train task
    memory_banks/<agent>/<domain>/          selection*.json, labels, task*/, index.npz

tau2-bench is found through `TAU2_ROOT`; leave it unset if `tau2` is already importable.
"""
import os
from pathlib import Path

DATA = Path(os.environ.get('MEMCOMPOSER_DATA', Path(__file__).resolve().parents[2] / 'data')) / 'tau2'
TAU2_ROOT = os.environ.get('TAU2_ROOT')
DOMAINS = ['airline', 'retail', 'telecom']


def rollouts(agent, domain):
    return DATA / 'rollouts' / agent / domain


def openings(domain, j):
    return DATA / 'openings' / domain / f'train_op{j}.parquet'


def bank(agent):
    return DATA / 'memory_banks' / agent
