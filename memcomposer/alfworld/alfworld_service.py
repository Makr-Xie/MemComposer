#!/usr/bin/env python
"""Local HTTP wrapper around one ALFWorld simulation -- memcomposer/tau2/tau2_service.py with
only the simulator swapped. Runs in the ALFWorld Python environment.

    python alfworld_service.py --port 8160 --workers 32

Everything below the import is the same as tau2_service.py: the scoped dedup table, in-flight
futures, JOIN_TIMEOUT fallback, /health, /reset_dedup. Differences: the simulator comes from
alfworld_runner, the domain is fixed to 'alfworld', and MEMSEL_JOIN_TIMEOUT defaults to 900 s
because an ALFWorld game with many memories runs much longer than a tau2 conversation.

Env vars: MEMSEL_JOIN_TIMEOUT (seconds a duplicate request waits for the leader, default 900),
MEMSEL_DEDUP_TTL (seconds a cached result stays valid, default 1800); the simulator's own
settings are documented in alfworld_runner.py.

  POST /run   {"task_id","opening","folders":[...],"seed","scope","dedup","max_steps"}  ->  {"reward","termination","n_messages",...}
  GET  /health
  POST /reset_dedup
"""
import argparse, asyncio, hashlib, os, sys, time
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault('MEMSEL_JOIN_TIMEOUT', '900')
from alfworld_runner import run_one, experience_from, _tasks, _meta, MAX_STEPS, DOMAIN

WORKERS = 48   # overwritten from argv before uvicorn starts


@asynccontextmanager
async def lifespan(app):
    # starlette's default threadpool is 40 threads and a blocking run_one holds one for
    # minutes, so the limiter has to be raised above the reward function's concurrency.
    # It can only be read from inside the loop, hence lifespan rather than module scope.
    anyio.to_thread.current_default_thread_limiter().total_tokens = WORKERS
    print(f'threadpool limit set to {WORKERS}', flush=True)
    yield


app = FastAPI(title='alfworld-sim', lifespan=lifespan)


# Deduplication has to live here, not in the reward function. verl runs several
# RewardLoopWorker actors, each a separate process with its own state, and a GRPO group's
# rollouts are split across them -- so a per-process cache would let the same selection be
# simulated once per actor, and identical actions in one group could then receive different
# rewards and a spurious advantage. This service is a single process, so a table here is global:
# the first request for a key runs the simulation, every concurrent duplicate awaits the same
# future and gets the identical reward.
#
# The key hashes the rendered memory block, not the folder list, so any two selections that
# produce the same executor prompt share one simulation (an empty selection and an unparseable
# one, say).
#
# The table is scoped: a request carries `scope`, the reward function's label for the current
# training step (_step_scope() in memcomposer/reward.py), and a result is reused only by a request
# with the same scope. Merging is meant for the rollouts of one GRPO group in one step --
# identical actions there must earn identical reward. It must not carry a result across steps: a
# prompt drawn again later has to be simulated afresh, otherwise every later step, and every eval
# that happens to make the same selection, inherits one old draw of the simulator's noise. So when
# a request arrives with a new scope the whole table is dropped: verl finishes all of a step's
# rewards before the next step generates, so nothing from the old scope is still wanted. Eval
# requests come with dedup=false and never touch the table. As a second guard an entry also
# expires DEDUP_TTL seconds after it was made, which only matters if the scope label cannot move
# (rollout dumps disabled).
_INFLIGHT: dict[str, asyncio.Future] = {}
_RESULTS: dict[str, dict] = {}          # scoped key -> result, for the current scope only
_SCOPE = ['']                           # scope the table currently belongs to
_STATS = {'requests': 0, 'simulated': 0, 'inflight_hits': 0, 'result_hits': 0, 'scope_resets': 0,
          'join_timeouts': 0, 'join_fallbacks': 0}
# How long a follower waits for the leader's simulation before running its own (set to 900 s
# above for ALFWorld): long enough that a merely slow leader is still shared, short enough that a
# dead one costs minutes instead of the whole run.
JOIN_TIMEOUT = float(os.environ.get('MEMSEL_JOIN_TIMEOUT', '420'))
DEDUP_TTL = float(os.environ.get('MEMSEL_DEDUP_TTL', '1800'))


def _key(task_id, opening, folders, seed, scope=''):
    exp = experience_from(folders)
    return (f"{scope}|{task_id}|{hashlib.sha1(opening.strip().encode()).hexdigest()[:12]}"
            f"|{seed}|{hashlib.sha1(exp.encode()).hexdigest()[:16]}")


def _enter_scope(scope):
    """Retire the table when the step changes. In-flight futures are left alone: a straggler
    from the old step still gets its own result, it just cannot be reused by the new one."""
    if scope != _SCOPE[0]:
        if _RESULTS:
            _STATS['scope_resets'] += 1
        _RESULTS.clear()
        _SCOPE[0] = scope


class RunReq(BaseModel):
    task_id: str
    opening: str
    folders: list[str] = []
    seed: int = 0
    max_steps: int = MAX_STEPS
    dedup: bool = True
    scope: str = ''         # training-step label from the reward function; '' = unscoped


@app.get('/health')
def health():
    return {'ok': True, 'domain': DOMAIN, 'tasks': len(_tasks()), 'memories': len(_meta()), **_STATS,
            'scope': _SCOPE[0], 'table': len(_RESULTS), 'inflight': len(_INFLIGHT), 'ttl': DEDUP_TTL}


@app.post('/reset_dedup')
def reset_dedup():
    """Drop the completed-result table between runs; in-flight entries are left alone."""
    n = len(_RESULTS)
    _RESULTS.clear()
    _SCOPE[0] = ''
    for k in _STATS: _STATS[k] = 0
    return {'cleared': n}


async def _simulate(req: RunReq, t0):
    r = await run_in_threadpool(run_one, req.task_id, req.opening, req.folders,
                                req.seed, None, req.max_steps)
    _STATS['simulated'] += 1
    return {**r, 'secs': round(time.time() - t0, 1), 'error': None, 'dedup': 'miss'}


@app.post('/run')
async def run(req: RunReq):
    t0 = time.time()
    _STATS['requests'] += 1
    try:
        if not req.dedup:
            return await _simulate(req, t0)
        _enter_scope(req.scope)
        k = _key(req.task_id, req.opening, req.folders, req.seed, req.scope)
        hit = _RESULTS.get(k)
        if hit is not None and time.time() - hit['_done_at'] < DEDUP_TTL:
            _STATS['result_hits'] += 1
            return {**{a: b for a, b in hit.items() if a != '_done_at'},
                    'secs': round(time.time() - t0, 1), 'dedup': 'result'}
        fut = _INFLIGHT.get(k)
        if fut is not None:
            _STATS['inflight_hits'] += 1
            # shield so our timeout never cancels the leader's simulation; wait_for so a leader
            # that dies, hangs or is cancelled cannot park us here forever (a bare await would
            # stall the whole training step on one missing reward).
            try:
                out = await asyncio.wait_for(asyncio.shield(fut), timeout=JOIN_TIMEOUT)
            except (asyncio.TimeoutError, TimeoutError):
                _STATS['join_timeouts'] += 1
                out = None
            if out is not None:
                return {**out, 'secs': round(time.time() - t0, 1), 'dedup': 'inflight'}
            # leader failed, was cancelled, or is past JOIN_TIMEOUT -- simulate ourselves. The
            # draw will differ from the leader's, which only happens on the failure path.
            _STATS['join_fallbacks'] += 1
            return await _simulate(req, t0)
        fut = asyncio.get_running_loop().create_future()
        _INFLIGHT[k] = fut
        try:
            out = await _simulate(req, t0)
            if _SCOPE[0] == req.scope:          # the step may have moved on while this ran
                _RESULTS[k] = {**out, '_done_at': time.time()}
            if not fut.done(): fut.set_result(out)
            return out
        finally:
            _INFLIGHT.pop(k, None)
            # Always settle the future. _simulate raising is caught by the outer `except
            # Exception` (which returns an error dict and never touches fut), and cancellation
            # is a BaseException that handler cannot catch at all -- either way every follower
            # parked on this future would otherwise wait forever. None means "leader failed,
            # run your own".
            if not fut.done():
                fut.set_result(None)
    except Exception as e:
        err = {'task_id': req.task_id, 'folders': req.folders, 'seed': req.seed,
               'reward': None, 'secs': round(time.time() - t0, 1),
               'error': f'{type(e).__name__}: {e}'[:300]}
        return err


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8160)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--workers', type=int, default=32,
                    help='threadpool size = simulations this process runs at once')
    a = ap.parse_args()
    WORKERS = a.workers
    import uvicorn
    print(f'alfworld service on {a.host}:{a.port}, domain {DOMAIN}, threadpool {a.workers}, '
          f'tasks {len(_tasks())}, memories {len(_meta())}, join_timeout {JOIN_TIMEOUT}', flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level='warning')
