#!/usr/bin/env python
"""Local HTTP wrapper around one tau2 simulation. Runs in the tau2 venv.

    python tau2_service.py --domain retail --port 8100 --workers 48

verl and tau2 cannot share an interpreter (tau2 needs litellm and a newer pydantic than the verl
env has), so the reward function calls this service over localhost instead of importing tau2.
This also keeps the simulation threads out of verl's driver process.

One service serves one domain, fixed by --domain at startup; a request for another domain's task
would be graded against the wrong task set without any error. The service ignores SIGTERM; stop
it with kill -9.

  POST /run   {"task_id","opening","folders":[...],"seed","scope","dedup"}  ->  {"reward", "termination", ...}
  GET  /health
  POST /reset_dedup

Env vars: MEMSEL_JOIN_TIMEOUT (s, default 420) and MEMSEL_DEDUP_TTL (s, default 1800), see below;
the simulation itself is configured by tau2_runner's env vars.
"""
import argparse, asyncio, hashlib, os, sys, time
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

# The domain is process-wide (tau2_runner builds its task map and memory map once), so it has
# to be in the environment BEFORE that import. Pre-parse argv rather than relying on the caller
# to export it: one service per domain, and a mismatch would silently grade against the wrong
# task set.
_dom = 'airline'
for _i, _a in enumerate(sys.argv):
    if _a == '--domain' and _i + 1 < len(sys.argv):
        _dom = sys.argv[_i + 1]
    elif _a.startswith('--domain='):
        _dom = _a.split('=', 1)[1]
assert _dom in ('airline', 'retail', 'telecom'), f'unknown domain {_dom!r}'
os.environ['MEMSEL_DOMAIN'] = _dom

from tau2_runner import run_one, experience_from, _tasks, _meta, MAX_STEPS, DOMAIN

WORKERS = 48   # overwritten from argv before uvicorn starts


@asynccontextmanager
async def lifespan(app):
    # starlette's default threadpool is 40 threads and a blocking run_one holds one for
    # minutes, so the limiter has to be raised above the reward function's concurrency.
    # It can only be read from inside the loop, hence lifespan rather than module scope.
    anyio.to_thread.current_default_thread_limiter().total_tokens = WORKERS
    print(f'threadpool limit set to {WORKERS}', flush=True)
    yield


app = FastAPI(title='tau2-sim', lifespan=lifespan)


# Deduplication lives here, not in the reward function: verl's RewardLoopWorker actors are
# separate processes and a GRPO group's rollouts are split across them, so a per-process cache
# would simulate the same selection once per actor, and identical actions could then earn
# different rewards (a spurious advantage). This service is one process, so the table is global:
# the first request for a key runs the simulation and every concurrent duplicate awaits the same
# future and gets the identical reward.
#
# The key hashes the rendered <experience> block rather than the folder list, so any two
# selections that yield the same prompt (e.g. an empty and an unparseable one) share a result.
#
# The table is scoped: a request carries `scope`, the reward function's label for the current
# training step (memcomposer/reward.py, _step_scope), and a result is reused only within the same
# scope. Merging is meant for one GRPO group in one step; a result must not carry over to later
# steps or evals, or they would all inherit one old draw of tau2's noise. When a request arrives
# with a new scope the whole table is dropped (verl finishes a step's rewards before the next step
# generates). Eval requests send dedup=false and never touch the table. As a second guard an
# entry expires DEDUP_TTL seconds after it was made, which matters only if the scope never changes.
_INFLIGHT: dict[str, asyncio.Future] = {}
_RESULTS: dict[str, dict] = {}          # scoped key -> result, for the current scope only
_SCOPE = ['']                           # scope the table currently belongs to
_STATS = {'requests': 0, 'simulated': 0, 'inflight_hits': 0, 'result_hits': 0, 'scope_resets': 0,
          'join_timeouts': 0, 'join_fallbacks': 0}
# Seconds a follower waits for the leader's simulation of the same key before running its own:
# long enough (about 2x a slow normal simulation) that a slow leader is still shared, short
# enough that a dead or hung leader costs minutes instead of stalling the run.
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
            # that dies, hangs or is cancelled cannot park this request forever.
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
            # ALWAYS settle the future, or followers parked on it wait forever: an exception in
            # _simulate goes to the outer handler, which never touches fut, and cancellation is a
            # BaseException it cannot catch. None means "leader failed, run your own".
            if not fut.done():
                fut.set_result(None)
    except Exception as e:
        err = {'task_id': req.task_id, 'folders': req.folders, 'seed': req.seed,
               'reward': None, 'secs': round(time.time() - t0, 1),
               'error': f'{type(e).__name__}: {e}'[:300]}
        return err


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', default='airline', choices=['airline', 'retail', 'telecom'],
                    help='pre-parsed at import time; declared here so --help shows it')
    ap.add_argument('--port', type=int, default=8100)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--workers', type=int, default=48,
                    help='threadpool size; must exceed the reward function concurrency')
    a = ap.parse_args()
    WORKERS = a.workers
    import uvicorn
    assert a.domain == DOMAIN, f'--domain {a.domain} but module bound to {DOMAIN}'
    print(f'tau2 service on {a.host}:{a.port}, domain {DOMAIN}, '
          f'threadpool {a.workers}', flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level='warning')
