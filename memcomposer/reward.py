#!/usr/bin/env python
"""verl reward function for the memory selector.

    reward.custom_reward_function.path=<repo>/memcomposer/reward.py
    reward.custom_reward_function.name=compute_score

The policy's reply names memories by their 1-based rank in the candidate pool it was shown. We map
those back to memory-bank folders, the simulation service renders them into <experience>, replays
the conversation from the same customer opening the retrieval used, and returns tau2's own reward.

    score = task_reward - fmt_cost - k_cost

  * task_reward: mean success over the row's simulations (MEMSEL_TRAIN_TRIALS for training rows,
    the row's own n_trials for eval rows).
  * fmt_cost: PEN_NO_REASONING when a reply gave a legal <answer> but no <reasoning> block.
  * k_cost: K_PENALTY per selected case above K_MAX, plus K_MIN_PENALTY per case below K_MIN.

Two cases that look alike and are not:
  * no closed <answer> block, or one naming numbers that were never shown -- a format failure.
    Scores 0 WITHOUT simulating (see read_answer), i.e. an effective penalty of 1.0.
  * an explicit `[]` -- a real decision that no case helps. This IS simulated and earns whatever
    the agent earns without memory.

By default the k penalty is charged only on solved rollouts, so a failed rollout scores a flat 0
whatever its size and the penalty ranks ways of succeeding, never ways of failing. Every penalty
must stay below one trial step (1/8 at 8 trials), or a fully solved selection could rank below a
partially solved one. All knobs default to off/neutral here; the training scripts under training/
set them per run.

Eval rows (n_trials > 1, or is_eval=1) are scored as raw pass^1 with no penalties, never deduped
and never use partial credit, so they stay comparable to tau2's leaderboard numbers.

Simulations go to a simulation service over HTTP rather than through an import: tau2 needs
litellm and a newer pydantic than the verl env has, and this module is imported by verl. Start
the service first, e.g.

    python memcomposer/tau2/tau2_service.py --domain retail --port 8100 --workers 48

Main env vars: MEMSEL_TAU2_URL / MEMSEL_TAU2_URLS (service routing), MEMSEL_TRAIN_TRIALS,
MEMSEL_PEN_NO_REASONING, MEMSEL_K_PENALTY / MEMSEL_K_MAX, MEMSEL_K_MIN_PENALTY / MEMSEL_K_MIN,
MEMSEL_REWARD_DEDUP, MEMSEL_REWARD_LOG, MEMSEL_REWARD_CACHE (off by default).
"""
import asyncio, hashlib, json, os, re, sys, time, traceback
from pathlib import Path

import aiohttp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'tau2'))

from memcomposer_prompt import parse_selection, extract_answer   # noqa: E402

TAU2_URL = os.environ.get('MEMSEL_TAU2_URL', 'http://127.0.0.1:8100')
# Per-domain routing. A service is bound to ONE domain and the /run payload carries none, so a
# retail row sent to the airline service is silently graded on whatever airline task shares its
# id. Mixed-domain runs therefore route on verl's data_source:
#   MEMSEL_TAU2_URLS='memsel/retail=http://127.0.0.1:8101,memsel/airline=http://127.0.0.1:8102'
# Unset -> every row goes to TAU2_URL. A data_source may list several services ('|'-separated);
# throughput is bounded per service process, so more services means more simulations per second.
# Among several services the choice is a stable hash of the simulation key, NOT round-robin:
# identical (task, opening, folders, seed) must hit the same service so its dedup still collapses
# them, or two identical selections could get different rewards.
TAU2_URLS = {}
for _pair in os.environ.get('MEMSEL_TAU2_URLS', '').split(','):
    if '=' in _pair:
        _k, _v = _pair.split('=', 1)
        TAU2_URLS[_k.strip()] = [u.strip().rstrip('/') for u in _v.split('|') if u.strip()]


def _urls_for(data_source):
    if not TAU2_URLS:
        return [TAU2_URL]
    u = TAU2_URLS.get(str(data_source))
    if not u:
        raise KeyError(f'MEMSEL_TAU2_URLS has no entry for data_source {data_source!r}; '
                       f'known: {sorted(TAU2_URLS)}')
    return u


def _url_for(data_source, task_id='', opening='', folders=(), seed=0):
    urls = _urls_for(data_source)
    if len(urls) == 1:
        return urls[0]
    h = hashlib.sha1(f'{task_id}|{opening.strip()}|{",".join(folders)}|{seed}'.encode()).digest()
    return urls[int.from_bytes(h[:4], 'big') % len(urls)]
SIM_TIMEOUT = int(os.environ.get('MEMSEL_SIM_TIMEOUT', '1800'))
MAX_CONCURRENT_SIMS = int(os.environ.get('MEMSEL_REWARD_CONCURRENCY', '32'))
# Cost of a reply that gave a legal <answer> but skipped the <reasoning> block (a missing <answer>
# never reaches here: read_answer scores it 0 without simulating). It exists because the hard
# <answer> gate makes "stop reasoning" the cheapest way never to derail, and dropping reasoning
# collapses within-group diversity of selections. Keep it small: a larger value amplifies every
# gradient the reward carries, including the pressure to write short replies.
# MEMSEL_FORMAT_PENALTY is accepted as an older alias.
PEN_NO_REASONING = float(os.environ.get('MEMSEL_PEN_NO_REASONING',
                                        os.environ.get('MEMSEL_FORMAT_PENALTY', '0.05')))
FORMAT_PENALTY = PEN_NO_REASONING          # alias kept for existing callers
# Upper-bound k penalty: each selected case beyond K_MAX costs K_PENALTY. 0 disables it.
K_PENALTY = float(os.environ.get('MEMSEL_K_PENALTY', '0.0'))
# Free allowance: selections of K_MAX cases or fewer pay nothing. Charging from k=0 would make the
# penalty the only term separating two solved selections and push toward small k even where more
# memories help; an allowance keeps it as a brake on runaway k only. Defaults to 0 when the penalty
# is off and to 4 when it is on; an explicit MEMSEL_K_MAX always wins.
K_MAX = int(os.environ.get('MEMSEL_K_MAX', '4' if K_PENALTY > 0 else '0'))
# MEMSEL_K_PEN_ALWAYS=1 charges the k penalty at every task_reward level, not only on solved
# rollouts. Gated on success, the term can order only the rare tied pairs that both solved; charged
# always, it orders far more tied pairs, all toward smaller k.
# Requires DAPO filter_groups on task_reward: it drops all-failed (zero-spread) groups, which are
# exactly the groups where an ungated penalty would teach "select nothing".
# Magnitude only needs to break ties; keep K_PENALTY * (k_pool - K_MAX) < 0.125 (one trial step).
K_PEN_ALWAYS = os.environ.get('MEMSEL_K_PEN_ALWAYS', '0') not in ('0', 'false', 'False')

# ---------------------------------------------------------------------------------------------
# Lower bound on k, independent of the upper bound above: a brake on k COLLAPSING rather than
# running away. The task reward itself can favour smaller selections within a group, so k may
# drift below the useful range even with the upper-bound penalty off.
# Each case below K_MIN costs K_MIN_PENALTY; k >= K_MIN pays nothing. As a tiebreaker the worst
# case (k=0) costs K_MIN_PENALTY * K_MIN, which must stay under one trial step (0.125).
# Defaults 0/0 = off; with the penalty on, K_MIN defaults to 3.
K_MIN_PENALTY = float(os.environ.get('MEMSEL_K_MIN_PENALTY', '0.0'))
K_MIN = int(os.environ.get('MEMSEL_K_MIN', '3' if K_MIN_PENALTY > 0 else '0'))
# Charge at every task_reward level by default when the penalty is on: a lower bound that fires
# only on solved rollouts cannot stop a collapse driven by failures.
K_MIN_ALWAYS = os.environ.get('MEMSEL_K_MIN_ALWAYS',
                              '1' if K_MIN_PENALTY > 0 else '0') not in ('0', 'false', 'False')
if K_MIN_PENALTY > 0:
    _worst_lo = K_MIN_PENALTY * max(0, K_MIN)
    if _worst_lo >= 0.125:
        import warnings as _w2
        _w2.warn(f"MEMSEL_K_MIN_PENALTY={K_MIN_PENALTY} with K_MIN={K_MIN} gives a worst-case "
                 f"cost of {_worst_lo:.4f} >= 0.125, so a fully solved k=0 selection can rank "
                 f"below a 7/8 one. Lower it below {0.125 / max(1, K_MIN):.4f}.", RuntimeWarning)
# ---------------------------------------------------------------------------------------------
if K_PEN_ALWAYS and K_PENALTY > 0:
    # Ordering bound: a fully solved selection must not rank below a 7/8 one. The worst case is a
    # full-pool selection, so K_PENALTY * (pool - K_MAX) must stay under one trial step (1/8).
    _worst = K_PENALTY * max(0, 10 - K_MAX)
    if _worst >= 0.125:
        import warnings as _w
        _w.warn(f"MEMSEL_K_PENALTY={K_PENALTY} with K_MAX={K_MAX} gives a worst-case k cost of "
                f"{_worst:.4f} >= 0.125 (one trial step at 8 trials): a fully solved selection at "
                f"k=10 would rank BELOW a 7/8 solve. Lower K_PENALTY to under "
                f"{0.125/max(1, 10-K_MAX):.4f} or raise K_MAX.", RuntimeWarning)
# Weights for partial_credit(). Only used when MEMSEL_PARTIAL=1, and never for eval rows.
W_DB = float(os.environ.get('MEMSEL_W_DB', '0.6'))
W_COMM = float(os.environ.get('MEMSEL_W_COMM', '0.2'))
W_ACTION = float(os.environ.get('MEMSEL_W_ACTION', '0.2'))
# Score TRAINING rollouts on the graded blend instead of tau2's 0/1. Off by default because it
# changes what the number means. Eval rows ignore it.
PARTIAL = os.environ.get('MEMSEL_PARTIAL', '0') not in ('0', 'false', 'False')
SIM_RETRIES = int(os.environ.get('MEMSEL_SIM_RETRIES', '1'))
# tau2's own per-trial seeds, reproduced exactly: tau2 seeds the global RNG with DEFAULT_SEED=300
# and draws one integer per trial, reusing the list for every task. The seed reaches agent and
# user LLM calls, so at temperature 0 it is only a determinism hint; the trials measure residual
# nondeterminism, as in the leaderboard protocol.
TAU2_SEED = int(os.environ.get('MEMSEL_TAU2_SEED', '300'))
# Eval rows are reported like tau2's pass^1: raw mean task success, no format or k penalty, since
# a k penalty would shift the number against a no-memory reference that has no k at all.
# Set 0 to score eval rows like training rows.
EVAL_RAW_SCORE = os.environ.get('MEMSEL_EVAL_RAW_SCORE', '1') not in ('0', 'false', 'False')
# Number of simulations a TRAINING rollout is averaged over (pass^1). A single 0/1 draw on a task
# with pass rate near 0.5 is a coin flip, while penalties are deterministic, so with one trial
# GRPO tends to learn the penalties rather than memory selection. More trials cut the reward
# noise at a proportional cost in simulations.
TRAIN_TRIALS = max(1, int(os.environ.get('MEMSEL_TRAIN_TRIALS', '1')))
# Turn cap for TRAINING simulations only, sent per request; 0 = service default (tau2's official
# value, which eval always uses). A step's simulation phase waits for its slowest conversation,
# and the longest conversations are mostly failures anyway. A capped simulation scores 0.
TRAIN_MAX_STEPS = int(os.environ.get('MEMSEL_TRAIN_MAX_STEPS', '0'))


def _trial_seeds(n):
    import random
    r = random.Random(TAU2_SEED)
    return [r.randint(0, 1000000) for _ in range(n)]
# Rollouts in a GRPO group are scored concurrently, so an on-disk cache never helps: duplicates
# are all in flight at once. DEDUP collapses them instead: the first arrival runs the simulation,
# the rest await the same future and get the identical reward, so identical selections get an
# advantage of exactly 0 between them rather than a spurious gradient from tau2's noise.
# MEMSEL_REWARD_DEDUP=0 scores every rollout independently.
DEDUP = os.environ.get('MEMSEL_REWARD_DEDUP', '1') not in ('0', 'false', 'False')
LOG_PATH = os.environ.get('MEMSEL_REWARD_LOG', str(HERE / 'out' / 'reward_calls.jsonl'))
# Where verl dumps rollouts/<step>.jsonl (trainer.rollout_data_dir). Used to detect the current
# training step -- see _step_scope().
ROLLOUT_DIR = os.environ.get('MEMSEL_ROLLOUT_DIR', str(Path(LOG_PATH).parent / 'rollouts'))
CACHE_PATH = os.environ.get('MEMSEL_REWARD_CACHE')          # unset -> no caching

_sem = None
_session = None
_cache = None
_log_lock = None
_inflight = None      # sim key -> asyncio.Future, for the in-flight deduplication


def _lazy():
    """Built on first use: verl imports this module in the driver before the loop exists."""
    global _sem, _session, _cache, _log_lock, _inflight
    if _sem is None:
        _inflight = {}
        _sem = asyncio.Semaphore(MAX_CONCURRENT_SIMS)
        _session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=SIM_TIMEOUT),
            connector=aiohttp.TCPConnector(limit=MAX_CONCURRENT_SIMS + 8))
        _log_lock = asyncio.Lock()
        Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        if CACHE_PATH:
            from reward_cache import RewardCache
            _cache = RewardCache(CACHE_PATH)
    return _sem, _session, _cache, _log_lock


def _step_scope():
    """A label that changes exactly once per training step, obtained without help from verl.

    The service's dedup table must only merge identical selections within ONE GRPO group of ONE
    step; a prompt drawn again later has to be simulated afresh. verl does not pass uid or
    global_steps to a custom reward function, but it writes rollouts/<step>.jsonl after each
    step's rewards and before the next step generates, so the newest dump's mtime moves once per
    step. The service keys its table on this label and drops it when the label changes. The group
    itself is already in the key (task, opening). With rollout dumps disabled the label is constant
    and the service falls back to its TTL.
    """
    try:
        m = max((p.stat().st_mtime for p in Path(ROLLOUT_DIR).glob('*.jsonl')), default=None)
    except OSError:
        m = None
    return f'step@{m:.0f}' if m else 'step@none'


_REASONING = re.compile(r'<reasoning>.*?</reasoning>', re.S)
_LIST = re.compile(r'\[\s*(?:\d+\s*(?:,\s*\d+\s*)*)?\]')


def read_answer(reply, valid):
    """The selection the reply committed to, and why it is not one when it is not.

    status is one of:
      'ok'         a closed <answer> block holding a list every number of which was shown
      'no_answer'  no closed <answer> block, or one that is not EXACTLY one bare-integer
                   list -- a FORMAT FAILURE. Quoted numbers (`["1", "2"]`), numbers each in
                   their own brackets (`["[1]", "[2]"]`, `[3][2][1]`, `[7], [6], [5]`) and
                   anything with prose beside the list all land here.
      'no_valid'   a closed block holding a NON-EMPTY list with ANY number that was not shown

    'no_valid' is all-or-nothing on purpose: `[2, 99]` names a case that was never on the menu,
    and keeping the 2 would reward a half-hallucinated reply as if it had chosen deliberately.

    `[]` inside a closed block is a decision -- use no memory -- and is simulated, since that is
    the no-memory baseline and the policy may choose it. The other two statuses score 0 without
    a simulation. Without this split, a derailed reply ("Case [142] is ...") would be parsed,
    filtered to an empty selection and scored as a deliberate no-memory choice.
    """
    block = extract_answer(reply)
    if block is None:
        return [], 'no_answer'
    # fullmatch, not findall: the WHOLE block must be one bare-integer list. Otherwise a block like
    # `["[1]", "[2]", "[7]"]` or `[3][2][1]` would silently become a selection of a single case.
    m = _LIST.fullmatch(block)
    if m is None:
        return [], 'no_answer'
    # Parse the BLOCK, not the whole reply: parse_selection falls back to "the last bracketed list
    # after </reasoning>" when it sees no tags, which would let a derailed reply through.
    raw = json.loads(m.group(0))
    if raw and not all(i in valid for i in raw):
        return [], 'no_valid'
    picked = parse_selection(block, valid=valid)
    return picked, 'ok'


def check_format(reply, thinking=True):
    """Sloppy means parseable but not the shape the prompt asked for.

    With thinking=0 the prompt asks for the answer block alone and never mentions <reasoning>,
    so requiring one would penalise every rollout and distort the logged format metrics.
    """
    text = (reply or '').strip()
    has_reasoning = bool(_REASONING.search(text))
    block = extract_answer(text)
    has_answer = block is not None and _LIST.fullmatch(block) is not None
    clean = has_answer and (has_reasoning if thinking else not has_reasoning)
    return {'has_reasoning': has_reasoning, 'has_answer': has_answer, 'clean': clean}


# verl turns every key of the returned dict into np.array over the batch, taking the key set
# from the FIRST sample. So every call must return the same keys and every value must be a
# scalar (a variable-length list crashes with "inhomogeneous shape"). The selection and the
# reply text go to the JSONL log instead.
_FIELDS = ('score', 'task_reward', 'n_trials', 'trials_ok', 'trials_passed', 'k_cost', 'parse_failed',
           'is_eval', 'partial_reward', 'db_reward', 'comm_reward', 'action_match', 'action_total',
           'simulated', 'cache_hit', 'dedup_hit',
           'format_clean', 'has_reasoning', 'has_answer', 'n_selected', 'k_pool', 'window',
           'n_messages', 'secs', 'termination', 'sim_error', 'task_id')


def _result(**kw):
    """Fixed key set, scalars only, in a stable order."""
    base = {'score': 0.0, 'task_reward': 0.0, 'n_trials': 1, 'trials_ok': 0, 'trials_passed': 0,
            'is_eval': 0, 'partial_reward': 0.0, 'db_reward': -1.0, 'comm_reward': -1.0,
            'action_match': 0, 'action_total': 0,
            'k_cost': 0.0, 'parse_failed': 0, 'simulated': 0, 'cache_hit': 0,
            'dedup_hit': 0, 'format_clean': 0, 'has_reasoning': 0, 'has_answer': 0, 'n_selected': 0,
            'k_pool': 0, 'window': -1, 'n_messages': -1, 'secs': 0.0,
            'termination': '', 'sim_error': '', 'task_id': ''}
    base.update(kw)
    out = {}
    for k in _FIELDS:
        v = base[k]
        if isinstance(v, bool):
            v = int(v)
        elif hasattr(v, 'item'):            # numpy scalar
            v = v.item()
        out[k] = v if isinstance(v, (int, float, str)) else str(v)
    return out


def partial_credit(sim_rows):
    """A graded stand-in for tau2's all-or-nothing reward, from parts it already computed.

    tau2 multiplies the components a task lists in reward_basis, so one zero sinks the run and
    the rest is discarded; e.g. "said the right thing, left the database wrong" scores the same as
    getting everything wrong. A weighted blend of DB / COMMUNICATE / action-match gives GRPO
    groups more within-group variance at no extra simulation cost.

    The weights are ours (tau2 has none). DB keeps the majority so a solved task never ranks below
    an unsolved one, and satisfying every basis component still scores exactly 1.0.
    EVALUATION MUST NOT USE THIS: eval stays on tau2's own number so results compare to baselines
    and the leaderboard.
    """
    out = []
    for r in sim_rows:
        bd = r.get('reward_breakdown') or {}
        basis = set(r.get('reward_basis') or bd.keys())
        parts, weights = [], []
        for name, w in (('DB', W_DB), ('ENV_ASSERTION', W_DB), ('COMMUNICATE', W_COMM),
                        ('NL_ASSERTION', W_COMM)):
            for key in (name, f'RewardType.{name}'):
                if key in bd and (name in basis or f'RewardType.{name}' in basis):
                    parts.append(float(bd[key])); weights.append(w); break
        tot = int(r.get('action_total') or 0)
        if tot:
            parts.append(int(r.get('action_match') or 0) / tot); weights.append(W_ACTION)
        out.append(sum(p * w for p, w in zip(parts, weights)) / sum(weights) if parts
                   else float(r['reward']))
    return sum(out) / len(out)


def _k_cost(task_reward, k):
    """What a selection's size costs it.

    Upper term: K_PENALTY per case beyond K_MAX (the first K_MAX are free). Lower term:
    K_MIN_PENALTY per case short of K_MIN. Each term is charged only on a solved rollout unless
    MEMSEL_K_PEN_ALWAYS / MEMSEL_K_MIN_ALWAYS is set; when solved-only, picking fewer and still
    failing earns nothing, so the penalty never pushes toward the empty (no-memory) selection.
    Both terms default to 0 and are independent.
    """
    hi = K_PENALTY * max(0, k - K_MAX) if (K_PEN_ALWAYS or task_reward >= 1) else 0.0
    lo = K_MIN_PENALTY * max(0, K_MIN - k) if (K_MIN_ALWAYS or task_reward >= 1) else 0.0
    return hi + lo


# Charge the missing-<reasoning> penalty on EVERY rollout, not only on solved ones. If it were
# gated on success, dropping <reasoning> would be a certain saving on every rollout but penalised
# only on the few that solve the task, making "never reason" strictly cheaper; the policy then
# quickly stops reasoning and within-group diversity collapses.
# MEMSEL_FMT_PEN_ALWAYS=0 restores the solved-only rule.
FMT_PEN_ALWAYS = os.environ.get('MEMSEL_FMT_PEN_ALWAYS', '1') not in ('0', 'false', 'False')


def _fmt_cost(task_reward, fmt_pen):
    """Cost of a missing <reasoning> block on a reply that did give a legal <answer>."""
    if FMT_PEN_ALWAYS:
        return fmt_pen
    return fmt_pen if task_reward >= 1 else 0.0


def _score(task_reward, fmt_pen, k):
    return task_reward - _fmt_cost(task_reward, fmt_pen) - _k_cost(task_reward, k)


def _as_list(x):
    """extra_info survives parquet -> datasets -> numpy, so lists arrive as arrays."""
    if x is None:
        return []
    return [str(v) for v in (x.tolist() if hasattr(x, 'tolist') else list(x))]


def _sim_key(url, task_id, opening, folders, seed):
    """Everything that determines a simulation's inputs. Order matters: render() keeps the
    given order inside each WENT WRONG / WENT RIGHT group, so a reordering is a different
    <experience> block and a genuinely different action."""
    h = hashlib.sha1(opening.strip().encode()).hexdigest()[:12]
    return f"{url}|{task_id}|{h}|{seed}|{','.join(folders)}"


async def _sim_dedup(session, url, task_id, opening, folders, seed, scope='', dedup=True, max_steps=0):
    """One simulation per distinct key, however many rollouts ask for it concurrently.

    Returns (result, error, was_deduped). The winner runs the simulation while holding the
    concurrency semaphore; the followers wait on its future and never occupy a slot, so
    duplicates cost neither an API call nor concurrency budget.
    """
    if not DEDUP or not dedup:
        async with _sem:
            r, e = await _sim(session, url, task_id, opening, folders, seed, scope, dedup, max_steps)
        return r, e, False

    key = scope + '|' + _sim_key(url, task_id, opening, folders, seed)
    fut = _inflight.get(key)
    if fut is not None:
        r, e = await asyncio.shield(fut)
        return r, e, True

    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    _inflight[key] = fut
    try:
        async with _sem:
            r, e = await _sim(session, url, task_id, opening, folders, seed, scope, dedup, max_steps)
        if not fut.done():
            fut.set_result((r, e))
        return r, e, False
    except Exception as exc:
        if not fut.done():
            fut.set_result((None, f'{type(exc).__name__}: {exc}'))
        raise
    finally:
        _inflight.pop(key, None)


async def _sim(session, url, task_id, opening, folders, seed, scope='', dedup=True, max_steps=0):
    """One simulation via the tau2 service. Returns (result, error)."""
    payload = {'task_id': str(task_id), 'opening': opening,
               'folders': list(folders), 'seed': int(seed), 'scope': scope, 'dedup': bool(dedup)}
    if max_steps:
        payload['max_steps'] = int(max_steps)        # training only; eval omits it -> service default
    last = None
    for attempt in range(SIM_RETRIES + 1):
        try:
            async with session.post(f'{url}/run', json=payload) as resp:
                r = await resp.json()
            if r.get('error') or r.get('reward') is None:
                last = r.get('error') or 'service returned no reward'
            else:
                return r, None
        except Exception as e:
            last = f'{type(e).__name__}: {e}'
        await asyncio.sleep(2 * (attempt + 1))
    return None, last


async def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """One rollout -> one reward. Returns a dict so verl logs the parts separately."""
    sem, session, cache, log_lock = _lazy()
    t0 = time.time()
    info = dict(extra_info or {})
    task_id = str(info.get('task_id', ground_truth))
    opening = info.get('opening') or ''
    all_folders = _as_list(info.get('folders'))
    k = len(all_folders)

    picked, status = read_answer(solution_str, valid=range(1, k + 1))
    thinking = bool(info.get('thinking', 1))
    fmt = check_format(solution_str, thinking)
    common = dict(task_id=task_id, window=int(info.get('window', -1)), k_pool=k,
                  n_selected=len(picked), format_clean=fmt['clean'],
                  has_reasoning=fmt['has_reasoning'], has_answer=fmt['has_answer'])
    fmt_pen = PEN_NO_REASONING * (not fmt['clean'])

    if status != 'ok':
        out = _result(**common, parse_failed=1, sim_error=status,
                      secs=round(time.time() - t0, 1))
        await _write_log(log_lock, out, picked, solution_str, data_source=data_source,
                         pool=all_folders)
        return out

    folders = [all_folders[i - 1] for i in picked]

    # An eval row is one the dataset built with n_trials > 1, or one that sets is_eval explicitly
    # (ALFWorld evaluates a single attempt per game, so n_trials cannot mark it). Training rows
    # take TRAIN_TRIALS draws, so the trial count alone does not identify an eval row.
    row_trials = max(1, int(info.get('n_trials', 1)))
    is_eval = bool(int(info.get('is_eval', 0) or 0)) or row_trials > 1
    n_trials = row_trials if is_eval else TRAIN_TRIALS
    common['is_eval'] = int(is_eval)
    # the on-disk cache stores one reward per (task, opening, folders); a multi-trial row wants
    # fresh simulations, so it must not be short-circuited by a single cached value
    if cache is not None and not is_eval and n_trials == 1:
        hit = cache.get(task_id, opening, folders)
        if hit:
            task_reward = sum(hit) / len(hit)
            out = _result(**common, cache_hit=1,
                          score=_score(task_reward, fmt_pen, len(picked)),
                          k_cost=_k_cost(task_reward, len(picked)),
                          task_reward=task_reward, secs=round(time.time() - t0, 1))
            await _write_log(log_lock, out, picked, solution_str, data_source=data_source,
                             pool=all_folders)
            return out

    # Both kinds use tau2's per-trial seeds: n_trials simulations of the SAME opening and SAME
    # selection, averaged. The selector ran once, so trials vary only in the simulation.
    #
    # Training dedups identical actions within one GRPO group of one step, so two rollouts that
    # chose the same memories get the SAME averaged reward; otherwise tau2's noise would read as a
    # preference. Eval never dedups, so every eval is a fresh measurement.
    seeds = _trial_seeds(n_trials)
    if is_eval:
        dedup, scope = False, ''
    else:
        dedup, scope = True, _step_scope()

    # return_exceptions: one trial blowing up must not lose the others. _sim already turns
    # transport and service errors into (None, msg); this only catches the unexpected.
    runs = await asyncio.gather(*[_sim_dedup(session, _url_for(data_source, task_id, opening, folders, sd), task_id, opening, folders, sd, scope, dedup, 0 if is_eval else TRAIN_MAX_STEPS)
                                  for sd in seeds], return_exceptions=True)
    errs = [f'{type(x).__name__}: {x}' if isinstance(x, BaseException) else x[1]
            for x in runs if isinstance(x, BaseException) or x[0] is None]
    ok = [(r, d) for x in runs if not isinstance(x, BaseException)
          for r, e, d in [x] if r is not None]
    if not ok:
        out = _result(**common, n_trials=n_trials, sim_error=str(errs[0] if errs else 'all trials failed')[:300],
                      secs=round(time.time() - t0, 1))
        await _write_log(log_lock, out, picked, solution_str, data_source=data_source,
                         pool=all_folders)
        return out

    # Like tau2's own metrics, infrastructure errors are dropped rather than scored 0: a dead
    # trial shrinks the denominator, and trials_ok records what that denominator really was.
    rewards = [float(r['reward']) for r, _ in ok]
    binary_reward = sum(rewards) / len(rewards)
    sim_rows = [r for r, _ in ok]
    partial = partial_credit(sim_rows)
    # Eval rows always keep tau2's own number, whatever PARTIAL says, so they stay comparable.
    task_reward = partial if (PARTIAL and not is_eval) else binary_reward
    bd0 = sim_rows[0].get('reward_breakdown') or {}
    def _bd(name):
        for key in (name, f'RewardType.{name}'):
            if key in bd0:
                return float(bd0[key])
        return -1.0
    sim = ok[0][0]
    # a service-side hit ('inflight'/'result') costs no simulation even though this process's
    # own in-flight table missed it, so count both as dedup hits
    hits = sum(1 for r, d in ok if d or r.get('dedup') in ('inflight', 'result'))
    if cache is not None and not is_eval and n_trials == 1:
        cache.put(task_id, opening, folders, task_reward, seed=seeds[0])
    if is_eval and EVAL_RAW_SCORE:
        score, k_cost = task_reward, 0.0           # tau2 pass^1: raw mean success
    else:
        score = _score(task_reward, fmt_pen, len(picked))
        k_cost = _k_cost(task_reward, len(picked))
    out = _result(**common, simulated=len(ok) - hits, dedup_hit=hits,
                  n_trials=n_trials, trials_ok=len(ok),
                  trials_passed=sum(1 for x in rewards if x >= 1),
                  partial_reward=partial, db_reward=_bd('DB'), comm_reward=_bd('COMMUNICATE'),
                  action_match=sum(int(r.get('action_match') or 0) for r in sim_rows),
                  action_total=sum(int(r.get('action_total') or 0) for r in sim_rows),
                  sim_error='; '.join(str(e) for e in errs)[:300],
                  score=score, k_cost=k_cost,
                  task_reward=task_reward, termination=str(sim['termination']),
                  n_messages=int(sim['n_messages']), secs=round(time.time() - t0, 1))
    await _write_log(log_lock, out, picked, solution_str, scope, data_source,
                     pool=all_folders, folders=folders)
    return out


async def _write_log(lock, out, selection, reply, scope='', data_source=None, pool=None, folders=None):
    """Full record: the fixed-shape result plus the things too ragged to return to verl.

    `pool` is the candidate memory folders in the order the prompt showed them (`selection` is
    1-based into it) and `folders` is what the simulation actually received, so each row is
    self-contained for offline analysis. With `scope` (one value per training step) this file is
    the PRE-DAPO-FILTER record: it includes the zero-variance groups DAPO discards and never writes
    to rollouts/<step>.jsonl, so per-task analysis should read this file, not rollouts/.

    data_source is logged because task ids overlap between domains, so a mixed-domain run cannot
    be split by id alone. It stays out of _FIELDS because verl wants scalar per-sample fields.
    """
    try:
        async with lock:
            with open(LOG_PATH, 'a') as f:
                f.write(json.dumps({**out, 'selection': list(selection), 'scope': scope,
                                    'data_source': data_source,
                                    'pool': list(pool or []), 'folders': list(folders or []),
                                    'reply_tail': (reply or '')[-300:]}) + '\n')
    except Exception:
        traceback.print_exc()
