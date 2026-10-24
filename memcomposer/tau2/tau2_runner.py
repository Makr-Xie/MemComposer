"""Run one tau2 simulation for a given memory selection, on the domain MEMSEL_DOMAIN names.

Used by evaluation and, through tau2_service.py, by the RL reward, so both run the identical
pipeline.

    from tau2_runner import run_one
    r = run_one(task_id='15', opening='Hello, I need to...', folders=['task12_trial6'], seed=0)
    r['reward']   # 0.0 or 1.0

`folders` are memory-bank folder names in the order the selector chose; render() keeps that
order, so order is meaningful. An empty list gives experience='' -- the stock tau2 prompt.

The customer's opening is forced through `task.initial_state.message_history`, so the utterance
retrieval used is the one actually spoken. The agent's first message is tau2's constant
DEFAULT_FIRST_AGENT_MESSAGE, so the memory block cannot influence it.

Main env vars (all read once at import): MEMSEL_DOMAIN, MEMSEL_MEMORY_BASE, TAU2_ROOT,
MEMSEL_AGENT_LLM / _BASES / _API_KEY / _TEMP / _TOP_P / _MAX_TOKENS / _CHAT_TEMPLATE_KWARGS,
BEDROCK_BASE_URL, MEMSEL_USER_MAX_TOKENS, MEMSEL_MAX_STEPS, MEMSEL_AGENT_CONC_PER_CARD,
MEMSEL_USER_CONC, MEMSEL_SIM_DUMP, MEMSEL_LLM_TIMING. Each is described where it is read.
"""
import collections, copy, itertools, json, os, re, sys, threading, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
# One process serves one domain; tau2_service sets MEMSEL_DOMAIN before importing this module.
# MEMSEL_MEMORY_BASE is the memory-bank root (one folder per domain); default is the Qwen3-4B bank.
DOMAIN = os.environ.get('MEMSEL_DOMAIN', 'airline')
BASE = Path(os.environ.get('MEMSEL_MEMORY_BASE',
                    Path(os.environ.get('MEMCOMPOSER_DATA', Path(__file__).resolve().parents[2] / 'data'))
                    / 'tau2' / 'memory_banks' / 'qwen')) / DOMAIN

# TAU2_ROOT: a tau2-bench checkout; its .env (the user simulator's credentials) is loaded if present.
if os.environ.get('TAU2_ROOT') and (Path(os.environ['TAU2_ROOT']) / '.env').exists():
    for _k, _v in re.findall(r"^(\w+)=['\"]?([^'\"\n]+)",
                             (Path(os.environ['TAU2_ROOT']) / '.env').read_text(), re.M):
        os.environ.setdefault(_k, _v)
sys.path.insert(0, str(HERE))

from render import render                                              # noqa: E402
from memory_agent import MemoryLLMAgent                                # noqa: E402
from tau2.data_model.message import UserMessage                        # noqa: E402
from tau2.data_model.tasks import InitialState                         # noqa: E402
from tau2.orchestrator.orchestrator import (                           # noqa: E402
    DEFAULT_FIRST_AGENT_MESSAGE, Orchestrator)
from tau2.runner.build import build_environment, build_user            # noqa: E402
from tau2.runner.helpers import get_tasks                              # noqa: E402
from tau2.runner.simulation import run_simulation                      # noqa: E402

# Connection pooling. tau2's llm_utils pins litellm to one process-global httpx client, so agent
# calls and user-simulator calls share one pool. The pool caps concurrent HTTP calls, not
# simulations (a simulation holds a slot only while it is sending a turn). Below, every known host
# gets its own pool, injected per call through litellm's `client=`: agent and user-simulator calls
# use separate pools because a shared pool made them queue behind each other. Agent calls are
# capped per server because too many conversations resident on one vLLM server thrash its prefix
# cache and every turn's prefill is redone. The global pool (MEMSEL_HTTPX_MAXCONN, default 16;
# MEMSEL_HTTPX_KEEPALIVE, default 8) only serves api_bases that match neither role.
# This is patched here rather than in the tau2-bench checkout so that checkout stays clean.
import httpx, litellm                                                  # noqa: E402
import tau2.utils.llm_utils                                            # noqa: E402,F401
# Max concurrent calls to EACH agent server (default 6). Higher values thrash the prefix cache.
_AGENT_CONC = int(os.environ.get('MEMSEL_AGENT_CONC_PER_CARD', '6'))
# Max concurrent user-simulator calls from THIS process (default 64). A client-side limit only; it
# is per process, so separate tau2_service processes each get their own. Raise it when the agent
# servers sit idle waiting on the user simulator.
_USER_CONC = int(os.environ.get('MEMSEL_USER_CONC', '64'))
_HTTPX_LIMITS = httpx.Limits(max_keepalive_connections=int(os.environ.get('MEMSEL_HTTPX_KEEPALIVE', '8')),
                             max_connections=int(os.environ.get('MEMSEL_HTTPX_MAXCONN', '16')))
litellm.client_session = httpx.Client(limits=_HTTPX_LIMITS)
litellm.aclient_session = httpx.AsyncClient(limits=_HTTPX_LIMITS)
import openai                                                          # noqa: E402
_CLIENTS = {}
def _client_for(api_base, api_key):
    """One pooled client per (base, api_key); None for an unknown base (global pool).

    Keyed by the pair, not the base alone, because a remote agent model can share its host with
    the user simulator (MEMSEL_AGENT_BASES equal to USER_ARGS['api_base']); a base-only cache
    would give both roles one pool and their calls would queue behind each other. The agent
    always passes an explicit api_key (MEMSEL_AGENT_API_KEY) and the user never does, so the pair
    separates them.
    """
    base = (api_base or '').rstrip('/')
    if not base: return None
    ck = (base, api_key or '')
    c = _CLIENTS.get(ck)
    if c is None:
        is_agent = base in {b.rstrip('/') for b in AGENT_BASES} and bool(api_key)
        if is_agent:
            lim = httpx.Limits(max_connections=_AGENT_CONC, max_keepalive_connections=_AGENT_CONC)
        elif base == USER_ARGS['api_base'].rstrip('/'):
            lim = httpx.Limits(max_connections=_USER_CONC, max_keepalive_connections=_USER_CONC)
        elif base in {b.rstrip('/') for b in AGENT_BASES}:
            lim = httpx.Limits(max_connections=_AGENT_CONC, max_keepalive_connections=_AGENT_CONC)
        else:
            return None
        c = _CLIENTS[ck] = openai.OpenAI(base_url=base, api_key=api_key or os.environ.get('OPENAI_API_KEY') or 'local',
                                         http_client=httpx.Client(limits=lim, timeout=httpx.Timeout(600.0)))
    return c
_completion_nopool = litellm.completion
def _completion_split_pools(*a, **kw):
    if 'client' not in kw:
        c = _client_for(kw.get('api_base'), kw.get('api_key'))
        if c is not None: kw['client'] = c
    return _completion_nopool(*a, **kw)
litellm.completion = _completion_split_pools
# tau2's llm_utils did `from litellm import completion` at import, so the name it calls is a
# module-level binding that `litellm.completion = ...` never touches. Rebind it too, or the
# split pools silently never engage.
tau2.utils.llm_utils.completion = _completion_split_pools

# Turn cap for one simulation (tau2's default is 200). Normal conversations end far below 100, so
# the cap only stops runs that have already gone wrong (e.g. an agent looping on a tool). If
# legitimate conversations hit it, termination reads MAX_STEPS instead of USER_STOP; raise it.
MAX_STEPS = int(os.environ.get('MEMSEL_MAX_STEPS', '100'))

# The agent model, as the vLLM servers advertise it (--served-model-name). To swap the agent the
# servers must be started with the matching --served-model-name AND --tool-call-parser (hermes for
# Qwen, llama3_json for Llama 3.1). A mismatched parser silently yields zero tool calls and every
# task fails.
AGENT_LLM = os.environ.get('MEMSEL_AGENT_LLM', 'openai/Qwen3-4B-Instruct-2507')

# MEMSEL_SIM_DUMP=<path>: append the full simulation (messages, reward_info, task, timings) to a
# jsonl, one line per run. run_one otherwise returns only summary fields; these transcripts are
# what memory banks are built from. Off by default.
SIM_DUMP = os.environ.get('MEMSEL_SIM_DUMP', '')
_DUMP_LOCK = threading.Lock()

# The agent settings below are read once at import, so a service must be STARTED with them set.
# Agent sampling temperature. 0 for training and eval; 1.0 matches how the memory banks' source
# rollouts were sampled.
AGENT_TEMP = float(os.environ.get('MEMSEL_AGENT_TEMP', '0'))
# Agent top_p. Unset -> not sent, so the server default applies. Set it to follow a model's own
# recommended sampling pair.
AGENT_TOP_P = os.environ.get('MEMSEL_AGENT_TOP_P')
AGENT_TOP_P = float(AGENT_TOP_P) if AGENT_TOP_P not in (None, '') else None
# Extra kwargs for the model's chat template, as JSON; unset -> nothing is sent. Needed for Qwen3
# hybrid-reasoning models, which emit <think>...</think> by default: {"enable_thinking": false} is
# the only way to turn that off through the OpenAI API.
_CTK = os.environ.get('MEMSEL_AGENT_CHAT_TEMPLATE_KWARGS')
AGENT_CHAT_TEMPLATE_KWARGS = json.loads(_CTK) if _CTK else None
# Agent API key. 'local' suits a local vLLM server; a hosted endpoint needs a real key.
AGENT_API_KEY = os.environ.get('MEMSEL_AGENT_API_KEY', 'local')
# Agent reply cap (default 4096). A hosted endpoint such as bedrock-mantle reserves
# `input_tokens + max_tokens` against its input tokens-per-minute quota at admission, so lower this
# for a remote agent; a tau2 agent turn is a short tool call.
AGENT_MAX_TOKENS = int(os.environ.get('MEMSEL_AGENT_MAX_TOKENS', '4096'))
# Comma-separated agent server URLs, all serving the same model.
AGENT_BASES = [b for b in os.environ.get(
    'MEMSEL_AGENT_BASES', 'http://127.0.0.1:8020/v1,http://127.0.0.1:8022/v1').split(',') if b]
USER_LLM = 'openai/zai.glm-4.7'
# User-simulator reply cap (MEMSEL_USER_MAX_TOKENS, default 512). The endpoint reserves
# `input_tokens + max_tokens` against its input tokens-per-minute quota at admission and has no
# request-rate quota, so a small reservation is what leaves room for concurrency. User turns are
# far shorter than 512 tokens.
# https://docs.aws.amazon.com/bedrock/latest/userguide/quotas-mantle.html
# The user simulator (and the NL-assertion judge below) is reached through BEDROCK_BASE_URL, an
# OpenAI-compatible endpoint; it is also read from TAU2_ROOT's .env.
USER_ARGS = {'api_base': os.environ.get('BEDROCK_BASE_URL', 'https://bedrock-mantle.us-east-1.api.aws/v1'),
             'temperature': 0, 'max_tokens': int(os.environ.get('MEMSEL_USER_MAX_TOKENS', '512'))}

# tau2 hardcodes the NL-assertion judge to gpt-4.1 on api.openai.com. Here OPENAI_API_KEY holds
# the user-simulator endpoint's key, so that call would fail during grading and the finished
# conversation would be lost. Only retail's reward_basis uses NL_ASSERTION. The judge is rebound
# on evaluator_nl_assertions itself because that module did `from tau2.config import ...` at
# import time, so patching tau2.config would come too late.
NL_JUDGE = 'openai/zai.glm-4.7'
NL_JUDGE_ARGS = {'api_base': USER_ARGS['api_base'], 'temperature': 0, 'max_tokens': 4096}
_FENCE = re.compile(r"^\s*```(?:json)?\s*\n(.*?)\n?\s*```\s*$", re.S)


def _patch_nl_judge():
    import tau2.evaluator.evaluator_nl_assertions as ev
    ev.DEFAULT_LLM_NL_ASSERTIONS = NL_JUDGE
    ev.DEFAULT_LLM_NL_ASSERTIONS_ARGS = NL_JUDGE_ARGS
    inner = ev.generate

    def generate(*a, **kw):
        # The judge's reply is parsed with a bare json.loads; GLM wraps it in a code fence.
        msg = inner(*a, **kw)
        content = getattr(msg, 'content', None)
        if isinstance(content, str):
            m = _FENCE.match(content)
            if m:
                try:
                    msg.content = m.group(1)
                except Exception:               # frozen model - hand back a copy instead
                    msg = msg.model_copy(update={'content': m.group(1)})
        return msg

    ev.generate = generate


_patch_nl_judge()

_TASKS = None
_META = None
# All agent servers host the same frozen model, so choosing one is pure load balancing. Each
# simulation goes to the base with the fewest simulations outstanding, ties broken round-robin.
# Plain round-robin balances arrivals, not load: simulation length is heavy-tailed, so one
# server's queue can keep growing while another sits idle.
_RR = itertools.count()
_OUTSTANDING = collections.Counter()
_OUT_LOCK = threading.Lock()


def _pick_base():
    with _OUT_LOCK:
        base = min(AGENT_BASES, key=lambda b: (_OUTSTANDING[b], next(_RR)))
        _OUTSTANDING[base] += 1
    return base


def _release_base(base):
    with _OUT_LOCK:
        _OUTSTANDING[base] -= 1


def _tasks():
    # Build fully, then publish: concurrent first callers must never see a half-filled dict.
    global _TASKS
    if _TASKS is None:
        _TASKS = {str(t.id): t for t in get_tasks(DOMAIN)}
    return _TASKS


def _meta():
    """folder -> {sim_id, label_kind}; render() needs both."""
    # Build the whole map in a local, then assign once, so a concurrent first caller never sees a
    # partial dict (a KeyError on a folder that exists). Same pattern as _tasks above.
    global _META
    if _META is None:
        meta = {}
        for f in sorted(BASE.glob('task*/memory.json')):
            m = json.loads(f.read_text())
            meta[m['folder']] = {'sim_id': m['sim_id'], 'label_kind': m['label_kind']}
        _META = meta
    return _META


def experience_from(folders):
    """<experience> block from full.txt, or '' for an empty selection."""
    if not folders:
        return ''
    meta = _meta()
    return render([{'sim_id': meta[f]['sim_id'], 'label_kind': meta[f]['label_kind']}
                   for f in folders])


def run_one(task_id, opening, folders, seed=0, agent_base=None, max_steps=MAX_STEPS):
    """One simulation. Returns reward plus enough context to audit it later."""
    task = copy.deepcopy(_tasks()[str(task_id)])
    if task.initial_state is None:
        task.initial_state = InitialState(initialization_data=None,
                                          initialization_actions=None, message_history=None)
    task.initial_state.message_history = [
        copy.deepcopy(DEFAULT_FIRST_AGENT_MESSAGE),
        UserMessage(role='user', content=opening),
    ]
    env = build_environment(DOMAIN)
    base = agent_base or _pick_base()
    owned = agent_base is None
    agent = MemoryLLMAgent(
        tools=env.get_tools(), domain_policy=env.get_policy(), llm=AGENT_LLM,
        llm_args={'api_base': base, 'api_key': AGENT_API_KEY, 'temperature': AGENT_TEMP,
                  'max_tokens': AGENT_MAX_TOKENS,
                  **({'top_p': AGENT_TOP_P} if AGENT_TOP_P is not None else {}),
                  **({'extra_body': {'chat_template_kwargs': AGENT_CHAT_TEMPLATE_KWARGS}}
                     if AGENT_CHAT_TEMPLATE_KWARGS is not None else {})},
        experience=experience_from(folders))
    user = build_user('user_simulator', env, task, llm=USER_LLM, llm_args=USER_ARGS)
    orch = Orchestrator(domain=DOMAIN, agent=agent, user=user, environment=env, task=task,
                        max_steps=max_steps, max_errors=10, seed=seed)
    try:
        sim = run_simulation(orch)
    finally:
        if owned:
            _release_base(base)
    ri = sim.reward_info
    # tau2 multiplies the components listed in a task's reward_basis, so one failed component
    # zeroes the reward. The per-component breakdown and action_checks (computed for every task,
    # even when ACTION is not in its basis) are returned as well, so a graded reward can be built
    # from them without extra simulations (partial_credit in memcomposer/reward.py).
    ac = ri.action_checks or []
    if SIM_DUMP:
        try:
            rec = sim.model_dump(mode='json')
            rec['_meta'] = {'domain': DOMAIN, 'task_id': str(task_id), 'seed': seed,
                            'folders': list(folders), 'opening': opening,
                            'agent_llm': AGENT_LLM, 'agent_temperature': AGENT_TEMP,
                            'agent_top_p': AGENT_TOP_P,
                            'agent_chat_template_kwargs': AGENT_CHAT_TEMPLATE_KWARGS,
                            'agent_max_tokens': AGENT_MAX_TOKENS,
                            'user_llm': USER_LLM, 'max_steps': max_steps,
                            'task': task.model_dump(mode='json') if hasattr(task, 'model_dump') else None}
            with _DUMP_LOCK:
                with open(SIM_DUMP, 'a') as _f:
                    _f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
        except Exception:
            traceback.print_exc()
    return {'task_id': str(task_id), 'folders': list(folders), 'seed': seed,
            'reward_breakdown': {str(k): float(v) for k, v in (ri.reward_breakdown or {}).items()},
            'reward_basis': [str(x) for x in (ri.reward_basis or [])],
            'action_match': sum(1 for a in ac if a.action_match),
            'action_total': len(ac),
            'reward': float(sim.reward_info.reward),
            'termination': str(sim.termination_reason),
            'n_messages': len(sim.messages), 'agent_base': base}


# ---- per-call timing hook (diagnostic; active only when MEMSEL_LLM_TIMING names a file) ----
# Splits every LLM call into: pool wait (httpx connection checkout) / wire (on the socket) /
# python-other, keyed by thread so calls can be attributed to the simulation that made them.
if os.environ.get('MEMSEL_LLM_TIMING'):
    import threading as _th, time as _tm, json as _js
    _TL = _th.local(); _TLOCK = _th.Lock(); _TF = open(os.environ['MEMSEL_LLM_TIMING'], 'a')
    def _tlog(d):
        with _TLOCK: _TF.write(_js.dumps(d) + '\n'); _TF.flush()
    from httpcore._sync.connection_pool import ConnectionPool as _CP
    from httpcore._sync.connection import HTTPConnection as _HC
    _cp_hr, _hc_hr = _CP.handle_request, _HC.handle_request
    def _cp_wrapped(self, request):
        t0 = _tm.time()
        try: return _cp_hr(self, request)
        finally: _TL.pool = getattr(_TL, 'pool', 0.0) + (_tm.time() - t0)
    def _hc_wrapped(self, request):
        t0 = _tm.time()
        try: return _hc_hr(self, request)
        finally: _TL.wire = getattr(_TL, 'wire', 0.0) + (_tm.time() - t0)
    _CP.handle_request, _HC.handle_request = _cp_wrapped, _hc_wrapped
    _orig_completion = litellm.completion
    def _timed_completion(*a, **kw):
        _TL.pool = 0.0; _TL.wire = 0.0; t0 = _tm.time()
        try: return _orig_completion(*a, **kw)
        finally:
            _tlog({'k': 'call', 'tid': _th.get_ident(), 'model': str(kw.get('model', a[0] if a else '?')),
                   'max_tokens': kw.get('max_tokens'), 't0': t0, 'total': _tm.time() - t0,
                   'pool': _TL.pool, 'wire': _TL.wire})
    litellm.completion = _timed_completion
    for _modname in ('tau2.utils.llm_utils',):
        _m = sys.modules.get(_modname)
        if _m is not None and getattr(_m, 'completion', None) is not None: _m.completion = _timed_completion
    _orig_run_one = run_one
    def run_one(*a, **kw):                                              # noqa: F811
        t0 = _tm.time()
        try: return _orig_run_one(*a, **kw)
        finally: _tlog({'k': 'sim', 'tid': _th.get_ident(), 'task_id': str(a[0] if a else kw.get('task_id')),
                        't0': t0, 'total': _tm.time() - t0})
    print(f'[timing hook] active -> {os.environ["MEMSEL_LLM_TIMING"]}', flush=True)
