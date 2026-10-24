#!/usr/bin/env python3
"""Post-hoc failure attribution for tau2-bench agent trajectories.

Given a FAILED trajectory plus the full answer key (reference actions, the
graded assertions, and the grader's verdict), ask a model to locate the turn
where the agent's own behaviour caused that failure and say what it should
have done instead.

This is deliberately NOT a blind self-diagnosis: the critic is told exactly
what the grader concluded, so its only job is attribution and localization,
and every failed trajectory gets a root cause (unlike `tau2 review`, which
does not see reward_info). A mechanical audit then marks each critique
usable or not.

Usage:
    python self_critique.py <results.json> -o <out.json>
        [--model hosted_vllm/qwen3-4b] [--temperature 0.0] [--concurrency 8]
        [--limit N] [--include-passed] [--siblings <all_rollouts.json>]

    --include-passed   also critique trajectories that scored 1.0
    --siblings         show a passing run of the same task as a correct path

Input is a tau2 results.json (see sims_to_tau2.py). Writes a copy of it with
a `self_critique` object added to each critiqued simulation; `usable` and
`unusable_reasons` in that object are the audit's verdict.
"""
import argparse, json, os, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# TAU2_ROOT: a tau2-bench checkout, if `tau2` is not installed; its .env is loaded as tau2 does.
if os.environ.get("TAU2_ROOT"):
    sys.path.insert(0, str(Path(os.environ["TAU2_ROOT"]) / "src"))
    from dotenv import load_dotenv
    load_dotenv(Path(os.environ["TAU2_ROOT"]) / ".env")
import litellm
from tau2.utils.display import MarkdownDisplay
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task
from tau2.registry import registry


def domain_context(dom):
    """The same material the agent itself had: its full policy (for telecom this
    includes the tech-support manual) and the split between the tools the agent
    owns and the tools that live on the customer's phone."""
    env = registry.get_env_constructor(dom)()
    try:
        user_tools = [(t.name, (t.short_desc or "").split("\n")[0][:90]) for t in env.get_user_tools()]
    except Exception:
        user_tools = []
    agent_tools = [(t.name, (t.short_desc or "").split("\n")[0][:90]) for t in env.get_tools()]
    return env.get_policy(), agent_tools, user_tools, registry.get_env_constructor(dom)


def tool_index(messages):
    """turn -> what the agent actually did, so the critic localizes instead of guessing."""
    rows = []
    for i, m in enumerate(messages):
        if m["role"] == "assistant" and m.get("tool_calls"):
            for t in m["tool_calls"]:
                args = t.get("arguments")
                if isinstance(args, str):
                    try: args = json.loads(args)
                    except Exception: pass
                rows.append(f"  [TURN {i}] called {t['name']}({json.dumps(args, ensure_ascii=False)[:110]})")
        elif m["role"] == "tool" and m.get("requestor") == "assistant":
            c = (m.get("content") or "").replace("\n", " ")
            if str(m.get("error")) == "True" or c.startswith("Error"):
                rows.append(f"  [TURN {i}] ^ that call FAILED: {c[:90]}")
        elif m["role"] == "user" and m.get("tool_calls"):
            rows.append(f"  [TURN {i}] (the CUSTOMER used their phone: {', '.join(t['name'] for t in m['tool_calls'])})")
    return "\n".join(rows) or "  (the agent never called a tool)"

SYSTEM = """
You are analyzing a customer-service agent's own transcript to find out why it failed the task.

You are given:
- <Policy>: the rules the agent had to follow.
- <Task>: what the customer wanted.
- <Answer Key>: the reference solution and the exact conditions the grader checked.
- <Grader Verdict>: what the grader concluded — this trajectory FAILED, and these are the specific checks that did not pass.
- <Conversation>: the full transcript, every line prefixed with its turn number as [TURN n].

The verdict is ground truth. Do not re-litigate whether the trajectory failed.
Your job is attribution: find the moment in THIS transcript that caused THAT verdict.

How to work:
1. Read the verdict and state plainly, in your own words, which condition was not met.
2. Walk the transcript and find the FIRST turn where the agent's own output was wrong — a
   wrong statement to the customer, a wrong or missing tool call, a wrong argument. That is
   the root cause. It is never a lookup that returned correctly; if the agent looked
   something up and then drew the wrong conclusion, the root cause is the turn with the
   wrong conclusion. If a tool errored at [TURN 31], the root cause is the agent's call at
   [TURN 30].
   If the verdict marks a write as [FORBIDDEN WRITE], the agent's error is having made that write;
   "should_have" must then say to refuse or leave it unchanged, and must NOT describe how to make
   the same write correctly. A task can mix both: some writes required, others forbidden.
   Many tasks are REFUSAL tasks: the correct outcome is to decline or do nothing, and the
   reference solution then contains only lookups (the verdict tells you when this is the
   case). In such a task the agent's error is that it wrote to the database at all — do
   not tell it to perform writes it should have refused.
3. Say concretely what the agent should have done at that turn instead, grounded in the
   policy and the answer key.
4. List any further agent mistakes that contributed, each with its own turn.

Rules:
- Only analyze the AGENT. The customer's behaviour is not your concern; never blame the customer for an agent mistake.
- Every turn_idx you cite MUST be a [TURN n] number that appears in the transcript, and it
  MUST be an **agent** turn — a turn labelled **assistant**. Never cite a **tool** turn (that
  is the system answering) or a **user** turn (that is the customer). If the mistake shows
  up in a tool error at [TURN 31], the agent's turn is [TURN 30]; if the agent answered a
  customer question badly at [TURN 50], cite 50, not the question at 49.
  <Agent Actions By Turn> lists every tool the agent called and at which turn — use it so
  the turn you cite is the turn where the thing you describe actually happened.
- Lines in the transcript marked "**user**" followed by "Tool Call" are the CUSTOMER
  operating their own phone at the agent's request. They are not agent actions. Never write
  "the agent called toggle_roaming" when the customer did it.
- The root cause must be a turn where the agent did something wrong or failed to do
  something. A successful read-only lookup (get_*, search_*, find_*, list_*) is never the
  root cause by itself — if a lookup was fine and the mistake came later, cite the later turn.
- "should_have" must be an action the agent could actually have taken with the information
  in the transcript and the tools it owns. Never write "state the answer-key value" or
  "match the reference solution"; instead say which step would have produced the right
  result (which lookup, which calculation, which tool with which arguments, which question
  to ask the customer). If the correct value can be computed from numbers in the transcript,
  show the computation.
- The reference solution is ONE valid path. Failing to match it exactly is not itself an
  error; only cite it to explain what the failed condition required.
- When you say the agent violated the policy, put the exact policy sentence you rely on in
  "policy_quote". It must be copied from <Policy> — never from <Task>, the answer key, or
  your own words. If the failure was not a policy violation (e.g. a wrong calculation, a
  missed step, a wrong argument), set "policy_quote" to null rather than inventing one.
- Quote or paraphrase what actually happened at that turn. Never invent tool names,
  arguments, or messages that are not in the transcript.
- <Agent Tools> and <Customer Device Tools> are two different sets. A tool in the second
  set exists but belongs to the customer's phone: the agent cannot call it and must ask
  the customer to do it. A call that came back "Tool 'X' not found" is this mistake.
  Never write that such a tool "does not exist".
- Assertion names from the answer key (assert_*) are grading checks, not tools. Do not
  describe the agent as calling them.
- If a <Correct Run Of The Same Task> block is present, it is what this SAME agent did in another
  attempt at this SAME task that PASSED. It is the most concrete example of a correct path. Use it
  to spot which step the failed run skipped, did in the wrong order, or did with different
  arguments. The customer's wording differs between runs, so align by intent, not turn by turn,
  and never describe the correct run's actions as things the failed run did.
- If the failure was NOT the agent's fault (the customer hung up mid-task, the conversation
  hit the step limit while the agent was making progress, the task was impossible), say so
  by setting "blame" accordingly and explain. Do not manufacture an agent error.

Output ONLY this JSON object:
{
  "failed_condition": "<plain-language restatement of what the grader says was not met>",
  "blame": "agent" | "customer" | "environment" | "unclear",
  "root_cause": {
    "turn_idx": <int, an assistant turn>,
    "what_happened": "<what the agent did at this turn>",
    "why_it_caused_the_failure": "<causal link to the failed condition>",
    "should_have": "<the concrete correct action at this turn>",
    "policy_quote": "<exact sentence from <Policy> if a rule was broken, else null>"
  },
  "contributing_errors": [
    {"turn_idx": <int>, "what_happened": "<...>", "should_have": "<...>"}
  ],
  "summary": "<two sentences: the failure in one line, the fix in one line>"
}
""".strip()

USER = """<Policy>
{policy}
</Policy>

<Agent Tools>
{agent_tools}
</Agent Tools>
{customer_tools}
<Task>
{task}
</Task>

<Answer Key>
Reference solution (one valid path; other paths may also be correct):
{actions}
{criteria}
</Answer Key>

{sibling}<Grader Verdict>
This trajectory scored {reward}. The following checks did not pass:
{verdict}
</Grader Verdict>

<Agent Actions By Turn>
{tool_index}
</Agent Actions By Turn>

<Conversation>
{conversation}
</Conversation>"""


def _flatten(o, path="", out=None):
    """Leaf paths of a nested dict; any non-dict value, lists included, is one leaf."""
    if out is None: out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            _flatten(v, f"{path}.{k}" if path else str(k), out)
    else:
        out[path] = o
    return out


def _flatten_pair(a, b, path, out):
    """Flatten two trees in lockstep into {path: (a_value, b_value)}. Equal-length lists of
    dicts are recursed element by element, tagged [id=val] when both share an id-like field,
    else [i]; any other list is one leaf compared as a whole."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            p = f"{path}.{k}" if path else str(k)
            if k in a and k in b:
                _flatten_pair(a[k], b[k], p, out)
            elif k in a:
                out[p] = (a[k], "__MISSING__")
            else:
                out[p] = ("__MISSING__", b[k])
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and all(isinstance(x, dict) for x in a + b):
        for i, (x, y) in enumerate(zip(a, b)):
            idk = next((k for k in x if k.endswith("_id") or k == "id"), None)
            tag = f"[{idk}={x[idk]}]" if idk and x.get(idk) == y.get(idk) else f"[{i}]"
            _flatten_pair(x, y, f"{path}{tag}", out)
    else:
        out[path] = (a, b)


def _get(tree, path):
    """Resolve a flattened path (dict keys, [i] or [id=val] list tags) inside a raw tree."""
    cur = tree
    for tok in re.findall(r"\[[^\]]+\]|[^.\[]+", path):
        if cur is None: return "__MISSING__"
        if tok.startswith("["):
            inner = tok[1:-1]
            if not isinstance(cur, list): return "__MISSING__"
            if "=" in inner:
                k, v = inner.split("=", 1)
                cur = next((x for x in cur if isinstance(x, dict) and str(x.get(k)) == v), "__MISSING__")
            else:
                i = int(inner); cur = cur[i] if i < len(cur) else "__MISSING__"
        else:
            cur = cur.get(tok, "__MISSING__") if isinstance(cur, dict) else "__MISSING__"
        if cur == "__MISSING__": return cur
    return cur


def _j(v): return "(absent)" if v == "__MISSING__" else json.dumps(v, default=str)[:140]


DERIVED = re.compile(r"surroundings\.(line_active|roaming_allowed|mobile_data_usage_exceeded)")


def db_diff(task_obj, messages, env_ctor, limit=25):
    """Replay exactly what the grader replays (gold = reference actions, predicted = the agent's
    trajectory) but instead of comparing hashes, diff the two databases field by field — and
    against the ORIGINAL database, so every difference is classified as one of:
      FORBIDDEN  the target leaves this field unchanged; the agent's write should not have happened
      MISSING    the target changes this field; the agent never did
      WRONG      both changed it, with different values (wrong argument)"""
    init = task_obj.initial_state
    kw = dict(initialization_data=getattr(init, "initialization_data", None),
              initialization_actions=getattr(init, "initialization_actions", None))
    hist = list(getattr(init, "message_history", None) or [])
    base = env_ctor(); base.set_state(message_history=hist, strict=False, **kw)
    gold = env_ctor(); gold.set_state(message_history=hist, strict=False, **kw)
    for a in task_obj.evaluation_criteria.actions or []:
        try: gold.make_tool_call(tool_name=a.name, requestor=a.requestor, **a.arguments)
        except Exception: pass
        # make_tool_call does not run sync_tools, but a live turn does (and `mine` gets it via
        # set_state(message_history=...)). Without this, fields sync_tools derives (in telecom:
        # line_active, roaming_allowed, bill status after a payment) would show up as agent writes.
        try: gold.sync_tools()
        except Exception: pass
    mine = env_ctor(); mine.set_state(message_history=list(messages), strict=False, **kw)
    out = []; counts = {"FORBIDDEN": 0, "MISSING": 0, "WRONG": 0}; dev = 0
    for side, b, g, m in (("agent-side DB", base.tools, gold.tools, mine.tools),
                          ("customer-device DB", base.user_tools, gold.user_tools, mine.user_tools)):
        if g is None or m is None or getattr(g, "db", None) is None:
            continue
        agent_side = side == "agent-side DB"
        bd, gd, md = b.db.model_dump(), g.db.model_dump(), m.db.model_dump()
        pairs = {}; _flatten_pair(gd, md, "", pairs)
        rows = []
        for path, (gv, mv) in sorted(pairs.items()):
            if gv == mv: continue
            # derived by sync_tools from state already reported; nobody can write them directly
            if DERIVED.match(path): continue
            bv = _get(bd, path)
            if bv == gv:
                kind = "FORBIDDEN"
                txt = (f"  [FORBIDDEN WRITE] {path}: the target leaves this UNCHANGED at {_j(bv)}, but the agent changed it to {_j(mv)} — this write should not have been made at all."
                       if agent_side else
                       f"  [DEVICE CHANGED] {path}: the target leaves this at {_j(bv)}, the run ended at {_j(mv)}. The customer operates this field; the agent cannot write it.")
            elif bv == mv:
                kind = "MISSING"
                txt = (f"  [MISSING WRITE] {path}: originally {_j(bv)}, the target changes it to {_j(gv)}, the agent never changed it."
                       if agent_side else
                       f"  [DEVICE NOT FIXED] {path}: originally {_j(bv)}, the target ends at {_j(gv)}, the run ended at {_j(mv)}. Only the customer can change this — the agent's job was to get them to.")
            else:
                kind = "WRONG"
                txt = (f"  [WRONG VALUE] {path}: originally {_j(bv)}, target = {_j(gv)}, agent left = {_j(mv)} — the agent performed this write with the wrong argument."
                       if agent_side else
                       f"  [DEVICE WRONG] {path}: originally {_j(bv)}, target = {_j(gv)}, the run ended at {_j(mv)}. The customer operates this field; the agent cannot write it.")
            if agent_side: counts[kind] += 1
            else: dev += 1
            rows.append(txt)
        if rows:
            out.append(f"  [{side}]"); out += rows[:limit]
            if len(rows) > limit: out.append(f"  … {len(rows) - limit} more differing fields not shown")
    if out:
        head = [f"  Summary: {counts['FORBIDDEN']} forbidden write(s) (target unchanged, agent wrote), "
                f"{counts['MISSING']} missing write(s) (target changed, agent did not), {counts['WRONG']} wrong-value write(s)"
                + (f", plus {dev} customer-device field(s) that ended differently — those are the customer's "
                   f"actions on their own phone, never the agent's writes." if dev else ".")]
        if counts["MISSING"] == 0 and counts["WRONG"] == 0 and counts["FORBIDDEN"] > 0:
            ref_writes = [a for a in (task_obj.evaluation_criteria.actions or [])
                          if a.requestor == "assistant" and not re.match(r"^(get_|search_|find_|list_|calculate$|think$)", a.name)]
            if ref_writes:
                head.append("  => The required write(s) WERE performed; every difference is an EXTRA write the target does not make. "
                            "The fix is to stop after the required writes and refuse the extra ones — not to perform them 'correctly'.")
            else:
                head.append("  => Every difference is a FORBIDDEN write and the reference solution makes no writes at all: the correct "
                            "outcome was to make NO changes. The fix is to refuse/not perform the write — not to perform it "
                            "'correctly' or 'after confirmation'.")
        out = head + out
    return out


def build_verdict(ri, diff_rows=None):
    """Human-readable list of exactly what the grader faulted."""
    out = []
    db = ri.get("db_check") or {}
    if db.get("db_match") is False:
        out.append("- The database end-state does not match the target end-state (the agent's writes were wrong, missing, or extra).")
        if diff_rows:
            out.append("  Field-level difference between the TARGET end-state and what the agent actually left behind, "
                       "classified against the ORIGINAL database (this is exactly what the grader compared). Read it first. "
                       "[FORBIDDEN WRITE] = the agent changed something the target leaves untouched: the correct behaviour was "
                       "NOT to make that write (refuse / explain), never to make it 'after confirmation' or 'with better "
                       "arguments'. [MISSING WRITE] = a write the agent never performed. [WRONG VALUE] = the right write with "
                       "the wrong argument. A record id like lines[line_id=L1002] tells you WHICH record the correct write had to touch.")
            out += diff_rows
        elif diff_rows is not None:
            out.append("  (replaying the trajectory produced no field-level difference; the mismatch may be in a field not shown or in record ordering)")
    for a in ri.get("env_assertions") or []:
        if not a.get("met"):
            e = a.get("env_assertion") or {}
            out.append(f"- Environment assertion NOT satisfied: {e.get('func_name')}({json.dumps(e.get('arguments') or {})}) — the world was never put into this state.")
    for a in ri.get("nl_assertions") or []:
        if not a.get("met"):
            j = (a.get("justification") or "").strip()
            out.append(f"- Required outcome NOT satisfied: \"{a.get('nl_assertion')}\"" + (f" (grader's note: {j})" if j else ""))
    for c in ri.get("communicate_checks") or []:
        if not c.get("met"):
            j = (c.get("justification") or "").strip()
            out.append(f"- The agent never told the customer this required information: \"{c.get('info')}\"" + (f" (grader's note: {j})" if j else ""))
    checks = ri.get("action_checks") or []
    if checks:
        hit = [a["action"]["name"] for a in checks if a.get("action_match")]
        missed = [a["action"]["name"] for a in checks if not a.get("action_match")]
        out.append(f"- Against the reference solution — performed: {', '.join(dict.fromkeys(hit)) or 'none'}; "
                   f"never performed: {', '.join(dict.fromkeys(missed)) or 'none'}. "
                   "(Diagnostic only: another correct path may exist, so ask whether the *effect* was achieved at all.)")
        # the grader already labels each reference action read/write; fall back to a name regex only when it didn't
        def is_write(a):
            tt = a.get("tool_type")
            if tt in ("read", "write", "generic", "think"):
                return tt == "write"
            return not re.match(r"^(get_|search_|find_|list_|calculate$|think$)", a["action"]["name"])
        writes = [a["action"]["name"] for a in checks if a.get("action", {}).get("requestor") == "assistant" and is_write(a)]
        if not writes and db.get("db_match") is False:
            out.append("- NOTE: the reference solution makes NO changes to the database — the correct outcome was to "
                       "look things up and then refuse, explain, or leave things as they were. The database mismatch "
                       "therefore means the agent performed a write it should not have performed.")
    elif db.get("db_match") is False and not ri.get("action_checks"):
        out.append("- NOTE: the reference solution contains NO actions at all — the correct outcome was to refuse or explain "
                   "and change nothing. Every write the agent made was a write it should not have made.")
    return "\n".join(out) or "- (no per-check detail recorded; the trajectory was scored 0)"


def build_criteria(ec, desc=None):
    """Everything the grader could check — including the fields tau2's own reviewer never shows,
    plus the task author's stated purpose (what the task was designed to test)."""
    out = []
    desc = desc or {}
    if desc.get("purpose"):
        out.append(f"\nWhat this task was designed to test (task author's note): {desc['purpose'].strip()}")
    if desc.get("notes"):
        out.append(f"Task author's additional note: {str(desc['notes']).strip()}")
    if ec.get("reward_basis"):
        out.append(f"\nScored on: {' × '.join(ec['reward_basis'])}")
    if ec.get("env_assertions"):
        out.append("\nEnvironment assertions that had to hold at the end:")
        for a in ec["env_assertions"]:
            out.append(f"  - {a.get('func_name')}({json.dumps(a.get('arguments') or {})}) == {a.get('assert_value')}  [checked on the {a.get('env_type')} side]")
    if ec.get("nl_assertions"):
        out.append("\nStatements the agent had to satisfy:")
        out += [f"  - {x}" for x in ec["nl_assertions"]]
    if ec.get("communicate_info"):
        out.append("\nInformation the agent had to tell the customer:")
        out += [f"  - {x}" for x in ec["communicate_info"]]
    return "\n".join(out)


def load_siblings(path):
    """task_id -> passing sims of the same task (the same agent, same customer model, other trials)."""
    d = json.load(open(path)); out = {}
    for s in d["simulations"]:
        if ((s.get("reward_info") or {}).get("reward") or 0) >= 1:
            out.setdefault(s["task_id"], []).append(s)
    return out


def sibling_block(sibs, exclude_id, budget_chars=7000):
    """Compact agent-side trace of the shortest passing sibling: what the agent said (truncated) and
    every tool it called with full arguments, plus the customer's lines heavily truncated for
    alignment. Tool results are omitted."""
    cands = [x for x in sibs if x.get("id") != exclude_id]
    if not cands:
        return "", None
    sib = min(cands, key=lambda x: len(x["messages"]))
    rows = []
    for m in sib["messages"]:
        if m["role"] == "assistant":
            if m.get("tool_calls"):
                for t in m["tool_calls"]:
                    a = t.get("arguments")
                    if isinstance(a, str):
                        try: a = json.loads(a)
                        except Exception: pass
                    rows.append(f"  agent -> tool: {t['name']}({json.dumps(a, ensure_ascii=False)[:220]})")
            elif (m.get("content") or "").strip():
                rows.append(f"  agent: {' '.join((m.get('content') or '').split())[:200]}")
        elif m["role"] == "user":
            if m.get("tool_calls"):
                rows.append(f"  customer (on their phone): {', '.join(t['name'] for t in m['tool_calls'])}")
            elif (m.get("content") or "").strip():
                rows.append(f"  customer: {' '.join((m.get('content') or '').split())[:110]}")
    txt = "\n".join(rows)
    if len(txt) > budget_chars:
        half = budget_chars // 2
        txt = txt[:half] + "\n  [… middle of the correct run elided …]\n" + txt[-half:]
    block = ("<Correct Run Of The Same Task>\nThe same agent PASSED this task in another attempt (trial "
             f"{sib.get('trial')}). Its agent-side trace, in order:\n{txt}\n</Correct Run Of The Same Task>\n\n")
    return block, {"trial": sib.get("trial"), "id": sib.get("id"), "num_messages": len(sib["messages"]), "chars": len(block)}


POLICY_TEXT = {}
PIPELINE_VERSION = "self_critique"

STOP = set("a an the and or but if of to in on at for from with without by as is are was were be been being "
           "this that these those it its their there here what which who whom when where why how not no nor "
           "should would could must need needs needed may might can will shall do does did done have has had "
           "you your they them he she we our us i me my than then so such only also any all more most other "
           "into over under about after before during while because since agent user customer tool tools call "
           "called calls make made makes take taken takes give given gives get gets got".split())


def content_words(text):
    """Lower-case words of 4+ characters that are not stopwords — enough to tell whether two
    sentences are about the same thing without pretending to do semantics."""
    return {w for w in re.findall(r"[a-z_]{4,}", (text or "").lower()) if w not in STOP}
CONTEXT_BUDGET_CHARS = 100_000  # ~25k tokens of prompt for a 32k-context model, leaving room for the answer


def fit_context(conv, budget_chars):
    """Shrink a transcript that, with a long policy (telecom), would overflow the critic's context.
    Two stages, never touching the [TURN n] labels so citations stay valid:
    1) truncate tool-result bodies (the agent's own words and calls are kept whole);
    2) if still too long, elide the middle turns with an explicit marker."""
    if len(conv) <= budget_chars:
        return conv
    conv = re.sub(r"(\*\*tool[^\n]*\*\*:\nReponse to: \w+\n```\n)(.{240,}?)(\n```)",
                  lambda m: m.group(1) + m.group(2)[:240] + " …[truncated]" + m.group(3), conv, flags=re.S)
    if len(conv) <= budget_chars:
        return conv
    turns = re.split(r"(?=\[TURN \d+\])", conv)
    head, tail = turns[:30], turns[-30:]
    body = "".join(head) + "\n[… middle turns elided to fit the context window; the agent's actions there are still listed in <Agent Actions By Turn> …]\n\n" + "".join(tail)
    while len(body) > budget_chars and len(tail) > 10:
        tail = tail[2:]
        body = "".join(head) + "\n[… middle turns elided to fit the context window; the agent's actions there are still listed in <Agent Actions By Turn> …]\n\n" + "".join(tail)
    return body


def critique(sim, task, ctx, model, temperature, siblings=None):
    policy, agent_tools, user_tools, env_ctor = ctx
    sib_txt, sib_meta = sibling_block((siblings or {}).get(sim["task_id"], []), sim.get("id"))
    POLICY_TEXT["current"] = policy
    ri = sim.get("reward_info") or {}
    ec = task.get("evaluation_criteria") or {}
    run = SimulationRun(**sim)
    conv = MarkdownDisplay.display_messages(run.messages)
    diff_rows = None
    if (ri.get("db_check") or {}).get("db_match") is False:
        try:
            diff_rows = db_diff(Task(**task), run.messages, env_ctor)
        except Exception as e:
            diff_rows = [f"  (could not replay the trajectory to compute the diff: {type(e).__name__}: {str(e)[:120]})"]
    # make the customer's phone actions unmistakable in the transcript
    conv = re.sub(r"\*\*user\*\*:\n\*\*Tool Call\*\*", "**user — CUSTOMER operating their own phone (not the agent)**:\n**Tool Call**", conv)
    conv = fit_context(conv, budget_chars=CONTEXT_BUDGET_CHARS - len(policy) - 6000 - len(sib_txt))
    acts = json.dumps([{k: a[k] for k in ("action_id", "requestor", "name", "arguments") if k in a}
                       for a in ec.get("actions") or []], indent=1)
    task_blob = {"user_scenario": task.get("user_scenario")}
    for k in ("ticket", "initial_state"):
        if task.get(k): task_blob[k] = task[k]
    ct = ""
    if user_tools:
        ct = ("\n<Customer Device Tools>\nThese run on the CUSTOMER's phone. The agent cannot call them — "
              "it must ask the customer to do it.\n"
              + "\n".join(f"  {n}: {d}" for n, d in user_tools) + "\n</Customer Device Tools>\n")
    prompt = USER.format(
        policy=policy,
        agent_tools="\n".join(f"  {n}: {d}" for n, d in agent_tools),
        customer_tools=ct,
        task=json.dumps(task_blob, indent=1),
        actions=acts, criteria=build_criteria(ec, task.get("description")),
        reward=ri.get("reward"), verdict=build_verdict(ri, diff_rows), sibling=sib_txt,
        tool_index=tool_index(sim["messages"]), conversation=conv)
    out = None; txt = ""
    for attempt in range(2):  # one retry: a truncated/fenced JSON is the usual failure
        r = litellm.completion(model=model, temperature=temperature if attempt == 0 else 0.3, max_tokens=3500,
                               messages=[{"role": "system", "content": SYSTEM},
                                         {"role": "user", "content": prompt}])
        txt = (r.choices[0].message.content or "").strip()
        txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.S)
        m = re.search(r"\{.*\}", txt, re.S)
        try:
            # strict=False: weaker critics emit raw newlines inside JSON strings. This only adds
            # tolerance; anything that parses strictly parses identically.
            out = json.loads(m.group(0) if m else txt, strict=False); break
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
    if out is None:
        return {"parse_error": err, "raw": txt[:1500], "prompt_chars": len(prompt)}
    out["db_diff"] = diff_rows
    out.update(audit(sim, out, agent_tools, user_tools, task_obj=Task(**task)))
    out["prompt_chars"] = len(prompt)
    out["sibling_used"] = sib_meta
    return out


GOODBYE = re.compile(r"\b(thank\w*|thn?a?ks|thx|bye|good.?bye|great|that.s (all|everything|it)|everything i need|appreciate|no further|"
                     r"nothing (else|further|more)|perfect|good|all set|done|au revoir|merci|gracias|adi[oó]s|ciao|danke|take care)\b", re.I)
# an explicit instruction to act: a hang-up even if wrapped in thanks
STRONG_REQUEST = re.compile(r"\b(proceed|go ahead|please (do|proceed|book|cancel|change|update|process|make|apply|exchange|return|keep looking)|"
                            r"let.s do (it|that)|do it|i(.ll| will) take (the|that|option)|book it|cancel (it|them|all)|i confirm|make it so)\b", re.I)
# a bare affirmation ("Yes." / "Sure, please") answering the agent's "shall I ...?"
WEAK_REQUEST = re.compile(r"^(yes|yeah|yep|sure|ok(ay)?)[,.! ]*(please|go ahead|do that|do it)?[.! ]*$", re.I)


def customer_hung_up(last_user_msg):
    """True only when the customer's FINAL message asks the agent to act and carries ###STOP###
    in the same breath — the agent never got a turn to act, so the failure is not attributable
    to an agent turn. A goodbye (any language, typos included), a bare 'No.' answering
    'anything else?', or a thank-you with a trailing question is a normal close, not a hang-up."""
    if "###STOP###" not in last_user_msg:
        return False
    rest = re.sub(r"</?think>", "", last_user_msg.replace("###STOP###", "")).strip()
    if not rest:
        return False
    if STRONG_REQUEST.search(rest) or WEAK_REQUEST.match(rest):
        return True
    if GOODBYE.search(rest) or re.fullmatch(r"(no|nope|nothing)[.! ]*", rest, re.I):
        return False
    return len(rest) > 60  # long, no goodbye, no request: the customer gave up mid-task


SHALLOW = {"to", "the", "for", "and", "by", "id", "get", "set", "details", "order", "user"}
NEGATES = re.compile(r"(should not|shouldn't|must not|mustn't|should have (refrained|avoided|refused|declined)|"
                     r"not have (made|performed|called|attempted|enabled|sent|initiated|proceeded)|"
                     r"never have|refrain(ed|ing)? from|avoid(ed|ing)? (making|calling|performing|enabling|sending)|"
                     r"no (database )?(writes?|changes?)|without (making|performing|calling))", re.I)
ARITH = re.compile(r"\$?([\d,]+\.\d{2})\s*([-+])\s*\$?([\d,]+\.\d{2})"
                   r"(?:\s*([-+])\s*\$?([\d,]+\.\d{2}))?\s*=\s*\$?([\d,]+\.\d{2})")
SELF_REFUTING = re.compile(r"(—\s*contradiction|\bcontradiction\b\s*[;,]\s*(thus|so|therefore)|"
                           r"however,\s*upon\s*(recheck|re-check|review|reexamin)|"
                           r"which indicates a discrepancy)", re.I)


def audit(sim, out, agent_tools, user_tools, task_obj=None):
    """Mechanical checks that decide whether a critique is safe to use as a label.
    Sets `usable`, `unusable_reasons` (hard rejects) and `warnings` (repairs: snapped turns,
    dropped quotes or contributing errors). The critic's own `blame` field is unreliable
    (it almost always says 'agent'), so customer-caused failures are detected here instead."""
    ms = sim["messages"]; n = len(ms); ri = sim.get("reward_info") or {}
    reasons = []; warnings = []
    rc = out.get("root_cause") or {}
    t = rc.get("turn_idx")
    cited = [t] + [c.get("turn_idx") for c in out.get("contributing_errors") or []]
    out["turn_idx_valid"] = all(isinstance(x, int) and 0 <= x < n for x in cited if x is not None)
    if not out["turn_idx_valid"]:
        reasons.append("turn_idx out of range")

    # Contributing errors must also cite agent turns and describe agent actions, not things the
    # customer did on their phone. Snap off-by-one pointers, drop the rest, and keep the critique:
    # a bad contributing error is a bad line, not a bad root cause.
    kept_ce, dropped_ce = [], []
    for ce in out.get("contributing_errors") or []:
        ct = ce.get("turn_idx")
        if not (isinstance(ct, int) and 0 <= ct < n):
            dropped_ce.append((ct, "turn out of range")); continue
        if ms[ct]["role"] == "tool" and ct > 0 and ms[ct - 1]["role"] == "assistant":
            ce["turn_idx_as_written"] = ct; ce["turn_idx"] = ct = ct - 1
        elif ms[ct]["role"] == "user" and not ms[ct].get("tool_calls") and ct + 1 < n and ms[ct + 1]["role"] == "assistant":
            ce["turn_idx_as_written"] = ct; ce["turn_idx"] = ct = ct + 1
        if ms[ct]["role"] != "assistant":
            dropped_ce.append((ct, f"points at a {ms[ct]['role']} turn"
                                   + (" (a customer phone action)" if ms[ct].get("tool_calls") else ""))); continue
        if re.match(r"\s*the (user|customer)\b", (ce.get("what_happened") or ""), re.I):
            dropped_ce.append((ct, "describes something the customer did, not the agent")); continue
        kept_ce.append(ce)
    if dropped_ce:
        out["contributing_errors"] = kept_ce
        out["contributing_errors_dropped"] = dropped_ce
        warnings.append(f"{len(dropped_ce)} contributing error(s) dropped: "
                        + "; ".join(f"turn {a}: {b}" for a, b in dropped_ce))
        if not kept_ce and len(dropped_ce) >= 2:
            reasons.append("every contributing error pointed at a non-agent turn or described the customer")
    if isinstance(t, int) and 0 <= t < n:
        # snap an off-by-one pointer (tool result / customer question) to the adjacent agent turn
        orig = t
        if ms[t]["role"] == "tool" and t > 0 and ms[t - 1]["role"] == "assistant":
            t -= 1
        elif ms[t]["role"] == "user" and not ms[t].get("tool_calls") and t + 1 < n and ms[t + 1]["role"] == "assistant":
            t += 1
        if t != orig:
            rc["turn_idx"] = t; rc["turn_idx_as_written"] = orig; out["turn_idx_snapped"] = True
        m_ = ms[t]
        out["root_cause_role"] = m_["role"] + (" (tool_call)" if m_.get("tool_calls") else "")
        if m_["role"] != "assistant":
            reasons.append(f"root cause points at a {m_['role']} turn, not the agent"
                           + (" (a customer phone action)" if m_.get("tool_calls") else ""))
        # Only POSITIVE, ACTUAL claims ("the agent called X") are checked against the transcript.
        # A sentence that is negated ("never called X"), normative ("should have called X",
        # "the reference requires X") or hypothetical ("the only way was to call X") is skipped —
        # those are the whole point of many valid critiques.
        names = {x for x, _ in agent_tools} | {x for x, _ in user_tools}
        text = f"{rc.get('what_happened','')} {rc.get('why_it_caused_the_failure','')}"
        NONACTUAL = re.compile(r"\b(not|never|without|fail\w*|instead|omit\w*|miss\w*|should|would|could|must|need\w*|necessary|require\w*|reference|solution|expected|answer key|correct action|valid action|was to|is to|to achieve|via the)\b", re.I)
        positive = set()
        for sent in re.split(r"(?<=[.;])\s+", text):
            if NONACTUAL.search(sent):
                continue
            positive |= {x for x in re.findall(r"\b([a-z_]{4,})\b", sent) if x in names}
        window = {tc["name"] for k in range(max(0, t - 2), min(n, t + 3)) for tc in (ms[k].get("tool_calls") or [])}
        anywhere = {tc["name"] for k in range(n) for tc in (ms[k].get("tool_calls") or [])}
        ghost = positive - window
        if positive and ghost and not (positive & window):
            if ghost & anywhere:
                reasons.append(f"root cause describes a call made elsewhere, not at turn {t}: {sorted(ghost)}")
            else:
                reasons.append(f"root cause names tools the agent never called: {sorted(ghost)}")
        # A successful read-only lookup is not a root cause: the mistake is what the agent did
        # with the result. Advance the pointer to the agent's next output (its wrong conclusion
        # or wrong write) and record the move, rather than discarding an otherwise-good critique.
        if m_.get("tool_calls") and all(re.match(r"^(get_|search_|find_|list_|calculate$)", tc["name"]) for tc in m_["tool_calls"]):
            nxt = ms[t + 1] if t + 1 < n else {}
            if nxt.get("role") == "tool" and str(nxt.get("error")) != "True" and not (nxt.get("content") or "").startswith("Error"):
                k = next((j for j in range(t + 2, n) if ms[j]["role"] == "assistant"), None)
                if k is not None:
                    rc.setdefault("turn_idx_as_written", t); rc["turn_idx"] = k
                    out["turn_idx_advanced_past_lookup"] = True
                    m_ = ms[k]; t = k
                    out["root_cause_role"] = m_["role"] + (" (tool_call)" if m_.get("tool_calls") else "")
                else:
                    reasons.append("root cause is a successful read-only lookup with no agent turn after it")
        if rc.get("policy_quote") and str(rc["policy_quote"]).strip().lower() not in ("null", "none", ""):
            # accept exact, reformatted, or lightly paraphrased quotes; flag invented ones
            import difflib
            norm = lambda x: re.sub(r"\s+", " ", re.sub(r"[*_`>#\-]+", " ", x)).strip().lower()
            q = norm(str(rc["policy_quote"]))
            pol = norm(POLICY_TEXT.get("current", ""))
            lines = [norm(l) for l in POLICY_TEXT.get("current", "").split("\n") if len(l.strip()) > 20]
            best = max((difflib.SequenceMatcher(None, q[:160], l[:160]).ratio() for l in lines), default=0.0)
            # any 40-char window of the quote appearing verbatim in the policy counts (handles a
            # quote that starts mid-sentence or is one clause of a long policy line)
            window_hit = any(q[i:i + 40] in pol for i in range(0, max(1, len(q) - 40), 8))
            out["policy_quote_similarity"] = round(best, 2)
            out["policy_quote_found"] = window_hit or best >= 0.7
            if not out["policy_quote_found"]:
                # an invented quote is dropped (the field is optional) rather than sinking the whole critique
                rc["policy_quote_as_written"] = rc["policy_quote"]; rc["policy_quote"] = None
                warnings.append("policy_quote was not found in the policy and was dropped (paraphrase of the task/answer key)")
            else:
                # Being IN the policy is not the same as being ABOUT this failure: boilerplate such
                # as "You should not make up any information" passes the lookup but explains
                # nothing. Require the quote to share content words with what went wrong.
                rc_words = content_words(f"{rc.get('what_happened','')} {rc.get('why_it_caused_the_failure','')} "
                                         f"{rc.get('should_have','')} {out.get('failed_condition','')}")
                q_words = content_words(str(rc["policy_quote"]))
                shared = rc_words & q_words
                out["policy_quote_overlap"] = sorted(shared)[:8]
                out["policy_quote_relevant"] = len(shared) >= 2
                if not out["policy_quote_relevant"]:
                    rc["policy_quote_as_written"] = rc["policy_quote"]; rc["policy_quote"] = None
                    warnings.append(f"policy_quote is in the policy but shares {len(shared)} content word(s) with the "
                                    f"critique, so it does not explain this failure; dropped")
    else:
        reasons.append("no valid root-cause turn")
    # failures not attributable to an agent turn (the critic does not flag these reliably)
    tr = sim.get("termination_reason")
    if tr in ("max_steps", "too_many_errors"):
        reasons.append(f"conversation terminated by {tr}: task never completed, not attributable to one agent turn")
    elif ri.get("reward") == 0 and not ri.get("reward_basis"):
        reasons.append("unscored (empty reward_basis): task never evaluated")
    utext = [m.get("content") or "" for m in ms if m["role"] == "user" and (m.get("content") or "").strip()]
    if utext:
        if customer_hung_up(utext[-1]):
            reasons.append("customer hung up mid-instruction (asked the agent to act, then STOP in the same message)")
        if len(utext) >= 8 and len({u[:60] for u in utext[-8:]}) <= 2:
            reasons.append("customer goodbye loop")
    if "assert_" in f"{rc.get('what_happened','')}":
        reasons.append("describes an assertion as if it were an agent action")
    sh = (rc.get("should_have") or "").lower()
    cust_only = {x for x, _ in user_tools} - {x for x, _ in agent_tools}
    told = {x for x in re.findall(r"\b([a-z_]{4,})\b", sh) if x in cust_only}
    if told:
        # "the agent should have called/used/run <customer tool>" is wrong advice; "asked the customer to run <tool>" is fine
        for m in re.finditer(r"\b(call|run|use|using|invoke|execute|perform)\w*\b([^.]{0,80}?)\b(" + "|".join(sorted(told)) + r")\b", sh):
            clause = sh[max(0, m.start() - 80):m.end()]
            # "ask the user to run X" is right; "toggled the user's airplane mode via a tool call"
            # is the agent doing it itself. Only an ACTOR mention exonerates the sentence.
            if not re.search(r"\b(ask|tell|instruct|guide|have|request|direct|walk)\w*\s+(the\s+)?(customer|user)\b", clause):
                reasons.append(f"should_have tells the agent itself to run a customer-device tool it cannot call: {m.group(3)}")
                break
    # Reject only telling the agent to emit the key's value regardless of what it could derive;
    # a concrete action that merely ends with "as per the reference" is fine.
    parrot = re.search(r"(even if (the|its) calculated value|regardless of (the|its) (own )?calculat|state(d)? (the )?(answer.key|reference) value|say (the )?(number|value|amount) (from|in) the (answer key|reference))", sh)
    no_action = not re.search(r"\b(calculat|sum|add|look ?up|call|ask|check|retriev|verif|refus|cancel|includ|select|perform|confirm|list|proceed|updat|modif|book|search|explain|deny|transfer|grant|enabl|disabl|refuel|reset|toggle|instruct|guide|tell|inform|offer|return|refund|exchang|process|remov|resum|suspend|send|pay|downgrad|upgrad|declin|us(e|ing)|appl(y|ied))\w*", sh)
    if parrot or (no_action and re.search(r"(answer key|reference solution)", sh)):
        reasons.append("should_have just points at the answer key instead of an action")
    dd = out.get("db_diff") or []
    # On a pure-refusal task (every diff row is a forbidden write), should_have must not teach
    # the agent how to perform the write.
    pure_refusal = any("reference solution makes no writes at all" in r for r in dd)
    WRITE_VERBS = r"(proceed\w*|cancel\w*|book\w*|updat\w*|upgrad\w*|issu\w*|send|sent|modif\w*|perform\w*|mak\w* (the )?(change|update)s?)"
    teaches_write = (re.search(r"\b(then|only then|only after|after (confirm|verif|obtain|receiv|get|ask|present|list)\w*)\b[^.]{0,90}?\b" + WRITE_VERBS + r"\b", sh)
                     or re.search(r"\b(confirm\w*|'yes'|\"yes\")\b[^.]{0,80}?\bbefore (making|proceeding|updating|cancel\w*|book\w*|the (change|update))", sh)
                     or re.search(r"\bbefore " + WRITE_VERBS + r"[^.]{0,60}\b(confirm\w*|'yes')", sh))
    refuses = re.search(r"\b(refus\w*|declin\w*|should not have (made|performed|called|cancel|book|updat|upgrad|issu|sent|proceed)\w*|must not (have )?(made|perform|call|cancel|book|updat|upgrad|issu|proceed)\w*|without making any|no (database )?(writes?|changes?) (at all|should))\b", sh)
    if pure_refusal and teaches_write and not refuses:
        reasons.append("pure-refusal task (every diff is a forbidden write) but should_have tells the agent how to perform the write")
    # The mirror: when the diff has MISSING writes the agent had to DO something, so a
    # should_have that only tells it to refuse contradicts the answer key.
    has_missing = any("[MISSING WRITE]" in r for r in dd)
    if has_missing and refuses and not teaches_write:
        reasons.append("the diff lists writes the agent FAILED to make, but should_have only tells it to refuse")
    # should_have must not contradict a reference write: a negated sentence ("should not have
    # cancelled ...") naming every distinctive word of a required write tool is rejected. Keying
    # on the tool name, not generic refusal wording, avoids misreading a prescribed write.
    if task_obj is not None:
        acts = [a for a in (task_obj.evaluation_criteria.actions or [])
                if not re.match(r"^(get_|search_|find_|list_|calculate$|think$|check_|transfer_to_human)", a.name)]
        shs = re.split(r"(?<=[.;])\s+", rc.get("should_have") or "")
        for sent in shs:
            if not NEGATES.search(sent):
                continue
            for a in acts:
                words = [w for w in a.name.split("_") if len(w) > 2 and w not in SHALLOW]
                if words and all(re.search(r"\b" + w + r"\w*", sent, re.I) for w in words):
                    reasons.append(f"should_have tells the agent not to do what the answer key requires: {a.name}")
                    break

    # An arithmetic claim in should_have must evaluate. Narrow on purpose: only fully
    # written-out expressions are checked, never a number the critic merely asserts.
    for mm in ARITH.finditer(rc.get("should_have") or ""):
        f = lambda x: float(x.replace(",", "")) if x else None
        a1, op1, b1, op2, c1, res = f(mm.group(1)), mm.group(2), f(mm.group(3)), mm.group(4), f(mm.group(5)), f(mm.group(6))
        v = a1 + b1 if op1 == "+" else a1 - b1
        if c1 is not None:
            v = v + c1 if op2 == "+" else v - c1
        if abs(v - res) > 0.011:
            reasons.append(f"should_have contains an arithmetic claim that does not check out: {mm.group(0)}")
            break

    # A draft that argues with itself instead of prescribing an action.
    if SELF_REFUTING.search(rc.get("should_have") or ""):
        reasons.append("should_have argues with itself instead of prescribing an action")

    out["pipeline_version"] = PIPELINE_VERSION
    out["usable"] = not reasons
    out["unusable_reasons"] = reasons
    out["warnings"] = warnings
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results"); ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--model", default="hosted_vllm/qwen3-4b")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--include-passed", action="store_true")
    ap.add_argument("--siblings", help="results.json with all rollouts of the same tasks; a PASSING sibling run "
                                        "of the same task is shown to the critic as a concrete correct path")
    a = ap.parse_args()
    siblings = load_siblings(a.siblings) if a.siblings else None

    d = json.load(open(a.results))
    tasks = {t["id"]: t for t in d["tasks"]}
    dom = d["info"]["environment_info"]["domain_name"]
    ctx = domain_context(dom)

    def graded(s):
        return ((s.get("reward_info") or {}).get("reward") is not None
                and s.get("termination_reason") != "infrastructure_error")

    # A simulation with no reward (e.g. the grader crashed) or an infrastructure error was never
    # graded; it is not a failure, so it is skipped rather than critiqued.
    ungraded = [s for s in d["simulations"] if not graded(s)]
    targets = [s for s in d["simulations"]
               if graded(s) and (a.include_passed or s["reward_info"]["reward"] < 1)]
    if a.limit: targets = targets[:a.limit]
    if ungraded:
        print(f"skipping {len(ungraded)} ungraded simulation(s) (no reward_info / infrastructure_error)", flush=True)
    print(f"domain {dom} | policy {len(ctx[0])} chars | {len(ctx[1])} agent tools, {len(ctx[2])} customer tools "
          f"| critiquing {len(targets)}/{len(d['simulations'])} sims with {a.model}", flush=True)

    done = 0
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs = {ex.submit(critique, s, tasks[s["task_id"]], ctx, a.model, a.temperature, siblings): s for s in targets}
        for f in as_completed(futs):
            s = futs[f]
            try:
                s["self_critique"] = f.result()
            except Exception as e:
                s["self_critique"] = {"parse_error": f"call failed: {type(e).__name__}: {e}"}
            done += 1
            if done % 10 == 0 or done == len(targets): print(f"  {done}/{len(targets)}", flush=True)
    json.dump(d, open(a.output, "w"))
    ok = [s for s in targets if "parse_error" not in s["self_critique"]]
    if siblings is not None:
        print(f"sibling shown in {sum(1 for s in targets if (s['self_critique'].get('sibling_used')))}/{len(targets)} prompts")
    print(f"parsed {len(ok)}/{len(targets)} | turn_idx valid {sum(1 for s in ok if s['self_critique'].get('turn_idx_valid'))}/{len(ok)} "
          f"| blame {dict((b, sum(1 for s in ok if s['self_critique'].get('blame') == b)) for b in ('agent','customer','environment','unclear'))}")
    print(f"→ {a.output}")


if __name__ == "__main__":
    main()
