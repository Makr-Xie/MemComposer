# Memory bank construction

Builds the per-agent, per-domain memory banks described in Appendix C of the paper. A bank is built
only from the agent's own no-memory rollouts on the τ²-bench **train** split, and the agent reviews its own trajectories.

## Pipeline

| step | script | reads | writes |
|---|---|---|---|
| 1. rollouts | `collect.sh` → `collect_rollouts.py` | `openings/<d>/train_op{0..3}.parquet` | `rollouts/<agent>/<d>/sims.jsonl`, `k0.op*.jsonl` |
| 2. select | `select_memories.py` | rollouts | `memory_banks/<agent>/<d>/selection.json` |
| 3. review | `label.sh` → `review/self_critique.py`, `review/pass_audit.py` | selection | `labelled_sc.json`, `labelled_pa.json` |
| 4. audit + top-up | `topup.py` | labels, rollouts | `labels.json`, `selection_final.json` |
| 5. render | `render.py` (uses `digest_memory.py`) | `selection_final.json` | `task<NN>_trial<t>/{memory.json,digest.txt,full.txt}`, `query_keys.jsonl`, `curation.jsonl` |
| 6. index | `build_index.py` | `query_keys.jsonl` | `index.npz`, `index_manifest.json` |
| 7. retrieve | `retrieve.py` | `index.npz`, a file of query openings | top-k candidate pools (`retrieval.json`) |

**Rollouts.** The frozen agent runs 8 rollouts per train task (4 sampled customer openings × 2 τ²-bench seeds) at
temperature 1.0, against the GLM-4.7 user simulator at temperature 0. Pass rates count valid simulations only. A
simulation that τ²-bench aborted, for example on invalid tool-call JSON, has no transcript and is neither counted nor
selected.

**Selection.** Each task keeps 2 trajectories, chosen by its pass rate: two passes if ≥ 7/8, two failures if
≤ 1/8, otherwise one of each. The pick within a bucket is random, seeded by `(seed, domain)`.

**Review.** Failures get `self_critique.py`, which locates the first wrong agent turn and states what the agent should
have done; a passing sibling of the same task is shown as a contrastive example when one exists. Passes get
`pass_audit.py`, which labels them `clean_pass` or `fake_pass`. Each review is then checked mechanically against the
transcript and the grader. A trajectory whose review fails the check is replaced by another of the same outcome and
reviewed again (`topup.py`), for up to 4 rounds.

**Representation.** `digest.txt` is the compact form the selector reads; `full.txt` is the full form the executor
receives. Each memory is keyed by the customer's opening request, embedded with Qwen3-Embedding-8B and
L2-normalised.

## Setup

```bash
export MEMCOMPOSER_DATA=/path/to/data      # default: <repo>/data; this pipeline uses <data>/tau2/
export TAU2_ROOT=/path/to/tau2-bench       # only if `tau2` is not pip-installed
export PYTHON=python                       # interpreter for the shell drivers; must import tau2

# credentials, read by agents.py — never written to disk or logs
export BEDROCK_API_KEY=... BEDROCK_BASE_URL=...      # Bedrock agents (ministral, qwen32b, qwen235b)
export OPENAI_API_KEY=...                            # OpenAI agents (gpt54, gpt54mini)
```

The agents are registered in `agents.py`: `ministral`, `qwen`, `qwen32b`, `qwen235b`, `gpt54mini` and `gpt54`. To add
an agent, add one line there; no other script needs editing.

Steps 6 and 7 need a Qwen3-Embedding-8B server with an OpenAI-compatible `/v1/embeddings` endpoint (default
`http://127.0.0.1:8012/v1`, override with `--url`).

## Running

```bash
AGENT=qwen32b DOMAIN=retail ./collect.sh              # repeat per domain
python select_memories.py --agent qwen32b             # all three domains
AGENT=qwen32b bash label.sh
python topup.py --agent qwen32b --domain retail       # per domain
python render.py --agent qwen32b retail               # per domain
python build_index.py --agent qwen32b retail          # per domain
python retrieve.py --agent qwen32b --domain retail \
    --openings openings.json --out retrieval.json --top-k 10 --cap 2
```

## Reproducibility

- **Deterministic steps.** Steps 2 and 5 make no model calls. Given the same rollouts and labels,
  they produce the same output.
- **Top-up (step 4) depends on run history.** When several reviewed candidates qualify for a slot,
  the pick depends on the order of the earlier rounds. Re-running it from the final `labels.json`
  alone can pick a different trajectory of the same task and outcome.
