# Memory bank construction: ALFWorld

Builds one memory bank per executor model (`ministral8b`, `qwen3-4b`) from that model's own
no-memory trajectories on the ALFWorld **train** split (3,553 games, one attempt each). The same
model reviews its own trajectories.

## Pipeline

| step | script | reads | writes |
|---|---|---|---|
| 0. task index | `build_task_index.py` | `$ALFWORLD_DATA/json_2.1.1` | `task_index.jsonl` |
| 1. rollouts | `collect_raw.py` | task index | `rollouts/<model>/train_full_s20.jsonl` |
| 2. facts | `build_facts.py` | rollouts, game files | `memory_banks/<model>/_work/facts.jsonl` |
| 3. self-review | `self_review.py` (prompts in `self_review_prompt.py`) | rollouts, facts, walkthroughs | `_work/reviews_<tag>.jsonl` |
| 3b. re-audit | `reaudit_reviews.py` | `_work/reviews_full.jsonl` | `_work/reviews_full_final.jsonl` |
| 4. assemble | `build_memory_base.py` | rollouts, facts, final reviews | `<group_id>__<trial>/{memory.json,full.txt,digest.txt}`, `README.md` |
| 5. retrieval | `build_retrieval.py`, `build_train_retrieval.py` | bank, task index | `retrieval/<model>/valid_unseen_top10.json`, `train_q256_top20.json` |

All paths are under `<MEMCOMPOSER_DATA>/alfworld/`. Step 1 plays games through the executor in
`memcomposer/alfworld/`.

**Rollouts.** Each game is played once, without memory: temperature 0.4, at most 20 steps,
max_tokens 2048, history length 10. Games run in a seeded shuffle, and the run can be resumed.

**Facts.** A deterministic replay of each trajectory from its observation text. It records the
target, where the target was at the start, progress (seen / taken / transformed / placed),
deviations, search coverage, revisits and wasted runs. For failures it also records the earliest
failed stage. Facts are evidence for the critic and are rendered for the executor. A game won
through ALFWorld's double-`inReceptacle` quirk is flagged `env_quirk_early_win` and left out of the
bank.

**Self-review.** The critic is the executor model itself, at temperature 0. It sees the rules,
the task, the walkthrough as an answer key, the verdict, the facts, and the trajectory with think
text cut to about 300 characters. Failures get a root cause, contributing errors, a failed stage
and `should_have` actions. Successes get key steps, wasted steps and a reusable strategy.

Every answer goes through an 11-point mechanical audit: cited steps must exist, quotes must be
verbatim, advice must not break the rules, the failed stage must be consistent with the facts,
and so on. A rejected answer is retried once at temperature 0.3, with the reasons attached. If it
fails again, the trajectory is dropped. Near misses, such as a missing instance number or a short
wasted run, are repaired and flagged rather than rejected.

**Representation.** `full.txt` (for the executor) holds every step with its think, the full
observations and all annotations. `digest.txt` (for the selector) drops the think, cuts each
observation to 160 characters, and keeps only "what happened" and "should have". Each memory is
keyed by its goal text.

**Retrieval.** The top-10 pool for each `valid_unseen` game is computed once and then frozen.
Every method chooses from the same 10 memories. The query is embedded with Qwen3-Embedding-8B
using an instruction; the memory keys are embedded raw. Ties are broken by
`sha256(seed|query|memory)`, and each source taskdir contributes at most 2 memories. Training
queries exclude their own taskdir. Both scripts refuse to overwrite an existing pool, and each
pool records a SHA-256 of the bank it was built from.

## Setup

```bash
export MEMCOMPOSER_DATA=/path/to/data       # default: <repo>/data; this pipeline uses <data>/alfworld/
export ALFWORLD_DATA=/path/to/alfworld_data # what `alfworld-download` filled; default ~/.cache/alfworld
export BEDROCK_API_KEY=... BEDROCK_BASE_URL=...         # ministral8b, as executor and as critic
export ALFW_AGENT_BASE=http://host:port/v1              # collect_raw.py with a local vLLM executor
export ALFW_CRITIC_BASE=http://host:port/v1[,...]       # self_review.py critic for qwen3-4b
```

The executor and the reviewer only need `textworld` and `alfworld`, so they run in the plain
`alfworld` environment. The retrieval scripts also need `numpy` and an OpenAI-compatible
Qwen3-Embedding-8B server (`--url`, default `http://127.0.0.1:8012/v1`).

## Running

```bash
python build_task_index.py
python collect_raw.py --split train --run-id train_full_s20 \
    --out $MEMCOMPOSER_DATA/alfworld/rollouts/ministral8b/train_full_s20.jsonl
python build_facts.py --model ministral8b
python self_review.py --model ministral8b --pilot 48 --tag pilot --dry-run 6   # prompts only
python self_review.py --model ministral8b --all --tag full
python reaudit_reviews.py --model ministral8b --reviews reviews_full
python build_memory_base.py --model ministral8b
python build_retrieval.py --model ministral8b --split valid_unseen
python build_train_retrieval.py --model ministral8b --n 256
```

## Reproducibility

- **Deterministic steps.** Steps 2, 3b and 4 make no model calls. Given the same rollouts and
  reviews, they produce the same output.
- **Re-audit (step 3b).** `reaudit_reviews.py` replays every stored critic answer through the
  current audit, without re-sampling. `self_review.py` already applies the current audit, so step
  3b matters only for reviews written by an earlier version of the audit.
