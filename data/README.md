# Data

This directory holds the released data. The code finds it through `MEMCOMPOSER_DATA`, which
defaults to this directory. Only this README is tracked in git; the data itself is downloaded from
Hugging Face.

## Memory banks

Each executor model has its own memory bank. It is built from that model's own no-memory
trajectories on the training split (see `memory_bank/`).

```
tau2/memory_banks/<agent>/<domain>/
  task<NN>_trial<t>/memory.json     the full record: transcript, task, review, grader evidence
  task<NN>_trial<t>/digest.txt      compact rendering, read by MemComposer
  task<NN>_trial<t>/full.txt        full rendering, given to the executor
  index.npz                         Qwen3-Embedding-8B vectors of each memory's retrieval key
                                    (the customer's opening request), L2-normalised, one row per folder
  index_manifest.json               embedding model, query instruction, row order
  query_keys.jsonl                  the retrieval key of each row

alfworld/memory_banks/<model>/
  <taskdir>__<trial>/memory.json, digest.txt, full.txt      (retrieval key: the goal text)
```

### τ²-bench

| agent | model | airline | retail | telecom |
|---|---|---|---|---|
| `qwen` | Qwen3-4B-Instruct-2507 | 58 | 143 | 141 |
| `ministral` | Ministral-3-8B-Instruct-2512 | 57 | 144 | 144 |
| `qwen32b` | Qwen3-32B | 60 | 147 | 146 |
| `qwen235b` | Qwen3-235B-A22B-Instruct-2507 | 60 | 146 | 148 |
| `gpt54` | GPT-5.4 | 59 | 147 | 148 |
| `gpt54mini` | GPT-5.4-mini | 60 | 148 | 148 |

The `qwen` and `ministral` banks are used by their own executors. The other four supply memories
to the Qwen3-4B executor in the memory-source transfer experiment.

### ALFWorld

| model | memories |
|---|---|
| `ministral8b` (Ministral-3-8B-Instruct-2512) | 3,433 |
| `qwen3-4b` (Qwen3-4B-Instruct-2507) | 3,267 |

These are built from all 3,553 training games, one trajectory each. A trajectory is left out when its
self-review failed the audit, or when the game was won through an environment quirk.

## Training data

One parquet row per MemComposer prompt (format: `training/README.md`). The prompts are already
rendered from the memory banks above, so `extra_info.folders` names folders of the matching bank.

| path | bank | train rows | val rows |
|---|---|---|---|
| `training/tau2/qwen/retail/` | `tau2/memory_banks/qwen/retail` | 464 | 40 |
| `training/tau2/qwen/telecom/` | `tau2/memory_banks/qwen/telecom` | 424 | 40 |
| `training/tau2/ministral/retail/` | `tau2/memory_banks/ministral/retail` | 472 | 40 |
| `training/tau2/ministral/airline/` | `tau2/memory_banks/ministral/airline` | 108 | 20 |
| `training/tau2/ministral/telecom/` | `tau2/memory_banks/ministral/telecom` | 284 | 40 |
| `training/alfworld/ministral/` | `alfworld/memory_banks/ministral8b` | 512 | 133 |

Each τ²-bench validation row is one test task scored over 4 trials; an ALFWorld validation row is
one `valid_unseen` game, played once.

`alfworld/init_obs.jsonl` holds the opening observation of every train and `valid_unseen` game. The
ALFWorld prompt builder (`memcomposer/alfworld/memcomposer_prompt.py`) reads the starting room from
it. `alfworld/task_index.jsonl` is machine-specific (it holds absolute game-file paths); build it
with `memory_bank/alfworld/build_task_index.py`.
