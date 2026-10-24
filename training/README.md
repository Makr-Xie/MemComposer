# Training MemComposer (ExPO)

The selector is trained with LoRA on [verl](../verl/README.md). Each rollout is a selection of
memories. Its reward comes from running the frozen executor with those memories in the real
environment, through the simulation services in `memcomposer/`. The reward function is
`memcomposer/reward.py`.

## Runs

| benchmark | selector / executor | domain | script |
|---|---|---|---|
| τ²-bench | Qwen3-4B-Instruct-2507 | retail | `tau2_qwen/run_retail.sh` |
| τ²-bench | Qwen3-4B-Instruct-2507 | telecom | `tau2_qwen/run_telecom.sh` |
| τ²-bench | Ministral-3-8B-Instruct | retail | `tau2_ministral/run_retail.sh` |
| τ²-bench | Ministral-3-8B-Instruct | airline | `tau2_ministral/run_airline.sh` |
| τ²-bench | Ministral-3-8B-Instruct | telecom | `tau2_ministral/run_telecom.sh` |
| ALFWorld | Ministral-3-8B-Instruct | — | `alfworld/run_alfworld.sh` |

The selector and the executor are the same base model; only the selector is trained. Each
`run_<domain>.sh` sets the hyperparameters and calls the launcher in its folder (`run_train.sh`).
Any variable it sets can be overridden from the environment.

## Setup

1. **verl.** Install it into `../verl/` and read [`../verl/README.md`](../verl/README.md) first.
   Two problems there go unnoticed during training:
   - The LoRA adapter may never reach vLLM.
   - Ministral-3-8B needs four fixes.
2. **Environments.** You need three Python environments:
   - **Training:** verl and vLLM. Pass it as `VENV=/path/to/venv`, or activate it.
   - **τ²-bench services:** [tau2-bench](https://github.com/sierra-research/tau2-bench) installed. Set `TAU2_PYTHON`, and set `TAU2_ROOT` to the checkout; its `.env` holds the user simulator's credentials.
   - **ALFWorld services:** `alfworld`, `textworld` and `fastapi`. Set `ALFWORLD_PYTHON` and `ALFWORLD_DATA`.
3. **Ministral model.** Download `mistralai/Ministral-3-8B-Instruct-2512-BF16`, then run
   `tau2_ministral/convert_ministral_text_only.py --src <dir> --apply`, then apply the tokenizer
   and RoPE fixes in `../verl/README.md` to the output (`<repo>/models/Ministral-3-8B-Instruct-2512-text`).
4. **Data and memory banks** from Hugging Face, laid out under `MEMCOMPOSER_DATA` (default `<repo>/data`):

```
data/
  training/tau2/qwen/{retail,telecom}/{train,val}.parquet
  training/tau2/ministral/{retail,airline,telecom}/{train,val}.parquet
  training/alfworld/ministral/{train,val}.parquet
  tau2/memory_banks/{qwen,ministral}/<domain>/            the banks the parquets point into
  alfworld/memory_banks/ministral8b/
  alfworld/task_index.jsonl                               memory_bank/alfworld/build_task_index.py
```

5. **Credentials** for Bedrock-hosted models: `BEDROCK_BASE_URL`, `BEDROCK_API_KEY`.

## Launching

Start the services, check that their `/health` lines report the right domain, then train.

```bash
# τ²-bench, Qwen3-4B: the executor runs locally on vLLM
tau2_qwen/launch_agents.sh                    # Qwen3-4B executor servers, ports 8020.., one GPU each
tau2_qwen/launch_services.sh retail           # 4 tau2 services on 8140-8143
tau2_qwen/run_retail.sh

# τ²-bench, Ministral-3-8B: the executor runs on Bedrock, so the services use no GPU
tau2_ministral/launch_services.sh airline     # 4 services on 8131-8134
tau2_ministral/run_airline.sh

# ALFWorld
alfworld/launch_services.sh 4 200             # 4 services on 8160-8163
alfworld/run_alfworld.sh
```

- **Each service serves one domain**, fixed at start-up by `--domain`. The launchers refuse to
  train against a service that reports another domain.
- **Stopping services:** use `kill -9 <pid>`, because they ignore SIGTERM. A service left behind
  keeps calling Bedrock.
- **ALFWorld services:** they grow in memory over long runs, so restart them before every launch.
- **Hardware the runs used:**
  - τ²-bench: 8×H100. verl ran on 1 GPU (Qwen) or 2 GPUs (Ministral, TP 2), and the Qwen
    executor servers used the remaining cards.
  - ALFWorld: 2×32 GB GPUs, with actor offload.
  - The GPU assignment is set by `VERL_GPUS` / `N_GPUS` / `TP`; the executor ports by `AGENT_PORTS`.
- **Outputs** go to `runs/<name>/`:
  - `checkpoints/` (LoRA plus full FSDP shards)
  - `validation/<step>.jsonl`
  - `rollouts/`
  - `reward_calls.jsonl` (one line per reward call)
  - `train.log`

  Evaluate the base model plus the adapter, not a bf16-merged model (see `../verl/README.md`).

## Evaluating a released adapter

Any training script evaluates instead of training when it is given an adapter and both validation
switches. The adapter is loaded into the LoRA before the first validation, and the run exits after it:

```bash
VAL_ONLY=True VAL_BEFORE=True tau2_ministral/run_airline.sh \
    actor_rollout_ref.model.lora_adapter_path=<repo>/checkpoints/tau2_ministral_airline
```

The services for the domain must be up, as for training. Without `lora_adapter_path` the same
command evaluates the untrained selector (a fresh LoRA). Per-task results are written to
`runs/<name>/validation/`, with one row per test task; its reward is the mean success over the
task's trials.

## Recipe

| | Qwen3-4B | Ministral-3-8B | ALFWorld |
|---|---|---|---|
| LoRA | r 64, α 128, all linear layers | same | same |
| learning rate | 2e-5 | 1e-5 | 1e-5 |
| FSDP | `fsdp` (with peft 0.19.1) | `fsdp2` | `fsdp2`, fp32 |
| batch | 8 groups × 8 rollouts; 16 prompts drawn per round | same | same |
| advantage | group mean only (no std normalisation), no KL term | same | same |
| dynamic sampling | drop groups whose task reward does not vary | same | same |
| reward per rollout | mean success over 8 simulations | same | same |
| prompt / response | 16384 / 1024 tokens | 24576 / 1024 | 16384 / 1024 |
| rollout sampling | temperature 1.0, top_p 0.9, top_k 20 | same | same |
| validation | greedy decode; 4 τ²-bench trials per task, tau2's pass^1 | same | 1 attempt per game |

**Reward.** A rollout's reward is its task success minus two penalties:

| penalty | charged when | Qwen telecom | everything else |
|---|---|---|---|
| no-reasoning | the reply has no `<reasoning>` block | 0.2 | 0.05 |
| k penalty | per selected case beyond `K_MAX` | 0.02 per case beyond 7 | off |

Both penalties are charged on every rollout. A reply without a parseable `<answer>` scores 0 and
is not simulated. The executor may take at most `MEMSEL_TRAIN_MAX_STEPS` steps per training
simulation: 50 on τ²-bench, except 70 for Qwen telecom (telecom conversations are about twice as
long as retail's), and 20 in ALFWorld.

## Data format

Each parquet row is one MemComposer prompt. The fields are:

| field | content |
|---|---|
| `prompt` | the chat messages: the query plus the k retrieved candidates, numbered `[1]..[k]`, each as its `digest.txt` |
| `data_source` | `memsel/<domain>`; the reward routes the row to that domain's services |
| `extra_info.task_id` | the task (τ²-bench task id or ALFWorld game id) |
| `extra_info.opening` | the customer's opening request (τ²-bench) or the game's goal and starting room (ALFWorld) |
| `extra_info.folders` | the bank folder of each candidate, in prompt order; the reply's numbers index into this list |
| `extra_info.n_trials` | simulations per scoring; `> 1` or `is_eval = 1` marks an eval row |
| `extra_info.window`, `rank_offset`, `k` | which retrieval ranks the candidates are |
| `extra_info.template`, `thinking` | which MemComposer prompt rendered the row (`qwen`, `ministral` or `alfworld`; see `memcomposer/tau2/memcomposer_prompt.py` and `memcomposer/alfworld/memcomposer_prompt.py`), and whether the reply must reason first |

How the data was built:
- **Queries.** τ²-bench training queries are 4 sampled customer openings per training task. Eval
  uses one fixed opening per test task. ALFWorld uses 256 training games from distinct taskdirs,
  and the 133 `valid_unseen` games for eval.
- **Candidates.** Training rows use retrieval ranks 1–10 and 11–20 as two separate rows; eval rows
  use the top 10. A training query never sees memories from its own task (its own taskdir in
  ALFWorld). Each source task contributes at most 2 candidates.
- **Task filtering (τ²-bench training sets only).** Two filters remove tasks that carry no
  training signal:
  1. tasks the executor already solves at least 7 times in 8 without memory;
  2. tasks whose reward does not change with the selection (within-group spread ≤ 0.125).

  Filter 1 is applied everywhere. Filter 2 is applied only to Qwen telecom and Ministral retail.
