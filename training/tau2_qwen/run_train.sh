#!/usr/bin/env bash
# ExPO training of the MemComposer selector on tau2-bench, Qwen3-4B-Instruct-2507 (LoRA, verl, FSDP1).
# The per-domain scripts (run_retail.sh, run_telecom.sh) set the hyperparameters and call this one.
#
# Before launching: the tau2 services for the domain must be up (launch_services.sh), and so must the
# Qwen3-4B executor servers they call (launch_agents.sh). Each service is bound to one domain at
# start-up, and the check below refuses to train against a service serving another domain.
#
#   * LoRA r=64 / alpha=128 on all linear layers. The ref model is the actor with the adapter off.
#   * Prompt <= 16384 tokens, response <= MAX_RESP (1024). data.truncation=error makes an overlong
#     prompt a crash; an overlong response is clipped and scored 0 by the <answer> gate, so watch
#     response_length/clip_ratio.
#   * Rollouts: temperature 1.0, top_p 0.9, top_k 20 (the model's own generation config).
#     Validation: greedy, one decode per task, each scored on the row's own 4 tau2 trials
#     (tau2 pass^1, no penalties).
#   * GRPO without std normalisation (Dr.GRPO), no KL term, DAPO dynamic sampling (DAPO=1): drop
#     groups whose FILTER_METRIC has zero variance and keep generating until BATCH informative
#     groups are in hand. Requires verl's recipe/ (recipe.dapo.main_dapo).
#   * Checkpoints must include `model`; with '[optimizer,extra]' only Adam state is saved and the
#     adapter cannot be recovered.
#
# Env knobs: BATCH GROUP GEN_BATCH STEPS LR TRAIN_FILES VAL_FILES MODEL_PATH VERL_DIR VENV
#            VERL_GPUS N_GPUS ROLLOUT_GPU_FRAC AGENT_PORTS TOP_P TOP_K MAX_RESP USE_KL_LOSS KL_COEF
#            TRAIN_TRIALS DAPO FILTER_METRIC MAX_GEN_BATCHES NORM_ADV_BY_STD REMOVE_PADDING ATTN_IMPL
#            VAL_N VAL_TEMP VAL_SAMPLE VAL_ONLY VAL_BEFORE TEST_FREQ SAVE_FREQ RESUME_MODE WANDB RUN_DIR
# VERL_GPUS (which cards) and N_GPUS (how many verl shards over) must agree.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
[ -n "${VENV:-}" ] && export PATH="$VENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${VERL_GPUS:-6,7}"
if [ "${WANDB:-1}" = "1" ]; then LOGGER='[console,wandb]'; else LOGGER='[console]'; export WANDB_MODE=disabled; fi
export RAY_USAGE_STATS_ENABLED=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export HYDRA_FULL_ERROR=1 VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn

RUN_DIR="${RUN_DIR:-$REPO/runs/train_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"
export MEMSEL_REWARD_LOG="$RUN_DIR/reward_calls.jsonl"

# ---- reward (memcomposer/reward.py reads these) ----
# k penalty: each selected case beyond K_MAX costs MEMSEL_K_PENALTY. With the penalty on, K_MAX
# defaults to 4; with it off, to 0.
export MEMSEL_K_PENALTY="${MEMSEL_K_PENALTY:-0.005}"
if [ "${MEMSEL_K_PENALTY:-0}" = "0" ] || [ "${MEMSEL_K_PENALTY:-0}" = "0.0" ]; then
  export MEMSEL_K_MAX="${K_MAX:-0}"
else
  export MEMSEL_K_MAX="${K_MAX:-4}"
fi
# Charge the k penalty at every task_reward level, not only on solved rollouts. Needs DAPO with
# FILTER_METRIC=task_reward.
export MEMSEL_K_PEN_ALWAYS="${K_PEN_ALWAYS:-0}"
# Cost of answering without a <reasoning> block. (A reply without <answer> scores 0, unsimulated.)
export MEMSEL_PEN_NO_REASONING="${MEMSEL_PEN_NO_REASONING:-0.05}"
# Simulations kept in flight by the reward function; keep it matched to the services' --workers.
export MEMSEL_REWARD_CONCURRENCY="${MEMSEL_REWARD_CONCURRENCY:-96}"
# A training rollout's task reward is the mean over this many tau2 simulations (pass^1).
# Eval rows keep the trial count their parquet was built with.
export MEMSEL_TRAIN_TRIALS="${TRAIN_TRIALS:-8}"
echo "run dir: $RUN_DIR"

# ---- preflight: every tau2 service must be up and serve the domain this run trains ----
TAU2_PORT="${TAU2_PORT:-8100}"
export MEMSEL_TAU2_URL="${MEMSEL_TAU2_URL:-http://127.0.0.1:$TAU2_PORT}"
if [ -n "${MEMSEL_TAU2_URLS:-}" ]; then
  export MEMSEL_TAU2_URLS
  IFS=',' read -ra _PAIRS <<< "$MEMSEL_TAU2_URLS"
  for _p in "${_PAIRS[@]}"; do
    _k="${_p%%=*}"; _want="${_k##*/}"
    # a data_source may list several services, '|'-separated (the reward hash-routes across them)
    IFS='|' read -ra _US <<< "${_p#*=}"
    for _u in "${_US[@]}"; do
      _h=$(curl -sf -m 15 "$_u/health") || { echo "tau2_service ($_u) for $_k is not up"; exit 1; }
      _got=$(printf '%s' "$_h" | sed -n 's/.*"domain":"\([a-z]*\)".*/\1/p')
      [ "$_got" = "$_want" ] || { echo "DOMAIN MISMATCH: $_u serves '$_got' but $_k needs '$_want'"; exit 1; }
      echo "tau2_service $_u  domain $_got  OK  ($_k)"
    done
  done
else
_health=$(curl -sf -m 15 "$MEMSEL_TAU2_URL/health") || { echo "tau2_service ($MEMSEL_TAU2_URL) is not up"; exit 1; }
_svc_dom=$(printf '%s' "$_health" | sed -n 's/.*"domain":"\([a-z]*\)".*/\1/p')
_want_dom="${MEMSEL_DOMAIN:?set MEMSEL_DOMAIN}"
[ "$_svc_dom" = "$_want_dom" ] || { echo "DOMAIN MISMATCH: service at $MEMSEL_TAU2_URL serves '$_svc_dom' but this run trains '$_want_dom'"; exit 1; }
echo "tau2_service $MEMSEL_TAU2_URL  domain $_svc_dom  OK"
fi
for p in ${AGENT_PORTS:-8020 8022 8024 8026}; do curl -sf -m 5 http://127.0.0.1:$p/v1/models >/dev/null || { echo "agent server $p is not up"; exit 1; }; done

# DAPO dynamic sampling: draw GEN_BATCH prompts per round, drop zero-variance groups, repeat until
# BATCH informative groups. reward_manager=naive (the recipe's 'dapo' manager adds an overlong
# penalty we do not want). MAX_GEN_BATCHES=0 = unlimited.
MODULE=verl.trainer.main_ppo; DAPO_ARGS=()
if [ "${DAPO:-0}" = "1" ]; then
    MODULE=recipe.dapo.main_dapo
    DAPO_ARGS=(data.gen_batch_size="${GEN_BATCH:-16}"
               algorithm.filter_groups.enable=True
               algorithm.filter_groups.metric="${FILTER_METRIC:-score}"
               algorithm.filter_groups.max_num_gen_batches="${MAX_GEN_BATCHES:-0}"
               reward.reward_manager.name=naive)
fi
cd "${VERL_DIR:-$REPO/verl}"
python -u -m "$MODULE" "${DAPO_ARGS[@]}" \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo="${NORM_ADV_BY_STD:-False}" \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILES:?set TRAIN_FILES}" \
    data.val_files="${VAL_FILES:?set VAL_FILES}" \
    data.train_batch_size="${BATCH:-8}" \
    data.max_prompt_length=16384 \
    data.max_response_length="${MAX_RESP:-1024}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}" \
    actor_rollout_ref.model.use_remove_padding="${REMOVE_PADDING:-True}" \
    +actor_rollout_ref.model.override_config.attn_implementation="${ATTN_IMPL:-flash_attention_2}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank="${LORA_RANK:-64}" \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr="${LR:-2e-5}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${BATCH:-8}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS:-False}" \
    actor_rollout_ref.actor.kl_loss_coef="${KL_COEF:-0.01}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_FRAC:-0.35}" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.max_model_len=$((16384 + ${MAX_RESP:-1024})) \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((16384 + ${MAX_RESP:-1024})) \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.n="${GROUP:-8}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p="${TOP_P:-0.9}" \
    actor_rollout_ref.rollout.top_k="${TOP_K:-20}" \
    actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMP:-0}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample="${VAL_SAMPLE:-False}" \
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_N:-1}" \
    actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_TOP_P:-1.0}" \
    actor_rollout_ref.rollout.val_kwargs.top_k="${VAL_TOP_K:--1}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward.custom_reward_function.path="$REPO/memcomposer/reward.py" \
    reward.custom_reward_function.name=compute_score \
    reward.num_workers=2 \
    trainer.logger="$LOGGER" \
    trainer.project_name="${WANDB_PROJECT:-MemComposer}" \
    trainer.experiment_name="$(basename "$RUN_DIR")" \
    trainer.n_gpus_per_node="${N_GPUS:-2}" \
    trainer.nnodes=1 \
    trainer.total_epochs=100 \
    trainer.total_training_steps="${STEPS:-1}" \
    trainer.val_only="${VAL_ONLY:-False}" \
    trainer.val_before_train="${VAL_BEFORE:-True}" \
    trainer.test_freq="${TEST_FREQ:-5}" \
    trainer.save_freq="${SAVE_FREQ:-5}" \
    trainer.resume_mode="${RESUME_MODE:-disable}" \
    actor_rollout_ref.actor.checkpoint.save_contents='[model,optimizer,extra]' \
    trainer.default_local_dir="$RUN_DIR/checkpoints" \
    trainer.validation_data_dir="$RUN_DIR/validation" \
    trainer.rollout_data_dir="$RUN_DIR/rollouts" \
    ray_kwargs.ray_init.num_cpus=32 \
    +ray_kwargs.ray_init.num_gpus="${N_GPUS:-2}" \
    +ray_kwargs.ray_init.address=local \
    +ray_kwargs.ray_init.include_dashboard=False \
    +ray_kwargs.ray_init.object_store_memory=4294967296 \
    +ray_kwargs.ray_init._temp_dir="/tmp/verl-memcomposer-$USER-$$" \
    hydra.run.dir="$RUN_DIR/hydra" \
    "$@" 2>&1 | tee -a "$RUN_DIR/train.log"   # append: a resume into the same RUN_DIR must not clobber the log
