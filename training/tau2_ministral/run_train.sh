#!/usr/bin/env bash
# ExPO training of the MemComposer selector on tau2-bench, Ministral-3-8B-Instruct (LoRA, verl, FSDP2).
# The per-domain scripts (run_retail.sh, run_airline.sh, run_telecom.sh) set the hyperparameters and
# call this one. Before launching, the tau2 services for the domain must be up (launch_services.sh).
#
# Differences from the Qwen3-4B launcher (../tau2_qwen/run_train.sh):
#   * strategy=fsdp2 is REQUIRED. With FSDP1 and peft >= 0.21 the LoRA sync collects zero tensors
#     and vLLM keeps serving the base model, silently (see verl/README.md).
#   * Group filtering is built into verl.trainer.main_ppo (no recipe/).
#   * max_prompt_length 24576: Ministral's memory bank renders longer cases, and the longest eval
#     prompt is ~20.5k tokens. With a smaller limit filter_overlong_prompts drops eval tasks
#     silently.
#   * The model must be the text-only conversion with the fixes in verl/README.md applied
#     (convert_ministral_text_only.py).
#   * The executor runs on Bedrock, so the tau2 services need no local GPUs.
#   * Do not set PYTORCH_CUDA_ALLOC_CONF=expandable_segments with TP > 1: vLLM's custom all-reduce
#     cannot export that memory over CUDA IPC.
#   * Do not set reward.reward_manager.compute_score_timeout to a small value: one reward call runs
#     8 tau2 simulations (median ~630 s, max ~1430 s), and a timeout silently scores 0.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
[ -n "${VENV:-}" ] && export PATH="$VENV/bin:$PATH"

export CUDA_VISIBLE_DEVICES="${VERL_GPUS:-0,1}"
if [ "${WANDB:-1}" = "1" ]; then LOGGER='[console,wandb]'; else LOGGER='[console]'; export WANDB_MODE=disabled; fi
export RAY_USAGE_STATS_ENABLED=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export HYDRA_FULL_ERROR=1 VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
unset PYTORCH_CUDA_ALLOC_CONF || true

RUN_DIR="${RUN_DIR:-$REPO/runs/train_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"
export MEMSEL_REWARD_LOG="$RUN_DIR/reward_calls.jsonl"

# ---- reward (memcomposer/reward.py reads these) ----
export MEMSEL_K_PENALTY="${MEMSEL_K_PENALTY:-0}"
if [ "${MEMSEL_K_PENALTY:-0}" = "0" ] || [ "${MEMSEL_K_PENALTY:-0}" = "0.0" ]; then
  export MEMSEL_K_MAX="${K_MAX:-0}"
else
  export MEMSEL_K_MAX="${K_MAX:-4}"
fi
export MEMSEL_K_PEN_ALWAYS="${K_PEN_ALWAYS:-0}"
export MEMSEL_PEN_NO_REASONING="${MEMSEL_PEN_NO_REASONING:-0.05}"
export MEMSEL_FMT_PEN_ALWAYS="${MEMSEL_FMT_PEN_ALWAYS:-1}"
export MEMSEL_TRAIN_TRIALS="${TRAIN_TRIALS:-8}"
export MEMSEL_REWARD_CONCURRENCY="${MEMSEL_REWARD_CONCURRENCY:-480}"
export MEMSEL_TRAIN_MAX_STEPS="${MEMSEL_TRAIN_MAX_STEPS:-0}"

# ---- preflight: every tau2 service must be up and serve the domain its data_source names ----
if [ -n "${MEMSEL_TAU2_URLS:-}" ]; then
  export MEMSEL_TAU2_URLS
  IFS=',' read -ra _PAIRS <<< "$MEMSEL_TAU2_URLS"
  for _p in "${_PAIRS[@]}"; do
    _k="${_p%%=*}"; _want="${_k##*/}"
    IFS='|' read -ra _US <<< "${_p#*=}"
    for _u in "${_US[@]}"; do
      _h=$(curl -sf -m 15 "$_u/health") || { echo "tau2_service ($_u) for $_k is not up"; exit 1; }
      _got=$(printf '%s' "$_h" | sed -n 's/.*"domain":"\([a-z]*\)".*/\1/p')
      [ "$_got" = "$_want" ] || { echo "DOMAIN MISMATCH: $_u serves '$_got' but $_k needs '$_want'"; exit 1; }
      echo "tau2_service $_u  domain $_got  OK  ($_k)"
    done
  done
fi

MAXP="${MAX_PROMPT:-24576}"
MAXR="${MAX_RESP:-1024}"
echo "=== MemComposer / tau2-bench / Ministral-3-8B ==="
echo "  verl      ${VERL_DIR:-$REPO/verl}"
echo "  model     ${MODEL_PATH:-$REPO/models/Ministral-3-8B-Instruct-2512-text}"
echo "  GPU       ${VERL_GPUS:-0,1}  (n_gpus=${N_GPUS:-2}, TP=${TP:-2})"
echo "  prompt    $MAXP   response $MAXR"
echo "  lora      r=${LORA_RANK:-64} alpha=${LORA_ALPHA:-128}   lr=${LR:-1e-5}"
echo "  run dir   $RUN_DIR"

cd "${VERL_DIR:-$REPO/verl}"
python -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo="${NORM_ADV_BY_STD:-False}" \
    algorithm.use_kl_in_reward=False \
    algorithm.filter_groups.enable="${FILTER_ENABLE:-True}" \
    algorithm.filter_groups.metric="${FILTER_METRIC:-task_reward}" \
    algorithm.filter_groups.max_num_gen_batches="${MAX_GEN_BATCHES:-0}" \
    reward.reward_manager.name=naive \
    data.gen_batch_size="${GEN_BATCH:-16}" \
    data.train_files="${TRAIN_FILES:?set TRAIN_FILES}" \
    data.val_files="${VAL_FILES:?set VAL_FILES}" \
    data.train_batch_size="${BATCH:-8}" \
    data.max_prompt_length="$MAXP" \
    data.max_response_length="$MAXR" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH:-$REPO/models/Ministral-3-8B-Instruct-2512-text}" \
    actor_rollout_ref.model.use_remove_padding="${REMOVE_PADDING:-True}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank="${LORA_RANK:-64}" \
    actor_rollout_ref.model.lora_alpha="${LORA_ALPHA:-128}" \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr="${LR:-1e-5}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${BATCH:-8}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BSZ:-1}" \
    actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS:-False}" \
    actor_rollout_ref.actor.kl_loss_coef="${KL_COEF:-0.01}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload="${REF_OFFLOAD:-True}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TP:-2}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_FRAC:-0.4}" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.max_model_len=$(( MAXP + MAXR )) \
    actor_rollout_ref.rollout.max_num_batched_tokens=$(( MAXP + MAXR )) \
    actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS:-16}" \
    actor_rollout_ref.rollout.n="${GROUP:-8}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p="${TOP_P:-0.9}" \
    actor_rollout_ref.rollout.top_k="${TOP_K:-20}" \
    actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMP:-0}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample="${VAL_SAMPLE:-False}" \
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_N:-1}" \
    actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_TOP_P:-1.0}" \
    actor_rollout_ref.rollout.val_kwargs.top_k="${VAL_TOP_K:--1}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${MICRO_BSZ:-1}" \
    actor_rollout_ref.rollout.agent.num_workers="${AGENT_WORKERS:-4}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${MICRO_BSZ:-1}" \
    reward.custom_reward_function.path="$REPO/memcomposer/reward.py" \
    reward.custom_reward_function.name=compute_score \
    reward.num_workers="${REWARD_WORKERS:-2}" \
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
    "$@" 2>&1 | tee -a "$RUN_DIR/train.log"
