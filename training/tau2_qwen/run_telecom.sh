#!/usr/bin/env bash
# MemComposer on tau2-bench telecom, Qwen3-4B-Instruct-2507 selector and executor.
# Hardware as run: verl on 1 GPU, the Qwen3-4B executor served on the other cards (launch_agents.sh),
# four telecom tau2 services on 8150-8153 (launch_services.sh telecom).
#
# Differences from retail: k penalty on (0.02 per case beyond K_MAX=7, charged on every rollout),
# no-reasoning penalty 0.2, MEMSEL_TRAIN_MAX_STEPS 70 (telecom conversations are about twice as
# long as retail's). A no-reasoning penalty this strong can collapse the reasoning and push k toward
# 1-2; watch has_reasoning and the mean k.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DATA="${MEMCOMPOSER_DATA:-$(cd "$HERE/../.." && pwd)/data}"

export VERL_GPUS="${VERL_GPUS:-7}"
export N_GPUS="${N_GPUS:-1}"
export AGENT_PORTS="${AGENT_PORTS:-8020 8022 8024 8026 8028}"
export MEMSEL_DOMAIN=telecom

export TRAIN_FILES="${TRAIN_FILES:-$DATA/training/tau2/qwen/telecom/train.parquet}"
export VAL_FILES="${VAL_FILES:-$DATA/training/tau2/qwen/telecom/val.parquet}"
export MEMSEL_TAU2_URLS="${MEMSEL_TAU2_URLS:-memsel/telecom=http://127.0.0.1:8150|http://127.0.0.1:8151|http://127.0.0.1:8152|http://127.0.0.1:8153}"

export LR="${LR:-2e-5}"
export USE_KL_LOSS="${USE_KL_LOSS:-False}"
export TRAIN_TRIALS="${TRAIN_TRIALS:-8}"
export MEMSEL_K_PENALTY="${MEMSEL_K_PENALTY:-0.02}"
export K_PEN_ALWAYS="${K_PEN_ALWAYS:-1}"
export K_MAX="${K_MAX:-7}"
export MEMSEL_TRAIN_MAX_STEPS="${MEMSEL_TRAIN_MAX_STEPS:-70}"
export MEMSEL_REWARD_CONCURRENCY="${MEMSEL_REWARD_CONCURRENCY:-480}"
export BATCH="${BATCH:-8}"
export GEN_BATCH="${GEN_BATCH:-16}"
export STEPS="${STEPS:-30}"
export TEST_FREQ="${TEST_FREQ:-5}"
export SAVE_FREQ="${SAVE_FREQ:-5}"
export DAPO="${DAPO:-1}"                                     # K_PEN_ALWAYS needs DAPO filtering on task_reward
export VAL_N="${VAL_N:-1}"
export VAL_BEFORE="${VAL_BEFORE:-False}"
export MEMSEL_PEN_NO_REASONING="${MEMSEL_PEN_NO_REASONING:-0.2}"
export MEMSEL_FMT_PEN_ALWAYS="${MEMSEL_FMT_PEN_ALWAYS:-1}"
export FILTER_METRIC="${FILTER_METRIC:-task_reward}"

export WANDB="${WANDB:-1}"
export RUN_DIR="${RUN_DIR:-$HERE/../../runs/tau2_qwen_telecom}"
exec "$HERE/run_train.sh" "$@"
