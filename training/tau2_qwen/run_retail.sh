#!/usr/bin/env bash
# MemComposer on tau2-bench retail, Qwen3-4B-Instruct-2507 selector and executor.
# Hardware as run: verl on 1 GPU, the Qwen3-4B executor served on the other cards (launch_agents.sh),
# four retail tau2 services on 8140-8143 (launch_services.sh retail).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DATA="${MEMCOMPOSER_DATA:-$(cd "$HERE/../.." && pwd)/data}"

export VERL_GPUS="${VERL_GPUS:-7}"
export N_GPUS="${N_GPUS:-1}"
export AGENT_PORTS="${AGENT_PORTS:-8020 8022 8024 8026 8028}"
export MEMSEL_DOMAIN=retail

export TRAIN_FILES="${TRAIN_FILES:-$DATA/training/tau2/qwen/retail/train.parquet}"
export VAL_FILES="${VAL_FILES:-$DATA/training/tau2/qwen/retail/val.parquet}"
export MEMSEL_TAU2_URLS="${MEMSEL_TAU2_URLS:-memsel/retail=http://127.0.0.1:8140|http://127.0.0.1:8141|http://127.0.0.1:8142|http://127.0.0.1:8143}"

export LR="${LR:-2e-5}"
export USE_KL_LOSS="${USE_KL_LOSS:-False}"
export TRAIN_TRIALS="${TRAIN_TRIALS:-8}"
export K_MAX="${K_MAX:-0}"
export MEMSEL_K_PENALTY="${MEMSEL_K_PENALTY:-0}"
export MEMSEL_TRAIN_MAX_STEPS="${MEMSEL_TRAIN_MAX_STEPS:-50}"
export MEMSEL_REWARD_CONCURRENCY="${MEMSEL_REWARD_CONCURRENCY:-480}"
export BATCH="${BATCH:-8}"
export GEN_BATCH="${GEN_BATCH:-16}"
export STEPS="${STEPS:-30}"
export TEST_FREQ="${TEST_FREQ:-5}"
export SAVE_FREQ="${SAVE_FREQ:-5}"
export DAPO="${DAPO:-1}"
export VAL_N="${VAL_N:-1}"
export VAL_BEFORE="${VAL_BEFORE:-False}"
export MEMSEL_FMT_PEN_ALWAYS="${MEMSEL_FMT_PEN_ALWAYS:-1}"   # charge the no-reasoning penalty on every rollout
export FILTER_METRIC="${FILTER_METRIC:-task_reward}"         # DAPO filters groups on task success

export WANDB="${WANDB:-1}"
export RUN_DIR="${RUN_DIR:-$HERE/../../runs/tau2_qwen_retail}"
exec "$HERE/run_train.sh" "$@"
