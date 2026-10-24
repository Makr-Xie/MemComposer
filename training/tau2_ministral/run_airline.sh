#!/usr/bin/env bash
# MemComposer on tau2-bench airline, Ministral-3-8B-Instruct selector and executor.
# Hardware as run: verl on 2 GPUs (TP 2); the executor on Bedrock; four airline tau2 services
# (launch_services.sh airline).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DATA="${MEMCOMPOSER_DATA:-$(cd "$HERE/../.." && pwd)/data}"

export VERL_GPUS="${VERL_GPUS:-6,7}"
export N_GPUS="${N_GPUS:-2}"
export TP="${TP:-2}"
export MEMSEL_DOMAIN=airline

export TRAIN_FILES="${TRAIN_FILES:-$DATA/training/tau2/ministral/airline/train.parquet}"
export VAL_FILES="${VAL_FILES:-$DATA/training/tau2/ministral/airline/val.parquet}"
export MEMSEL_TAU2_URLS="${MEMSEL_TAU2_URLS:-memsel/airline=http://127.0.0.1:8131|http://127.0.0.1:8132|http://127.0.0.1:8133|http://127.0.0.1:8134}"

export LR="${LR:-1e-5}"
export LORA_RANK="${LORA_RANK:-64}"
export LORA_ALPHA="${LORA_ALPHA:-128}"
export USE_KL_LOSS="${USE_KL_LOSS:-False}"
export TRAIN_TRIALS="${TRAIN_TRIALS:-8}"
export MEMSEL_K_PENALTY="${MEMSEL_K_PENALTY:-0}"
export K_MAX="${K_MAX:-0}"
export MEMSEL_K_MIN_PENALTY="${MEMSEL_K_MIN_PENALTY:-0}"
export MEMSEL_PEN_NO_REASONING="${MEMSEL_PEN_NO_REASONING:-0.05}"
export MEMSEL_FMT_PEN_ALWAYS="${MEMSEL_FMT_PEN_ALWAYS:-1}"
export MEMSEL_TRAIN_MAX_STEPS="${MEMSEL_TRAIN_MAX_STEPS:-50}"
export MEMSEL_REWARD_CONCURRENCY="${MEMSEL_REWARD_CONCURRENCY:-480}"
export BATCH="${BATCH:-8}"
export GEN_BATCH="${GEN_BATCH:-16}"
export GROUP="${GROUP:-8}"
export STEPS="${STEPS:-20}"
export TEST_FREQ="${TEST_FREQ:-5}"
export SAVE_FREQ="${SAVE_FREQ:-5}"
export DAPO="${DAPO:-1}"
export FILTER_METRIC="${FILTER_METRIC:-task_reward}"
export VAL_N="${VAL_N:-1}"
export VAL_BEFORE="${VAL_BEFORE:-False}"
export MAX_PROMPT="${MAX_PROMPT:-24576}"
export MAX_RESP="${MAX_RESP:-1024}"

export WANDB="${WANDB:-1}"
export RUN_DIR="${RUN_DIR:-$HERE/../../runs/tau2_ministral_airline}"
exec "$HERE/run_train.sh" "$@"
