#!/usr/bin/env bash
# MemComposer on ALFWorld, Ministral-3-8B-Instruct selector and executor.
# Same recipe as the tau2-bench Ministral runs. ALFWorld-specific: 256 training games from distinct
# taskdirs x 2 rank windows = 512 rows (32 steps per epoch); validation is valid_unseen, 133 games,
# one attempt each; the environment's step limit is 20.
# Hardware as run: verl on 2 x 32 GB GPUs (TP 2); the executor on Bedrock; four ALFWorld services on
# 8160-8163 (launch_services.sh). fp32 training on 2 x 32 GB needs ACTOR_OFFLOAD=True (the default);
# without it the run runs out of memory at step 2.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DATA="${MEMCOMPOSER_DATA:-$(cd "$HERE/../.." && pwd)/data}"

export VERL_GPUS="${VERL_GPUS:-0,1}"
export N_GPUS="${N_GPUS:-2}"
export TP="${TP:-2}"

export TRAIN_FILES="${TRAIN_FILES:-$DATA/training/alfworld/ministral/train.parquet}"
export VAL_FILES="${VAL_FILES:-$DATA/training/alfworld/ministral/val.parquet}"
export MEMSEL_TAU2_URLS="${MEMSEL_TAU2_URLS:-memsel/alfworld=http://127.0.0.1:8160|http://127.0.0.1:8161|http://127.0.0.1:8162|http://127.0.0.1:8163}"

export LR="${LR:-1e-5}"
export LORA_RANK="${LORA_RANK:-64}"
export LORA_ALPHA="${LORA_ALPHA:-128}"
export USE_KL_LOSS="${USE_KL_LOSS:-False}"
export TRAIN_TRIALS="${TRAIN_TRIALS:-8}"
export MEMSEL_K_PENALTY="${MEMSEL_K_PENALTY:-0}"
export K_MAX="${K_MAX:-0}"
export MEMSEL_PEN_NO_REASONING="${MEMSEL_PEN_NO_REASONING:-0.05}"
export MEMSEL_FMT_PEN_ALWAYS="${MEMSEL_FMT_PEN_ALWAYS:-1}"
export MEMSEL_TRAIN_MAX_STEPS="${MEMSEL_TRAIN_MAX_STEPS:-20}"
export MEMSEL_REWARD_CONCURRENCY="${MEMSEL_REWARD_CONCURRENCY:-480}"
export BATCH="${BATCH:-8}"
export GEN_BATCH="${GEN_BATCH:-16}"
export GROUP="${GROUP:-8}"
export STEPS="${STEPS:-30}"
export TEST_FREQ="${TEST_FREQ:-5}"
export SAVE_FREQ="${SAVE_FREQ:-5}"
export DAPO="${DAPO:-1}"
export FILTER_METRIC="${FILTER_METRIC:-task_reward}"
export VAL_N="${VAL_N:-1}"
export VAL_BEFORE="${VAL_BEFORE:-True}"
export MAX_PROMPT="${MAX_PROMPT:-16384}"
export MAX_RESP="${MAX_RESP:-1024}"
export ACTOR_DTYPE="${ACTOR_DTYPE:-fp32}"
export ACTOR_OFFLOAD="${ACTOR_OFFLOAD:-True}"
export ROLLOUT_GPU_FRAC="${ROLLOUT_GPU_FRAC:-0.4}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
export TOP_P="${TOP_P:-0.9}"
export TOP_K="${TOP_K:-20}"

export WANDB="${WANDB:-1}"
export RUN_DIR="${RUN_DIR:-$HERE/../../runs/alfworld_ministral}"
exec "$HERE/run_train.sh" "$@"
