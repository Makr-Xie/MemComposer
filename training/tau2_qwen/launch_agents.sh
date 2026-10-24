#!/usr/bin/env bash
# Qwen3-4B-Instruct-2507 executor servers (vLLM, one per GPU), called by the tau2 services.
#
#   AGENT_PORTS_LIST="8020 8022 8024 8026 8028" ./launch_agents.sh     # GPUs 0..4
#
# VLLM_PYTHON: interpreter of an environment with vLLM (default: python). Agent temperature is set
# per request by the tau2 services (0 for training and eval).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
V="${VLLM_PYTHON:-python}"
M="${AGENT_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
LOGDIR="${LOGDIR:-$HERE/../../runs/logs}"
mkdir -p "$LOGDIR"
PORTS="${AGENT_PORTS_LIST:-8020 8022 8024 8026 8028}"
i=0
for p in $PORTS; do
  g=$i
  CUDA_VISIBLE_DEVICES=$g nohup $V -m vllm.entrypoints.openai.api_server \
    --model $M --served-model-name Qwen3-4B-Instruct-2507 \
    --host 127.0.0.1 --port $p --max-model-len 65536 --gpu-memory-utilization ${GPU_MEM_UTIL:-0.92} \
    --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser hermes \
    > "$LOGDIR/agent${p}.log" 2>&1 &
  echo "agent $p -> GPU $g pid $!"; i=$((i+1))
done
for n in $(seq 1 90); do ok=1; for p in $PORTS; do curl -sf -m 3 http://127.0.0.1:$p/v1/models >/dev/null || ok=0; done; [ $ok = 1 ] && break; sleep 10; done
for p in $PORTS; do curl -sf -m 3 http://127.0.0.1:$p/v1/models >/dev/null && echo "  $p OK" || echo "  $p DOWN"; done
