#!/usr/bin/env bash
# ALFWorld simulation services: N processes, each with its own textworld lock and libdownward pool.
# The executor is Ministral-3-8B on Bedrock (BEDROCK_BASE_URL / BEDROCK_API_KEY); no GPUs.
#
#   ./launch_services.sh <n_services> <workers_per_service> [first_port=8160]      # as run: 4 200
#
# ALFWORLD_PYTHON: an environment with alfworld/textworld and fastapi. ALFW_TASK_INDEX: the task
# index (memory_bank/alfworld/build_task_index.py). The memory bank is MEMSEL_MEMORY_BASE (default
# <data>/alfworld/memory_banks/ministral8b).
# The services grow in RAM over long runs; restart them before every launch.
# Stop with kill -9 by pid, then check the ports are free (ss -ltn).
set -euo pipefail
N="${1:?usage: $0 <n_services> <workers_per_service> [first_port]}"
W="${2:?usage: $0 <n_services> <workers_per_service> [first_port]}"
BASE_PORT="${3:-8160}"
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
DATA="${MEMCOMPOSER_DATA:-$REPO/data}"
PY="${ALFWORLD_PYTHON:-python}"
MEMBASE="${MEMSEL_MEMORY_BASE:-$DATA/alfworld/memory_banks/ministral8b}"
LOGDIR="${LOGDIR:-$REPO/runs/logs}"
mkdir -p "$LOGDIR"

PORTS=""; for i in $(seq 0 $((N-1))); do PORTS="$PORTS $((BASE_PORT+i))"; done
for p in $PORTS; do
  curl -sf -m 2 "http://127.0.0.1:$p/health" >/dev/null 2>&1 && { echo "  port $p is busy"; exit 1; }
done
echo "=== ALFWorld services ==="
echo "  ports   $PORTS   workers per service $W"
echo "  memory   $MEMBASE"
echo "  executor ${ALFW_AGENT_MODEL:-mistral.ministral-3-8b-instruct}  temp ${ALFW_AGENT_TEMP:-0.4}  max_steps ${ALFW_MAX_STEPS:-20}  llm timeout ${ALFW_LLM_TIMEOUT:-600}s"
cd "$REPO/memcomposer/alfworld"
for p in $PORTS; do
  MEMSEL_MEMORY_BASE="$MEMBASE" \
  ALFW_LLM_TIMEOUT="${ALFW_LLM_TIMEOUT:-600}" \
  ALFW_SIM_LOG="$LOGDIR/sims_$p.jsonl" \
  MEMSEL_JOIN_TIMEOUT="${MEMSEL_JOIN_TIMEOUT:-900}" \
    setsid nohup "$PY" alfworld_service.py --port "$p" --workers "$W" > "$LOGDIR/svc$p.log" 2>&1 < /dev/null &
  echo "  launched $p  pid $!"
done
for i in $(seq 1 60); do
  ok=1; for p in $PORTS; do curl -sf -m 2 "http://127.0.0.1:$p/health" >/dev/null || ok=0; done
  [ $ok = 1 ] && break; sleep 2
done
for p in $PORTS; do
  echo "  $p: $(curl -s -m 10 "http://127.0.0.1:$p/health" | tr ',' '\n' | sed 's/[{}\"]//g' | grep -E '^(ok|domain|tasks|memories):' | tr '\n' ' ')"
done
