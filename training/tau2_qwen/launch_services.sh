#!/usr/bin/env bash
# tau2 simulation services for one domain, fed by the local Qwen3-4B executor servers.
#
#   ./launch_services.sh retail      # 4 services on 8140-8143
#   ./launch_services.sh telecom     # 4 services on 8150-8153
#
# The domain given here is the ONLY thing that decides which domain gets graded; read the health
# line printed at the end. TAU2_PYTHON must be an environment where tau2-bench is installed;
# TAU2_ROOT points at the tau2-bench checkout whose .env holds the user simulator's credentials.
# The memory bank is MEMSEL_MEMORY_BASE (default <data>/tau2/memory_banks/qwen).
#
# MEMSEL_USER_CONC is the user-simulator connection pool per service and usually the throughput
# limit. MEMSEL_JOIN_TIMEOUT is how long a deduplicated request waits for the identical one in
# flight; telecom conversations are about twice as long as retail's, so its default is doubled.
# Stop the services with kill -9 by pid: they ignore SIGTERM.
set -euo pipefail
DOMAIN="${1:?usage: $0 <retail|telecom|airline>}"
HERE=$(cd "$(dirname "$0")" && pwd)
SVC_DIR="$HERE/../../memcomposer/tau2"
TAU2="${TAU2_PYTHON:-python}"
case "$DOMAIN" in
  airline) BASE_PORT=8131; JT=420 ;;
  retail)  BASE_PORT=8140; JT=420 ;;
  telecom) BASE_PORT=8150; JT=840 ;;
  *) echo "unknown domain $DOMAIN"; exit 1 ;;
esac
PORTS="${SERVICE_PORTS:-$BASE_PORT $((BASE_PORT+1)) $((BASE_PORT+2)) $((BASE_PORT+3))}"
AGENTS="${AGENT_PORTS_LIST:-8020 8022 8024 8026 8028 8030 8032}"
BASES=""
for p in $AGENTS; do BASES="${BASES:+$BASES,}http://127.0.0.1:$p/v1"; done
for p in $AGENTS; do curl -sf -m 5 http://127.0.0.1:$p/v1/models >/dev/null || { echo "agent $p down"; exit 1; }; done
for p in $PORTS; do curl -sf -m 2 http://127.0.0.1:$p/health >/dev/null 2>&1 && { echo "port $p busy"; exit 1; }; done
LOGDIR="${LOGDIR:-$HERE/../../runs/logs}"
mkdir -p "$LOGDIR"
cd "$SVC_DIR"
for p in $PORTS; do
  MEMSEL_AGENT_BASES="$BASES" MEMSEL_AGENT_CONC_PER_CARD="${MEMSEL_AGENT_CONC_PER_CARD:-6}" \
  MEMSEL_USER_CONC="${MEMSEL_USER_CONC:-64}" MEMSEL_JOIN_TIMEOUT="${MEMSEL_JOIN_TIMEOUT:-$JT}" \
    nohup $TAU2 tau2_service.py --port $p --domain $DOMAIN --workers 200 > "$LOGDIR/svc${p}_${DOMAIN}.log" 2>&1 &
  echo "launched $DOMAIN service on $p pid $!"
done
for i in $(seq 1 60); do ok=1; for p in $PORTS; do curl -sf -m 2 http://127.0.0.1:$p/health >/dev/null || ok=0; done; [ $ok = 1 ] && break; sleep 3; done
for p in $PORTS; do echo "$p: $(curl -s -m 10 http://127.0.0.1:$p/health | tr ',' '\n' | sed 's/[{}"]//g' | grep -E 'domain|tasks|memories' | tr '\n' ' ')"; done
