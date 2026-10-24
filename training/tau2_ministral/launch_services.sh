#!/usr/bin/env bash
# tau2 simulation services for one domain, with the Ministral-3-8B executor on Bedrock (no GPUs).
#
#   ./launch_services.sh <airline|retail|telecom> [n_services]     # default 4
#   ports: airline 8131-, retail 8140-, telecom 8150-
#
# Needs BEDROCK_BASE_URL and BEDROCK_API_KEY in the environment (the executor and the GLM-4.7 user
# simulator both go through it; their connection pools are kept apart by api_key).
# TAU2_PYTHON: an environment with tau2-bench installed. TAU2_ROOT: the tau2-bench checkout.
# The memory bank is MEMSEL_MEMORY_BASE (default <data>/tau2/memory_banks/ministral); the
# Ministral parquets name folders of THIS bank, and a wrong bank fails every simulation.
# The domain given here is the ONLY thing that decides which domain gets graded.
# Stop the services with kill -9 by pid: they ignore SIGTERM.
set -euo pipefail
DOMAIN="${1:?usage: $0 <airline|retail|telecom> [n_services]}"
case "$DOMAIN" in airline|retail|telecom) ;; *) echo "domain must be airline/retail/telecom"; exit 1;; esac
N="${2:-4}"
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
DATA="${MEMCOMPOSER_DATA:-$REPO/data}"
TAU2="${TAU2_PYTHON:-python}"
case "$DOMAIN" in
  airline) BASE_PORT=8131 ;;
  retail)  BASE_PORT=8140 ;;
  telecom) BASE_PORT=8150 ;;
esac
PORTS=""; for i in $(seq 0 $((N-1))); do PORTS="$PORTS $((BASE_PORT+i))"; done

BEDROCK_BASE="${BEDROCK_BASE_URL:?set BEDROCK_BASE_URL}"; BEDROCK_BASE="${BEDROCK_BASE%/}"
BEDROCK_KEY="${BEDROCK_API_KEY:?set BEDROCK_API_KEY}"
AGENT_LLM="${MEMSEL_AGENT_LLM:-openai/mistral.ministral-3-8b-instruct}"
# Bedrock reserves input_tokens + max_tokens of quota at admission, so keep this modest.
AGENT_MAXTOK="${MEMSEL_AGENT_MAX_TOKENS:-1024}"
MEMBANK="${MEMSEL_MEMORY_BASE:-$DATA/tau2/memory_banks/ministral}"

echo "=== tau2 services (Ministral-3-8B on Bedrock) ==="
echo "  domain   $DOMAIN"
echo "  ports   $PORTS"
echo "  agent    $AGENT_LLM  (max_tokens $AGENT_MAXTOK)"
echo "  memory   $MEMBANK"

# probe Bedrock once before starting anything
code=$(curl -s -o /dev/null -w "%{http_code}" -m 30 -X POST "$BEDROCK_BASE/chat/completions" \
  -H "Content-Type: application/json" -H "Authorization: Bearer $BEDROCK_KEY" \
  -d "{\"model\":\"${AGENT_LLM#openai/}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":5}")
[ "$code" = "200" ] || { echo "  Bedrock probe failed: HTTP $code"; exit 1; }

for p in $PORTS; do
  curl -sf -m 2 "http://127.0.0.1:$p/health" >/dev/null 2>&1 && { echo "  port $p is busy"; exit 1; }
done

LOGDIR="${LOGDIR:-$REPO/runs/logs}"
mkdir -p "$LOGDIR"
cd "$REPO/memcomposer/tau2"
for p in $PORTS; do
  MEMSEL_MEMORY_BASE="$MEMBANK" \
  MEMSEL_AGENT_BASES="$BEDROCK_BASE" \
  MEMSEL_AGENT_API_KEY="$BEDROCK_KEY" \
  MEMSEL_AGENT_LLM="$AGENT_LLM" \
  MEMSEL_AGENT_MAX_TOKENS="$AGENT_MAXTOK" \
  MEMSEL_AGENT_CONC_PER_CARD="${MEMSEL_AGENT_CONC_PER_CARD:-32}" \
  MEMSEL_USER_CONC="${MEMSEL_USER_CONC:-64}" \
  MEMSEL_JOIN_TIMEOUT="${MEMSEL_JOIN_TIMEOUT:-840}" \
    nohup $TAU2 tau2_service.py --port "$p" --domain "$DOMAIN" --workers 200 \
    > "$LOGDIR/svc${p}_${DOMAIN}_ministral.log" 2>&1 &
  echo "  launched $p  pid $!"
done

for i in $(seq 1 60); do
  ok=1; for p in $PORTS; do curl -sf -m 2 "http://127.0.0.1:$p/health" >/dev/null || ok=0; done
  [ $ok = 1 ] && break; sleep 3
done
for p in $PORTS; do
  echo "  $p: $(curl -s -m 10 "http://127.0.0.1:$p/health" | tr ',' '\n' | sed 's/[{}\"]//g' | grep -E 'ok|domain|tasks|memories' | tr '\n' ' ')"
done
