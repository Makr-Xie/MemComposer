#!/bin/bash
# Collect the no-memory (k=0) TRAIN rollouts that a memory bank is built from, for any API agent.
#
#   AGENT=qwen32b DOMAIN=retail ./collect.sh
#   AGENT=gpt54mini DOMAIN=airline SMOKE=2 ./collect.sh   # smoke test: 2 tasks, 1 opening, 1 seed
#
# Protocol: agent temp 1.0, user zai.glm-4.7 temp 0, 4 sampled openings per task x 2 tau2 seeds
# = 8 sims/task, dedup off, tau2's own grader. The four openings are agent-independent
# (<data>/openings/<d>/train_op{0..3}.parquet) and are reused as-is, so every agent's bank is built
# from the same customer requests.
#
# Only API agents (Bedrock / OpenAI). A local vLLM agent needs its servers started first.
#
# Starts a tau2 simulation service (SERVICE, default <repo>/memcomposer/tau2/tau2_service.py) and
# drives it with collect_rollouts.py, one process per opening.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
PY=${PYTHON:-python}
SERVICE=${SERVICE:-$REPO/memcomposer/tau2/tau2_service.py}
: "${AGENT:?set AGENT=qwen32b|qwen235b|gpt54mini|gpt54|ministral}"
: "${DOMAIN:?set DOMAIN=airline|retail|telecom}"
SMOKE=${SMOKE:-0}
CONC=${CONC:-40}
TRIALS=${TRIALS:-2}

read -r MODEL OUT APIBASE APIKEY <<<"$(cd "$HERE" && $PY - "$AGENT" "$DOMAIN" <<'PYEOF'
import sys
import paths
from agents import AGENTS, creds
a = AGENTS[sys.argv[1]]; c = creds(a)
print(a.model, paths.rollouts(a.tag, sys.argv[2]), c['OPENAI_API_BASE'], c['OPENAI_API_KEY'])
PYEOF
)"
[ -n "$APIKEY" ] || { echo "no credentials for $AGENT"; exit 1; }

OPENINGS="0 1 2 3"
LIMIT=0
if [ "$SMOKE" != 0 ]; then OUT=$OUT-smoke; OPENINGS=0; TRIALS=1; CONC=4; LIMIT=$SMOKE; fi
mkdir -p "$OUT/logs"
LOG=$OUT/logs/cell.log
say(){ echo "[$AGENT/$DOMAIN] $* $(date '+%H:%M:%S')" | tee -a "$LOG"; }

case $DOMAIN in airline) o=0;; retail) o=1;; telecom) o=2;; esac
case $AGENT in ministral) b=0;; qwen32b) b=3;; qwen235b) b=6;; gpt54mini) b=9;; *) b=12;; esac
TAUPORT=${TAUPORT:-$((8220 + b + o))}

PIDS=""
cleanup(){ for p in $PIDS; do for x in $(pgrep -P "$p" 2>/dev/null) "$p"; do kill -9 "$x" 2>/dev/null; done; done; }
trap cleanup EXIT

say "start (model $MODEL, tau2 port $TAUPORT, smoke=$SMOKE, openings '$OPENINGS' x $TRIALS seeds)"
cd "$(dirname "$SERVICE")"
MEMSEL_AGENT_BASES="$APIBASE" MEMSEL_AGENT_API_KEY="$APIKEY" \
MEMSEL_AGENT_LLM="$MODEL" MEMSEL_AGENT_TEMP=1.0 MEMSEL_AGENT_MAX_TOKENS=1024 \
MEMSEL_AGENT_CONC_PER_CARD=64 MEMSEL_SIM_DUMP="$OUT/sims.jsonl" \
  nohup $PY "$SERVICE" --port $TAUPORT --domain $DOMAIN --workers 120 \
  > "$OUT/logs/service_$TAUPORT.log" 2>&1 &
PIDS="$PIDS $!"
for n in $(seq 1 80); do sleep 3; curl -sf -m 2 "http://127.0.0.1:$TAUPORT/health" >/dev/null 2>&1 && break; done
curl -sf -m 5 "http://127.0.0.1:$TAUPORT/health" >/dev/null || { say "tau2 never came up"; exit 1; }
say "tau2 up"

# The four openings run side by side against one service: they are independent eval sets and the
# agent is an API, so there is nothing to serialise on.
cd "$HERE"
# wait on the opening jobs BY PID: a bare `wait` would also wait on the tau2 service in $PIDS,
# which never exits.
JOBS=""
for j in $OPENINGS; do
  $PY collect_rollouts.py --agent "$AGENT" --domain $DOMAIN --opening $j --url "http://127.0.0.1:$TAUPORT" \
      --out-dir "$OUT" --trials $TRIALS --limit $LIMIT --concurrency $CONC \
      > "$OUT/logs/run_op$j.log" 2>&1 &
  JOBS="$JOBS $!"
done
wait $JOBS
for j in $OPENINGS; do say "opening $j: $(tail -1 "$OUT/logs/run_op$j.log" | cut -c1-90)"; done

cat > "$OUT/meta.json" <<META
{"domain":"$DOMAIN","agent":"$AGENT","split":"train","condition":"nomem",
 "agent_model":"$MODEL","agent_temp":1.0,"agent_max_tokens":1024,
 "trials":$TRIALS,"openings":"$OPENINGS",
 "note":"4 sampled openings x 2 tau2 seeds = 8 sims/task",
 "user_model":"zai.glm-4.7","user_temp":0.0,"max_steps":100,"dedup":false,
 "grader":"tau2 own grader","smoke":$SMOKE,
 "purpose":"no-memory train rollouts for memory construction",
 "started":"$(date -u +%Y-%m-%dT%H:%M:%SZ)","sims":$(cat "$OUT"/k0.op*.jsonl 2>/dev/null | wc -l)}
META
say "DONE: $(cat "$OUT"/k0.op*.jsonl 2>/dev/null | wc -l) rows, $(wc -l < "$OUT/sims.jsonl" 2>/dev/null || echo 0) transcripts"
