#!/bin/bash
# Self-judge the selected trajectories: the agent is its own critic. Failing trajectories get
# review/self_critique.py, passing ones review/pass_audit.py.
#
#   AGENT=qwen32b bash label.sh                 # all three domains
#   AGENT=gpt54mini DOMAINS=retail CONC=4 bash label.sh
#
# Both critics read a tau2 results.json, so sims.jsonl is repacked first. Credentials come from
# agents.creds and reach the children through the environment only; they are never echoed.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python}
: "${AGENT:?set AGENT=qwen32b|qwen235b|gpt54mini|gpt54|ministral}"
DOMAINS=${DOMAINS:-"airline retail telecom"}
CONC=${CONC:-8}

read -r MODEL OUTBASE APIBASE APIKEY <<<"$(cd "$HERE" && $PY - "$AGENT" <<'PYEOF'
import sys
from agents import AGENTS, creds
a = AGENTS[sys.argv[1]]; c = creds(a)
print(a.model, a.base, c['OPENAI_API_BASE'], c['OPENAI_API_KEY'])
PYEOF
)"
[ -n "$APIKEY" ] || { echo "no credentials for $AGENT"; exit 1; }
export OPENAI_API_BASE="$APIBASE" OPENAI_API_KEY="$APIKEY" OPENAI_BASE_URL="$APIBASE"
W=${W:-$OUTBASE/_work}
mkdir -p "$W"
echo "[$AGENT] critic = $MODEL, base = $OUTBASE, domains = $DOMAINS"

for d in $DOMAINS; do
  $PY "$HERE/sims_to_tau2.py" --agent "$AGENT" "$d" -o "$W/${d}_all.json"
  $PY "$HERE/sims_to_tau2.py" --agent "$AGENT" "$d" --selection -o "$W/${d}_selected.json"
done

for d in $DOMAINS; do
  mkdir -p "$OUTBASE/_work/$d/logs"
  # self_critique labels only the failing sims in its input; --siblings lets it show the critic a
  # passing run of the same task as a concrete correct path (excluding the sim itself).
  $PY "$HERE/review/self_critique.py" "$W/${d}_selected.json" -o "$OUTBASE/$d/labelled_sc.json" \
      --model "$MODEL" --concurrency "$CONC" --siblings "$W/${d}_all.json" \
      > "$OUTBASE/_work/$d/logs/self_critique.log" 2>&1 &
  # pass_audit labels only the passing ones.
  $PY "$HERE/review/pass_audit.py" "$W/${d}_selected.json" -o "$OUTBASE/$d/labelled_pa.json" \
      --model "$MODEL" --concurrency "$CONC" \
      > "$OUTBASE/_work/$d/logs/pass_audit.log" 2>&1 &
done
wait
echo "=========== [$AGENT] all done $(date '+%H:%M:%S') ==========="
for d in $DOMAINS; do
  echo "--- $d ---"
  tail -2 "$OUTBASE/_work/$d/logs/self_critique.log"
  tail -3 "$OUTBASE/_work/$d/logs/pass_audit.log"
done
