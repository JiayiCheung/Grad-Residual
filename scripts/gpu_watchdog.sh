#!/usr/bin/env bash
# keep the total gpu memory of a shared machine under a limit: every 20 s, if the used memory exceeds LIMIT_MIB, stop the most recently started
# gradresidual_train worker. a stopped worker loses at most one checkpoint interval; release its claims (or all claims once no worker runs)
# and start it again later.
#
#   LIMIT_MIB=11000 bash scripts/gpu_watchdog.sh
LIMIT_MIB=${LIMIT_MIB:-11000}
INTERVAL=${INTERVAL:-20}
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' \r')
  if [ -n "$used" ] && [ "$used" -gt "$LIMIT_MIB" ]; then
    victim=$(pgrep -n -f "gradresidual_train.*--worker|gradresidual.run.train.*--worker")
    if [ -n "$victim" ]; then
      kill "$victim"
      echo "$(date '+%F %T') ${used} MiB > ${LIMIT_MIB} MiB: stopped worker $victim"
      sleep 10
      continue
    fi
  fi
  sleep "$INTERVAL"
done
