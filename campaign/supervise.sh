#!/usr/bin/env bash
# Start the campaign supervisor detached, under caffeinate so the laptop does not
# sleep on it. Idempotent: a second start refuses while one is already running.
#
#   bash campaign/supervise.sh            # start
#   tail -f campaign/supervisor.log       # watch
#   bash campaign/supervise.sh stop       # stop (workers keep running)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ "${1:-}" == "stop" ]]; then
  pid=$(cat "campaign/.supervisor${ARM:-}.lock/pid" 2>/dev/null || true)
  [[ -n "${pid:-}" ]] && kill "$pid" 2>/dev/null && echo "stopped supervisor pid $pid" || echo "no supervisor running"
  exit 0
fi

INTERVAL="${INTERVAL:-600}"
nohup caffeinate -i -m -s python3 campaign/supervise.py --interval "$INTERVAL" \
  >>"campaign/supervisor${ARM:-}.nohup.log" 2>&1 &
disown
sleep 2
echo "supervisor started (interval ${INTERVAL}s); watch with: tail -f campaign/supervisor${ARM:-}.log"
