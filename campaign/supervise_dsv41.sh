#!/usr/bin/env bash
# Start the DeepSeek-V4.1-Flash campaign supervisor detached, under caffeinate so
# the laptop does not sleep on it. Idempotent: a second start refuses while one
# is already running. The workers themselves live on Hetzner and keep playing
# whether or not this process is up - stopping the supervisor only stops the
# supervision (status logging, rebalancing, unblocking).
#
#   bash campaign/supervise_dsv41.sh              # start
#   tail -f campaign/supervisor_dsv41.log         # watch
#   bash campaign/supervise_dsv41.sh stop         # stop (remote workers keep running)
#
# To drain the campaign itself:
#   RESULTS_PREFIX=results_dsv41_grid_n15 CAMPAIGN_SCRIPT=campaign/dsv41_grid.sh \
#     bash campaign/hetzner_dsflash.sh stop
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export TAG="${TAG:-dsv41}"   # MODEL=DeepSeek-V4.1-Flash-NoThink TAG=dsv41nt: the no-think arm
if [[ "${1:-}" == "stop" ]]; then
  pid=$(cat "campaign/.supervisor_${TAG}${ARM:-}.lock/pid" 2>/dev/null || true)
  [[ -n "${pid:-}" ]] && kill "$pid" 2>/dev/null && echo "stopped supervisor pid $pid" || echo "no supervisor running"
  exit 0
fi

INTERVAL="${INTERVAL:-600}"
TOTAL="${TOTAL:-24}"
# USD floor the workers get. Drop it for the endgame (MIN_BALANCE=2) so the last
# few episodes can be paid for out of the tail of the account.
MIN_BALANCE="${MIN_BALANCE:-12}"
nohup caffeinate -i -m -s python3 campaign/supervise_dsv41.py --interval "$INTERVAL" --total "$TOTAL" --min-balance "$MIN_BALANCE" \
  >>"campaign/supervisor_${TAG}${ARM:-}.nohup.log" 2>&1 &
disown
sleep 2
echo "supervisor started (interval ${INTERVAL}s, $TOTAL streams, floor ${MIN_BALANCE} \$); watch with: tail -f campaign/supervisor_${TAG}${ARM:-}.log"
