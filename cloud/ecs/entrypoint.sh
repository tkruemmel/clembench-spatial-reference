#!/usr/bin/env bash
# One ECS task drives one (model, suite) pair to completion.
#
# The task owns the whole suite, so the campaign driver's "one driver per suite"
# invariant holds by construction and its shard split can rebalance every pass.
# Parallelism lives INSIDE the task: PROCS concurrent `clem run` processes, which
# is what "shard size" means here. The workload is API-bound, so a fraction of a
# vCPU carries several streams - and Fargate is billed per vCPU, not per task.
#
# S3 is the durable copy. The task pulls the suite's results before starting and
# pushes them back periodically and on SIGTERM, so stopping tasks to change PROCS
# costs at most the episodes in flight.
set -uo pipefail

: "${S3_BUCKET:?S3_BUCKET is required}"
: "${RUN_ID:?RUN_ID is required}"
: "${MODEL_KEY:?MODEL_KEY is required}"
: "${CLEM_MODEL:?CLEM_MODEL is required}"
: "${SUITE:?SUITE is required}"
PROCS="${PROCS:-4}"
TEMP="${TEMP:-0.6}"
SYNC_SECONDS="${SYNC_SECONDS:-180}"

REPO=/clembench
RESULTS=/work/results
PREFIX="$RUN_ID/$MODEL_KEY/$SUITE"
cd "$REPO" || exit 1
mkdir -p "$RESULTS"

say() { echo "[$(date -u '+%F %T')] $MODEL_KEY/$SUITE: $*"; }
sync_down() { python3 "$REPO/cloud/ecs/s3sync.py" pull --bucket "$S3_BUCKET" --prefix "$PREFIX" --root "$RESULTS"; }
sync_up() { python3 "$REPO/cloud/ecs/s3sync.py" push --bucket "$S3_BUCKET" --prefix "$PREFIX" --root "$RESULTS"; }

say "task start: model=$CLEM_MODEL procs=$PROCS temp=$TEMP s3://$S3_BUCKET/$PREFIX"
sync_down || say "WARNING: initial pull failed - starting from whatever is on disk"

# A lock that travelled here inside the S3 mirror belongs to a container that no
# longer exists. This task is the suite's only driver, so it is always ours.
rm -rf "$RESULTS/_campaign/driver.lock"

( while sleep "$SYNC_SECONDS"; do sync_up || say "periodic push failed"; done ) &
SYNCER=$!

CHAIN=""
shutdown() {
  trap - TERM INT
  say "SIGTERM - stopping the chain and pushing what is finished"
  kill "$SYNCER" 2>/dev/null
  [[ -n "$CHAIN" ]] && { kill "$CHAIN" 2>/dev/null; wait "$CHAIN" 2>/dev/null; }
  sync_up || say "final push failed"
  exit 143
}
trap shutdown TERM INT

SUITE="$SUITE" MODEL="$CLEM_MODEL" TEMP="$TEMP" SHARDS="$PROCS" RESULTS="$RESULTS" \
  bash "$REPO/campaign/run_chain.sh" &
CHAIN=$!
wait "$CHAIN"
rc=$?
CHAIN=""

kill "$SYNCER" 2>/dev/null
sync_up || { say "final push failed - RETRYING once"; sleep 15; sync_up; }
say "chain exited rc=$rc"
exit "$rc"
