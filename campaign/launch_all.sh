#!/usr/bin/env bash
# Launch all four suites in parallel, each with its own resume loop, results
# root and shard count. Safe to re-run after a reboot or a kill: every suite
# picks up exactly where it stopped.
#
# Shard counts are proportional to MEASURED remaining work per suite (a turn is
# one API call regardless of player count, so cost tracks episode length, not
# player count: S1 exits early on success at ~32 calls, S2+ runs its full ~168-turn
# budget). Total concurrent requests = 24.
# Dial the whole campaign down with e.g. SCALE=0.5.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SOLO_SHARDS="${SOLO_SHARDS:-3}"
DYAD_SHARDS="${DYAD_SHARDS:-7}"
P3_SHARDS="${P3_SHARDS:-7}"
P6_SHARDS="${P6_SHARDS:-7}"

launch() {
  local suite="$1" shards="$2"
  local results="$PWD/results_qwen38_reason_n15_$suite"
  mkdir -p "$results/_campaign"
  SUITE="$suite" SHARDS="$shards" nohup caffeinate -i -m -s \
    bash campaign/run_chain.sh >>"$results/_campaign/nohup.log" 2>&1 &
  disown
  echo "launched $suite with $shards shard(s) -> $results"
}

launch solo "$SOLO_SHARDS"
launch dyad "$DYAD_SHARDS"
launch 3p   "$P3_SHARDS"
launch 6p   "$P6_SHARDS"
