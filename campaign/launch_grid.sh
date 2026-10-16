#!/usr/bin/env bash
# Launch the four grid-rendered suites (matrixgame_covered_*_grid) against the
# Potsdam Qwen3.8 endpoint, one resume-loop driver per suite. Idempotent: re-run
# after a kill or reboot and every suite resumes where it stopped (a missing
# interactions.json is the resume signal, see campaign/run_chain.sh).
#
# Shard counts are proportional to the measured episode-hours of the compact
# campaign (solo 314 h, dyad 970 h, 3p 883 h, 6p 676 h at n=15), so the four
# suites finish at about the same time. Total streams = 16, one step above the
# 14 that ran cleanly at the end of the compact campaign; 24 tipped the shared
# endpoint into a 300 s timeout cascade. Override per suite, e.g.
#   DYAD_SHARDS=6 bash campaign/launch_grid.sh
# Re-shard only while nothing is in flight (after a clean stop): restarting a
# driver discards its in-flight multi-hour episodes.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SOLO_SHARDS="${SOLO_SHARDS:-2}"
DYAD_SHARDS="${DYAD_SHARDS:-5}"
P3_SHARDS="${P3_SHARDS:-5}"
P6_SHARDS="${P6_SHARDS:-4}"
export MODEL="${MODEL:-Qwen3.8-27B-FP8}"
export TEMP="${TEMP:-0.6}"
export STALL_LIMIT="${STALL_LIMIT:-10}"
export MAX_PASSES="${MAX_PASSES:-60}"

# Refuse to start a driver on a suite that worker.sh processes are playing
# (campaign/qwen38_grid.sh): a driver deals shards over the same outstanding
# episodes and would play them a second time into the same directories.
for suite in solo dyad 3p 6p; do
  for pidfile in "$PWD/results_qwen38_grid_n15_${suite}/_campaign"/worker_w*.pid; do
    [[ -e "$pidfile" ]] || continue
    if kill -0 "$(cat "$pidfile")" 2>/dev/null; then
      echo "refusing: live worker $(basename "$pidfile") on $suite - this campaign runs on campaign/qwen38_grid.sh now" >&2
      exit 3
    fi
  done
done

launch() {
  local suite="$1" shards="$2"
  local results="$PWD/results_qwen38_grid_n15_${suite%_grid}"
  mkdir -p "$results/_campaign"
  SUITE="$suite" SHARDS="$shards" RESULTS="$results" nohup caffeinate -i -m -s \
    bash campaign/run_chain.sh >>"$results/_campaign/nohup.log" 2>&1 &
  disown
  echo "launched $suite with $shards shard(s) -> $results"
}

launch solo_grid "$SOLO_SHARDS"
launch dyad_grid "$DYAD_SHARDS"
launch 3p_grid   "$P3_SHARDS"
launch 6p_grid   "$P6_SHARDS"
