#!/usr/bin/env bash
# Launch the three updated-prompt grid suites against the Potsdam Qwen3.8
# endpoint, one resume-loop driver per suite. Idempotent: re-run after a kill or
# reboot and every suite resumes where it stopped (a missing interactions.json
# is the resume signal, see campaign/run_chain.sh).
#
# Shards are proportional to each suite's measured episode cost (dyad 71 min,
# 3p 53 min, 6p 21 min, measured 2026-09-26 17:25 over the first 15 episodes), so
# the three suites finish at about the same time: 8/6/4 gives a makespan of ~35 h
# against the ~56 h of an even 5/5/4 split. Total is 18 streams — two above the 16
# that ran cleanly on the compact campaign, chosen because this endpoint is
# currently unshared (num_requests_waiting stays at 0); 24 tipped it into a 300 s
# timeout cascade, so do not go higher. Re-shard only while nothing is in flight:
# restarting a driver discards its in-flight multi-hour episodes.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DYAD_SHARDS="${DYAD_SHARDS:-8}"
P3_SHARDS="${P3_SHARDS:-6}"
P6_SHARDS="${P6_SHARDS:-4}"
export MODEL="${MODEL:-Qwen3.8-27B-FP8}"
export TEMP="${TEMP:-0.6}"
export STALL_LIMIT="${STALL_LIMIT:-10}"
export MAX_PASSES="${MAX_PASSES:-60}"

launch() {
  local suite="$1" shards="$2"
  local results="$PWD/results_qwen38_grid_updated_prompt_n15_$3"
  mkdir -p "$results/_campaign"
  SUITE="$suite" SHARDS="$shards" RESULTS="$results" nohup caffeinate -i -m -s \
    bash campaign/run_chain.sh >>"$results/_campaign/nohup.log" 2>&1 &
  disown
  echo "launched $suite with $shards shard(s) -> $results"
}

launch dyad_grid_updated_prompt "$DYAD_SHARDS" dyad
launch 3p_grid_updated_prompt   "$P3_SHARDS"   3p
launch 6p_grid_updated_prompt   "$P6_SHARDS"   6p
