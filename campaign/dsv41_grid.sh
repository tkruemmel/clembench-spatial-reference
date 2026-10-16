#!/usr/bin/env bash
# DeepSeek-V4.1-Flash on the four grid-rendered suites, worker-based (see
# scale_workers.sh / worker.sh / claim_next.py). Runs ON THE CAMPAIGN HOST
# (Hetzner); the laptop drives it through campaign/supervise_dsv41.py, which
# re-runs this script over ssh with new counts whenever rebalance.py proposes a
# different split. Re-run with new counts to scale at any time;
# `bash campaign/dsv41_grid.sh stop` drains all workers.
#
#   SOLO=3 DYAD=8 P3=8 P6=5 bash campaign/dsv41_grid.sh      # 24 streams
#   ARM=_final SOLO=2 DYAD=8 P3=8 P6=6 bash campaign/dsv41_grid.sh   # final arm
#   ARM=_final_chrono SUITES="3p 6p" P3=12 P6=12 bash campaign/dsv41_grid.sh   # chrono arm
#
# The model is the one served as `deepseek-flash` since 2026-09-10. Do NOT point
# a V4.1 campaign at the DeepSeek-V4-Flash registry entry: its model_id
# (deepseek-v4-flash) is a temporary alias that now routes to V4.1 as well, so
# episodes would land in the V4-Flash tree under the wrong model name (that is
# exactly what contaminated results_dsflash_grid_n15_* on 2026-09-11).
#
# Default split is proportional to the compact DeepSeek campaign's measured
# episode-hours (solo 11% / dyad 34% / 3p 31% / 6p 24%) so the suites finish
# together; the supervisor replaces it with a measured one within a few hours.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ "${1:-}" == "stop" ]]; then SOLO=0 DYAD=0 P3=0 P6=0; fi
export MODEL="${MODEL:-DeepSeek-V4.1-Flash}"
export TAG="${TAG:-dsv41}"
ARM="${ARM:-}"   # suite suffix: empty = original grid arm, _final = final arm
export RESULTS_PREFIX="${RESULTS_PREFIX:-results_dsv41_grid${ARM}_n15}"
export TEMP="${TEMP:-0.6}"
export MIN_BALANCE="${MIN_BALANCE:-12}"   # workers wait instead of claiming below this (USD)
suites=""; targets=""
for s in ${SUITES:-solo dyad 3p 6p}; do   # short names, as in hetzner_dsflash.sh
  case "$s" in solo) n=${SOLO:-3} ;; dyad) n=${DYAD:-8} ;; 3p) n=${P3:-8} ;; 6p) n=${P6:-5} ;; esac
  suites+="${s}_grid$ARM "; targets+="$n "
done
SUITES="$suites" TARGETS="$targets" bash campaign/scale_workers.sh
