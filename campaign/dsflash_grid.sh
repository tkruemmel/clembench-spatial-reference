#!/usr/bin/env bash
# DeepSeek-V4-Flash on the four grid-rendered suites, worker-based (see
# scale_workers.sh / worker.sh / claim_next.py). Re-run with new counts to
# scale up or down at any time; `bash campaign/dsflash_grid.sh stop` drains all.
#
#   SOLO=4 DYAD=11 P3=10 P6=7 bash campaign/dsflash_grid.sh
#
# Default split is proportional to the compact campaign's measured episode-hours
# (solo 11% / dyad 34% / 3p 31% / 6p 24%) so the suites finish together.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ "${1:-}" == "stop" ]]; then SOLO=0 DYAD=0 P3=0 P6=0; fi
export MODEL="${MODEL:-DeepSeek-V4-Flash}"
export TAG="${TAG:-dsflash}"
export RESULTS_PREFIX="${RESULTS_PREFIX:-results_dsflash_grid_n15}"
export TEMP="${TEMP:-0.6}"
export MIN_BALANCE="${MIN_BALANCE:-12}"   # workers wait instead of claiming below this (USD)
SUITES="solo_grid dyad_grid 3p_grid 6p_grid" \
TARGETS="${SOLO:-4} ${DYAD:-11} ${P3:-10} ${P6:-7}" \
  bash campaign/scale_workers.sh
