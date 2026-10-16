#!/usr/bin/env bash
# DeepSeek-V4.1-Flash on the crossed-device suites (S3x/S4x, grid rendering,
# updated prompts + A_k budget; matrixgame_covered_crossed_shared). Sibling of
# dsv41_updated_prompt.sh, whose arm these results compare against. Runs ON THE
# CAMPAIGN HOST (Hetzner); drive it from the laptop with:
#
#   SUITES="dyad 6p" RESULTS_PREFIX=results_dsv41_grid_crossed_n15 \
#   CAMPAIGN_SCRIPT=campaign/dsv41_crossed.sh DYAD=12 P6=12 MIN_BALANCE=30 \
#     bash campaign/hetzner_dsflash.sh launch
#
# Pass MIN_BALANCE explicitly: hetzner_dsflash.sh forwards its own default (12),
# which is below the in-flight floor dsv41_updated_prompt.sh documents.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ "${1:-}" == "stop" ]]; then SOLO=0; DYAD=0; P3=0; P6=0; fi
export MODEL="${MODEL:-DeepSeek-V4.1-Flash}"
export TAG="${TAG:-dsv41x}"
export RESULTS_PREFIX="${RESULTS_PREFIX:-results_dsv41_grid_crossed_n15}"
export TEMP="${TEMP:-0.6}"
export MIN_BALANCE="${MIN_BALANCE:-30}"
SUITES="solo_grid_crossed dyad_grid_crossed 3p_grid_crossed 6p_grid_crossed" \
TARGETS="${SOLO:-0} ${DYAD:-0} ${P3:-0} ${P6:-0}" \
  bash campaign/scale_workers.sh
