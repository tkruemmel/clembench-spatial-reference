#!/usr/bin/env bash
# DeepSeek-V4.1-Flash on the updated-prompt grid suites (second prompt arm,
# prompt_updated/) - the DeepSeek counterpart of the Qwen3.8 arm that
# campaign/launch_updated_prompt.sh runs against Potsdam. Same worker/claim
# machinery as dsv41_grid.sh (scale_workers.sh / worker.sh / claim_next.py),
# three suites instead of four: there is no solo updated-prompt package. Runs ON
# THE CAMPAIGN HOST (Hetzner); drive it from the laptop with:
#
#   SUITES="dyad 3p" RESULTS_PREFIX=results_dsv41_grid_updated_prompt_n15 \
#   CAMPAIGN_SCRIPT=campaign/dsv41_updated_prompt.sh DYAD=12 P3=12 P6=0 \
#     bash campaign/hetzner_dsflash.sh launch
#
# Re-run with new counts to scale at any time; `... launch stop` (or
# `bash campaign/dsv41_updated_prompt.sh stop` on the host) drains the workers
# after their current episode. Model note from dsv41_grid.sh applies: the
# registry entry must be DeepSeek-V4.1-Flash (model_id deepseek-flash), never
# the DeepSeek-V4-Flash alias.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ "${1:-}" == "stop" ]]; then DYAD=0; P3=0; P6=0; fi
export MODEL="${MODEL:-DeepSeek-V4.1-Flash}"
export TAG="${TAG:-dsv41up}"
export RESULTS_PREFIX="${RESULTS_PREFIX:-results_dsv41_grid_updated_prompt_n15}"
export TEMP="${TEMP:-0.6}"
# Floor of 30, not dsflash_grid.sh's 12: the floor only blocks new claims, so it
# has to cover what the workers already in flight will still spend. At 24 workers
# holding S4 claims that run 40-120 min each, 12 did not - on 2026-09-29 the
# account went from the floor to -0.06 USD and two ~2 h episodes were 402-killed.
export MIN_BALANCE="${MIN_BALANCE:-30}"   # workers wait instead of claiming below this (USD)
# Counts default to 0: a suite runs only when asked for. A target of 0 against a
# finished suite is also how you leave it alone while another one runs.
SUITES="dyad_grid_updated_prompt 3p_grid_updated_prompt 6p_grid_updated_prompt" \
TARGETS="${DYAD:-0} ${P3:-0} ${P6:-0}" \
  bash campaign/scale_workers.sh
