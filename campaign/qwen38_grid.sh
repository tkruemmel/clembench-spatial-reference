#!/usr/bin/env bash
# Qwen3.8-27B-FP8 (Potsdam endpoint) on the four grid-rendered suites, worker
# based (scale_workers.sh / worker.sh / claim_next.py) so the stream count can be
# changed at any time without discarding in-flight episodes. Re-run with new
# counts to scale up or down; `bash campaign/qwen38_grid.sh stop` drains all.
#
#   SOLO=2 DYAD=9 P3=6 P6=3 bash campaign/qwen38_grid.sh      # 20 streams
#   ARM=_final SOLO=2 DYAD=9 P3=6 P6=3 bash campaign/qwen38_grid.sh   # final arm
#
# Replaces launch_grid.sh (fixed shards under run_chain.sh) for this campaign
# from 2026-09-05 on; the results trees are the same, a missing interactions.json
# is the resume signal in both models. Never run launch_grid.sh while workers are
# live: two drivers on one suite play the same episode twice.
#
# Keep the total at or below 24: the shared vLLM endpoint's efficiency knee is
# ~24 concurrent streams and 24 fixed shards once tipped it into a 300 s timeout
# cascade (see memory: project_qwen38_campaign). FAIL_LIMIT is raised from 3 to
# 10 because the endpoint stalls transiently; a worker only gives up after 10
# consecutive episodes without an interactions.json.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ "${1:-}" == "stop" ]]; then SOLO=0 DYAD=0 P3=0 P6=0; fi
export MODEL="${MODEL:-Qwen3.8-27B-FP8}"
export TAG="${TAG:-qwen38}"
ARM="${ARM:-}"   # suite suffix: empty = original grid arm, _final = final arm
export RESULTS_PREFIX="${RESULTS_PREFIX:-results_qwen38_grid${ARM}_n15}"
export TEMP="${TEMP:-0.6}"
export FAIL_LIMIT="${FAIL_LIMIT:-10}"
SUITES="solo_grid$ARM dyad_grid$ARM 3p_grid$ARM 6p_grid$ARM" \
TARGETS="${SOLO:-2} ${DYAD:-9} ${P3:-6} ${P6:-3}" \
  bash campaign/scale_workers.sh
