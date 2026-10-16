#!/usr/bin/env bash
# One campaign worker: claim an episode, play it, repeat (see claim_next.py).
#
# The single-instance file is named instances_<TAG>_<SUITE>_w<WORKER>: the suite
# is part of the name because 3p and 6p share one game directory (in/), and two
# suites' workers with the same id would otherwise overwrite each other's file
# (happened at launch on 2026-09-04: 3p clem runs played 6p instances).
#
#   SUITE=solo_grid MODEL=DeepSeek-V4-Flash TAG=dsflash WORKER=3 \
#   RESULTS=$PWD/results_dsflash_grid_n15_solo bash campaign/worker.sh
#
# Normally started by scale_workers.sh. Exits after the current episode when
# <results>/_campaign/stop or stop_w<WORKER> exists, when nothing is left to
# claim, or after FAIL_LIMIT consecutive episodes that ended without an
# interactions.json. A 402 (deepseek_api.py raises InsufficientBalanceError,
# which kills the clem process on the spot) is NOT recorded against the episode.
# With MIN_BALANCE=<usd> set, a worker checks the account before every claim
# (campaign/balance.py) and waits instead of claiming while the balance is below
# the floor, so a top-up resumes the campaign without anyone re-running scripts
# and in-flight episodes are not killed mid-way by the account running dry.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

SUITE="${SUITE:?}"; MODEL="${MODEL:?}"; RESULTS="${RESULTS:?}"; TAG="${TAG:?}"; WORKER="${WORKER:?}"
TEMP="${TEMP:-0.6}"
FAIL_LIMIT="${FAIL_LIMIT:-3}"
MIN_BALANCE="${MIN_BALANCE:-}"

case "$SUITE" in
  solo) GAME=matrixgame_covered_solo;        GAME_DIR=matrixgame_covered_solo;         SRC=instances ;;
  dyad) GAME=matrixgame_covered_comm;        GAME_DIR=matrixgame_covered_comm;         SRC=instances ;;
  3p)   GAME=matrixgame_covered_comm_3p;     GAME_DIR=matrixgame_covered_comm_nplayer; SRC=instances_3p ;;
  6p)   GAME=matrixgame_covered_comm_6p;     GAME_DIR=matrixgame_covered_comm_nplayer; SRC=instances_6p ;;
  solo_grid) GAME=matrixgame_covered_solo_grid;    GAME_DIR=matrixgame_covered_solo_grid;         SRC=instances ;;
  dyad_grid) GAME=matrixgame_covered_comm_grid;    GAME_DIR=matrixgame_covered_comm_grid;         SRC=instances ;;
  3p_grid)   GAME=matrixgame_covered_comm_3p_grid; GAME_DIR=matrixgame_covered_comm_nplayer_grid; SRC=instances_3p ;;
  6p_grid)   GAME=matrixgame_covered_comm_6p_grid; GAME_DIR=matrixgame_covered_comm_nplayer_grid; SRC=instances_6p ;;
  # Second prompt arm: same grid boards, revised initial prompts (prompt_updated/).
  dyad_grid_updated_prompt) GAME=matrixgame_covered_comm_grid_updated_prompt; GAME_DIR=matrixgame_covered_comm_grid_updated_prompt; SRC=instances ;;
  3p_grid_updated_prompt)   GAME=matrixgame_covered_comm_3p_grid_updated_prompt; GAME_DIR=matrixgame_covered_comm_nplayer_grid_updated_prompt; SRC=instances_3p ;;
  6p_grid_updated_prompt)   GAME=matrixgame_covered_comm_6p_grid_updated_prompt; GAME_DIR=matrixgame_covered_comm_nplayer_grid_updated_prompt; SRC=instances_6p ;;
  # Final arm: updated-prompt instances, recent-play record + cycle rule (prompt_final/).
  solo_grid_final) GAME=matrixgame_covered_solo_grid_final;    GAME_DIR=matrixgame_covered_solo_grid_final;         SRC=instances ;;
  dyad_grid_final) GAME=matrixgame_covered_comm_grid_final;    GAME_DIR=matrixgame_covered_comm_grid_final;         SRC=instances ;;
  3p_grid_final)   GAME=matrixgame_covered_comm_3p_grid_final; GAME_DIR=matrixgame_covered_comm_nplayer_grid_final; SRC=instances_3p ;;
  6p_grid_final)   GAME=matrixgame_covered_comm_6p_grid_final; GAME_DIR=matrixgame_covered_comm_nplayer_grid_final; SRC=instances_6p ;;
  # Final arm with teammates' messages in the order sent (prompt_final/chrono.py); no solo/dyad.
  3p_grid_final_chrono) GAME=matrixgame_covered_comm_3p_grid_final_chrono; GAME_DIR=matrixgame_covered_comm_nplayer_grid_final_chrono; SRC=instances_3p ;;
  6p_grid_final_chrono) GAME=matrixgame_covered_comm_6p_grid_final_chrono; GAME_DIR=matrixgame_covered_comm_nplayer_grid_final_chrono; SRC=instances_6p ;;
  # Crossed-device levels S3x/S4x (matrixgame_covered_crossed_shared), grid rendering.
  solo_grid_crossed) GAME=matrixgame_covered_solo_crossed_grid;    GAME_DIR=matrixgame_covered_solo_crossed_grid;         SRC=instances ;;
  dyad_grid_crossed) GAME=matrixgame_covered_comm_crossed_grid;    GAME_DIR=matrixgame_covered_comm_crossed_grid;         SRC=instances ;;
  3p_grid_crossed)   GAME=matrixgame_covered_comm_3p_crossed_grid; GAME_DIR=matrixgame_covered_comm_nplayer_crossed_grid; SRC=instances_3p ;;
  6p_grid_crossed)   GAME=matrixgame_covered_comm_6p_crossed_grid; GAME_DIR=matrixgame_covered_comm_nplayer_crossed_grid; SRC=instances_6p ;;
  *) echo "unknown SUITE '$SUITE'" >&2; exit 2 ;;
esac

LOGDIR="$RESULTS/_campaign"
mkdir -p "$LOGDIR/claims"
LOG="$LOGDIR/worker_w$WORKER.log"
PIDFILE="$LOGDIR/worker_w$WORKER.pid"
echo $$ >"$PIDFILE"

export SSL_CERT_FILE="$REPO/.certs/clem-ca-bundle.pem"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

say() { echo "[$(date '+%F %T')] w$WORKER: $*" >>"$LOG"; }

CHILD=""
CLAIM=""
stop() {
  trap - TERM INT
  say "signalled, killing clem ${CHILD:-none} and releasing claim ${CLAIM:-none}"
  [[ -n "$CHILD" ]] && kill "$CHILD" 2>/dev/null
  [[ -n "$CLAIM" ]] && rm -rf "$LOGDIR/claims/$CLAIM"
  rm -f "$PIDFILE"
  exit 143
}
trap stop TERM INT

say "WORKER_START suite=$SUITE game=$GAME model=$MODEL temp=$TEMP results=$RESULTS pid=$$"
fails=0
while :; do
  if [[ -e "$LOGDIR/stop" || -e "$LOGDIR/stop_w$WORKER" ]]; then
    say "stop file present, exiting"
    break
  fi
  if [[ -n "$MIN_BALANCE" ]]; then
    bal="$(python3 "$REPO/campaign/balance.py" 2>/dev/null)"
    if [[ "$bal" =~ ^-?[0-9]+(\.[0-9]+)?$ ]] && awk -v b="$bal" -v m="$MIN_BALANCE" 'BEGIN{exit !(b < m)}'; then
      say "balance $bal USD below floor $MIN_BALANCE USD - not claiming, retry in 300s"
      sleep 300
      continue
    fi
  fi
  out="$(python3 "$REPO/campaign/claim_next.py" --game "$GAME" --game-dir "$REPO/$GAME_DIR" \
          --src "$SRC" --results "$RESULTS" --model "$MODEL" --tag "${TAG}_${SUITE}" --worker "$WORKER" --pid $$ 2>&1)"
  rc=$?
  if [[ $rc -ne 0 ]]; then
    say "claim_next failed rc=$rc: $out"
    fails=$((fails + 1))
    [[ $fails -ge $FAIL_LIMIT ]] && { say "giving up after $fails consecutive failures"; break; }
    sleep 60
    continue
  fi
  if [[ "$out" == NONE* ]]; then
    say "nothing left to claim ($out), exiting"
    break
  fi
  CLAIM="${out#CLAIM=}"; CLAIM="${CLAIM%% *}"
  FILE="${out#*FILE=}"; FILE="${FILE%% *}"
  EXP="${out#*EXP=}"; EXP="${EXP%% *}"
  ID="${out#*ID=}"; ID="${ID%% *}"
  REM="${out#*REMAINING=}"; REM="${REM%% *}"
  say "playing $EXP/$ID (outstanding before claim: $REM)"

  t0=$(date +%s)
  clem run -g "$GAME" -m "$MODEL" -t "$TEMP" -i "$FILE" -r "$RESULTS" >>"$LOGDIR/clem_w$WORKER.log" 2>&1 &
  CHILD=$!
  wait "$CHILD"; rc=$?
  CHILD=""
  dt=$(( $(date +%s) - t0 ))

  episode="$RESULTS/$MODEL/$GAME/$EXP/instance_$(printf '%05d' "$ID")"
  if [[ -f "$episode/interactions.json" ]]; then
    say "done $EXP/$ID in ${dt}s (clem rc=$rc)"
    fails=0
  elif tail -c 6000 "$LOGDIR/clem_w$WORKER.log" | grep -q 'InsufficientBalanceError'; then
    say "FAILED $EXP/$ID after ${dt}s: DeepSeek 402 insufficient balance (not recorded against the episode)"
    fails=$((fails + 1))
  else
    say "FAILED $EXP/$ID after ${dt}s (clem rc=$rc, no interactions.json)"
    echo "$(date '+%F %T') w$WORKER rc=$rc after ${dt}s" >>"$LOGDIR/claims/$CLAIM.fails"
    fails=$((fails + 1))
  fi
  rm -rf "$LOGDIR/claims/$CLAIM"
  CLAIM=""
  if [[ $fails -ge $FAIL_LIMIT ]]; then
    say "giving up after $fails consecutive failed episodes (check balance / API)"
    break
  fi
  [[ $fails -gt 0 ]] && sleep $((120 * fails))
done
rm -f "$PIDFILE"
say "WORKER_END"
