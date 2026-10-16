#!/usr/bin/env bash
# Resume-capable driver for ONE matrixgame suite, played by SHARDS concurrent
# `clem run` processes against the same results tree.
#
# Every pass recomputes which episodes are still missing and replays only those,
# so killing this at any point (sleep, reboot, Ctrl-C) costs at most the episodes
# that were in flight. Re-run the identical command to continue where it stopped.
#
#   SUITE=6p SHARDS=7 bash campaign/run_chain.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

SUITE="${SUITE:?set SUITE to one of: solo dyad 3p 6p solo_grid dyad_grid 3p_grid 6p_grid dyad_grid_updated_prompt 3p_grid_updated_prompt 6p_grid_updated_prompt}"
MODEL="${MODEL:-Qwen3.8-27B-FP8}"
TEMP="${TEMP:-0.6}"
SHARDS="${SHARDS:-1}"
RESULTS="${RESULTS:-$REPO/results_qwen38_reason_n15_$SUITE}"
MAX_PASSES="${MAX_PASSES:-40}"
STALL_LIMIT="${STALL_LIMIT:-3}"

case "$SUITE" in
  solo) GAME=matrixgame_covered_solo;        GAME_DIR=matrixgame_covered_solo;         SRC=instances ;;
  dyad) GAME=matrixgame_covered_comm;        GAME_DIR=matrixgame_covered_comm;         SRC=instances ;;
  3p)   GAME=matrixgame_covered_comm_3p;     GAME_DIR=matrixgame_covered_comm_nplayer; SRC=instances_3p ;;
  6p)   GAME=matrixgame_covered_comm_6p;     GAME_DIR=matrixgame_covered_comm_nplayer; SRC=instances_6p ;;
  # Grid-rendered twins (ASCII grid instead of the compact listing, same boards).
  solo_grid) GAME=matrixgame_covered_solo_grid;    GAME_DIR=matrixgame_covered_solo_grid;         SRC=instances ;;
  dyad_grid) GAME=matrixgame_covered_comm_grid;    GAME_DIR=matrixgame_covered_comm_grid;         SRC=instances ;;
  3p_grid)   GAME=matrixgame_covered_comm_3p_grid; GAME_DIR=matrixgame_covered_comm_nplayer_grid; SRC=instances_3p ;;
  6p_grid)   GAME=matrixgame_covered_comm_6p_grid; GAME_DIR=matrixgame_covered_comm_nplayer_grid; SRC=instances_6p ;;
  # Second prompt arm: same grid boards, revised initial prompts (prompt_updated/).
  dyad_grid_updated_prompt) GAME=matrixgame_covered_comm_grid_updated_prompt; GAME_DIR=matrixgame_covered_comm_grid_updated_prompt; SRC=instances ;;
  3p_grid_updated_prompt)   GAME=matrixgame_covered_comm_3p_grid_updated_prompt; GAME_DIR=matrixgame_covered_comm_nplayer_grid_updated_prompt; SRC=instances_3p ;;
  6p_grid_updated_prompt)   GAME=matrixgame_covered_comm_6p_grid_updated_prompt; GAME_DIR=matrixgame_covered_comm_nplayer_grid_updated_prompt; SRC=instances_6p ;;
  # Crossed-device levels S3x/S4x (matrixgame_covered_crossed_shared), grid rendering.
  solo_grid_crossed) GAME=matrixgame_covered_solo_crossed_grid;    GAME_DIR=matrixgame_covered_solo_crossed_grid;         SRC=instances ;;
  dyad_grid_crossed) GAME=matrixgame_covered_comm_crossed_grid;    GAME_DIR=matrixgame_covered_comm_crossed_grid;         SRC=instances ;;
  3p_grid_crossed)   GAME=matrixgame_covered_comm_3p_crossed_grid; GAME_DIR=matrixgame_covered_comm_nplayer_crossed_grid; SRC=instances_3p ;;
  6p_grid_crossed)   GAME=matrixgame_covered_comm_6p_crossed_grid; GAME_DIR=matrixgame_covered_comm_nplayer_crossed_grid; SRC=instances_6p ;;
  *) echo "unknown SUITE '$SUITE'" >&2; exit 2 ;;
esac

LOGDIR="$RESULTS/_campaign"
mkdir -p "$LOGDIR"
CHAIN_LOG="$LOGDIR/chain.log"

# clem's openai_compatible backend disables TLS verification itself; the bundle
# keeps direct probes against the Potsdam endpoint working too.
export SSL_CERT_FILE="$REPO/.certs/clem-ca-bundle.pem"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

say() { echo "[$(date '+%F %T')] $SUITE: $*" | tee -a "$CHAIN_LOG"; }

# One driver per suite, enforced. Two drivers would deal overlapping shards and
# play the same episode twice into the same directory.
LOCK="$LOGDIR/driver.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  owner="$(cat "$LOCK/pid" 2>/dev/null || true)"
  if [[ -n "$owner" ]] && kill -0 "$owner" 2>/dev/null; then
    say "refusing to start: driver pid $owner already owns this suite"
    exit 3
  fi
  say "clearing stale lock from pid ${owner:-unknown}"
  rm -f "$LOCK/pid"; rmdir "$LOCK" 2>/dev/null || true
  mkdir "$LOCK" 2>/dev/null || { say "could not take lock"; exit 3; }
fi
echo $$ >"$LOCK/pid"

CHILDREN=()
release() { rm -f "$LOCK/pid"; rmdir "$LOCK" 2>/dev/null || true; }
stop() {
  trap - TERM INT
  say "signalled, stopping ${#CHILDREN[@]} shard process(es)"
  for child in ${CHILDREN[@]+"${CHILDREN[@]}"}; do kill "$child" 2>/dev/null || true; done
  release
  exit 143
}
trap stop TERM INT
trap release EXIT

say "SUITE_START $GAME model=$MODEL temp=$TEMP shards=$SHARDS results=$RESULTS pid=$$"
expected=0
prev=-1
stall=0

for ((pass = 1; pass <= MAX_PASSES; pass++)); do
  stats="$(python3 "$REPO/campaign/remaining.py" \
    --game "$GAME" --game-dir "$REPO/$GAME_DIR" --src "$SRC" \
    --out "instances_resume_$SUITE" --results "$RESULTS" --model "$MODEL" \
    --nshards "$SHARDS")" || { say "instance filter failed"; break; }

  rem="${stats#REMAINING=}"; rem="${rem%% *}"
  files="${stats##*SHARD_FILES=}"
  [[ $expected -eq 0 ]] && { expected="${stats#*TOTAL=}"; expected="${expected%% *}"; }
  say "pass $pass: $stats"

  [[ $rem -eq 0 ]] && break

  if [[ $rem -eq $prev ]]; then
    stall=$((stall + 1))
    if [[ $stall -ge $STALL_LIMIT ]]; then
      say "$STALL_LIMIT passes without progress ($rem still missing) - giving up on the rest"
      break
    fi
    say "no progress last pass, backing off $((60 * stall))s"
    sleep $((60 * stall))
  else
    stall=0
  fi
  prev=$rem

  pids=()
  for shard_file in ${files//,/ }; do
    clem run -g "$GAME" -m "$MODEL" -t "$TEMP" -i "$shard_file" -r "$RESULTS" \
      >>"$LOGDIR/$shard_file.log" 2>&1 &
    pids+=($!)
    CHILDREN+=($!)
    sleep 2   # stagger startup so the shards do not all hit the endpoint at once
  done
  say "pass $pass: launched ${#pids[@]} shard(s), waiting"
  for pid in "${pids[@]}"; do wait "$pid"; done
  say "pass $pass: all shards exited"
done

say "scoring"
clem score -g "$GAME" -r "$RESULTS" >>"$LOGDIR/score.log" 2>&1
say "score rc=$?"

# Verify every finished episode actually got scored. A scoring call that fails
# (bad args, unreadable tree) is silent otherwise, and the summary below would
# quietly describe a subset of the suite as if it were the whole thing.
played=$(find "$RESULTS" -name interactions.json | wc -l | tr -d " ")
scored=$(find "$RESULTS" -name scores.json | wc -l | tr -d " ")
if [ "$played" != "$scored" ]; then
  say "WARNING: $scored of $played episodes scored - retrying once"
  clem score -g "$GAME" -r "$RESULTS" >>"$LOGDIR/score.log" 2>&1
  scored=$(find "$RESULTS" -name scores.json | wc -l | tr -d " ")
  [ "$played" != "$scored" ] && say "WARNING: still $scored of $played scored - SUMMARY BELOW IS INCOMPLETE"
fi
say "scored $scored of $played episodes"
python3 "$REPO/campaign/summarize.py" --results "$RESULTS" --model "$MODEL" \
  --game "$GAME" --label "$SUITE" --expected "$expected" \
  >"$LOGDIR/summary_$SUITE.txt" 2>&1
cat "$LOGDIR/summary_$SUITE.txt" >>"$CHAIN_LOG"
say "SUITE_DONE"
