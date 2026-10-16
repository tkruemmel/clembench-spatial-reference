#!/usr/bin/env bash
# Scale each suite of a worker-based campaign to a target number of workers.
# Idempotent: run it again any time. Missing workers are started (lowest free
# ids), surplus workers get a stop_w<id> file and exit after their current
# episode - nothing in flight is ever discarded. Stale pid files are cleared.
#
#   MODEL=DeepSeek-V4-Flash TAG=dsflash RESULTS_PREFIX=results_dsflash_grid_n15 \
#   SUITES="solo_grid dyad_grid 3p_grid 6p_grid" TARGETS="4 11 10 7" bash campaign/scale_workers.sh
#
# Results root per suite: $RESULTS_PREFIX_<suite minus _grid>, e.g.
# results_dsflash_grid_n15_solo. On a laptop workers run under caffeinate so it
# does not sleep on them; on Linux they run bare.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL="${MODEL:?}"; TAG="${TAG:?}"; RESULTS_PREFIX="${RESULTS_PREFIX:?}"
export MODEL TAG TEMP="${TEMP:-0.6}" FAIL_LIMIT="${FAIL_LIMIT:-3}" MIN_BALANCE="${MIN_BALANCE:-}"
read -r -a suites <<<"${SUITES:?}"
read -r -a targets <<<"${TARGETS:?}"
[[ ${#suites[@]} -eq ${#targets[@]} ]] || { echo "SUITES and TARGETS differ in length" >&2; exit 2; }

# pid must still be a worker.sh: after a reboot the old pids belong to unrelated processes.
# caffeinate keeps a laptop awake; on a Linux server (Hetzner) it does not exist.
KEEPAWAKE=()
command -v caffeinate >/dev/null 2>&1 && KEEPAWAKE=(caffeinate -i -m -s)

alive() { ps -o command= -p "$1" 2>/dev/null | grep -q 'campaign/worker.sh'; }

for idx in "${!suites[@]}"; do
  suite="${suites[$idx]}"; target="${targets[$idx]}"
  # %%_grid* not %_grid: suite names carry the rendering and the prompt arm
  # (6p_grid_updated_prompt), the results tree only the team size (..._6p).
  results="$PWD/${RESULTS_PREFIX}_${suite%%_grid*}"
  logdir="$results/_campaign"
  mkdir -p "$logdir"

  live=()
  for pidfile in "$logdir"/worker_w*.pid; do
    [[ -e "$pidfile" ]] || continue
    id="${pidfile##*worker_w}"; id="${id%.pid}"
    if alive "$(cat "$pidfile")"; then
      live+=("$id")
    else
      rm -f "$pidfile"
    fi
  done
  # Sorted numerically; stop files of live workers we intend to keep are cleared.
  IFS=$'\n' live=($(printf '%s\n' ${live[@]+"${live[@]}"} | sort -n)); unset IFS
  rm -f "$logdir/stop"

  n=${#live[@]}
  if (( n > target )); then
    for id in "${live[@]:target}"; do
      touch "$logdir/stop_w$id"
      echo "$suite: worker w$id will stop after its current episode"
    done
    for id in "${live[@]:0:target}"; do rm -f "$logdir/stop_w$id"; done
  else
    for id in ${live[@]+"${live[@]}"}; do rm -f "$logdir/stop_w$id"; done
    id=0
    while (( n < target )); do
      if printf '%s\n' ${live[@]+"${live[@]}"} | grep -qx "$id"; then id=$((id + 1)); continue; fi
      rm -f "$logdir/stop_w$id"
      SUITE="$suite" WORKER="$id" RESULTS="$results" nohup ${KEEPAWAKE[@]+"${KEEPAWAKE[@]}"} \
        bash campaign/worker.sh >>"$logdir/nohup.log" 2>&1 &
      disown
      echo "$suite: started worker w$id -> $results"
      n=$((n + 1)); id=$((id + 1))
      sleep 2
    done
  fi
  echo "$suite: target $target, now ${n} worker(s) (${#live[@]} were live)"
done
