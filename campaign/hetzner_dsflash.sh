#!/usr/bin/env bash
# Run the DeepSeek-V4-Flash grid campaign on the Hetzner runner (cloud/config.env)
# instead of the laptop. api.deepseek.com needs no VPN, so the worker/claim
# campaign (dsflash_grid.sh) runs there unchanged; the laptop keeps a mirror of
# the four results trees via `fetch`.
#
#   bash campaign/hetzner_dsflash.sh sync             # working tree + the four DS results trees -> server
#   SOLO=3 DYAD=8 P3=8 P6=5 bash campaign/hetzner_dsflash.sh launch   # (re)scale the workers on the server
#   bash campaign/hetzner_dsflash.sh stop             # drain all remote workers
#   bash campaign/hetzner_dsflash.sh status [ds_status args]
#   bash campaign/hetzner_dsflash.sh fetch            # server results trees -> laptop mirror
#   bash campaign/hetzner_dsflash.sh ssh [cmd]
#
# The claim state lives in the results trees, so exactly ONE machine may run
# workers on them: sync/fetch refuse while local DS workers are alive, and the
# local trees are a read-only mirror once the campaign runs on the server.
# The ssh key (cloud/config.env SSH_KEY) is passphrase-protected: load it into
# the agent first (ssh-add --apple-use-keychain ~/.ssh/id_ed25519).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source cloud/common.sh

PREFIX="${RESULTS_PREFIX:-results_dsflash_grid_n15}"
# Which campaign wrapper `launch`/`stop` runs on the server. Set together with
# RESULTS_PREFIX to drive another model's campaign through the same plumbing,
# e.g. RESULTS_PREFIX=results_dsv41_grid_n15 CAMPAIGN_SCRIPT=campaign/dsv41_grid.sh.
CAMPAIGN_SCRIPT="${CAMPAIGN_SCRIPT:-campaign/dsflash_grid.sh}"
# Which suites sync/fetch move. Override for a single-suite arm, e.g. SUITES=6p.
read -r -a _suites <<<"${SUITES:-solo dyad 3p 6p}"
TREES=("${_suites[@]/#/${PREFIX}_}")
TREE_EXCLUDES=(--exclude '_campaign/claims/' --exclude '_campaign/*.pid' --exclude '_campaign/nohup.log' --exclude '.DS_Store')

local_workers_alive() {
  local pidfile pid
  for pidfile in "${TREES[@]/%//_campaign/worker_w*.pid}"; do
    for pidfile in $pidfile; do
      [[ -e "$pidfile" ]] || continue
      pid=$(cat "$pidfile")
      ps -o command= -p "$pid" 2>/dev/null | grep -q 'campaign/worker.sh' && return 0
    done
  done
  return 1
}

remote() { ssh "${SSH_OPTS[@]}" "$R" "$@"; }

cmd="${1:-}"; shift || true
case "$cmd" in
  sync)
    if local_workers_alive; then
      echo "local DeepSeek workers are alive - stop them first (bash campaign/dsflash_grid.sh stop, or kill the worker.sh pids)" >&2; exit 1
    fi
    wait_for_ssh
    echo "==> working tree -> $R:$REMOTE_DIR"
    rsync -az --delete -e "ssh ${SSH_OPTS[*]}" \
      --exclude .git --exclude 'results*' --exclude thesis --exclude venv --exclude __pycache__ \
      --exclude '*.pyc' --exclude .DS_Store --exclude '*.log' --exclude cloud-runs --exclude .worktrees \
      --exclude .idea --exclude .vscode --exclude .superpowers --exclude '*.pdf' \
      ./ "$R:$REMOTE_DIR/"
    echo "==> results trees -> server (${TREES[*]})"
    rsync -az -e "ssh ${SSH_OPTS[*]}" "${TREE_EXCLUDES[@]}" "${TREES[@]}" "$R:$REMOTE_DIR/"
    echo "==> venv + deps, stale claims/pids/partials cleared on the server"
    remote bash -s <<REMOTE
      set -e; cd "$REMOTE_DIR"; mkdir -p cloud-runs
      [ -d venv ] || python3 -m venv venv
      new=\$(cat requirements.txt cloud/requirements-extra.txt | md5sum | awk '{print \$1}')
      old=\$(cat cloud-runs/.pip-stamp 2>/dev/null || true)
      if [ "\$new" != "\$old" ]; then venv/bin/pip install -q --upgrade pip; venv/bin/pip install -q -r requirements.txt -r cloud/requirements-extra.txt; echo "\$new" > cloud-runs/.pip-stamp; fi
      for t in ${TREES[*]}; do
        rm -rf "\$t/_campaign/claims" "\$t"/_campaign/worker_w*.pid; mkdir -p "\$t/_campaign/claims"
        find "\$t" -name instance.json | while read -r f; do e=\$(dirname "\$f"); [ -e "\$e/interactions.json" ] || { [ "\$(ls "\$e" | tr '\n' ' ')" = "instance.json " ] && rm -rf "\$e"; }; done
        echo "  \$t: \$(find "\$t" -name interactions.json | wc -l) finished episodes"
      done
      venv/bin/clem --version 2>/dev/null || venv/bin/pip show clemcore | grep -i ^version
REMOTE
    ;;
  launch|stop)
    wait_for_ssh
    args=""; [[ "$cmd" == stop ]] && args=stop
    # MODEL/TAG/RESULTS_PREFIX pass through only when set; empty leaves the campaign script's defaults.
    remote "cd $REMOTE_DIR && source venv/bin/activate && export SSL_CERT_FILE=\$PWD/.certs/clem-ca-bundle.pem REQUESTS_CA_BUNDLE=\$PWD/.certs/clem-ca-bundle.pem && ARM=${ARM:-} SUITES='${_suites[*]}' MODEL='${MODEL:-}' TAG=${TAG:-} RESULTS_PREFIX=${RESULTS_PREFIX:-} SOLO=${SOLO:-3} DYAD=${DYAD:-8} P3=${P3:-8} P6=${P6:-5} MIN_BALANCE=${MIN_BALANCE:-12} bash $CAMPAIGN_SCRIPT $args"
    ;;
  status)
    remote "cd $REMOTE_DIR && ARM=${ARM:-} SUITES='${_suites[*]}' venv/bin/python3 campaign/ds_status.py $*"
    ;;
  fetch)
    if local_workers_alive; then echo "local DeepSeek workers are alive - refusing to overwrite their trees" >&2; exit 1; fi
    wait_for_ssh
    for t in "${TREES[@]}"; do
      rsync -az -e "ssh ${SSH_OPTS[*]}" "${TREE_EXCLUDES[@]}" "$R:$REMOTE_DIR/$t" ./
      echo "  $t: $(find "$t" -name interactions.json | wc -l | tr -d ' ') finished episodes (mirror)"
    done
    ;;
  ssh)
    remote "$@"
    ;;
  *)
    sed -n '2,19p' "$0"; exit 1
    ;;
esac
