#!/usr/bin/env bash
# Dispatch a benchmark run to the cloud runner. Several runs can be live at
# once, each in its own tmux session — call dispatch.sh once per model.
#
# Usage:
#   cloud/dispatch.sh [--sync] <command...>
#
# Examples:
#   cloud/dispatch.sh 'clem run -g matrixgame_covered_solo -m minimax-m3 -t 0.6 -r results_minimax_t06'
#   cloud/dispatch.sh 'clem run ... -r results_x && clem score -r results_x'
#
# A single quoted argument runs as a shell line (&& chains work); multiple
# bare arguments are re-quoted verbatim. Give each parallel run its own
# results dir (-r) so scoring stays per-model.
#
# The working tree is rsynced up first, but only while no run is active, so
# code never changes under a live run: the first dispatch syncs, back-to-back
# later ones reuse that tree. --sync forces it (only safe if the tree is
# unchanged since the running dispatches started).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

FORCE_SYNC=0
if [ "${1:-}" = "--sync" ]; then
    FORCE_SYNC=1
    shift
fi
if [ $# -eq 0 ]; then
    sed -n '2,19p' "$0"
    exit 1
fi

if [ $# -eq 1 ]; then
    CMD_STR="$1"
else
    CMD_STR=""
    for a in "$@"; do CMD_STR+=$(printf '%q ' "$a"); done
fi

wait_for_ssh
echo "==> Server $SERVER_NAME at $SERVER_IP"

echo "==> Checking base packages"
ssh "${SSH_OPTS[@]}" "$R" '
    if ! command -v tmux >/dev/null 2>&1 || ! command -v rsync >/dev/null 2>&1; then
        cloud-init status --wait >/dev/null 2>&1 || true
        apt-get -qq update && apt-get -qq install -y tmux rsync python3-venv
    fi'

ACTIVE=$(ssh "${SSH_OPTS[@]}" "$R" "tmux list-sessions -F '#S' 2>/dev/null | grep -c '^bench-'" || true)
ACTIVE=${ACTIVE:-0}

if [ "$ACTIVE" -gt 0 ] && [ "$FORCE_SYNC" = 0 ]; then
    echo "==> $ACTIVE run(s) already active — skipping sync + dependency install"
    echo "    (server tree stays as dispatched; use --sync only if it is unchanged)"
else
    echo "==> Syncing working tree (~260 MB on first run, deltas afterwards)"
    rsync -az --delete -e "ssh ${SSH_OPTS[*]}" \
        --exclude .git \
        --exclude 'results*' \
        --exclude thesis \
        --exclude venv \
        --exclude __pycache__ \
        --exclude '*.pyc' \
        --exclude .DS_Store \
        --exclude '*.log' \
        --exclude cloud-runs \
        --exclude .idea --exclude .vscode --exclude .superpowers \
        "$REPO_DIR/" "$R:$REMOTE_DIR/"

    echo "==> Ensuring venv and dependencies"
    ssh "${SSH_OPTS[@]}" "$R" bash -s <<EOF
    set -e
    cd "$REMOTE_DIR"
    mkdir -p cloud-runs
    [ -d venv ] || python3 -m venv venv
    new=\$(cat requirements.txt cloud/requirements-extra.txt | md5sum | awk '{print \$1}')
    old=\$(cat cloud-runs/.pip-stamp 2>/dev/null || true)
    if [ "\$new" != "\$old" ]; then
        echo "    installing python deps (first time takes a few minutes)"
        venv/bin/pip install -q --upgrade pip
        venv/bin/pip install -q -r requirements.txt -r cloud/requirements-extra.txt
        echo "\$new" > cloud-runs/.pip-stamp
    fi
EOF
fi

RUN_ID=$(date +%Y%m%d-%H%M%S)
SESSION="bench-$RUN_ID"
echo "==> Launching run $RUN_ID in tmux session '$SESSION'"
RUNNER=$(mktemp)
cat > "$RUNNER" <<EOF
#!/usr/bin/env bash
cd "$REMOTE_DIR"
source venv/bin/activate
export SSL_CERT_FILE=\$PWD/.certs/clem-ca-bundle.pem
export REQUESTS_CA_BUNDLE=\$PWD/.certs/clem-ca-bundle.pem
export AWS_DEFAULT_REGION=eu-central-1
echo "=== clembench cloud run $RUN_ID ==="
echo "+ $CMD_STR"
$CMD_STR
status=\$?
echo "=== EXIT=\$status ==="
EOF
rsync -az -e "ssh ${SSH_OPTS[*]}" "$RUNNER" "$R:$REMOTE_DIR/cloud-runs/run-$RUN_ID.sh"
rm -f "$RUNNER"

ssh "${SSH_OPTS[@]}" "$R" \
    "cd $REMOTE_DIR && tmux new-session -d -s $SESSION 'bash cloud-runs/run-$RUN_ID.sh 2>&1 | tee cloud-runs/run-$RUN_ID.log'"

echo
echo "Dispatched. Your laptop can sleep now."
echo "  progress:  cloud/status.sh"
echo "  attach:    ssh -i $SSH_KEY $R -t tmux attach -t $SESSION   (detach: Ctrl-b d)"
echo "  collect:   cloud/fetch-results.sh"
