#!/usr/bin/env bash
# Pull results and run logs back from the cloud runner.
#
# Usage:
#   cloud/fetch-results.sh
#
# Rsyncs every remote results* directory and cloud-runs/ (run logs) into the
# repo. Safe while runs are active — you just get a snapshot of progress.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

wait_for_ssh

ACTIVE=$(ssh "${SSH_OPTS[@]}" "$R" "tmux list-sessions -F '#S' 2>/dev/null | grep -c '^bench-'" || true)
[ "${ACTIVE:-0}" -gt 0 ] && echo "note: $ACTIVE run(s) still in progress — fetching what exists so far."

echo "==> Fetching results"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$R:$REMOTE_DIR/results*" "$REPO_DIR/" 2>/dev/null \
    || echo "    (no results directories on the server yet)"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$R:$REMOTE_DIR/cloud-runs" "$REPO_DIR/" 2>/dev/null \
    || echo "    (no run logs on the server yet)"
echo "    synced into $REPO_DIR (results* and cloud-runs/)"

LATEST=$(ls -t "$REPO_DIR"/cloud-runs/run-*.log 2>/dev/null | head -1 || true)
if [ -n "$LATEST" ]; then
    echo "==> Tail of latest run log ($LATEST):"
    tail -n 10 "$LATEST"
fi
