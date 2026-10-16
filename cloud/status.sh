#!/usr/bin/env bash
# Show what the cloud runner is doing: active runs and their log tails.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

echo "server:   $SERVER_NAME  ip: $SERVER_IP"
if ! ssh "${SSH_OPTS[@]}" -o ConnectTimeout=5 "$R" true 2>/dev/null; then
    echo "state:    UNREACHABLE (check: hcloud server list)"
    exit 1
fi

ssh "${SSH_OPTS[@]}" "$R" bash -s <<EOF
sessions=\$(tmux list-sessions -F '#S' 2>/dev/null | grep '^bench-' || true)
if [ -n "\$sessions" ]; then
    echo "runs:     \$(echo "\$sessions" | wc -l | tr -d ' ') IN PROGRESS"
    for s in \$sessions; do
        log="$REMOTE_DIR/cloud-runs/run-\${s#bench-}.log"
        echo
        echo "--- \$s — last 6 lines of \$(basename "\$log") ---"
        tail -n 6 "\$log" 2>/dev/null || echo "(no log yet)"
    done
else
    echo "runs:     none active"
    latest=\$(ls -t $REMOTE_DIR/cloud-runs/run-*.log 2>/dev/null | head -1)
    if [ -n "\$latest" ]; then
        echo "log:      \$latest"
        echo "--- last 15 lines ---"
        tail -n 15 "\$latest"
    fi
fi
EOF
