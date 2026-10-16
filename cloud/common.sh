# Shared config and helpers for the clembench cloud runner scripts.
# Sourced by dispatch.sh / fetch-results.sh / status.sh — not executable on its own.

CLOUD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$CLOUD_DIR")"
CONFIG="$CLOUD_DIR/config.env"

if [ ! -f "$CONFIG" ]; then
    echo "error: $CONFIG not found — run cloud/setup-infra.sh first" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$CONFIG"

SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)
R="$REMOTE_USER@$SERVER_IP"

# The server is always on; this just tolerates a reboot or brief network blip.
wait_for_ssh() {
    local i
    for i in $(seq 1 12); do
        if ssh "${SSH_OPTS[@]}" -o ConnectTimeout=5 "$R" true 2>/dev/null; then
            return 0
        fi
        sleep 5
    done
    echo "error: could not reach $R via ssh after 60s (server down? check: hcloud server list)" >&2
    exit 1
}
