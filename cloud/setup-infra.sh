#!/usr/bin/env bash
# One-time setup of the clembench benchmark runner on Hetzner Cloud.
#
# Prerequisites:
#   brew install hcloud
#   hcloud context create clembench     (paste a read/write API token)
#   ssh key added to the Hetzner project (console or hcloud ssh-key create)
#
# Creates (all named "clembench-runner", idempotent — safe to re-run):
#   - firewall allowing inbound ssh only (Hetzner has no default filtering)
#   - CX33 Ubuntu 24.04 x86 server (4 vCPU, 8 GB — sized for parallel runs),
#     always on, ~11 EUR/month all-in (CAX ARM was sold out at setup time)
#   - cloud/config.env consumed by dispatch.sh / fetch-results.sh / status.sh
#
# Ends with a reachability test of the Potsdam endpoint (blocked from AWS).
set -euo pipefail

NAME="clembench-runner"
SERVER_TYPE="${SERVER_TYPE:-cx33}"
LOCATION="${LOCATION:-fsn1}"
IMAGE="ubuntu-24.04"
PUBKEY="$HOME/.ssh/id_ed25519.pub"

CLOUD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_OPTS=(-i "${PUBKEY%.pub}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)

echo "==> Verifying hcloud CLI and token"
command -v hcloud >/dev/null 2>&1 || { echo "error: hcloud not installed (brew install hcloud)" >&2; exit 1; }
[ -n "$(hcloud context active 2>/dev/null)" ] || { echo "error: no hcloud context. Run: hcloud context create clembench" >&2; exit 1; }
hcloud server list >/dev/null

echo "==> SSH key"
FP=$(ssh-keygen -E md5 -lf "$PUBKEY" | awk '{print $2}' | sed 's/^MD5://')
KEY_NAME=$(hcloud ssh-key list -o noheader -o columns=name,fingerprint | awk -v fp="$FP" '$2==fp {print $1}')
if [ -z "$KEY_NAME" ]; then
    KEY_NAME="tom-ed25519"
    hcloud ssh-key create --name "$KEY_NAME" --public-key-from-file "$PUBKEY" >/dev/null
    echo "    created key '$KEY_NAME'"
else
    echo "    using existing key '$KEY_NAME'"
fi

echo "==> Firewall (inbound ssh only)"
if ! hcloud firewall describe "$NAME" >/dev/null 2>&1; then
    hcloud firewall create --name "$NAME" >/dev/null
    hcloud firewall add-rule "$NAME" --direction in --protocol tcp --port 22 \
        --source-ips 0.0.0.0/0 --source-ips ::/0 >/dev/null
    echo "    created"
else
    echo "    already exists"
fi

echo "==> Server"
if hcloud server describe "$NAME" >/dev/null 2>&1; then
    echo "    reusing existing server"
else
    USERDATA=$(mktemp)
    cat > "$USERDATA" <<'EOF'
#!/bin/bash
set -e
apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv rsync tmux
fallocate -l 4G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
EOF
    echo "    launching $SERVER_TYPE ($IMAGE) in $LOCATION"
    hcloud server create --name "$NAME" --type "$SERVER_TYPE" --image "$IMAGE" \
        --location "$LOCATION" --ssh-key "$KEY_NAME" --firewall "$NAME" \
        --user-data-from-file "$USERDATA" >/dev/null
    rm -f "$USERDATA"
fi
IP=$(hcloud server ip "$NAME")
echo "    $NAME at $IP"

echo "==> Writing cloud/config.env"
cat > "$CLOUD_DIR/config.env" <<EOF
SERVER_NAME=$NAME
SERVER_IP=$IP
REMOTE_USER=root
REMOTE_DIR=/root/clembench
SSH_KEY=${PUBKEY%.pub}
EOF

echo "==> Waiting for ssh + cloud-init (first boot installs packages)"
for i in $(seq 1 40); do
    if ssh "${SSH_OPTS[@]}" -o ConnectTimeout=5 "root@$IP" true 2>/dev/null; then break; fi
    sleep 5
done
ssh "${SSH_OPTS[@]}" "root@$IP" \
    'cloud-init status --wait >/dev/null 2>&1 || true; command -v tmux >/dev/null && command -v rsync >/dev/null' \
    && echo "    server is ready"

echo "==> Testing Potsdam endpoint (jarvis.ling.uni-potsdam.de — was blocked from AWS)"
ssh "${SSH_OPTS[@]}" "root@$IP" \
    "if curl -ksS -o /dev/null --connect-timeout 10 https://jarvis.ling.uni-potsdam.de/; then echo '    reachable'; else echo '    UNREACHABLE (endpoint likely allowlists university networks only)'; fi"

echo
echo "Done. Next: cloud/dispatch.sh 'clem run -g <game> -m <model> -t 0.6 -r results_<model>'"
