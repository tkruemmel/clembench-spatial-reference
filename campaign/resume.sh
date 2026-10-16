#!/usr/bin/env bash
# Resume the campaign after a reboot, sleep or kill. Idempotent: each suite's
# driver takes a lock, then replays only the episodes with no interactions.json.
exec bash "$(dirname "${BASH_SOURCE[0]}")/launch_all.sh"
