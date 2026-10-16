# Cloud runner

Dispatch benchmark runs to a Hetzner Cloud server (`clembench-runner`, CX33
x86, 4 vCPU / 8 GB, Falkenstein) so they keep running while the laptop sleeps.
Several runs can be live at once, each in its own tmux session — sized for
5-10 parallel model runs (the workload is API-bound, not CPU-bound).

## One-time setup

```sh
brew install hcloud
hcloud context create clembench    # paste a read/write API token
cloud/setup-infra.sh
```

Idempotent — safe to re-run. Creates a firewall (inbound ssh only) and the
server, writes `cloud/config.env`, and ends with a reachability test of the
Potsdam endpoint. The ssh key is picked up from the Hetzner project (matched
against `~/.ssh/id_ed25519.pub`, created there if missing).

## Daily use

```sh
# one dispatch per model — they run in parallel, each in its own tmux session
cloud/dispatch.sh 'clem run -g matrixgame_covered_solo -m minimax-m3 -t 0.6 -r results_minimax_t06'
cloud/dispatch.sh 'clem run -g matrixgame_covered_solo -m deepseek-flash -t 0.6 -r results_deepseek_t06'

cloud/status.sh          # all active runs + per-run log tails
cloud/fetch-results.sh   # pull results*/ + cloud-runs/ logs back
```

- A single quoted argument runs as a shell line (`&&` chains work); multiple
  bare arguments are re-quoted verbatim.
- Give each parallel run its own results dir (`-r results_<model>_t06`) and
  score/transcribe with the same `-r`. Without it everything lands in a shared
  `results/` and `clem score` would walk other runs' episodes too.
- The working tree is rsynced up only while no run is active: the first
  dispatch syncs, back-to-back later ones reuse that tree (dispatch says so).
  `--sync` forces a sync — only safe if the tree is unchanged since.
- Runs execute inside tmux session `bench-<run-id>` with `SSL_CERT_FILE`/
  `REQUESTS_CA_BUNDLE` pointed at `.certs/clem-ca-bundle.pem` (Potsdam
  endpoint) and `AWS_DEFAULT_REGION=eu-central-1` (Bedrock).
- Attach to a live run: `ssh -i ~/.ssh/id_ed25519 root@<ip> -t tmux attach -t bench-<run-id>`
  (ip and session names printed by `status.sh`; detach with `Ctrl-b d`).
- venv on the server installs `requirements.txt` plus
  `cloud/requirements-extra.txt`. Deps are reinstalled only when either file
  changes, and never while runs are active.
- Bedrock models: the EC2 instance role is gone, so `*-bedrock` runs need
  static AWS keys on the server (e.g. `~/.aws/credentials` for root, keys for
  the `Tom-Bedrock-Access` user). Fireworks/Potsdam models need nothing —
  their keys travel in `key.json` with the sync.
- Parallel runs against the same API provider share that account's rate
  limits; spreading models across providers parallelises best.

## Cost

Flat ~11 EUR/month (CX33 ~10.1 EUR + ~0.6 EUR IPv4), always on — Hetzner bills
powered-off servers at full rate, so there is no stop/start choreography.
API tokens, not the server, are the real cost driver. (CAX ARM at ~6.5 EUR
was sold out at setup time; the workload is API-bound, so the CPU type is
irrelevant. Switching to ARM later would mean delete + re-run setup-infra.sh —
cross-architecture change-type is not possible.)

## Teardown

```sh
hcloud server delete clembench-runner
hcloud firewall delete clembench-runner
```

## Old AWS runner

The previous EC2-based scripts live in `cloud/aws-legacy/`. The instance
`i-0e11e2de97bbface5` (eu-central-1) still exists in stopped state
(~$0.05/month for its disk). Terminate it once the migration has proven out:

```sh
aws ec2 terminate-instances --instance-ids i-0e11e2de97bbface5 \
    --profile AWSAdministratorAccess-106197011510 --region eu-central-1
```
