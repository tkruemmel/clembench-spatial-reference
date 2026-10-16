# ECS runner

Runs the matrixgame **grid** suites on AWS Fargate against Bedrock models. Built
for the fan-out the Hetzner box cannot do: eight suite drivers at once, each with
its own concurrency, all writing into one S3 mirror.

Shape of a campaign: **one Fargate task per (model, suite)**. Each task is a
`campaign/run_chain.sh` driver with `SHARDS=$PROCS` concurrent `clem run`
processes. So `PROCS` *is* the shard size, and the total number of Bedrock
streams is `tasks x PROCS`.

Why not one task per shard: Fargate bills per vCPU and the account's
`Fargate On-Demand vCPU` quota is the binding limit, while the workload is
API-bound and a fraction of a vCPU carries several streams. Packing shards into
a task buys concurrency without buying vCPU. It also preserves the campaign
driver's "one driver per suite" invariant for free — two drivers on one suite
deal overlapping shards and replay each other's episodes.

## One-time setup

```sh
aws sso login --profile AWSAdministratorAccess-106197011510
cloud/ecs/setup-infra.sh          # idempotent; writes cloud/ecs/config.env
```

Creates: S3 bucket `clembench-ecs-<account>`, ECR repo `clembench-runner`,
ECS cluster `clembench`, log group `/ecs/clembench`, an execution role, a task
role scoped to Bedrock invoke + that one bucket, and an egress-only security
group in the default VPC. Prints the Fargate vCPU quota — 8 tasks at 1 vCPU need
8, and the default in a fresh region is 6.

## Running a campaign

```sh
IMAGE=$(cloud/ecs/build-push.sh | tail -1)      # ARM64, staged subset of the tree
cloud/ecs/ctl.py launch --run-id grid-1 --image "$IMAGE" --procs 4
cloud/ecs/ctl.py status --run-id grid-1         # progress + throttling + verdict
cloud/ecs/ctl.py scale  --run-id grid-1 --procs 8
cloud/ecs/ctl.py fetch  --run-id grid-1 --score
```

`status` is the check that decides shard size. It prints, per (model, suite),
episodes finished and the rate since the previous `status` call; then Bedrock's
own `Invocations` / `InvocationThrottles` / `InvocationLatency`; then container
CPU against its reservation; then a verdict. The three numbers separate the
three reasons throughput stops rising:

| symptom | meaning | move |
|---|---|---|
| ep/h per stream fell after raising `--procs` | past the knee | go back |
| throttles > 10% | Bedrock is the limit; retries burn wall-clock | halve `--procs` |
| throttles 3–10% | at the knee | hold |
| server errors > 2% | load is costing episodes | do not push harder |
| CPU peak > 85% of reservation | the container is the limit | relaunch with a bigger `--cpu` |
| all low | nothing is saturated | double `--procs` |

**Throttles are not the primary signal.** Bedrock answers past-capacity load by
queueing, not by returning `ThrottlingException`: going 32 → 64 streams on
2026-09-04 left throttling at exactly 0% while latency went 0.9 s → 4.7 s. Judge
on delivered episodes/hour per stream first; treat throttles, server errors and
latency as corroboration.

**Two traps in reading the rate.** Counts come from S3, which each task
refreshes only every `--sync-seconds` (default 60), so any single reading trails
reality by up to one push interval. The *delta* between two readings is still
unbiased — the same lag sits on both ends — **unless a task restarted in
between**, because a stopping task does a final flush that dumps a whole
interval of backlog into the later reading. `status` detects a changed task
generation and refuses to report a rate across it. Comparing a flushed total
against a mid-cycle one is what briefly made 64 streams look *slower* than 32
when it was in fact 12% faster.

Rates are computed against the snapshot written by the previous `status`, so
call it on a regular cadence and ignore the first reading.

`scale` stops the affected tasks, **waits until they are really STOPPED**, then
starts replacements. Never start a second task for a suite that still has one.

## Resume

An episode counts as finished iff its `interactions.json` exists — clemcore
writes it from `on_game_end` only, so an episode killed mid-flight leaves nothing
and is replayed, while an episode the game legitimately aborted keeps its result.
Each task pulls the suite's tree from S3 before starting and pushes it back every
`SYNC_SECONDS` (default 180) and on SIGTERM. Stopping a task therefore costs at
most the episodes in flight, which is what makes rescaling cheap.

Fargate's `stopTimeout` is capped at 120 s; the entrypoint spends that window
killing its shards and running a final push.

## Costs

Fargate is the cheap part: 8 tasks x 1 vCPU / 4 GiB ARM64 is about 0.30 USD/h.
Bedrock tokens dominate — a full 840-episode campaign is ~50k calls per model.
`status` prints `tokens_in` / `tokens_out` per model so the spend is visible
while it runs.

Nothing here is billed when no task is running: no NAT gateway, no load
balancer, no always-on instance. S3 keeps the results (cents) and the ECR
lifecycle policy keeps only the last 10 images.

## Teardown

```sh
cloud/ecs/ctl.py stop --run-id grid-1 --wait
aws ecs delete-cluster --cluster clembench --profile "$PROFILE" --region "$REGION"
aws ecr delete-repository --repository-name clembench-runner --force --profile "$PROFILE" --region "$REGION"
# the bucket holds the only copy of anything not yet fetched - fetch first
```

## Relation to the other runners

`cloud/` (Hetzner) stays the runner for Potsdam-endpoint models: the Potsdam
endpoint is unreachable from cloud providers, and those runs need the CA bundle
and long-lived tmux sessions. This directory is for Bedrock models only, where
the task role removes the need for any key material in the image.
