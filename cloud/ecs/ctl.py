#!/usr/bin/env python3
"""Control plane for the matrixgame grid campaign on ECS Fargate.

    cloud/ecs/ctl.py launch  --run-id grid-1 --image <uri> --procs 4
    cloud/ecs/ctl.py status  --run-id grid-1
    cloud/ecs/ctl.py scale   --run-id grid-1 --procs 8
    cloud/ecs/ctl.py stop    --run-id grid-1
    cloud/ecs/ctl.py fetch   --run-id grid-1

One task per (model, suite). "Shard size" is PROCS: the number of concurrent
`clem run` processes inside a task. `scale` changes it by stopping the tasks and
starting them again - the S3 mirror makes that cost only the episodes in flight.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
STATE = HERE / ".state"

# model key -> (clem model name in model_registry.json, Bedrock model id)
MODELS = {
    "devstral": ("devstral-2-bedrock", "mistral.devstral-2-123b"),
    "nemotron": ("nemotron-super-3-120b-bedrock", "nvidia.nemotron-super-3-120b"),
}
# suite -> (clem game name, episodes in the suite)
SUITES = {
    "solo_grid": ("matrixgame_covered_solo_grid", 120),
    "dyad_grid": ("matrixgame_covered_comm_grid", 240),
    "3p_grid": ("matrixgame_covered_comm_3p_grid", 240),
    "6p_grid": ("matrixgame_covered_comm_6p_grid", 240),
    # Compact (listing-rendered) originals; same boards, run_chain.sh knows them.
    "solo": ("matrixgame_covered_solo", 120),
    "dyad": ("matrixgame_covered_comm", 240),
    "3p": ("matrixgame_covered_comm_3p", 240),
    "6p": ("matrixgame_covered_comm_6p", 240),
    # Second prompt arm of the grid twin (prompt_updated/); same boards, no solo.
    "dyad_grid_updated_prompt": ("matrixgame_covered_comm_grid_updated_prompt", 240),
    "3p_grid_updated_prompt": ("matrixgame_covered_comm_3p_grid_updated_prompt", 240),
    "6p_grid_updated_prompt": ("matrixgame_covered_comm_6p_grid_updated_prompt", 240),
}
# Shortest gap between two readings that yields a usable rate: comfortably
# more than one --sync-seconds push interval.
MIN_RATE_WINDOW_S = 240
FAMILY = "clembench-runner"
CONTAINER = "runner"


def config():
    cfg = {}
    for line in (HERE / "config.env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            cfg[key] = value
    return cfg


class Aws:
    def __init__(self, cfg):
        self.cfg = cfg
        self.session = boto3.Session(profile_name=cfg["PROFILE"], region_name=cfg["REGION"])
        self.ecs = self.session.client("ecs")
        self.s3 = self.session.client("s3")
        self.cw = self.session.client("cloudwatch")
        self.logs = self.session.client("logs")


def pairs(args):
    models = args.models.split(",") if args.models else list(MODELS)
    suites = args.suites.split(",") if args.suites else list(SUITES)
    for model in models:
        for suite in suites:
            if model not in MODELS:
                sys.exit(f"unknown model '{model}' (known: {', '.join(MODELS)})")
            if suite not in SUITES:
                sys.exit(f"unknown suite '{suite}' (known: {', '.join(SUITES)})")
            yield model, suite


# ---------------------------------------------------------------- task control

def register(aws, image, cpu, memory):
    cfg = aws.cfg
    response = aws.ecs.register_task_definition(
        family=FAMILY,
        networkMode="awsvpc",
        requiresCompatibilities=["FARGATE"],
        cpu=str(cpu),
        memory=str(memory),
        runtimePlatform={"cpuArchitecture": "ARM64", "operatingSystemFamily": "LINUX"},
        executionRoleArn=cfg["EXEC_ROLE_ARN"],
        taskRoleArn=cfg["TASK_ROLE_ARN"],
        containerDefinitions=[{
            "name": CONTAINER,
            "image": image,
            "essential": True,
            # Fargate's ceiling. The entrypoint uses the window to stop its shards
            # and push the finished episodes to S3.
            "stopTimeout": 120,
            "environment": [{"name": "S3_BUCKET", "value": cfg["BUCKET"]}],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": cfg["LOG_GROUP"],
                    "awslogs-region": cfg["REGION"],
                    "awslogs-stream-prefix": "run",
                    "mode": "blocking",
                },
            },
        }],
    )
    arn = response["taskDefinition"]["taskDefinitionArn"]
    print(f"task definition {arn} ({cpu} cpu / {memory} MiB, ARM64)")
    return arn


def run_task(aws, taskdef, run_id, model, suite, procs, temp, sync_seconds=60):
    cfg = aws.cfg
    clem_model, _ = MODELS[model]
    response = aws.ecs.run_task(
        cluster=cfg["CLUSTER"],
        taskDefinition=taskdef,
        launchType="FARGATE",
        platformVersion="LATEST",
        count=1,
        startedBy=run_id,
        networkConfiguration={"awsvpcConfiguration": {
            "subnets": cfg["SUBNETS"].split(","),
            "securityGroups": [cfg["SECURITY_GROUP"]],
            "assignPublicIp": "ENABLED",
        }},
        overrides={"containerOverrides": [{
            "name": CONTAINER,
            "environment": [
                {"name": "RUN_ID", "value": run_id},
                {"name": "MODEL_KEY", "value": model},
                {"name": "CLEM_MODEL", "value": clem_model},
                {"name": "SUITE", "value": suite},
                {"name": "PROCS", "value": str(procs)},
                {"name": "TEMP", "value": str(temp)},
                {"name": "SYNC_SECONDS", "value": str(sync_seconds)},
            ],
        }]},
    )
    for failure in response.get("failures", []):
        print(f"  FAILED {model}/{suite}: {failure.get('reason')} {failure.get('detail','')}")
    tasks = response.get("tasks", [])
    return tasks[0]["taskArn"] if tasks else None


def live_tasks(aws, run_id):
    """[(arn, task)] for every task of this run that has not stopped."""
    cfg = aws.cfg
    arns = []
    paginator = aws.ecs.get_paginator("list_tasks")
    for page in paginator.paginate(cluster=cfg["CLUSTER"], startedBy=run_id,
                                   desiredStatus="RUNNING"):
        arns += page["taskArns"]
    for page in paginator.paginate(cluster=cfg["CLUSTER"], startedBy=run_id,
                                   desiredStatus="PENDING"):
        arns += page["taskArns"]
    if not arns:
        return []
    described = []
    for i in range(0, len(arns), 100):
        described += aws.ecs.describe_tasks(cluster=cfg["CLUSTER"],
                                            tasks=arns[i:i + 100])["tasks"]
    return described


def task_env(task):
    for override in task.get("overrides", {}).get("containerOverrides", []):
        if override.get("name") == CONTAINER:
            return {e["name"]: e["value"] for e in override.get("environment", [])}
    return {}


# ------------------------------------------------------------------- progress

def episode_counts(aws, run_id):
    """(model, suite) -> finished episodes, straight from the S3 mirror."""
    counts = {}
    paginator = aws.s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=aws.cfg["BUCKET"], Prefix=f"{run_id}/"):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith("/interactions.json"):
                continue
            parts = obj["Key"].split("/")
            if len(parts) > 3:
                counts[(parts[1], parts[2])] = counts.get((parts[1], parts[2]), 0) + 1
    return counts


def bedrock_metrics(aws, minutes):
    """Bedrock's own view: calls, throttles and latency per model, last N min."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=minutes)
    period = max(60, minutes * 60)
    out = {}
    for model, (_, model_id) in MODELS.items():
        row = {}
        for metric, stat in (("Invocations", "Sum"), ("InvocationThrottles", "Sum"),
                             ("InvocationServerErrors", "Sum"), ("InvocationLatency", "Average"),
                             ("InputTokenCount", "Sum"), ("OutputTokenCount", "Sum")):
            points = aws.cw.get_metric_statistics(
                Namespace="AWS/Bedrock", MetricName=metric,
                Dimensions=[{"Name": "ModelId", "Value": model_id}],
                StartTime=start, EndTime=end, Period=period, Statistics=[stat],
            )["Datapoints"]
            row[metric] = sum(p[stat] for p in points) if stat == "Sum" else (
                sum(p[stat] for p in points) / len(points) if points else 0.0)
        out[model] = row
    return out


def container_cpu(aws, minutes):
    """Max CPU/memory utilisation across the fleet - is PROCS CPU-bound yet?"""
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=minutes)
    out = {}
    for metric in ("CpuUtilized", "CpuReserved", "MemoryUtilized", "MemoryReserved"):
        points = aws.cw.get_metric_statistics(
            Namespace="ECS/ContainerInsights", MetricName=metric,
            Dimensions=[{"Name": "ClusterName", "Value": aws.cfg["CLUSTER"]}],
            StartTime=start, EndTime=end, Period=60, Statistics=["Maximum", "Average"],
        )["Datapoints"]
        out[metric] = (max((p["Maximum"] for p in points), default=0.0),
                       (sum(p["Average"] for p in points) / len(points)) if points else 0.0)
    return out


def snapshot_path(run_id):
    return STATE / f"{run_id}.snapshot.json"


# --------------------------------------------------------------- subcommands

def check_instances(image, force=False):
    """Refuse to launch an image whose baked-in instances no longer match the
    working tree. Regenerating instances after a build and launching anyway
    plays the wrong boards and the run looks perfectly healthy while doing it."""
    fingerprint = STATE / f"image-{image.rsplit(':', 1)[-1]}.json"
    if not fingerprint.exists():
        print(f"! no fingerprint for {image} - cannot verify its instances "
              f"(built before this check existed, or by another checkout)")
        return
    baked = json.loads(fingerprint.read_text())["instances"]
    stale = [rel for rel, digest in baked.items()
             if hashlib.sha256((REPO / rel).read_bytes()).hexdigest() != digest]
    if not stale:
        print(f"instances in {image.rsplit('/', 1)[-1]} match the working tree")
        return
    message = ("image is STALE - these instances changed since it was built:\n  "
               + "\n  ".join(stale) + "\n  rebuild with cloud/ecs/build-push.sh")
    if force:
        print(f"! {message}\n! --force given, launching anyway")
    else:
        sys.exit(message)


def cmd_launch(aws, args):
    STATE.mkdir(exist_ok=True)
    check_instances(args.image, args.force)
    taskdef = register(aws, args.image, args.cpu, args.memory)
    todo = list(pairs(args))
    vcpu = len(todo) * args.cpu / 1024
    print(f"launching {len(todo)} task(s) = {vcpu:g} vCPU, {args.procs} clem process(es) each "
          f"= {len(todo) * args.procs} concurrent Bedrock streams")
    manifest = {"run_id": args.run_id, "image": args.image, "temp": args.temp,
                "taskdef": taskdef, "pairs": {},
                "launched_at": datetime.now(timezone.utc).isoformat()}
    for model, suite in todo:
        arn = run_task(aws, taskdef, args.run_id, model, suite, args.procs, args.temp,
                       args.sync_seconds)
        if arn:
            record_pair(manifest, model, suite, args.procs, args.cpu, args.memory, arn)
            print(f"  {model}/{suite} -> {arn.rsplit('/', 1)[-1]}")
    (STATE / f"{args.run_id}.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest: {STATE / (args.run_id + '.json')}")


def record_pair(manifest, model, suite, procs, cpu, memory, arn):
    """Per-pair, not per-run: `scale` takes --models/--suites selectors, so after
    scaling one suite a single global procs/cpu would describe none of the fleet
    correctly and would then be used as the default for the next rescale."""
    manifest.setdefault("pairs", {})[f"{model}/{suite}"] = {
        "procs": procs, "cpu": cpu, "memory": memory, "arn": arn,
        "since": datetime.now(timezone.utc).isoformat(),
    }


def cmd_status(aws, args):
    STATE.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc)
    tasks = live_tasks(aws, args.run_id)
    counts = episode_counts(aws, args.run_id)

    by_pair = {}
    for task in tasks:
        env = task_env(task)
        by_pair[(env.get("MODEL_KEY"), env.get("SUITE"))] = task

    previous = {}
    path = snapshot_path(args.run_id)
    if path.exists():
        previous = json.loads(path.read_text())
    elapsed_h = 0.0
    if previous:
        elapsed_h = (now - datetime.fromisoformat(previous["at"])).total_seconds() / 3600
    # A rate needs a window comfortably longer than the S3 push interval, or it
    # measures the sync cycle rather than the fleet. Below that: no rates, and
    # leave the baseline alone, so an ad-hoc `status` cannot report "0.0 ep/h,
    # stalled" or destroy the baseline a scheduled check depends on.
    too_soon = bool(previous) and elapsed_h * 3600 < MIN_RATE_WINDOW_S

    print(f"=== {args.run_id} @ {now.strftime('%F %T')}Z ===")
    if too_soon:
        print(f"(only {elapsed_h * 3600:.0f}s since the last check - rates suppressed, "
              f"baseline kept)")
    elif previous:
        print(f"(delta over the last {elapsed_h * 60:.0f} min)")
    print(f"{'pair':<22}{'state':<12}{'done':>10}{'left':>7}{'ep/h':>8}{'eta':>9}  procs")
    total_done = total_expected = 0
    fleet_rate = 0.0
    for model in MODELS:
        for suite, (_, expected) in SUITES.items():
            done = counts.get((model, suite), 0)
            total_done += done
            total_expected += expected
            task = by_pair.get((model, suite))
            state = task["lastStatus"] if task else "-"
            procs = task_env(task).get("PROCS", "-") if task else "-"
            was = (previous.get("counts", {}) or {}).get(f"{model}/{suite}")
            rate = eta = ""
            if was is not None and not too_soon and elapsed_h > 0:
                per_hour = (done - was) / elapsed_h
                fleet_rate += per_hour
                rate = f"{per_hour:.1f}"
                left = expected - done
                # "none" means no episode finished in this window - which for a
                # handful of long S4 episodes is normal. It is NOT run_chain.sh's
                # STALL_LIMIT, which is about passes that make no progress.
                eta = f"{left / per_hour:.1f}h" if per_hour > 0.05 and left else (
                    "-" if not left else "none")
            print(f"{model + '/' + suite:<22}{state:<12}{done:>7}/{expected:<3}"
                  f"{expected - done:>6}{rate:>8}{eta:>9}  {procs}")
    print(f"{'TOTAL':<22}{'':<12}{total_done:>7}/{total_expected:<3}"
          f"{total_expected - total_done:>6}"
          f"{(f'{fleet_rate:.1f}' if fleet_rate else ''):>8}")

    stopped = [t for t in tasks if t["lastStatus"] not in ("RUNNING", "PENDING", "PROVISIONING")]
    for task in stopped:
        env = task_env(task)
        print(f"  ! {env.get('MODEL_KEY')}/{env.get('SUITE')} {task['lastStatus']}: "
              f"{task.get('stoppedReason', '')}")

    print(f"\n--- Bedrock, last {args.window} min ---")
    metrics = bedrock_metrics(aws, args.window)
    for model, row in metrics.items():
        calls = row["Invocations"]
        throttles = row["InvocationThrottles"]
        denom = calls + throttles
        rate = (throttles / denom * 100) if denom else 0.0
        print(f"{model:<12} calls={calls:>7.0f} throttled={throttles:>6.0f} ({rate:4.1f}%) "
              f"errors={row['InvocationServerErrors']:>4.0f} "
              f"latency={row['InvocationLatency'] / 1000:>6.1f}s "
              f"tokens_in={row['InputTokenCount'] / 1e6:>5.2f}M "
              f"out={row['OutputTokenCount'] / 1e6:>5.2f}M")

    print(f"\n--- containers, last {args.window} min ---")
    cpu = container_cpu(aws, args.window)
    reserved = cpu["CpuReserved"][0] or 1
    print(f"CPU max={cpu['CpuUtilized'][0]:.0f} avg={cpu['CpuUtilized'][1]:.0f} "
          f"of {reserved:.0f} reserved ({cpu['CpuUtilized'][0] / reserved * 100:.0f}% peak)   "
          f"MEM max={cpu['MemoryUtilized'][0]:.0f} of {cpu['MemoryReserved'][0]:.0f} MiB")

    streams = sum(int(task_env(t).get("PROCS", 0) or 0) for t in by_pair.values())
    # Counts come from S3, which each task refreshes only every SYNC_SECONDS, so
    # a reading trails reality by up to one push interval. The DELTA between two
    # readings is still unbiased - the same lag sits on both ends - but only if
    # no task restarted in between: a stopping task does a final flush, which
    # dumps a whole interval of backlog into the later reading and inflates it.
    # Comparing a flushed total against a mid-cycle one is what made 64 streams
    # look slower than 32 on 2026-09-04, when it was in fact 12% faster.
    generation = sorted(t["taskArn"] for t in by_pair.values())
    comparable = (bool(previous) and not too_soon
                  and previous.get("generation") == generation)

    print(f"\n--- throughput ---")
    if streams:
        print(f"{streams} streams" + (f", {fleet_rate / streams:.1f} episodes/h per stream"
                                      if fleet_rate and comparable else ""))
    if previous and not comparable and not too_soon:
        print("tasks restarted since the last check - this delta spans a final flush "
              "and is NOT a valid rate. Compare from the next check on.")

    print("\n--- verdict ---")
    for line in advise(metrics, cpu, by_pair, previous, elapsed_h, fleet_rate, streams,
                       comparable, total_expected - total_done):
        print(f"  {line}")

    # Only advance the baseline when this reading can serve as one.
    if too_soon:
        print(f"\n(baseline from {previous['at'][11:19]}Z left in place)")
        return
    path.write_text(json.dumps({
        "at": now.isoformat(),
        "counts": {f"{m}/{s}": c for (m, s), c in counts.items()},
        "streams": streams,
        "fleet_rate": fleet_rate,
        "generation": generation,
        "latency": {m: row["InvocationLatency"] / 1000 for m, row in metrics.items()},
    }, indent=2))


def advise(metrics, cpu, by_pair, previous, elapsed_h, fleet_rate, streams, comparable=True,
           remaining=None):
    """Turn the numbers into the one decision this check exists for: PROCS up,
    down, or leave it. Deliberately blunt - the thresholds are the argument.

    Throttles alone are not enough. Bedrock answers past-capacity load by
    queueing rather than by returning ThrottlingException, so the tell is
    latency climbing while episodes/hour stays flat - which is exactly what
    happened going 32 -> 64 streams on 2026-09-04. Judge on delivered
    throughput first and on the error signals second."""
    out = []
    worst = errors = 0.0
    for row in metrics.values():
        denom = row["Invocations"] + row["InvocationThrottles"]
        if denom:
            worst = max(worst, row["InvocationThrottles"] / denom)
            errors = max(errors, row["InvocationServerErrors"] / denom)
    reserved = cpu["CpuReserved"][0] or 1
    cpu_peak = cpu["CpuUtilized"][0] / reserved

    # Did the last change actually buy anything? Compare episodes/h per stream.
    verdict = None
    was_streams = previous.get("streams") or 0
    was_rate = previous.get("fleet_rate") or 0.0
    if comparable and was_streams and streams and was_rate and fleet_rate and elapsed_h > 0.1:
        before, after = was_rate / was_streams, fleet_rate / streams
        if streams > was_streams and after < before * 0.75:
            verdict = (f"PAST THE KNEE: {was_streams} -> {streams} streams moved throughput "
                       f"{was_rate:.0f} -> {fleet_rate:.0f} ep/h, i.e. {before:.1f} -> {after:.1f} "
                       f"per stream. The extra streams are queueing. Go back to {was_streams}.")
        elif streams < was_streams and after > before * 1.25:
            verdict = (f"good: {was_streams} -> {streams} streams and each stream got "
                       f"{before:.1f} -> {after:.1f} ep/h faster. Hold here.")

    was_latency = previous.get("latency") or {}
    spikes = [f"{m} {was_latency[m]:.1f}s -> {row['InvocationLatency'] / 1000:.1f}s"
              for m, row in metrics.items()
              if was_latency.get(m, 0) > 0.2
              and row["InvocationLatency"] / 1000 > was_latency[m] * 2]
    if spikes and not verdict and comparable:
        verdict = ("latency more than doubled (" + "; ".join(spikes) + ") with no throughput "
                   "gain to show for it - the provider is queueing. Reduce PROCS.")

    # Every PROCS change costs a stop/start plus the episodes in flight. Near the
    # end of a campaign that is simply more than the change can win back.
    minutes_left = (remaining / fleet_rate * 60) if (remaining and fleet_rate) else None
    if minutes_left is not None and minutes_left < 15 and comparable:
        out.append(f"finishing: {remaining} episodes left, ~{minutes_left:.0f} min at the "
                   f"current rate. A rescale costs more than it saves - leave it alone.")
    elif verdict:
        out.append(verdict)
    elif worst > 0.10:
        out.append(f"THROTTLED HARD ({worst * 100:.0f}%) - halve PROCS: the retries are "
                   f"burning wall-clock, not buying throughput.")
    elif worst > 0.03:
        out.append(f"throttling at {worst * 100:.0f}% - at the knee. Hold PROCS here.")
    elif cpu_peak > 0.85:
        out.append(f"CPU peaked at {cpu_peak * 100:.0f}% of reservation - the container, not "
                   f"Bedrock, is the limit. Relaunch with a bigger --cpu before raising PROCS.")
    elif not comparable or not fleet_rate:
        out.append(f"no valid rate this check ({streams} streams) - nothing to judge PROCS on. "
                   f"Wait for the next check.")
    else:
        out.append(f"headroom: throttling {worst * 100:.1f}%, CPU peak {cpu_peak * 100:.0f}%, "
                   f"{fleet_rate / streams if streams else 0:.1f} ep/h per stream - "
                   f"raise PROCS (try double) with `ctl.py scale`.")

    if errors > 0.02:
        out.append(f"server errors at {errors * 100:.0f}% of calls - each one that outlives the "
                   f"backend's 4 retries costs an episode. Another reason not to push harder.")

    running = [p for p, t in by_pair.items() if t["lastStatus"] == "RUNNING"]
    if len(running) < len(by_pair):
        out.append(f"only {len(running)} of {len(by_pair)} tasks are RUNNING - check the "
                   f"stopped ones above before reading the rates.")
    if comparable and elapsed_h < 0.15:
        out.append(f"rates are over only {elapsed_h * 60:.0f} min - treat them as noisy.")
    return out


def cmd_scale(aws, args):
    manifest_path = STATE / f"{args.run_id}.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    targets = set(pairs(args))
    tasks = [t for t in live_tasks(aws, args.run_id)
             if (task_env(t).get("MODEL_KEY"), task_env(t).get("SUITE")) in targets]

    print(f"stopping {len(tasks)} task(s) so they can come back with PROCS={args.procs}")
    for task in tasks:
        env = task_env(task)
        aws.ecs.stop_task(cluster=aws.cfg["CLUSTER"], task=task["taskArn"],
                          reason=f"rescale to procs={args.procs}")
        print(f"  stopping {env.get('MODEL_KEY')}/{env.get('SUITE')}")
    wait_stopped(aws, [t["taskArn"] for t in tasks])

    image = args.image or manifest.get("image")
    if not image:
        sys.exit("no image known for this run - pass --image")
    check_instances(image, args.force)
    # Defaults come from what the TARGETED pairs are running now, not from a
    # run-wide value that some other suite's rescale may have last written.
    known = [manifest.get("pairs", {}).get(f"{m}/{s}", {}) for m, s in targets]
    cpu = args.cpu or max([p.get("cpu", 0) for p in known] + [0]) or 1024
    memory = args.memory or max([p.get("memory", 0) for p in known] + [0]) or 4096
    taskdef = register(aws, image, cpu, memory)
    for model, suite in sorted(targets):
        arn = run_task(aws, taskdef, args.run_id, model, suite, args.procs, args.temp,
                       args.sync_seconds)
        print(f"  {model}/{suite} -> {arn.rsplit('/', 1)[-1] if arn else 'FAILED'}")
        if arn:
            record_pair(manifest, model, suite, args.procs, cpu, memory, arn)
    manifest.update({"image": image, "taskdef": taskdef,
                     "rescaled_at": datetime.now(timezone.utc).isoformat()})
    manifest_path.write_text(json.dumps(manifest, indent=2))
    # From ECS, not the manifest: a suite that finished has already exited, and
    # the manifest has no way to learn that on its own.
    live = live_tasks(aws, args.run_id)
    total = sum(int(task_env(t).get("PROCS", 0) or 0) for t in live)
    print(f"fleet now: {total} streams across {len(live)} live task(s)")


def wait_stopped(aws, arns, timeout=300):
    """Two drivers on one suite would replay each other's episodes, so never
    start the replacements until the old tasks are really gone."""
    if not arns:
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        described = aws.ecs.describe_tasks(cluster=aws.cfg["CLUSTER"], tasks=arns)["tasks"]
        alive = [t for t in described if t["lastStatus"] != "STOPPED"]
        if not alive:
            print("  all stopped")
            return
        print(f"  waiting for {len(alive)} task(s) to stop "
              f"({', '.join(sorted({t['lastStatus'] for t in alive}))})")
        time.sleep(10)
    sys.exit("tasks did not stop within the timeout - not starting replacements")


def cmd_stop(aws, args):
    tasks = [t for t in live_tasks(aws, args.run_id)
             if (task_env(t).get("MODEL_KEY"), task_env(t).get("SUITE")) in set(pairs(args))]
    for task in tasks:
        env = task_env(task)
        aws.ecs.stop_task(cluster=aws.cfg["CLUSTER"], task=task["taskArn"],
                          reason=args.reason)
        print(f"stopping {env.get('MODEL_KEY')}/{env.get('SUITE')}")
    if args.wait:
        wait_stopped(aws, [t["taskArn"] for t in tasks])
    print(f"{len(tasks)} task(s) signalled; finished episodes are already in S3")


def cmd_fetch(aws, args):
    """Pull the S3 mirror into the local results_* trees the analysis expects."""
    sys.path.insert(0, str(HERE))
    import s3sync  # noqa: E402  - lives next to this file, in the image too

    for model, suite in pairs(args):
        dest = REPO / f"results_{model}_{args.label}_n15_{suite.split('_grid')[0]}"
        prefix = f"{args.run_id}/{model}/{suite}"
        moved = s3sync.pull(aws.s3, aws.cfg["BUCKET"], prefix, dest)
        episodes = len(list(dest.rglob("interactions.json")))
        print(f"{model}/{suite}: +{moved} file(s) -> {dest.name} ({episodes} episodes)")
        if args.score:
            game, _ = SUITES[suite]
            print(f"  scoring {game}")
            subprocess.run(["clem", "score", "-g", game, "-r", str(dest)],
                           cwd=REPO, check=False)


def cmd_logs(aws, args):
    streams = aws.logs.describe_log_streams(
        logGroupName=aws.cfg["LOG_GROUP"], orderBy="LastEventTime", descending=True,
        limit=50)["logStreams"]
    tasks = {t["taskArn"].rsplit("/", 1)[-1]: task_env(t) for t in live_tasks(aws, args.run_id)}
    for stream in streams:
        task_id = stream["logStreamName"].rsplit("/", 1)[-1]
        env = tasks.get(task_id)
        if not env:
            continue
        if args.models and env.get("MODEL_KEY") not in args.models.split(","):
            continue
        events = aws.logs.get_log_events(
            logGroupName=aws.cfg["LOG_GROUP"], logStreamName=stream["logStreamName"],
            limit=args.lines, startFromHead=False)["events"]
        print(f"--- {env.get('MODEL_KEY')}/{env.get('SUITE')} ({task_id}) ---")
        for event in events:
            print("   ", event["message"].rstrip())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(parser, selectors=True):
        parser.add_argument("--run-id", required=True)
        if selectors:
            parser.add_argument("--models", help=f"comma list of {', '.join(MODELS)}")
            parser.add_argument("--suites", help=f"comma list of {', '.join(SUITES)}")

    p = sub.add_parser("launch")
    common(p)
    p.add_argument("--image", required=True, help="ECR image URI from build-push.sh")
    p.add_argument("--procs", type=int, default=4, help="clem processes per task")
    p.add_argument("--cpu", type=int, default=1024)
    p.add_argument("--memory", type=int, default=4096)
    p.add_argument("--temp", default="0.6")
    p.add_argument("--force", action="store_true",
                   help="launch even if the image's instances are stale")
    p.add_argument("--sync-seconds", type=int, default=60,
                   help="how often a task mirrors results to S3; also how far the "
                        "progress numbers can trail reality")
    p.set_defaults(fn=cmd_launch)

    p = sub.add_parser("status")
    common(p, selectors=False)
    p.add_argument("--window", type=int, default=15, help="metric window in minutes")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("scale")
    common(p)
    p.add_argument("--procs", type=int, required=True)
    p.add_argument("--image")
    p.add_argument("--cpu", type=int)
    p.add_argument("--memory", type=int)
    p.add_argument("--temp", default="0.6")
    p.add_argument("--force", action="store_true",
                   help="rescale even if the image's instances are stale")
    p.add_argument("--sync-seconds", type=int, default=60,
                   help="how often a task mirrors results to S3; also how far the "
                        "progress numbers can trail reality")
    p.set_defaults(fn=cmd_scale)

    p = sub.add_parser("stop")
    common(p)
    p.add_argument("--reason", default="stopped by ctl.py")
    p.add_argument("--wait", action="store_true")
    p.set_defaults(fn=cmd_stop)

    p = sub.add_parser("fetch")
    common(p)
    p.add_argument("--score", action="store_true", help="run `clem score` on each tree")
    p.add_argument("--label", default="grid",
                   help="middle part of the local tree name results_<model>_<label>_n15_<suite>; "
                        "the default reproduces the grid campaign's names, a compact rerun "
                        "needs its own (e.g. fixthink)")
    p.set_defaults(fn=cmd_fetch)

    p = sub.add_parser("logs")
    common(p, selectors=False)
    p.add_argument("--models")
    p.add_argument("--lines", type=int, default=15)
    p.set_defaults(fn=cmd_logs)

    args = ap.parse_args()
    for name in ("models", "suites"):
        if not hasattr(args, name):
            setattr(args, name, None)
    args.fn(Aws(config()), args)


if __name__ == "__main__":
    main()
