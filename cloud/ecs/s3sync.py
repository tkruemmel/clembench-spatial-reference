#!/usr/bin/env python3
"""Mirror a results tree to/from S3 with boto3 only.

Used inside the runner container instead of the AWS CLI: awscli v1 pins its own
botocore and would fight the boto3 the Bedrock backend needs, and awscli v2 is
another 100 MB in the image for one `s3 sync`.

Episode files are write-once (clemcore writes interactions.json / scores.json
from on_game_end and never rewrites them), so comparing size is enough to decide
whether a file already made it across - no checksums, no timestamps.
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from botocore.config import Config

# The lock is a local coordination artefact of one container; mirroring it would
# hand the next task a lock file it then has to reason about.
EXCLUDED = {"driver.lock", "pid"}


def _client():
    return boto3.client("s3", config=Config(retries={"max_attempts": 8, "mode": "adaptive"},
                                            max_pool_connections=32))


def _remote(s3, bucket, prefix):
    """key -> size for everything under the prefix."""
    found = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
        for obj in page.get("Contents", []):
            found[obj["Key"]] = obj["Size"]
    return found


def _local(root: Path):
    """relative posix path -> size for every file under root."""
    found = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.name in EXCLUDED:
            continue
        if any(part in EXCLUDED for part in path.parts):
            continue
        found[path.relative_to(root).as_posix()] = path.stat().st_size
    return found


def _run(jobs, workers=16):
    if not jobs:
        return 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda job: job(), jobs))
    return len(jobs)


def push(s3, bucket, prefix, root: Path):
    remote = _remote(s3, bucket, prefix)
    jobs = []
    for rel, size in _local(root).items():
        key = f"{prefix}/{rel}"
        if remote.get(key) == size:
            continue
        jobs.append(lambda k=key, r=rel: s3.upload_file(str(root / r), bucket, k))
    return _run(jobs)


def pull(s3, bucket, prefix, root: Path):
    local = _local(root)
    jobs = []
    for key, size in _remote(s3, bucket, prefix).items():
        rel = key[len(prefix) + 1:]
        if not rel or local.get(rel) == size:
            continue
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        jobs.append(lambda k=key, d=dest: s3.download_file(bucket, k, str(d)))
    return _run(jobs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("direction", choices=["push", "pull"])
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", required=True, help="key prefix, no leading or trailing slash")
    ap.add_argument("--root", required=True, help="local directory to mirror")
    args = ap.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix.strip("/")
    action = push if args.direction == "push" else pull
    moved = action(_client(), args.bucket, prefix, root)
    print(f"{args.direction}: {moved} file(s) s3://{args.bucket}/{prefix} <-> {root}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
