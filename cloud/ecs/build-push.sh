#!/usr/bin/env bash
# Stage the files the runner actually needs, build the ARM64 image, push to ECR.
#
# Staging rather than a .dockerignore because the repo root holds ~10 GB of
# results trees, PDFs and clembench.log; the runner needs ~15 MB of it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

# --local builds into the local daemon and skips ECR entirely, so the image can
# be smoke-tested before any AWS resource exists.
LOCAL=0
[[ "${1:-}" == "--local" ]] && { LOCAL=1; shift; }

TAG="${1:-$(date -u +%Y%m%d-%H%M%S)}"
if (( LOCAL )); then
  IMAGE="clembench-runner:$TAG"
else
  source "$HERE/config.env"   # written by setup-infra.sh
  IMAGE="$ECR_URI:$TAG"
fi
STAGE="${TMPDIR:-/tmp}/clembench-ecs-build"

rm -rf "$STAGE"
mkdir -p "$STAGE"

# The three grid packages import their compact twins' master by path, so the
# compact packages and the shared board generator ship too.
PACKAGES=(
  matrixgame_covered_shared
  matrixgame_covered_solo        matrixgame_covered_solo_grid
  matrixgame_covered_comm        matrixgame_covered_comm_grid
  matrixgame_covered_comm_nplayer matrixgame_covered_comm_nplayer_grid
  matrixgame_covered_comm_grid_updated_prompt matrixgame_covered_comm_nplayer_grid_updated_prompt
)
for pkg in "${PACKAGES[@]}"; do
  rsync -a --exclude='__pycache__' --exclude='tests' --exclude='instances_resume_*' \
        "$REPO/$pkg" "$STAGE/"
done

mkdir -p "$STAGE/campaign" "$STAGE/cloud/ecs" "$STAGE/.certs"
cp "$REPO/campaign/run_chain.sh" "$REPO/campaign/remaining.py" \
   "$REPO/campaign/summarize.py" "$STAGE/campaign/"
cp "$HERE/entrypoint.sh" "$HERE/s3sync.py" "$STAGE/cloud/ecs/"
cp "$HERE/requirements-ecs.txt" "$HERE/Dockerfile" "$STAGE/"
cp "$HERE/logging.yaml" "$STAGE/logging.yaml"
# certifi plus the Potsdam CA. run_chain.sh points SSL_CERT_FILE at this path
# unconditionally, and botocore refuses to start if the file is missing.
cp "$REPO/.certs/clem-ca-bundle.pem" "$STAGE/.certs/"
cp "$REPO/amazon_bedrock_api.py" "$REPO/model_registry.json" "$STAGE/"
# Bedrock authenticates through the task role; nothing in the image needs a key.
# clemcore still expects the file to exist, so ship an empty one rather than the
# real key.json, which carries live third-party keys.
echo '{}' >"$STAGE/key.json"

echo "staged $(du -sh "$STAGE" | cut -f1) in $STAGE"

# Fingerprint the instances that went in. Regenerating them after a build and
# then launching the old image silently plays the wrong boards - it happened on
# 2026-09-04, 37 seconds after a build - so ctl.py refuses to launch an image
# whose fingerprint no longer matches the working tree.
mkdir -p "$HERE/.state"
python3 - "$STAGE" "$HERE/.state/image-$TAG.json" "$IMAGE" <<'PY'
import hashlib, json, sys
stage, out, image = sys.argv[1:4]
files = ["matrixgame_covered_solo_grid/in/instances.json",
         "matrixgame_covered_comm_grid/in/instances.json",
         "matrixgame_covered_comm_nplayer_grid/in/instances_3p.json",
         "matrixgame_covered_comm_nplayer_grid/in/instances_6p.json",
         "matrixgame_covered_solo/in/instances.json",
         "matrixgame_covered_comm/in/instances.json",
         "matrixgame_covered_comm_nplayer/in/instances_3p.json",
         "matrixgame_covered_comm_nplayer/in/instances_6p.json",
         "matrixgame_covered_comm_grid_updated_prompt/in/instances.json",
         "matrixgame_covered_comm_nplayer_grid_updated_prompt/in/instances_3p.json",
         "matrixgame_covered_comm_nplayer_grid_updated_prompt/in/instances_6p.json"]
digests = {f: hashlib.sha256(open(f"{stage}/{f}", "rb").read()).hexdigest() for f in files}
json.dump({"image": image, "instances": digests}, open(out, "w"), indent=2)
for f, d in digests.items():
    print(f"  {d[:16]}  {f}")
PY

# Fargate ARM64 (cheaper, and native on an Apple-silicon builder - no emulation).
if (( LOCAL )); then
  docker buildx build --platform linux/arm64 --provenance=false \
    -t "$IMAGE" --load "$STAGE"
else
  aws ecr get-login-password --region "$REGION" --profile "$PROFILE" \
    | docker login --username AWS --password-stdin "${ECR_URI%%/*}"
  docker buildx build --platform linux/arm64 --provenance=false \
    -t "$IMAGE" -t "$ECR_URI:latest" --push "$STAGE"
fi

echo "$IMAGE"
