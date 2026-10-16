# Runner image for the matrixgame grid suites on ECS Fargate.
#
# Built from a staged subset of the working tree (see build-push.sh), not from
# the repo root: the repo carries multi-GB results trees, PDFs and a 50 MB log
# that have no business in a build context.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# procps for the campaign driver's `kill -0` liveness checks on its shards.
RUN apt-get update \
 && apt-get install -y --no-install-recommends procps \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /clembench

COPY requirements-ecs.txt .
RUN pip install --no-cache-dir -r requirements-ecs.txt

COPY . /clembench
RUN mkdir -p /work/results && chmod +x /clembench/cloud/ecs/entrypoint.sh

ENTRYPOINT ["bash", "/clembench/cloud/ecs/entrypoint.sh"]
