#!/usr/bin/env bash
# Create the ECS/ECR/S3/IAM resources the runner needs. Idempotent.
#
#   aws sso login --profile AWSAdministratorAccess-106197011510
#   cloud/ecs/setup-infra.sh
#
# Ends by writing cloud/ecs/config.env, which every other script sources.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROFILE="${PROFILE:-AWSAdministratorAccess-106197011510}"
REGION="${REGION:-eu-central-1}"     # where both Bedrock models are enabled
CLUSTER="${CLUSTER:-clembench}"
REPO_NAME="${REPO_NAME:-clembench-runner}"
LOG_GROUP="${LOG_GROUP:-/ecs/clembench}"
EXEC_ROLE="${EXEC_ROLE:-clembench-ecs-execution}"
TASK_ROLE="${TASK_ROLE:-clembench-ecs-task}"
SG_NAME="${SG_NAME:-clembench-ecs-tasks}"

aws() { command aws --profile "$PROFILE" --region "$REGION" "$@"; }
step() { echo; echo "==> $*"; }

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="${BUCKET:-clembench-ecs-$ACCOUNT}"
ECR_URI="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME"
echo "account=$ACCOUNT region=$REGION"

step "S3 bucket $BUCKET"
if aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  echo "exists"
else
  aws s3api create-bucket --bucket "$BUCKET" \
    --create-bucket-configuration "LocationConstraint=$REGION" >/dev/null
  aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
  aws s3api put-bucket-encryption --bucket "$BUCKET" --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
  echo "created"
fi

step "ECR repository $REPO_NAME"
aws ecr describe-repositories --repository-names "$REPO_NAME" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$REPO_NAME" \
       --image-scanning-configuration scanOnPush=true >/dev/null
# Keep the last 10 images; a rebuild per launch would otherwise accumulate.
aws ecr put-lifecycle-policy --repository-name "$REPO_NAME" --lifecycle-policy-text '{
  "rules":[{"rulePriority":1,"description":"keep last 10",
            "selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":10},
            "action":{"type":"expire"}}]}' >/dev/null
echo "$ECR_URI"

step "CloudWatch log group $LOG_GROUP"
aws logs create-log-group --log-group-name "$LOG_GROUP" 2>/dev/null || echo "exists"
aws logs put-retention-policy --log-group-name "$LOG_GROUP" --retention-in-days 14

step "IAM execution role $EXEC_ROLE"
TRUST='{"Version":"2012-10-17","Statement":[{"Effect":"Allow",
  "Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam get-role --role-name "$EXEC_ROLE" >/dev/null 2>&1 \
  || aws iam create-role --role-name "$EXEC_ROLE" --assume-role-policy-document "$TRUST" >/dev/null
aws iam attach-role-policy --role-name "$EXEC_ROLE" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

step "IAM task role $TASK_ROLE"
aws iam get-role --role-name "$TASK_ROLE" >/dev/null 2>&1 \
  || aws iam create-role --role-name "$TASK_ROLE" --assume-role-policy-document "$TRUST" >/dev/null
# Scoped to model invocation and this one bucket: the container never needs to
# list models, change quotas, or touch anything else in the account.
aws iam put-role-policy --role-name "$TASK_ROLE" --policy-name clembench-runner \
  --policy-document "{
    \"Version\":\"2012-10-17\",
    \"Statement\":[
      {\"Sid\":\"InvokeModels\",\"Effect\":\"Allow\",
       \"Action\":[\"bedrock:InvokeModel\",\"bedrock:InvokeModelWithResponseStream\",
                   \"bedrock:Converse\",\"bedrock:ConverseStream\"],
       \"Resource\":[\"arn:aws:bedrock:*::foundation-model/*\",
                     \"arn:aws:bedrock:*:$ACCOUNT:inference-profile/*\"]},
      {\"Sid\":\"ResultsBucket\",\"Effect\":\"Allow\",
       \"Action\":[\"s3:ListBucket\"],\"Resource\":\"arn:aws:s3:::$BUCKET\"},
      {\"Sid\":\"ResultsObjects\",\"Effect\":\"Allow\",
       \"Action\":[\"s3:GetObject\",\"s3:PutObject\"],\"Resource\":\"arn:aws:s3:::$BUCKET/*\"}
    ]}"

step "ECS cluster $CLUSTER"
# Container Insights costs cents at this fleet size and is the only way to tell
# "the model got slower" apart from "the container ran out of CPU" when deciding
# whether to raise PROCS.
aws ecs create-cluster --cluster-name "$CLUSTER" \
  --settings name=containerInsights,value=enabled >/dev/null
echo "ready"

step "network (default VPC, public subnets, egress-only security group)"
VPC="$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true \
        --query 'Vpcs[0].VpcId' --output text)"
[[ "$VPC" == "None" || -z "$VPC" ]] && { echo "no default VPC in $REGION" >&2; exit 1; }
SUBNETS="$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC" \
            "Name=map-public-ip-on-launch,Values=true" \
            --query 'Subnets[].SubnetId' --output text | tr '\t' ',')"
[[ -z "$SUBNETS" ]] && { echo "no public subnets in $VPC" >&2; exit 1; }
SG="$(aws ec2 describe-security-groups --filters "Name=vpc-id,Values=$VPC" \
       "Name=group-name,Values=$SG_NAME" --query 'SecurityGroups[0].GroupId' --output text)"
if [[ "$SG" == "None" || -z "$SG" ]]; then
  SG="$(aws ec2 create-security-group --group-name "$SG_NAME" --vpc-id "$VPC" \
         --description "clembench Fargate runners: egress only" --query GroupId --output text)"
  # Default egress-all is what the tasks need (Bedrock, ECR, S3, CloudWatch);
  # no ingress rule is ever added.
fi
echo "vpc=$VPC subnets=$SUBNETS sg=$SG"

step "Fargate vCPU quota (L-3032A538)"
QUOTA="$(aws service-quotas get-service-quota --service-code fargate \
          --quota-code L-3032A538 --query 'Quota.Value' --output text 2>/dev/null || echo unknown)"
echo "Fargate On-Demand vCPU limit in $REGION: $QUOTA"
echo "(8 tasks at 1 vCPU need 8. Raise it with:"
echo "  aws service-quotas request-service-quota-increase --service-code fargate \\"
echo "    --quota-code L-3032A538 --desired-value 64 --profile $PROFILE --region $REGION)"

cat >"$HERE/config.env" <<EOF
# Written by cloud/ecs/setup-infra.sh - re-run it to refresh.
PROFILE=$PROFILE
REGION=$REGION
ACCOUNT=$ACCOUNT
CLUSTER=$CLUSTER
BUCKET=$BUCKET
ECR_URI=$ECR_URI
LOG_GROUP=$LOG_GROUP
EXEC_ROLE_ARN=arn:aws:iam::$ACCOUNT:role/$EXEC_ROLE
TASK_ROLE_ARN=arn:aws:iam::$ACCOUNT:role/$TASK_ROLE
SUBNETS=$SUBNETS
SECURITY_GROUP=$SG
EOF

step "wrote $HERE/config.env"
cat "$HERE/config.env"
