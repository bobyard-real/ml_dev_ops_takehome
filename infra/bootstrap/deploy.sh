#!/usr/bin/env bash
# Deploy (or update) the Terraform bootstrap CloudFormation stack.
# Run this once before running any `terraform init`.
#
# Prerequisites:
#   - AWS CLI configured with credentials that can create S3, DynamoDB, IAM, and Secrets Manager resources
#   - BucketName must be globally unique — override the default if needed
#
# Usage:
#   bash deploy.sh
#   BUCKET_NAME=my-unique-bucket bash deploy.sh

set -euo pipefail

STACK_NAME="${STACK_NAME:-tf-bootstrap}"
REGION="${AWS_REGION:-us-west-2}"
BUCKET_NAME="${BUCKET_NAME:-tf-state-ml-app}"
TABLE_NAME="${TABLE_NAME:-tf-state-lock}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Deploying stack: ${STACK_NAME} in ${REGION}"
echo "  S3 bucket  : ${BUCKET_NAME}"
echo "  DynamoDB   : ${TABLE_NAME}"

aws cloudformation deploy \
  --template-file "${SCRIPT_DIR}/cfn-bootstrap.yml" \
  --stack-name "${STACK_NAME}" \
  --region "${REGION}" \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    BucketName="${BUCKET_NAME}" \
    TableName="${TABLE_NAME}"

echo ""
echo "Stack outputs:"
aws cloudformation describe-stacks \
  --stack-name "${STACK_NAME}" \
  --region "${REGION}" \
  --query "Stacks[0].Outputs" \
  --output table

echo ""
echo "Retrieve Terraform deployer credentials:"
echo "  aws secretsmanager get-secret-value --secret-id terraform-deployer-credentials --region ${REGION} --query SecretString --output text"
