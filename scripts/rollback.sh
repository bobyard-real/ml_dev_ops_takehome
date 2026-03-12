#!/usr/bin/env bash
# Roll back the ECS service to a previous task definition revision.
#
# Usage:
#   ./scripts/rollback.sh                    # roll back to previous revision
#   ./scripts/rollback.sh 5                  # roll back to specific revision number
#   ./scripts/rollback.sh list               # list recent revisions
#
# Alternatively, deploy a specific image tag via Terraform:
#   terraform apply -var='image_tag=<git-sha>'

set -euo pipefail

CLUSTER="ml-inference-prod"
SERVICE="ml-inference-prod"
FAMILY="ml-inference-prod"
REGION="${AWS_REGION:-us-east-1}"

list_revisions() {
  echo "=== Recent task definition revisions ==="
  aws ecs list-task-definitions \
    --family-prefix "$FAMILY" \
    --sort DESC \
    --max-items 10 \
    --region "$REGION" \
    --query 'taskDefinitionArns[]' \
    --output table

  echo ""
  echo "Current deployment:"
  aws ecs describe-services \
    --cluster "$CLUSTER" \
    --services "$SERVICE" \
    --region "$REGION" \
    --query 'services[0].{taskDefinition:taskDefinition,running:runningCount,desired:desiredCount,status:status}' \
    --output table
}

rollback_to_revision() {
  local target_arn="$1"

  echo "Current deployment:"
  aws ecs describe-services \
    --cluster "$CLUSTER" \
    --services "$SERVICE" \
    --region "$REGION" \
    --query 'services[0].taskDefinition' \
    --output text

  echo ""
  echo "Rolling back to: $target_arn"
  echo ""

  aws ecs update-service \
    --cluster "$CLUSTER" \
    --service "$SERVICE" \
    --task-definition "$target_arn" \
    --region "$REGION" \
    --query 'service.{taskDefinition:taskDefinition,status:status}' \
    --output table

  echo ""
  echo "Rollback initiated. Waiting for deployment to stabilize..."
  aws ecs wait services-stable \
    --cluster "$CLUSTER" \
    --services "$SERVICE" \
    --region "$REGION" 2>/dev/null && echo "Deployment stable." || echo "Warning: timed out waiting for stability. Check the AWS console."
}

case "${1:-}" in
  list)
    list_revisions
    ;;
  "")
    # Roll back to previous revision
    CURRENT=$(aws ecs describe-services \
      --cluster "$CLUSTER" \
      --services "$SERVICE" \
      --region "$REGION" \
      --query 'services[0].taskDefinition' \
      --output text)

    CURRENT_REV=$(echo "$CURRENT" | grep -oE '[0-9]+$')
    PREV_REV=$((CURRENT_REV - 1))

    if [ "$PREV_REV" -lt 1 ]; then
      echo "Error: no previous revision to roll back to." >&2
      exit 1
    fi

    TARGET="${FAMILY}:${PREV_REV}"
    rollback_to_revision "$TARGET"
    ;;
  *)
    # Roll back to specific revision number
    TARGET="${FAMILY}:${1}"
    rollback_to_revision "$TARGET"
    ;;
esac
