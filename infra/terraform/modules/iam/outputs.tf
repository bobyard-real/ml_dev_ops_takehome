output "github_ecr_role_arn" {
  description = "ARN of the github-actions-ecr role"
  value       = aws_iam_role.github_ecr.arn
}

output "github_ecs_role_arns" {
  description = "Map of env → github-actions-ecs-{env} role ARN"
  value       = { for k, r in aws_iam_role.github_ecs : k => r.arn }
}

output "task_execution_role_arn" {
  description = "ARN of the ECS task execution role (ml-app-task-execution)"
  value       = aws_iam_role.task_execution.arn
}

output "oidc_provider_arn" {
  description = "ARN of the GitHub OIDC provider"
  value       = aws_iam_openid_connect_provider.github.arn
}
