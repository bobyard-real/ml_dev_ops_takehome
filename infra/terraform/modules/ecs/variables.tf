variable "env" {
  description = "Environment name (dev | staging | prod)"
  type        = string
}

variable "image_uri" {
  description = "Initial container image URI (CI will update this on each deploy)"
  type        = string
}

variable "desired_count" {
  description = "Number of running ECS task instances"
  type        = number
  default     = 1
}

variable "cpu" {
  description = "Task-level CPU units (256 | 512 | 1024 | 2048 | 4096)"
  type        = number
  default     = 256
}

variable "memory" {
  description = "Task-level memory in MiB"
  type        = number
  default     = 512
}

variable "use_spot" {
  description = "Use FARGATE_SPOT capacity provider (cheaper; suitable for non-prod)"
  type        = bool
  default     = false
}

variable "task_execution_role_arn" {
  description = "ARN of the ECS task execution role"
  type        = string
}

variable "target_group_arn" {
  description = "ARN of the ALB target group"
  type        = string
}

variable "private_subnet_ids" {
  description = "List of private subnet IDs for ECS tasks"
  type        = list(string)
}

variable "ecs_sg_id" {
  description = "Security group ID for ECS tasks"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}
