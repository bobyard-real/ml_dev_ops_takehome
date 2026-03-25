variable "env" {
  description = "Environment name (dev | staging | prod)"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "alb_arn_suffix" {
  description = "ALB ARN suffix for CloudWatch metric dimensions"
  type        = string
}

variable "target_group_arn_suffix" {
  description = "Target group ARN suffix for CloudWatch metric dimensions"
  type        = string
}

variable "ecs_cluster_name" {
  description = "ECS cluster name for CloudWatch metric dimensions"
  type        = string
}

variable "ecs_service_name" {
  description = "ECS service name for CloudWatch metric dimensions"
  type        = string
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}
