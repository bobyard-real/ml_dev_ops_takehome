variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "app_name" {
  description = "Application name used for resource naming"
  type        = string
  default     = "ml-inference"
}

variable "container_port" {
  description = "Port the container listens on"
  type        = number
  default     = 8000
}

variable "cpu" {
  description = "Fargate task CPU units (1024 = 1 vCPU)"
  type        = number
  default     = 512
}

variable "memory" {
  description = "Fargate task memory in MiB"
  type        = number
  default     = 1024
}

variable "desired_count" {
  description = "Number of ECS tasks to run"
  type        = number
  default     = 1
}

variable "environment" {
  description = "Deployment environment (e.g. staging, prod)"
  type        = string
  default     = "prod"
}

variable "image_tag" {
  description = "Docker image tag to deploy (git SHA or 'latest')"
  type        = string
  default     = "latest"
}

variable "autoscaling_min" {
  description = "Minimum number of ECS tasks"
  type        = number
  default     = 1
}

variable "autoscaling_max" {
  description = "Maximum number of ECS tasks"
  type        = number
  default     = 4
}
