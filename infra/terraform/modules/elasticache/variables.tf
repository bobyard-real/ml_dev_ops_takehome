variable "env" {
  type        = string
  description = "Environment name (dev, staging, prod)"
}

variable "vpc_id" {
  type        = string
  description = "VPC ID to place the Redis cluster in"
}

variable "private_subnet_ids" {
  type        = list(string)
  description = "Private subnet IDs for the ElastiCache subnet group"
}

variable "ecs_sg_id" {
  type        = string
  description = "ECS tasks security group ID — allowed to reach Redis on port 6379"
}

variable "node_type" {
  type        = string
  description = "ElastiCache node type (e.g. cache.t4g.micro, cache.t4g.small)"
  default     = "cache.t4g.micro"
}

variable "num_cache_clusters" {
  type        = number
  description = "Number of cache clusters (1 = single node, 2 = primary + replica)"
  default     = 1
}

variable "tags" {
  type        = map(string)
  description = "Tags to apply to all resources"
  default     = {}
}
