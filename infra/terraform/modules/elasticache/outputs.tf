output "redis_primary_endpoint" {
  description = "Redis primary endpoint hostname"
  value       = aws_elasticache_replication_group.this.primary_endpoint_address
}

output "redis_url" {
  description = "Full Redis URL for use in REDIS_URL env var"
  value       = "redis://${aws_elasticache_replication_group.this.primary_endpoint_address}:6379/0"
}
