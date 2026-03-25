locals {
  name = "ml-app-redis-${var.env}"
}

# ── Security Group ─────────────────────────────────────────────────────────────
resource "aws_security_group" "redis" {
  name        = "${local.name}-sg"
  description = "Allow Redis access from ECS tasks"
  vpc_id      = var.vpc_id
  tags        = var.tags
}

resource "aws_security_group_rule" "redis_ingress_from_ecs" {
  type                     = "ingress"
  from_port                = 6379
  to_port                  = 6379
  protocol                 = "tcp"
  security_group_id        = aws_security_group.redis.id
  source_security_group_id = var.ecs_sg_id
  description              = "Redis from ECS tasks"
}

resource "aws_security_group_rule" "redis_egress" {
  type              = "egress"
  from_port         = 0
  to_port           = 0
  protocol          = "-1"
  cidr_blocks       = ["0.0.0.0/0"]
  security_group_id = aws_security_group.redis.id
}

# ── Subnet Group ───────────────────────────────────────────────────────────────
resource "aws_elasticache_subnet_group" "this" {
  name       = "${local.name}-subnet-group"
  subnet_ids = var.private_subnet_ids
  tags       = var.tags
}

# ── Redis Replication Group ────────────────────────────────────────────────────
resource "aws_elasticache_replication_group" "this" {
  replication_group_id = local.name
  description          = "Redis cache for ML inference results (${var.env})"

  engine               = "redis"
  engine_version       = "7.1"
  node_type            = var.node_type
  num_cache_clusters   = var.num_cache_clusters
  parameter_group_name = "default.redis7"
  port                 = 6379

  subnet_group_name  = aws_elasticache_subnet_group.this.name
  security_group_ids = [aws_security_group.redis.id]

  # Failover and multi-AZ only make sense with a replica
  automatic_failover_enabled = var.num_cache_clusters >= 2
  multi_az_enabled           = var.num_cache_clusters >= 2

  at_rest_encryption_enabled = true

  tags = var.tags
}
