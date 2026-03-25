variable "aws_region" {
  type    = string
  default = "us-west-2"
}

variable "github_org" {
  type        = string
  description = "GitHub org or username that owns the repo"
}

variable "github_repo" {
  type        = string
  description = "Repository name (without org prefix)"
}

variable "aws_account_id" {
  type        = string
  description = "AWS account ID"
}

# ── ECR (shared across all environments) ──────────────────────────────────────
module "ecr" {
  source          = "../../modules/ecr"
  repository_name = "ml-app"
  tags            = local.tags
}

# ── IAM: GitHub OIDC + roles + task execution role ───────────────────────────
module "iam" {
  source             = "../../modules/iam"
  github_org         = var.github_org
  github_repo        = var.github_repo
  ecr_repository_arn = module.ecr.repository_arn
  aws_account_id     = var.aws_account_id
  tags               = local.tags
}

locals {
  tags = {
    Project   = "ml-app"
    ManagedBy = "Terraform"
  }
}

# ── Outputs consumed by per-env states via terraform_remote_state ─────────────
output "ecr_repository_url" {
  value = module.ecr.repository_url
}

output "task_execution_role_arn" {
  value = module.iam.task_execution_role_arn
}

output "github_ecr_role_arn" {
  value = module.iam.github_ecr_role_arn
}

output "github_ecs_role_arns" {
  value = module.iam.github_ecs_role_arns
}
