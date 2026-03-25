locals {
  github_subject_prefix = "repo:${var.github_org}/${var.github_repo}"
}

# ── GitHub OIDC provider ──────────────────────────────────────────────────────
# One provider per account; use `data` if it already exists.
resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  # Thumbprints are rotated by GitHub; these are current as of 2024.
  thumbprint_list = [
    "6938fd4d98bab03faadb97b34396831e3780aea1",
    "1c58a3a8518e8759bf075b76b750d4f2df264fcd",
  ]

  tags = var.tags
}

# ── Shared trust policy helpers ───────────────────────────────────────────────
data "aws_iam_policy_document" "oidc_trust" {
  for_each = {
    # Build job has no environment — sub contains the ref
    ecr     = "${local.github_subject_prefix}:*"
    # Deploy jobs have environment: <name> set — sub contains the environment name, not the ref
    dev     = "${local.github_subject_prefix}:environment:dev"
    staging = "${local.github_subject_prefix}:environment:staging"
    prod    = "${local.github_subject_prefix}:environment:production"
  }

  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = each.key == "ecr" ? "StringLike" : "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = [each.value]
    }
  }
}

# ── github-actions-ecr role (build job — any branch/tag) ─────────────────────
resource "aws_iam_role" "github_ecr" {
  name               = "github-actions-ecr"
  assume_role_policy = data.aws_iam_policy_document.oidc_trust["ecr"].json
  tags               = var.tags
}

resource "aws_iam_role_policy" "github_ecr" {
  name = "ecr-push"
  role = aws_iam_role.github_ecr.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "GetAuthToken"
        Effect   = "Allow"
        Action   = "ecr:GetAuthorizationToken"
        Resource = "*"
      },
      {
        Sid    = "PushPull"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
          "ecr:PutImage",
        ]
        Resource = var.ecr_repository_arn
      },
    ]
  })
}

# ── ECS deploy roles (one per environment) ────────────────────────────────────
locals {
  ecs_envs = {
    dev     = "github-actions-ecs-dev"
    staging = "github-actions-ecs-staging"
    prod    = "github-actions-ecs-prod"
  }
}

resource "aws_iam_role" "github_ecs" {
  for_each           = local.ecs_envs
  name               = each.value
  assume_role_policy = data.aws_iam_policy_document.oidc_trust[each.key].json
  tags               = var.tags
}

resource "aws_iam_role_policy" "github_ecs" {
  for_each = local.ecs_envs
  name     = "ecs-deploy"
  role     = aws_iam_role.github_ecs[each.key].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ECSDeployActions"
        Effect = "Allow"
        Action = [
          "ecs:DescribeTaskDefinition",
          "ecs:RegisterTaskDefinition",
          "ecs:UpdateService",
          "ecs:DescribeServices",
        ]
        Resource = "*"
      },
      {
        Sid      = "PassTaskExecutionRole"
        Effect   = "Allow"
        Action   = "iam:PassRole"
        Resource = "arn:aws:iam::${var.aws_account_id}:role/ml-app-task-execution"
      },
    ]
  })
}

# ── ECS task execution role ───────────────────────────────────────────────────
resource "aws_iam_role" "task_execution" {
  name = "ml-app-task-execution"
  tags = var.tags

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "task_execution_managed" {
  role       = aws_iam_role.task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "task_execution_logs" {
  name = "cloudwatch-logs"
  role = aws_iam_role.task_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "logs:CreateLogStream",
        "logs:PutLogEvents",
      ]
      Resource = "arn:aws:logs:*:${var.aws_account_id}:log-group:/ecs/ml-app-*:*"
    }]
  })
}
