variable "github_org" {
  description = "GitHub organisation or username that owns the repository"
  type        = string
}

variable "github_repo" {
  description = "GitHub repository name (without org prefix)"
  type        = string
}

variable "ecr_repository_arn" {
  description = "ARN of the ECR repository — granted to the github-actions-ecr role"
  type        = string
}

variable "aws_account_id" {
  description = "AWS account ID — used to scope IAM PassRole conditions"
  type        = string
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}
