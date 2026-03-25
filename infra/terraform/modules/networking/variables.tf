variable "env" {
  description = "Environment name (dev | staging | prod)"
  type        = string
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
}

variable "domain" {
  description = "FQDN for this environment (e.g. ml-app-dev.diyer.us) — used for ACM cert and Route53 alias"
  type        = string
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}
