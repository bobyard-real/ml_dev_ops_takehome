terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket         = "tf-state-ml-app"   # output StateBucketName from cfn-bootstrap
    key            = "shared/terraform.tfstate"
    region         = "us-west-2"
    dynamodb_table = "tf-state-lock"     # output LockTableName from cfn-bootstrap
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region
}
