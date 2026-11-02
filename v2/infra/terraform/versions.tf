# DESIGNED ONLY — NOT DEPLOYED. `terraform apply` has never been run (D15).
terraform {
  required_version = ">= 1.9.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.80"
    }
  }

  # A real deployment would keep state in S3 with locking. Left out on purpose:
  # creating the state bucket would be the first paid resource.
  # backend "s3" {}
}

provider "aws" {
  region = var.region

  # offline_plan = true lets CI run `terraform plan` with no AWS account and no
  # credentials (mock keys in the environment). Nothing is looked up or created;
  # the plan proves the configuration and the resource graph, not permissions.
  skip_credentials_validation = var.offline_plan
  skip_requesting_account_id  = var.offline_plan
  skip_metadata_api_check     = var.offline_plan
  skip_region_validation      = var.offline_plan

  default_tags {
    tags = {
      project    = "fraud-detection-ml"
      version    = "v2"
      managed_by = "terraform"
      status     = "designed-only"
    }
  }
}
