variable "region" {
  description = "AWS region. London, matching the UK framing of the project."
  type        = string
  default     = "eu-west-2"
}

variable "name" {
  description = "Prefix for every resource name."
  type        = string
  default     = "fraud-api"
}

variable "github_repository" {
  description = "owner/name of the only repository allowed to assume the deploy role."
  type        = string
  default     = "korede-folarin/fraud-detection-ml"
}

variable "github_branch" {
  description = "Only workflows on this branch can deploy."
  type        = string
  default     = "main"
}

variable "image_tag" {
  description = "Immutable image tag to run (the git SHA in a real pipeline)."
  type        = string
  default     = "v2-designed-only"
}

variable "task_cpu" {
  description = "Fargate vCPU units. 512 = 0.5 vCPU; v2 measured ~80 req/s per worker locally (D13)."
  type        = number
  default     = 512
}

variable "task_memory" {
  description = "Fargate memory in MiB. The API used ~270 MB under load locally (D13)."
  type        = number
  default     = 1024
}

variable "desired_count" {
  description = "Running tasks. 1 for a demo; production would run 2+ across AZs behind a load balancer."
  type        = number
  default     = 1
}

variable "allowed_cidr" {
  description = "Who may call the API on port 3000. Placeholder documentation range; set to a real range before any deployment."
  type        = string
  default     = "203.0.113.0/24"
}

variable "log_retention_days" {
  description = "CloudWatch log retention."
  type        = number
  default     = 7
}

variable "offline_plan" {
  description = "Skip AWS account/credential lookups so `terraform plan` runs in CI with no account (D15)."
  type        = bool
  default     = false
}
