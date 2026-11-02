output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "model_bucket" {
  value = aws_s3_bucket.models.bucket
}

output "github_deploy_role_arn" {
  description = "Set as the role-to-assume in a deploy workflow (aws-actions/configure-aws-credentials)."
  value       = aws_iam_role.github_deploy.arn
}

output "ecs_service" {
  value = "${aws_ecs_cluster.main.name}/${aws_ecs_service.api.name}"
}

output "status" {
  value = "DESIGNED ONLY - NOT DEPLOYED"
}
