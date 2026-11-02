# AWS infrastructure — DESIGNED ONLY, NOT DEPLOYED

> **Status: designed only.** Nothing in this folder has been deployed. `terraform apply` has never
> been run, there is no AWS account behind it, and CI has no apply step. See `docs/decisions.md` D15.

## What it would build

The v2 fraud API (`v2/serving/`) in AWS London (eu-west-2):

| Component | Why |
|---|---|
| **ECR** repository, immutable tags, scan on push, keep the last 10 images | where CI would push the API image |
| **S3** bucket, versioned, encrypted, public access blocked | model artifacts (the `v1_frozen` bundle, future candidates) |
| **ECS Fargate** cluster, task (0.5 vCPU, 1 GB) and service (1 task) | runs the container with no servers to manage |
| **IAM**: execution role, task role (read models only) | least privilege for the running task |
| **GitHub OIDC** provider + deploy role | GitHub Actions gets short-lived credentials. **No AWS keys are stored anywhere.** Only `korede-folarin/fraud-detection-ml` on branch `main` can use the role. |
| VPC, 2 public subnets, internet gateway, security group | network, with inbound port 3000 allowed from one CIDR only |
| CloudWatch log group, 7-day retention | container logs |

It is sized from real local measurements (D13): one worker sustained about 80 requests per second and used about 270 MB.

## What it costs per month (if it were deployed)

These are on-demand London prices from the **AWS Price List API**, fetched by `cost/estimate.py` on 2026-11-02 (USD, before tax and free tier). The full output is in `cost/estimate.json`.

| Component | USD / month | In design? |
|---|---:|---|
| ECS Fargate task, 0.5 vCPU + 1 GB, always on (730 h at $0.04656/vCPU-h, $0.00511/GB-h) | 20.72 | yes |
| Public IPv4 address for the task ($0.005/h) | 3.65 | yes |
| ECR image storage, ~3 GB ($0.10/GB-month) | 0.30 | yes |
| S3 model artifacts, ~0.1 GB ($0.024/GB-month) | 0.00 | yes |
| CloudWatch Logs, ~1 GB ingested ($0.5985/GB) | 0.61 | yes |
| VPC, subnets, IGW, IAM, OIDC provider, ECS cluster | 0.00 | yes (no charge) |
| **Total as designed** | **25.28** | |
| Application Load Balancer, 730 h + 1 LCU ($0.02646/h + $0.0084/LCU-h) | 25.45 | **no**: production add-on |
| NAT gateway, data transfer out, Terraform state bucket | not priced | **no** |

**Design trade-offs, stated plainly:**
- **No NAT gateway, no load balancer.** This keeps the demo at about $25 a month. The load balancer alone would add $25.45 (about $51 in total), and the NAT gateway was not priced, so there is no full production total here. The cost is that the task has a public IP, there is no TLS, and there is no zero-downtime rolling deploy behind a stable address. A production version would add an ALB with HTTPS and private subnets.
- **No remote state backend.** Creating its S3 bucket and lock would be the first real resource.

## How it is checked without deploying

CI (`.github/workflows/ci.yml`, job `terraform`) runs:

```
terraform fmt -check
terraform init -backend=false
terraform validate
terraform plan -refresh=false -var offline_plan=true     # mock credentials, no AWS account
```

`offline_plan=true` turns off every AWS lookup, so the plan proves the configuration and the resource graph (25 resources). It cannot prove that IAM permissions work. That would need a real account, and this project deliberately has none.

**Why no LocalStack or moto (D15):**
- LocalStack's free plan no longer includes ECR or ECS, and it now needs an account token.
- The API container has no AWS dependency at runtime: the model is baked into the image.
- So an AWS emulator would only test the emulator. The checks that matter here are validate and plan.
