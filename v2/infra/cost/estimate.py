"""
Monthly cost estimate for the designed-only AWS stack (D15)
Run from the repository root: python v2/infra/cost/estimate.py

Plain English: looks up today's London prices from AWS's own public price
list and works out what the Terraform design would cost per month if it were
ever deployed — per component, with the usage assumptions written down.
Nothing is deployed; nothing here talks to an AWS account.

Technical: downloads the AWS Price List API offer files for eu-west-2
(public, no key) and picks the on-demand unit price by usage type. Usage
assumptions come from the v2 measurements (D13) and the Terraform variables.

What the code does: writes v2/infra/cost/estimate.json and prints a table.
"""
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

OUT = Path(__file__).resolve().parent
URL = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/{svc}/current/eu-west-2/index.json"
HOURS = 730  # AWS's standard month

# usage assumptions (one demo task, always on)
TASK_VCPU, TASK_GB = 0.5, 1.0          # terraform task_cpu=512, task_memory=1024
ECR_GB = 3.0                           # ~1 GB compressed per image, layers shared across up to 10 tags
S3_GB = 0.1                            # v1_frozen bundle (~26 MB) plus a few versions
LOG_GB = 1.0                           # API logs per month at demo traffic


def offer(svc):
    with urllib.request.urlopen(URL.format(svc=svc), timeout=60) as r:
        return json.load(r)


def unit_price(data, usage_type, *, family=None, first_tier=True):
    """On-demand USD price for a usage type (lowest tier when tiered); family narrows shared usage types"""
    skus = {sku for sku, p in data["products"].items()
            if p["attributes"].get("usagetype") == usage_type and (family is None or p.get("productFamily") == family)}
    if not skus:
        raise KeyError(f"usage type {usage_type} not found")
    prices = []
    for sku in skus:
        for term in data["terms"]["OnDemand"].get(sku, {}).values():
            for dim in term["priceDimensions"].values():
                begin = float(dim.get("beginRange", 0))
                prices.append((begin, float(dim["pricePerUnit"]["USD"]), dim["unit"], dim["description"]))
    prices.sort()
    return prices[0] if first_tier else prices


def main():
    ecs, ecr, s3, cw, elb, vpc = (offer(s) for s in
                                 ("AmazonECS", "AmazonECR", "AmazonS3", "AmazonCloudWatch", "AWSELB", "AmazonVPC"))
    p = {
        "fargate_vcpu_hour": unit_price(ecs, "EUW2-Fargate-vCPU-Hours:perCPU"),
        "fargate_gb_hour": unit_price(ecs, "EUW2-Fargate-GB-Hours"),
        "public_ipv4_hour": unit_price(vpc, "EUW2-PublicIPv4:InUseAddress"),
        "ecr_gb_month": unit_price(ecr, "EUW2-TimedStorage-ByteHrs"),
        "s3_gb_month": unit_price(s3, "EUW2-TimedStorage-ByteHrs"),
        "logs_ingest_gb": unit_price(cw, "EUW2-DataProcessing-Bytes"),
        "logs_store_gb_month": unit_price(cw, "EUW2-TimedStorage-ByteHrs"),
        # usage types are shared by Application, Network and Gateway load balancers
        "alb_hour": unit_price(elb, "EUW2-LoadBalancerUsage", family="Load Balancer-Application"),
        "alb_lcu_hour": unit_price(elb, "EUW2-LCUUsage", family="Load Balancer-Application"),
    }
    price = {k: v[1] for k, v in p.items()}

    rows = [
        ("ECS Fargate task (0.5 vCPU, 1 GB, 730 h)",
         HOURS * (TASK_VCPU * price["fargate_vcpu_hour"] + TASK_GB * price["fargate_gb_hour"]), "in design"),
        ("Public IPv4 address for the task (730 h)", HOURS * price["public_ipv4_hour"], "in design"),
        (f"ECR image storage ({ECR_GB:g} GB)", ECR_GB * price["ecr_gb_month"], "in design"),
        (f"S3 model artifacts ({S3_GB:g} GB)", S3_GB * price["s3_gb_month"], "in design"),
        (f"CloudWatch Logs ({LOG_GB:g} GB ingested, 7-day retention)",
         LOG_GB * price["logs_ingest_gb"] + LOG_GB * 0.25 * price["logs_store_gb_month"], "in design"),
        ("VPC, subnets, IGW, IAM, OIDC provider, ECS cluster", 0.0, "in design (no charge)"),
        ("Application Load Balancer (730 h + 1 LCU)",
         HOURS * (price["alb_hour"] + price["alb_lcu_hour"]), "NOT in design (production add-on)"),
    ]
    in_design = sum(c for _, c, s in rows if s.startswith("in design"))
    result = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "region": "eu-west-2",
        "source": "AWS Price List API (public offer files), on-demand USD, before tax and free tier",
        "unit_prices": {k: {"usd": v[1], "unit": v[2], "aws_description": v[3]} for k, v in p.items()},
        "monthly_usd": [{"component": n, "usd": round(c, 2), "status": s} for n, c, s in rows],
        "total_in_design_usd_per_month": round(in_design, 2),
        "not_priced": "NAT gateway (deliberately avoided; would be the largest add-on), data transfer out, "
                      "Terraform state bucket",
        "deployed": False,
    }
    (OUT / "estimate.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for r in result["monthly_usd"]:
        print(f"  ${r['usd']:>7.2f}  {r['component']}  [{r['status']}]")
    print(f"  ${in_design:>7.2f}  TOTAL as designed (not deployed)")


if __name__ == "__main__":
    main()
