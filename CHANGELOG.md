# Changelog

v1 is frozen at tag `v1.0-evaluated` (commit `a152eb9`). Everything below is v2.
Status labels: **running**, **tested locally**, **designed only** (see `docs/decisions.md`).

## v2 — in progress

### Step 0: freeze v1 (D8–D11)
- Added `v1_frozen/`: the v1 models, scaler, and validate/test features, with `MANIFEST.json` (hashes, v1 MLflow run IDs, `dataset_version`, library versions). *Tested locally.*
- Added the local git tag `v1.0-evaluated`. *Not pushed yet.*
- Paused the weekly Airflow DAG, whose runs were not v1 (D10). *Running (paused).*
- Added `docs/decisions.md`: v1 decisions D1–D7 written down retroactively; v2 starts at D8.
- Added the promotion gate (`v2/promotion/`). First real run: both v1 challengers lost, so XGBoost stays champion. The result is logged to the MLflow experiment `fraud-v2-promotion` and `v2/promotion/records/`. *Tested locally.*

### Step 1: CI (D12)
- `.github/workflows/ci.yml` now has four jobs: lint (new code only), tests with coverage, the v1 regression check, and an API Docker build with smoke test. *Tested locally; first GitHub run is on the next push.*
- New scripts:
  - `v2/regression/check_v1.py`: recomputes all 181 v1 values exactly.
  - `v2/ci/register_v1_model.py`: registers the frozen model for the Docker build.
  - `v2/ci/smoke_test_api.py`: checks a running API against the published demo.
- New tests in `tests/v2/` (gate rules, v1 regression). Ruff and coverage settings are in `pyproject.toml`.
- Coverage: **25%** (see D12 for the breakdown).
- Fixed the first GitHub CI failure: the regression check now hashes text files with normalised line endings, so Windows and Linux agree (D17).

### Step 2: service monitoring (D13)
- `v2/serving/`: a v2 BentoML API with the same model and decisions as v1. It adds Prometheus metrics (decision mix, score distribution, model + SHAP time, rejected inputs, model info) and returns HTTP 400 for bad input. *Tested locally.*
- `v2/monitoring/`: a Docker Compose stack (API + Prometheus 3.4.2 + Grafana 12.0.2) with a provisioned 9-panel "service health" dashboard and the load generator `load.py`. *Tested locally.*
- Real results: at 50 /s, p95 38–47 ms; one worker saturates at about 80 /s; 0 server errors. See D13 and `v2/monitoring/results/`.
- CI's Docker job now builds the v2 API and smoke-tests its metrics and 400 handling.
- Screenshot: `docs/images/grafana-service-health.png`.

### Step 3: streaming (D14)
- `v2/streaming/`:
  - a Kafka + Flink 1.20 + PyFlink stack in Docker Compose;
  - `prepare_replay.py`, which verifies the dataset version and builds the v1 batch reference;
  - `producer.py`, which replays the test set with duplicates and late events injected;
  - `job/fraud_stream_job.py`, which orders, de-duplicates, computes features, scores in batches and checkpoints;
  - `parity.py`, which measures latency and throughput and checks parity against v1;
  - `run_stream.sh` and `log_runs.py`.

  *Tested locally.*
- Real results:
  - 992 events/s at a 1,000/s target, with p95 829 ms.
  - About 2,500 events/s sustained at a 3× overload.
  - Exact parity with v1 batch (scores within 9.3e-10, zero decision flips) in every run, including the fault-injection run and recovery after a TaskManager restart.
  - Results are in `v2/streaming/results/` and the MLflow experiment `fraud-v2-streaming`.
- Three failures were found and fixed (watermark run-ahead, slow state layout, direct-memory exhaustion under overload); see D14.
- New unit tests: `tests/v2/test_stream_features.py`.

### Step 4: infrastructure as code (D15)
- `v2/infra/terraform/`: ECR, S3, ECS Fargate, IAM, and a GitHub OIDC deploy role. *Designed only, not deployed.*
- CI job `terraform`: fmt, validate, and offline plan (25 resources) with mock credentials and no apply.
- `v2/infra/cost/estimate.py` → `estimate.json`: about $25.28 a month as designed, from the AWS Price List API (eu-west-2).
- `v2/infra/README.md`, with the cost table and a "not deployed" label.

### Step 5: Helm on kind, not done (D16)
- Skipped. It doesn't fit alongside the rest in this machine's Docker memory, and it would add no new evidence. See D16.
