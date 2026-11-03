# Decision log

Each entry records what was decided, why, the evidence, and what it changes.
Entries are append-only: a later decision may supersede an earlier one, but
earlier entries are never edited.

- **D1–D7** are v1 decisions, written down retroactively on 2026-11-02 from the
  README at tag `v1.0-evaluated`. They record what v1 did. They do not re-decide it.
- **D8 onwards** are v2 decisions, recorded as they are made.

Status labels used in v2: **running** (deployed and operating), **tested locally**
(built and exercised on this machine or in CI), **designed only** (written, not
deployed).

---

## v1 (retroactive)

### D1. Out-of-time validation, 70/15/15 by time
- **Decision:** split by transaction time: train on the earliest 70%, choose thresholds on the next 15%, look at the final 15% once.
- **Why:** a random split lets the model learn from the future and inflates every metric.
- **Evidence:** the fraud rate fell from 0.193% (train) to 0.122% (test), real drift that a random split would hide.

### D2. Information Value over correlation for feature selection
- **Decision:** select features by Information Value, with optimal binning fitted on the training set only.
- **Why:** correlation misses non-linear patterns.
- **Evidence:** the U-shaped fraud rate by amount: 0.67% below €1.19, 0.11% in the middle, 0.35% above €237. 34 candidate features were cut to 30.

### D3. Class weighting over SMOTE
- **Decision:** handle imbalance inside each model (`class_weight='balanced'`, XGBoost `scale_pos_weight=518`), not with synthetic fraud.
- **Why:** interpolated frauds do not exist in production.
- **Evidence:** XGBoost's mean validate score (0.16%) sits close to the true fraud rate (0.13%).

### D4. Champion: XGBoost, despite losing the test ranking metrics
- **Decision:** choose XGBoost as champion and keep Random Forest as challenger.
- **Why:** at the shared deployment threshold, XGBoost's alerts are 48.1% fraud. Random Forest's are 8.1%, and Logistic Regression flags 31.5% of all payments.
- **Evidence:** `evaluation_results.json` → `at_business_threshold.test`.

### D5. Threshold by business cost, chosen on validate and frozen
- **Decision:** minimise `missed fraud × £150 + false alarms × £10` on validate, then freeze the threshold at 0.0716.
- **Evidence:** on validate it beat the F1 threshold (£1,210 vs £1,810). On test it did not (£2,370 vs £2,150), and this was reported as found.

### D6. Velocity features: measure the limitation, don't hide it
- **Decision:** keep only rolling volatility. The other rolling features had near-zero IV because the data has no customer IDs.

### D7. Explanations: per-decision SHAP, positive reasons only, honest names
- **Decision:** every decline cites up to three features that pushed *towards* fraud. V1–V28 are named "behavioural risk indicators".
- **Evidence:** enforced by `tests/test_inference.py`.

---

## v2

### D8. Freeze v1 before any v2 work · 2026-11-02 · tested locally
- **Decision:**
  - Tag `v1.0-evaluated` on commit `a152eb9` (local tag, not yet pushed).
  - Identify v1 by a manifest (`v1_frozen/MANIFEST.json`), not by editing anything that exists.
- **Why:** v2 must not be able to change v1 by accident, and "v1" had no precise definition.
- **What the manifest records:**
  - **MLflow:** the 6 DagsHub runs from 2026-05-30 whose metrics match `evaluation_results.json`;
  - **Dataset:** `ulb-creditcard@md5:e90efcb83d69faf99fcab8b0255024de`;
  - **Environment:** Python 3.14.3, numpy 2.4.4, pandas 2.3.3, scikit-learn 1.8.0, xgboost 3.2.0;
  - **Files:** a SHA-256 hash of every frozen file.
- **Not done, on purpose:** the MLflow runs were not tagged, because adding a tag would modify them.

### D9. Commit a frozen v1 bundle so CI can check v1 with no secrets · 2026-11-02 · tested locally
- **Context:** the raw data sits behind DagsHub DVC (anonymous access returns HTTP 401), and the v1 model pickles are gitignored. CI therefore had no way to recompute v1.
- **Decision:** commit `v1_frozen/`, about 26 MB:
  - the three v1 models and the scaler;
  - the validate and test features with labels, as parquet, keeping the raw row number.
- **Evidence:** `v2/regression/check_v1.py` recomputes all 181 values in `evaluation_results.json` exactly, on Windows with Python 3.14 and on Linux with Python 3.11 (pinned libraries). Changing one value makes it fail and name the field.
- **Consequence:** this redistributes features derived from the Kaggle ULB dataset; the owner approved that. Scaled features are not stored; `scaler.transform` reproduces v1's scaled arrays exactly, and this is checked at freeze time.

### D10. Pause the weekly Airflow DAG; its runs are not v1 · 2026-11-02 · running (paused)
- **Finding:** the DAG has retrained weekly since June. The latest run was 2026-11-01, giving 48 runs in the v1 experiment. Its container uses xgboost 2.1.4 and scikit-learn 1.3.2, so it produces a different model (XGBoost test KS 0.8358 vs v1's 0.805), with no promotion gate.
- **Decision:** pause `fraud_detection_dag` during v2. The runs it already logged are left untouched.
- **Consequence:** the README line "scheduled retraining is demonstrated, not exercised" was inaccurate and is corrected in v2. Bringing scheduled retraining back requires the D11 gate and the v1 library versions.

### D11. Promotion gate (none existed) · 2026-11-02 · tested locally
- **Context:** v1 hard-coded `'XGBoost'` as champion, and the DAG only logged "swap recommended". There was no promotion logic.
- **Decision:** a challenger replaces the champion only if it passes **every** rule:
  1. Each model is judged at its own validate cost-optimal threshold.
  2. On validate: alert precision ≥ the champion's, business cost ≤ the champion's, and KS ≥ 0.75.
  3. Only if validate passes: one confirmation on test, with test cost ≤ the champion's and KS ≥ 0.75.
- **Outcome is logged either way:** to the MLflow experiment `fraud-v2-promotion` (tagged `dataset_version`) and to `v2/promotion/records/`.
- **First real run, 2026-11-02:**
  - **Random Forest did not win.** Its precision is *higher* at its own threshold (0.5904 vs 0.5258), but its cost is higher too (£1,390 vs £1,210).
  - **Logistic Regression did not win.** It fails on both precision (0.4667) and cost (£1,610).
  - **XGBoost stays champion.**
- **Note:** judged at its own threshold, Random Forest looks much better than the 8.1% figure in D4, which was measured at the *champion's* threshold. The gate's fairer comparison still keeps XGBoost.

### D12. CI: lint new code, test, rebuild v1, build and smoke-test the API · 2026-11-02 · tested locally
- **Decision:** `.github/workflows/ci.yml` runs four jobs with no secrets and no paid services:
  1. **Lint**, for new code only: `ruff check v2 tests/v2`. v1 code is frozen, so it is not reformatted.
  2. **Tests and coverage** on Python 3.11, with the v1 library versions pinned.
  3. **v1 regression check**, the D9 script.
  4. **Docker build of the BentoML API** from the hash-checked frozen model, then a smoke test. The four demo transactions must return the same decision and the word-for-word same notice as the published site.
- **Coverage, stated honestly:** 25% of lines in `src/frauddetection` + `v2` (240 of 961).
  - 100%: `inference.py`.
  - 94%: the regression check.
  - 88%: the gate.
  - 53%: v1's `model_evaluation.py`, which the regression test exercises.
  - 0%: the v1 ingestion, transformation and training stages, and the one-off scripts. They have no unit tests yet, and none were added, because v1 is frozen.
- **Evidence:**
  - Every Python step of the workflow passed in a clean `python:3.11-slim` container.
  - The image built locally in 260 s (2.66 GB) and passed the smoke test.
  - CI on GitHub has not run yet; it runs on the next push.
- **Finding (left unchanged in v1):** the v1 API answers a malformed request (3 features instead of 30) with HTTP 500. It should be a 4xx error. The fix belongs in the v2 API (D13).

### D13. Service monitoring: Prometheus metrics in a v2 API, Grafana in Docker Compose · 2026-11-02 · tested locally
- **Decision:**
  - Add a new v2 API (`v2/serving/`). The v1 `bento_service/` is left untouched.
  - It uses the same model and the same `inference.py`, so every decision and notice is identical to v1. It adds Prometheus metrics with `prometheus_client`.
  - It returns **HTTP 400** for malformed input; v1 returned 500 (see D12).
  - A Compose stack (`v2/monitoring/`) runs the API, Prometheus 3.4.2 (scrape every 5 s) and Grafana 12.0.2. Ports bind to localhost only, every container has a memory limit, and no password is set in config.
- **Why not build metrics from scratch:** BentoML already exports request count and latency per endpoint at `/metrics`. v2 adds only what it lacks:
  - the decision mix by risk band;
  - the score distribution, with a bucket edge at the 0.07159 threshold;
  - model + SHAP time;
  - rejected inputs by reason;
  - a model-info gauge.
- **Relation to drift monitoring:** data drift (PSI by feature) stays on the Streamlit Drift page. The Grafana board adds a service-side signal: the live decline rate against v1's validate flag rate (0.227%).
- **Real results, 2026-11-02:**
  - **Setup:** one API container (1 worker), with the load generator on the same laptop. Traffic was the out-of-time test set replayed in time order, with 1% deliberately malformed requests.

    | Target rate | Achieved | Client p50 / p95 / p99 | Notes |
    |---|---|---|---|
    | 20 /s | 20.0 /s | 29 / 36 / 41 ms | |
    | 50 /s | 50.0 /s | 26 / 47 / 74 ms | first 50 /s run |
    | 50 /s | 50.0 /s | 22 / 38 / 61 ms | clean repeat, shown in `docs/images/grafana-service-health.png` |
    | 100 /s | 76.4 /s | 409 / 494 / 563 ms | saturated |
    | 200 /s | 85.3 /s | 360 / 444 / 591 ms | saturated |

  - **Capacity:** one worker on this laptop saturates at about **80 requests per second**.
  - **Server-side model + SHAP time:** p50 8.2 ms, p95 18.0 ms. Most request time is HTTP and framework overhead, not the model.
  - **Errors:** 0 server errors. Every malformed request got 400 (310 in total).
  - **Decline rate:** 0.22–0.28% at a steady 50 /s, against the v1 reference of 0.227%.
  - The raw run files are in `v2/monitoring/results/`.
- **Measurement error found and corrected:** the first two runs showed a flat 2.05 s latency. That was the Windows `localhost` lookup trying IPv6 first, not the API. Those runs were discarded and are not reported. The load generator and smoke test now use `127.0.0.1`.
- **CI:** the Docker job now builds and smoke-tests the v2 API (`--v2`). It requires HTTP 400 for bad input and checks that the metric names are exported.
- **Not done:** no alerting rules. Thresholds for alerts should come from a longer baseline than one evening of replays.

### D14. Streaming: Kafka + PyFlink scoring with proven parity to v1 batch · 2026-11-02 · tested locally
- **Decision:** a local Kafka (KRaft, 1 partition) and a Flink 1.20 cluster in Docker (`v2/streaming/`). PyFlink runs in the container, because it does not run natively on Windows. The stack is kept separate from Airflow; both Airflow stacks were stopped for the runs and restarted after.
- **Input:** the stream replays the DVC-versioned raw file, checked against `dataset_version` md5 `e90efcb8…`: 99 validate rows to warm the window, then all 42,722 out-of-time test rows. The label is never sent to the scorer.
- **How the job works:** one operator does ordering, de-duplication, v1's three engineered features and batched scoring, with SHAP and the customer notice for declines.
  - **Parallelism 1:** v1's 100-transaction window is global (the data has no customer IDs), so this one step cannot be parallelised.
  - **Bound:** out-of-orderness of 60 s of event time.
  - **Checkpoints:** every 5 s.
  - **Sink:** at-least-once.
- **Parity tolerances, fixed before any run:**
  - features within 1e-9;
  - scores within 1e-6;
  - zero decision flips;
  - no missing rows.

  The reference is v1's batch features, from the frozen bundle, and v1's batch scores, from the frozen pickle. Rebuilding v1's batch features from the raw file reproduces the frozen features exactly.
- **Real runs, all on this laptop:**

  | Run | Sent | Throughput | End-to-end p50 / p95 / p99 | Parity |
  |---|---|---|---|---|
  | clean, 1,000/s | 42,821 events | 992 /s | 528 / 829 / 1,490 ms | passed, all 42,722 test rows |
  | faults, 1,000/s | + 388 duplicates, 200 late within bound, 20 too late | 982 /s | 482 / 715 / 1,749 ms | passed (see below) |
  | ceiling, 3,000/s | 42,821 events | **2,497 /s sustained** | 2,967 / 3,354 / 3,505 ms (backlog) | passed |
  | restart, 500/s | TaskManager restarted at +30 s | 492 /s | 705 / 19,097 / 21,950 ms (recovery stall) | passed |

  - **Largest differences across all runs:** Amount_log 8.9e-16, rolling volatility 1.6e-11, score 9.3e-10. There were 0 decision flips, and the 81 declines match v1's 39 frauds caught plus 42 false alarms.
  - **Fault handling:**
    - All 388 duplicates were dropped.
    - All 200 late-within-bound events were re-ordered and scored identically to batch.
    - Exactly the 20 too-late events went to `late-events` and were not scored.
    - Their 1,979 following rows lost one payment from their 100-payment window. They are reported separately: the volatility feature differs in 1,969 of them, the score differs by up to 0.00075 in 32, and there are 0 decision flips. They are excluded from the parity claim, not hidden.
  - **Recovery:**
    - Flink made 3 restore attempts while the TaskManager rebooted; 2 failed with "no resources", then it **restored from checkpoint 7**.
    - The at-least-once sink then re-emitted 1,776 outputs, all identical to the first copies.
    - Duplicate drops counted 476 against 457 injected, because 19 injected duplicates were re-read after the rewind.
- **Three failures found and fixed on the way (all kept in this record):**
  1. **Watermark run-ahead.** The first design used Flink's own watermark for lateness. Under PyFlink, the Java-side watermark ran ahead of events still queued for the Python worker, and 41,821 of 42,821 events were wrongly marked late. Fix: event-time progress is tracked inside the operator, in processing order, with the same 60 s rule.
  2. **State layout too slow.**
     - Storing whole events in one pickled value gave 628 /s; a seen-set map lookup per event gave 533 /s, because every new key missed PyFlink's state cache.
     - Using the dense sequence number for de-duplication, and keeping only `(time, seq)` pairs in the heap, gave 958 /s.
     - 100 ms Python bundles (down from 1,000 ms) gave 992 /s and cut p50 latency from 1.04 s to 0.53 s.
  3. **Out of direct memory at 3× overload.** The first 3,000/s run failed after 10 restarts with `OutOfMemoryError: Direct buffer memory` in the Kafka fetcher. Fix: bounded fetch sizes (5 MB per fetch, 1 MB per partition) and 128 MB of task off-heap memory. The rerun sustained 2,497 /s.
- **Not proven, stated plainly:**
  - Throughput is single-machine, with the producer, Kafka, Flink and the checker all sharing one laptop.
  - The parity tolerance for scores (1e-6) is needed because 25 rows differ by about 1e-9 between XGBoost's `Booster.predict` and `predict_proba`. Decisions are unaffected.
  - The seen-and-gap state has no TTL. That's fine for a 43k-event replay; an unbounded stream would need state TTL.
- **MLflow:** all four runs are logged to the new experiment `fraud-v2-streaming`, tagged `dataset_version`, `baseline` and `decision=D14`.

### D15. Infrastructure as code: Terraform for AWS, designed only · 2026-11-02 · designed only
- **Decision:** Terraform in `v2/infra/terraform/` for ECR, S3, ECS Fargate, IAM, and a GitHub OIDC deploy role (no stored keys). Only `korede-folarin/fraud-detection-ml` on branch `main` can assume the role. **Not deployed:** `terraform apply` has never been run, there is no AWS account, and CI has no apply step.
- **Checks:** CI runs `fmt -check`, `init -backend=false`, `validate`, and `plan -refresh=false -var offline_plan=true` with mock credentials. Locally (Terraform 1.9.8 in Docker) this passed with **25 resources to add**. The plan proves the configuration and the resource graph, not IAM permissions.
- **Cost, from the AWS Price List API (eu-west-2, 2026-11-02):** about **$25.28 a month** as designed.
  - Fargate 0.5 vCPU + 1 GB, always on: $20.72.
  - Public IPv4 address: $3.65.
  - ECR: $0.30. Logs: $0.61. S3: about $0.
  - An Application Load Balancer, deliberately left out, would add $25.45. A NAT gateway, also left out, was not priced.
  - The first run of the estimator picked up the Gateway Load Balancer price ($0.0147/h), because the usage type is shared. It was corrected by filtering on product family to the ALB's $0.02646/h.
- **Changed from the brief (with the owner's approval in the feasibility answers):**
  - **No LocalStack.** Its free plan excludes ECR and ECS and now needs an account token.
  - **No moto either.** The API container has no AWS dependency at runtime (the model is baked into the image), so an emulator would only test the emulator.

### D16. Helm chart on kind: not done · 2026-11-02
- **Decision:** skipped the optional Helm/kind step.
- **Evidence:**
  - kind and Helm are not installed. Docker has 7.5 GiB, of which the two Airflow stacks use about 2.6 GiB.
  - The streaming stack needed that memory with Airflow stopped, and an earlier Flink stack on this machine had been killed with exit code 137.
  - A kind cluster plus loading the 2.7 GB API image would only fit with everything else stopped. It would add a deployment format, not new evidence about the model or the service.
- **Next:** if wanted, use Docker Desktop's built-in Kubernetes instead of kind (no extra install), with the Airflow stacks stopped.

### D17. Regression check hashes text files with normalised line endings · 2026-11-03 · tested locally
- **Finding:** the first two CI runs on GitHub failed the v1 regression check and the test job. The freeze manifest (D8) had recorded the SHA-256 of `evaluation_results.json` as checked out on Windows, with CRLF line endings. GitHub's Linux runner checks out the same file with LF endings, so its hash differed and the check reported "evaluation_results.json changed".
- **Why local testing missed it:** the clean-container test in D12 used a clone made on Windows, so it saw CRLF too.
- **Decision:** hash text files with line endings normalised to LF (`text_sha256` in `check_v1.py` and `build_v1_bundle.py`), and record the LF hash in `v1_frozen/MANIFEST.json`. That is the hash of the content git stores, so it is the same on every OS. Binary files (models, parquet) are still hashed byte for byte.
- **What did not change:** no v1 file and no v1 number. Only the manifest's recorded hash changed, from `e61272f0…` (CRLF) to `5fa3db2b…` (LF, the git blob).
- **Evidence:** the check passes on Windows (CRLF working tree) and on a fresh Linux clone (LF) in `python:3.11-slim`, where all 29 tests pass too.
