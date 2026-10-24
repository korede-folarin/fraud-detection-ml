# Fraud Detection Pipeline with Explainable Decisioning

*A fraud detection pipeline built to UK banking model-validation practice, with a real-time,
plain-English explanation for every declined transaction.*

![CI](https://github.com/korede-folarin/fraud-detection-ml/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Champion_Model-orange)
![BentoML](https://img.shields.io/badge/BentoML-REST_API-green)
![Docker](https://img.shields.io/badge/Docker-Containerised-blue)
![Airflow](https://img.shields.io/badge/Apache_Airflow-Orchestration-017CEE)
![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-0194E2)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B)
![DVC](https://img.shields.io/badge/DVC-Data_Versioning-945DD6)

## 🔍 Try it live

1. **Live demo:** [korede-folarin.github.io/fraud-detection-ml](https://korede-folarin.github.io/fraud-detection-ml/).
   It loads instantly and is always on. It replays four real out-of-time test transactions (a fraud,
   a legitimate payment, the nearest approval and a false alarm), each with its decision, SHAP
   reasons and the exact customer notice. It also has the threshold/cost trade-off, the
   champion-challenger comparison and drift monitoring. Nothing on it calls a model, and every
   number is read from the pipeline's output files.
2. **Interactive dashboard:** [kf-fraud-detection-ml.streamlit.app](https://kf-fraud-detection-ml.streamlit.app)
   (it can take about 30 seconds to wake). It scores transactions live:

- **Score a real transaction.** Load a fraud, a legitimate payment, a borderline case or a false alarm
  from the out-of-time test set, and the champion model decides APPROVE or DECLINE.
- **Change its features** (amount, time of day, the top risk indicators) and watch the decision flip.
- **See why.** A SHAP waterfall shows what drove *this* decision, plus the exact notice the customer
  would receive. It runs the same scoring code as the REST API.
- **Move the decision threshold** and see fraud caught, false alarms and £ cost change on both
  hold-out sets.

## Links

- **Live demo:** https://korede-folarin.github.io/fraud-detection-ml/
- **Interactive dashboard:** https://kf-fraud-detection-ml.streamlit.app
- **GitHub repository:** https://github.com/korede-folarin/fraud-detection-ml
- **MLflow experiments:** https://dagshub.com/korede-folarin/fraud-detection-ml.mlflow

## Overview

The UK has no equivalent of the US adverse action notice requirement. When an automated system
declines a transaction here, the customer is rarely told why in any meaningful way. Under the FCA's
Consumer Duty, firms must support retail customer understanding so that communications "meet the
information needs of retail customers," "are likely to be understood," and "equip retail customers
to make decisions that are effective, timely and properly informed."[^1] Most fraud systems deployed
today do not meet that bar.

This pipeline is built to. Every decline it returns comes with a real-time, plain-English
explanation generated from SHAP values computed for that specific decision. It is not a static FAQ
or a generic disclaimer, and it is built into the scoring code rather than bolted on afterward.

Underneath is a fraud model developed the way UK bank model-validation teams expect:
out-of-time validation, Information Value-driven feature selection, cost-based threshold
optimisation, and a champion chosen for trustworthy behaviour at its deployment threshold rather
than the highest leaderboard score. It is orchestrated with Airflow, tracked in MLflow, versioned
with DVC, served with BentoML and Docker, and monitored in a live Streamlit dashboard.

Full reasoning for every modelling decision is in [Why Each Decision Was Made](#why-each-decision-was-made).
The explainability design is in [Explainability & FCA Consumer Duty](#explainability--fca-consumer-duty).

### At a glance

Champion XGBoost on the **out-of-time test set** (the last 6 hours of data, never used for training
or tuning), at the deployment threshold chosen on the validate set:

| | |
|---|---|
| Fraud caught | **75.0%** (39 of 52) |
| Alert precision | **48.1%** (about 1 in 2 alerts is real fraud) |
| Manual review load | **19 alerts per 10,000 transactions** |
| KS / AUPRC | **0.805 / 0.760** (AUPRC is 624× the random baseline) |
| Explanations | Per-transaction SHAP reasons, generated at scoring time |
| Governance | Three-model champion-challenger comparison with documented rationale |
| Deployment | BentoML REST API, containerised with Docker; live Streamlit demo |
| Monitoring | KS retraining trigger (Airflow), PSI drift by feature, threshold cost tracking |

---

## Why Each Decision Was Made

### Out-of-time validation (70/15/15)

Random train/test splits are the most common source of optimistic bias in fraud ML. When future
transactions appear in training data, the model learns patterns it could never have seen in
production. That inflates reported performance and produces models that degrade as soon as they
are deployed.

I enforced strict temporal ordering:

| Split | Hours of the 48-hour window | Transactions | Fraud | Fraud rate |
|---|---|---|---|---|
| Train | 0 – 36.9 | 199,364 | 384 | 0.193% |
| Validate | 36.9 – 42.0 | 42,721 | 56 | 0.131% |
| Test | 42.0 – 48 | 42,722 | 52 | 0.122% |

This is the out-of-sample discipline that UK bank model-risk frameworks expect (see the PRA's
SS1/23, *Model risk management principles for banks*). The validate set picks the threshold, and
the test set is only looked at once.

**Impact:** the fraud rate fell from 0.193% in train to 0.122% in test, a genuine distribution
shift over time that a random split would have hidden.

### Information Value over Pearson correlation

Correlation measures linear relationships, and fraud is not linear. Relying on correlation alone
undervalues features with non-monotonic relationships to the target.

I applied optimal binning to the training set only, never touching validate or test, to calculate
Information Value for every candidate feature. V14 and V12 came out as the dominant predictors. IV
also exposed a U-shaped fraud profile in transaction amount that correlation misses entirely:

| Amount (training set) | Fraud rate |
|---|---|
| below €1.19 | 0.67% |
| €1.19 – €237 | 0.11% |
| above €237 | 0.35% |

High fraud at tiny amounts is consistent with card testing, and at large amounts with account
takeover.

**Impact:** IV-based selection reduced 34 candidate features to 30 with a principled cut-off.

### `class_weight` over SMOTE

Class imbalance here is extreme: 492 frauds against 284,315 legitimate transactions. SMOTE is the
instinctive fix, but at this ratio it is counterproductive. Fraud patterns made up by
interpolation do not exist in production, so a model trained on them learns a distribution it will
never see.

I handled imbalance inside each model instead, keeping the real training distribution:

- Logistic Regression → `class_weight='balanced'`
- Random Forest → `class_weight='balanced'`
- XGBoost → `scale_pos_weight=518` (the legit:fraud ratio in the training set)

**Impact:** XGBoost's average predicted score on validate (0.16%) closely matched the true fraud rate
(0.13%). Its probabilities reflected the real fraud distribution, which becomes decisive when one
threshold is applied to all three models.

### Champion-challenger model development

Good model governance means benchmarking candidate models under identical conditions and giving a
defensible reason for the champion, not just deploying the leaderboard winner.

I trained three models under identical out-of-time conditions:

- Logistic Regression, the interpretable baseline
- Random Forest, the ensemble benchmark
- XGBoost, the industry-standard fraud approach

Hyperparameters were set from domain knowledge rather than cross-validation tuning. Standard
cross-validation breaks the time ordering of transaction data, and tuning on a 48-hour window risks
fitting its particular fraud bursts.

On validate, all three scored similarly (KS > 0.92). On test, Random Forest edged ahead of XGBoost
on both KS and AUPRC, the two threshold-independent ranking metrics. By leaderboard rules,
Random Forest wins.

I still chose XGBoost. KS and AUPRC measure *ranking*: do fraud scores tend to be higher than
legitimate ones? They do not measure whether a score means anything on its own, and a bank deploys
a threshold, not a ranking.

So I applied XGBoost's deployment threshold (0.0716) to all three models on the test set:

| Model | Alert precision | Share of all transactions flagged | Behaviour |
|---|---|---|---|
| Logistic Regression | 0.4% | 31.5% | Flags almost a third of all payments. Unusable. |
| Random Forest | 8.1% | 1.3% | About 12 of every 13 alerts are false alarms |
| **XGBoost** | **48.1%** | **0.19%** | About half of alerts are real fraud |

Random Forest's ranking advantage disappears once a real threshold is applied. Its scores are not
calibrated to the true fraud rate (its mean score on validate is 1.5% vs a true rate of 0.13%), so
a low threshold makes it flag far too much. XGBoost's scores are calibrated, so the same threshold
stays meaningful.

**Impact:** XGBoost was chosen because it was the only model whose scores could be trusted at the
threshold a bank would actually deploy. Random Forest remains the registered challenger.

**How much weight the evidence can bear.** With 52 test frauds, one case moves recall by 1.9 points.
The KS and AUPRC gaps between the models are small next to that noise; the precision gap at the
threshold (48.1% vs 8.1%) is not, and it is what the decision rests on. The fair challenge to this
choice is that Random Forest's scores could be recalibrated (for example with isotonic regression)
and might then compete. That was not tested, and it is the first experiment I would run on the
challenger.

### Business cost threshold over F1 optimisation

F1 treats false positives and false negatives as equally costly. In fraud detection they are not:
missing a £150 fraud costs 15× more than a £10 false-alarm review.

Rather than pick a threshold by hand, I computed the exact expected cost at every possible threshold
on the **validate set**:

```
cost = (missed fraud × £150) + (false alarms × £10)
```

and froze the minimum-cost threshold (0.0716) before looking at test.

| Threshold (chosen on validate) | Validate: caught / precision / cost | Test: caught / precision / cost |
|---|---|---|
| Cost-optimal, 0.0716 | 91.1% / 52.6% / **£1,210** | 75.0% / 48.1% / £2,370 |
| F1-optimal, 0.9311 | 78.6% / 97.8% / £1,810 | 73.1% / 88.4% / **£2,150** |

**Impact, stated honestly:** on validate, the cost-optimal threshold caught 12.5 points more fraud
and cut cost by a third. On the later test window it did not hold up: it caught one more fraud than
the F1 threshold for 37 more false alarms, and cost £220 more. With only 56 validate frauds, the
cost-optimal point is unstable over time.

That is the case for what a production version would add: recalibrate the threshold on a rolling
window, and monitor alert precision alongside KS. You can explore this trade-off with the threshold
slider in the live dashboard.

### Velocity features: production design considerations

Rolling transaction features were engineered over a 100-transaction window: average amount,
volatility, fraud density and transaction count. Their Information Value was calculated, and only
rolling volatility survived feature selection.

This is a data limitation, not a method failure. Without customer identifiers, rolling statistics
mix different customers' histories. They are mathematically valid but behaviourally meaningless.
In production, velocity features are computed per customer through a feature store (Redis/Feast)
over 1h/24h/7d windows, and they typically rank among the strongest predictors. The dataset's
anonymisation removes the customer dimension they need.

**Impact:** IV quantified the limitation rather than hiding it, and points to per-customer velocity
features as the clearest improvement with real customer-level data.

---

## Explainability & FCA Consumer Duty

The UK does not currently have a direct equivalent of the US adverse action notice requirement
(ECOA / Regulation B), which obliges lenders to give consumers specific reasons when a credit
decision goes against them. The UK's Data (Use and Access) Act 2025 narrowed Article 22 UK GDPR's
protections around solely automated decision-making. Since then, legal commentators have argued
the UK should adopt a comparable framework, because consumers here have materially weaker rights to
an explanation than in the US. (See: Oxford Business Law Blog, "Reasons for Automated Decisions";
FCA Consumer Duty, fca.org.uk/firms/consumer-duty.)

The FCA's Consumer Duty, in force since 2023, requires that consumers receive communications they
can understand, and the consumer understanding outcome remains a supervisory focus.[^2][^3] A model
that declines a transaction with no explanation does not meet that bar, even where no explicit legal
requirement yet forces the point.

I built this pipeline to anticipate that direction rather than wait for it. Every decline returned
by the API, and by the live demo, includes:

- The fraud probability and risk band.
- The top contributing features, ranked by SHAP values computed for *that* transaction. This is not
  global feature importance and not raw feature size. An earlier version ranked by raw value, which
  surfaced large but irrelevant numbers ahead of the features that actually drove the decision;
  that was fixed.
- A plain-English notice naming the main contributing factors.
- An explicit statement that the decision was automated and can be reviewed.

What the explanation can and cannot do: on this dataset most reasons are anonymised PCA
components, so a customer learns that a reason exists, not what it is. That is a limit of the data,
not of the method; with a bank's named features the same code would name them. And an explanation
describes the model, it does not prove it right. A false alarm gets reasons that sound just as
confident as a caught fraud, which is why every notice offers a manual review.

Two design constraints, both enforced by unit tests in [`tests/test_inference.py`](tests/test_inference.py):

1. **Only positive SHAP contributions are cited as reasons for a decline.** A feature that pushed a
   transaction *away* from fraud must never appear as a reason it was blocked; that would
   misrepresent the model's own reasoning.
2. **Honest feature naming.** V1–V28 are PCA components from the original anonymised bank data, and
   their real-world meaning is unknown. The notice calls them "behavioural risk indicators" rather
   than inventing a plausible-sounding meaning. Only the features I engineered (transaction amount,
   time of day, recent spending volatility) are described in concrete terms.

SHAP values are computed with XGBoost's native TreeSHAP (`pred_contribs`). This is exact, matches
`shap.TreeExplainer` to floating-point precision, and adds no extra dependency at serving time.

This is not a claim that UK regulation currently requires this. It shows what a fraud system
designed ahead of that direction looks like.

### Why this matters commercially, not just academically

An unexplained decline is not a neutral outcome for a bank. It generates a support call, and at
scale some of those escalate to the Financial Ombudsman Service. Firms pay a case fee there
regardless of outcome, and absorb the compensation, reputational and operational cost of upheld
complaints.

An accurate, per-transaction explanation helps on two fronts:

- **Regulatory exposure.** It gives a defensible, contemporaneous record of why a specific
  transaction was declined. It is created at the moment of decision, not reconstructed after a
  complaint, which is the difference between "the model said no" and an auditable rationale a
  compliance team can stand behind.
- **Operational cost.** A customer given a specific, accurate reason is less likely to call the
  contact centre or escalate than one given a generic decline. At fraud-detection volumes, even a
  small reduction in unnecessary escalations is a material saving.

Neither requires the UK to pass an adverse-action-notice law first.

---

## Results

All figures come from [`artifacts/model_evaluation/evaluation_results.json`](artifacts/model_evaluation/evaluation_results.json),
written by the pipeline's evaluation stage. The dashboard reads the same file.

**Validate set** (ranking metrics; F1 at each model's own best threshold)

| Model | KS | AUPRC | F1 |
|---|---|---|---|
| XGBoost | 0.926 | 0.874 | 0.871 |
| Random Forest | 0.920 | 0.867 | 0.871 |
| Logistic Regression | 0.922 | 0.832 | 0.811 |

**Test set, out-of-time**

| Model | KS | AUPRC | F1 | Precision @ deployment threshold |
|---|---|---|---|---|
| **XGBoost (champion)** | 0.805 | 0.760 | 0.832 | **0.481** |
| Random Forest | 0.843 | 0.775 | 0.848 | 0.081 |
| Logistic Regression | 0.857 | 0.697 | 0.787 | 0.004 |

Random Forest scores higher on test KS, AUPRC and F1, and XGBoost was still chosen as champion; see
[Champion-challenger model development](#champion-challenger-model-development). Calibration and
behaviour at the deployment threshold mattered more than ranking metrics alone.

## Monitoring & drift

- **Retraining trigger:** the Airflow `drift_check` task reads the evaluation results after each run
  and flags retraining when test KS falls below 0.75 (currently 0.805).
- **Feature drift:** the dashboard reports the Population Stability Index (train vs test) for the
  eight most influential features. V1 shows a significant shift (PSI 1.01) and V11 a moderate-to-
  significant one (0.34). V14, the top feature by IV and gain, is stable (0.08).
- **What the signals disagree on:** KS still reads healthy while V1's PSI is above 1, and the
  threshold section shows the operating point has already broken. A KS-only trigger would miss
  both, so a production monitor should watch PSI and alert precision as well.
- **Fraud bursts:** fraud is concentrated in short bursts in low-volume hours. Hour 26 (2.05% fraud)
  is 54% higher than the same clock hour on day one (hour 2, 1.33%).

---

## Live components

- **Static demo site (GitHub Pages):** [korede-folarin.github.io/fraud-detection-ml](https://korede-folarin.github.io/fraud-detection-ml/).
  It is plain HTML/CSS/JS with self-hosted fonts and makes no outside requests. Its data is written by
  `python -m src.site.build`, which scores the four samples with the same `inference.py` code as the
  API. A test checks that the site's notices match the API word for word.
- **Streamlit dashboard + demo:** [kf-fraud-detection-ml.streamlit.app](https://kf-fraud-detection-ml.streamlit.app).
  It has an interactive scoring page plus model performance, threshold/cost, drift and champion
  rationale pages. Every chart is drawn from pipeline outputs; nothing is simulated.
- **REST API (BentoML):** real-time scoring with SHAP explanations and decline notices,
  containerised with Docker. It runs locally (instructions below) and is not publicly hosted.

## Production stack

Apache Airflow (scheduled retraining) · MLflow / DagsHub (experiment tracking) · DVC (data
versioning) · BentoML (model serving) · SHAP (explainability) · Docker (containerisation) ·
Streamlit (interactive dashboard) · GitHub Pages (static demo) · GitHub Actions (tests, deploy)

## Project structure

```
src/frauddetection/
  components/          # ingestion, transformation, training, evaluation, dashboard export
  pipeline/            # stage runners used by main.py and the Airflow DAG
  inference.py         # scoring + SHAP explanation, shared by the API and the dashboard
dags/                  # Airflow DAG: ingest → transform → train → evaluate → drift check
bento_service/         # BentoML service and model registration
dashboard/             # Streamlit app + assets (champion model as XGBoost JSON, demo sample)
site/                  # static GitHub Pages demo (data/ is generated by src/site/build.py)
src/site/build.py      # builds site/data/*.json from the pipeline outputs
artifacts/             # pipeline outputs (evaluation results, dashboard data, IV rankings)
notebooks/             # EDA, feature engineering, modelling and evaluation research
tests/                 # decision/explanation rules; site data matches the API
```

---

## Setup

```bash
git clone https://github.com/korede-folarin/fraud-detection-ml
cd fraud-detection-ml
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Place `creditcard.csv` in `data/` (or `dvc pull`), then run the full pipeline:

```bash
python main.py
```

The evaluation stage logs to MLflow on DagsHub when `DAGSHUB_TOKEN` is set, and skips remote
logging otherwise. It also writes the dashboard data and the demo assets.

### Running the dashboard

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

### Rebuilding the demo site

```bash
python -m src.site.build
python -m http.server 8600 --directory site
```

Pushing changes under `site/` to `main` deploys it via `.github/workflows/pages.yml`.

### Running the API

From the repository root:

```bash
python bento_service/save_model.py
bentoml serve bento_service.service:FraudDetectionService --reload
```

Or via Docker:

```bash
bentoml build -f bento_service/bentofile.yaml .
bentoml containerize fraud_detection:latest
docker run --rm -p 3000:3000 fraud_detection:latest
```

API docs are at http://localhost:3000 once running.

### Running the tests

```bash
pytest tests
```

## Dataset

Credit Card Fraud Detection, ULB Machine Learning Group:
[kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
It contains two days of European cardholder transactions from September 2013. Amounts are in euros;
the £ cost assumptions above are illustrative business inputs.

## Limitations

- **48 hours of data.** Out-of-time results rest on 52 test frauds, so every test metric has wide
  uncertainty. A production model would be validated on 12–24 months.
- **No hyperparameter tuning**, by design (see above); time-series-aware tuning is the next step.
- **Cost inputs are assumptions.** £150 and £10 are illustrative, and the optimal threshold depends
  only on their ratio. In a bank that ratio would be agreed with fraud operations and complaints
  teams before the optimiser's output is trusted.
- **Threshold stability.** The cost-optimal threshold did not generalise to the test window; it
  needs rolling recalibration.
- **Scheduled retraining is demonstrated, not exercised.** The Airflow DAG retrains weekly, but on
  this static dataset each run sees the same data.

---

*Korede Folarin*

[^1]: FCA, Consumer Duty: navigating the Consumer Understanding Outcome. The Outcome requires firms
    to ensure communications "meet the information needs of retail customers," "are likely to be
    understood by retail customers," and "equip retail customers to make decisions that are
    effective, timely and properly informed" (PRIN 2A.5).
    https://www.hoganlovells.com/en/publications/consumer-duty-navigating-the-consumer-understanding-outcome

[^2]: FCA, Our Consumer Duty focus areas. Confirms the Consumer Duty remains a 2025/26 supervisory
    priority, with a dedicated review of the consumer understanding outcome.
    https://www.fca.org.uk/publications/corporate-documents/consumer-duty-focus-areas

[^3]: FCA, Consumer understanding: good practice and areas for improvement (March 2026). Review of
    38 firms across banking, payments and consumer finance on how communications support customer
    understanding under the Duty.
    https://www.fca.org.uk/publications/good-and-poor-practice/consumer-understanding-good-practice-areas-improvement
