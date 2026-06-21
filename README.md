

A UK banking-standard fraud detection pipeline with real-time,
transaction-level explainability

Overview

The UK has no equivalent of the US adverse action notice requirement.
When an automated system declines a transaction here, the customer is
rarely told why in any meaningful way. Under the FCA's Consumer Duty,
firms must support retail customer understanding so that communications
"meet the information needs of retail customers," "are likely to be
understood," and "equip retail customers to make decisions that are
effective, timely and properly informed."1 Most fraud systems
deployed today do not meet that bar.

This pipeline does. Every declined transaction is accompanied by a
real-time, plain-English explanation generated from SHAP values computed
for that specific decision, not a static FAQ or a generic disclaimer,
built directly into the live REST API rather than bolted on afterward.

It sits on top of a fraud detection system built to UK banking production
standards: out-of-time validation, Information Value-driven feature
selection, cost-based threshold optimisation, and a champion model chosen
for trustworthy behaviour at its actual deployment threshold rather than
the highest leaderboard score. Orchestrated end to end with Airflow,
MLflow, DVC, BentoML, Docker, and a live monitoring dashboard.

Full reasoning for the explainability design is in
Explainability & FCA Consumer Duty
below. Full reasoning for every modelling decision is in
Why Each Decision Was Made.

Business impact, at a glance

Fraud capture at cost-optimal threshold87.5%Precision at deployment threshold (champion)52.7%Per-transaction explanation latencyReal-time, generated at scoringModel governanceFull champion-challenger comparison, documented rationaleDeploymentContainerised REST API (BentoML + Docker)MonitoringLive dashboard with automated drift alertsExperiment trackingEvery run logged to MLflow, data versioned with DVC


Why Each Decision Was Made

Out-of-Time Validation (70/15/15)

Random train/test splits are the most common source of optimistic bias in
fraud ML. When future transactions appear in training data, the model
learns patterns it could never have seen in production, inflating
reported performance and producing models that degrade immediately on
deployment.

I enforced strict temporal ordering: train on the earliest 70% of
transactions, validate on the next 15%, test on the final 15%. This
mirrors FCA model risk guidance and produces performance estimates that
reflect genuine out-of-sample capability.

Impact: Test set fraud rate (0.122%) diverged from train (0.193%),
confirming the split captured a genuine distributional shift rather than
a statistical artefact.

Information Value over Pearson Correlation

Correlation measures linear relationships. Fraud is not linear. Relying
on correlation alone systematically undervalues features with complex,
non-monotonic relationships to the target.

I applied optimal binning independently on the training set, never
touching validate or test data, to calculate Information Value for all
features. This revealed V14 and V12 as dominant predictors, and exposed
Amount's U-shaped fraud profile: high fraud concentration below £1.19
(card testing) and above £237 (account takeover): a pattern completely
invisible to correlation analysis.

Impact: Features that correlation ranked as moderate were revealed by IV
as the strongest predictors in the dataset. Feature selection based on IV
reduced dimensionality from 34 to 30 features with principled
justification rather than arbitrary cutoffs.

class_weight over SMOTE

Class imbalance in fraud detection is extreme: 492 fraud cases against
284,315 legitimate transactions. The instinct to rebalance via SMOTE is
understandable but counterproductive at this ratio: synthetic fraud
patterns fabricated by interpolation do not exist in production, and a
model trained on them learns a distribution that will never appear in
live data.

I handled imbalance natively within each model instead, preserving the
real training distribution while correcting for class frequency:


Logistic Regression → class_weight='balanced'
Random Forest → class_weight='balanced'
XGBoost → scale_pos_weight=518


All three approaches penalise missed fraud proportionally to its rarity
without distorting the data the models learn from.

Impact: XGBoost's average predicted score (0.16%) closely matched the
true validate fraud rate (0.131%). Its probabilities reflected the
actual fraud distribution rather than an artificially rebalanced one.
This becomes critical later, when the same business threshold is applied
across all three models.

Champion-Challenger Model Development

Regulatory best practice in UK banking requires model developers to
benchmark candidate models under identical conditions and articulate a
defensible rationale for champion selection, not simply deploy the
model with the highest score on a leaderboard.

I trained three models under identical out-of-time conditions: Logistic
Regression as the interpretable, FCA-aligned baseline, Random Forest as
the ensemble benchmark, and XGBoost as the industry-standard fraud
detection approach. Hyperparameters were set based on domain knowledge
rather than cross-validation tuning. Standard cross-validation violates
the temporal independence assumption of transaction data, and tuning on
this 48-hour window risks overfitting to its specific fraud patterns
rather than generalising.

On the validate set, all three models scored similarly well (KS >
0.90). On the test set, the picture changed: Random Forest edged
ahead of XGBoost on both KS and AUPRC, the two threshold-independent
ranking metrics. By leaderboard rules, Random Forest wins.

I still chose XGBoost. Here is why.

KS and AUPRC measure ranking quality: do fraud scores tend to be higher
than legitimate scores? They do not measure whether a score, taken in
isolation, means anything. A model can rank well while its
probabilities are uncalibrated.

I tested this directly. I applied XGBoost's cost-optimal threshold
(0.0784) to all three models and measured what happened at that single
operating point:

ModelPrecision at thresholdBehaviourLogistic Regression0.004Flags ~99.6% of all transactions, unusableRandom Forest0.1049 in 10 alerts are false alarmsXGBoost0.527Fewer than half of alerts are false

Random Forest's ranking advantage evaporated the moment a real decision
threshold was applied. Its scores were not calibrated to the true fraud
rate, so a low threshold caused it to flag almost everything. XGBoost's
scores were calibrated. Its average prediction matched the true fraud
rate closely enough that a low threshold remained meaningful.

Impact: XGBoost was selected as champion not because it won the
leaderboard, but because it was the only model whose scores remained
trustworthy at the threshold the bank would actually deploy. In a
regulated institution, a model risk team signs off the model it can
trust at the operating point, not the model with the best metric in
isolation.

Business Cost Threshold over F1 Optimisation

F1 score treats false positives and false negatives as equally costly.
In fraud detection they are not. Missing a £150 fraud transaction costs
15× more than a £10 false alarm investigation (the cost of a human
reviewing the flag).

I did not pick a threshold by hand. For XGBoost, I tested every possible
threshold and calculated total expected cost at each:

cost = (missed fraud × £150) + (false alarms × £10)

The F1-optimal threshold sat at 0.9929, blocking only at near-total
certainty, achieving precision of 100% but recall of only 71.2% on test.
The cost-optimal threshold sat at 0.0784, blocking far more aggressively.

Impact: The cost-optimal threshold lifted fraud capture from 78.6% to
87.5% on the validate set, an 11.2% improvement. Blocking more
aggressively does mean more flags reaching the manual review queue, but
that exact cost is already priced into the £10 false-alarm figure in the
formula above. The cost-optimal threshold is the point where catching
more fraud and overloading the review team are already balanced, not a
trade-off left for someone downstream to discover.

Velocity Features: Production Design Considerations

Rolling transaction features (average amount, volatility, fraud density,
transaction count over a 100-transaction window) were engineered and
their Information Value calculated. Three of the four produced near-zero
IV. Only rolling volatility survived feature selection.

This is a data structure limitation, not a methodology failure. Without
customer identifiers, rolling statistics mix different customers'
transaction histories, producing features that are mathematically valid
but behaviourally meaningless. In production, velocity features are
computed per customer via a feature store (Redis/Feast), capturing
individual behavioural baselines over 1h/24h/7d windows. These typically
rank among the highest-IV predictors in deployed fraud systems. This
dataset's anonymisation, a necessary confidentiality measure, removes the
customer dimension required for meaningful velocity computation.

Impact: IV analysis quantified the limitation precisely rather than
obscuring it, confirming that per-customer velocity features would be
expected to materially improve discrimination in a production
implementation with customer-level data.


Explainability & FCA Consumer Duty

The UK does not currently have a direct equivalent of the US adverse
action notice requirement (ECOA / Regulation B), which obliges lenders to
give consumers specific, individual reasons when a credit decision goes
against them. Following the UK's Data (Use and Access) Act 2025, which
narrowed Article 22 UK GDPR's protections around solely automated
decision-making, legal commentators have argued the UK should adopt a
comparable framework, since the current position leaves consumers with
materially weaker rights to an explanation than in the US. (See: Oxford
Business Law Blog, "Reasons for Automated Decisions"; FCA Consumer Duty,
fca.org.uk/firms/consumer-duty.)

The FCA's Consumer Duty, in force since 2023 and a stated supervisory
focus for 2026, requires that AI-influenced decisions affecting
consumers be tested for fairness and that consumers receive
communications they can understand. A model that declines a transaction
with no accompanying explanation does not meet that bar, even where no
explicit legal requirement yet forces the point.

I built this pipeline to anticipate that direction rather than wait for
it. Every declined transaction served by the API includes:


The fraud probability and risk band
The top contributing features, ranked by SHAP value computed for that
specific transaction, not global feature importance, and not raw
feature magnitude (an earlier version of this service ranked by raw
value, which surfaced large-but-irrelevant numbers ahead of the
features that actually drove the decision; this was corrected)
A plain-English adverse action notice naming the principal contributing
factors
An explicit statement that the decision was automated and may be
reviewed


Two design constraints worth being explicit about:


Only positive SHAP contributions are cited as reasons for decline.
A feature that pushed a transaction away from fraud must never
appear as a reason it was blocked. That would misrepresent the
model's own reasoning.
Honest feature naming. V1-V28 are PCA components from the original
anonymised bank data; their real-world meaning is genuinely unknown.
The notice describes them as "behavioural risk indicators" rather than
inventing a plausible-sounding interpretation. Only the engineered
features I built myself (transaction amount, time of day, recent
spending volatility) are described in concrete terms.


This is not a claim that UK regulation currently requires this. It is a
demonstration of what a fraud system designed ahead of that direction
looks like.

Why this matters commercially, not just academically

An unexplained decline is not a neutral outcome for a bank. It generates
a customer support call, and at scale, a measurable share of those
escalate to a complaint with the Financial Ombudsman Service, where firms
already pay a case fee regardless of outcome and absorb the
compensation, reputational, and operational cost of upheld complaints.
The consumer understanding outcome is an active FCA supervisory focus
area,2 with a dedicated review of how firms deliver it published in
March 2026,3 which makes this a cost banks are increasingly unable to
treat as background noise.

An automated, accurate, per-transaction explanation addresses this on
two fronts at once:


Regulatory exposure. It gives the bank a defensible, contemporaneous
record of why a specific transaction was declined, generated at the
moment of decision rather than reconstructed after a complaint is
raised. That record is the difference between "the model said no" and
a documented, auditable rationale a compliance team can stand behind.
Operational cost. A customer who receives a specific, accurate
reason is measurably less likely to call the contact centre or escalate
a complaint than one who receives a generic decline. At fraud-detection
volumes, even a small reduction in unnecessary escalations is a
material saving in contact centre and complaints-handling cost, on top
of the £10-per-review figure already built into the threshold
calculation above.


Neither of these requires the UK to pass an adverse-action-notice law
first. Both are available today to any institution willing to build the
explainability layer in before it is mandated rather than after.


Results

Validate set

ModelKSAUPRCF1XGBoost0.9240.8690.871Random Forest0.9030.8660.880Logistic Regression0.9220.8320.811

Test set (out-of-time)

ModelKSAUPRCF1Precision @ business thresholdXGBoost (champion)0.8050.7600.8320.527Random Forest0.8430.7750.8480.104Logistic Regression0.8570.6970.7870.004

Random Forest scores higher on test-set KS, AUPRC and F1. XGBoost was
selected as champion regardless. See Champion-Challenger Model
Development above for the full reasoning. Score calibration and
behaviour at the actual deployment threshold mattered more than ranking
metrics in isolation.


Experiment Tracking

All model runs tracked via MLflow on DagsHub:
👉 https://dagshub.com/korede-folarin/fraud-detection-ml

Includes KS, AUPRC, F1, Precision, Recall per model per dataset, business
cost threshold per run, model artifacts, and champion-challenger
comparison across validate and test sets.


Live Components


Monitoring dashboard (Streamlit):
kf-fraud-detection-ml.streamlit.app: model performance, drift alerts,
business threshold analysis, champion rationale
REST API (BentoML): real-time fraud scoring with SHAP-based
explainability and adverse action notices, containerised with Docker



Production Stack

Apache Airflow (weekly retraining) · MLflow / DagsHub (experiment
tracking) · DVC (data versioning) · BentoML (model serving) · SHAP
(explainability) · Docker (containerisation) · Streamlit (monitoring)


Running the API

bashcd bento_service
python save_model.py          # registers model + SHAP explainer
bentoml serve service:FraudDetectionService --reload

Or via Docker:

bashbentoml build
bentoml containerize fraud_detection:latest
docker run --rm -p 3000:3000 fraud_detection:latest

API docs available at http://localhost:3000 once running.


Dataset

Credit Card Fraud Detection, ULB Machine Learning Group
👉 kaggle.com/datasets/mlg-ulb/creditcardfraud

Place creditcard.csv in data/ folder before running notebooks.


References


Setup

bashgit clone https://github.com/korede-folarin/fraud-detection-ml
cd fraud-detection-ml
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt


Korede Folarin

Footnotes



FCA, Consumer Duty: navigating the Consumer Understanding
Outcome. The Outcome requires firms to ensure communications "meet
the information needs of retail customers," "are likely to be
understood by retail customers," and "equip retail customers to make
decisions that are effective, timely and properly informed" (PRIN
2A.5). https://www.hoganlovells.com/en/publications/consumer-duty-navigating-the-consumer-understanding-outcome ↩



FCA, Our Consumer Duty focus areas. Confirms the Consumer
Duty remains a 2025/26 supervisory priority, with a dedicated review
of the consumer understanding outcome.
https://www.fca.org.uk/publications/corporate-documents/consumer-duty-focus-areas ↩



FCA, Consumer understanding: good practice and areas for
improvement (March 2026). Review of 38 firms across banking,
payments and consumer finance on how communications support customer
understanding under the Duty.
https://www.fca.org.uk/publications/good-and-poor-practice/consumer-understanding-good-practice-areas-improvement ↩