import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from frauddetection.inference import decide, shap_contributions, describe_feature  # noqa: E402

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection · Dashboard · Korede Folarin",
    page_icon="🟪",
    layout="wide",
    initial_sidebar_state="auto"  # collapsed on phones so the demo is the first thing seen
)

SITE_URL = "https://korede-folarin.github.io/fraud-detection-ml/"
REPO_URL = "https://github.com/korede-folarin/fraud-detection-ml"
MLFLOW_URL = "https://dagshub.com/korede-folarin/fraud-detection-ml.mlflow"

# ─────────────────────────────────────────────
# AUBERGINE VISUAL DIRECTION — same tokens as site/style.css
# Theme colours and self-hosted fonts are set in .streamlit/config.toml;
# this block adds the shared components (cards, stamp, track, notes)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    :root{
        --paper:#ebe7ea; --card:#f6f3f5; --ink:#110d12; --muted:#665c68;
        --key:#3b2142; --on-key:#f1eaf2; --soft:#d3c7d5; --line:#cfc6d1; --orange:#ff4f12;
        --rule:1.5px solid var(--ink); --mono:"JetBrains Mono",ui-monospace,monospace;
    }
    .main .block-container, [data-testid="stMainBlockContainer"]{ max-width:1280px; padding-top:2.2rem; }

    /* type */
    .kicker{ font-family:var(--mono); font-weight:500; font-size:12px; letter-spacing:.04em;
             text-transform:uppercase; color:var(--key); margin:0 0 .6rem; }
    .page-title{ font-weight:800; text-transform:uppercase; letter-spacing:-.05em; line-height:.92;
                 font-size:clamp(34px,4.4vw,64px); color:var(--ink); margin:0 0 .9rem; }
    .lede{ color:var(--muted); max-width:72ch; font-size:16.5px; line-height:1.5; margin:0 0 1.2rem; }
    .mono-label{ font-family:var(--mono); font-weight:500; font-size:11.5px; letter-spacing:.04em;
                 text-transform:uppercase; color:var(--muted); line-height:1.5; margin:0; }
    .muted{ color:var(--muted); font-size:14.5px; }

    /* stat cards — first one solid aubergine, the rest outlined */
    .stat{ border:var(--rule); border-radius:14px; padding:16px 18px; margin-bottom:.8rem;
           display:flex; flex-direction:column; gap:8px; min-height:132px; }
    .stat-solid{ background:var(--key); border-color:var(--key); }
    .stat-solid .mono-label, .stat-solid .stat-sub{ color:color-mix(in srgb,var(--on-key) 78%,transparent); }
    .stat-solid .stat-value{ color:var(--on-key); }
    .stat-value{ font-weight:800; letter-spacing:-.06em; font-variant-numeric:tabular-nums;
                 font-size:clamp(28px,2.6vw,40px); line-height:1; color:var(--ink); }
    .stat-sub{ color:var(--muted); font-size:13.5px; }
    .chip-orange{ display:inline-block; background:var(--orange); color:var(--ink); font-family:var(--mono);
                  font-size:11px; letter-spacing:.04em; text-transform:uppercase; padding:2px 9px; border-radius:999px; }

    /* verdict card */
    .verdict{ background:var(--key); color:var(--on-key); border-radius:14px; padding:22px 24px; margin-bottom:12px; }
    .verdict .mono-label{ color:color-mix(in srgb,var(--on-key) 78%,transparent); }
    .stamp-row{ display:flex; align-items:center; gap:14px; margin:12px 0 18px; flex-wrap:wrap; }
    .stamp{ display:inline-block; font-weight:800; text-transform:uppercase; line-height:1; }
    .stamp.decline{ background:var(--orange); color:var(--ink); border:2px solid var(--ink); border-radius:8px;
                    font-size:24px; padding:11px 16px; transform:rotate(-3deg); }
    .stamp.approve{ color:var(--on-key); border:1.5px solid var(--on-key); border-radius:999px; font-size:19px; padding:9px 18px; }
    .prob{ font-weight:800; letter-spacing:-.06em; font-variant-numeric:tabular-nums;
           font-size:clamp(52px,6vw,92px); line-height:.95; margin:4px 0 14px; color:var(--on-key); }
    .track{ position:relative; height:12px; border-radius:999px; background:color-mix(in srgb,var(--on-key) 18%,transparent); margin-top:26px; }
    .track-fill{ position:absolute; inset:0 auto 0 0; border-radius:999px; background:var(--on-key); }
    .track-dot{ position:absolute; top:50%; width:20px; height:20px; margin:-10px 0 0 -10px; border-radius:50%;
                background:var(--on-key); border:3px solid var(--key); box-shadow:0 0 0 1.5px var(--on-key); }
    .track-mark{ position:absolute; top:-14px; bottom:-14px; border-left:3px solid var(--orange); }
    .track-mark span{ position:absolute; bottom:100%; left:6px; white-space:nowrap; color:var(--on-key); }
    .track-scale{ display:flex; justify-content:space-between; margin-top:14px; }
    .actual{ margin-top:18px; font-size:17px; }

    /* customer notice */
    .notice{ background:var(--card); border:var(--rule); border-left:4px solid var(--key); border-radius:14px; padding:16px 20px; }
    .notice p:last-child{ margin:6px 0 0; font-size:16.5px; line-height:1.5; color:var(--ink); }

    /* judgement + honest notes */
    .judgement{ border-top:var(--rule); border-bottom:var(--rule); padding:16px 0; margin:1.4rem 0 .6rem;
                display:grid; grid-template-columns:130px 1fr; gap:6px 24px; }
    .judgement .mono-label{ color:var(--key); font-weight:700; }
    .judgement p:not(.mono-label){ grid-column:2; max-width:80ch; margin:0; color:var(--ink); font-size:15.5px; line-height:1.55; }
    .judgement p+p:not(.mono-label){ margin-top:6px; }
    .honest{ background:var(--card); border:var(--rule); border-radius:14px; padding:16px 20px; margin-top:.8rem; }
    .honest p:last-child{ margin:6px 0 0; color:var(--ink); font-size:15.5px; line-height:1.55; max-width:84ch; }
    .health{ border:var(--rule); border-radius:14px; padding:12px 16px; margin:.2rem 0 1rem; font-size:15px; }
    .health strong{ font-weight:700; }

    /* confusion matrix */
    .cm{ display:grid; grid-template-columns:auto 1fr 1fr; gap:6px; font-variant-numeric:tabular-nums; }
    .cm div{ border:1px solid var(--line); border-radius:8px; padding:10px 12px; background:var(--card); }
    .cm .h{ border:0; background:transparent; padding:2px 4px; }
    .cm .v{ font-weight:800; font-size:22px; letter-spacing:-.03em; }
    .cm .tp{ background:var(--key); border-color:var(--key); color:var(--on-key); }
    .cm-title{ font-weight:700; margin:0 0 6px; }

    /* sidebar (aubergine via theme.sidebar) */
    section[data-testid="stSidebar"] .wordmark{ font-weight:700; font-size:21px; letter-spacing:-.02em; color:var(--on-key); margin:0; }
    section[data-testid="stSidebar"] .mono-label{ color:color-mix(in srgb,var(--on-key) 72%,transparent); }
    section[data-testid="stSidebar"] .facts{ font-family:var(--mono); font-size:12px; line-height:1.9; color:var(--on-key); }
    section[data-testid="stSidebar"] .facts a{ color:var(--on-key); }
    section[data-testid="stSidebar"] hr{ border-color:color-mix(in srgb,var(--on-key) 30%,transparent); }

    /* Streamlit's markdown sets p font-size; keep the shared type scale */
    [data-testid="stMarkdownContainer"] .mono-label{ font-size:11.5px !important; line-height:1.5 !important; }
    [data-testid="stMarkdownContainer"] .kicker{ font-size:12px !important; }
    [data-testid="stMarkdownContainer"] .lede{ font-size:16.5px !important; }
    [data-testid="stMarkdownContainer"] .muted{ font-size:14.5px !important; }
    [data-testid="stMarkdownContainer"] .facts{ font-size:12px !important; }
    [data-testid="stMarkdownContainer"] .verdict .actual{ font-size:17px !important; }
    [data-testid="stMainBlockContainer"]{ padding-top:4.5rem !important; }
    [data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"]:has(.js-plotly-plot){ background:var(--card); }

    @media (max-width:640px){
        .judgement{ grid-template-columns:1fr; }
        .judgement p:not(.mono-label){ grid-column:1; }
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING — every number below comes from these pipeline outputs
# (written by the model evaluation stage, see dashboard_export.py)
# ─────────────────────────────────────────────
EVAL_DIR   = ROOT / "artifacts" / "model_evaluation"
ASSETS_DIR = ROOT / "dashboard" / "assets"


@st.cache_data
def load_json(path):
    with open(path) as f:
        return json.load(f)


@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model(ASSETS_DIR / "xgboost_model.json")
    return model


@st.cache_data
def load_demo():
    return pd.read_csv(ASSETS_DIR / "demo_transactions.csv", index_col="test_row")


try:
    results = load_json(EVAL_DIR / "evaluation_results.json")
    dash    = load_json(EVAL_DIR / "dashboard_data.json")
    meta    = load_json(ASSETS_DIR / "model_meta.json")
    model   = load_model()
    demo    = load_demo()
except FileNotFoundError as e:
    st.error(f"Missing pipeline output: `{e.filename}`. Run `python main.py` to regenerate it.")
    st.stop()

FEATURES   = meta["features"]
THRESHOLD  = meta["business_threshold"]
COST_FN    = meta["cost_fn"]
COST_FP    = meta["cost_fp"]
models_list = ['XGBoost', 'Random Forest', 'Logistic Regression']
champion   = results["champion"]
at_biz     = results["at_business_threshold"]
at_f1      = results["xgboost_at_f1_threshold"]
splits     = dash["splits"]
hours      = dash["hourly"]["split_hours"]
WINDOW_HOURS = len(dash["hourly"]["hour"])
TOTAL_ROWS   = sum(s["rows"] for s in splits.values())
TOTAL_FRAUD  = sum(s["fraud"] for s in splits.values())

INK, MUTED, KEY, SOFT, LINE, CARD, ORANGE = "#110d12", "#665c68", "#3b2142", "#d3c7d5", "#cfc6d1", "#f6f3f5", "#ff4f12"

PLOT_CONFIG = dict(
    paper_bgcolor=CARD,
    plot_bgcolor=CARD,
    font=dict(color=MUTED, family='JetBrains Mono, monospace', size=11),
    margin=dict(l=48, r=20, t=40, b=44),
    hoverlabel=dict(font_family='JetBrains Mono, monospace'),
)

# the model or series being discussed is aubergine, comparisons are muted / lilac;
# orange is reserved for the threshold line
COLORS = {
    'XGBoost': KEY,
    'Random Forest': MUTED,
    'Logistic Regression': SOFT,
    'validate': KEY,
    'test': MUTED,
    'threshold': ORANGE,
    'grid': LINE,
}


def axis(**kw):
    return dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=INK, **kw)


def legend(**kw):
    return dict(bgcolor='rgba(0,0,0,0)', font=dict(color=INK), **kw)


def chart(fig):
    with st.container(border=True):
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def card(col, label, value, sub="", solid=False):
    col.markdown(f"""
    <div class='stat{" stat-solid" if solid else ""}'>
        <p class='mono-label'>{label}</p>
        <div class='stat-value'>{value}</div>
        <div class='stat-sub'>{sub}</div>
    </div>""", unsafe_allow_html=True)


def header(kicker, title, lede=""):
    st.markdown(f"<p class='kicker'>{kicker}</p><div class='page-title' role='heading' aria-level='1'>{title}</div>"
                + (f"<p class='lede'>{lede}</p>" if lede else ""), unsafe_allow_html=True)


def eyebrow(text):
    st.markdown(f"<p class='kicker' style='margin-top:.6rem'>{text}</p>", unsafe_allow_html=True)


def caption(text):
    st.markdown(f"<p class='muted'>{text}</p>", unsafe_allow_html=True)


def judgement(*paragraphs):
    body = "".join(f"<p>{p}</p>" for p in paragraphs)
    st.markdown(f"<aside class='judgement'><p class='mono-label'>Judgement</p>{body}</aside>", unsafe_allow_html=True)


def pct(x, digits=1):
    return f"{x * 100:.{digits}f}%"


def prob_text(p):
    if p >= 0.0001:
        return pct(p, 2) if p >= 0.001 else pct(p, 3)
    return "<0.01%"


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
PAGES = ["Score a transaction", "Model performance", "Threshold & cost", "Drift", "Champion rationale"]

with st.sidebar:
    st.markdown("<p class='wordmark'>Fraud Detection</p>"
                "<p class='mono-label'>Interactive dashboard · Korede Folarin</p>", unsafe_allow_html=True)
    st.divider()

    page = st.radio("Navigation", PAGES, label_visibility="collapsed")

    st.divider()
    st.markdown(f"""
    <p class='mono-label'>Dataset</p>
    <div class='facts'>
    ULB credit-card fraud (Kaggle)<br>
    {WINDOW_HOURS} hours · Sept 2013<br>
    {TOTAL_ROWS:,} transactions<br>
    {TOTAL_FRAUD:,} fraud ({TOTAL_FRAUD / TOTAL_ROWS:.3%})<br>
    70/15/15 out-of-time split<br>
    Test = hours {hours['test_start']:.0f}–{WINDOW_HOURS}, never trained on
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown(f"""
    <p class='mono-label'>Links</p>
    <div class='facts'>
    <a href='{SITE_URL}' target='_blank'>Live demo site ↗</a><br>
    <a href='{REPO_URL}' target='_blank'>GitHub ↗</a><br>
    <a href='{MLFLOW_URL}' target='_blank'>MLflow on DagsHub ↗</a><br>
    <a href='https://koredefolarin.com' target='_blank'>koredefolarin.com ↗</a>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE 1 — SCORE A TRANSACTION (live demo)
# ═══════════════════════════════════════════════
EDITABLE = [f for f in dash["importance"]["features"] if f.startswith("V")][:6]
RANGES = {
    f: (float(min(demo[f].min(), meta["feature_ranges"][f][0])),
        float(max(demo[f].max(), meta["feature_ranges"][f][1])))
    for f in EDITABLE
}
MAX_AMOUNT = float(np.ceil(np.expm1(demo["Amount_log"].max())))


def load_row(row_id):
    """Copies a real test transaction into the input widgets"""
    row = demo.loc[row_id]
    st.session_state.row_id = int(row_id)
    for f in EDITABLE:
        st.session_state[f"f_{f}"] = float(row[f])
    st.session_state.f_Amount = round(float(np.expm1(row["Amount_log"])), 2)
    st.session_state.f_Hour = int(row["Hour"])


def pick(kind):
    rng = np.random.default_rng()
    if kind == "legit":
        pool = demo[(demo["Class"] == 0) & (demo["score"] < THRESHOLD)]
    elif kind == "fraud":
        pool = demo[demo["Class"] == 1]
    elif kind == "borderline":
        pool = demo[(demo["score"] > THRESHOLD / 4) & (demo["score"] < THRESHOLD * 4)]
    else:  # false alarm
        pool = demo[(demo["Class"] == 0) & (demo["score"] >= THRESHOLD)]
    load_row(rng.choice(pool.index))


if "row_id" not in st.session_state:
    # open on a real fraud the model catches, so the first view shows a decline notice
    load_row(demo[(demo["Class"] == 1) & (demo["score"] > 0.9)].index[0])

if page == "Score a transaction":

    header("01 · Score a transaction", "One payment, one decision, one reason",
           f"Load a real transaction from the out-of-time test set (hours {hours['test_start']:.0f}–{WINDOW_HOURS}, "
           f"never seen in training) or change its features. The champion XGBoost model scores it live against the "
           f"deployment threshold ({THRESHOLD:.4f}) and explains the decision with SHAP values for this transaction — "
           f"the same code the REST API runs.")

    b1, b2, b3, b4 = st.columns(4)
    b1.button("Random legitimate", on_click=pick, args=("legit",), width="stretch")
    b2.button("Random fraud", on_click=pick, args=("fraud",), width="stretch")
    b3.button("Borderline", on_click=pick, args=("borderline",), width="stretch",
              help="Score within 4× of the threshold either side")
    b4.button("False alarm", on_click=pick, args=("false_alarm",), width="stretch",
              help="A legitimate transaction the model flags")

    original = demo.loc[st.session_state.row_id]
    # verdict column first so phones (where columns stack) show the decision before the inputs
    right, left = st.columns([1.25, 1], gap="large")

    with left:
        eyebrow(f"Inputs · test transaction #{st.session_state.row_id}")

        c1, c2 = st.columns(2)
        c1.number_input("Amount (€)", min_value=0.0, max_value=MAX_AMOUNT, step=1.0, key="f_Amount")
        c2.slider("Hour of day", 0, 23, key="f_Hour")

        st.markdown("<p class='mono-label' style='margin:.4rem 0'>Top behavioural risk indicators · "
                    "anonymised PCA components, ranked by SHAP importance</p>", unsafe_allow_html=True)
        for f in EDITABLE:
            lo, hi = RANGES[f]
            st.slider(f, lo, hi, step=0.01, key=f"f_{f}", format="%.2f")

        st.button("Reset to original values", on_click=load_row, args=(st.session_state.row_id,))
        caption(f"The other {len(FEATURES) - len(EDITABLE) - 2} features keep this transaction's real values.")

    # build the model input from the real row plus any edits
    x = original[FEATURES].astype(float).copy()
    for f in EDITABLE:
        x[f] = st.session_state[f"f_{f}"]
    x["Amount_log"] = np.log1p(st.session_state.f_Amount)
    x["Hour"] = st.session_state.f_Hour
    X = pd.DataFrame([x.values], columns=FEATURES)

    fraud_prob = float(model.predict_proba(X)[0, 1])
    contributions, base_value = shap_contributions(model, X)
    result = decide(fraud_prob, contributions, THRESHOLD)
    edited = not np.allclose(x.values, original[FEATURES].astype(float).values, atol=1e-4)

    with right:
        declined = result["decision"] == "DECLINE"
        pos = min(100.0, fraud_prob * 100)
        actual = "fraud" if original["Class"] == 1 else "legitimate"
        st.markdown(f"""
        <div class='verdict' aria-live='polite'>
            <p class='mono-label'>Test transaction #{st.session_state.row_id} · €{st.session_state.f_Amount:,.2f} · hour of day {st.session_state.f_Hour}{" · edited" if edited else ""}</p>
            <div class='stamp-row'>
                <span class='stamp {"decline" if declined else "approve"}'>{result['decision']}</span>
                <span class='mono-label'>Risk band · {result['risk_level']}</span>
            </div>
            <p class='mono-label'>Fraud probability</p>
            <div class='prob'>{prob_text(fraud_prob)}</div>
            <div class='track'>
                <div class='track-fill' style='width:{pos}%'></div>
                <div class='track-mark' style='left:{THRESHOLD * 100}%'><span class='mono-label'>threshold {THRESHOLD:.4f}</span></div>
                <div class='track-dot' style='left:{pos}%'></div>
            </div>
            <div class='track-scale mono-label'><span>0%</span><span>50%</span><span>100%</span></div>
            <p class='actual'>Actually: <strong>{actual}</strong>{" (before your edits)" if edited else ""}</p>
        </div>""", unsafe_allow_html=True)

        notice = result['adverse_action_notice'] or \
            "No notice. The payment is approved, so the API returns no decline reason."
        st.markdown(f"""
        <div class='notice'>
            <p class='mono-label'>What the customer receives</p>
            <p>{notice}</p>
        </div>""", unsafe_allow_html=True)

    eyebrow("Why — SHAP contributions for this transaction")
    caption("Each bar is how far one feature pushed this score towards fraud (aubergine) or away from it (lilac), "
            "in log-odds. Only aubergine bars can be cited as decline reasons.")

    contrib = pd.Series(contributions)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:10])
    rest = contrib.drop(top.index).sum()

    names  = ["Model baseline"] + [f"{f} — {describe_feature(f)}" if not f.startswith("V") else describe_feature(f)
                                   for f in top.index] \
             + [f"{len(contrib) - len(top)} other features", "Final score"]
    values = [base_value] + top.tolist() + [rest, 0]
    measure = ["absolute"] + ["relative"] * (len(top) + 1) + ["total"]

    wf = go.Figure(go.Waterfall(
        orientation="h",
        y=names, x=values, measure=measure,
        increasing=dict(marker=dict(color=KEY)),
        decreasing=dict(marker=dict(color=SOFT)),
        totals=dict(marker=dict(color=MUTED)),
        connector=dict(line=dict(color=LINE)),
        text=[f"{v:+.2f}" if 0 < i < len(values) - 1 else "" for i, v in enumerate(values)],
        textposition="outside", textfont=dict(color=INK),
    ))
    wf.update_layout(
        **{**PLOT_CONFIG, 'margin': dict(l=48, r=64, t=30, b=44)}, height=460, showlegend=False,
        yaxis=axis(autorange="reversed", tickfont=dict(color=INK)),
        xaxis=axis(title="Log-odds of fraud"),
    )
    chart(wf)

    judgement(
        "The reasons are faithful to the model, but on this dataset most of them are anonymised components, so the "
        "customer learns that a reason exists, not what it is. That is a limit of the data, not the method: with a "
        "bank's named features the same code would name them. I chose honest-but-vague over inventing plausible "
        "meanings, because a notice that misdescribes the model is worse than one that says little.",
        "Try the False alarm button. Those payments are declined with reasons that sound just as confident, and they "
        "are wrong. That is why the notice says the decision is automated and offers a manual review: the "
        "explanation describes the model, it does not prove it right.",
    )

    with st.expander("API response for this transaction (JSON)"):
        st.json(result)


# ═══════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════
elif page == "Model performance":

    header("02 · Model performance", "How well it ranks, and what it catches",
           f"Champion XGBoost on the out-of-time test set (hours {hours['test_start']:.0f}–{WINDOW_HOURS}) "
           f"at the deployment threshold {THRESHOLD:.4f}.")

    xgb_test = results['test']['XGBoost']
    xgb_biz  = at_biz['test']['XGBoost']
    trigger_state = "above" if xgb_test['KS'] >= 0.75 else "below"
    st.markdown(f"""<div class='health'><strong>Retraining trigger:</strong> test KS {xgb_test['KS']:.3f} is
        {trigger_state} the 0.75 line the Airflow <code>drift_check</code> task uses
        (validate {results['validate']['XGBoost']['KS']:.3f}).</div>""", unsafe_allow_html=True)

    c = st.columns(3) + st.columns(3)
    card(c[0], "Fraud caught", pct(xgb_biz['Recall']), f"{xgb_biz['TP']} of {xgb_biz['TP'] + xgb_biz['FN']} frauds", solid=True)
    card(c[1], "Alert precision", pct(xgb_biz['Precision']), f"{xgb_biz['FP']} false alarms")
    card(c[2], "Alerts per 10k", f"{xgb_biz['Flag_rate'] * 1e4:.0f}", "Manual review load")
    card(c[3], "KS statistic", f"{xgb_test['KS']:.3f}", "Ranking separation")
    card(c[4], "AUPRC", f"{xgb_test['AUPRC']:.3f}",
         f"{xgb_test['AUPRC'] / splits['test']['fraud_rate']:.0f}× random baseline")
    card(c[5], "Business cost", f"£{xgb_biz['Cost']:,.0f}",
         f"<span class='chip-orange'>missed fraud £{xgb_biz['FN'] * COST_FN:,}</span>")

    eyebrow("Model comparison — test set")
    rows = []
    for m in models_list:
        t, v, b = results['test'][m], results['validate'][m], at_biz['test'][m]
        rows.append({
            'Model': m + (' · champion' if m == 'XGBoost' else ''),
            'Validate KS': v['KS'], 'Test KS': t['KS'], 'Test AUPRC': t['AUPRC'],
            'Catch rate @ threshold': b['Recall'], 'Precision @ threshold': b['Precision'],
            'Share flagged': b['Flag_rate'], 'Cost @ threshold (£)': b['Cost'],
        })
    st.dataframe(
        pd.DataFrame(rows).set_index('Model'),
        width="stretch",
        column_config={
            'Validate KS': st.column_config.NumberColumn(format="%.3f"),
            'Test KS': st.column_config.NumberColumn(format="%.3f"),
            'Test AUPRC': st.column_config.NumberColumn(format="%.3f"),
            'Catch rate @ threshold': st.column_config.NumberColumn(format="percent"),
            'Precision @ threshold': st.column_config.NumberColumn(format="percent"),
            'Share flagged': st.column_config.NumberColumn(format="percent"),
            'Cost @ threshold (£)': st.column_config.NumberColumn(format="£%,.0f"),
        },
    )

    split = st.radio("Dataset", ["test", "validate"], horizontal=True,
                     format_func=lambda s: f"{s.title()} set")

    col1, col2 = st.columns(2)
    with col1:
        eyebrow("KS curves — fraud vs legitimate captured")
        fig = go.Figure()
        for m in models_list:
            k = dash['ks_curves'][split][m]
            gap = np.array(k['fraud']) - np.array(k['legit'])
            fig.add_trace(go.Scatter(x=k['population'], y=k['fraud'], name=f"{m} — fraud",
                                     line=dict(color=COLORS[m], width=2.5)))
            fig.add_trace(go.Scatter(x=k['population'], y=k['legit'], name=f"{m} — legit",
                                     line=dict(color=COLORS[m], width=1.2, dash='dot'), showlegend=False))
            i = int(gap.argmax())
            fig.add_trace(go.Scatter(x=[k['population'][i]] * 2, y=[k['legit'][i], k['fraud'][i]],
                                     mode='lines', line=dict(color=COLORS[m], width=1, dash='dash'),
                                     showlegend=False, hoverinfo='skip'))
        fig.update_layout(
            **PLOT_CONFIG, height=420,
            xaxis=axis(type='log', title='Share reviewed, highest score first (log)'),
            yaxis=axis(title='Cumulative share captured', range=[0, 1.02]),
            legend=legend(orientation='h', y=-0.28),
        )
        chart(fig)
        caption("Solid = fraud captured, dotted = legitimate captured. KS is the largest vertical gap (dashed).")

    with col2:
        eyebrow("Precision–recall curves")
        fig2 = go.Figure()
        for m in models_list:
            p = dash['pr_curves'][split][m]
            fig2.add_trace(go.Scatter(x=p['recall'], y=p['precision'], name=f"{m} (AUPRC {results[split][m]['AUPRC']:.3f})",
                                      line=dict(color=COLORS[m], width=2.5)))
        base = splits[split]['fraud_rate']
        fig2.add_hline(y=base, line_dash='dot', line_color=INK,
                       annotation_text=f'Random baseline ({base:.3%})', annotation_font_color=MUTED)
        fig2.update_layout(
            **PLOT_CONFIG, height=420,
            xaxis=axis(title='Recall (fraud caught)', range=[0, 1.02]),
            yaxis=axis(title='Precision', range=[0, 1.02]),
            legend=legend(orientation='h', y=-0.28),
        )
        chart(fig2)

    eyebrow(f"Confusion matrices — {split} set at deployment threshold {THRESHOLD:.4f}")
    cols = st.columns(3)
    for col, m in zip(cols, models_list):
        b = at_biz[split][m]
        col.markdown(f"""
        <p class='cm-title'>{m}{' · champion' if m == 'XGBoost' else ''}</p>
        <p class='mono-label' style='margin-bottom:6px'>Precision {b['Precision']:.3f} · recall {b['Recall']:.3f}</p>
        <div class='cm'>
            <div class='h'></div><div class='h mono-label'>Predicted legit</div><div class='h mono-label'>Predicted fraud</div>
            <div class='h mono-label'>Actual legit</div><div><span class='v'>{b['TN']:,}</span></div><div><span class='v'>{b['FP']:,}</span></div>
            <div class='h mono-label'>Actual fraud</div><div><span class='v'>{b['FN']:,}</span></div><div class='tp'><span class='v'>{b['TP']:,}</span></div>
        </div>""", unsafe_allow_html=True)

    n_test = splits['test']['fraud']
    judgement(
        f"Read every test figure with the sample size in mind: there are {n_test} frauds in the test window, so one "
        f"case moves recall by {100 / n_test:.1f} points. Small gaps between models in this table are within that noise.",
        "KS and AUPRC say how well the model ranks. They say nothing about whether a single cut-off produces usable "
        "alerts — that is what the threshold and champion pages are for.",
    )


# ═══════════════════════════════════════════════
# PAGE 3 — THRESHOLD & COST
# ═══════════════════════════════════════════════
elif page == "Threshold & cost":

    header("03 · Threshold & cost", "Where to draw the line",
           "F1 treats a missed fraud and a false alarm as equally bad. They are not, so the deployment threshold "
           "minimises expected cost instead. It is chosen on the validate set only and then frozen for the test set.")

    st.markdown(f"<p class='notice' style='font-family:var(--mono);display:inline-block;border-left-width:1.5px;'>"
                f"cost = missed fraud × £{COST_FN} + false alarms × £{COST_FP}</p>", unsafe_allow_html=True)

    c = st.columns(4)
    card(c[0], "Deployment threshold", f"{THRESHOLD:.4f}", "Cost-optimal on validate", solid=True)
    card(c[1], "F1-optimal threshold", f"{champion['f1_threshold']:.4f}", "Also chosen on validate")
    card(c[2], "Cost per missed fraud", f"£{COST_FN}", "Assumed average loss")
    card(c[3], "Cost per false alarm", f"£{COST_FP}", "Manual review + friction")

    eyebrow("Try a threshold")
    cc = dash['cost_curves']
    grid = cc['validate']['threshold']
    default = min(grid, key=lambda t: abs(t - THRESHOLD))
    t = st.select_slider("Decision threshold — flag when score ≥ threshold", options=grid, value=default,
                         format_func=lambda v: f"{v:.4f}")
    i = grid.index(t)

    for split in ['validate', 'test']:
        d = cc[split]
        frauds = d['tp'][i] + d['fn'][i]
        alerts = d['tp'][i] + d['fp'][i]
        with st.container():
            eyebrow(f"{split.title()} set · {frauds} frauds in {splits[split]['rows']:,} transactions")
            k = st.columns(3)
            card(k[0], "Fraud caught", f"{d['tp'][i]} of {frauds}", f"{pct(d['tp'][i] / frauds)} · {d['fn'][i]} missed",
                 solid=(split == 'validate'))
            card(k[1], "False alarms", f"{d['fp'][i]:,}", f"precision {pct(d['tp'][i] / alerts)}" if alerts else "no alerts")
            card(k[2], "Total cost", f"£{d['cost'][i]:,.0f}",
                 f"<span class='chip-orange'>missed £{d['fn'][i] * COST_FN:,}</span>")

    fig = go.Figure()
    for split in ['validate', 'test']:
        fig.add_trace(go.Scatter(x=cc[split]['threshold'], y=cc[split]['cost'], name=f"{split.title()} set",
                                 line=dict(color=COLORS[split], width=2.5)))
    fig.add_vline(x=THRESHOLD, line_color=ORANGE, line_width=2.5,
                  annotation_text=f'cost-optimal {THRESHOLD:.4f}', annotation_font_color=INK,
                  annotation_position='top left')
    fig.add_vline(x=champion['f1_threshold'], line_dash='dot', line_color=INK, line_width=1.5,
                  annotation_text=f"F1-optimal {champion['f1_threshold']:.4f}", annotation_font_color=INK,
                  annotation_position='top left')
    fig.add_vline(x=t, line_color=INK, line_width=1)
    fig.update_layout(
        **PLOT_CONFIG, height=420,
        xaxis=axis(type='log', title='Threshold (log scale)'),
        yaxis=axis(title='Cost (£)', range=[0, 8000]),
        legend=legend(orientation='h', y=-0.22),
    )
    chart(fig)

    eyebrow("The two thresholds, frozen")
    v_biz, v_f1 = at_biz['validate']['XGBoost'], at_f1['validate']
    t_biz, t_f1 = at_biz['test']['XGBoost'], at_f1['test']
    table = pd.DataFrame([
        {'Threshold': f"Cost-optimal ({THRESHOLD:.4f})", 'Set': 'Validate', 'Caught': v_biz['Recall'], 'Precision': v_biz['Precision'], 'Cost (£)': v_biz['Cost']},
        {'Threshold': f"F1-optimal ({champion['f1_threshold']:.4f})", 'Set': 'Validate', 'Caught': v_f1['Recall'], 'Precision': v_f1['Precision'], 'Cost (£)': v_f1['Cost']},
        {'Threshold': f"Cost-optimal ({THRESHOLD:.4f})", 'Set': 'Test', 'Caught': t_biz['Recall'], 'Precision': t_biz['Precision'], 'Cost (£)': t_biz['Cost']},
        {'Threshold': f"F1-optimal ({champion['f1_threshold']:.4f})", 'Set': 'Test', 'Caught': t_f1['Recall'], 'Precision': t_f1['Precision'], 'Cost (£)': t_f1['Cost']},
    ])
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        'Caught': st.column_config.NumberColumn(format="percent"),
        'Precision': st.column_config.NumberColumn(format="percent"),
        'Cost (£)': st.column_config.NumberColumn(format="£%,.0f"),
    })
    points = (v_biz['Recall'] - v_f1['Recall']) * 100
    saving = 1 - v_biz['Cost'] / v_f1['Cost']
    st.markdown(f"""<div class='honest'><p class='mono-label'>Stated honestly</p><p>
    On validate, the cost-optimal threshold caught {points:.1f} points more fraud and cut cost by {pct(saving, 0)}.
    On the later test window it did not hold up: it caught {t_biz['TP'] - t_f1['TP']} more fraud than the F1 threshold
    for {t_biz['FP'] - t_f1['FP']} more false alarms, and cost £{t_biz['Cost'] - t_f1['Cost']:,.0f} more.
    With only {splits['validate']['fraud']} validate frauds, the cost-optimal point is unstable over time.
    That is the case for recalibrating the threshold on a rolling window and monitoring alert precision alongside KS.
    </p></div>""", unsafe_allow_html=True)

    judgement(
        "Choosing the threshold on the test set would have produced a better-looking number and a meaningless one, "
        "so it was fixed on validate first and the test result is reported as it came out.",
        f"The costs (£{COST_FN} per missed fraud, £{COST_FP} per false alarm) are assumptions, not measured figures, "
        f"and the optimal threshold depends only on their ratio. A bank would agree that ratio with its "
        f"fraud-operations and complaints teams before trusting the optimiser.",
    )


# ═══════════════════════════════════════════════
# PAGE 4 — DRIFT
# ═══════════════════════════════════════════════
elif page == "Drift":

    header("04 · Drift & monitoring", "What a monitor should catch",
           "Out-of-time validation means each split is a later slice of time, so differences between them are real "
           "temporal shift, not sampling noise from a random split.")

    c = st.columns(4)
    card(c[0], "Fraud rate · train", pct(splits['train']['fraud_rate'], 3), f"Hours 0–{hours['validate_start']:.1f}", solid=True)
    card(c[1], "Fraud rate · validate", pct(splits['validate']['fraud_rate'], 3),
         f"Hours {hours['validate_start']:.1f}–{hours['test_start']:.1f}")
    card(c[2], "Fraud rate · test", pct(splits['test']['fraud_rate'], 3), f"Hours {hours['test_start']:.1f}–{WINDOW_HOURS}")
    ks_drop = results['validate']['XGBoost']['KS'] - results['test']['XGBoost']['KS']
    card(c[3], "XGBoost KS drop", f"{ks_drop:.3f}", "Validate → test")

    eyebrow(f"Hourly fraud rate — all {TOTAL_ROWS:,} transactions")
    h = dash['hourly']
    rate = np.array(h['fraud_rate']) * 100
    fig = go.Figure()
    for x0, x1, name, op in [(0, hours['validate_start'], 'Train', 0.0),
                             (hours['validate_start'], hours['test_start'], 'Validate', 0.35),
                             (hours['test_start'], WINDOW_HOURS, 'Test', 0.7)]:
        fig.add_vrect(x0=x0 - 0.5, x1=x1 - 0.5, fillcolor=SOFT, opacity=op, line_width=0,
                      annotation_text=name, annotation_position='top left', annotation_font_color=MUTED)
    fig.add_trace(go.Bar(x=h['hour'], y=rate, marker_color=KEY,
                         customdata=np.c_[h['fraud'], h['count']],
                         hovertemplate='Hour %{x}<br>%{y:.2f}% fraud<br>%{customdata[0]} of %{customdata[1]:,}<extra></extra>'))
    fig.update_layout(
        **PLOT_CONFIG, height=380, showlegend=False,
        xaxis=axis(title='Hours since start of window', dtick=4),
        yaxis=axis(title='Fraud rate (%)'),
    )
    chart(fig)
    r2, r26 = h['fraud_rate'][2], h['fraud_rate'][26]
    caption(f"Fraud clusters in short bursts during the low-volume hours of each day (hours 2–7 and 26–28). "
            f"Hour 26 ({pct(r26, 2)}) is {r26 / r2 - 1:.0%} higher than the same clock hour on day one "
            f"(hour 2, {pct(r2, 2)}).")

    col1, col2 = st.columns(2)
    with col1:
        eyebrow("Population Stability Index — train vs test")
        psi = pd.Series(dash['psi']).sort_values()
        labels = [f"{f}  {v:.3f}" for f, v in psi.items()]
        fig2 = go.Figure(go.Bar(x=psi.values, y=labels, orientation='h', marker_color=KEY,
                                hovertemplate='%{y}<extra></extra>'))
        flagged = [(lab, v) for lab, v in zip(labels, psi.values) if v > 0.25]
        fig2.add_trace(go.Scatter(x=[v + psi.max() * 0.05 for _, v in flagged], y=[lab for lab, _ in flagged], mode='markers',
                                  marker=dict(symbol='circle-open', size=16, color=ORANGE, line=dict(width=2.5)),
                                  hovertemplate='%{y}: significant shift<extra></extra>'))
        for x, label, pos in [(0.10, 'moderate', 'top left'), (0.25, 'significant', 'top right')]:
            fig2.add_vline(x=x, line_dash='dash', line_color=INK, line_width=1.2, annotation_position=pos,
                           annotation_text=label, annotation_font_color=MUTED)
        fig2.update_layout(**PLOT_CONFIG, height=380, showlegend=False,
                           xaxis=axis(title='PSI', range=[0, max(psi.max() * 1.25, 0.3)]),
                           yaxis=axis(tickfont=dict(color=INK)))
        chart(fig2)
        caption("PSI for the eight most influential features. Open orange rings mark a significant shift (PSI > 0.25).")

    with col2:
        eyebrow("KS — validate → test")
        fig3 = go.Figure()
        for m in models_list:
            v, t = results['validate'][m]['KS'], results['test'][m]['KS']
            fig3.add_trace(go.Scatter(
                x=['Validate', 'Test'], y=[v, t], mode='lines+markers+text', name=m,
                line=dict(color=COLORS[m], width=3), marker=dict(size=10),
                text=['', f'↓{v - t:.3f}'], textposition='middle right', textfont=dict(color=INK),
            ))
        fig3.add_hline(y=0.75, line_color=ORANGE, line_width=2, annotation_position='top left',
                       annotation_text='retraining trigger (KS < 0.75)', annotation_font_color=INK)
        fig3.update_layout(**PLOT_CONFIG, height=380, yaxis=axis(range=[0.7, 1.0], title='KS'),
                           xaxis=axis(range=[-0.3, 1.7]), legend=legend(orientation='h', y=-0.15))
        chart(fig3)

    top_psi = max(dash['psi'], key=dash['psi'].get)
    judgement(
        f"The KS trigger still reads healthy ({results['test']['XGBoost']['KS']:.3f} against 0.75), yet {top_psi} has a "
        f"PSI of {dash['psi'][top_psi]:.2f}: the inputs moved a lot while the ranking held up. The threshold page shows "
        f"the operating point had already broken. A KS-only trigger would miss both, so a production monitor should "
        f"also watch PSI and alert precision.",
        f"The weekly Airflow retraining is demonstrated, not exercised: on a fixed {WINDOW_HOURS}-hour dataset every "
        f"run sees the same data.",
    )


# ═══════════════════════════════════════════════
# PAGE 5 — CHAMPION RATIONALE
# ═══════════════════════════════════════════════
elif page == "Champion rationale":

    header("05 · Champion", "The leaderboard is not the decision",
           "Why XGBoost was chosen over Random Forest, even though Random Forest scores higher on test KS and AUPRC.")

    xgb_b, rf_b, lr_b = (at_biz['test'][m] for m in models_list)
    st.markdown(f"""<div class='verdict'><p class='mono-label'>Champion · XGBoost</p>
        <p style='margin:.5rem 0 0;font-size:17px;line-height:1.55;max-width:90ch;color:var(--on-key)'>
        KS and AUPRC measure <em>ranking</em>. A bank deploys a <em>threshold</em>. At the same deployment threshold
        ({THRESHOLD:.4f}) on the test set, {pct(xgb_b['Precision'])} of XGBoost's alerts are fraud, against
        {pct(rf_b['Precision'])} for Random Forest; Logistic Regression flags {pct(lr_b['Flag_rate'])} of all
        transactions. Only XGBoost's probabilities are calibrated closely enough to the true fraud rate for a single
        threshold to mean something.</p></div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        eyebrow("Calibration — mean predicted score vs actual fraud rate (validate)")
        actual = results['validate']['XGBoost']['Fraud_rate']
        means = [results['validate'][m]['Mean_score'] for m in models_list]
        fig = go.Figure(go.Bar(x=models_list, y=[v * 100 for v in means], marker_color=[COLORS[m] for m in models_list],
                               text=[f"{v:.2%}" for v in means], textposition='outside', textfont=dict(color=INK)))
        fig.add_hline(y=actual * 100, line_dash='dash', line_color=INK, annotation_position='bottom left',
                      annotation_text=f'actual fraud rate {actual:.3%}', annotation_font_color=INK)
        fig.update_layout(**PLOT_CONFIG, height=380, showlegend=False,
                          xaxis=axis(tickfont=dict(color=INK)),
                          yaxis=axis(type='log', title='Mean score (%, log scale)', range=[-1.3, 1.4]))
        chart(fig)

    with col2:
        eyebrow(f"At the shared threshold {THRESHOLD:.4f} — test set")
        fig2 = make_subplots(rows=2, cols=1, vertical_spacing=0.28,
                             subplot_titles=['Alert precision', 'Share of all transactions flagged'])
        for row, key, digits in [(1, 'Precision', 1), (2, 'Flag_rate', 2)]:
            vals = [at_biz['test'][m][key] * 100 for m in models_list]
            fig2.add_trace(go.Bar(x=vals, y=models_list, orientation='h', marker_color=[COLORS[m] for m in models_list],
                                  text=[f"{v:.{digits}f}%" for v in vals], textposition='outside',
                                  textfont=dict(color=INK), showlegend=False), row=row, col=1)
            fig2.update_yaxes(autorange='reversed', tickfont=dict(color=INK), row=row, col=1)
            fig2.update_xaxes(range=[0, max(vals) * 1.3], gridcolor=LINE, row=row, col=1)
        fig2.update_layout(**PLOT_CONFIG, height=380)
        fig2.update_annotations(font=dict(color=INK, family='JetBrains Mono, monospace', size=11))
        chart(fig2)

    eyebrow("Feature importance — three independent views")
    imp = dash['importance']
    fig_feat = make_subplots(rows=1, cols=3, subplot_titles=[
        'Mean |SHAP| on test (log-odds)', 'Information Value (train)', 'XGBoost gain (normalised)'])
    for col_idx, (key, color) in enumerate([('shap', KEY), ('iv', MUTED), ('gain', SOFT)], 1):
        fig_feat.add_trace(go.Bar(x=imp[key][::-1], y=imp['features'][::-1], orientation='h',
                                  marker_color=color, showlegend=False), row=1, col=col_idx)
    fig_feat.update_layout(**PLOT_CONFIG, height=460)
    fig_feat.update_annotations(font=dict(color=INK, family='JetBrains Mono, monospace', size=11))
    for i in range(1, 4):
        fig_feat.update_xaxes(gridcolor=LINE, row=1, col=i)
        fig_feat.update_yaxes(gridcolor=LINE, tickfont=dict(color=INK), row=1, col=i)
    chart(fig_feat)

    top_psi = max(dash['psi'], key=dash['psi'].get)
    caption(f"V4, V14, V12 and V10 rank in the top five on all three views. IV is calculated on the training set only. "
            f"Of the top features, {top_psi} shows the largest train→test shift (PSI {dash['psi'][top_psi]:.2f}).")

    n_test = splits['test']['fraud']
    judgement(
        f"On {n_test} test frauds, one case moves recall by {100 / n_test:.1f} points, so the KS and AUPRC gaps "
        f"between the models are small relative to sampling noise. The precision gap at the threshold "
        f"({pct(xgb_b['Precision'])} vs {pct(rf_b['Precision'])}) is not. That is the difference the decision rests on.",
        "The fair challenge is that Random Forest's scores could be recalibrated, for example with isotonic regression, "
        "and might then compete. That was not tested. It is the first experiment I would run on the challenger.",
        "Hyperparameters were set from domain knowledge, not tuned: standard cross-validation breaks the time "
        f"ordering, and tuning on a {WINDOW_HOURS}-hour window risks fitting its particular fraud bursts.",
    )
