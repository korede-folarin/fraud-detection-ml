"""
Smoke test a running fraud API (container or local)
Run: python v2/ci/smoke_test_api.py [base_url] [--v2]     default http://localhost:3000

Plain English: sends the four real demo transactions to the API and checks it
returns the same decision and the word-for-word same customer notice as the
published demo, plus a clean error for a malformed request. With --v2 it also
requires HTTP 400 (not 500) for bad input and checks the service metrics
(D13) are exported at /metrics.

Technical: reads site/data/samples.json (built with the same inference code)
and dashboard/assets/demo_transactions.csv; posts to /predict; compares
decision, notice and probability (within 5e-5, the site's rounding).

What the code does: retries until the API is ready (up to ~2 minutes), then
exits 0 if every check passes, 1 otherwise.
"""
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
V2 = "--v2" in sys.argv
BASE = (ARGS[0] if ARGS else "http://127.0.0.1:3000").rstrip("/")


def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def wait_ready():
    for _ in range(60):
        try:
            with urllib.request.urlopen(BASE + "/readyz", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    sys.exit(f"API at {BASE} never became ready")


def main():
    wait_ready()
    meta = json.loads((ROOT / "dashboard" / "assets" / "model_meta.json").read_text())
    samples = json.loads((ROOT / "site" / "data" / "samples.json").read_text(encoding="utf-8"))
    with open(ROOT / "dashboard" / "assets" / "demo_transactions.csv", newline="") as f:
        rows = {int(r["test_row"]): r for r in csv.DictReader(f)}

    failures = []
    for kind, s in samples.items():
        features = [float(rows[s["test_row"]][c]) for c in meta["features"]]
        status, r = post("/predict", {"transaction": {"features": features}})
        if status != 200:
            failures.append(f"{kind}: HTTP {status} {r}")
            continue
        ok = (r["decision"] == s["decision"]
              and r["adverse_action_notice"] == s["notice"]
              and abs(r["fraud_probability"] - s["fraud_probability"]) <= 5e-5)
        print(f"{'ok  ' if ok else 'FAIL'} {kind:12s} {r['decision']:8s} p={r['fraud_probability']}")
        if not ok:
            failures.append(f"{kind}: got {r['decision']} / {r['adverse_action_notice']!r}")

    status, _ = post("/predict", {"transaction": {"features": [0.0] * 3}})
    rejected_ok = status == 400 if V2 else status >= 400
    print(f"{'ok  ' if rejected_ok else 'FAIL'} malformed request rejected with HTTP {status}")
    if not rejected_ok:
        failures.append(f"malformed request returned HTTP {status}")

    if V2:
        with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
            text = r.read().decode()
        for name in ("fraud_api_predictions_total", "fraud_api_fraud_probability_bucket",
                     "fraud_api_inference_seconds_bucket", "fraud_api_rejected_requests_total",
                     "fraud_api_model_info", "bentoml_service_request_duration_seconds_bucket"):
            present = name in text
            print(f"{'ok  ' if present else 'FAIL'} /metrics exports {name}")
            if not present:
                failures.append(f"/metrics is missing {name}")

    if failures:
        print("\n".join(["SMOKE TEST FAILED:"] + failures))
        sys.exit(1)
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
