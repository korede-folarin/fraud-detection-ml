"""
Replay real transactions against the API at a steady rate and measure it
Run: python v2/monitoring/load.py --rps 50 --seconds 120 [--bad-share 0.01]

Plain English: plays the out-of-time test transactions (the final six hours of
v1's data) into the running API at a fixed pace, like live traffic, and records
how long each answer took from the caller's side. A small share of deliberately
broken requests can be mixed in so the error panels have something to show.

Technical: open-loop load — requests are scheduled on a fixed clock and sent by
a thread pool, so a slow API cannot slow the arrival rate down (closed-loop
tools hide latency that way). Client-side latency includes HTTP and JSON; the
Grafana panels show the server-side view. Rows come from
v1_frozen/data/test.parquet in time order.

What the code does: prints achieved rate, latency p50/p95/p99, status counts
and decision mix, and writes the same as JSON to v2/monitoring/results/.
"""
import argparse
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "v2" / "monitoring" / "results"


def percentile(values, q):
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, round(q / 100 * (len(s) - 1)))]


def main():
    ap = argparse.ArgumentParser()
    # 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 first and adds ~2 s per
    # request when the container only listens on IPv4 (found while measuring, D13)
    ap.add_argument("--url", default="http://127.0.0.1:3000")
    ap.add_argument("--rps", type=float, default=50)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--bad-share", type=float, default=0.0, help="share of malformed requests, e.g. 0.01")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    df = pd.read_parquet(ROOT / "v1_frozen" / "data" / "test.parquet")
    features = df.drop(columns="Class").to_numpy().tolist()
    total = int(args.rps * args.seconds)
    every_bad = int(1 / args.bad_share) if args.bad_share > 0 else 0

    latencies, statuses, decisions = [], {}, {}
    lock = threading.Lock()

    def send(i):
        bad = every_bad and i % every_bad == every_bad - 1
        body = {"transaction": {"features": [0.0] * 3 if bad else features[i % len(features)]}}
        req = urllib.request.Request(args.url.rstrip("/") + "/predict", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                status, payload = r.status, json.load(r)
        except urllib.error.HTTPError as e:
            status, payload = e.code, None
        except Exception:
            status, payload = "client_error", None
        elapsed = time.perf_counter() - t0
        with lock:
            statuses[status] = statuses.get(status, 0) + 1
            if status == 200:
                latencies.append(elapsed)
                decisions[payload["decision"]] = decisions.get(payload["decision"], 0) + 1

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i in range(total):
            delay = start + i / args.rps - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            pool.submit(send, i)
    wall = time.perf_counter() - start

    ms = [x * 1000 for x in latencies]
    report = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "target_rps": args.rps, "requests": total, "wall_seconds": round(wall, 2),
        "achieved_rps": round(total / wall, 1),
        "latency_ms_client": {
            "p50": round(percentile(ms, 50), 2) if ms else None,
            "p95": round(percentile(ms, 95), 2) if ms else None,
            "p99": round(percentile(ms, 99), 2) if ms else None,
            "mean": round(statistics.fmean(ms), 2) if ms else None,
        },
        "status_counts": {str(k): v for k, v in sorted(statuses.items(), key=str)},
        "decisions": decisions,
        "data": "v1_frozen/data/test.parquet (out-of-time test set, time order)",
    }
    print(json.dumps(report, indent=2))
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"load-{report['run_at'][:19].replace(':', '')}-{int(args.rps)}rps.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"written {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
