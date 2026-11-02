"""
Streaming sink + parity check (D14) — runs in the tools container, started
BEFORE the producer:  python parity.py --name run-1000 --expect 42821

Plain English: listens to everything the Flink job writes, times how long
each payment took from being sent to being scored, and checks every scored
test payment against v1's batch answer: same three features, same score, same
decision. Payments the stream could not score (far too late) are counted
separately, as are the payments whose 100-payment window was missing them.

Technical:
  latency     consumer receive time - producer send time (same Docker VM
              clock), per scored event; reported as p50/p95/p99/max
  throughput  unique scored events / (last receive - first send)
  parity      join on seq with reference.parquet (v1 batch features from the
              frozen bundle + v1 pickle scores). Tolerances, set before the run:
                Amount_log, rolling_std_amount: |diff| <= 1e-9 (abs) or 1e-9 relative
                Hour: exact;  score: |diff| <= 1e-6;  decisions: zero flips
  exclusions  only rows whose window lost a too-late payment (seq s+1..s+99
              for each too-late s) — reported, not hidden
  dedupe      the sink is at-least-once: repeated outputs for one seq must be
              identical, and are counted

What the code does: writes results/<name>.json and prints a summary.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from confluent_kafka import Consumer

TOL_FEATURE, TOL_SCORE = 1e-9, 1e-6


def pct(values, q):
    return round(float(np.percentile(values, q)), 1) if len(values) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default="kafka:9092")
    ap.add_argument("--name", required=True)
    ap.add_argument("--expect", type=int, required=True, help="unique events the producer will send")
    ap.add_argument("--idle-seconds", type=float, default=30)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--faults", default="/work/results/last_faults.json")
    ap.add_argument("--reference", default="/work/data/reference.parquet")
    ap.add_argument("--out", default="/work/results")
    args = ap.parse_args()

    consumer = Consumer({"bootstrap.servers": args.bootstrap, "group.id": f"parity-{args.name}-{time.time()}",
                         "auto.offset.reset": "earliest", "enable.auto.commit": False})
    consumer.subscribe(["scores", "late-events", "duplicate-events"])

    scores, outputs_per_seq, late, dup_in = {}, {}, [], 0
    recv_ms, conflicts = {}, 0
    start, last_msg = time.time(), time.time()
    while time.time() - start < args.timeout:
        msg = consumer.poll(1.0)
        now_ms = int(time.time() * 1000)
        if msg is None:
            got = len(scores) + len(late)
            if got and time.time() - last_msg > args.idle_seconds:
                break
            continue
        if msg.error():
            continue
        last_msg = time.time()
        value = json.loads(msg.value())
        if msg.topic() == "late-events":
            late.append(int(value["seq"]))
        elif msg.topic() == "duplicate-events":
            dup_in += 1
        else:
            seq = int(value["seq"])
            outputs_per_seq[seq] = outputs_per_seq.get(seq, 0) + 1
            if seq in scores:
                a, b = scores[seq], value
                if any(a[k] != b[k] for k in ("score", "decision", "Amount_log", "Hour", "rolling_std_amount")):
                    conflicts += 1
            else:
                scores[seq] = value
                recv_ms[seq] = now_ms - int(value["produce_ts_ms"])
    consumer.close()

    if not scores:
        result = {"name": args.name, "parity_passed": False, "error": "no scored events received (job failed?)",
                  "late_events": len(late), "duplicate_inputs_dropped": dup_in}
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / f"{args.name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return

    faults = json.loads(Path(args.faults).read_text())
    ref = pd.read_parquet(args.reference).set_index("seq")
    got = pd.DataFrame.from_dict(scores, orient="index")
    got.index.name = "seq"

    too_late = set(faults["too_late"])
    affected = set()
    for s in too_late:
        affected.update(range(s + 1, s + 100))
    test_seqs = ref.index
    scored_test = got.index.intersection(test_seqs)
    strict = scored_test.difference(pd.Index(sorted(affected)))

    def compare(idx):
        if len(idx) == 0:
            return {}
        r, g = ref.loc[idx], got.loc[idx]
        out = {}
        for col in ("Amount_log", "rolling_std_amount"):
            d = (g[col].astype(float) - r[col]).abs()
            rel = d / r[col].abs().clip(lower=1e-12)
            out[f"{col}_max_abs_diff"] = float(d.max())
            out[f"{col}_rows_outside_tolerance"] = int(((d > TOL_FEATURE) & (rel > TOL_FEATURE)).sum())
        out["Hour_mismatches"] = int((g["Hour"].astype(int) != r["Hour"]).sum())
        sd = (g["score"].astype(float) - r["v1_score"]).abs()
        out["score_max_abs_diff"] = float(sd.max())
        out["score_rows_outside_tolerance"] = int((sd > TOL_SCORE).sum())
        out["decision_flips"] = int((g["decision"] != r["v1_decision"]).sum())
        out["declines"] = int((g["decision"] == "DECLINE").sum())
        return out

    strict_result = compare(strict)
    affected_result = compare(scored_test.intersection(pd.Index(sorted(affected))))
    missing_test = test_seqs.difference(got.index).difference(pd.Index(sorted(too_late)))

    lat = np.array([recv_ms[s] for s in scores], dtype=float)
    first_send = min(int(v["produce_ts_ms"]) for v in scores.values())
    last_recv = max(int(v["produce_ts_ms"]) + recv_ms[s] for s, v in scores.items())
    batch = np.array([v["batch_size"] for v in scores.values()])

    passed = (strict_result.get("Amount_log_rows_outside_tolerance") == 0
              and strict_result.get("rolling_std_amount_rows_outside_tolerance") == 0
              and strict_result.get("Hour_mismatches") == 0
              and strict_result.get("score_rows_outside_tolerance") == 0
              and strict_result.get("decision_flips") == 0
              and len(missing_test) == 0
              and set(late) == too_late
              and conflicts == 0)

    result = {
        "name": args.name,
        "parity_passed": bool(passed),
        "tolerances": {"features_abs_or_rel": TOL_FEATURE, "score_abs": TOL_SCORE, "decision_flips": 0},
        "producer": {k: faults[k] for k in ("rate_target", "achieved_send_rate", "events_unique",
                                            "messages_sent", "send_seconds")},
        "faults_injected": {"duplicates": len(faults["duplicates"]),
                            "late_within_bound": len(faults["late_within_bound"]),
                            "too_late": len(too_late)},
        "handled": {"duplicate_inputs_dropped": dup_in, "late_events_side_output": len(late),
                    "too_late_all_routed_to_side_output": set(late) == too_late,
                    "scored_unique": len(scores), "sink_repeats": int(sum(c - 1 for c in outputs_per_seq.values())),
                    "repeat_conflicts": conflicts},
        "test_rows": {"total": len(test_seqs), "scored": len(scored_test), "missing_unexpected": len(missing_test),
                      "strict_parity_rows": len(strict), "excluded_window_lost_too_late_event": len(affected)},
        "parity_strict": strict_result,
        "rows_after_too_late_events": affected_result,
        "end_to_end_latency_ms": {"p50": pct(lat, 50), "p95": pct(lat, 95), "p99": pct(lat, 99),
                                  "max": float(lat.max()) if len(lat) else None},
        "throughput_events_per_s": round(len(scores) / max((last_recv - first_send) / 1000, 1e-9), 1),
        "scoring_batch_size": {"median": float(np.median(batch)), "p95": float(np.percentile(batch, 95)),
                               "max": int(batch.max())},
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"{args.name}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
