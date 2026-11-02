"""
Replay producer for the v2 streaming test (D14) — runs in the tools container
python producer.py --rate 1000 [--dup-share 0.01 --late-share 0.005 --too-late 20]

Plain English: plays the real raw transactions into Kafka at a chosen rate,
like a card network would, and deliberately misbehaves the way real feeds do:
some payments are sent twice, some arrive a little late, and a few arrive far
too late. Every injected fault is written down so the results can be checked
against it.

Technical: one partition, so arrival order is exactly the send order.
event_time_ms = 2013-09-01T00:00Z + Time (the dataset's seconds-since-start);
produce_ts_ms = wall clock at send (for end-to-end latency). Faults:
  duplicate   same payload re-sent 1-20 positions later (dedupe by seq)
  late        held back by 1-40 s of EVENT time — inside the job's 60 s bound,
              so it must be re-ordered and scored exactly like batch
  too late    held back by 600 s of event time — beyond the bound, so it must
              go to 'late-events' and is NOT scored (the parity check then
              reports the rows whose 100-payment window lost it)
A final control event with event time +1 day flushes the job's last window.
Seeded RNG, so the same flags give the same faults.

What the code does: recreates the four topics (1 partition each), sends the
events at the target rate, and writes the fault manifest next to the results.
"""
import argparse
import heapq
import json
import random
import time
from pathlib import Path

import pandas as pd
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic

BASE_EPOCH_MS = 1377993600000  # 2013-09-01T00:00:00Z
TOPICS = ["transactions", "scores", "late-events", "duplicate-events"]


def recreate_topics(bootstrap):
    admin = AdminClient({"bootstrap.servers": bootstrap})
    existing = set(admin.list_topics(timeout=10).topics)
    to_delete = [t for t in TOPICS if t in existing]
    if to_delete:
        for f in admin.delete_topics(to_delete).values():
            f.result()
        time.sleep(3)
    for f in admin.create_topics([NewTopic(t, num_partitions=1, replication_factor=1) for t in TOPICS]).values():
        f.result()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default="kafka:9092")
    ap.add_argument("--data", default="/work/data/replay.parquet")
    ap.add_argument("--rate", type=float, default=1000)
    ap.add_argument("--dup-share", type=float, default=0.0)
    ap.add_argument("--late-share", type=float, default=0.0)
    ap.add_argument("--too-late", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--manifest", default="/work/results/last_faults.json")
    ap.add_argument("--no-recreate", action="store_true")
    ap.add_argument("--topics-only", action="store_true", help="recreate the topics and exit")
    args = ap.parse_args()

    if args.topics_only:
        recreate_topics(args.bootstrap)
        print("topics recreated:", ", ".join(TOPICS))
        return

    df = pd.read_parquet(args.data)
    events = []
    for row in df.itertuples(index=False):
        e = row._asdict()
        e["seq"] = int(e["seq"])
        e["event_time_ms"] = BASE_EPOCH_MS + int(round(e["Time"] * 1000))
        events.append(e)

    rng = random.Random(args.seed)
    n = len(events)
    test_start = events[0]["seq"] + 99
    candidates = [i for i in range(n) if events[i]["seq"] >= test_start + 200 and i < n - 2000]
    too_late = set(rng.sample(candidates, args.too_late)) if args.too_late else set()
    late = {i for i in candidates if i not in too_late and rng.random() < args.late_share}
    dups = {i for i in range(n) if rng.random() < args.dup_share}

    # build the send order: delayed events are re-inserted by event time
    send, held = [], []   # held: heap of (release_event_time_ms, seq, idx)
    late_delay = {}
    for i, e in enumerate(events):
        while held and held[0][0] <= e["event_time_ms"]:
            send.append(("late", heapq.heappop(held)[2]))
        if i in too_late:
            heapq.heappush(held, (e["event_time_ms"] + 600_000, e["seq"], i))
            late_delay[e["seq"]] = 600.0
        elif i in late:
            delay = rng.uniform(1.0, 40.0)
            heapq.heappush(held, (e["event_time_ms"] + int(delay * 1000), e["seq"], i))
            late_delay[e["seq"]] = round(delay, 2)
        else:
            send.append(("on_time", i))
        if i in dups:
            send.append(("dup_pending", i))
    while held:
        send.append(("late", heapq.heappop(held)[2]))

    # place each duplicate 1-20 positions after its first copy
    final, pending = [], []
    for kind, i in send:
        if kind == "dup_pending":
            pending.append([rng.randint(1, 20), i])
            continue
        final.append((kind, i))
        for p in pending:
            p[0] -= 1
        while pending and min(p[0] for p in pending) <= 0:
            p = min(pending, key=lambda x: x[0])
            pending.remove(p)
            final.append(("duplicate", p[1]))
    final += [("duplicate", p[1]) for p in pending]

    if not args.no_recreate:
        recreate_topics(args.bootstrap)
    producer = Producer({"bootstrap.servers": args.bootstrap, "linger.ms": 5, "acks": "all",
                         "enable.idempotence": True})

    start = time.perf_counter()
    for k, (_, i) in enumerate(final):
        delay = start + k / args.rate - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        payload = {**events[i], "produce_ts_ms": int(time.time() * 1000)}
        producer.produce("transactions", json.dumps(payload).encode())
        if k % 1000 == 0:
            producer.poll(0)
    send_seconds = time.perf_counter() - start
    end_marker = {"control": "end", "seq": -1, "event_time_ms": events[-1]["event_time_ms"] + 86_400_000,
                  "produce_ts_ms": int(time.time() * 1000)}
    producer.produce("transactions", json.dumps(end_marker).encode())
    producer.flush(60)

    manifest = {
        "rate_target": args.rate, "events_unique": n, "messages_sent": len(final),
        "send_seconds": round(send_seconds, 2), "achieved_send_rate": round(len(final) / send_seconds, 1),
        "seed": args.seed, "bound_note": "job watermark bound is 60 s of event time",
        "duplicates": sorted(events[i]["seq"] for i in dups),
        "late_within_bound": {str(events[i]["seq"]): late_delay[events[i]["seq"]] for i in sorted(late)},
        "too_late": sorted(events[i]["seq"] for i in too_late),
        "test_start_seq": test_start,
    }
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in manifest.items()}, indent=2))


if __name__ == "__main__":
    main()
