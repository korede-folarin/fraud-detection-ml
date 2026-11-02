"""
PyFlink streaming scorer, v2 (D14)
Submitted inside the Flink container — see v2/streaming/run_stream.sh

Plain English: reads card payments from Kafka as they arrive, puts them back
in the right order if some arrive a little late, drops repeats, rebuilds v1's
features (amount, hour, recent spending volatility), scores them with the v1
champion in small batches, and writes each decision to Kafka. Payments that
turn up too late to fit are not silently mixed in: they go to their own topic.

Technical:
  source      KafkaSource 'transactions' (1 partition), JSON events
  operator    one KeyedProcessFunction on a constant key (v1's rolling window is
              global, not per customer, so this step is parallelism 1)
  event time  bounded out-of-orderness, tracked INSIDE the operator:
                progress = max(event_time seen) - bound   (--bound-seconds, 60)
              in the order the Python worker actually processes events.
              Flink's own watermark is NOT used for lateness: under PyFlink the
              Java-side watermark can run ahead of events still queued for the
              Python worker (observed: 41,821 of 42,821 events wrongly marked
              late in the first run — see docs/decisions.md D14).
  duplicates  seq is a dense per-feed sequence number: a repeat is either still
              in the buffer or at/below the last released seq -> 'duplicate-events'
  late        an event whose place was skipped because it had not arrived when
              progress passed it (more than the bound late) -> 'late-events',
              not scored; the rows after it lose it from their window
  ordering    raw events in MapState by seq, a min-heap of (event_time, seq)
              decides release; released in seq order once progress passes
              them; features computed at release
  batching    released rows are scored together: flush at --batch-max rows or
              after --flush-ms (processing-time timer); SHAP + customer notice
              only for declines
  state       buffer, window, progress, pending batch and gap sets are keyed
              state, checkpointed every --checkpoint-ms (hashmap backend,
              filesystem checkpoints on a Docker volume)
  sink        KafkaSink 'scores', at-least-once (after a restart some outputs
              repeat; the parity check dedupes by seq and verifies they agree)
  end marker  {"control": "end"} moves progress to +infinity and flushes

What the code does: builds and executes the Flink job graph.
"""
import argparse
import heapq
import json
import sys
import time

sys.path.insert(0, "/opt/job")
sys.path.insert(0, "/opt/src")

from pyflink.common import Configuration, Types, WatermarkStrategy  # noqa: E402
from pyflink.common.serialization import SimpleStringSchema  # noqa: E402
from pyflink.datastream import CheckpointingMode, OutputTag, StreamExecutionEnvironment  # noqa: E402
from pyflink.datastream.connectors.base import DeliveryGuarantee  # noqa: E402
from pyflink.datastream.connectors.kafka import (  # noqa: E402
    KafkaOffsetsInitializer,
    KafkaRecordSerializationSchema,
    KafkaSink,
    KafkaSource,
)
from pyflink.datastream.functions import KeyedProcessFunction, RuntimeContext  # noqa: E402
from pyflink.datastream.state import ListStateDescriptor, MapStateDescriptor, ValueStateDescriptor  # noqa: E402

LATE = OutputTag("late-events", Types.STRING())
DUPLICATE = OutputTag("duplicate-events", Types.STRING())
INFINITY = 2**62


class ScoreInOrder(KeyedProcessFunction):
    def __init__(self, model_path, meta_path, bound_ms, batch_max, flush_ms):
        self.model_path, self.meta_path = model_path, meta_path
        self.bound_ms, self.batch_max, self.flush_ms = bound_ms, batch_max, flush_ms

    def open(self, runtime_context: RuntimeContext):
        import xgboost as xgb

        meta = json.load(open(self.meta_path))
        self.features = meta["features"]
        self.threshold = float(meta["business_threshold"])
        # plain Booster: same probabilities as XGBClassifier.predict_proba[:, 1] without
        # needing scikit-learn in the Flink image (checked by parity.py)
        self.model = xgb.Booster()
        self.model.load_model(self.model_path)
        self.xgb = xgb

        # state is kept small per event: raw JSON in a map, only (time, seq) pairs in
        # the ordering heap — re-serialising whole events on every arrival capped the
        # first version at ~630 events/s (D14)
        get = runtime_context
        self.events = get.get_map_state(MapStateDescriptor("events", Types.LONG(), Types.STRING()))
        # seq is a dense per-feed sequence number, so "already seen" needs no per-event
        # lookup: anything <= the last released seq is either a repeat or a skipped gap
        self.max_released = get.get_state(ValueStateDescriptor("max_released", Types.LONG()))
        self.skipped = get.get_map_state(MapStateDescriptor("skipped", Types.LONG(), Types.BOOLEAN()))
        self.late_seen = get.get_map_state(MapStateDescriptor("late_seen", Types.LONG(), Types.BOOLEAN()))
        self.order = get.get_state(ValueStateDescriptor("order", Types.PICKLED_BYTE_ARRAY()))
        self.window = get.get_state(ValueStateDescriptor("window", Types.PICKLED_BYTE_ARRAY()))
        self.progress = get.get_state(ValueStateDescriptor("progress", Types.LONG()))
        self.pending = get.get_list_state(ListStateDescriptor("pending", Types.STRING()))
        self.pending_n = get.get_state(ValueStateDescriptor("pending_n", Types.INT()))
        self.timer_set = get.get_state(ValueStateDescriptor("timer_set", Types.BOOLEAN()))

    # ── arrival ──────────────────────────────────────────────
    def process_element(self, value, ctx: KeyedProcessFunction.Context):
        event = json.loads(value)
        if event.get("control") == "end":
            self.progress.update(INFINITY)
            self._release()
            yield from self._flush()
            return

        seq, event_time = int(event["seq"]), int(event["event_time_ms"])
        heap = self.order.value() or []
        max_released = self.max_released.value()

        if max_released is not None and seq <= max_released:
            # its place in the sequence has passed: a repeat, or the first copy of
            # an event that arrived after the job had to skip it (too late)
            if self.skipped.contains(seq):
                self.skipped.remove(seq)
                self.late_seen.put(seq, True)
                yield LATE, value
            else:
                yield DUPLICATE, value
            return
        if any(s == seq for _, s in heap):
            yield DUPLICATE, value          # repeat of an event still waiting in the buffer
            return

        progress = self.progress.value()
        self.events.put(seq, value)
        heapq.heappush(heap, (event_time, seq))
        self.order.update(heap)
        self.progress.update(max(progress if progress is not None else -INFINITY, event_time - self.bound_ms))

        released = self._release()
        if (self.pending_n.value() or 0) >= self.batch_max:
            yield from self._flush()
        elif released and not self.timer_set.value():
            ctx.timer_service().register_processing_time_timer(
                ctx.timer_service().current_processing_time() + self.flush_ms)
            self.timer_set.update(True)

    def on_timer(self, timestamp, ctx: KeyedProcessFunction.OnTimerContext):
        self.timer_set.update(False)
        yield from self._flush()

    # ── release in v1 order, then features ──────────────────
    def _release(self):
        from stream_features import AmountWindow, engineered, model_row

        heap, progress = self.order.value() or [], self.progress.value()
        if not heap or heap[0][0] > progress:
            return 0
        window = AmountWindow(self.window.value() or [])
        last = self.max_released.value()
        released = 0
        while heap and heap[0][0] <= progress:
            _, seq = heapq.heappop(heap)
            if last is not None:
                for gap in range(last + 1, seq):     # never arrived within the bound
                    self.skipped.put(gap, True)
            last = seq
            event = json.loads(self.events.get(seq))
            self.events.remove(seq)
            feats = engineered(event["Amount"], event["Time"], window.push(event["Amount"]))
            self.pending.add(json.dumps([seq, model_row(event, feats, self.features), feats,
                                         event.get("produce_ts_ms")]))
            released += 1
        self.order.update(heap)
        self.window.update(list(window.amounts))
        self.max_released.update(last)
        self.pending_n.update((self.pending_n.value() or 0) + released)
        return released

    # ── batched scoring ─────────────────────────────────────
    def _flush(self):
        import pandas as pd
        from inference import decide, shap_contributions

        pending = [json.loads(p) for p in (self.pending.get() or [])]
        if not pending:
            return
        X = pd.DataFrame([row for _, row, _, _ in pending], columns=self.features)
        scores = self.model.predict(self.xgb.DMatrix(X))       # one call for the whole batch
        out_ms = int(time.time() * 1000)
        for i, (seq, _, feats, produce_ts) in enumerate(pending):
            score = float(scores[i])
            out = {"seq": seq, "score": score,
                   "decision": "DECLINE" if score >= self.threshold else "APPROVE",
                   **feats, "produce_ts_ms": produce_ts, "flink_out_ms": out_ms, "batch_size": len(pending)}
            if out["decision"] == "DECLINE":
                contributions, _ = shap_contributions(self.model, X.iloc[[i]])
                out["notice"] = decide(score, contributions, self.threshold)["adverse_action_notice"]
            yield json.dumps(out)
        self.pending.clear()
        self.pending_n.update(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default="kafka:9092")
    ap.add_argument("--group", default="fraud-stream")
    ap.add_argument("--bound-seconds", type=int, default=60)
    ap.add_argument("--batch-max", type=int, default=200)
    ap.add_argument("--flush-ms", type=int, default=100)
    ap.add_argument("--checkpoint-ms", type=int, default=5000)
    ap.add_argument("--bundle-ms", type=int, default=100)
    ap.add_argument("--model", default="/opt/model/xgboost_model.json")
    ap.add_argument("--meta", default="/opt/model/model_meta.json")
    args, _ = ap.parse_known_args()

    # results leave the Python worker once per bundle; the 1,000 ms default put a ~1 s
    # floor under end-to-end latency (D14), so bundles close at 100 ms or 1,000 events
    config = Configuration()
    config.set_integer("python.fn-execution.bundle.time", args.bundle_ms)
    config.set_integer("python.fn-execution.bundle.size", 1000)
    env = StreamExecutionEnvironment.get_execution_environment(config)
    env.set_parallelism(1)
    env.enable_checkpointing(args.checkpoint_ms, CheckpointingMode.EXACTLY_ONCE)

    source = (KafkaSource.builder()
              .set_bootstrap_servers(args.bootstrap)
              .set_topics("transactions")
              .set_group_id(args.group)
              .set_starting_offsets(KafkaOffsetsInitializer.earliest())
              .set_value_only_deserializer(SimpleStringSchema())
              # bounded fetches: with the 50 MB default, a 3x overload backlog exhausted
              # the TaskManager's direct memory and failed the job (D14)
              .set_property("fetch.max.bytes", "5242880")
              .set_property("max.partition.fetch.bytes", "1048576")
              .build())

    def sink(topic):
        return (KafkaSink.builder()
                .set_bootstrap_servers(args.bootstrap)
                .set_record_serializer(KafkaRecordSerializationSchema.builder()
                                       .set_topic(topic)
                                       .set_value_serialization_schema(SimpleStringSchema())
                                       .build())
                .set_delivery_guarantee(DeliveryGuarantee.AT_LEAST_ONCE)
                .build())

    fn = ScoreInOrder(args.model, args.meta, args.bound_seconds * 1000, args.batch_max, args.flush_ms)
    scored = (env.from_source(source, WatermarkStrategy.no_watermarks(), "transactions")
              .key_by(lambda _: 0, key_type=Types.INT())
              .process(fn, output_type=Types.STRING())
              .name("order-dedupe-features-score"))

    scored.sink_to(sink("scores")).name("scores")
    scored.get_side_output(LATE).sink_to(sink("late-events")).name("late-events")
    scored.get_side_output(DUPLICATE).sink_to(sink("duplicate-events")).name("duplicate-events")

    env.execute("fraud-v2-stream")


if __name__ == "__main__":
    main()
