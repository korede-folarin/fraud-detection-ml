#!/usr/bin/env bash
# One streaming test run (D14). From the repository root:
#   bash v2/streaming/run_stream.sh <name> <rate> [dup_share] [late_share] [too_late] [kill_tm_after_s]
# e.g.  bash v2/streaming/run_stream.sh run-1000 1000 0.01 0.005 20
#
# Plain English: resets the Kafka topics, starts a fresh Flink job, starts the
# checker listening, replays the payments, and waits for the checker's verdict.
# The optional last argument restarts the Flink TaskManager mid-run to prove the
# job recovers from its checkpoint (state survives, answers stay identical).
set -euo pipefail
cd "$(dirname "$0")"

NAME=${1:?name}; RATE=${2:?rate}; DUP=${3:-0}; LATE=${4:-0}; TOO_LATE=${5:-0}; KILL_AFTER=${6:-0}
DC="docker compose -f docker-compose.yml"
export MSYS_NO_PATHCONV=1

$DC up -d kafka jobmanager taskmanager tools >/dev/null
until $DC exec -T kafka /opt/kafka/bin/kafka-broker-api-versions.sh --bootstrap-server kafka:9092 >/dev/null 2>&1; do sleep 2; done
until curl -sf http://127.0.0.1:8081/taskmanagers | grep -q '"slotsNumber":1'; do sleep 2; done

# cancel any job left from an earlier run
for id in $(curl -s http://127.0.0.1:8081/jobs | python -c "import json,sys; print(' '.join(j['id'] for j in json.load(sys.stdin)['jobs'] if j['status'] in ('RUNNING','RESTARTING','CREATED')))"); do
  curl -s -X PATCH "http://127.0.0.1:8081/jobs/$id?mode=cancel" >/dev/null
done
sleep 3

$DC exec -T tools python producer.py --topics-only
$DC exec -T jobmanager flink run -d -py /opt/job/fraud_stream_job.py -pyfs /opt/job,/opt/src --group "stream-$NAME" | tail -1
until curl -s http://127.0.0.1:8081/jobs/overview | grep -q '"state":"RUNNING"'; do sleep 1; done
sleep 5   # let the Python worker start before traffic arrives

EXPECT=$($DC exec -T tools python -c "import pandas as pd; print(len(pd.read_parquet('/work/data/replay.parquet')))")
$DC exec -d tools sh -c "python parity.py --name $NAME --expect $EXPECT > /work/results/$NAME.log 2>&1"

if [ "$KILL_AFTER" != "0" ]; then
  ( sleep "$KILL_AFTER"; echo "restarting TaskManager at +${KILL_AFTER}s"; $DC restart taskmanager >/dev/null ) &
fi
$DC exec -T tools python producer.py --no-recreate --rate "$RATE" --dup-share "$DUP" --late-share "$LATE" --too-late "$TOO_LATE"
wait

until [ -s "results/$NAME.json" ]; do sleep 5; done
python - "$NAME" <<'EOF'
import json, sys
r = json.load(open(f"results/{sys.argv[1]}.json"))
print(f"parity_passed={r['parity_passed']} scored={r['handled']['scored_unique']} late={r['handled']['late_events_side_output']}"
      f" dups_dropped={r['handled']['duplicate_inputs_dropped']} throughput={r['throughput_events_per_s']}/s"
      f" latency_ms={r['end_to_end_latency_ms']}")
EOF
