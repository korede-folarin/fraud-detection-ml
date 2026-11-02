"""
Log streaming test results to MLflow (D14)
Run from the repository root: python v2/streaming/log_runs.py results/run-1000.json [...]

Plain English: copies each streaming run's real measurements (parity verdict,
latency, throughput, faults handled) into a separate MLflow experiment, so v2
evidence sits next to — but never inside — v1's experiment.

Technical: experiment fraud-v2-streaming on the DagsHub tracking server from
v1_frozen/MANIFEST.json; every run tagged dataset_version and baseline. The
DagsHub token is read from the environment or .env, never from code; without
it the script only prints what it would log.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "fraud-v2-streaming"


def flatten(d, prefix=""):
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            yield from flatten(v, f"{key}.")
        elif isinstance(v, bool):
            yield key, int(v)
        elif isinstance(v, (int, float)) and v is not None:
            yield key, v


def main(paths):
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    manifest = json.loads((ROOT / "v1_frozen" / "MANIFEST.json").read_text(encoding="utf-8"))
    token = os.getenv("DAGSHUB_TOKEN")
    if token:
        import mlflow
        os.environ.setdefault("MLFLOW_TRACKING_USERNAME", "korede-folarin")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        mlflow.set_tracking_uri(manifest["mlflow"]["tracking_uri"])
        mlflow.set_experiment(EXPERIMENT)

    for path in paths:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        metrics = dict(flatten({k: v for k, v in result.items() if k not in ("name", "tolerances")}))
        tags = {"dataset_version": manifest["dataset_version"], "baseline": manifest["git_tag"],
                "decision": "D14", "parity_passed": str(result["parity_passed"])}
        if not token:
            print(f"[dry run, DAGSHUB_TOKEN not set] {result['name']}: {len(metrics)} metrics, tags {tags}")
            continue
        with mlflow.start_run(run_name=result["name"]):
            mlflow.set_tags(tags)
            mlflow.log_metrics(metrics)
            mlflow.log_dict(result, f"{result['name']}.json")
        print(f"logged {result['name']} to {EXPERIMENT}")


if __name__ == "__main__":
    main(sys.argv[1:])
