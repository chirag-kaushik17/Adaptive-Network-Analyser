"""Measure aggregation, streaming pipeline, and in-process API throughput."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi.testclient import TestClient

from .api import create_app
from .features import window_features
from .pipeline import AtaPipeline
from .run import DEFAULTS_PATH
from .stream import FlowStream, iter_flow_batches


def _throughput(count: int, elapsed: float) -> float | None:
    return count / elapsed if elapsed > 0 else None


def run_benchmark(
    csv_path: str | Path,
    *,
    dataset: str = "synth",
    batch_size: int = 2000,
    output: str | Path = "reports/throughput.json",
) -> dict[str, Any]:
    defaults = json.loads(DEFAULTS_PATH.read_text(encoding="utf-8"))
    # Reuse one loaded input for all three stage measurements.
    source = FlowStream(csv_path, dataset=dataset)
    flows = list(source)
    load_stats = source.load_stats
    started = time.perf_counter()
    windows = window_features(
        pd.DataFrame.from_records(flows),
        window=1.0,
    )
    aggregation_elapsed = time.perf_counter() - started

    pipeline = AtaPipeline(
        window=1.0,
        warmup=defaults["warmup_windows"],
        history=defaults["history_windows"],
        refit_every=defaults["refit_every_windows"],
        contamination=defaults["contamination"],
        model_seed=defaults["model_seed"],
        drift=defaults["drift"],
    )
    started = time.perf_counter()
    pipeline_alerts = []
    for batch in iter_flow_batches(iter(flows), batch_size):
        pipeline_alerts.extend(pipeline.process_flows(batch))
    pipeline_alerts.extend(pipeline.flush())
    pipeline_elapsed = time.perf_counter() - started

    app = create_app(
        {
            "database": ":memory:",
            "warmup": defaults["warmup_windows"],
            "history": defaults["history_windows"],
            "refit_every": defaults["refit_every_windows"],
            "contamination": defaults["contamination"],
            "model_seed": defaults["model_seed"],
            "drift": defaults["drift"],
            "max_batch_size": max(batch_size, 1),
        }
    )
    accepted = api_windows = api_alerts = 0
    started = time.perf_counter()
    with TestClient(app) as client:
        for batch in iter_flow_batches(iter(flows), batch_size):
            response = client.post("/flows", json={"flows": batch})
            response.raise_for_status()
            result = response.json()
            accepted += result["flows_accepted"]
            api_windows += result["windows_processed"]
            api_alerts += len(result["alerts"])
        final_response = client.post("/flush")
        final_response.raise_for_status()
        final_result = final_response.json()
        api_windows += final_result["windows_processed"]
        api_alerts += len(final_result["alerts"])
        final_stats = client.get("/stats").json()
    api_elapsed = time.perf_counter() - started

    report = {
        "status": "PASS",
        "dataset_type": dataset,
        "input_file": str(Path(csv_path).resolve()),
        "flow_count": len(flows),
        "window_count_batch_aggregation": len(windows),
        "window_count_pipeline": pipeline.windows_processed,
        "window_count_api": api_windows,
        "configuration": {
            "window_seconds": 1.0,
            "warmup_windows": defaults["warmup_windows"],
            "history_windows": defaults["history_windows"],
            "refit_every_windows": defaults["refit_every_windows"],
            "contamination": defaults["contamination"],
            "model_seed": defaults["model_seed"],
            "replay_batch_size": batch_size,
        },
        "stages": {
            "window_aggregation": {
                "elapsed_seconds": aggregation_elapsed,
                "flows_per_second": _throughput(len(flows), aggregation_elapsed),
                "windows_per_second": _throughput(len(windows), aggregation_elapsed),
            },
            "full_pipeline": {
                "elapsed_seconds": pipeline_elapsed,
                "flows_per_second": _throughput(len(flows), pipeline_elapsed),
                "windows_per_second": _throughput(
                    pipeline.windows_processed, pipeline_elapsed
                ),
                "alerts_generated": len(pipeline_alerts),
            },
            "api_ingestion": {
                "transport": "FastAPI TestClient in-process; excludes socket/server overhead",
                "elapsed_seconds": api_elapsed,
                "flows_per_second": _throughput(accepted, api_elapsed),
                "windows_per_second": _throughput(api_windows, api_elapsed),
                "alerts_generated": api_alerts,
                "final_stats": final_stats,
            },
        },
        "machine": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": __import__("os").cpu_count(),
        },
        "loader_stats": load_stats,
        "errors_or_limitations": [
            "Synthetic throughput is a local engineering measurement, not real-world capacity evidence."
            if dataset == "synth"
            else "CIC throughput measures only the supplied local file and does not establish general capacity.",
            "API timing uses in-process TestClient and excludes Uvicorn socket and process overhead.",
        ],
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ATA processing stages.")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--dataset", choices=["synth", "cicids2017"], default="synth")
    parser.add_argument("--batch", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("reports/throughput.json"))
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch must be at least one.")
    run_benchmark(args.csv, dataset=args.dataset, batch_size=args.batch, output=args.out)


if __name__ == "__main__":
    main()
