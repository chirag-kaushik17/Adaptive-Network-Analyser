"""Replay a canonical ATA CSV into a running local API."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

from .stream import FlowStream, iter_flow_batches


def replay(
    csv_path: str | Path,
    *,
    url: str = "http://127.0.0.1:8000",
    dataset: str = "synth",
    batch_size: int = 2000,
    speed: float = 0.0,
    timeout: float = 30.0,
    output: str | Path = "reports/phase7_replay_summary.json",
    date_order: str = "auto",
    allow_any_dates: bool = False,
) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be at least one.")
    if timeout <= 0:
        raise ValueError("timeout must be positive.")
    stream = FlowStream(
        csv_path,
        dataset=dataset,
        speed=speed,
        date_order=date_order,
        allow_any_dates=allow_any_dates,
    )
    started = time.perf_counter()
    sent_flows = processed_windows = returned_alerts = 0
    error: str | None = None
    try:
        with httpx.Client(timeout=timeout) as client:
            for batch in iter_flow_batches(iter(stream), batch_size):
                response = client.post(
                    f"{url.rstrip('/')}/flows",
                    json={"flows": batch},
                )
                response.raise_for_status()
                result = response.json()
                sent_flows += int(result["flows_accepted"])
                processed_windows += int(result["windows_processed"])
                returned_alerts += len(result["alerts"])
            final_response = client.post(f"{url.rstrip('/')}/flush")
            final_response.raise_for_status()
            final_result = final_response.json()
            processed_windows += int(final_result["windows_processed"])
            returned_alerts += len(final_result["alerts"])
        response = httpx.get(f"{url.rstrip('/')}/stats", timeout=timeout)
        response.raise_for_status()
        final_stats = response.json()
    except (httpx.HTTPError, KeyError, ValueError, OSError) as exc:
        error = str(exc)
        final_stats = None
    elapsed = time.perf_counter() - started
    summary = {
        "status": "FAIL" if error else "PASS",
        "dataset": dataset,
        "csv": str(Path(csv_path).resolve()),
        "api_url": url,
        "batch_size": batch_size,
        "replay_speed_multiplier": speed,
        "flows_sent": sent_flows,
        "windows_processed_in_responses": processed_windows,
        "alerts_returned": returned_alerts,
        "elapsed_seconds": elapsed,
        "flows_per_second": sent_flows / elapsed if elapsed else None,
        "final_api_stats": final_stats,
        "load_stats": stream.load_stats,
        "error": error,
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))
    if error:
        raise RuntimeError(f"Replay failed: {error}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay flow CSVs into ATA's API.")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset", choices=["synth", "cicids2017"], default="synth")
    parser.add_argument("--batch", type=int, default=2000)
    parser.add_argument("--speed", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--out", type=Path, default=Path("reports/phase7_replay_summary.json"))
    parser.add_argument("--date-order", choices=["auto", "dmy", "mdy"], default="auto")
    parser.add_argument("--allow-any-dates", action="store_true")
    args = parser.parse_args()
    try:
        replay(
            args.csv,
            url=args.url,
            dataset=args.dataset,
            batch_size=args.batch,
            speed=args.speed,
            timeout=args.timeout,
            output=args.out,
            date_order=args.date_order,
            allow_any_dates=args.allow_any_dates,
        )
    except (RuntimeError, ValueError) as exc:
        parser.exit(1, f"ata.replay: error: {exc}\n")


if __name__ == "__main__":
    main()
