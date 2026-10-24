"""Run the local ATA FastAPI service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn

from .api import create_app
from .pipeline import train_demo_classifier
from .run import DEFAULTS_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local ATA streaming API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--database", default="data/ata_alerts.sqlite3")
    parser.add_argument("--with-classifier", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")

    try:
        defaults = json.loads(DEFAULTS_PATH.read_text(encoding="utf-8"))
        config = {
            "database": args.database,
            "warmup": defaults["warmup_windows"],
            "history": defaults["history_windows"],
            "refit_every": defaults["refit_every_windows"],
            "contamination": defaults["contamination"],
            "model_seed": defaults["model_seed"],
            "drift": defaults["drift"],
        }
        if args.with_classifier:
            model, explainer, model_name = train_demo_classifier(seed=1)
            config.update(
                classifier=model,
                explainer=explainer,
                model_name=model_name,
            )
        app = create_app(config)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ATA service configuration failed: {exc}") from exc

    print(f"Starting ATA API on http://{args.host}:{args.port}")
    print("Authentication is not configured; keep this service on localhost.")
    if not args.with_classifier:
        print("No classifier artifact is configured; serving Layer 1-only alerts.")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
