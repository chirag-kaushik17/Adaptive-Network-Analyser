"""Local FastAPI interface for the ATA streaming pipeline."""

from __future__ import annotations

import importlib.metadata
import math
import threading
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .features import FEATURE_COLUMNS
from .pipeline import AtaPipeline, validate_flow_batch
from .store import AlertStore

Protocol = Literal["tcp", "udp", "icmp", "other"]


class FlowRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ts: float = Field(ge=0, allow_inf_nan=False)
    src: str = Field(min_length=1, max_length=128)
    dst: str = Field(min_length=1, max_length=128)
    dport: int = Field(strict=True, ge=0, le=65535)
    proto: Protocol
    bytes: float = Field(ge=0, allow_inf_nan=False)
    packets: float = Field(ge=0, allow_inf_nan=False)
    duration: float = Field(ge=0, allow_inf_nan=False)
    syn: int = Field(strict=True, ge=0, le=1)
    label: str | None = Field(default=None, max_length=64)
    label_raw: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def validate_addresses_and_numbers(self):
        if not self.src.strip() or not self.dst.strip():
            raise ValueError("src and dst must be non-empty.")
        if not all(math.isfinite(value) for value in (self.ts, self.bytes, self.packets, self.duration)):
            raise ValueError("numeric values must be finite.")
        return self


class FlowBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flows: list[FlowRequest] = Field(min_length=1)


class FeatureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flows: float = Field(ge=0, allow_inf_nan=False)
    bytes_total: float = Field(ge=0, allow_inf_nan=False)
    bytes_mean: float = Field(ge=0, allow_inf_nan=False)
    pkts_mean: float = Field(ge=0, allow_inf_nan=False)
    dur_mean: float = Field(ge=0, allow_inf_nan=False)
    syn_ratio: float = Field(ge=0, le=1, allow_inf_nan=False)
    uniq_src: float = Field(ge=0, allow_inf_nan=False)
    uniq_dst: float = Field(ge=0, allow_inf_nan=False)
    uniq_dport: float = Field(ge=0, allow_inf_nan=False)
    dport_entropy: float = Field(ge=0, allow_inf_nan=False)
    top_src_share: float = Field(ge=0, le=1, allow_inf_nan=False)


def _application_version() -> str:
    try:
        return importlib.metadata.version("ata")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0"


def create_app(config: dict[str, Any] | None = None) -> FastAPI:
    """Create an app instance; optional injected pipeline/store simplify tests."""
    options = dict(config or {})
    pipeline = options.get("pipeline") or AtaPipeline(
        window=float(options.get("window", 1.0)),
        warmup=int(options.get("warmup", 120)),
        history=int(options.get("history", 300)),
        refit_every=options.get("refit_every", 30),
        contamination=float(options.get("contamination", 0.005)),
        model_seed=int(options.get("model_seed", 0)),
        refit_trigger=str(options.get("refit_trigger", "periodic")),
        drift=options.get("drift"),
        classifier=options.get("classifier"),
        explainer=options.get("explainer"),
        model_name=options.get("model_name"),
        telemetry_limit=int(options.get("telemetry_limit", 2000)),
        adaptation_policy=str(options.get("adaptation_policy", "strict")),
    )
    database = options.get("database", "data/ata_alerts.sqlite3")
    store = options.get("store") or AlertStore(database)
    max_batch_size = int(options.get("max_batch_size", 2000))
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be positive.")
    state_lock = threading.RLock()

    app = FastAPI(
        title="ATA local streaming API",
        version=_application_version(),
        docs_url="/docs",
        redoc_url=None,
    )
    app.state.pipeline = pipeline
    app.state.alert_store = store
    app.state.state_lock = state_lock
    app.state.max_batch_size = max_batch_size

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": "request_validation_failed",
                "details": [
                    {
                        "location": list(error["loc"]),
                        "message": error["msg"],
                        "type": error["type"],
                    }
                    for error in exc.errors()
                ],
            },
        )

    @app.get("/health")
    def health():
        with state_lock:
            stats = pipeline.stats()
            return {
                "status": "ok",
                "version": app.version,
                "models_loaded": {
                    "layer1": pipeline.ready,
                    "layer2": pipeline.classifier is not None,
                },
                "windows_processed": stats["windows_processed"],
                "warmup_complete": stats["warmup_complete"],
                "pipeline_ready": stats["pipeline_ready"],
            }

    @app.post("/flows")
    def ingest_flows(request: FlowBatchRequest):
        if len(request.flows) > max_batch_size:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "batch_too_large",
                    "message": f"Batch limit is {max_batch_size} flows.",
                },
            )
        candidate_flows = [flow.model_dump(exclude_none=True) for flow in request.flows]
        with state_lock:
            try:
                checked = validate_flow_batch(
                    candidate_flows,
                    previous_ts=pipeline.last_flow_ts,
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=409,
                    detail={"error": "invalid_flow_order_or_value", "message": str(exc)},
                ) from exc
            before_windows = pipeline.windows_processed
            before_alerts = pipeline.alerts_generated
            try:
                alerts = pipeline.process_flows(checked)
                for alert in alerts:
                    store.add(alert)
            except ValueError as exc:
                raise HTTPException(
                    status_code=422,
                    detail={"error": "flow_processing_failed", "message": str(exc)},
                ) from exc
            except Exception as exc:
                raise HTTPException(
                    status_code=503,
                    detail={"error": "pipeline_or_storage_unavailable"},
                ) from exc
            return {
                "flows_accepted": len(checked),
                "windows_processed": pipeline.windows_processed - before_windows,
                "total_windows_processed": pipeline.windows_processed,
                "alerts_generated": pipeline.alerts_generated - before_alerts,
                "alerts": alerts,
                "warmup_complete": pipeline.ready,
            }

    @app.post("/score")
    def score_features(request: FeatureRequest):
        with state_lock:
            if not pipeline.ready:
                raise HTTPException(
                    status_code=503,
                    detail={"error": "warmup_incomplete"},
                )
            try:
                return pipeline.score_features(request.model_dump())
            except ValueError as exc:
                raise HTTPException(
                    status_code=422,
                    detail={"error": "invalid_features", "message": str(exc)},
                ) from exc

    @app.post("/flush")
    def flush_pipeline():
        """Finish the last aggregation window and close this stream session."""
        with state_lock:
            before_windows = pipeline.windows_processed
            try:
                alerts = pipeline.flush()
                for alert in alerts:
                    store.add(alert)
            except Exception as exc:
                raise HTTPException(
                    status_code=503,
                    detail={"error": "pipeline_or_storage_unavailable"},
                ) from exc
            return {
                "windows_processed": pipeline.windows_processed - before_windows,
                "total_windows_processed": pipeline.windows_processed,
                "alerts_generated": len(alerts),
                "alerts": alerts,
                "pipeline_closed": True,
            }

    @app.get("/alerts")
    def list_alerts(
        limit: int = Query(default=100, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
        attack_type: str | None = Query(default=None, min_length=1, max_length=64),
        since: float | None = Query(default=None, ge=0),
    ):
        try:
            records = store.list(
                limit=limit,
                offset=offset,
                attack_type=attack_type,
                since=since,
            )
            return {
                "items": records,
                "count": len(records),
                "total": store.count(),
                "limit": limit,
                "offset": offset,
            }
        except (ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": "invalid_alert_query", "message": str(exc)},
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail={"error": "alert_store_unavailable"},
            ) from exc

    @app.get("/alerts/{alert_id}")
    def get_alert(alert_id: str):
        try:
            record = store.get(alert_id)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail={"error": "alert_store_unavailable"},
            ) from exc
        if record is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "alert_not_found"},
            )
        return record

    @app.get("/stats")
    def stats():
        with state_lock:
            values = pipeline.stats()
        try:
            values["stored_alerts"] = store.count()
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail={"error": "alert_store_unavailable"},
            ) from exc
        return values

    return app
