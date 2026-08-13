"""FastAPI application for the interactive degradation demo.

Engineering choices that follow the spec:

* models are loaded **once** into a registry at startup, never per request;
* endpoints are async and inference runs in a thread pool, so a slow forward pass cannot
  block the event loop that is serving the WebSocket;
* responses are cached on ``(sample_id, model, corruption_hash)`` — the demo is highly
  repetitive, so the hit rate is excellent;
* every response carries a request id, and logs are structured JSON.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from wfb import __version__
from wfb.serving.inference import InferenceRegistry, RegistryConfig, sentiment_label
from wfb.serving.results_store import ResultsStore
from wfb.serving.schemas import (
    CompareRequest,
    CompareResponse,
    CorruptionInfo,
    CorruptionRequest,
    DegradationResponse,
    HealthResponse,
    ModelInfo,
    ParetoResponse,
    PredictRequest,
    PredictResponse,
    RelianceResponse,
    SampleInfo,
)

logger = logging.getLogger("wfb.serving")

_STATE: dict[str, Any] = {}

RATE_LIMIT_PER_MINUTE = int(os.environ.get("WFB_RATE_LIMIT", "240"))
_rate_buckets: dict[str, list[float]] = {}


def registry_config_from_env() -> RegistryConfig:
    """Build the registry config from environment variables (12-factor style)."""
    models = os.environ.get("WFB_MODELS", "")
    return RegistryConfig(
        dataset=os.environ.get("WFB_DATASET", "mosi"),
        checkpoint_dir=os.environ.get("WFB_CHECKPOINTS", "outputs"),
        results_dir=os.environ.get("WFB_RESULTS", "experiments/results"),
        redis_url=os.environ.get("WFB_REDIS_URL") or None,
        seed=int(os.environ.get("WFB_SEED", "0")),
        device=os.environ.get("WFB_DEVICE", "cpu"),
        models=tuple(m.strip() for m in models.split(",") if m.strip()) or RegistryConfig().models,
        force_synthetic=os.environ.get("WFB_SYNTHETIC", "").lower() in {"1", "true", "yes"},
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001 - FastAPI signature
    """Load the dataset, models and results once, before the first request."""
    started = time.perf_counter()
    cfg = _STATE.get("config") or registry_config_from_env()
    _STATE["registry"] = InferenceRegistry(cfg)
    _STATE["results"] = ResultsStore.load(cfg.results_dir)
    logger.info(
        json.dumps(
            {
                "event": "startup",
                "seconds": round(time.perf_counter() - started, 3),
                "fingerprint": _STATE["registry"].fingerprint(),
            }
        )
    )
    yield
    _STATE.clear()


def get_registry() -> InferenceRegistry:
    """Dependency: the warm model registry."""
    registry = _STATE.get("registry")
    if registry is None:
        raise HTTPException(status_code=503, detail="Registry is not ready")
    return registry  # type: ignore[no-any-return]


def get_results() -> ResultsStore:
    """Dependency: the precomputed sweep results."""
    results = _STATE.get("results")
    if results is None:
        raise HTTPException(status_code=503, detail="Results are not loaded")
    return results  # type: ignore[no-any-return]


def create_app(config: RegistryConfig | None = None) -> FastAPI:
    """Build the ASGI application. Tests pass an explicit ``config``."""
    if config is not None:
        _STATE["config"] = config

    app = FastAPI(
        title="When Fusion Breaks",
        version=__version__,
        summary="Graceful degradation in multimodal emotion recognition",
        description=(
            "Interactive API behind the degradation explorer. Corrupt a sample's "
            "modalities and watch every fusion architecture respond to the *same* "
            "corrupted input."
        ),
        lifespan=lifespan,
    )

    origins = os.environ.get(
        "WFB_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in origins if o.strip()],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next: Callable[..., Any]) -> Any:
        """Attach a request id, enforce a simple rate limit, and log structurally."""
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        client = request.client.host if request.client else "unknown"

        now = time.time()
        bucket = [t for t in _rate_buckets.get(client, []) if now - t < 60.0]
        if len(bucket) >= RATE_LIMIT_PER_MINUTE:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded", "request_id": request_id},
            )
        bucket.append(now)
        _rate_buckets[client] = bucket

        started = time.perf_counter()
        response = await call_next(request)
        duration = (time.perf_counter() - started) * 1000
        response.headers["x-request-id"] = request_id
        logger.info(
            json.dumps(
                {
                    "event": "request",
                    "id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "ms": round(duration, 2),
                }
            )
        )
        return response

    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:
    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    async def health() -> HealthResponse:
        """Liveness and readiness, including what data is actually being served."""
        registry = get_registry()
        trained = sum(1 for m in registry.models.values() if m.trained)
        return HealthResponse(
            status="ok" if trained else "degraded",
            version=__version__,
            models_loaded=len(registry.models),
            trained_models=trained,
            dataset=registry.bundle.name,
            dataset_source=registry.bundle.provenance.source,
            cache=registry.cache.backend,  # type: ignore[arg-type]
            uptime_seconds=registry.uptime,
        )

    @app.get("/api/models", response_model=list[ModelInfo], tags=["registry"])
    async def list_models(
        registry: InferenceRegistry = Depends(get_registry),
    ) -> list[ModelInfo]:
        """Every loaded architecture, its size, and whether it is actually trained."""
        return [
            ModelInfo(
                name=entry.name,
                architecture=type(entry.module).__name__,
                modalities=[m.value for m in entry.module.active],
                parameters=entry.module.num_parameters,
                trained=entry.trained,
                checkpoint=entry.checkpoint,
                clean_metrics=entry.clean_metrics,
                fusion_rank=entry.fusion_rank,
            )
            for entry in registry.models.values()
        ]

    @app.get("/api/corruptions", response_model=list[CorruptionInfo], tags=["registry"])
    async def list_corruptions() -> list[CorruptionInfo]:
        """The corruption registry, so the UI can build its controls from the server."""
        from wfb.corruption.registry import catalogue

        return [CorruptionInfo(**entry) for entry in catalogue()]  # type: ignore[arg-type]

    @app.get("/api/samples", response_model=list[SampleInfo], tags=["data"])
    async def list_samples(
        dataset: str | None = Query(default=None),
        limit: int = Query(default=24, ge=1, le=200),
        registry: InferenceRegistry = Depends(get_registry),
    ) -> list[SampleInfo]:
        """Curated demo clips."""
        if dataset is not None and dataset != registry.bundle.name:
            raise HTTPException(
                status_code=404,
                detail=f"This instance serves {registry.bundle.name!r}, not {dataset!r}",
            )
        return [
            SampleInfo(
                id=sample_id,
                dataset=registry.bundle.name,
                split=registry.cfg.split,
                label=registry.label_for(sample_id),
                sentiment=sentiment_label(registry.label_for(sample_id)),
                media_url=None,
                transcript=None,
            )
            for sample_id in registry.sample_ids(limit)
        ]

    @app.post("/api/predict", response_model=PredictResponse, tags=["inference"])
    async def predict(
        request: PredictRequest, registry: InferenceRegistry = Depends(get_registry)
    ) -> PredictResponse:
        """One sample, one model, one corruption setting."""
        return PredictResponse(**await _predict_cached(registry, request))

    @app.post("/api/compare", response_model=CompareResponse, tags=["inference"])
    async def compare(
        request: CompareRequest, registry: InferenceRegistry = Depends(get_registry)
    ) -> CompareResponse:
        """The same corrupted input through every architecture — the hero view.

        Watching one architecture fall off a cliff while another holds steady, under
        provably identical inputs, is the entire hypothesis in a single interaction.
        """
        started = time.perf_counter()
        names = request.models or list(registry.models)
        unknown = [n for n in names if n not in registry.models]
        if unknown:
            raise HTTPException(status_code=404, detail=f"Unknown models: {unknown}")
        if not registry.has_sample(request.sample_id):
            raise HTTPException(status_code=404, detail=f"Unknown sample {request.sample_id!r}")

        results = []
        for name in names:
            single = PredictRequest(
                sample_id=request.sample_id,
                model=name,
                corruption=request.corruption,
                return_attention=request.return_attention,
            )
            results.append(PredictResponse(**await _predict_cached(registry, single)))

        return CompareResponse(
            sample_id=request.sample_id,
            corruption_description=results[0].corruption_description if results else "clean",
            results=results,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    @app.get("/api/results/degradation", response_model=DegradationResponse, tags=["results"])
    async def degradation(
        dataset: str | None = Query(default=None),
        results: ResultsStore = Depends(get_results),
    ) -> DegradationResponse:
        """Precomputed retention curves with seed bands."""
        return DegradationResponse(
            dataset=dataset or (results.datasets()[0] if results.datasets() else "none"),
            metric=results.metric(dataset),
            provenance=results.provenance(dataset),
            curves=results.degradation_curves(dataset),  # type: ignore[arg-type]
            brittleness=results.brittleness(dataset),
        )

    @app.get("/api/results/reliance", response_model=RelianceResponse, tags=["results"])
    async def reliance(
        dataset: str | None = Query(default=None),
        results: ResultsStore = Depends(get_results),
    ) -> RelianceResponse:
        """Modality Reliance Score matrices and the 7-subset removal grid."""
        return RelianceResponse(
            dataset=dataset or (results.datasets()[0] if results.datasets() else "none"),
            metric=results.metric(dataset),
            provenance=results.provenance(dataset),
            entries=results.reliance(dataset),  # type: ignore[arg-type]
        )

    @app.get("/api/results/pareto", response_model=ParetoResponse, tags=["results"])
    async def pareto(
        dataset: str | None = Query(default=None),
        results: ResultsStore = Depends(get_results),
    ) -> ParetoResponse:
        """Clean performance against AUDC, with the Pareto frontier flagged."""
        return ParetoResponse(
            dataset=dataset or (results.datasets()[0] if results.datasets() else "none"),
            metric=results.metric(dataset),
            points=results.pareto(dataset),  # type: ignore[arg-type]
        )

    @app.websocket("/ws/live")
    async def live(websocket: WebSocket) -> None:
        """Streaming updates as the sliders move.

        Each message is a :class:`CompareRequest`; each reply is a
        :class:`CompareResponse`. Using the compare shape (rather than single predict)
        means one round trip per slider tick regardless of how many architectures the
        client is watching.
        """
        await websocket.accept()
        registry = _STATE.get("registry")
        if registry is None:
            await websocket.close(code=1011)
            return
        try:
            while True:
                payload = await websocket.receive_json()
                try:
                    request = CompareRequest.model_validate(payload)
                except ValueError as exc:
                    await websocket.send_json({"error": str(exc)})
                    continue
                if not registry.has_sample(request.sample_id):
                    await websocket.send_json({"error": f"unknown sample {request.sample_id}"})
                    continue

                names = request.models or list(registry.models)
                results = []
                for name in names:
                    if name not in registry.models:
                        continue
                    single = PredictRequest(
                        sample_id=request.sample_id,
                        model=name,
                        corruption=request.corruption,
                        return_attention=request.return_attention,
                    )
                    results.append(await _predict_cached(registry, single))
                await websocket.send_json(
                    {
                        "sample_id": request.sample_id,
                        "corruption_description": (
                            results[0]["corruption_description"] if results else "clean"
                        ),
                        "results": results,
                    }
                )
        except WebSocketDisconnect:
            logger.info(json.dumps({"event": "ws_disconnect"}))


def _plan_from_request(registry: InferenceRegistry, corruption: CorruptionRequest) -> Any:
    settings = [
        (modality, setting.type, setting.severity, setting.params)
        for modality, setting in corruption.as_pairs()
    ]
    try:
        return registry.build_plan(settings)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


async def _predict_cached(registry: InferenceRegistry, request: PredictRequest) -> dict[str, Any]:
    """Cache-aware inference, off the event loop."""
    if not registry.has_sample(request.sample_id):
        raise HTTPException(status_code=404, detail=f"Unknown sample {request.sample_id!r}")
    if request.model not in registry.models:
        raise HTTPException(status_code=404, detail=f"Unknown model {request.model!r}")

    plan = _plan_from_request(registry, request.corruption)
    key = registry.cache.key(request.sample_id, request.model, plan.hash())
    if not request.return_attention:
        cached = registry.cache.get(key)
        if cached is not None:
            return {**cached, "cached": True}

    try:
        payload = await run_in_threadpool(
            registry.infer,
            request.sample_id,
            request.model,
            plan,
            request.return_attention,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not request.return_attention:
        registry.cache.set(key, payload)
    return payload


app = create_app()
