from __future__ import annotations

import logging

from fastapi import FastAPI, Body

from src.config.logging import setup_logging
from src.config.settings import settings

try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
except Exception:  # pragma: no cover
    generate_latest = None  # type: ignore
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"  # type: ignore


setup_logging()
logger = logging.getLogger("mini_aime.api")

app = FastAPI(title="Mini-Aime API", debug=settings.debug)


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("API starting up", extra={"debug": settings.debug})


@app.get("/health")
async def health() -> dict:
    logger.debug("Health check called")
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():  # type: ignore[override]
    if not settings.enable_metrics:
        return {"detail": "metrics disabled"}
    if generate_latest is None:
        return {"detail": "prometheus-client not installed"}

    data = generate_latest()
    from fastapi import Response

    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.post("/agent/resolution")
async def submit_resolution(
    task_id: str = Body(..., embed=True),
    resolution_type: str = Body(..., embed=True, description="resume|retry|replan"),
    user_hint: str | None = Body(None, embed=True)
) -> dict:
    """提交用户指引用于恢复/重试/重规划。

    注：这里暂为占位实现，实际应调用 ProgressManager/Planner 对应逻辑。
    """
    if not settings.human_in_loop_enabled:
        return {"detail": "human-in-loop disabled"}

    return {
        "task_id": task_id,
        "resolution_type": resolution_type,
        "user_hint": (user_hint or "").strip(),
        "status": "accepted"
    }
