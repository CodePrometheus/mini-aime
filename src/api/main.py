from __future__ import annotations

import logging

from fastapi import FastAPI

from src.config.logging import setup_logging
from src.config.settings import settings


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
