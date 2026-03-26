"""
FastAPI application for the Wikipedia Country Infobox API.
Run with: uvicorn src.app:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI

from api.controllers import health_controller, wikiinfobox_controller

tags_metadata = [
    {"name": "health", "description": "API health check"},
    {"name": "wikiinfobox", "description": "Wikipedia country infobox data: countries list, JSON, tree, HTML"},
]

app = FastAPI(
    version="1.0",
    title="Wikipedia Country Infobox API",
    description="API for browsing Wikipedia country infobox data (JSON, trees, raw HTML).",
    openapi_tags=tags_metadata,
)

app.include_router(
    health_controller.router,
    prefix="/health",
    tags=["health"],
)

app.include_router(
    wikiinfobox_controller.router,
    prefix="/wikiinfobox",
    tags=["wikiinfobox"],
)
