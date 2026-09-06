"""
Entrypoint for the AI Research Assistant backend.

Run with:
    uvicorn main:app --reload
(from inside the backend/ directory)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.api.image import router as image_router
from app.api.research import router as research_router
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.DEBUG,
)

# The frontend (Step 9A) is a standalone static page with no build step, so
# it's opened directly (file://) or served from a different port than this
# API. CORS must be open for its fetch() calls to reach /api/* at all.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api")
app.include_router(research_router, prefix="/api")
app.include_router(image_router, prefix="/api")
