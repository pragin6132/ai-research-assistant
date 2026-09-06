"""
POST /api/generate-image endpoint.

Exposes the existing `PixazoImageService` over HTTP. Accepts a text prompt
and returns the generated image's URL. The Pixazo API key never leaves the
backend — it's read server-side from Settings inside `PixazoImageService`
and never sent to or accepted from the client.

Independent of the research pipeline: does not touch `app.graph`,
`app.agents`, or the `/api/research` streaming endpoint.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.pixazo_service import (
    PixazoConfigurationError,
    PixazoImageService,
    PixazoServiceError,
)

router = APIRouter(tags=["image"])


class ImageRequest(BaseModel):
    """Request body for POST /api/generate-image."""

    prompt: str = Field(..., min_length=1, description="Description of the image to generate.")


class ImageResponse(BaseModel):
    """Response body for POST /api/generate-image."""

    image_url: str
    model: str


@router.post("/generate-image", response_model=ImageResponse)
def generate_image(request: ImageRequest) -> ImageResponse:
    """Generate an image for the given prompt via Pixazo and return its URL."""
    service = PixazoImageService()

    try:
        result = service.generate_image(request.prompt)
    except PixazoConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PixazoServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ImageResponse(image_url=result.image_url, model=result.model)
