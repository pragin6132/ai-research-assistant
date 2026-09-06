"""
Pixazo image generation service.

Wraps the Pixazo image-generation API behind a small, dependency-injectable
class that returns a generated image URL for a given text prompt.

Pixazo's real contract (submit -> poll -> result) is asynchronous:
  1. POST https://gateway.pixazo.ai/{model}/v1/{model}-request
     header: Ocp-Apim-Subscription-Key: <api_key>
     body:   {"prompt": "..."}
     ->      {"request_id": "...", "status": "QUEUED", "polling_url": "..."}
  2. GET https://gateway.pixazo.ai/v2/requests/status/{request_id}
     header: Ocp-Apim-Subscription-Key: <api_key>
     ->      {"status": "QUEUED" | "PROCESSING" | "COMPLETED" | "FAILED" | "ERROR",
               "output": {"media_url": ["https://..."]}, ...}
     polled repeatedly until COMPLETED (or a failure/timeout).

NOTE ON MODELS: unlike a single unified "pass any model name" endpoint,
Pixazo assigns each model its own endpoint slug (confirmed by comparing
multiple of Pixazo's own model docs pages) — e.g. "z-image-base" submits to
".../z-image-base/v1/z-image-base-request", but other models' request-path
suffixes don't always mirror their slug exactly. This service builds the
submit URL from PIXAZO_IMAGE_MODEL using the common "{model}/v1/{model}-request"
pattern, which matches Pixazo's documented free-tier models (the default,
"z-image-base", is one of them). Switching PIXAZO_IMAGE_MODEL to a different
model requires confirming that model's exact endpoint slug on Pixazo's own
model page — it is not guaranteed to follow the same template for every
model in Pixazo's catalog.

This service is independent of the LLM Manager, Tavily service, and the
research pipeline — it does one thing: given a prompt, return a generated
image URL (or raise a clear error). The Pixazo API key never leaves the
backend.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import Settings, get_settings

logger = logging.getLogger("app.pixazo_service")

_BASE_URL = "https://gateway.pixazo.ai"
_POLL_INTERVAL_SECONDS = 2.0
_MAX_POLL_ATTEMPTS = 30  # ~60 seconds total, in line with Pixazo's documented
# generation times (a few seconds for fast models, up to ~30s for slower ones).
_REQUEST_TIMEOUT_SECONDS = 30.0


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PixazoServiceError(Exception):
    """Base class for all Pixazo image-generation errors."""


class PixazoConfigurationError(PixazoServiceError):
    """Raised when the Pixazo API key is missing/not configured."""


class PixazoConnectionError(PixazoServiceError):
    """Raised on network failures: timeouts, DNS errors, connection refused, etc."""


class PixazoAPIError(PixazoServiceError):
    """Raised for a Pixazo API failure (bad request, auth, model error, etc.)."""


class PixazoTimeoutError(PixazoServiceError):
    """Raised when the generation job never reaches COMPLETED within the poll budget."""


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class PixazoImageResult:
    """A normalized image-generation result."""

    image_url: str
    model: str


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class PixazoImageService:
    """
    Thin wrapper around the Pixazo image-generation API.

    Usage:
        service = PixazoImageService()
        result = service.generate_image("a lighthouse at dawn, watercolor")

    For tests, an httpx client can be injected directly:
        service = PixazoImageService(client=FakeClient())
    """

    def __init__(self, settings: Settings | None = None, client: Any | None = None):
        """
        Args:
            settings: Optional Settings override (mainly for tests). Defaults
                to the app-wide cached settings.
            client: Optional object exposing `.post(url, headers=, json=)` and
                `.get(url, headers=)` (mainly for tests, e.g. a fake or an
                `httpx.Client`). Defaults to a real `httpx.Client()` built
                lazily on first use.
        """
        settings = settings or get_settings()
        self.api_key = settings.PIXAZO_API_KEY
        self.model = settings.PIXAZO_IMAGE_MODEL
        self._client = client

    def is_configured(self) -> bool:
        """Whether a Pixazo API key is available."""
        return bool(self.api_key)

    def generate_image(self, prompt: str) -> PixazoImageResult:
        """
        Generate an image for the given prompt.

        Args:
            prompt: The text prompt describing the desired image.

        Returns:
            A `PixazoImageResult` with the generated image's URL.

        Raises:
            ValueError: if `prompt` is empty/blank.
            PixazoConfigurationError: if the API key is not configured.
            PixazoConnectionError: on network/timeout failures.
            PixazoAPIError: on any other Pixazo API failure.
            PixazoTimeoutError: if the job never completes within the poll budget.
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")

        if not self.is_configured():
            logger.error("Pixazo API key is not configured; cannot generate an image.")
            raise PixazoConfigurationError("Pixazo API key is not configured.")

        client = self._get_client()
        headers = {
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": self.api_key,
        }

        submit_url = f"{_BASE_URL}/{self.model}/v1/{self.model}-request"
        logger.info("Submitting Pixazo image generation job (model=%s)", self.model)

        try:
            submit_response = client.post(submit_url, headers=headers, json={"prompt": prompt})
        except Exception as exc:  # noqa: BLE001 - normalized immediately below
            raise self._classify_exception(exc) from exc

        submit_data = self._parse_json_response(submit_response, context="submit")
        request_id = submit_data.get("request_id")
        if not request_id:
            raise PixazoAPIError(
                f"Pixazo submit response did not include a request_id: {submit_data}"
            )

        return self._poll_for_result(client, headers, request_id)

    # -- internals ------------------------------------------------------

    def _poll_for_result(
        self, client: Any, headers: dict, request_id: str
    ) -> PixazoImageResult:
        """Poll the status endpoint until COMPLETED, FAILED/ERROR, or timeout."""
        status_url = f"{_BASE_URL}/v2/requests/status/{request_id}"

        for attempt in range(1, _MAX_POLL_ATTEMPTS + 1):
            try:
                status_response = client.get(status_url, headers=headers)
            except Exception as exc:  # noqa: BLE001 - normalized immediately below
                raise self._classify_exception(exc) from exc

            status_data = self._parse_json_response(status_response, context="status")
            status = status_data.get("status")

            if status == "COMPLETED":
                output = status_data.get("output") or {}
                media_urls = output.get("media_url") or []
                if not media_urls:
                    raise PixazoAPIError(
                        f"Pixazo job {request_id} completed with no media_url: {status_data}"
                    )
                logger.info("Pixazo job %s completed.", request_id)
                return PixazoImageResult(image_url=media_urls[0], model=self.model)

            if status in ("FAILED", "ERROR"):
                message = status_data.get("error") or status_data.get("message") or "unknown error"
                raise PixazoAPIError(f"Pixazo job {request_id} failed: {message}")

            logger.info(
                "Pixazo job %s still %s (attempt %d/%d)",
                request_id,
                status,
                attempt,
                _MAX_POLL_ATTEMPTS,
            )
            time.sleep(_POLL_INTERVAL_SECONDS)

        raise PixazoTimeoutError(
            f"Pixazo job {request_id} did not complete within "
            f"{_MAX_POLL_ATTEMPTS * _POLL_INTERVAL_SECONDS:.0f}s."
        )

    def _get_client(self) -> Any:
        """Lazily construct a real httpx.Client, unless one was injected."""
        if self._client is not None:
            return self._client
        self._client = httpx.Client(timeout=_REQUEST_TIMEOUT_SECONDS)
        return self._client

    def _parse_json_response(self, response: Any, context: str) -> dict:
        """Validate an HTTP response's status code and parse its JSON body."""
        status_code = getattr(response, "status_code", None)
        if status_code is not None and status_code >= 400:
            try:
                body = response.json()
                message = body.get("message") or body.get("error") or str(body)
            except Exception:  # noqa: BLE001
                message = getattr(response, "text", "") or f"HTTP {status_code}"
            raise PixazoAPIError(f"Pixazo {context} request failed ({status_code}): {message}")

        try:
            return response.json()
        except Exception as exc:  # noqa: BLE001
            raise PixazoAPIError(
                f"Pixazo {context} response was not valid JSON: {exc}"
            ) from exc

    def _classify_exception(self, exc: Exception) -> PixazoServiceError:
        """Normalize a raw httpx exception into a PixazoServiceError subtype."""
        message = str(exc).lower()
        type_name = type(exc).__name__.lower()

        if (
            "timeout" in message
            or "timeout" in type_name
            or "connect" in message
            or "connect" in type_name
        ):
            return PixazoConnectionError(f"Pixazo connection/timeout error: {exc}")

        return PixazoAPIError(f"Pixazo API error: {exc}")
