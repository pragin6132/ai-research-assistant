"""
Tavily web search service.

Wraps the Tavily search API behind a small, dependency-injectable class
that returns structured `SearchResult` objects (see app/schemas/models.py).

This service is intentionally independent of the LLM Manager and of any
future LangGraph orchestration — it does one thing: given a query string,
return structured search results (or raise a clear error).
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.schemas.models import SearchResult

logger = logging.getLogger("app.tavily_service")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TavilyServiceError(Exception):
    """Base class for all Tavily service errors."""


class TavilyConfigurationError(TavilyServiceError):
    """Raised when the Tavily API key is missing/not configured."""


class TavilyConnectionError(TavilyServiceError):
    """Raised on network failures: timeouts, DNS errors, connection refused, etc."""


class TavilyAPIError(TavilyServiceError):
    """Raised for any other Tavily API failure (bad request, auth, server error, etc.)."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class TavilySearchService:
    """
    Thin wrapper around the Tavily search API.

    Usage:
        service = TavilySearchService()
        results = service.search("latest advances in RAG")

    For tests, a fake client can be injected directly:
        service = TavilySearchService(client=FakeTavilyClient())
    """

    def __init__(self, settings: Settings | None = None, client: Any | None = None):
        """
        Args:
            settings: Optional Settings override (mainly for tests). Defaults
                to the app-wide cached settings.
            client: Optional pre-built Tavily client (mainly for tests).
                Defaults to lazily constructing a real `TavilyClient` on
                first use.
        """
        settings = settings or get_settings()
        self.api_key = settings.TAVILY_API_KEY
        self.default_max_results = settings.TAVILY_MAX_RESULTS
        self.default_search_depth = settings.TAVILY_SEARCH_DEPTH
        self._client = client

    def is_configured(self) -> bool:
        """Whether a Tavily API key is available."""
        return bool(self.api_key)

    def search(
        self,
        query: str,
        max_results: int | None = None,
        search_depth: str | None = None,
    ) -> list[SearchResult]:
        """
        Run a Tavily search for the given query and return structured results.

        Args:
            query: The search query text. Never hardcoded — always supplied
                by the caller (e.g. a future query planner).
            max_results: Optional override for this call; defaults to
                `TAVILY_MAX_RESULTS` from settings.
            search_depth: Optional override for this call ("basic" or
                "advanced"); defaults to `TAVILY_SEARCH_DEPTH` from settings.

        Returns:
            A list of `SearchResult` objects. Empty list if Tavily returns
            no results — this is not treated as an error.

        Raises:
            ValueError: if `query` is empty/blank.
            TavilyConfigurationError: if the API key is not configured.
            TavilyConnectionError: on network/timeout failures.
            TavilyAPIError: on any other Tavily API failure.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")

        if not self.is_configured():
            logger.error("Tavily API key is not configured; cannot run search.")
            raise TavilyConfigurationError("Tavily API key is not configured.")

        client = self._get_client()

        effective_max_results = max_results or self.default_max_results
        effective_search_depth = search_depth or self.default_search_depth

        logger.info(
            "Searching Tavily for query=%r (max_results=%s, search_depth=%s)",
            query,
            effective_max_results,
            effective_search_depth,
        )

        try:
            raw_response = client.search(
                query=query,
                max_results=effective_max_results,
                search_depth=effective_search_depth,
            )
        except Exception as exc:  # noqa: BLE001 - normalized immediately below
            classified = self._classify_exception(exc)
            logger.error("Tavily search failed for query=%r: %s", query, classified)
            raise classified from exc

        return self._parse_results(raw_response, query)

    # -- internals ----------------------------------------------------------

    def _get_client(self) -> Any:
        """Lazily construct the real Tavily client, unless one was injected."""
        if self._client is not None:
            return self._client

        try:
            from tavily import TavilyClient
        except ImportError as exc:
            raise TavilyAPIError(
                f"Tavily SDK 'tavily-python' is not installed ({exc})."
            ) from exc

        self._client = TavilyClient(api_key=self.api_key)
        return self._client

    def _parse_results(self, raw_response: Any, query: str) -> list[SearchResult]:
        """
        Turn Tavily's raw response into a list of `SearchResult`.

        Never fabricates data: any item missing a usable title/url/content
        is skipped (and logged) rather than filled in with placeholders.
        """
        raw_results = []
        if isinstance(raw_response, dict):
            raw_results = raw_response.get("results") or []
        elif isinstance(raw_response, list):
            raw_results = raw_response

        if not raw_results:
            logger.warning("Tavily returned no results for query=%r", query)
            return []

        results: list[SearchResult] = []
        for item in raw_results:
            try:
                results.append(
                    SearchResult(
                        title=item.get("title") or "Untitled",
                        url=item["url"],
                        content=item.get("content") or "",
                        relevance_score=item.get("score"),
                    )
                )
            except (KeyError, ValidationError) as exc:
                logger.warning("Skipping malformed Tavily result %r: %s", item, exc)
                continue

        logger.info(
            "Tavily returned %d structured result(s) for query=%r", len(results), query
        )
        return results

    def _classify_exception(self, exc: Exception) -> TavilyServiceError:
        """Normalize a raw SDK/HTTP exception into a TavilyServiceError subtype."""
        message = str(exc).lower()
        type_name = type(exc).__name__.lower()

        if (
            "timeout" in message
            or "timeout" in type_name
            or "connection" in message
            or "connection" in type_name
        ):
            return TavilyConnectionError(f"Tavily connection/timeout error: {exc}")

        return TavilyAPIError(f"Tavily API error: {exc}")
