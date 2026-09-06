"""
LLM Manager: a single entry point for calling LLM providers with automatic
fallback.

Fallback order for EVERY independent request:
    1. Gemini  (primary)
    2. NVIDIA  (first fallback)
    3. Groq    (second fallback)

Design notes:
- Each call to `LLMManager.generate()` is fully independent: it always
  starts by trying Gemini. A fallback taken during one request never
  changes the primary provider for the next request.
- The rest of the application should only ever call `LLMManager.generate()`
  and shouldn't need to know which provider actually served the request
  (that's available on the returned `LLMResponse.provider` if needed).
- Provider-specific SDK calls and error handling are isolated inside each
  provider class. `LLMManager` only reasons about the common `LLMError`
  subtypes, never about a provider's raw SDK exceptions.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.core.config import Settings, get_settings

logger = logging.getLogger("app.llm_manager")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Base class for all LLM-related errors."""


class LLMConfigurationError(LLMError):
    """Raised when a provider is missing required configuration (e.g. API key)."""


class LLMAuthenticationError(LLMError):
    """Raised when a provider rejects credentials (invalid/expired API key)."""


class LLMRateLimitError(LLMError):
    """Raised on HTTP 429 / quota-exceeded / rate-limit responses."""


class LLMConnectionError(LLMError):
    """Raised on network failures: timeouts, DNS errors, connection refused, etc."""


class LLMUnexpectedError(LLMError):
    """Raised for any other provider failure that doesn't fit the categories above."""


class AllProvidersFailedError(LLMError):
    """Raised when every provider in the fallback chain has failed for a request."""


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """A normalized response, regardless of which provider produced it."""

    provider: str
    model: str
    text: str


# ---------------------------------------------------------------------------
# Shared error classification helper
# ---------------------------------------------------------------------------


def classify_provider_exception(provider_name: str, exc: Exception) -> LLMError:
    """
    Turn a raw SDK/HTTP exception into one of our normalized LLMError types.

    This is intentionally generic (status-code / keyword based) so it works
    across different provider SDKs without each provider needing to know
    every exception class the others raise.
    """
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None) if response else None

    message = str(exc).lower()
    type_name = type(exc).__name__.lower()

    if (
        status_code == 429
        or "429" in message
        or "rate limit" in message
        or "rate_limit" in type_name
        or "quota" in message
        or "resourceexhausted" in type_name
    ):
        return LLMRateLimitError(f"{provider_name}: rate limit / quota exceeded ({exc}).")

    if (
        status_code in (401, 403)
        or "unauthorized" in message
        or "invalid api key" in message
        or "authentication" in type_name
        or "permissiondenied" in type_name
    ):
        return LLMAuthenticationError(f"{provider_name}: authentication failed ({exc}).")

    if (
        "timeout" in message
        or "timeout" in type_name
        or "connection" in message
        or "connection" in type_name
    ):
        return LLMConnectionError(f"{provider_name}: connection/timeout error ({exc}).")

    return LLMUnexpectedError(f"{provider_name}: unexpected error ({exc}).")


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------


class BaseLLMProvider(ABC):
    """
    Common interface every provider must implement.

    Subclasses only need to implement `_call_api`, which should return an
    `LLMResponse` on success and raise an `LLMError` subclass on failure.
    """

    name: str = "base"

    def __init__(self, api_key: str | None, model_name: str):
        self.api_key = api_key
        self.model_name = model_name

    def is_configured(self) -> bool:
        """Whether this provider has the minimum configuration to be tried."""
        return bool(self.api_key)

    def generate(self, prompt: str) -> LLMResponse:
        """Public entry point used by LLMManager. Enforces the config check."""
        if not self.is_configured():
            raise LLMConfigurationError(f"{self.name}: API key is not configured.")
        return self._call_api(prompt)

    @abstractmethod
    def _call_api(self, prompt: str) -> LLMResponse:
        """Provider-specific call. Must raise an LLMError subclass on failure."""
        raise NotImplementedError


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider (primary)."""

    name = "Gemini"

    def _call_api(self, prompt: str) -> LLMResponse:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise LLMUnexpectedError(
                f"{self.name}: SDK 'google-generativeai' is not installed ({exc})."
            ) from exc

        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model_name)
            result = model.generate_content(prompt)
            text = getattr(result, "text", "") or ""
            return LLMResponse(provider=self.name, model=self.model_name, text=text)
        except Exception as exc:  # noqa: BLE001 - normalized immediately below
            raise classify_provider_exception(self.name, exc) from exc


class NvidiaProvider(BaseLLMProvider):
    """NVIDIA NIM provider (first fallback). Uses NVIDIA's OpenAI-compatible API."""

    name = "NVIDIA"
    BASE_URL = "https://integrate.api.nvidia.com/v1"

    def _call_api(self, prompt: str) -> LLMResponse:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMUnexpectedError(
                f"{self.name}: SDK 'openai' is not installed ({exc})."
            ) from exc

        try:
            client = OpenAI(api_key=self.api_key, base_url=self.BASE_URL)
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            text = completion.choices[0].message.content or ""
            return LLMResponse(provider=self.name, model=self.model_name, text=text)
        except Exception as exc:  # noqa: BLE001 - normalized immediately below
            raise classify_provider_exception(self.name, exc) from exc


class GroqProvider(BaseLLMProvider):
    """Groq provider (second fallback)."""

    name = "Groq"

    def _call_api(self, prompt: str) -> LLMResponse:
        try:
            from groq import Groq
        except ImportError as exc:
            raise LLMUnexpectedError(
                f"{self.name}: SDK 'groq' is not installed ({exc})."
            ) from exc

        try:
            client = Groq(api_key=self.api_key)
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            text = completion.choices[0].message.content or ""
            return LLMResponse(provider=self.name, model=self.model_name, text=text)
        except Exception as exc:  # noqa: BLE001 - normalized immediately below
            raise classify_provider_exception(self.name, exc) from exc


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class LLMManager:
    """
    Orchestrates the Gemini -> NVIDIA -> Groq fallback chain for one request.

    Every call to `generate()` is independent and always starts with Gemini.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        providers: list[BaseLLMProvider] | None = None,
    ):
        """
        Args:
            settings: Optional Settings override (mainly for tests). Defaults
                to the app-wide cached settings.
            providers: Optional explicit provider list, in fallback order
                (mainly for tests). Defaults to the standard
                Gemini -> NVIDIA -> Groq chain built from `settings`.
        """
        settings = settings or get_settings()
        self._providers: list[BaseLLMProvider] = providers or [
            GeminiProvider(settings.GEMINI_API_KEY, settings.GEMINI_MODEL_NAME),
            NvidiaProvider(settings.NVIDIA_API_KEY, settings.NVIDIA_MODEL_NAME),
            GroqProvider(settings.GROQ_API_KEY, settings.GROQ_MODEL_NAME),
        ]

    def generate(self, prompt: str) -> LLMResponse:
        """
        Run the fallback chain for a single request, always starting at
        Gemini. Returns the first successful response, or raises
        `AllProvidersFailedError` if every provider in the chain fails.
        """
        errors: list[str] = []

        for index, provider in enumerate(self._providers):
            logger.info("Trying %s", provider.name)

            try:
                response = provider.generate(prompt)
            except LLMConfigurationError as exc:
                logger.warning("%s is not configured, skipping. (%s)", provider.name, exc)
                errors.append(str(exc))
                continue
            except LLMRateLimitError as exc:
                errors.append(str(exc))
                if index + 1 < len(self._providers):
                    next_provider = self._providers[index + 1].name
                    logger.warning(
                        "%s failed with 429, falling back to %s", provider.name, next_provider
                    )
                else:
                    logger.warning("%s failed with 429. No more providers to fall back to.", provider.name)
                continue
            except (LLMAuthenticationError, LLMConnectionError, LLMUnexpectedError) as exc:
                errors.append(str(exc))
                if index + 1 < len(self._providers):
                    next_provider = self._providers[index + 1].name
                    logger.warning(
                        "%s failed (%s), falling back to %s",
                        provider.name,
                        type(exc).__name__,
                        next_provider,
                    )
                else:
                    logger.warning("%s failed (%s). No more providers to fall back to.", provider.name, type(exc).__name__)
                continue
            else:
                logger.info("%s succeeded", provider.name)
                return response

        message = "All providers failed for this request: " + " | ".join(errors)
        logger.error(message)
        raise AllProvidersFailedError(message)
