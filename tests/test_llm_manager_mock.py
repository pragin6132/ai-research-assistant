"""
Mock-based test script for app/services/llm_manager.py.

This verifies ONLY the fallback *decision logic* inside LLMManager.
No real network calls are made: each provider's `_call_api` method is
patched with a mock so no SDK needs to be installed or reach the network.

Run from backend/:
    python ../tests/test_llm_manager_mock.py
"""

import logging
import sys
from pathlib import Path
from unittest.mock import patch

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.core.config import Settings  # noqa: E402
from app.services.llm_manager import (  # noqa: E402
    AllProvidersFailedError,
    GeminiProvider,
    GroqProvider,
    LLMManager,
    LLMResponse,
    LLMRateLimitError,
    LLMUnexpectedError,
    NvidiaProvider,
)

PASSED = []
FAILED = []


def check(condition: bool, description: str):
    if condition:
        PASSED.append(description)
        print(f"  PASS: {description}")
    else:
        FAILED.append(description)
        print(f"  FAIL: {description}")


def dummy_settings() -> Settings:
    """Settings with plausible (but fake) keys for all three providers."""
    return Settings(
        _env_file=None,
        GEMINI_API_KEY="fake-gemini-key",
        NVIDIA_API_KEY="fake-nvidia-key",
        GROQ_API_KEY="fake-groq-key",
    )


def test_1_gemini_success_short_circuits():
    print("\n=== Test 1: Gemini success -> NVIDIA and Groq are NOT called ===")
    manager = LLMManager(settings=dummy_settings())

    with patch.object(
        GeminiProvider, "_call_api",
        return_value=LLMResponse(provider="Gemini", model="gemini-1.5-flash", text="hello"),
    ) as gemini_mock, \
         patch.object(NvidiaProvider, "_call_api") as nvidia_mock, \
         patch.object(GroqProvider, "_call_api") as groq_mock:

        response = manager.generate("test prompt")

        check(response.provider == "Gemini", "response came from Gemini")
        check(gemini_mock.call_count == 1, "Gemini was called exactly once")
        check(nvidia_mock.call_count == 0, "NVIDIA was NOT called")
        check(groq_mock.call_count == 0, "Groq was NOT called")


def test_2_gemini_429_falls_back_to_nvidia():
    print("\n=== Test 2: Gemini 429 -> NVIDIA is called ===")
    manager = LLMManager(settings=dummy_settings())

    with patch.object(
        GeminiProvider, "_call_api",
        side_effect=LLMRateLimitError("Gemini: rate limit / quota exceeded (429)."),
    ) as gemini_mock, \
         patch.object(
             NvidiaProvider, "_call_api",
             return_value=LLMResponse(provider="NVIDIA", model="llama-3.1-8b-instruct", text="hi"),
         ) as nvidia_mock, \
         patch.object(GroqProvider, "_call_api") as groq_mock:

        response = manager.generate("test prompt")

        check(gemini_mock.call_count == 1, "Gemini was tried once")
        check(nvidia_mock.call_count == 1, "NVIDIA was called after Gemini's 429")
        check(response.provider == "NVIDIA", "response came from NVIDIA")
        check(groq_mock.call_count == 0, "Groq was NOT called")


def test_3_gemini_429_and_nvidia_failure_falls_back_to_groq():
    print("\n=== Test 3: Gemini 429 + NVIDIA failure -> Groq is called ===")
    manager = LLMManager(settings=dummy_settings())

    with patch.object(
        GeminiProvider, "_call_api",
        side_effect=LLMRateLimitError("Gemini: rate limit / quota exceeded (429)."),
    ) as gemini_mock, \
         patch.object(
             NvidiaProvider, "_call_api",
             side_effect=LLMUnexpectedError("NVIDIA: unexpected error (503)."),
         ) as nvidia_mock, \
         patch.object(
             GroqProvider, "_call_api",
             return_value=LLMResponse(provider="Groq", model="llama-3.1-8b-instant", text="hey"),
         ) as groq_mock:

        response = manager.generate("test prompt")

        check(gemini_mock.call_count == 1, "Gemini was tried once")
        check(nvidia_mock.call_count == 1, "NVIDIA was tried once")
        check(groq_mock.call_count == 1, "Groq was called after NVIDIA's failure")
        check(response.provider == "Groq", "response came from Groq")


def test_4_new_request_restarts_at_gemini():
    print("\n=== Test 4: A new request after a fallback starts with Gemini again ===")
    manager = LLMManager(settings=dummy_settings())

    # First call: Gemini fails with 429, NVIDIA succeeds.
    # Second call: Gemini succeeds directly (simulating the 429 was transient).
    gemini_side_effects = [
        LLMRateLimitError("Gemini: rate limit / quota exceeded (429)."),
        LLMResponse(provider="Gemini", model="gemini-1.5-flash", text="back to normal"),
    ]

    with patch.object(GeminiProvider, "_call_api", side_effect=gemini_side_effects) as gemini_mock, \
         patch.object(
             NvidiaProvider, "_call_api",
             return_value=LLMResponse(provider="NVIDIA", model="llama-3.1-8b-instruct", text="hi"),
         ) as nvidia_mock, \
         patch.object(GroqProvider, "_call_api") as groq_mock:

        first_response = manager.generate("request 1")
        check(first_response.provider == "NVIDIA", "request 1 fell back to NVIDIA")
        check(gemini_mock.call_count == 1, "Gemini was tried once on request 1")
        check(nvidia_mock.call_count == 1, "NVIDIA was called once on request 1")

        second_response = manager.generate("request 2")
        check(gemini_mock.call_count == 2, "Gemini was tried AGAIN on request 2 (not skipped)")
        check(second_response.provider == "Gemini", "request 2 was served by Gemini directly")
        check(nvidia_mock.call_count == 1, "NVIDIA was NOT called again on request 2")
        check(groq_mock.call_count == 0, "Groq was never called in this scenario")


def test_5_missing_keys_handled_clearly():
    print("\n=== Test 5: Missing keys are handled clearly ===")
    empty_settings = Settings(
        _env_file=None,
        GEMINI_API_KEY=None,
        NVIDIA_API_KEY=None,
        GROQ_API_KEY=None,
    )
    manager = LLMManager(settings=empty_settings)

    with patch.object(GeminiProvider, "_call_api") as gemini_mock, \
         patch.object(NvidiaProvider, "_call_api") as nvidia_mock, \
         patch.object(GroqProvider, "_call_api") as groq_mock:

        raised = False
        error_message = ""
        try:
            manager.generate("test prompt")
        except AllProvidersFailedError as exc:
            raised = True
            error_message = str(exc)

        check(raised, "AllProvidersFailedError was raised when no keys are configured")
        check("Gemini" in error_message, "error message mentions Gemini")
        check("NVIDIA" in error_message, "error message mentions NVIDIA")
        check("Groq" in error_message, "error message mentions Groq")
        check(gemini_mock.call_count == 0, "Gemini API was never actually called (no key)")
        check(nvidia_mock.call_count == 0, "NVIDIA API was never actually called (no key)")
        check(groq_mock.call_count == 0, "Groq API was never actually called (no key)")


if __name__ == "__main__":
    test_1_gemini_success_short_circuits()
    test_2_gemini_429_falls_back_to_nvidia()
    test_3_gemini_429_and_nvidia_failure_falls_back_to_groq()
    test_4_new_request_restarts_at_gemini()
    test_5_missing_keys_handled_clearly()

    print("\n=== Summary ===")
    print(f"Passed: {len(PASSED)}")
    print(f"Failed: {len(FAILED)}")

    if FAILED:
        print("\nFailed checks:")
        for item in FAILED:
            print(f"  - {item}")
        sys.exit(1)
    else:
        print("All fallback-logic tests passed.")
