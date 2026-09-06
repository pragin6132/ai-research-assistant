"""
Test for the Settings UI's "Inbuilt API Key" On/Off configuration flow
(app/api/research.py's `_build_llm_manager` and `_redact` helpers).

No real LLM/network calls are made; no real API keys are required.

Run from backend/:
    python ../tests/test_research_api_key_override.py
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.api.research import _build_llm_manager, _redact  # noqa: E402
from app.core.config import Settings  # noqa: E402

PASSED = []
FAILED = []


def check(condition: bool, description: str):
    if condition:
        PASSED.append(description)
        print(f"  PASS: {description}")
    else:
        FAILED.append(description)
        print(f"  FAIL: {description}")


def _providers_by_name(manager):
    return {p.name: p for p in manager._providers}


def test_1_no_overrides_uses_app_settings():
    print("\n=== Test 1: no custom_api_keys -> every provider uses the app's own (inbuilt) key ===")
    manager = _build_llm_manager(None)
    providers = _providers_by_name(manager)
    app_settings = Settings(_env_file=None)

    check(
        providers["Gemini"].api_key == app_settings.GEMINI_API_KEY,
        "Gemini uses the app's configured key when no override is given",
    )
    check(
        providers["NVIDIA"].api_key == app_settings.NVIDIA_API_KEY,
        "NVIDIA uses the app's configured key when no override is given",
    )
    check(
        providers["Groq"].api_key == app_settings.GROQ_API_KEY,
        "Groq uses the app's configured key when no override is given",
    )


def test_2_single_provider_override_does_not_affect_others():
    print("\n=== Test 2: overriding one provider's key leaves the others on the inbuilt key ===")
    manager = _build_llm_manager({"gemini": "user-secret-gemini-key"})
    providers = _providers_by_name(manager)
    app_settings = Settings(_env_file=None)

    check(providers["Gemini"].api_key == "user-secret-gemini-key", "Gemini uses the user-supplied override key")
    check(
        providers["NVIDIA"].api_key == app_settings.NVIDIA_API_KEY,
        "NVIDIA is unaffected and still uses the app's inbuilt key",
    )
    check(
        providers["Groq"].api_key == app_settings.GROQ_API_KEY,
        "Groq is unaffected and still uses the app's inbuilt key",
    )
    check(
        manager._providers[0].name == "Gemini"
        and manager._providers[1].name == "NVIDIA"
        and manager._providers[2].name == "Groq",
        "fallback order is still Gemini -> NVIDIA -> Groq, unchanged",
    )


def test_3_all_three_providers_overridden():
    print("\n=== Test 3: all three providers can be independently overridden at once ===")
    manager = _build_llm_manager(
        {"gemini": "g-key", "nvidia": "n-key", "groq": "q-key"}
    )
    providers = _providers_by_name(manager)

    check(providers["Gemini"].api_key == "g-key", "Gemini override applied")
    check(providers["NVIDIA"].api_key == "n-key", "NVIDIA override applied")
    check(providers["Groq"].api_key == "q-key", "Groq override applied")


def test_4_empty_string_override_is_ignored():
    print("\n=== Test 4: an empty-string override falls back to the app's inbuilt key ===")
    manager = _build_llm_manager({"gemini": ""})
    providers = _providers_by_name(manager)
    app_settings = Settings(_env_file=None)

    check(
        providers["Gemini"].api_key == app_settings.GEMINI_API_KEY,
        "an empty override value does not blank out the inbuilt key",
    )


def test_5_redaction_strips_key_values():
    print("\n=== Test 5: _redact() removes known secret values from text ===")
    message = "Gemini rejected key user-secret-gemini-key: invalid credential"
    redacted = _redact(message, ["user-secret-gemini-key"])

    check("user-secret-gemini-key" not in redacted, "the raw key value no longer appears in the text")
    check("[redacted]" in redacted, "a placeholder marks where the key was")
    check(_redact("no secrets here", []) == "no secrets here", "text with no secrets list is left unchanged")


if __name__ == "__main__":
    test_1_no_overrides_uses_app_settings()
    test_2_single_provider_override_does_not_affect_others()
    test_3_all_three_providers_overridden()
    test_4_empty_string_override_is_ignored()
    test_5_redaction_strips_key_values()

    print("\n=== Summary ===")
    print(f"Passed: {len(PASSED)}")
    print(f"Failed: {len(FAILED)}")

    if FAILED:
        print("\nFailed checks:")
        for item in FAILED:
            print(f"  - {item}")
        sys.exit(1)
    else:
        print("All API-key override / redaction tests passed.")
