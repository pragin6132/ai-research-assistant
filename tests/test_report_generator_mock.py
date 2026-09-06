"""
Mock-based test script for app/agents/report_generator.py.

No real LLM/network calls are made: a fake LLM manager double is
injected directly into `ReportGenerator`, simulating `LLMManager.generate()`.

Run from backend/:
    python ../tests/test_report_generator_mock.py
"""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.agents.report_generator import (  # noqa: E402
    ReportGenerator,
    ReportGeneratorLLMError,
    ReportGeneratorParsingError,
)
from app.graph.state import detect_response_language  # noqa: E402
from app.schemas.models import Finding, ResearchReport  # noqa: E402
from app.services.llm_manager import AllProvidersFailedError, LLMResponse  # noqa: E402

PASSED = []
FAILED = []


def check(condition: bool, description: str):
    if condition:
        PASSED.append(description)
        print(f"  PASS: {description}")
    else:
        FAILED.append(description)
        print(f"  FAIL: {description}")


class FakeLLMManager:
    """A stand-in for `LLMManager`, exposing only `.generate(prompt)`."""

    def __init__(self, text: str | None = None, exception: Exception | None = None):
        self.text = text
        self.exception = exception
        self.calls: list[str] = []

    def generate(self, prompt: str) -> LLMResponse:
        self.calls.append(prompt)
        if self.exception is not None:
            raise self.exception
        return LLMResponse(provider="Gemini", model="gemini-1.5-flash", text=self.text)


def _sample_findings() -> list[Finding]:
    return [
        Finding(
            topic="Battery materials",
            summary="Ceramic and polymer electrolytes are the main approaches.",
            key_points=["Ceramic offers high conductivity", "Polymer is cheaper"],
            sources=["https://example.com/materials"],
        ),
        Finding(
            topic="Manufacturing challenges",
            summary="Scaling production remains costly and complex.",
            key_points=["High capital costs"],
            sources=["https://example.com/manufacturing", "https://example.com/materials"],
        ),
    ]


def test_1_successful_report_generation():
    print("\n=== Test 1: successful generation produces a valid ResearchReport ===")
    fake_json = json.dumps(
        {
            "title": "Solid-State Battery Research Overview",
            "executive_summary": "Solid-state batteries show promise but face manufacturing hurdles.",
            "key_findings": [
                "Ceramic electrolytes offer the highest conductivity",
                "Manufacturing at scale remains costly",
            ],
        }
    )
    manager = FakeLLMManager(text=fake_json)
    generator = ReportGenerator(llm_manager=manager)
    findings = _sample_findings()

    report = generator.generate_report(findings, main_topic="Solid-state batteries")

    check(isinstance(report, ResearchReport), "a ResearchReport instance was returned")
    check(report.title == "Solid-State Battery Research Overview", "title from the LLM narrative was used")
    check(len(report.key_findings) == 2, "key_findings from the LLM narrative were used")
    check(report.findings == findings, "the original Finding objects are preserved verbatim, not regenerated")
    check(
        [str(u) for u in report.sources] == ["https://example.com/materials", "https://example.com/manufacturing"],
        "sources are deduplicated across findings, preserving first-seen order, and never fabricated",
    )
    check(len(manager.calls) == 1, "the LLM was called exactly once")


def test_1b_report_language_instruction():
    print("\n=== Test 1B: report narrative follows the requested language ===")
    tamil_json = json.dumps(
        {
            "title": "திடநிலை பேட்டரி ஆராய்ச்சி சுருக்கம்",
            "executive_summary": "திடநிலை பேட்டரிகள் நல்ல திறனை வழங்குகின்றன.",
            "key_findings": ["செராமிக் எலக்ட்ரோலைட்டுகள் அதிக கடத்துத்திறன் வழங்குகின்றன."],
        }
    )
    manager = FakeLLMManager(text=tamil_json)
    report = ReportGenerator(llm_manager=manager).generate_report(
        _sample_findings(), main_topic="திடநிலை பேட்டரிகள்", response_language="Tamil"
    )

    check("REPORT LANGUAGE: Tamil" in manager.calls[0], "Tamil report language is included in the LLM prompt")
    check(_contains_tamil(report.title), "the Tamil narrative returned by the LLM is preserved in the final report")
    check(
        detect_response_language("மின்பேருந்துகளின் பயன்கள் என்ன?") == "Tamil",
        "Tamil-script input is detected as Tamil",
    )
    check(
        detect_response_language("Research electric vehicle battery recycling.") == "English",
        "English input is detected as English",
    )
    check(
        detect_response_language("EV battery பற்றி explain செய்யுங்கள்") == "Tanglish / Tamil-English mixed",
        "Tamil-English mixed input is detected as Tanglish",
    )


def test_2_malformed_llm_output():
    print("\n=== Test 2: malformed LLM output raises ReportGeneratorParsingError ===")

    # Case A: not valid JSON at all.
    manager_a = FakeLLMManager(text="Here is your report: ...")
    generator_a = ReportGenerator(llm_manager=manager_a)
    raised_a = False
    try:
        generator_a.generate_report(_sample_findings())
    except ReportGeneratorParsingError:
        raised_a = True
    check(raised_a, "non-JSON LLM output raises ReportGeneratorParsingError")

    # Case B: valid JSON but missing required narrative fields.
    manager_b = FakeLLMManager(text=json.dumps({"key_findings": ["only this"]}))
    generator_b = ReportGenerator(llm_manager=manager_b)
    raised_b = False
    try:
        generator_b.generate_report(_sample_findings())
    except ReportGeneratorParsingError:
        raised_b = True
    check(raised_b, "missing title/executive_summary raises ReportGeneratorParsingError")


def test_3_llm_failure():
    print("\n=== Test 3: LLM failure raises ReportGeneratorLLMError ===")
    manager = FakeLLMManager(
        exception=AllProvidersFailedError(
            "All providers failed for this request: Gemini: ... | NVIDIA: ... | Groq: ..."
        )
    )
    generator = ReportGenerator(llm_manager=manager)

    raised = False
    try:
        generator.generate_report(_sample_findings())
    except ReportGeneratorLLMError:
        raised = True

    check(raised, "ReportGeneratorLLMError was raised when the LLM manager fails")
    check(len(manager.calls) == 1, "the LLM was invoked exactly once")


def test_4_empty_findings_raises_value_error():
    print("\n=== Test 4: empty validated_findings raises ValueError without calling the LLM ===")
    manager = FakeLLMManager(text="should never be used")
    generator = ReportGenerator(llm_manager=manager)

    raised = False
    try:
        generator.generate_report([])
    except ValueError:
        raised = True

    check(raised, "ValueError was raised for empty validated_findings")
    check(manager.calls == [], "the LLM was never called for empty input")


def _contains_tamil(text: str) -> bool:
    return any("\u0B80" <= char <= "\u0BFF" for char in text)


if __name__ == "__main__":
    test_1_successful_report_generation()
    test_1b_report_language_instruction()
    test_2_malformed_llm_output()
    test_3_llm_failure()
    test_4_empty_findings_raises_value_error()

    print("\n=== Summary ===")
    print(f"Passed: {len(PASSED)}")
    print(f"Failed: {len(FAILED)}")

    if FAILED:
        print("\nFailed checks:")
        for item in FAILED:
            print(f"  - {item}")
        sys.exit(1)
    else:
        print("All ReportGenerator tests passed.")
