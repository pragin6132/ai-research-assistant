"""
Mock-based test script for app/agents/summarizer.py.

No real LLM/network calls are made: a fake LLM manager double is
injected directly into `Summarizer`, simulating `LLMManager.generate()`.

Run from backend/:
    python ../tests/test_summarizer_mock.py
"""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.agents.summarizer import (  # noqa: E402
    Summarizer,
    SummarizerLLMError,
    SummarizerParsingError,
)
from app.schemas.models import Finding, SearchResult  # noqa: E402
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


def _sample_results() -> list[SearchResult]:
    return [
        SearchResult(
            title="Solid-State Battery Materials",
            url="https://example.com/materials",
            content="Solid-state batteries use ceramic or polymer electrolytes...",
            relevance_score=0.9,
        ),
        SearchResult(
            title="Manufacturing Challenges",
            url="https://example.com/manufacturing",
            content="Scaling solid-state battery production remains costly...",
            relevance_score=0.8,
        ),
    ]


def test_1_successful_summarization():
    print("\n=== Test 1: successful summarization produces valid Finding objects ===")
    fake_json = json.dumps(
        {
            "findings": [
                {
                    "topic": "Solid-state battery materials",
                    "summary": "Ceramic and polymer electrolytes are the main materials being explored.",
                    "key_points": ["Ceramic electrolytes offer high conductivity", "Polymer electrolytes are cheaper"],
                    "sources": ["https://example.com/materials"],
                },
                {
                    "topic": "Manufacturing challenges",
                    "summary": "Scaling production is currently costly and complex.",
                    "key_points": ["High capital costs", "Difficult to scale beyond pilot lines"],
                    "sources": ["https://example.com/manufacturing"],
                },
            ]
        }
    )
    manager = FakeLLMManager(text=fake_json)
    summarizer = Summarizer(llm_manager=manager)

    findings = summarizer.summarize(
        _sample_results(), main_topic="Solid-state batteries", sub_topics=["Materials", "Manufacturing"]
    )

    check(len(findings) == 2, "two findings were produced")
    check(all(isinstance(f, Finding) for f in findings), "all findings are Finding instances")
    check(findings[0].topic == "Solid-state battery materials", "first finding topic preserved")
    check(str(findings[0].sources[0]) == "https://example.com/materials", "source URL preserved exactly, not fabricated")
    check(len(manager.calls) == 1, "the LLM was called exactly once")
    check("Solid-state batteries" in manager.calls[0], "the prompt includes the main topic for context")
    check("https://example.com/manufacturing" in manager.calls[0], "the prompt includes the actual source URLs")


def test_2_malformed_llm_output():
    print("\n=== Test 2: malformed LLM output raises SummarizerParsingError ===")

    # Case A: not valid JSON at all.
    manager_a = FakeLLMManager(text="Here is a summary of the findings: ...")
    summarizer_a = Summarizer(llm_manager=manager_a)
    raised_a = False
    try:
        summarizer_a.summarize(_sample_results())
    except SummarizerParsingError:
        raised_a = True
    check(raised_a, "non-JSON LLM output raises SummarizerParsingError")

    # Case B: valid JSON but findings entries missing required fields.
    manager_b = FakeLLMManager(text=json.dumps({"findings": [{"topic": "Missing fields"}]}))
    summarizer_b = Summarizer(llm_manager=manager_b)
    raised_b = False
    try:
        summarizer_b.summarize(_sample_results())
    except SummarizerParsingError:
        raised_b = True
    check(raised_b, "findings missing required fields raise SummarizerParsingError")


def test_3_llm_failure():
    print("\n=== Test 3: LLM failure raises SummarizerLLMError ===")
    manager = FakeLLMManager(
        exception=AllProvidersFailedError(
            "All providers failed for this request: Gemini: ... | NVIDIA: ... | Groq: ..."
        )
    )
    summarizer = Summarizer(llm_manager=manager)

    raised = False
    try:
        summarizer.summarize(_sample_results())
    except SummarizerLLMError:
        raised = True

    check(raised, "SummarizerLLMError was raised when the LLM manager fails")
    check(len(manager.calls) == 1, "the LLM was invoked exactly once")


def test_4_empty_search_results_skips_llm_call():
    print("\n=== Test 4: empty search_results returns [] without calling the LLM ===")
    manager = FakeLLMManager(text="should never be used")
    summarizer = Summarizer(llm_manager=manager)

    findings = summarizer.summarize([])

    check(findings == [], "an empty list of findings is returned for empty input")
    check(manager.calls == [], "the LLM was never called for empty search_results")


if __name__ == "__main__":
    test_1_successful_summarization()
    test_2_malformed_llm_output()
    test_3_llm_failure()
    test_4_empty_search_results_skips_llm_call()

    print("\n=== Summary ===")
    print(f"Passed: {len(PASSED)}")
    print(f"Failed: {len(FAILED)}")

    if FAILED:
        print("\nFailed checks:")
        for item in FAILED:
            print(f"  - {item}")
        sys.exit(1)
    else:
        print("All Summarizer tests passed.")
