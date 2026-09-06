"""
Mock-based test script for app/agents/query_planner.py.

No real LLM/API calls are made: a fake LLM manager double is injected
directly into `QueryPlanner`, simulating `LLMManager.generate()`'s
behavior (both successful responses and failures).

Run from backend/:
    python ../tests/test_query_planner_mock.py
"""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.agents.query_planner import (  # noqa: E402
    QueryPlanner,
    QueryPlannerLLMError,
    QueryPlannerParsingError,
)
from app.schemas.models import ResearchPlan  # noqa: E402
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
    """
    A stand-in for `LLMManager`. Exposes the same `.generate(prompt)`
    interface the QueryPlanner relies on, without touching any real
    provider or network.
    """

    def __init__(self, text: str | None = None, exception: Exception | None = None):
        self.text = text
        self.exception = exception
        self.calls: list[str] = []

    def generate(self, prompt: str) -> LLMResponse:
        self.calls.append(prompt)
        if self.exception is not None:
            raise self.exception
        return LLMResponse(provider="Gemini", model="gemini-1.5-flash", text=self.text)


def test_1_short_research_prompt():
    print("\n=== Test 1: Short research prompt produces a valid ResearchPlan ===")
    fake_json = json.dumps(
        {
            "main_topic": "Electric vehicle battery recycling",
            "sub_topics": ["Recycling methods", "Environmental impact"],
            "search_queries": [
                {"query": "EV battery recycling methods 2026", "purpose": "Find current recycling techniques"},
                {"query": "environmental impact of EV battery recycling", "purpose": "Understand environmental trade-offs"},
            ],
        }
    )
    manager = FakeLLMManager(text=fake_json)
    planner = QueryPlanner(llm_manager=manager)

    prompt = "Tell me about EV battery recycling."
    plan = planner.create_plan(prompt)

    check(isinstance(plan, ResearchPlan), "a ResearchPlan instance was returned")
    check(plan.original_prompt == prompt, "original_prompt is preserved exactly")
    check(plan.main_topic == "Electric vehicle battery recycling", "main_topic was extracted")
    check(len(plan.sub_topics) == 2, "sub_topics were extracted")
    check(len(plan.search_queries) == 2, "search_queries were extracted")
    check(
        all(q.query != prompt for q in plan.search_queries),
        "no search query is just the raw prompt repeated back",
    )
    check(manager.calls[0] != prompt, "the prompt sent to the LLM includes planning instructions, not just the raw prompt")


def test_2_long_research_prompt_multiple_focused_queries():
    print("\n=== Test 2: Long research prompt produces multiple focused queries ===")
    long_prompt = (
        "I'm trying to understand the current landscape of large language model "
        "agents used in enterprise software: how they're architected, what "
        "orchestration frameworks are popular, what the main safety and reliability "
        "concerns are, how companies are evaluating ROI, and what the regulatory "
        "environment looks like across the US and EU."
    )
    fake_json = json.dumps(
        {
            "main_topic": "Enterprise LLM agent adoption",
            "sub_topics": [
                "Agent architecture patterns",
                "Orchestration frameworks",
                "Safety and reliability concerns",
                "ROI evaluation",
                "Regulatory environment (US/EU)",
            ],
            "search_queries": [
                {"query": "LLM agent architecture patterns enterprise 2026", "purpose": "Survey common architectures"},
                {"query": "popular LLM orchestration frameworks comparison", "purpose": "Identify leading frameworks"},
                {"query": "LLM agent safety reliability concerns enterprise", "purpose": "Understand risk factors"},
                {"query": "measuring ROI of enterprise LLM agents", "purpose": "Find ROI evaluation methods"},
                {"query": "EU AI Act LLM agents regulation 2026", "purpose": "Understand EU regulatory stance"},
                {"query": "US LLM agent regulation policy 2026", "purpose": "Understand US regulatory stance"},
            ],
        }
    )
    manager = FakeLLMManager(text=fake_json)
    planner = QueryPlanner(llm_manager=manager)

    plan = planner.create_plan(long_prompt)

    check(len(plan.search_queries) > 1, "multiple search queries were produced")
    check(len(plan.search_queries) >= 5, "the long prompt produced several focused queries (>=5)")
    check(
        all(len(q.query) < len(long_prompt) for q in plan.search_queries),
        "every individual query is shorter/more focused than the full prompt (not the whole paragraph)",
    )
    check(plan.original_prompt == long_prompt, "original_prompt is preserved exactly, even when long")


def test_3_malformed_llm_output():
    print("\n=== Test 3: Malformed LLM output raises QueryPlannerParsingError ===")

    # Case A: not valid JSON at all.
    manager_a = FakeLLMManager(text="Sure! Here are some search queries you could use: ...")
    planner_a = QueryPlanner(llm_manager=manager_a)
    raised_a = False
    try:
        planner_a.create_plan("Research the history of the printing press.")
    except QueryPlannerParsingError:
        raised_a = True
    check(raised_a, "non-JSON LLM output raises QueryPlannerParsingError")

    # Case B: valid JSON but missing required fields (no main_topic, no search_queries).
    manager_b = FakeLLMManager(text=json.dumps({"sub_topics": ["a", "b"]}))
    planner_b = QueryPlanner(llm_manager=manager_b)
    raised_b = False
    try:
        planner_b.create_plan("Research the history of the printing press.")
    except QueryPlannerParsingError:
        raised_b = True
    check(raised_b, "valid JSON missing required fields raises QueryPlannerParsingError")

    # Case C: valid JSON, main_topic present, but zero search queries.
    manager_c = FakeLLMManager(
        text=json.dumps({"main_topic": "Printing press history", "sub_topics": [], "search_queries": []})
    )
    planner_c = QueryPlanner(llm_manager=manager_c)
    raised_c = False
    try:
        planner_c.create_plan("Research the history of the printing press.")
    except QueryPlannerParsingError:
        raised_c = True
    check(raised_c, "zero search queries raises QueryPlannerParsingError")


def test_4_llm_failure():
    print("\n=== Test 4: LLM failure raises QueryPlannerLLMError ===")
    manager = FakeLLMManager(
        exception=AllProvidersFailedError(
            "All providers failed for this request: Gemini: ... | NVIDIA: ... | Groq: ..."
        )
    )
    planner = QueryPlanner(llm_manager=manager)

    raised = False
    try:
        planner.create_plan("Research renewable energy storage technologies.")
    except QueryPlannerLLMError as exc:
        raised = True
        check("failed" in str(exc).lower(), "the raised error message is clear about the LLM failure")

    check(raised, "QueryPlannerLLMError was raised when the LLM manager fails")
    check(len(manager.calls) == 1, "the LLM was invoked exactly once (planner does not implement its own fallback/retries)")


if __name__ == "__main__":
    test_1_short_research_prompt()
    test_2_long_research_prompt_multiple_focused_queries()
    test_3_malformed_llm_output()
    test_4_llm_failure()

    print("\n=== Summary ===")
    print(f"Passed: {len(PASSED)}")
    print(f"Failed: {len(FAILED)}")

    if FAILED:
        print("\nFailed checks:")
        for item in FAILED:
            print(f"  - {item}")
        sys.exit(1)
    else:
        print("All Query Planner tests passed.")
