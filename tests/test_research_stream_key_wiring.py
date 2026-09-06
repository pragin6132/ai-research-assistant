"""
End-to-end wiring test: confirms `_stream_research_events`'s custom_api_keys
argument actually reaches the SAME LLMManager instance shared by QueryPlanner,
Summarizer, and ReportGenerator when it builds the real graph — not just that
`_build_llm_manager` works in isolation (see test_research_api_key_override.py).

No real LLM/Tavily/network calls are made.

Run from backend/:
    python ../tests/test_research_stream_key_wiring.py
"""

import logging
import sys
from pathlib import Path
from unittest.mock import patch

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import app.api.research as research_module  # noqa: E402

PASSED = []
FAILED = []


def check(condition: bool, description: str):
    if condition:
        PASSED.append(description)
        print(f"  PASS: {description}")
    else:
        FAILED.append(description)
        print(f"  FAIL: {description}")


class RecordingGraph:
    """Stand-in for the compiled graph; just needs a working .stream()."""

    def stream(self, state, stream_mode=None):
        return iter([])  # no nodes to run; we only care about construction args


def test_same_llm_manager_shared_across_agents_with_override_applied():
    print("\n=== Test: custom_api_keys reaches the exact LLMManager given to all three agents ===")

    captured = {}

    def fake_query_planner(llm_manager=None):
        captured["query_planner_manager"] = llm_manager
        return "fake-query-planner"

    def fake_summarizer(llm_manager=None):
        captured["summarizer_manager"] = llm_manager
        return "fake-summarizer"

    def fake_report_generator(llm_manager=None):
        captured["report_generator_manager"] = llm_manager
        return "fake-report-generator"

    def fake_build_research_graph(**kwargs):
        captured["build_kwargs"] = kwargs
        return RecordingGraph()

    with patch.object(research_module, "QueryPlanner", side_effect=fake_query_planner), \
         patch.object(research_module, "Summarizer", side_effect=fake_summarizer), \
         patch.object(research_module, "ReportGenerator", side_effect=fake_report_generator), \
         patch.object(research_module, "build_research_graph", side_effect=fake_build_research_graph):

        list(research_module._stream_research_events("test prompt", {"gemini": "user-secret-xyz"}))

    manager = captured.get("query_planner_manager")
    check(manager is not None, "an LLMManager was constructed and passed to QueryPlanner")
    check(
        captured.get("summarizer_manager") is manager,
        "the exact same LLMManager instance was passed to Summarizer (not a second, separate one)",
    )
    check(
        captured.get("report_generator_manager") is manager,
        "the exact same LLMManager instance was passed to ReportGenerator",
    )

    providers = {p.name: p for p in manager._providers}
    check(providers["Gemini"].api_key == "user-secret-xyz", "the override actually reached the Gemini provider used by the real agents")

    build_kwargs = captured.get("build_kwargs", {})
    check("tavily_service" not in build_kwargs, "no tavily_service override was passed — Tavily always uses its default (app-configured) service")
    check(
        build_kwargs.get("query_planner") == "fake-query-planner"
        and build_kwargs.get("summarizer") == "fake-summarizer"
        and build_kwargs.get("report_generator") == "fake-report-generator",
        "build_research_graph received the constructed agents (all sharing the overridden LLMManager)",
    )


def test_no_override_when_all_providers_left_inbuilt():
    print("\n=== Test: no custom_api_keys -> agents still get a valid LLMManager using inbuilt keys ===")

    captured = {}

    def fake_query_planner(llm_manager=None):
        captured["manager"] = llm_manager
        return "fake-query-planner"

    with patch.object(research_module, "QueryPlanner", side_effect=fake_query_planner), \
         patch.object(research_module, "Summarizer", side_effect=lambda llm_manager=None: "s"), \
         patch.object(research_module, "ReportGenerator", side_effect=lambda llm_manager=None: "r"), \
         patch.object(research_module, "build_research_graph", side_effect=lambda **kw: RecordingGraph()):

        list(research_module._stream_research_events("test prompt", None))

    from app.core.config import Settings

    app_settings = Settings(_env_file=None)
    providers = {p.name: p for p in captured["manager"]._providers}
    check(
        providers["Gemini"].api_key == app_settings.GEMINI_API_KEY,
        "with no custom_api_keys, Gemini falls back to the app's own configured key",
    )


if __name__ == "__main__":
    test_same_llm_manager_shared_across_agents_with_override_applied()
    test_no_override_when_all_providers_left_inbuilt()

    print("\n=== Summary ===")
    print(f"Passed: {len(PASSED)}")
    print(f"Failed: {len(FAILED)}")

    if FAILED:
        print("\nFailed checks:")
        for item in FAILED:
            print(f"  - {item}")
        sys.exit(1)
    else:
        print("All request-to-agent key-wiring tests passed.")
