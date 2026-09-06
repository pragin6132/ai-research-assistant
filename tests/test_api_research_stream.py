"""
Minimal streaming API test for POST /api/research (Step 8).

No real LLM/Tavily calls are made: `build_research_graph` is patched
with a fake compiled graph whose `.stream()` yields canned per-node
update chunks (matching LangGraph's real `stream_mode="updates"` shape),
so this only exercises the SSE streaming layer itself.

Run from backend/:
    python ../tests/test_api_research_stream.py
"""

import logging
import sys
from pathlib import Path
from unittest.mock import patch

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

from app.schemas.models import Finding, ResearchReport  # noqa: E402

PASSED = []
FAILED = []


def check(condition: bool, description: str):
    if condition:
        PASSED.append(description)
        print(f"  PASS: {description}")
    else:
        FAILED.append(description)
        print(f"  FAIL: {description}")


class FakeGraph:
    """
    Stand-in for the compiled LangGraph graph. `.stream()` yields the same
    shape LangGraph's real `stream_mode="updates"` produces: one
    {node_name: partial_update_dict} per completed node.
    """

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.streamed_with = None

    def stream(self, state, stream_mode=None):
        self.streamed_with = state
        for chunk in self.chunks:
            yield chunk


def _parse_sse_events(raw_text: str) -> list[dict]:
    """Parse a raw SSE response body into a list of {"event": ..., "data": ...} dicts."""
    import json

    events = []
    for block in raw_text.strip().split("\n\n"):
        if not block.strip():
            continue
        event_name, data_line = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_line = line[len("data:"):].strip()
        if event_name is not None and data_line is not None:
            events.append({"event": event_name, "data": json.loads(data_line)})
    return events


def test_post_research_streams_progress_then_final_report():
    print("\n=== Test: POST /api/research streams progress events, then a final_report event ===")

    finding = Finding(
        topic="Solid-state battery materials",
        summary="Ceramic electrolytes are the leading approach.",
        key_points=["High conductivity"],
        sources=["https://example.com/materials"],
    )
    report = ResearchReport(
        title="Solid-State Battery Research Overview",
        executive_summary="Ceramic electrolytes show the most promise so far.",
        key_findings=["Ceramic electrolytes offer high conductivity"],
        findings=[finding],
        sources=["https://example.com/materials"],
    )

    fake_chunks = [
        {"analyze": {}},
        {"plan": {"errors": []}},
        {"search": {"errors": []}},
        {"process": {"errors": []}},
        {"summarize": {"errors": []}},
        {"validate": {"errors": []}},
        {"quality_check": {"errors": []}},
        {"report": {"final_report": report, "errors": []}},
    ]
    fake_graph = FakeGraph(fake_chunks)

    with patch("app.api.research.build_research_graph", return_value=fake_graph):
        import main  # imported inside the patch context; the route calls

        # build_research_graph() at request time, so patching
        # app.api.research.build_research_graph takes effect regardless
        # of when main.py itself was imported.

        client = TestClient(main.app)
        response = client.post(
            "/api/research", json={"prompt": "Research solid-state batteries."}
        )

    check(response.status_code == 200, "response status code is 200")
    check(
        response.headers["content-type"].startswith("text/event-stream"),
        "response content-type is text/event-stream",
    )

    events = _parse_sse_events(response.text)
    progress_events = [e for e in events if e["event"] == "progress"]
    final_events = [e for e in events if e["event"] == "final_report"]

    check(len(progress_events) == 1 + len(fake_chunks), "one 'start' progress event plus one per node was streamed")
    check(progress_events[0]["data"]["node"] == "start", "the first progress event marks the pipeline start")
    check(
        [e["data"]["node"] for e in progress_events[1:]]
        == ["analyze", "plan", "search", "process", "summarize", "validate", "quality_check", "report"],
        "progress events are emitted in the exact node execution order",
    )
    check(len(final_events) == 1, "exactly one final_report event was streamed, at the end")
    check(
        final_events[0]["data"]["final_report"]["title"] == "Solid-State Battery Research Overview",
        "the final_report event carries the assembled report",
    )
    check(final_events[0]["data"]["errors"] == [], "the final_report event carries the (empty) errors list")
    check(
        fake_graph.streamed_with["user_prompt"] == "Research solid-state batteries.",
        "the request's prompt was passed into the graph's initial state",
    )


if __name__ == "__main__":
    test_post_research_streams_progress_then_final_report()

    print("\n=== Summary ===")
    print(f"Passed: {len(PASSED)}")
    print(f"Failed: {len(FAILED)}")

    if FAILED:
        print("\nFailed checks:")
        for item in FAILED:
            print(f"  - {item}")
        sys.exit(1)
    else:
        print("Streaming API research endpoint test passed.")
