"""
Mock-based test for Step 6E: the `validate` node checking `Finding`
objects against the existing Pydantic schema and a minimal business rule
(must cite at least one source).

No LLM/Tavily calls are involved — `validate_node` is tested directly
against hand-built state dicts.

Run from backend/:
    python ../tests/test_graph_validate_mock.py
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.graph.research_graph import validate_node  # noqa: E402
from app.schemas.models import Finding  # noqa: E402

PASSED = []
FAILED = []


def check(condition: bool, description: str):
    if condition:
        PASSED.append(description)
        print(f"  PASS: {description}")
    else:
        FAILED.append(description)
        print(f"  FAIL: {description}")


def test_1_valid_findings_all_pass():
    print("\n=== Test 1: valid findings all pass validation unchanged ===")
    f1 = Finding(
        topic="Battery materials",
        summary="Ceramic and polymer electrolytes are the main approaches.",
        key_points=["Ceramic offers high conductivity"],
        sources=["https://example.com/materials"],
    )
    f2 = Finding(
        topic="Manufacturing challenges",
        summary="Scaling production remains costly.",
        key_points=["High capital costs"],
        sources=["https://example.com/manufacturing", "https://example.com/costs"],
    )

    state = {"findings": [f1, f2], "errors": []}
    result = validate_node(state)

    check(len(result["validated_findings"]) == 2, "both valid findings are validated")
    check(result["validated_findings"][0].topic == "Battery materials", "first finding's data is unchanged")
    check(result["validated_findings"][1].sources[1] and True, "multi-source finding preserved intact")
    check(result["errors"] == [], "no errors recorded when everything is valid")


def test_2_finding_with_no_sources_is_rejected():
    print("\n=== Test 2: a finding with no sources is rejected and recorded ===")
    good = Finding(
        topic="Good finding",
        summary="Has a source.",
        key_points=["ok"],
        sources=["https://example.com/good"],
    )
    unsourced = Finding(
        topic="Unsourced claim",
        summary="This has no citations.",
        key_points=["risky"],
        sources=[],
    )

    state = {"findings": [good, unsourced], "errors": []}
    result = validate_node(state)

    check(len(result["validated_findings"]) == 1, "only the sourced finding survives")
    check(result["validated_findings"][0].topic == "Good finding", "the surviving finding is the sourced one")
    check(len(result["errors"]) == 1, "one validation error was recorded")
    check("Unsourced claim" in result["errors"][0], "the error identifies the rejected finding by topic")
    check("no sources cited" in result["errors"][0], "the error explains why it was rejected")


def test_3_non_finding_item_is_rejected():
    print("\n=== Test 3: a non-Finding item in the list is rejected, not fabricated into a Finding ===")
    good = Finding(
        topic="Good finding",
        summary="Has a source.",
        key_points=["ok"],
        sources=["https://example.com/good"],
    )

    state = {"findings": [good, {"topic": "not a real Finding object"}, "just a string"], "errors": []}
    result = validate_node(state)

    check(len(result["validated_findings"]) == 1, "only the genuine Finding instance survives")
    check(len(result["errors"]) == 2, "both non-Finding items produced a recorded error")
    check(all("non-Finding item" in e for e in result["errors"]), "errors clearly explain the rejection reason")


def test_4_empty_findings_produce_no_errors():
    print("\n=== Test 4: empty findings input is handled cleanly ===")
    state = {"findings": [], "errors": ["pre-existing error"]}
    result = validate_node(state)

    check(result["validated_findings"] == [], "validated_findings is empty for empty input")
    check(result["errors"] == ["pre-existing error"], "pre-existing errors pass through untouched, no new ones added")


if __name__ == "__main__":
    test_1_valid_findings_all_pass()
    test_2_finding_with_no_sources_is_rejected()
    test_3_non_finding_item_is_rejected()
    test_4_empty_findings_produce_no_errors()

    print("\n=== Summary ===")
    print(f"Passed: {len(PASSED)}")
    print(f"Failed: {len(FAILED)}")

    if FAILED:
        print("\nFailed checks:")
        for item in FAILED:
            print(f"  - {item}")
        sys.exit(1)
    else:
        print("All validate-node tests passed.")
