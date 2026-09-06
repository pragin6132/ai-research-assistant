"""
Mock-based test for Step 6C: the `process` node cleaning/deduplicating
`search_results`.

No LLM/Tavily calls are involved — `process_node` is tested directly
against hand-built state dicts.

Run from backend/:
    python ../tests/test_graph_process_mock.py
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.graph.research_graph import process_node  # noqa: E402
from app.schemas.models import SearchResult  # noqa: E402

PASSED = []
FAILED = []


def check(condition: bool, description: str):
    if condition:
        PASSED.append(description)
        print(f"  PASS: {description}")
    else:
        FAILED.append(description)
        print(f"  FAIL: {description}")


def test_1_duplicate_urls_are_removed():
    print("\n=== Test 1: duplicate URLs are removed, first occurrence kept ===")
    r1 = SearchResult(title="Article A", url="https://example.com/a", content="First copy", relevance_score=0.9)
    r2 = SearchResult(title="Article A mirror", url="https://example.com/a/", content="Second copy, trailing slash", relevance_score=0.5)
    r3 = SearchResult(title="Article A upper", url="HTTPS://EXAMPLE.COM/a", content="Third copy, different case", relevance_score=0.4)
    r4 = SearchResult(title="Article B", url="https://example.com/b", content="A different article", relevance_score=0.7)

    state = {"search_results": [r1, r2, r3, r4], "errors": []}
    result = process_node(state)

    check(len(result["search_results"]) == 2, "only 2 unique URLs remain after dedup (a, b)")
    check(result["search_results"][0].content == "First copy", "the first occurrence of a duplicated URL is the one kept")
    check(result["search_results"][1].title == "Article B", "the distinct second URL is preserved")


def test_2_invalid_or_empty_results_are_ignored():
    print("\n=== Test 2: invalid/empty results are ignored ===")
    good = SearchResult(title="Good Result", url="https://example.com/good", content="Real content here", relevance_score=0.8)
    empty_content = SearchResult(title="No Content", url="https://example.com/empty-content", content="   ", relevance_score=0.5)
    empty_title = SearchResult(title=" ", url="https://example.com/empty-title", content="Has content but no title") \
        if _title_allows_blank() else None

    search_results = [good, empty_content, "not-a-search-result", None]
    if empty_title is not None:
        search_results.append(empty_title)

    state = {"search_results": search_results, "errors": []}
    result = process_node(state)

    check(len(result["search_results"]) == 1, "only the single genuinely valid, non-empty result survives")
    check(result["search_results"][0].title == "Good Result", "the surviving result is the well-formed one")


def _title_allows_blank() -> bool:
    """Whether SearchResult's schema even allows a blank/whitespace title to be constructed."""
    try:
        SearchResult(title=" ", url="https://example.com/x", content="content")
        return True
    except Exception:
        return False


def test_3_valid_data_is_preserved_unchanged():
    print("\n=== Test 3: valid SearchResult data is preserved unchanged ===")
    r1 = SearchResult(title="Preserved Title", url="https://example.com/preserved", content="Preserved content", relevance_score=0.73)
    r2 = SearchResult(title="No Score Result", url="https://example.com/no-score", content="Some content", relevance_score=None)

    state = {"search_results": [r1, r2], "errors": ["pre-existing error"]}
    result = process_node(state)

    check(len(result["search_results"]) == 2, "both valid, distinct results are kept")
    check(result["search_results"][0].title == "Preserved Title", "title preserved")
    check(str(result["search_results"][0].url) == "https://example.com/preserved", "url preserved")
    check(result["search_results"][0].content == "Preserved content", "content preserved")
    check(result["search_results"][0].relevance_score == 0.73, "relevance_score preserved when present")
    check(result["search_results"][1].relevance_score is None, "relevance_score preserved as None when absent")
    check(result["errors"] == ["pre-existing error"], "existing errors are passed through untouched (process doesn't add its own)")


if __name__ == "__main__":
    test_1_duplicate_urls_are_removed()
    test_2_invalid_or_empty_results_are_ignored()
    test_3_valid_data_is_preserved_unchanged()

    print("\n=== Summary ===")
    print(f"Passed: {len(PASSED)}")
    print(f"Failed: {len(FAILED)}")

    if FAILED:
        print("\nFailed checks:")
        for item in FAILED:
            print(f"  - {item}")
        sys.exit(1)
    else:
        print("All process-node cleanup/dedup tests passed.")
