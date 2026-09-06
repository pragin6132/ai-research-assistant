"""
Mock-based test script for app/services/tavily_service.py.

No real network calls are made and no real Tavily API key is required:
a fake client object is injected directly into `TavilySearchService`,
simulating Tavily's raw response shape.

Run from backend/:
    python ../tests/test_tavily_service_mock.py
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.core.config import Settings  # noqa: E402
from app.schemas.models import SearchResult  # noqa: E402
from app.services.tavily_service import (  # noqa: E402
    TavilyConfigurationError,
    TavilyConnectionError,
    TavilySearchService,
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
    """Settings with a plausible (but fake) Tavily key."""
    return Settings(_env_file=None, TAVILY_API_KEY="fake-tavily-key")


class FakeTavilyClient:
    """A stand-in for the real `tavily.TavilyClient`, used only in tests."""

    def __init__(self, response=None, exception=None):
        self.response = response
        self.exception = exception
        self.calls = []

    def search(self, query, max_results=None, search_depth=None):
        self.calls.append(
            {"query": query, "max_results": max_results, "search_depth": search_depth}
        )
        if self.exception is not None:
            raise self.exception
        return self.response


def test_1_successful_results():
    print("\n=== Test 1: Successful search returns structured SearchResult objects ===")
    fake_response = {
        "results": [
            {
                "title": "Retrieval-Augmented Generation Explained",
                "url": "https://example.com/rag-explained",
                "content": "RAG combines retrieval with generation...",
                "score": 0.91,
            },
            {
                "title": "Vector Databases 101",
                "url": "https://example.com/vector-db-101",
                "content": "Vector databases store embeddings...",
                "score": 0.77,
            },
        ]
    }
    client = FakeTavilyClient(response=fake_response)
    service = TavilySearchService(settings=dummy_settings(), client=client)

    results = service.search("what is RAG")

    check(len(results) == 2, "two results were returned")
    check(all(isinstance(r, SearchResult) for r in results), "all results are SearchResult instances")
    check(results[0].title == "Retrieval-Augmented Generation Explained", "first result title preserved")
    check(str(results[0].url) == "https://example.com/rag-explained", "first result URL preserved")
    check(results[0].relevance_score == 0.91, "first result relevance_score preserved")
    check(results[1].relevance_score == 0.77, "second result relevance_score preserved")
    check(client.calls[0]["query"] == "what is RAG", "the exact query text was passed through (not hardcoded)")


def test_2_empty_results():
    print("\n=== Test 2: Empty Tavily response is handled gracefully ===")
    client = FakeTavilyClient(response={"results": []})
    service = TavilySearchService(settings=dummy_settings(), client=client)

    results = service.search("a topic with no coverage")

    check(results == [], "an empty list is returned (not an error) for zero results")


def test_3_missing_api_key():
    print("\n=== Test 3: Missing API key is handled clearly ===")
    empty_settings = Settings(_env_file=None, TAVILY_API_KEY=None)
    service = TavilySearchService(settings=empty_settings)

    raised = False
    try:
        service.search("any query")
    except TavilyConfigurationError:
        raised = True

    check(raised, "TavilyConfigurationError was raised when no API key is configured")
    check(not service.is_configured(), "is_configured() correctly reports False")


def test_4_connection_timeout_error():
    print("\n=== Test 4: Network/timeout errors are classified clearly ===")
    client = FakeTavilyClient(exception=TimeoutError("Request timed out"))
    service = TavilySearchService(settings=dummy_settings(), client=client)

    raised = False
    try:
        service.search("any query")
    except TavilyConnectionError:
        raised = True

    check(raised, "TavilyConnectionError was raised on a timeout")


def test_5_malformed_result_is_skipped_not_fabricated():
    print("\n=== Test 5: A malformed result (missing URL) is skipped, never fabricated ===")
    fake_response = {
        "results": [
            {"title": "Good Result", "url": "https://example.com/good", "content": "fine", "score": 0.5},
            {"title": "Bad Result - no URL", "content": "this has no url key"},
        ]
    }
    client = FakeTavilyClient(response=fake_response)
    service = TavilySearchService(settings=dummy_settings(), client=client)

    results = service.search("query with one bad result")

    check(len(results) == 1, "only the well-formed result was kept")
    check(results[0].title == "Good Result", "the surviving result is the well-formed one")


if __name__ == "__main__":
    test_1_successful_results()
    test_2_empty_results()
    test_3_missing_api_key()
    test_4_connection_timeout_error()
    test_5_malformed_result_is_skipped_not_fabricated()

    print("\n=== Summary ===")
    print(f"Passed: {len(PASSED)}")
    print(f"Failed: {len(FAILED)}")

    if FAILED:
        print("\nFailed checks:")
        for item in FAILED:
            print(f"  - {item}")
        sys.exit(1)
    else:
        print("All Tavily service tests passed.")
