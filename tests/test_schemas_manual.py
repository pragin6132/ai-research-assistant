"""
Small standalone validation script for the schemas in app/schemas/models.py.

This is not wired into a test framework yet (e.g. pytest) — it's a quick
manual check that:
  1. Valid sample data passes validation for every model.
  2. An invalid URL is rejected as expected.

Run from backend/:
    python ../tests/test_schemas_manual.py
"""

import sys
from pathlib import Path

from pydantic import ValidationError

# Make `app` importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.schemas.models import (  # noqa: E402
    Finding,
    ResearchPlan,
    ResearchReport,
    SearchQuery,
    SearchResult,
)


def test_valid_data():
    print("=== Testing valid data ===")

    search_query = SearchQuery(
        query="latest advances in retrieval-augmented generation",
        purpose="Understand current RAG techniques",
    )
    print("SearchQuery OK:", search_query)

    plan = ResearchPlan(
        original_prompt="Research the state of RAG in 2026",
        main_topic="Retrieval-Augmented Generation",
        sub_topics=["Vector databases", "Reranking", "Agentic RAG"],
        search_queries=[search_query],
    )
    print("ResearchPlan OK:", plan.main_topic)

    search_result = SearchResult(
        title="A Survey of RAG Techniques",
        url="https://example.com/rag-survey",
        content="This paper surveys retrieval-augmented generation methods...",
        relevance_score=0.87,
    )
    print("SearchResult OK:", search_result.url)

    finding = Finding(
        topic="Vector databases",
        summary="Vector databases are widely used to store embeddings for retrieval.",
        key_points=["FAISS is popular for local use", "Managed options scale better"],
        sources=["https://example.com/vector-db-guide"],
    )
    print("Finding OK:", finding.topic)

    report = ResearchReport(
        title="The State of RAG in 2026",
        executive_summary="RAG has matured into agentic, multi-step pipelines...",
        key_findings=["Agentic RAG is now common", "Reranking improves precision"],
        findings=[finding],
        sources=["https://example.com/rag-survey", "https://example.com/vector-db-guide"],
    )
    print("ResearchReport OK:", report.title)

    print("All valid data passed validation.\n")


def test_invalid_url():
    print("=== Testing invalid URL rejection ===")
    try:
        SearchResult(
            title="Bad Result",
            url="not-a-valid-url",
            content="This should fail validation.",
        )
        print("FAILED: invalid URL was incorrectly accepted!")
    except ValidationError as e:
        print("Invalid URL correctly rejected. Error:")
        print(e)
        print()


if __name__ == "__main__":
    test_valid_data()
    test_invalid_url()
