"""
Central research state for the LangGraph orchestration pipeline (Step 6A).

This module only defines the shared state shape that flows through the
graph's nodes. No node logic, no LLM calls, no Tavily calls live here —
just the state definition.

`ResearchState` is a `TypedDict` (LangGraph's expected state shape for
`StateGraph`), built from the existing Pydantic schemas in
`app.schemas.models` where a field already has a well-defined shape
(e.g. `research_plan`, `search_results`, `findings`), and plain Python
types elsewhere.
"""

from __future__ import annotations

import re
from typing import Optional, TypedDict

from app.schemas.models import Finding, ResearchPlan, ResearchReport, SearchQuery, SearchResult


_TAMIL_SCRIPT = re.compile(r"[\u0B80-\u0BFF]")
_LATIN_WORD = re.compile(r"[A-Za-z]+")
_ROMANIZED_TAMIL_MARKERS = re.compile(
    r"\b(?:anna|enna|enakku|ennaku|epdi|eppadi|irukku|irukka|pathi|patti|sollu|"
    r"sollunga|venum|vendum|theriyuma|theriyum|thayavu|nandri)\b",
    re.IGNORECASE,
)


def detect_response_language(user_prompt: str) -> str:
    """Choose the report language from the original user prompt.

    Tamil script is an unambiguous signal. A Tamil-script prompt with
    multiple Latin words is treated as Tamil-English mixed, while a small
    set of common romanized Tamil markers covers typical Tanglish prompts
    that contain no Tamil script. Ambiguous input defaults to English.
    """
    has_tamil_script = bool(_TAMIL_SCRIPT.search(user_prompt))
    latin_word_count = len(_LATIN_WORD.findall(user_prompt))

    if has_tamil_script:
        return "Tanglish / Tamil-English mixed" if latin_word_count >= 2 else "Tamil"
    if _ROMANIZED_TAMIL_MARKERS.search(user_prompt):
        return "Tanglish / Tamil-English mixed"
    return "English"


class ResearchState(TypedDict, total=False):
    """
    The single state object threaded through every node of the research graph.

    Fields:
        user_prompt: The raw research request as given by the user.
        response_language: The report language inferred from `user_prompt`.
        research_plan: The plan produced by the query planner (Step 5),
            once available.
        search_queries: The individual search queries to run (may start as
            a copy of `research_plan.search_queries`, but kept separate so
            later steps can add/remove/reorder queries independently).
        search_results: Raw structured results gathered from the search
            provider (e.g. Tavily), across all queries.
        findings: Synthesized findings derived from `search_results`,
            before validation.
        validated_findings: Findings that have passed a validation/quality
            pass.
        final_report: The final assembled research report.
        errors: A running log of error messages encountered by any node,
            so later nodes (and the caller) can see what went wrong
            without the graph necessarily halting.
        retry_count: A simple counter nodes can use to cap retries of a
            given step (e.g. re-searching, re-summarizing).
    """

    user_prompt: str
    response_language: str
    research_plan: Optional[ResearchPlan]
    search_queries: list[SearchQuery]
    search_results: list[SearchResult]
    findings: list[Finding]
    validated_findings: list[Finding]
    final_report: Optional[ResearchReport]
    errors: list[str]
    retry_count: int


def create_initial_state(user_prompt: str) -> ResearchState:
    """
    Build a fresh `ResearchState` for a new research request.

    All downstream fields start empty/None — nodes are responsible for
    populating them as the graph executes.
    """
    return ResearchState(
        user_prompt=user_prompt,
        response_language=detect_response_language(user_prompt),
        research_plan=None,
        search_queries=[],
        search_results=[],
        findings=[],
        validated_findings=[],
        final_report=None,
        errors=[],
        retry_count=0,
    )
