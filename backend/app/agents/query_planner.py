"""
Query Planner (Step 5).

Turns a raw user research prompt (short or long) into a validated
`ResearchPlan`: a main topic, a handful of sub-topics, and a set of
focused search queries — never the whole user paragraph passed through
as a single search query.

This module only talks to the LLM via the existing `LLMManager`. It has
no knowledge of *which* provider actually served the request, and it
implements no provider fallback of its own — that logic lives entirely
inside `LLMManager`. It does not call Tavily and does not touch any
graph/orchestration logic.
"""

from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from app.schemas.models import ResearchPlan, SearchQuery
from app.services.llm_manager import LLMError, LLMManager

logger = logging.getLogger("app.query_planner")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class QueryPlannerError(Exception):
    """Base class for all Query Planner errors."""


class QueryPlannerLLMError(QueryPlannerError):
    """Raised when the underlying LLM request itself fails (all providers exhausted)."""


class QueryPlannerParsingError(QueryPlannerError):
    """Raised when the LLM's output can't be parsed/validated into a ResearchPlan."""


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_PLANNING_INSTRUCTIONS = """You are a research planning assistant.

Given a user's research request (it may be short or long, vague or detailed),
do the following:
1. Identify the single main topic of the research.
2. Break the request down into 2-5 focused sub-topics.
3. Generate several focused, specific web-search queries — never one giant
   query that just repeats the whole request. Each query should target one
   narrow aspect of the research, and each must include a short "purpose"
   explaining why that query is useful.

Respond with ONLY a single JSON object — no markdown code fences, no
commentary before or after it — in exactly this shape:
{
  "main_topic": "<string>",
  "sub_topics": ["<string>", "..."],
  "search_queries": [
    {"query": "<string>", "purpose": "<string>"},
    "..."
  ]
}

USER RESEARCH REQUEST:
"""


# ---------------------------------------------------------------------------
# Query Planner
# ---------------------------------------------------------------------------


class QueryPlanner:
    """
    Uses an LLM (via `LLMManager`) to turn a user prompt into a `ResearchPlan`.
    """

    def __init__(self, llm_manager: LLMManager | None = None):
        """
        Args:
            llm_manager: Optional LLMManager instance (or any object exposing
                a compatible `.generate(prompt) -> LLMResponse` method, mainly
                for tests). Defaults to a real `LLMManager()`.
        """
        self._llm_manager = llm_manager or LLMManager()

    def create_plan(self, prompt: str) -> ResearchPlan:
        """
        Build a validated `ResearchPlan` from a raw user research prompt.

        Args:
            prompt: The user's research request, any length.

        Returns:
            A validated `ResearchPlan`.

        Raises:
            ValueError: if `prompt` is empty/blank.
            QueryPlannerLLMError: if the LLM request itself fails (all
                providers in LLMManager's fallback chain were exhausted, or
                any other LLM error).
            QueryPlannerParsingError: if the LLM's output is not valid JSON,
                or doesn't validate against the ResearchPlan/SearchQuery
                schemas.
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")

        instruction = _PLANNING_INSTRUCTIONS + prompt

        logger.info("Requesting research plan from LLM for prompt=%r", _truncate(prompt))

        try:
            response = self._llm_manager.generate(instruction)
        except LLMError as exc:
            logger.error("Query planning failed: LLM request failed (%s).", exc)
            raise QueryPlannerLLMError(
                f"Query planning failed: LLM request failed ({exc})."
            ) from exc

        plan = self._parse_response(response.text, original_prompt=prompt)
        logger.info(
            "Research plan created: main_topic=%r, %d sub-topic(s), %d search quer(y/ies)",
            plan.main_topic,
            len(plan.sub_topics),
            len(plan.search_queries),
        )
        return plan

    # -- internals ------------------------------------------------------

    def _parse_response(self, raw_text: str, original_prompt: str) -> ResearchPlan:
        """Parse and validate the LLM's raw text into a ResearchPlan."""
        json_text = self._extract_json(raw_text)

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise QueryPlannerParsingError(
                f"LLM returned invalid JSON output: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise QueryPlannerParsingError(
                "LLM output was valid JSON but not a JSON object as expected."
            )

        try:
            search_queries_data = data.get("search_queries") or []
            search_queries = [SearchQuery(**item) for item in search_queries_data]

            plan = ResearchPlan(
                original_prompt=original_prompt,
                main_topic=data["main_topic"],
                sub_topics=data.get("sub_topics") or [],
                search_queries=search_queries,
            )
        except (KeyError, TypeError, ValidationError) as exc:
            raise QueryPlannerParsingError(
                f"LLM output did not match the expected ResearchPlan schema: {exc}"
            ) from exc

        if not plan.search_queries:
            raise QueryPlannerParsingError(
                "LLM output produced zero search queries; a valid plan needs at least one."
            )

        return plan

    @staticmethod
    def _extract_json(text: str) -> str:
        """
        Best-effort extraction of a JSON object from raw LLM text.

        Handles the common case of the LLM wrapping its JSON in a markdown
        code fence (```json ... ```) despite being asked not to, and falls
        back to slicing out the first {...} block if there's stray text
        around it.
        """
        text = text.strip()

        if text.startswith("```"):
            lines = text.splitlines()
            lines = lines[1:]  # drop opening fence (``` or ```json)
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start : end + 1]

        return text


def _truncate(text: str, limit: int = 120) -> str:
    """Shorten long prompts for log messages."""
    text = text.strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."
