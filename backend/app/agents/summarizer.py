"""
Summarizer (Step 6D).

Turns cleaned `SearchResult` objects into structured `Finding` objects
(see app/schemas/models.py), using the existing `LLMManager`. Focused
purely on factual synthesis with attached source URLs — no validation
pass, no retries, and no provider-fallback logic of its own (that lives
entirely inside `LLMManager`).
"""

from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from app.schemas.models import Finding, SearchResult
from app.services.llm_manager import LLMError, LLMManager

logger = logging.getLogger("app.summarizer")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SummarizerError(Exception):
    """Base class for all Summarizer errors."""


class SummarizerLLMError(SummarizerError):
    """Raised when the underlying LLM request itself fails (all providers exhausted)."""


class SummarizerParsingError(SummarizerError):
    """Raised when the LLM's output can't be parsed/validated into Findings."""


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SUMMARY_INSTRUCTIONS = """You are a research synthesis assistant.

You will be given a research topic, optional sub-topics, and a list of
web search results (each with a title, URL, and content excerpt).

Your job is to synthesize these into a set of factual findings. Rules:
- Base every finding ONLY on the provided search results. Never invent
  facts, sources, or URLs that are not present in the given data.
- Each finding must cite the exact source URL(s), copied verbatim from
  the input, that support it.
- Group related search results into one finding per topic/sub-topic
  where it makes sense; don't force exactly one finding per result.
- Keep each summary factual and concise.

Respond with ONLY a single JSON object — no markdown code fences, no
commentary before or after it — in exactly this shape:
{
  "findings": [
    {
      "topic": "<string>",
      "summary": "<string>",
      "key_points": ["<string>", "..."],
      "sources": ["<url>", "..."]
    }
  ]
}
"""


# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------


class Summarizer:
    """
    Uses an LLM (via `LLMManager`) to synthesize `SearchResult`s into
    structured `Finding`s.
    """

    def __init__(self, llm_manager: LLMManager | None = None):
        """
        Args:
            llm_manager: Optional LLMManager instance (or any object
                exposing a compatible `.generate(prompt) -> LLMResponse`
                method, mainly for tests). Defaults to a real `LLMManager()`.
        """
        self._llm_manager = llm_manager or LLMManager()

    def summarize(
        self,
        search_results: list[SearchResult],
        main_topic: str | None = None,
        sub_topics: list[str] | None = None,
    ) -> list[Finding]:
        """
        Synthesize a list of `SearchResult`s into structured `Finding`s.

        Args:
            search_results: Cleaned search results to synthesize (e.g. the
                output of the `process` node).
            main_topic: Optional overall research topic, for context.
            sub_topics: Optional list of sub-topics, for context.

        Returns:
            A list of `Finding`s. Returns an empty list (without calling
            the LLM) if `search_results` is empty.

        Raises:
            SummarizerLLMError: if the LLM request itself fails.
            SummarizerParsingError: if the LLM's output is not valid JSON,
                or doesn't validate against the `Finding` schema.
        """
        if not search_results:
            logger.info("Summarizer: no search results to summarize; skipping LLM call.")
            return []

        instruction = self._build_prompt(search_results, main_topic, sub_topics)

        logger.info(
            "Requesting synthesis from LLM for %d search result(s).", len(search_results)
        )

        try:
            response = self._llm_manager.generate(instruction)
        except LLMError as exc:
            logger.error("Summarization failed: LLM request failed (%s).", exc)
            raise SummarizerLLMError(
                f"Summarization failed: LLM request failed ({exc})."
            ) from exc

        findings = self._parse_response(response.text)
        logger.info("Summarizer produced %d finding(s).", len(findings))
        return findings

    # -- internals ------------------------------------------------------

    def _build_prompt(
        self,
        search_results: list[SearchResult],
        main_topic: str | None,
        sub_topics: list[str] | None,
    ) -> str:
        parts = [_SUMMARY_INSTRUCTIONS]

        if main_topic:
            parts.append(f"MAIN TOPIC:\n{main_topic}\n")
        if sub_topics:
            parts.append("SUB-TOPICS:\n" + "\n".join(f"- {topic}" for topic in sub_topics) + "\n")

        result_lines = ["SEARCH RESULTS:"]
        for index, result in enumerate(search_results, start=1):
            result_lines.append(
                f"{index}. Title: {result.title}\n   URL: {result.url}\n   Content: {result.content}"
            )
        parts.append("\n".join(result_lines))

        return "\n".join(parts)

    def _parse_response(self, raw_text: str) -> list[Finding]:
        """Parse and validate the LLM's raw text into a list of Findings."""
        json_text = self._extract_json(raw_text)

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise SummarizerParsingError(
                f"LLM returned invalid JSON output: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise SummarizerParsingError(
                "LLM output was valid JSON but not a JSON object as expected."
            )

        findings_data = data.get("findings") or []

        try:
            findings = [Finding(**item) for item in findings_data]
        except (KeyError, TypeError, ValidationError) as exc:
            raise SummarizerParsingError(
                f"LLM output did not match the expected Finding schema: {exc}"
            ) from exc

        return findings

    @staticmethod
    def _extract_json(text: str) -> str:
        """
        Best-effort extraction of a JSON object from raw LLM text.

        Handles the common case of the LLM wrapping its JSON in a markdown
        code fence despite being asked not to, and falls back to slicing
        out the first {...} block if there's stray text around it.
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
