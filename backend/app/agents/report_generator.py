"""
Report Generator (Step 6G).

Turns a list of validated `Finding` objects into a fully assembled,
schema-validated `ResearchReport` (see app/schemas/models.py), using the
existing `LLMManager`.

To avoid any risk of fabricated citations, the LLM is only ever asked to
write the report's *narrative* parts — title, executive summary, and a
short list of overall key findings. The `findings` and `sources` fields
of the final `ResearchReport` are built directly from the
already-validated data this module is given, never regenerated or
reworded by the LLM.
"""

from __future__ import annotations

import json
import logging

from pydantic import HttpUrl, ValidationError

from app.schemas.models import Finding, ResearchReport
from app.services.llm_manager import LLMError, LLMManager

logger = logging.getLogger("app.report_generator")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ReportGeneratorError(Exception):
    """Base class for all Report Generator errors."""


class ReportGeneratorLLMError(ReportGeneratorError):
    """Raised when the underlying LLM request itself fails (all providers exhausted)."""


class ReportGeneratorParsingError(ReportGeneratorError):
    """Raised when the LLM's output can't be parsed into a valid report narrative."""


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_REPORT_INSTRUCTIONS = """You are a research report writer.

You will be given a research topic and a list of already-validated
findings (each with a topic, summary, key points, and sources already
attached).

Your job is to write ONLY:
1. A concise, descriptive report title.
2. A short executive summary (2-4 sentences) synthesizing the overall
   research across all findings.
3. A list of the most important overall key findings, as short
   bullet-point strings (aim for 3-6 items).

Rules:
- Write the title, executive summary, and key findings in the requested
  report language. For Tamil-English mixed requests, use natural Tanglish
  where appropriate; do not force the prose into English.
- Base everything ONLY on the provided findings. Never invent facts.
- Do NOT include URLs/sources in your output — those are handled
  separately and must not be restated or altered by you.

Respond with ONLY a single JSON object — no markdown code fences, no
commentary before or after it — in exactly this shape:
{
  "title": "<string>",
  "executive_summary": "<string>",
  "key_findings": ["<string>", "..."]
}
"""


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------


class ReportGenerator:
    """
    Uses an LLM (via `LLMManager`) to write a report's narrative, then
    assembles a fully validated `ResearchReport` around the untouched
    `Finding` data it was given.
    """

    def __init__(self, llm_manager: LLMManager | None = None):
        """
        Args:
            llm_manager: Optional LLMManager instance (or any object
                exposing a compatible `.generate(prompt) -> LLMResponse`
                method, mainly for tests). Defaults to a real `LLMManager()`.
        """
        self._llm_manager = llm_manager or LLMManager()

    def generate_report(
        self,
        validated_findings: list[Finding],
        main_topic: str | None = None,
        response_language: str = "English",
    ) -> ResearchReport:
        """
        Assemble a `ResearchReport` from validated findings.

        Args:
            validated_findings: The findings that passed validation (e.g.
                the `validate` node's output). Must be non-empty.
            main_topic: Optional overall research topic, for context.
            response_language: Language/style to use for the report narrative.

        Returns:
            A validated `ResearchReport`.

        Raises:
            ValueError: if `validated_findings` is empty.
            ReportGeneratorLLMError: if the LLM request itself fails.
            ReportGeneratorParsingError: if the LLM's output can't be
                parsed into a usable title/summary/key_findings, or the
                assembled report fails schema validation.
        """
        if not validated_findings:
            raise ValueError("validated_findings must be a non-empty list.")

        instruction = self._build_prompt(validated_findings, main_topic, response_language)

        logger.info(
            "Requesting report narrative from LLM for %d validated finding(s).",
            len(validated_findings),
        )

        try:
            response = self._llm_manager.generate(instruction)
        except LLMError as exc:
            logger.error("Report generation failed: LLM request failed (%s).", exc)
            raise ReportGeneratorLLMError(
                f"Report generation failed: LLM request failed ({exc})."
            ) from exc

        title, executive_summary, key_findings = self._parse_response(response.text)
        sources = self._collect_sources(validated_findings)

        try:
            report = ResearchReport(
                title=title,
                executive_summary=executive_summary,
                key_findings=key_findings,
                findings=validated_findings,
                sources=sources,
            )
        except ValidationError as exc:
            raise ReportGeneratorParsingError(
                f"Assembled report failed schema validation: {exc}"
            ) from exc

        logger.info(
            "ReportGenerator assembled report %r with %d finding(s) and %d source(s).",
            report.title,
            len(report.findings),
            len(report.sources),
        )
        return report

    # -- internals ------------------------------------------------------

    def _build_prompt(
        self,
        validated_findings: list[Finding],
        main_topic: str | None,
        response_language: str,
    ) -> str:
        parts = [_REPORT_INSTRUCTIONS]

        parts.append(f"REPORT LANGUAGE: {response_language}\n")

        if main_topic:
            parts.append(f"RESEARCH TOPIC:\n{main_topic}\n")

        lines = ["VALIDATED FINDINGS:"]
        for index, finding in enumerate(validated_findings, start=1):
            key_points = "; ".join(finding.key_points) if finding.key_points else "(none)"
            lines.append(
                f"{index}. Topic: {finding.topic}\n"
                f"   Summary: {finding.summary}\n"
                f"   Key points: {key_points}"
            )
        parts.append("\n".join(lines))

        return "\n".join(parts)

    def _parse_response(self, raw_text: str) -> tuple[str, str, list[str]]:
        """Parse the LLM's narrative JSON into (title, executive_summary, key_findings)."""
        json_text = self._extract_json(raw_text)

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise ReportGeneratorParsingError(
                f"LLM returned invalid JSON output: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise ReportGeneratorParsingError(
                "LLM output was valid JSON but not a JSON object as expected."
            )

        try:
            title = data["title"]
            executive_summary = data["executive_summary"]
            key_findings = data.get("key_findings") or []

            if not isinstance(title, str) or not title.strip():
                raise ValueError("title must be a non-empty string.")
            if not isinstance(executive_summary, str) or not executive_summary.strip():
                raise ValueError("executive_summary must be a non-empty string.")
            if not isinstance(key_findings, list) or not all(isinstance(k, str) for k in key_findings):
                raise ValueError("key_findings must be a list of strings.")
        except (KeyError, ValueError, TypeError) as exc:
            raise ReportGeneratorParsingError(
                f"LLM output did not match the expected report shape: {exc}"
            ) from exc

        return title, executive_summary, key_findings

    @staticmethod
    def _collect_sources(validated_findings: list[Finding]) -> list[HttpUrl]:
        """
        Deduplicate sources across all findings, preserving first-seen
        order. Never invents a URL — only copies what's already present.
        """
        seen: set[str] = set()
        sources: list[HttpUrl] = []
        for finding in validated_findings:
            for url in finding.sources:
                key = str(url)
                if key not in seen:
                    seen.add(key)
                    sources.append(url)
        return sources

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
