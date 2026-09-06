"""
POST /api/research endpoint (Step 7, streaming added in Step 8).

Exposes the existing LangGraph research workflow (`build_research_graph`)
over HTTP as a Server-Sent Events (SSE) stream. Accepts a raw research
prompt and streams:
  - one "progress" event per graph node as it completes, and
  - a final "final_report" event once the graph finishes, carrying the
    assembled `ResearchReport` (if any) plus any errors collected along
    the way by the graph's nodes.

This is intentionally thin: it builds the graph, walks its execution via
`.stream(..., stream_mode="updates")`, forwards each node's completion as
an SSE event, and emits the final state at the end. No new pipeline
logic lives here — everything is reused from `app.graph.research_graph`.

Per-provider API key configuration: the request may optionally include
`custom_api_keys` for Gemini/NVIDIA/Groq (Settings UI's "Inbuilt API Key:
Off" case). When present, that provider's key overrides the app's own
configured (inbuilt) key for this request only — nothing is persisted
server-side. Tavily always uses the app's configured key; there is no
override for it. Key values are never logged, and are stripped from any
error text before it's logged or sent to the client.
"""

from __future__ import annotations

import json
import logging
from typing import Iterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agents.query_planner import QueryPlanner
from app.agents.report_generator import ReportGenerator
from app.agents.summarizer import Summarizer
from app.core.config import get_settings
from app.graph.research_graph import build_research_graph
from app.graph.state import ResearchState, create_initial_state
from app.services.llm_manager import LLMManager

logger = logging.getLogger("app.api.research")

router = APIRouter(tags=["research"])

# Maps the Settings UI's provider keys to the corresponding Settings field.
_PROVIDER_KEY_FIELDS = {
    "gemini": "GEMINI_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "groq": "GROQ_API_KEY",
}


class ResearchRequest(BaseModel):
    """Request body for POST /api/research."""

    prompt: str = Field(..., min_length=1, description="The user's research request.")
    custom_api_keys: dict[str, str] | None = Field(
        default=None,
        description=(
            "Optional per-provider API key overrides for gemini/nvidia/groq "
            "(used when the Settings UI's 'Inbuilt API Key' is Off for that "
            "provider). Any provider not present here uses the app's own "
            "configured key. Tavily has no override. Used for this request "
            "only — never persisted server-side, never logged."
        ),
    )


def _build_llm_manager(custom_api_keys: dict[str, str] | None) -> LLMManager:
    """
    Build an `LLMManager` whose provider keys reflect the request's
    per-provider overrides (if any), falling back to the app's own
    configured (inbuilt) keys for everything else.
    """
    base_settings = get_settings()

    overrides = {}
    if custom_api_keys:
        for provider, field_name in _PROVIDER_KEY_FIELDS.items():
            value = custom_api_keys.get(provider)
            if value:
                overrides[field_name] = value

    effective_settings = base_settings.model_copy(update=overrides) if overrides else base_settings
    return LLMManager(settings=effective_settings)


def _redact(text: str, secrets: list[str]) -> str:
    """Replace any occurrence of a known secret value with a placeholder."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[redacted]")
    return text


def _format_sse(event: str, data: dict) -> str:
    """Format a single Server-Sent Event message."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stream_research_events(
    prompt: str, custom_api_keys: dict[str, str] | None = None
) -> Iterator[str]:
    """
    Run the research graph for `prompt`, yielding SSE events:
      - "progress": one per graph node as it completes (e.g. "plan", "search").
      - "final_report": once, at the end, with the assembled report + errors.
      - "error": only if the graph itself raises an unexpected exception.

    Maintains a local running copy of the state so the fully-assembled
    final state is available once streaming completes, without needing a
    LangGraph checkpointer.
    """
    secrets_to_redact = list((custom_api_keys or {}).values())

    llm_manager = _build_llm_manager(custom_api_keys)
    graph = build_research_graph(
        query_planner=QueryPlanner(llm_manager=llm_manager),
        summarizer=Summarizer(llm_manager=llm_manager),
        report_generator=ReportGenerator(llm_manager=llm_manager),
        # tavily_service intentionally left as default: Tavily always uses
        # the app's own configured key, with no per-request override.
    )
    state: ResearchState = create_initial_state(prompt)

    yield _format_sse("progress", {"node": "start", "message": "Research pipeline started."})

    try:
        for chunk in graph.stream(state, stream_mode="updates"):
            for node_name, update in chunk.items():
                state.update(update)
                yield _format_sse(
                    "progress",
                    {"node": node_name, "message": f"Node '{node_name}' completed."},
                )
    except Exception as exc:  # noqa: BLE001 - surface unexpected failures over SSE, don't crash silently
        safe_message = _redact(f"Research pipeline failed: {exc}", secrets_to_redact)
        logger.error("Research stream failed: %s", safe_message)
        yield _format_sse("error", {"message": safe_message})
        return

    final_report = state.get("final_report")
    errors = [_redact(e, secrets_to_redact) for e in (state.get("errors") or [])]

    yield _format_sse(
        "final_report",
        {
            "final_report": final_report.model_dump(mode="json") if final_report else None,
            "errors": errors,
        },
    )


@router.post("/research")
def run_research_stream(request: ResearchRequest) -> StreamingResponse:
    """
    Run the full research pipeline for the given prompt:
        analyze -> plan -> search -> process -> summarize
                -> validate -> quality_check -> report
    (with a bounded retry loop between `quality_check` and `plan`),
    streaming node-completion progress events over SSE and finishing with
    a "final_report" event carrying the resulting report plus any errors.
    """
    return StreamingResponse(
        _stream_research_events(request.prompt, request.custom_api_keys),
        media_type="text/event-stream",
    )

