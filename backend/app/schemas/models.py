"""
Pydantic schemas for the AI Research Assistant pipeline.

These models define the data contracts between pipeline stages
(planning -> search -> findings -> report). No pipeline logic,
LLM calls, or Tavily integration live here yet — schemas only.
"""

from pydantic import BaseModel, Field, HttpUrl


class SearchQuery(BaseModel):
    """A single search query generated as part of a research plan."""

    query: str = Field(..., min_length=1, description="The search query text.")
    purpose: str = Field(..., min_length=1, description="Why this query is being run.")


class ResearchPlan(BaseModel):
    """The plan produced from an original user prompt, before searching begins."""

    original_prompt: str = Field(..., min_length=1)
    main_topic: str = Field(..., min_length=1)
    sub_topics: list[str] = Field(default_factory=list)
    search_queries: list[SearchQuery] = Field(default_factory=list)


class SearchResult(BaseModel):
    """A single raw result returned by a search provider."""

    title: str = Field(..., min_length=1)
    url: HttpUrl
    content: str
    relevance_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Optional relevance score (0-1)."
    )


class Finding(BaseModel):
    """A synthesized finding for one sub-topic, derived from search results."""

    topic: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    key_points: list[str] = Field(default_factory=list)
    sources: list[HttpUrl] = Field(default_factory=list)


class ResearchReport(BaseModel):
    """The final assembled research report."""

    title: str = Field(..., min_length=1)
    executive_summary: str = Field(..., min_length=1)
    key_findings: list[str] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    sources: list[HttpUrl] = Field(default_factory=list)
