"""Typed data contracts shared by the graph and its agents."""

from typing import Literal, TypedDict

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A source location supporting an answer."""

    source: str
    page: int = Field(ge=1)
    snippet: str


class AnswerOutput(BaseModel):
    """Structured user-facing answer with document citations."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]


class GraphState(TypedDict, total=False):
    """State passed between LangGraph nodes."""

    query: str
    history: list[dict]
    collection: str
    retrieval_query: str
    input_verdict: Literal["safe", "blocked"]
    block_reason: str
    route: Literal["in_scope", "off_topic"]
    chunks: list[dict]
    answer: AnswerOutput | None
    grounded: bool
    grounding_report: str
    retries: int
    llm_calls: int
    final_response: str
