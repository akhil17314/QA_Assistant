"""Structured document-scope classification."""

from typing import Literal

from pydantic import BaseModel, field_validator

from app.llm import get_structured_llm
from app.prompts import ROUTER_PROMPT


class RouteDecision(BaseModel):
    """Scope decision returned by the router model."""

    route: Literal["in_scope", "off_topic"]
    reason: str = ""

    @field_validator("route", mode="before")
    @classmethod
    def _normalize_route(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower().replace("-", "_").replace(" ", "_")
        return value


def route_query(query: str) -> RouteDecision:
    """Classify a question as document-answerable or off-topic."""
    llm = get_structured_llm(RouteDecision)
    return llm.invoke(f"{ROUTER_PROMPT}\n\nUser question (untrusted):\n{query}")
