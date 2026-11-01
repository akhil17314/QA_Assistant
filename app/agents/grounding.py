"""Structured claim-by-claim groundedness checking."""

from pydantic import BaseModel, Field

from app.config import settings
from app.llm import get_structured_llm
from app.prompts import CLAIM_EXTRACTION_PROMPT, GROUNDING_JUDGE_PROMPT
from app.state import AnswerOutput


class ClaimSet(BaseModel):
    """Atomic claims extracted from an answer."""

    claims: list[str] = Field(default_factory=list)


class ClaimVerdict(BaseModel):
    """Judge result for one atomic claim."""

    claim: str
    supported: bool


class GroundingVerdict(BaseModel):
    """Structured grounding decision."""

    verdicts: list[ClaimVerdict] = Field(default_factory=list)
    report: str = ""


def check_grounding(answer: AnswerOutput, chunks: list[dict]) -> tuple[bool, str]:
    """Return true only when every factual claim is supported by retrieved text."""
    normalized_answer = answer.answer.strip().lower()
    if not normalized_answer or normalized_answer.startswith(
        ("i don't know", "i do not know", "the documents do not contain")
    ):
        return True, "The answer explicitly abstains because the documents do not provide it."
    llm = get_structured_llm(ClaimSet, model=settings.judge_model)
    claims = llm.invoke(f"{CLAIM_EXTRACTION_PROMPT}\n\nAnswer:\n{answer.answer}")
    if not claims.claims:
        return True, "No factual claims were identified."
    context = _format_chunks(chunks)
    judge = get_structured_llm(GroundingVerdict, model=settings.judge_model)
    result = judge.invoke(
        f"{GROUNDING_JUDGE_PROMPT}\n\nClaims:\n{claims.model_dump_json()}\n\n"
        f"Document excerpts:\n{context}"
    )
    unsupported = [item.claim for item in result.verdicts if not item.supported]
    all_claims_checked = len(result.verdicts) == len(claims.claims)
    grounded = not unsupported and all_claims_checked
    report = result.report or (
        "All claims are supported." if grounded else "Unsupported claims: " + "; ".join(unsupported)
    )
    return grounded, report


def _format_chunks(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"source={item.get('source', 'unknown')} page={item.get('page', 1)}\n{item.get('text', '')}"
        for item in chunks
    )
