"""Structured answer generation with document citations."""

import logging

from app.guardrails.output_guard import filter_citations, redact_secrets, validate_schema
from app.llm import get_structured_llm
from app.prompts import ANSWER_PROMPT
from app.state import AnswerOutput

logger = logging.getLogger(__name__)


def _format_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{index}] source={chunk.get('source', 'unknown')} page={chunk.get('page', 1)}\n"
        f"{chunk.get('text', '')}"
        for index, chunk in enumerate(chunks, start=1)
    )


def answer_question(query: str, chunks: list[dict]) -> AnswerOutput:
    """Generate and validate an answer, retrying malformed structured output once."""
    llm = get_structured_llm(AnswerOutput)
    prompt = f"{ANSWER_PROMPT}\n\nContext:\n{_format_context(chunks)}\n\nQuestion:\n{query}"
    for attempt in range(2):
        try:
            answer = filter_citations(validate_schema(llm.invoke(prompt)), chunks)
            return answer.model_copy(
                update={
                    "answer": redact_secrets(answer.answer),
                    "citations": [
                        citation.model_copy(update={"snippet": redact_secrets(citation.snippet)})
                        for citation in answer.citations
                    ],
                }
            )
        except Exception as error:
            logger.warning(
                "Answer generation failed on attempt %s: %s: %s",
                attempt + 1,
                type(error).__name__,
                error,
                exc_info=True,
            )
    return AnswerOutput(
        answer="I could not produce a validated answer from the available documents.",
        citations=[],
        confidence="low",
    )
