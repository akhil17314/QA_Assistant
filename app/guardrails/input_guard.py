"""Rule-based and model-assisted checks for unsafe user input."""

import logging
import re
from collections.abc import Callable

from pydantic import BaseModel, Field

from app.llm import get_structured_llm
from app.prompts import INJECTION_CLASSIFIER_PROMPT

logger = logging.getLogger(__name__)

INJECTION_PATTERNS = (
    re.compile(r"\bignore\s+(all\s+)?(previous|prior|above)\s+instructions?\b", re.I),
    re.compile(
        r"\b(reveal|show|print|repeat|expose)\s+(your\s+)?(system|developer)\s+prompt\b", re.I
    ),
    re.compile(r"\byou\s+are\s+now\b", re.I),
    re.compile(r"\bdeveloper\s+mode\b", re.I),
    re.compile(r"\bact\s+as\s+(dan|an?\s+unrestricted)\b", re.I),
    re.compile(r"\bextract\s+(the\s+)?(api\s+key|secrets?|credentials?)\b", re.I),
)


class InjectionVerdict(BaseModel):
    """Structured classifier result."""

    safe: bool
    reason: str = Field(default="")


def sanitize_input(text: str) -> str:
    """Remove control characters except ordinary whitespace."""
    return "".join(character for character in text if character in "\n\r\t" or ord(character) >= 32)


def validate_input_rules(text: str, max_length: int = 2000) -> tuple[str, str]:
    """Apply deterministic input checks, returning a safe/blocked verdict and reason."""
    if len(text) > max_length:
        return "blocked", f"Message exceeds the {max_length}-character limit."
    cleaned = sanitize_input(text)
    for pattern in INJECTION_PATTERNS:
        if pattern.search(cleaned):
            return "blocked", "The message matches a prompt-injection or secret-extraction pattern."
    return "safe", ""


def check_input(
    text: str,
    classifier: Callable[[str], InjectionVerdict] | None = None,
) -> tuple[str, str]:
    """Run fast rules followed by a structured LLM safety classifier."""
    verdict, reason = validate_input_rules(text)
    if verdict == "blocked":
        logger.warning("Blocked user input by deterministic guardrail")
        return verdict, reason

    clean_text = sanitize_input(text)
    if classifier is None:
        classifier = get_structured_llm(InjectionVerdict).invoke
    result = classifier(f"{INJECTION_CLASSIFIER_PROMPT}\n\nUser message (untrusted):\n{clean_text}")
    if not isinstance(result, InjectionVerdict):
        result = InjectionVerdict.model_validate(result)
    if not result.safe:
        logger.warning("Blocked user input by classifier")
        return "blocked", result.reason or "The message was blocked by the safety classifier."
    return "safe", ""
