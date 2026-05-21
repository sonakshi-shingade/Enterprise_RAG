import logging
import re
from typing import Dict, List

try:
    from langdetect import detect, LangDetectException
except ImportError:  # pragma: no cover
    detect = None
    LangDetectException = Exception

logger = logging.getLogger(__name__)


class InputValidationError(ValueError):
    """Raised when user input fails guardrail validation."""


class InputGuardrails:
    """Production-grade input guardrails for user queries."""

    MAX_QUERY_LENGTH = 5000
    MIN_QUERY_LENGTH = 2
    SUPPORTED_LANGUAGES = {"en"}

    BLOCKED_PATTERNS: List[str] = [
        "ignore previous instructions",
        "forget previous instructions",
        "reveal system prompt",
        "bypass security",
        "developer mode",
        "jailbreak",
        "show confidential data",
        "do not answer",
        "do not respond",
        "disable safety",
        "override all restrictions",
    ]

    PII_PATTERNS: List[str] = [
        r"\b\d{12}\b",  # Aadhaar
        r"[A-Z]{5}[0-9]{4}[A-Z]{1}",  # PAN
        r"\b\d{10}\b",  # phone number
        r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b",  # SSN-style
        r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",  # credit card
        r"\b\d{15,16}\b",  # credit card fallback
        r"\S+@\S+\.\S+",  # email
        r"\b[A-Z]{2}\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\b",  # passport / PAN-like fallback
    ]

    UNSAFE_WORDS: List[str] = [
        "hack",
        "kill",
        "bomb",
        "steal",
        "attack",
        "exploit",
        "malware",
        "terror",
        "threat",
    ]

    ALLOWED_TOPICS: List[str] = [
        "loan",
        "bank",
        "interest",
        "credit",
        "investment",
        "finance",
        "account",
        "transaction",
        "mortgage",
        "savings",
    ]

    def __init__(self, enforce_domain: bool = True) -> None:
        self.enforce_domain = enforce_domain
        self._pii_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.PII_PATTERNS
        ]

    @staticmethod
    def normalize(query: str) -> str:
        return re.sub(r"\s+", " ", query or "").strip()

    def detect_prompt_injection(self, query: str) -> bool:
        normalized = query.lower()
        return any(pattern in normalized for pattern in self.BLOCKED_PATTERNS)

    def detect_pii(self, query: str) -> bool:
        for compiled in self._pii_patterns:
            if compiled.search(query):
                return True
        return False

    def detect_unsafe_content(self, query: str) -> bool:
        normalized = query.lower()
        return any(word in normalized for word in self.UNSAFE_WORDS)

    def validate_length(self, query: str) -> bool:
        length = len(query)
        return self.MIN_QUERY_LENGTH <= length <= self.MAX_QUERY_LENGTH

    def validate_language(self, query: str) -> bool:
        if detect is None:
            logger.warning(
                "langdetect is not installed; skipping language validation."
            )
            return True

        try:
            detected_language = detect(query)
            return detected_language in self.SUPPORTED_LANGUAGES
        except LangDetectException:
            logger.warning(
                "Language detection failed for query; blocking by default."
            )
            return False

    def validate_domain(self, query: str) -> bool:
        if not self.enforce_domain:
            return True

        normalized = query.lower()
        return any(topic in normalized for topic in self.ALLOWED_TOPICS)

    def validate(self, query: str) -> Dict[str, str]:
        query = self.normalize(query)

        if not query:
            return {
                "status": "BLOCKED",
                "reason": "Empty query is not allowed.",
            }

        if not self.validate_length(query):
            return {
                "status": "BLOCKED",
                "reason": (
                    f"Query length must be between {self.MIN_QUERY_LENGTH} "
                    f"and {self.MAX_QUERY_LENGTH} characters."
                ),
            }

        if self.detect_prompt_injection(query):
            return {
                "status": "BLOCKED",
                "reason": "Prompt injection pattern detected.",
            }

        if self.detect_pii(query):
            return {
                "status": "BLOCKED",
                "reason": "Potential PII or sensitive data detected.",
            }

        if self.detect_unsafe_content(query):
            return {
                "status": "BLOCKED",
                "reason": "Unsafe or malicious content detected.",
            }

        if not self.validate_language(query):
            return {
                "status": "BLOCKED",
                "reason": "Unsupported or undetectable language.",
            }

        if not self.validate_domain(query):
            return {
                "status": "BLOCKED",
                "reason": "Query is outside the allowed domain.",
            }

        return {
            "status": "SAFE",
            "reason": "Validation Passed.",
            "query": query,
        }

    def assert_safe_query(self, query: str) -> Dict[str, str]:
        validation = self.validate(query)

        if validation["status"] != "SAFE":
            logger.warning(
                "Input guardrail blocked request: %s", validation["reason"]
            )
            raise InputValidationError(
                validation["reason"]
            )

        logger.info(
            "Input guardrail validation succeeded for query: %s",
            validation["query"],
        )

        return validation
input_length(self,)