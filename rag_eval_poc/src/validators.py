"""
Validation module for RAG Bot
"""
from typing import Any, Tuple
from config import config
import logging
import html

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error"""
    pass


class InputValidator:
    """Validates user inputs"""

    @staticmethod
    def validate_question(question: str) -> Tuple[bool, str]:
        """
        Validate user question
        Returns: (is_valid, error_message)
        """
        if not isinstance(question, str):
            return False, "Question must be a string"

        # XSS protection: escape HTML characters
        question = html.escape(question, quote=True)
        question = question.strip()

        if len(question) == 0:
            return False, "Question cannot be empty"

        if len(question) < config.MIN_QUESTION_LENGTH:
            return False, f"Question must be at least {config.MIN_QUESTION_LENGTH} characters"

        if len(question) > config.MAX_QUESTION_LENGTH:
            return False, f"Question must not exceed {config.MAX_QUESTION_LENGTH} characters"

        # Check for valid characters (alphanumeric, spaces, punctuation)
        if not any(c.isalnum() for c in question):
            return False, "Question must contain at least one alphanumeric character"

        return True, ""

    @staticmethod
    def validate_file_path(file_path: str) -> Tuple[bool, str]:
        """
        Validate file path
        Returns: (is_valid, error_message)
        """
        from pathlib import Path

        if not isinstance(file_path, str):
            return False, "File path must be a string"

        path = Path(file_path)

        if not path.exists():
            return False, f"File does not exist: {file_path}"

        if not path.is_file():
            return False, f"Path is not a file: {file_path}"

        if path.suffix.lower() not in config.ALLOWED_EXTENSIONS:
            return False, f"File type not supported. Allowed: {config.ALLOWED_EXTENSIONS}"

        # Check file size (max 100MB)
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > 100:
            return False, f"File is too large ({file_size_mb:.2f}MB). Maximum is 100MB"

        return True, ""


class OutputValidator:
    """Validates bot outputs"""

    @staticmethod
    def validate_answer(answer: str) -> Tuple[bool, str]:
        """
        Validate bot answer
        Returns: (is_valid, error_message)
        """
        if not isinstance(answer, str):
            return False, "Answer must be a string"

        # XSS protection: escape HTML characters
        answer = html.escape(answer, quote=True)

        if len(answer) == 0:
            return False, "Answer cannot be empty"

        if len(answer) < config.MIN_ANSWER_LENGTH:
            return False, f"Answer seems too short (< {config.MIN_ANSWER_LENGTH} chars)"

        if len(answer) > config.MAX_ANSWER_LENGTH:
            return False, f"Answer exceeds maximum length ({config.MAX_ANSWER_LENGTH} chars)"

        return True, ""

    @staticmethod
    def validate_confidence_score(score: float) -> Tuple[bool, str]:
        """
        Validate confidence score
        Returns: (is_valid, error_message)
        """
        if not isinstance(score, (int, float)):
            return False, "Score must be a number"

        if not (0 <= score <= 1):
            return False, "Score must be between 0 and 1"

        return True, ""


class DocumentValidator:
    """Validates documents"""

    @staticmethod
    def validate_chunks(chunks: list) -> Tuple[bool, str]:
        """
        Validate document chunks
        Returns: (is_valid, error_message)
        """
        if not isinstance(chunks, list):
            return False, "Chunks must be a list"

        if len(chunks) == 0:
            return False, "No chunks found in document"

        for i, chunk in enumerate(chunks):
            if not hasattr(chunk, 'page_content'):
                return False, f"Chunk {i} is missing 'page_content' attribute"

            if not isinstance(chunk.page_content, str):
                return False, f"Chunk {i} content must be a string"

            if len(chunk.page_content.strip()) == 0:
                return False, f"Chunk {i} is empty"

        return True, ""

    @staticmethod
    def validate_metadata(metadata: dict) -> Tuple[bool, str]:
        """
        Validate document metadata
        Returns: (is_valid, error_message)
        """
        if not isinstance(metadata, dict):
            return False, "Metadata must be a dictionary"

        # Check for required fields
        required_fields = ["source"]
        for field in required_fields:
            if field not in metadata:
                return False, f"Missing required metadata field: {field}"

        return True, ""


def validate_or_raise(condition: bool, message: str):
    """Helper function to validate or raise error"""
    if not condition:
        raise ValidationError(message)
