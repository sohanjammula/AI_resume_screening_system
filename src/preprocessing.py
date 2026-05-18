"""Text preprocessing utilities for resume classification."""

from __future__ import annotations

import re


_WHITESPACE_RE = re.compile(r"\s+")
_NON_TEXT_RE = re.compile(r"[^a-zA-Z0-9+#.\s-]")


def clean_resume_text(text: str) -> str:
    """Normalize noisy resume text while preserving useful skill tokens."""
    text = text.lower()
    text = _NON_TEXT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()
