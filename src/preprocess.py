import re
from typing import Iterable

import pandas as pd

# Keep Bengali characters and a small punctuation set.
_KEEP_PATTERN = re.compile(r"[^ \t\n\r\u0980-\u09FF.,!?;:।“”\"'()\-]")
_MULTISPACE = re.compile(r"\s+")


def clean_bengali_text(text: str) -> str:
    """Normalize text by keeping Bengali characters and lightweight punctuation."""
    if not isinstance(text, str):
        return ""
    cleaned = _KEEP_PATTERN.sub(" ", text.strip())
    cleaned = _MULTISPACE.sub(" ", cleaned)
    return cleaned.strip()


def preprocess_text_series(texts: Iterable[str]) -> pd.Series:
    """Apply Bengali cleaning to a sequence of texts."""
    return pd.Series([clean_bengali_text(t) for t in texts], dtype="string")
