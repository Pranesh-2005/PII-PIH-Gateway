"""PHI/PII de-identification gateway.

Public surface is two functions, matching the assessment brief exactly::

    deidentify(text) -> (masked_text, mapping)
    rehydrate(response, mapping) -> text
"""

from .types import Category, DeidResult, Mapping, Span  # noqa: F401
from .pipeline import deidentify, deidentify_full, rehydrate  # noqa: F401

__version__ = "0.1.0"
__all__ = [
    "deidentify",
    "deidentify_full",
    "rehydrate",
    "Category",
    "DeidResult",
    "Mapping",
    "Span",
]
