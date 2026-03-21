"""
Zhang–Shasha tree edit distance — re-export for the services/ted layout.

Implementation lives in core.similarity.zhang_shasha.
"""
from core.similarity.zhang_shasha import (
    compute_ted_zhang_shasha,
    normalize_tree,
    zhang_shasha_distance,
)

__all__ = [
    "compute_ted_zhang_shasha",
    "normalize_tree",
    "zhang_shasha_distance",
]
