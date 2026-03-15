"""
Unified Tree Edit Distance (TED) entry point.
Dispatches to Chawathe (LD-pair) or Nierman & Jagadish algorithm based on algorithm=.
"""
from __future__ import annotations

from typing import Optional, Union

from domain.models.edit_script import NJTedResult, TedResult
from domain.models.tree import TreeNode

from core.similarity.common import clone_tree
from core.similarity.chawathe import chawathe_tree_to_ld_pairs, compute_ted_chawathe
from core.similarity.nj import compute_ted_nj

ALGORITHM_CHAWATHE = "chawathe"
ALGORITHM_NJ = "nj"


def compute_ted(
    source_root: TreeNode,
    target_root: TreeNode,
    *,
    algorithm: str = ALGORITHM_CHAWATHE,
    coerce_root_label: Optional[str] = None,
) -> Union[TedResult, NJTedResult]:
    """
    Compute TED between two trees using the chosen algorithm.

    algorithm: "chawathe" (LD-pair) or "nj" (Nierman & Jagadish).
    Returns TedResult (Chawathe) or NJTedResult (NJ); both have .to_dict() for JSON.
    """
    if algorithm == ALGORITHM_NJ:
        return compute_ted_nj(
            source_root, target_root, coerce_root_label=coerce_root_label
        )
    return compute_ted_chawathe(
        source_root, target_root, coerce_root_label=coerce_root_label
    )


def tree_to_ld_pairs(root: TreeNode):
    """Export tree to LD-pair sequence (Chawathe). Alias for chawathe_tree_to_ld_pairs."""
    return chawathe_tree_to_ld_pairs(root)
