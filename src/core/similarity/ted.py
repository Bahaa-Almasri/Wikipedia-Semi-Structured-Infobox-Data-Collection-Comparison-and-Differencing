"""
Unified Tree Edit Distance (TED) entry point.
Dispatches to Chawathe (LD-pair), Nierman & Jagadish, or Zhang–Shasha based on algorithm=.
"""
from __future__ import annotations

from typing import Optional, Union

from domain.models.edit_script import NJTedResult, TedResult
from domain.models.tree import TreeNode

from core.similarity.common import clone_tree
from core.similarity.chawathe import chawathe_tree_to_ld_pairs, compute_ted_chawathe
from core.similarity.nj import compute_ted_nj
from core.similarity.zhang_shasha import compute_ted_zhang_shasha

ALGORITHM_CHAWATHE = "chawathe"
ALGORITHM_NJ = "nj"
ALGORITHM_ZHANG_SHASHA = "zhang_shasha"


def compute_ted(
    source_root: TreeNode,
    target_root: TreeNode,
    *,
    algorithm: str = ALGORITHM_CHAWATHE,
    coerce_root_label: Optional[str] = None,
) -> Union[TedResult, NJTedResult]:
    """
    Compute TED between two trees using the chosen algorithm.

    algorithm: "chawathe" (LD-pair), "nj" (Nierman & Jagadish), or "zhang_shasha".
    Returns TedResult (Chawathe / Zhang–Shasha) or NJTedResult (NJ); both have .to_dict() for JSON.
    """
    al = (algorithm or "").lower()
    if al == ALGORITHM_NJ:
        return compute_ted_nj(
            source_root, target_root, coerce_root_label=coerce_root_label
        )
    if al == ALGORITHM_ZHANG_SHASHA:
        return compute_ted_zhang_shasha(
            source_root, target_root, coerce_root_label=coerce_root_label
        )
    return compute_ted_chawathe(
        source_root, target_root, coerce_root_label=coerce_root_label
    )


def tree_to_ld_pairs(root: TreeNode):
    """Export tree to LD-pair sequence (Chawathe). Alias for chawathe_tree_to_ld_pairs."""
    return chawathe_tree_to_ld_pairs(root)
