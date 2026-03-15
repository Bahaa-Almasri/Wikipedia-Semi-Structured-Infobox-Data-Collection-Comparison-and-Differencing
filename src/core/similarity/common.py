"""Shared helpers for TED algorithms (Chawathe and NJ)."""
from __future__ import annotations

from domain.models.tree import TreeNode


def clone_tree(node: TreeNode) -> TreeNode:
    """Deep-clone a TreeNode. Used by both Chawathe and NJ algorithms."""
    return TreeNode(
        label=node.label,
        value=node.value,
        children=[clone_tree(child) for child in node.children],
    )


def similarity_from_distance(distance: int, size_a: int, size_b: int) -> float:
    """Convert edit distance to similarity in [0, 1]. Used by both algorithms."""
    denom = size_a + size_b
    if denom == 0:
        return 1.0
    return max(0.0, 1.0 - (distance / denom))
