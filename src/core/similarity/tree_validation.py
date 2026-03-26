from __future__ import annotations

from typing import List, Sequence, Set

from domain.models.edit_script import LDPairNode
from domain.models.tree import TreeNode


class TreeValidationError(ValueError):
    pass


class PatchValidationError(ValueError):
    pass


class LDPairValidationError(ValueError):
    """Raised when an LD-pair sequence is invalid (e.g. inconsistent depths)."""
    pass


def validate_ld_pair_sequence(seq: Sequence[LDPairNode]) -> List[LDPairNode]:
    """
    Validate a preorder LD-pair sequence and return it as a list.
    Root must have depth 0; each node's depth must be at most parent_depth + 1.
    """
    items = list(seq)
    if not items:
        raise LDPairValidationError("LD-pair sequence is empty.")
    if items[0].depth != 0:
        raise LDPairValidationError(f"Root node must have depth 0, got {items[0].depth}.")
    stack: List[int] = [0]
    for i in range(1, len(items)):
        d = items[i].depth
        if d <= 0:
            raise LDPairValidationError(f"Non-root node at index {i} has depth {d}.")
        while stack and items[stack[-1]].depth >= d:
            stack.pop()
        if not stack:
            raise LDPairValidationError(
                f"Node at index {i} has depth {d}; no valid parent in preorder."
            )
        stack.append(i)
    return items


def validate_tree(
    root: TreeNode,
    *,
    max_nodes: int = 100_000,
    max_depth: int = 256,
) -> int:
    """
    Validate a TreeNode graph before TED or patching.

    Returns the number of visited nodes.
    Raises TreeValidationError on malformed structure.
    """

    if root is None:
        raise TreeValidationError("Tree root is None.")

    seen: Set[int] = set()
    node_count = 0

    def _walk(node: TreeNode, depth: int) -> None:
        nonlocal node_count

        if depth > max_depth:
            raise TreeValidationError(f"Tree depth exceeded max_depth={max_depth}.")

        object_id = id(node)
        if object_id in seen:
            raise TreeValidationError("Cycle detected in TreeNode graph.")
        seen.add(object_id)

        if not isinstance(node.label, str) or not node.label.strip():
            raise TreeValidationError("Every node must have a non-empty string label.")

        if node.value is not None and not isinstance(node.value, str):
            raise TreeValidationError(
                f"Node '{node.label}' has non-string value of type {type(node.value).__name__}."
            )

        if node.children is None:
            raise TreeValidationError(f"Node '{node.label}' has children=None; expected a list.")
        if not isinstance(node.children, list):
            raise TreeValidationError(f"Node '{node.label}' children must be a list.")

        node_count += 1
        if node_count > max_nodes:
            raise TreeValidationError(f"Tree exceeds max_nodes={max_nodes}.")

        for child in node.children:
            if not isinstance(child, TreeNode):
                raise TreeValidationError(
                    f"Node '{node.label}' contains a non-TreeNode child of type {type(child).__name__}."
                )
            _walk(child, depth + 1)

        seen.remove(object_id)

    _walk(root, depth=0)
    return node_count
