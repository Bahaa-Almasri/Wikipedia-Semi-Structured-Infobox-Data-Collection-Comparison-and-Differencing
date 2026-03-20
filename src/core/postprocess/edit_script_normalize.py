"""
Post-process Chawathe (LD-pair) edit scripts for display and API responses:

- Drop operations on ignored metadata paths (noise).
- Split updates where the node label changes into delete + insert (semantic clarity).
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Sequence

from core.patch.patch import PatchApplicationError, _ld_pairs_to_tree
from core.similarity.chawathe import chawathe_tree_to_ld_pairs
from domain.models.edit_script import EditOperation, LDPairNode
from domain.models.tree import TreeNode

# Fields that should not appear in the returned edit script (noise / provenance).
IGNORE_FIELDS: frozenset[str] = frozenset(
    {
        "retrieved_at",
        "slug",
        "wikipedia_url",
        "source",
    }
)


def _segment_base(segment: str) -> str:
    return segment.split("[", 1)[0]


def ignored_path(path: Sequence[str]) -> bool:
    """True if any path segment (or list key base) is in IGNORE_FIELDS."""
    for segment in path:
        if _segment_base(segment) in IGNORE_FIELDS:
            return True
    return False


def _preorder_paths(root: TreeNode) -> List[List[str]]:
    paths: List[List[str]] = []
    stack: List[str] = []

    def walk(node: TreeNode) -> None:
        stack.append(node.label)
        paths.append(list(stack))
        for child in node.children:
            walk(child)
        stack.pop()

    walk(root)
    return paths


def _path_for_delete_or_update(working: List[LDPairNode], position: int) -> List[str]:
    root = _ld_pairs_to_tree(working)
    paths = _preorder_paths(root)
    if position < 0 or position >= len(paths):
        return []
    return paths[position]


def _path_for_insert(working: List[LDPairNode], position: int, node: LDPairNode) -> List[str]:
    w2: List[LDPairNode] = list(working)
    w2.insert(position, node)
    root = _ld_pairs_to_tree(w2)
    paths = _preorder_paths(root)
    if position < 0 or position >= len(paths):
        return []
    return paths[position]


def _should_skip_operation(op: EditOperation, working: List[LDPairNode]) -> bool:
    if op.op == "insert":
        if op.node is None:
            return False
        return ignored_path(_path_for_insert(working, op.position, op.node))
    if op.op == "delete":
        return ignored_path(_path_for_delete_or_update(working, op.position))
    if op.op == "update":
        return ignored_path(_path_for_delete_or_update(working, op.position))
    return False


def _apply_one(working: List[LDPairNode], op: EditOperation) -> None:
    if op.position < 0:
        raise PatchApplicationError(f"Negative position {op.position}.")
    if op.op == "insert":
        if op.node is None:
            raise PatchApplicationError("Insert operation is missing node payload.")
        if op.position > len(working):
            raise PatchApplicationError(
                f"Insert position {op.position} exceeds sequence length {len(working)}."
            )
        working.insert(op.position, op.node)
        return
    if op.op == "delete":
        if op.position >= len(working):
            raise PatchApplicationError(
                f"Delete position {op.position} exceeds sequence length {len(working)}."
            )
        del working[op.position]
        return
    if op.op == "update":
        if op.position >= len(working):
            raise PatchApplicationError(
                f"Update position {op.position} exceeds sequence length {len(working)}."
            )
        if op.new_node is None:
            raise PatchApplicationError("Update operation is missing new_node payload.")
        working[op.position] = op.new_node
        return
    raise PatchApplicationError(f"Unsupported operation type '{op.op}'.")


def _paired_split_should_skip(delete_op: EditOperation, insert_op: EditOperation, working: List[LDPairNode]) -> bool:
    """
    For update split into delete+insert: skip both if either step targets an ignored path.
    Insert path is evaluated on the sequence *after* the delete (matches patch semantics).
    """
    if _should_skip_operation(delete_op, working):
        return True
    w_after = list(working)
    _apply_one(w_after, delete_op)
    return _should_skip_operation(insert_op, w_after)


def _expand_label_mismatch(op: EditOperation) -> List[EditOperation]:
    """Turn label-changing updates into delete(old) + insert(new) at the same index."""
    if op.op != "update" or op.old_node is None or op.new_node is None:
        return [op]
    if op.old_node.label == op.new_node.label:
        return [op]
    return [
        EditOperation(
            op="delete",
            position=op.position,
            node=op.old_node,
            note=(op.note or "") + " [split: delete old label]",
        ),
        EditOperation(
            op="insert",
            position=op.position,
            node=op.new_node,
            note=(op.note or "") + " [split: insert new label]",
        ),
    ]


def normalize_chawathe_edit_script(
    source_root: TreeNode,
    operations: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Replay operations on the source LD-pair sequence: drop ignored paths, split
    label-changing updates. Positions stay consistent with sequential application.
    """
    working = list(chawathe_tree_to_ld_pairs(source_root))
    out: List[Dict[str, Any]] = []

    for raw in operations:
        op = EditOperation.from_dict(deepcopy(raw))
        steps = _expand_label_mismatch(op)
        if (
            len(steps) == 2
            and steps[0].op == "delete"
            and steps[1].op == "insert"
        ):
            if _paired_split_should_skip(steps[0], steps[1], working):
                continue
            for step in steps:
                out.append(step.to_dict())
                _apply_one(working, step)
            continue
        for step in steps:
            if _should_skip_operation(step, working):
                continue
            out.append(step.to_dict())
            _apply_one(working, step)

    return out


def normalize_nj_edit_script(operations: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Light NJ cleanup: drop operations whose primary label is in IGNORE_FIELDS.
    (NJ does not use LD-pair positions; full path-based replay would require ref maps.)
    """
    out: List[Dict[str, Any]] = []
    for raw in operations:
        op = raw.get("op")
        if op == "update":
            if raw.get("old_label") in IGNORE_FIELDS or raw.get("new_label") in IGNORE_FIELDS:
                continue
        elif op == "delete_tree":
            if raw.get("old_label") in IGNORE_FIELDS:
                continue
        elif op == "insert_tree":
            subtree = raw.get("subtree") or {}
            if isinstance(subtree, dict) and subtree.get("label") in IGNORE_FIELDS:
                continue
        out.append(dict(raw))
    return out


def normalize_edit_script_for_algorithm(
    algorithm: str,
    source_root: TreeNode,
    operations: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Dispatch normalization by TED algorithm name."""
    al = (algorithm or "").lower()
    if "nj" in al or "nierman" in al:
        return normalize_nj_edit_script(operations)
    return normalize_chawathe_edit_script(source_root, operations)
