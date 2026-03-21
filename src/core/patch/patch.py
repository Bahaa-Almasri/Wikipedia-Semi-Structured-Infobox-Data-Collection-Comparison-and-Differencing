"""
NOTE:
Zhang–Shasha TED does not use LD-pair operations. For ``algorithm="zhang_shasha"``,
``apply_patch`` / ``apply_patch_from_dict`` require postorder ``mappings`` (from the
Zhang–Shasha alignment) plus the **target** tree, and apply ``apply_zhang_shasha_patch``.

Chawathe / NJ continue to use positional / ref-based edit scripts.

---

Unified tree patching: apply edit script from TED to source tree.
Chawathe (LD-pair), Nierman & Jagadish, and Zhang–Shasha (node mapping) implementations.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from core.similarity.chawathe import chawathe_tree_to_ld_pairs
from core.similarity.tree_validation import (
    LDPairValidationError,
    PatchValidationError,
    validate_ld_pair_sequence,
    validate_tree,
)
from core.patch.zhang_shasha_patch import apply_zhang_shasha_patch
from domain.models.edit_script import EditOperation, LDPairNode, NJEditOperation, NJTedResult, TedResult
from domain.models.tree import TreeNode

ALGORITHM_CHAWATHE = "chawathe"
ALGORITHM_NJ = "nj"
ALGORITHM_ZHANG_SHASHA = "zhang_shasha"


class PatchApplicationError(ValueError):
    pass


# --- Chawathe (LD-pair) patcher ---


def _same_ld_node(a: LDPairNode, b: LDPairNode) -> bool:
    return a.label == b.label and a.value == b.value and a.depth == b.depth


def _ld_pairs_to_tree(seq: Iterable[LDPairNode]) -> TreeNode:
    items = validate_ld_pair_sequence(seq)
    root_item = items[0]
    root = TreeNode(label=root_item.label, value=root_item.value)
    stack: List[tuple[TreeNode, int]] = [(root, root_item.depth)]
    for item in items[1:]:
        node = TreeNode(label=item.label, value=item.value)
        while stack and item.depth <= stack[-1][1]:
            stack.pop()
        if not stack:
            raise LDPairValidationError(
                f"No valid parent found while rebuilding node '{item.label}' at depth {item.depth}."
            )
        parent, _ = stack[-1]
        parent.children.append(node)
        stack.append((node, item.depth))
    validate_tree(root)
    return root


def _apply_chawathe_operations(
    source_root: TreeNode,
    operations: Sequence[EditOperation],
) -> TreeNode:
    validate_tree(source_root)
    working = list(chawathe_tree_to_ld_pairs(source_root))
    for step_no, op in enumerate(operations, start=1):
        if op.position < 0:
            raise PatchApplicationError(f"Operation #{step_no} has negative position {op.position}.")
        if op.op == "insert":
            if op.node is None:
                raise PatchApplicationError(f"Insert operation #{step_no} is missing node payload.")
            if op.position > len(working):
                raise PatchApplicationError(
                    f"Insert operation #{step_no} position {op.position} exceeds sequence length {len(working)}."
                )
            working.insert(op.position, op.node)
            continue
        if op.op == "delete":
            if op.position >= len(working):
                raise PatchApplicationError(
                    f"Delete operation #{step_no} position {op.position} exceeds sequence length {len(working)}."
                )
            current = working[op.position]
            expected = op.node
            if expected is not None and not _same_ld_node(current, expected):
                raise PatchApplicationError(
                    f"Delete operation #{step_no} node mismatch at position {op.position}: "
                    f"expected {expected.to_dict()}, found {current.to_dict()}."
                )
            del working[op.position]
            continue
        if op.op == "update":
            if op.position >= len(working):
                raise PatchApplicationError(
                    f"Update operation #{step_no} position {op.position} exceeds sequence length {len(working)}."
                )
            if op.new_node is None:
                raise PatchApplicationError(f"Update operation #{step_no} is missing new_node payload.")
            current = working[op.position]
            expected_old = op.old_node
            if expected_old is not None and not _same_ld_node(current, expected_old):
                raise PatchApplicationError(
                    f"Update operation #{step_no} node mismatch at position {op.position}: "
                    f"expected {expected_old.to_dict()}, found {current.to_dict()}."
                )
            working[op.position] = op.new_node
            continue
        raise PatchApplicationError(f"Unsupported operation type '{op.op}' at step #{step_no}.")
    return _ld_pairs_to_tree(working)


def _trees_equal_by_ld_pairs(left: TreeNode, right: TreeNode) -> bool:
    left_seq = chawathe_tree_to_ld_pairs(left)
    right_seq = chawathe_tree_to_ld_pairs(right)
    if len(left_seq) != len(right_seq):
        return False
    return all(_same_ld_node(a, b) for a, b in zip(left_seq, right_seq))


# --- NJ patcher ---


def _assign_source_refs(root: TreeNode, prefix: str = "s") -> None:
    counter = 0

    def _walk(node: TreeNode) -> None:
        nonlocal counter
        setattr(node, "_nj_source_ref", f"{prefix}{counter}")
        counter += 1
        for child in node.children:
            _walk(child)

    _walk(root)


def _build_ref_maps(root: TreeNode) -> Tuple[Dict[str, TreeNode], Dict[str, Optional[TreeNode]]]:
    ref_to_node: Dict[str, TreeNode] = {}
    ref_to_parent: Dict[str, Optional[TreeNode]] = {}

    def _walk(node: TreeNode, parent: Optional[TreeNode]) -> None:
        ref = getattr(node, "_nj_source_ref", None)
        if ref is not None:
            ref_to_node[ref] = node
            ref_to_parent[ref] = parent
        for child in node.children:
            _walk(child, node)

    _walk(root, None)
    return ref_to_node, ref_to_parent


def _remove_child(parent: TreeNode, child: TreeNode) -> None:
    for idx, current in enumerate(parent.children):
        if current is child:
            del parent.children[idx]
            return
    raise PatchApplicationError("Failed to remove child from parent; tree structure is inconsistent.")


def _apply_nj_operations(source_root: TreeNode, operations: List[NJEditOperation]) -> TreeNode:
    validate_tree(source_root)
    _assign_source_refs(source_root)
    for step_no, op in enumerate(operations, start=1):
        ref_to_node, ref_to_parent = _build_ref_maps(source_root)
        if op.op == "update":
            if not op.source_ref or op.source_ref not in ref_to_node:
                raise PatchApplicationError(
                    f"Step #{step_no}: update references missing source_ref '{op.source_ref}'."
                )
            node = ref_to_node[op.source_ref]
            if op.new_label is not None:
                node.label = op.new_label
            node.value = op.new_value
            continue
        if op.op == "delete_tree":
            if not op.source_ref or op.source_ref not in ref_to_node:
                raise PatchApplicationError(
                    f"Step #{step_no}: delete_tree references missing source_ref '{op.source_ref}'."
                )
            node = ref_to_node[op.source_ref]
            parent = ref_to_parent[op.source_ref]
            if parent is None:
                raise PatchApplicationError("delete_tree cannot remove the root node.")
            _remove_child(parent, node)
            continue
        if op.op == "insert_tree":
            if not op.parent_ref or op.parent_ref not in ref_to_node:
                raise PatchApplicationError(
                    f"Step #{step_no}: insert_tree references missing parent_ref '{op.parent_ref}'."
                )
            if not op.subtree:
                raise PatchApplicationError(f"Step #{step_no}: insert_tree is missing subtree payload.")
            parent = ref_to_node[op.parent_ref]
            subtree = TreeNode.from_dict(op.subtree)
            insert_index = 0 if op.position is None else max(0, op.position - 1)
            if insert_index > len(parent.children):
                insert_index = len(parent.children)
            parent.children.insert(insert_index, subtree)
            continue
        raise PatchApplicationError(f"Step #{step_no}: unsupported operation type '{op.op}'.")
    validate_tree(source_root)
    return source_root


def _trees_equal_nj(left: TreeNode, right: TreeNode) -> bool:
    if left.label != right.label or left.value != right.value:
        return False
    if len(left.children) != len(right.children):
        return False
    return all(_trees_equal_nj(a, b) for a, b in zip(left.children, right.children))


# --- Unified API ---


def apply_patch_from_dict(
    source_tree_dict: Dict[str, Any],
    edit_script_dict: Dict[str, Any],
    *,
    algorithm: str = ALGORITHM_CHAWATHE,
    target_tree_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Apply a serialized edit script to a source tree (both as dicts).
    edit_script_dict is the output of TedResult.to_dict() or NJTedResult.to_dict().
    For Zhang–Shasha, pass ``target_tree_dict`` and a ``mappings`` list in ``edit_script_dict``.
    """
    al = (algorithm or "").lower()
    source_root = TreeNode.from_dict(source_tree_dict)
    if al == ALGORITHM_ZHANG_SHASHA:
        if target_tree_dict is None:
            raise TypeError("Zhang–Shasha patching requires target_tree_dict.")
        raw = edit_script_dict.get("mappings") or []
        mappings = {int(m["source_id"]): int(m["target_id"]) for m in raw}
        target_root = TreeNode.from_dict(target_tree_dict)
        patched = apply_zhang_shasha_patch(source_root, target_root, mappings)
        return patched.to_dict()
    ops_raw = edit_script_dict.get("operations") or []
    if al == ALGORITHM_NJ:
        operations = [NJEditOperation.from_dict(o) for o in ops_raw]
        patched = _apply_nj_operations(source_root, operations)
    else:
        operations = [EditOperation.from_dict(o) for o in ops_raw]
        patched = _apply_chawathe_operations(source_root, operations)
    return patched.to_dict()


def apply_patch(
    source_root: TreeNode,
    ted_result: Union[TedResult, NJTedResult],
    *,
    algorithm: str = ALGORITHM_CHAWATHE,
    target_root: Optional[TreeNode] = None,
) -> TreeNode:
    """
    Apply the edit script from a TED result to the source tree.
    algorithm must match the algorithm that produced ted_result ("chawathe" or "nj").
    For Zhang–Shasha, pass ``target_root`` and ``ted_result.zhang_shasha_mappings``.
    """
    al = (algorithm or "").lower()
    if al == ALGORITHM_ZHANG_SHASHA:
        if target_root is None:
            raise TypeError("Zhang–Shasha requires target_root for apply_patch.")
        if not isinstance(ted_result, TedResult) or not ted_result.zhang_shasha_mappings:
            raise TypeError("Zhang–Shasha requires TedResult with zhang_shasha_mappings.")
        mp = {m["source_id"]: m["target_id"] for m in ted_result.zhang_shasha_mappings}
        return apply_zhang_shasha_patch(source_root, target_root, mp)
    if al == ALGORITHM_NJ:
        if not isinstance(ted_result, NJTedResult):
            raise TypeError("NJ algorithm requires NJTedResult.")
        return _apply_nj_operations(source_root, ted_result.operations)
    if not isinstance(ted_result, TedResult):
        raise TypeError("Chawathe algorithm requires TedResult.")
    return _apply_chawathe_operations(source_root, ted_result.operations)


def trees_equal(left: TreeNode, right: TreeNode, *, algorithm: str = ALGORITHM_NJ) -> bool:
    """
    Return True if two trees are equal.

    Chawathe: LD-pair sequence equality (same as patch replay semantics).
    NJ: ordered recursive label/value/children equality.

    Zhang–Shasha (distance-only algorithm): does not define a patch representation. Callers
    comparing ``patched`` vs ``target`` after a skipped patch should not use this for
    ``algorithm==zhang_shasha``; here we fall back to the same structural check as NJ
    (label/value/children) when callers pass a generic ``algorithm`` string.
    """
    al = (algorithm or "").lower()
    if al == ALGORITHM_NJ or al == ALGORITHM_ZHANG_SHASHA:
        return _trees_equal_nj(left, right)
    return _trees_equal_by_ld_pairs(left, right)
