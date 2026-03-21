"""
Node-based patching for Zhang–Shasha TED alignments (postorder id ↔ postorder id).

Does not use LD-pair positions. Requires a mapping dict from optimal alignment recovery.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from core.similarity.common import clone_tree
from core.similarity.zhang_shasha import normalize_tree as zs_normalize_tree
from core.similarity.zhang_shasha import postorder_nodes_and_zs_ids
from domain.models.tree import TreeNode


def find_parent(root: TreeNode, node: TreeNode) -> Optional[TreeNode]:
    """Return parent of ``node`` in tree rooted at ``root``, or None if ``node`` is root / not found."""
    if root is node:
        return None

    def walk(n: TreeNode) -> Optional[TreeNode]:
        for ch in n.children:
            if ch is node:
                return n
            r = walk(ch)
            if r is not None:
                return r
        return None

    return walk(root)


def remove_node(parent: TreeNode, child: TreeNode) -> None:
    for idx, c in enumerate(parent.children):
        if c is child:
            del parent.children[idx]
            return
    raise ValueError("Child not found under parent.")


def insert_node(parent: TreeNode, child: TreeNode, position: int) -> None:
    pos = max(0, min(position, len(parent.children)))
    parent.children.insert(pos, child)


def _trees_equal_struct(a: TreeNode, b: TreeNode) -> bool:
    if a.label != b.label or a.value != b.value:
        return False
    if len(a.children) != len(b.children):
        return False
    return all(_trees_equal_struct(x, y) for x, y in zip(a.children, b.children))


def apply_zhang_shasha_patch(
    source_root: TreeNode,
    target_root: TreeNode,
    mappings: Dict[int, int],
) -> TreeNode:
    """
    Transform ``source_root`` toward ``target_root`` using postorder id alignment.

    ``mappings`` maps source postorder id → target postorder id (from the same TED run
    as the trees, after identical ``normalize_tree`` if used before TED).

    Rebuilds each mapped node's children list to match the target (order-preserving),
    cloning unmapped target subtrees. Uses the same postorder numbering as TED via
    ``postorder_nodes_and_zs_ids`` (``_AnnotatedTree``).

    Clones ``source_root``; does not mutate the original ``source_root`` / ``target_root``.
    """
    src = clone_tree(source_root)
    tgt = clone_tree(target_root)
    zs_normalize_tree(src)
    zs_normalize_tree(tgt)

    _, s_map = postorder_nodes_and_zs_ids(src)
    _, _t_map = postorder_nodes_and_zs_ids(tgt)

    mapped_tgt_ids = set(mappings.values())

    tgt_to_src: Dict[int, TreeNode] = {
        tid: s_map[sid] for sid, tid in mappings.items() if sid in s_map and tid in _t_map
    }

    tgt_root = tgt
    if tgt_root._zs_id not in mapped_tgt_ids:
        raise ValueError(
            "Zhang–Shasha patch: target root is not in the alignment mapping; "
            "cannot rebuild the tree."
        )

    def build(tp: TreeNode) -> TreeNode:
        if tp._zs_id not in mapped_tgt_ids:
            return clone_tree(tp)
        sc = tgt_to_src[tp._zs_id]
        sc.label = tp.label
        sc.value = tp.value
        sc.children = [build(tc) for tc in tp.children]
        return sc

    out = build(tgt_root)
    assert _trees_equal_struct(out, tgt_root), "Zhang–Shasha patch did not reproduce target tree."
    return out
