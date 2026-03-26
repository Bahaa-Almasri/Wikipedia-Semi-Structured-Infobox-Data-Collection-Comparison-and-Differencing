from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from core.similarity.common import clone_tree, similarity_from_distance
from core.similarity.tree_validation import validate_tree
from domain.models.edit_script import EditOperation, LDPairNode, TedResult
from domain.models.tree import TreeNode


INF = 10 ** 9


@dataclass(frozen=True)
class AlignmentStep:
    kind: str  # keep | update | delete | insert
    a_node: Optional[LDPairNode] = None
    b_node: Optional[LDPairNode] = None


def chawathe_tree_to_ld_pairs(root: TreeNode) -> List[LDPairNode]:
    validate_tree(root)
    items: List[LDPairNode] = []

    def _walk(node: TreeNode, depth: int) -> None:
        items.append(
            LDPairNode(
                label=node.label,
                value=node.value,
                depth=depth,
                preorder_index=len(items),
            )
        )
        for child in node.children:
            _walk(child, depth + 1)

    _walk(root, depth=0)
    return items


def _node_update_cost(a: LDPairNode, b: LDPairNode) -> int:
    return 0 if (a.label == b.label and a.value == b.value) else 1


def _update_allowed(a: LDPairNode, b: LDPairNode) -> bool:
    return a.depth == b.depth


def _delete_allowed(a: LDPairNode, b: LDPairNode, *, j: int, n: int) -> bool:
    return j == n or a.depth >= b.depth


def _insert_allowed(a: LDPairNode, b: LDPairNode, *, i: int, m: int) -> bool:
    return i == m or b.depth >= a.depth


def _compute_matrix(a_seq: Sequence[LDPairNode], b_seq: Sequence[LDPairNode]) -> List[List[int]]:
    m = len(a_seq)
    n = len(b_seq)
    dist = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dist[i][0] = dist[i - 1][0] + 1
    for j in range(1, n + 1):
        dist[0][j] = dist[0][j - 1] + 1

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            a_node = a_seq[i - 1]
            b_node = b_seq[j - 1]

            delete_cost = INF
            if _delete_allowed(a_node, b_node, j=j, n=n):
                delete_cost = dist[i - 1][j] + 1

            insert_cost = INF
            if _insert_allowed(a_node, b_node, i=i, m=m):
                insert_cost = dist[i][j - 1] + 1

            update_cost = INF
            if _update_allowed(a_node, b_node):
                update_cost = dist[i - 1][j - 1] + _node_update_cost(a_node, b_node)

            dist[i][j] = min(delete_cost, insert_cost, update_cost)

    return dist


def _backtrack_alignment(
    a_seq: Sequence[LDPairNode],
    b_seq: Sequence[LDPairNode],
    dist: Sequence[Sequence[int]],
) -> List[AlignmentStep]:
    i = len(a_seq)
    j = len(b_seq)
    n = len(b_seq)
    m = len(a_seq)
    raw_steps: List[AlignmentStep] = []

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            a_node = a_seq[i - 1]
            b_node = b_seq[j - 1]
            diag_cost = INF
            if _update_allowed(a_node, b_node):
                diag_cost = dist[i - 1][j - 1] + _node_update_cost(a_node, b_node)
            if dist[i][j] == diag_cost:
                raw_steps.append(
                    AlignmentStep(
                        kind="keep" if _node_update_cost(a_node, b_node) == 0 else "update",
                        a_node=a_node,
                        b_node=b_node,
                    )
                )
                i -= 1
                j -= 1
                continue

        if i > 0:
            a_node = a_seq[i - 1]
            if j == 0:
                if dist[i][j] == dist[i - 1][j] + 1:
                    raw_steps.append(AlignmentStep(kind="delete", a_node=a_node))
                    i -= 1
                    continue
            else:
                b_node = b_seq[j - 1]
                if _delete_allowed(a_node, b_node, j=j, n=n) and dist[i][j] == dist[i - 1][j] + 1:
                    raw_steps.append(AlignmentStep(kind="delete", a_node=a_node))
                    i -= 1
                    continue

        if j > 0:
            b_node = b_seq[j - 1]
            if i == 0:
                if dist[i][j] == dist[i][j - 1] + 1:
                    raw_steps.append(AlignmentStep(kind="insert", b_node=b_node))
                    j -= 1
                    continue
            else:
                a_node = a_seq[i - 1]
                if _insert_allowed(a_node, b_node, i=i, m=m) and dist[i][j] == dist[i][j - 1] + 1:
                    raw_steps.append(AlignmentStep(kind="insert", b_node=b_node))
                    j -= 1
                    continue

        raise RuntimeError(f"Backtracking failed at matrix cell ({i}, {j}).")

    raw_steps.reverse()
    return raw_steps


def _alignment_to_edit_ops(
    source_seq: Sequence[LDPairNode],
    steps: Sequence[AlignmentStep],
) -> List[EditOperation]:
    working = list(source_seq)
    cursor = 0
    operations: List[EditOperation] = []

    for step in steps:
        if step.kind == "keep":
            cursor += 1
            continue

        if step.kind == "update":
            if step.a_node is None or step.b_node is None:
                raise RuntimeError("Update step missing source or target node.")
            operations.append(
                EditOperation(
                    op="update",
                    position=cursor,
                    old_node=working[cursor],
                    new_node=LDPairNode(
                        label=step.b_node.label,
                        value=step.b_node.value,
                        depth=step.b_node.depth,
                        preorder_index=step.b_node.preorder_index,
                    ),
                    note=f"Update {step.a_node.label} -> {step.b_node.label}",
                )
            )
            working[cursor] = LDPairNode(
                label=step.b_node.label,
                value=step.b_node.value,
                depth=step.b_node.depth,
                preorder_index=step.b_node.preorder_index,
            )
            cursor += 1
            continue

        if step.kind == "delete":
            if step.a_node is None:
                raise RuntimeError("Delete step missing source node.")
            operations.append(
                EditOperation(
                    op="delete",
                    position=cursor,
                    node=working[cursor],
                    note=f"Delete {step.a_node.label}",
                )
            )
            del working[cursor]
            continue

        if step.kind == "insert":
            if step.b_node is None:
                raise RuntimeError("Insert step missing target node.")
            node = LDPairNode(
                label=step.b_node.label,
                value=step.b_node.value,
                depth=step.b_node.depth,
                preorder_index=step.b_node.preorder_index,
            )
            operations.append(
                EditOperation(
                    op="insert",
                    position=cursor,
                    node=node,
                    note=f"Insert {step.b_node.label}",
                )
            )
            working.insert(cursor, node)
            cursor += 1
            continue

        raise RuntimeError(f"Unsupported alignment step kind: {step.kind}")

    return operations


def compute_ted_chawathe(
    source_root: TreeNode,
    target_root: TreeNode,
    *,
    coerce_root_label: Optional[str] = None,
) -> TedResult:
    """
    Compute Chawathe-style TED on the current TreeNode representation.
    """
    source = clone_tree(source_root)
    target = clone_tree(target_root)

    if coerce_root_label is not None:
        source.label = coerce_root_label
        target.label = coerce_root_label

    a_seq = chawathe_tree_to_ld_pairs(source)
    b_seq = chawathe_tree_to_ld_pairs(target)
    dist = _compute_matrix(a_seq, b_seq)
    steps = _backtrack_alignment(a_seq, b_seq, dist)
    operations = _alignment_to_edit_ops(a_seq, steps)

    distance = dist[len(a_seq)][len(b_seq)]
    similarity = similarity_from_distance(distance, len(a_seq), len(b_seq))

    return TedResult(
        algorithm="chawathe_ld_pair_ted",
        distance=distance,
        similarity=similarity,
        source_size=len(a_seq),
        target_size=len(b_seq),
        source_ld_pairs=a_seq,
        target_ld_pairs=b_seq,
        operations=operations,
    )


def diff_trees(
    source_root: TreeNode,
    target_root: TreeNode,
    *,
    coerce_root_label: Optional[str] = None,
) -> Tuple[int, float, List[EditOperation]]:
    result = compute_ted_chawathe(source_root, target_root, coerce_root_label=coerce_root_label)
    return result.distance, result.similarity, result.operations
