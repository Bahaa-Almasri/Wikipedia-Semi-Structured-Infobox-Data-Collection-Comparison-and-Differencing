from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from core.similarity.common import clone_tree, similarity_from_distance
from core.similarity.tree_validation import validate_tree
from domain.models.edit_script import NJEditOperation, NJTedResult
from domain.models.tree import TreeNode


@dataclass(frozen=True)
class _PairResult:
    distance: int
    operations: Tuple[NJEditOperation, ...]


class NiermanJagadishError(ValueError):
    pass


class _RefAssigner:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.counter = 0
        self.by_obj_id: Dict[int, str] = {}

    def assign(self, root: TreeNode) -> Dict[int, str]:
        def _walk(node: TreeNode) -> None:
            ref = f"{self.prefix}{self.counter}"
            self.counter += 1
            self.by_obj_id[id(node)] = ref
            for child in node.children:
                _walk(child)

        _walk(root)
        return self.by_obj_id


def nj_tree_size(root: TreeNode) -> int:
    """Tree size (node count). NJ-specific helper for cost computation."""
    return 1 + sum(nj_tree_size(child) for child in root.children)


def _node_equal(a: TreeNode, b: TreeNode) -> bool:
    return a.label == b.label and (a.value or None) == (b.value or None)


def _update_cost(a: TreeNode, b: TreeNode) -> int:
    return 0 if _node_equal(a, b) else 1


def _iter_nodes(root: TreeNode):
    yield root
    for child in root.children:
        yield from _iter_nodes(child)


def _contains_at(pattern: TreeNode, candidate: TreeNode) -> bool:
    if not _node_equal(pattern, candidate):
        return False

    if not pattern.children:
        return True

    start = 0
    for p_child in pattern.children:
        found = False
        for idx in range(start, len(candidate.children)):
            if _contains_at(p_child, candidate.children[idx]):
                found = True
                start = idx + 1
                break
        if not found:
            return False

    return True


@lru_cache(maxsize=None)
def _contained_in_cached(pattern_key: Tuple, tree_key: Tuple) -> bool:
    pattern = _nj_tree_from_key(pattern_key)
    tree = _nj_tree_from_key(tree_key)
    for candidate in _iter_nodes(tree):
        if _contains_at(pattern, candidate):
            return True
    return False


def _nj_tree_to_key(root: TreeNode) -> Tuple:
    return (
        root.label,
        root.value,
        tuple(_nj_tree_to_key(child) for child in root.children),
    )


def _nj_tree_from_key(key: Tuple) -> TreeNode:
    label, value, children = key
    return TreeNode(label=label, value=value, children=[_nj_tree_from_key(c) for c in children])


class _NJEngine:
    def __init__(self, source_root: TreeNode, target_root: TreeNode) -> None:
        validate_tree(source_root)
        validate_tree(target_root)

        self.source_root = source_root
        self.target_root = target_root

        self.source_refs = _RefAssigner("s").assign(source_root)
        self.target_refs = _RefAssigner("t").assign(target_root)

        self.source_size = nj_tree_size(source_root)
        self.target_size = nj_tree_size(target_root)

        self._source_key = _nj_tree_to_key(source_root)
        self._target_key = _nj_tree_to_key(target_root)
        self._memo: Dict[Tuple[int, int], _PairResult] = {}

    def source_ref(self, node: TreeNode) -> str:
        return self.source_refs[id(node)]

    def target_ref(self, node: TreeNode) -> str:
        return self.target_refs[id(node)]

    def del_tree_cost(self, subtree: TreeNode) -> int:
        if _contained_in_cached(_nj_tree_to_key(subtree), self._target_key):
            return 1
        return nj_tree_size(subtree)

    def ins_tree_cost(self, subtree: TreeNode) -> int:
        if _contained_in_cached(_nj_tree_to_key(subtree), self._source_key):
            return 1
        return nj_tree_size(subtree)

    def compare(self, a: TreeNode, b: TreeNode) -> _PairResult:
        key = (id(a), id(b))
        if key in self._memo:
            return self._memo[key]

        root_cost = _update_cost(a, b)
        m = len(a.children)
        n = len(b.children)

        dist = [[0] * (n + 1) for _ in range(m + 1)]
        choice: List[List[Optional[Tuple[str, int, int]]]] = [
            [None] * (n + 1) for _ in range(m + 1)
        ]

        dist[0][0] = root_cost
        choice[0][0] = ("root", -1, -1)

        for i in range(1, m + 1):
            child_a = a.children[i - 1]
            dist[i][0] = dist[i - 1][0] + self.del_tree_cost(child_a)
            choice[i][0] = ("delete_tree", i - 1, -1)

        for j in range(1, n + 1):
            child_b = b.children[j - 1]
            dist[0][j] = dist[0][j - 1] + self.ins_tree_cost(child_b)
            choice[0][j] = ("insert_tree", -1, j - 1)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                child_a = a.children[i - 1]
                child_b = b.children[j - 1]

                sub = self.compare(child_a, child_b)
                match_cost = dist[i - 1][j - 1] + sub.distance
                delete_cost = dist[i - 1][j] + self.del_tree_cost(child_a)
                insert_cost = dist[i][j - 1] + self.ins_tree_cost(child_b)

                best_cost = min(match_cost, delete_cost, insert_cost)
                dist[i][j] = best_cost

                if best_cost == match_cost:
                    choice[i][j] = ("match", i - 1, j - 1)
                elif best_cost == delete_cost:
                    choice[i][j] = ("delete_tree", i - 1, j)
                else:
                    choice[i][j] = ("insert_tree", i, j - 1)

        ops: List[NJEditOperation] = []
        if root_cost > 0:
            ops.append(
                NJEditOperation(
                    op="update",
                    source_ref=self.source_ref(a),
                    old_label=a.label,
                    new_label=b.label,
                    old_value=a.value,
                    new_value=b.value,
                    note=f"Upd(R({self.source_ref(a)}), {self.target_ref(b)})",
                )
            )

        ops.extend(self._build_operations(a, b, choice, m, n))

        result = _PairResult(distance=dist[m][n], operations=tuple(ops))
        self._memo[key] = result
        return result

    def _build_operations(
        self,
        a: TreeNode,
        b: TreeNode,
        choice: List[List[Optional[Tuple[str, int, int]]]],
        i: int,
        j: int,
    ) -> List[NJEditOperation]:
        if i == 0 and j == 0:
            return []

        current = choice[i][j]
        if current is None:
            raise NiermanJagadishError(f"Missing backpointer at matrix cell ({i}, {j}).")

        kind, left, right = current

        if kind == "match":
            prefix = self._build_operations(a, b, choice, i - 1, j - 1)
            child_result = self.compare(a.children[left], b.children[right])
            return prefix + list(child_result.operations)

        if kind == "delete_tree":
            prefix = self._build_operations(a, b, choice, i - 1, j)
            victim = a.children[left]
            prefix.append(
                NJEditOperation(
                    op="delete_tree",
                    source_ref=self.source_ref(victim),
                    old_label=victim.label,
                    old_value=victim.value,
                    subtree_node_count=nj_tree_size(victim),
                    note=f"DelTree({self.source_ref(victim)})",
                )
            )
            return prefix

        if kind == "insert_tree":
            prefix = self._build_operations(a, b, choice, i, j - 1)
            subtree = b.children[right]
            prefix.append(
                NJEditOperation(
                    op="insert_tree",
                    parent_ref=self.source_ref(a),
                    position=right + 1,
                    subtree=clone_tree(subtree).to_dict(),
                    new_label=subtree.label,
                    new_value=subtree.value,
                    subtree_node_count=nj_tree_size(subtree),
                    note=(
                        f"InsTree({self.target_ref(subtree)}, {self.source_ref(a)}, {right + 1})"
                    ),
                )
            )
            return prefix

        raise NiermanJagadishError(f"Unsupported backpointer kind '{kind}'.")


def compute_ted_nj(
    source_root: TreeNode,
    target_root: TreeNode,
    *,
    coerce_root_label: Optional[str] = None,
) -> NJTedResult:
    """
    Compute a single minimum-cost Nierman & Jagadish (2002) edit script.
    """
    src = clone_tree(source_root)
    tgt = clone_tree(target_root)

    if coerce_root_label is not None:
        src.label = coerce_root_label
        tgt.label = coerce_root_label

    engine = _NJEngine(src, tgt)
    pair_result = engine.compare(src, tgt)
    similarity = similarity_from_distance(pair_result.distance, engine.source_size, engine.target_size)

    return NJTedResult(
        algorithm="Nierman & Jagadish (2002)",
        distance=pair_result.distance,
        similarity=similarity,
        source_size=engine.source_size,
        target_size=engine.target_size,
        source_root_ref=engine.source_ref(src),
        operations=list(pair_result.operations),
        meta={
            "coerced_root_label": coerce_root_label,
            "notes": [
                "Returns one deterministic minimum-cost script.",
                "Tree insertion/deletion costs use the contained-in relation against the full source/target trees.",
            ],
        },
    )
