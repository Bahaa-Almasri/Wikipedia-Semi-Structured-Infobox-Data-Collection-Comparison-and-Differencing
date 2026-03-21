"""
Zhang–Shasha (1989) ordered tree edit distance.

Postorder numbering, leftmost-leaf indices, keyroots, and forest DP (zss-style).
Recovers optimal postorder node ↔ node mappings for patching.
"""
from __future__ import annotations

import collections
import time
from typing import Dict, List, Optional, Tuple

from core.similarity.common import clone_tree
from domain.models.edit_script import TedResult
from domain.models.tree import TreeNode


def normalize_tree(node: TreeNode) -> None:
    """Sort children alphabetically by label; mutates in place."""
    node.children.sort(key=lambda x: x.label)
    for child in node.children:
        normalize_tree(child)


def insert_cost(_node: TreeNode) -> int:
    return 1


def delete_cost(_node: TreeNode) -> int:
    return 1


def rename_cost(node1: TreeNode, node2: TreeNode) -> int:
    if node1.label == node2.label and node1.value == node2.value:
        return 0
    return 1


def _similarity_from_max_norm(distance: int, size_a: int, size_b: int) -> float:
    denom = max(size_a, size_b)
    if denom == 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - (distance / denom)))


def _zeros(rows: int, cols: int) -> List[List[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def _pick_min_index(costs: List[float], order: List[int]) -> int:
    best = min(costs)
    for idx in order:
        if costs[idx] == best:
            return idx
    return 0


class _AnnotatedTree:
    """Postorder nodes, lmd per node, keyroots (zss construction)."""

    def __init__(self, root: TreeNode) -> None:
        self.nodes: List[TreeNode] = []
        self.lmds: List[int] = []
        self.keyroots: List[int]

        stack: List[Tuple[TreeNode, collections.deque]] = [(root, collections.deque())]
        pstack: List[Tuple[Tuple[TreeNode, int], collections.deque]] = []
        j = 0
        while stack:
            n, anc = stack.pop()
            nid = j
            for c in n.children:
                a = collections.deque(anc)
                a.appendleft(nid)
                stack.append((c, a))
            pstack.append(((n, nid), anc))
            j += 1

        lmds_map: dict = {}
        keyroots_map: dict = {}
        i = 0
        while pstack:
            (n, nid), anc = pstack.pop()
            self.nodes.append(n)
            if not n.children:
                lmd = i
                for a in anc:
                    if a not in lmds_map:
                        lmds_map[a] = i
                    else:
                        break
            else:
                lmd = lmds_map[nid]
            self.lmds.append(lmd)
            keyroots_map[lmd] = i
            i += 1
        self.keyroots = sorted(keyroots_map.values())


def _assign_zs_ids(a: _AnnotatedTree, b: _AnnotatedTree) -> None:
    for i, n in enumerate(a.nodes):
        n._zs_id = i
    for i, n in enumerate(b.nodes):
        n._zs_id = i


def postorder_nodes_and_zs_ids(root: TreeNode) -> Tuple[List[TreeNode], Dict[int, TreeNode]]:
    """
    Postorder node list and ``_zs_id`` indices identical to TED (``_AnnotatedTree``).

    Call after ``normalize_tree`` so numbering matches ``zhang_shasha_mappings``.
    """
    at = _AnnotatedTree(root)
    id_to_node: Dict[int, TreeNode] = {}
    for i, n in enumerate(at.nodes):
        n._zs_id = i
        id_to_node[i] = n
    return at.nodes, id_to_node


def _zhang_shasha_distance_only(
    a: _AnnotatedTree,
    b: _AnnotatedTree,
    treedists: List[List[float]],
) -> None:
    al, bl = a.lmds, b.lmds
    an, bn = a.nodes, b.nodes

    def treedist(i: int, j: int) -> None:
        m = i - al[i] + 2
        n = j - bl[j] + 2
        fd = _zeros(m, n)
        ioff = al[i] - 1
        joff = bl[j] - 1

        for x in range(1, m):
            node = an[x + ioff]
            fd[x][0] = fd[x - 1][0] + delete_cost(node)
        for y in range(1, n):
            node = bn[y + joff]
            fd[0][y] = fd[0][y - 1] + insert_cost(node)

        for x in range(1, m):
            for y in range(1, n):
                node1 = an[x + ioff]
                node2 = bn[y + joff]
                if al[i] == al[x + ioff] and bl[j] == bl[y + joff]:
                    costs = [
                        fd[x - 1][y] + delete_cost(node1),
                        fd[x][y - 1] + insert_cost(node2),
                        fd[x - 1][y - 1] + rename_cost(node1, node2),
                    ]
                    fd[x][y] = min(costs)
                else:
                    p = al[x + ioff] - 1 - ioff
                    q = bl[y + joff] - 1 - joff
                    costs = [
                        fd[x - 1][y] + delete_cost(node1),
                        fd[x][y - 1] + insert_cost(node2),
                        fd[p][q] + treedists[x + ioff][y + joff],
                    ]
                    fd[x][y] = min(costs)

        treedists[i][j] = fd[m - 1][n - 1]

    for i in a.keyroots:
        for j in b.keyroots:
            treedist(i, j)


def _build_fd_with_back(
    a: _AnnotatedTree,
    b: _AnnotatedTree,
    treedists: List[List[float]],
    i: int,
    j: int,
) -> Tuple[
    List[List[float]],
    List[List[int]],
    List[List[int]],
    List[List[int]],
    int,
    int,
    int,
    int,
]:
    """Forest DP for subtree (i,j) with backtracking tags: 0 del, 1 ins, 2 ren, 3 merge."""
    al, bl = a.lmds, b.lmds
    an, bn = a.nodes, b.nodes
    m = i - al[i] + 2
    n = j - bl[j] + 2
    fd = _zeros(m, n)
    back = [[-1 for _ in range(n)] for _ in range(m)]
    merge_si = [[-1 for _ in range(n)] for _ in range(m)]
    merge_sj = [[-1 for _ in range(n)] for _ in range(m)]
    ioff = al[i] - 1
    joff = bl[j] - 1

    for x in range(1, m):
        node = an[x + ioff]
        fd[x][0] = fd[x - 1][0] + delete_cost(node)
        back[x][0] = 0
    for y in range(1, n):
        node = bn[y + joff]
        fd[0][y] = fd[0][y - 1] + insert_cost(node)
        back[0][y] = 1

    for x in range(1, m):
        for y in range(1, n):
            node1 = an[x + ioff]
            node2 = bn[y + joff]
            if al[i] == al[x + ioff] and bl[j] == bl[y + joff]:
                c_del = fd[x - 1][y] + delete_cost(node1)
                c_ins = fd[x][y - 1] + insert_cost(node2)
                c_ren = fd[x - 1][y - 1] + rename_cost(node1, node2)
                costs = [c_del, c_ins, c_ren]
                fd[x][y] = min(costs)
                back[x][y] = _pick_min_index(costs, [2, 0, 1])
            else:
                p = al[x + ioff] - 1 - ioff
                q = bl[y + joff] - 1 - joff
                c_del = fd[x - 1][y] + delete_cost(node1)
                c_ins = fd[x][y - 1] + insert_cost(node2)
                c_merge = fd[p][q] + treedists[x + ioff][y + joff]
                costs = [c_del, c_ins, c_merge]
                fd[x][y] = min(costs)
                bi = _pick_min_index(costs, [2, 0, 1])
                if bi == 2:
                    back[x][y] = 3
                    merge_si[x][y] = x + ioff
                    merge_sj[x][y] = y + joff
                else:
                    back[x][y] = bi

    return fd, back, merge_si, merge_sj, ioff, joff, m, n


def _recover_pairings(
    a: _AnnotatedTree,
    b: _AnnotatedTree,
    treedists: List[List[float]],
    i: int,
    j: int,
    pairs: List[Tuple[int, int]],
) -> None:
    al, bl = a.lmds, b.lmds
    an, bn = a.nodes, b.nodes
    fd, back, merge_si, merge_sj, ioff, joff, m, n = _build_fd_with_back(a, b, treedists, i, j)

    x, y = m - 1, n - 1
    while x > 0 or y > 0:
        if x == 0:
            y -= 1
            continue
        if y == 0:
            x -= 1
            continue
        ch = back[x][y]
        if ch == 0:
            x -= 1
        elif ch == 1:
            y -= 1
        elif ch == 2:
            pairs.append((an[x + ioff]._zs_id, bn[y + joff]._zs_id))
            x -= 1
            y -= 1
        elif ch == 3:
            si = merge_si[x][y]
            sj = merge_sj[x][y]
            _recover_pairings(a, b, treedists, si, sj, pairs)
            p = al[si] - 1 - ioff
            q = bl[sj] - 1 - joff
            x, y = p, q
        else:
            break


def _zhang_shasha_core(
    source: TreeNode,
    target: TreeNode,
) -> Tuple[int, int, int, List[Dict[str, int]]]:
    a = _AnnotatedTree(source)
    b = _AnnotatedTree(target)
    _assign_zs_ids(a, b)
    size_a = len(a.nodes)
    size_b = len(b.nodes)
    if size_a == 0 and size_b == 0:
        return 0, 0, 0, []
    treedists = _zeros(size_a, size_b)
    _zhang_shasha_distance_only(a, b, treedists)
    dist = int(treedists[-1][-1])

    pairs: List[Tuple[int, int]] = []
    _recover_pairings(a, b, treedists, size_a - 1, size_b - 1, pairs)
    mappings = [{"source_id": s, "target_id": t} for s, t in pairs]
    return dist, size_a, size_b, mappings


def zhang_shasha_distance(tree1: TreeNode, tree2: TreeNode) -> dict:
    """Zhang–Shasha TED with unit costs; includes postorder ``mappings``."""
    t0 = time.perf_counter()
    s = clone_tree(tree1)
    t = clone_tree(tree2)
    normalize_tree(s)
    normalize_tree(t)
    distance, source_size, target_size, mappings = _zhang_shasha_core(s, t)
    similarity = _similarity_from_max_norm(distance, source_size, target_size)
    runtime_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "distance": distance,
        "similarity": similarity,
        "source_size": source_size,
        "target_size": target_size,
        "mappings": mappings,
        "runtime_ms": runtime_ms,
    }


def compute_ted_zhang_shasha(
    source_root: TreeNode,
    target_root: TreeNode,
    *,
    coerce_root_label: Optional[str] = None,
) -> TedResult:
    source = clone_tree(source_root)
    target = clone_tree(target_root)
    if coerce_root_label is not None:
        source.label = coerce_root_label
        target.label = coerce_root_label
    normalize_tree(source)
    normalize_tree(target)
    distance, source_size, target_size, mappings = _zhang_shasha_core(source, target)
    similarity = _similarity_from_max_norm(distance, source_size, target_size)
    return TedResult(
        algorithm="zhang_shasha",
        distance=distance,
        similarity=similarity,
        source_size=source_size,
        target_size=target_size,
        source_ld_pairs=[],
        target_ld_pairs=[],
        operations=[],
        zhang_shasha_mappings=mappings,
    )
