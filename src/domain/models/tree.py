from __future__ import annotations

import io
from collections import defaultdict, deque
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class TreeNode:
    """
    Simple rooted, ordered, labeled tree node for TED.

    - label: node label (e.g. field name, meta key)
    - value: optional scalar value as text (leaf content)
    - children: ordered list of child nodes
    """

    label: str
    value: Optional[str] = None
    children: List["TreeNode"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "value": self.value,
            "children": [child.to_dict() for child in self.children],
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TreeNode":
        return TreeNode(
            label=data["label"],
            value=data.get("value"),
            children=[TreeNode.from_dict(c) for c in data.get("children", [])],
        )


def pretty_print(node: TreeNode, indent: int = 0, max_depth: Optional[int] = None) -> None:
    """
    Print a human-readable view of the tree for debugging.

    max_depth (if set) limits how deep to recurse.
    """
    depth = indent // 2
    if max_depth is not None and depth > max_depth:
        return

    prefix = " " * indent
    if node.value is not None and not node.children:
        line = f"{node.label} = {node.value}"
    elif node.value is not None:
        line = f"{node.label}: {node.value}"
    else:
        line = node.label

    print(prefix + line)

    if max_depth is not None and depth == max_depth:
        if node.children:
            print(prefix + "  ...")
        return

    for child in node.children:
        pretty_print(child, indent=indent + 2, max_depth=max_depth)


def _tree_line(node: TreeNode) -> str:
    """Single-line text for a node (label, or label = value when value is set)."""
    if node.value is not None:
        return f"{node.label} = {node.value}"
    return node.label


def draw_tree(
    node: TreeNode,
    prefix: str = "",
    is_last: bool = True,
    show_root: bool = True,
    max_depth: Optional[int] = None,
    _depth: int = 0,
) -> None:
    """
    Print the tree using box-drawing characters (├──, └──, │) for debugging and demos.

    - Leaf or node with a value: ``label = value``
    - Otherwise: ``label``

    If ``show_root`` is False, the root label is omitted and only children are drawn
    (useful when the root is a dummy wrapper).

    If ``max_depth`` is set, nodes deeper than that (root depth = 0) are not expanded;
    a ``...`` line is printed when children are cut off.
    """

    def emit(n: TreeNode, pfx: str, last: bool, depth: int) -> None:
        if max_depth is not None and depth > max_depth:
            return
        branch = "└── " if last else "├── "
        print(pfx + branch + _tree_line(n))
        if max_depth is not None and depth == max_depth and n.children:
            ext = pfx + ("    " if last else "│   ")
            print(ext + "...")
            return
        ext = pfx + ("    " if last else "│   ")
        for i, ch in enumerate(n.children):
            emit(ch, ext, i == len(n.children) - 1, depth + 1)

    if show_root:
        if max_depth is not None and _depth > max_depth:
            return
        print(prefix + _tree_line(node))
        if max_depth is not None and _depth == max_depth and node.children:
            print(prefix + "...")
            return
        for i, ch in enumerate(node.children):
            emit(ch, prefix, i == len(node.children) - 1, _depth + 1)
    else:
        for i, ch in enumerate(node.children):
            emit(ch, prefix, i == len(node.children) - 1, _depth + 1)


def format_draw_tree(
    node: TreeNode,
    prefix: str = "",
    is_last: bool = True,
    show_root: bool = True,
    max_depth: Optional[int] = None,
) -> str:
    """Return the same text ``draw_tree`` would print (no trailing newline)."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        draw_tree(
            node,
            prefix=prefix,
            is_last=is_last,
            show_root=show_root,
            max_depth=max_depth,
        )
    return buf.getvalue().rstrip("\n")


def format_draw_tree_dict(
    tree_dict: Dict[str, Any],
    **kwargs: Any,
) -> str:
    """Build box-drawn tree text from a ``TreeNode``-shaped dict (API / JSON)."""
    return format_draw_tree(TreeNode.from_dict(tree_dict), **kwargs)


def _normalize_scalar(value: Optional[str]) -> Optional[Union[str, float, int]]:
    """
    Match wikiinfobox_service._normalize_scalar: strip, unify empty, parse numbers.

    Used so tree_similarity agrees with semantic diff / patch validation when values
    differ only by whitespace or numeric formatting.
    """
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def _node_is_empty(node: TreeNode) -> bool:
    """True if node has no meaningful content (empty value and no non-empty descendants)."""
    if node.children:
        return all(_node_is_empty(c) for c in node.children)
    return node.value is None or (
        isinstance(node.value, str) and not node.value.strip()
    )


def tree_similarity(t1: TreeNode, t2: TreeNode) -> float:
    """
    Compute similarity score between two trees (0.0 to 1.0).

    - 1.0 → identical trees
    - 0.0 → completely different

    Compares nodes recursively:
    - +1 if labels match
    - +1 if values match (None and "" treated as equal)
    - Children matched by label (order-independent); duplicate labels pair in FIFO order
    - Unmatched children penalized by traversing their subtrees (empty nodes excluded)
    """
    matches = 0
    total = 0

    def _unmatched_subtree(node: TreeNode) -> None:
        """Penalize a subtree that has no paired counterpart in the other tree."""
        nonlocal matches, total
        if not _node_is_empty(node):
            total += 2
        for ch in node.children:
            _unmatched_subtree(ch)

    def _score(n1: TreeNode, n2: TreeNode) -> None:
        nonlocal matches, total

        total += 1
        if n1.label == n2.label:
            matches += 1

        total += 1
        v1 = n1.value if n1.value is not None else ""
        v2 = n2.value if n2.value is not None else ""
        if v1 == v2:
            matches += 1

        by_label: Dict[str, deque[TreeNode]] = defaultdict(deque)
        for c in n2.children:
            by_label[c.label].append(c)

        for c1 in n1.children:
            q = by_label[c1.label]
            if q:
                _score(c1, q.popleft())
            else:
                _unmatched_subtree(c1)

        for q in by_label.values():
            while q:
                _unmatched_subtree(q.popleft())

    _score(t1, t2)
    if total == 0:
        return 1.0
    return matches / total
