from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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

