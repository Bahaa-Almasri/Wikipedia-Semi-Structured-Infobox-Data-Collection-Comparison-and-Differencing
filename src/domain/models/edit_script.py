from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class LDPairNode:
    """
    One node in Chawathe's LD-pair preorder sequence.

    depth is the depth of the node in the source tree.
    preorder_index is 0-based within the exported sequence.
    """

    label: str
    depth: int
    value: Optional[str] = None
    preorder_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "depth": self.depth,
            "value": self.value,
            "preorder_index": self.preorder_index,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "LDPairNode":
        return LDPairNode(
            label=str(data["label"]),
            depth=int(data["depth"]),
            value=data.get("value"),
            preorder_index=data.get("preorder_index"),
        )


@dataclass(frozen=True)
class EditOperation:
    """
    Patch operation expressed against the current LD-pair sequence.

    position is 0-based and refers to the sequence state *at the moment the
    operation is applied*.
    """

    op: str  # insert | delete | update
    position: int
    node: Optional[LDPairNode] = None
    old_node: Optional[LDPairNode] = None
    new_node: Optional[LDPairNode] = None
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op,
            "position": self.position,
            "node": self.node.to_dict() if self.node else None,
            "old_node": self.old_node.to_dict() if self.old_node else None,
            "new_node": self.new_node.to_dict() if self.new_node else None,
            "note": self.note,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "EditOperation":
        return EditOperation(
            op=str(data["op"]),
            position=int(data["position"]),
            node=LDPairNode.from_dict(data["node"]) if data.get("node") else None,
            old_node=LDPairNode.from_dict(data["old_node"]) if data.get("old_node") else None,
            new_node=LDPairNode.from_dict(data["new_node"]) if data.get("new_node") else None,
            note=data.get("note"),
        )


@dataclass
class TedResult:
    algorithm: str
    distance: int
    similarity: float
    source_size: int
    target_size: int
    source_ld_pairs: List[LDPairNode] = field(default_factory=list)
    target_ld_pairs: List[LDPairNode] = field(default_factory=list)
    operations: List[EditOperation] = field(default_factory=list)
    # Zhang–Shasha: postorder node alignments (source_id -> target_id in postorder numbering)
    zhang_shasha_mappings: Optional[List[Dict[str, int]]] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "algorithm": self.algorithm,
            "distance": self.distance,
            "similarity": self.similarity,
            "source_size": self.source_size,
            "target_size": self.target_size,
            "source_ld_pairs": [n.to_dict() for n in self.source_ld_pairs],
            "target_ld_pairs": [n.to_dict() for n in self.target_ld_pairs],
            "operations": [op.to_dict() for op in self.operations],
        }
        if self.zhang_shasha_mappings is not None:
            out["mappings"] = list(self.zhang_shasha_mappings)
        return out


# --- Nierman & Jagadish edit script (tree-based refs) ---


@dataclass(frozen=True)
class NJEditOperation:
    """
    One forward edit operation for the Nierman & Jagadish tree-to-tree script.

    Supported operations:
    - update: update an existing source node referenced by source_ref
    - delete_tree: delete an existing source subtree rooted at source_ref
    - insert_tree: insert a destination subtree snapshot under parent_ref at 1-based position
    """

    op: str  # update | delete_tree | insert_tree
    source_ref: Optional[str] = None
    parent_ref: Optional[str] = None
    position: Optional[int] = None  # 1-based sibling position for insert_tree
    old_label: Optional[str] = None
    new_label: Optional[str] = None
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    subtree: Optional[Dict[str, Any]] = None
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op,
            "source_ref": self.source_ref,
            "parent_ref": self.parent_ref,
            "position": self.position,
            "old_label": self.old_label,
            "new_label": self.new_label,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "subtree": self.subtree,
            "note": self.note,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "NJEditOperation":
        return NJEditOperation(
            op=str(data["op"]),
            source_ref=data.get("source_ref"),
            parent_ref=data.get("parent_ref"),
            position=data.get("position"),
            old_label=data.get("old_label"),
            new_label=data.get("new_label"),
            old_value=data.get("old_value"),
            new_value=data.get("new_value"),
            subtree=data.get("subtree"),
            note=data.get("note"),
        )


@dataclass
class NJTedResult:
    algorithm: str
    distance: int
    similarity: float
    source_size: int
    target_size: int
    operations: List[NJEditOperation] = field(default_factory=list)
    source_root_ref: str = "s0"
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "distance": self.distance,
            "similarity": self.similarity,
            "source_size": self.source_size,
            "target_size": self.target_size,
            "source_root_ref": self.source_root_ref,
            "operations": [op.to_dict() for op in self.operations],
            "meta": self.meta,
        }
