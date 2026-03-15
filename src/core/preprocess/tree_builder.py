from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from core.data.storage import list_slugs, read_json_document, write_tree_document
from domain.models.tree import TreeNode


def _build_subtree(label: str, value: Any) -> TreeNode:
    if isinstance(value, dict):
        node = TreeNode(label=label)
        for key in sorted(value.keys()):
            node.children.append(_build_subtree(key, value[key]))
        return node

    if isinstance(value, list):
        node = TreeNode(label=label)
        for item in value:
            node.children.append(_build_subtree(f"{label}_item", item))
        return node

    if value is None:
        return TreeNode(label=label, value="null")

    return TreeNode(label=label, value=str(value))


def _select_comparison_fields(document: Dict[str, Any]) -> Dict[str, Any]:
    normalized = document.get("normalized", {}) or {}
    comparison_fields = normalized.get("comparison_fields", {}) or {}
    if comparison_fields:
        return comparison_fields
    return normalized.get("fields", {}) or {}


def build_country_tree(document: Dict[str, Any]) -> TreeNode:
    meta = document.get("meta", {}) or {}
    fields = _select_comparison_fields(document)

    root_label = meta.get("slug") or meta.get("country_name") or "country"
    root = TreeNode(label=str(root_label))

    if meta:
        root.children.append(_build_subtree("meta", meta))

    if fields:
        fields_node = TreeNode(label="fields")
        for field_key in sorted(fields.keys()):
            fields_node.children.append(_build_subtree(field_key, fields[field_key]))
        root.children.append(fields_node)

    return root


def build_and_save_tree_for_slug(slug: str) -> Optional[str]:
    """Load country JSON from MongoDB, build tree, save tree to MongoDB. Returns slug if saved, else None."""
    data = read_json_document(slug)
    if data is None:
        return None
    tree_root = build_country_tree(data)
    write_tree_document(slug, tree_root.to_dict())
    return slug


def build_and_save_trees_for_all(slugs: Optional[Iterable[str]] = None) -> List[str]:
    """Build and save trees for all country documents in MongoDB. Returns list of slugs written."""
    if slugs is None:
        slugs = list_slugs()

    written: List[str] = []
    for slug in slugs:
        result = build_and_save_tree_for_slug(slug)
        if result is not None:
            written.append(result)

    return written
