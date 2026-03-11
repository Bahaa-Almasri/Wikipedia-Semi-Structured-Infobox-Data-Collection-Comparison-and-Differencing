from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .storage import (
    json_path_for_slug,
    tree_path_for_slug,
    write_tree_document,
)
from .tree import TreeNode
from .config import PATHS


def _build_subtree(label: str, value: Any) -> TreeNode:
    """
    Recursively convert a JSON value into a TreeNode subtree.

    Rules:
    - dict: one node per key (children ordered by key name)
    - list: one child per element, labeled "<label>_item"
    - scalar / null: leaf node with stringified value
    """
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
    """
    Select the best field section for building the comparison tree.

    Preferred:
    - normalized.comparison_fields

    Fallback:
    - normalized.fields

    This lets us keep backward compatibility while moving toward a
    cleaner, TED-friendly canonical representation.
    """
    normalized = document.get("normalized", {}) or {}

    comparison_fields = normalized.get("comparison_fields", {}) or {}
    if comparison_fields:
        return comparison_fields

    fallback_fields = normalized.get("fields", {}) or {}
    return fallback_fields


def build_country_tree(document: Dict[str, Any]) -> TreeNode:
    """
    Build a rooted, ordered, labeled tree from a country JSON document.

    Preferred comparison source:
    - normalized.comparison_fields

    Fallback:
    - normalized.fields

    Raw HTML is intentionally ignored.
    """
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


def build_and_save_tree_for_slug(slug: str) -> Optional[Path]:
    """
    Load one country JSON by slug, build its tree, and save as JSON.

    Returns the output path, or None if the JSON file does not exist.
    """
    json_path = json_path_for_slug(slug)
    if not json_path.exists():
        return None

    data = json.loads(json_path.read_text(encoding="utf-8"))
    tree_root = build_country_tree(data)
    write_tree_document(slug, tree_root.to_dict())
    return tree_path_for_slug(slug)


def build_and_save_trees_for_all(slugs: Optional[Iterable[str]] = None) -> List[Path]:
    """
    Build and save trees for all country JSON documents.

    If slugs is provided, only those slugs are processed.
    """
    PATHS.trees_dir.mkdir(parents=True, exist_ok=True)

    if slugs is None:
        json_dir = PATHS.json_dir
        slugs = sorted(p.stem for p in json_dir.glob("*.json"))

    written: List[Path] = []
    for slug in slugs:
        result = build_and_save_tree_for_slug(slug)
        if result is not None:
            written.append(result)

    return written

