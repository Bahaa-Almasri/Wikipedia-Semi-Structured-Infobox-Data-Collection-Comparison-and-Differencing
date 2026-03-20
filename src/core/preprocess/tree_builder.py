from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from core.data.storage import list_slugs, read_json_document, write_tree_document
from domain.models.tree import TreeNode


def _build_subtree(label: str, value: Any) -> TreeNode:
    print("BUILD SUBTREE CALLED:", label, type(value), value)
    # Local tokenizer to satisfy "modify only _build_subtree".
    import re

    def tokenize(text: str):
        tokens = re.split(r"[^\w]+", text)
        return [t for t in tokens if t]

    if isinstance(value, dict):
        node = TreeNode(label=label)
        primitive_keys = []
        nested_keys = []

        for key, val in value.items():
            if isinstance(val, (dict, list)):
                nested_keys.append(key)
            else:
                primitive_keys.append(key)

        primitive_keys.sort()
        nested_keys.sort()

        ordered_keys = primitive_keys + nested_keys

        for key in ordered_keys:
            node.children.append(_build_subtree(key, value[key]))
        return node

    if isinstance(value, list):
        node = TreeNode(label=label)
        for item in value:
            node.children.append(_build_subtree(f"{label}_item", item))
        return node

    # For scalars (including None), do NOT store value directly on the label node.
    # Instead, create child leaf nodes.
    node = TreeNode(label=label)

    value_str = "null" if value is None else str(value)

    # 1) Control tokenization based on field type (skip selected meta fields).
    if label in {"retrieved_at", "wikipedia_url", "slug", "source"}:
        node.children.append(TreeNode(label="value", value=value_str))
        print("SCALAR CASE:", label, value)
        return node

    # Special-case: split UTC offsets like "UTC+4:30" into ["UTC", "4", "30"].
    # This must run before the generic ":" timestamp rule below.
    if isinstance(value, str) and value.startswith("UTC"):
        match = re.match(r"^UTC\+?(\d+):?(\d+)?$", value_str)
        if match:
            node.children.append(TreeNode(label="token", value="UTC"))
            hours = match.group(1)
            minutes = match.group(2)
            if hours:
                node.children.append(TreeNode(label="token", value=hours))
            if minutes:
                node.children.append(TreeNode(label="token", value=minutes))
            print("SCALAR CASE:", label, value)
            return node

    # 2) Handle numbers: keep simple floats as one token (avoid "0.496" -> ["0","496"]).
    if re.match(r"^\d+\.\d+$", value_str):
        node.children.append(TreeNode(label="token", value=value_str))
        print("SCALAR CASE:", label, value)
        return node

    # 3) Handle dates/timestamps: preserve as one token if it looks like a timestamp/date.
    if ("T" in value_str) or ("-" in value_str) or (":" in value_str):
        node.children.append(TreeNode(label="token", value=value_str))
        print("SCALAR CASE:", label, value)
        return node

    # 4) Keep normal tokenization for real text.
    raw_tokens = re.split(r"[^\w]+", value_str)

    # 5) Clean token filtering to reduce TED noise.
    tokens: List[str] = []
    for t in raw_tokens:
        if not t:
            continue
        # Remove very short tokens unless they're known meaningful compass/cardinal letters.
        if len(t) == 1 and t not in {"N", "E"}:
            continue
        tokens.append(t)

    for token in tokens:
        node.children.append(TreeNode(label="token", value=token))

    print("SCALAR CASE:", label, value)
    return node


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
    print(tree_root.to_dict())
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
