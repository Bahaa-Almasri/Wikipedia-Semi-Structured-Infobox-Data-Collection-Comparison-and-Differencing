from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from core.data.storage import list_slugs, read_json_document, write_tree_document
from domain.models.tree import TreeNode

# Field typing for semantic merging (token lists → single string values on the field node).
SEMANTIC_FIELDS: Dict[str, List[str]] = {
    "name_fields": [
        "head_of_government",
        "head_of_state",
        "capital",
        "country_name",
    ],
    "numeric_fields": [
        "gdp_nominal_total",
        "gdp_ppp_total",
    ],
    "list_fields": [
        "languages",
    ],
}

# Coordinate-like: merge parts with spaces (e.g. "34 31 N").
_COORDINATE_FIELDS = frozenset({"latitude", "longitude"})

_IGNORED_META_SCALAR_LABELS = frozenset({"retrieved_at", "wikipedia_url", "slug", "source"})


def merge_tokens(tokens: List[str], field_name: str) -> str:
    """
    Merge a list of word-like tokens into one display/comparison string.

    Rules:
    - name_fields: space-join
    - numeric_fields: digit groups joined by '.'; optional trailing unit word
    - latitude/longitude: space-join
    - default: space-join; if all tokens are digits, join with '.'
    """
    if not tokens:
        return ""

    name_fields = SEMANTIC_FIELDS["name_fields"]
    numeric_fields = SEMANTIC_FIELDS["numeric_fields"]

    if field_name in name_fields:
        return " ".join(tokens)

    if field_name in _COORDINATE_FIELDS:
        return " ".join(tokens)

    if field_name in numeric_fields:
        if len(tokens) == 1:
            return tokens[0]
        body, last = tokens[:-1], tokens[-1]
        # e.g. ["17", "329", "billion"] → "17.329 billion"
        if len(tokens) >= 2 and all(t.isdigit() for t in body) and not last.isdigit():
            return ".".join(body) + " " + last
        # e.g. ["17", "329"] → "17.329"
        if all(t.isdigit() for t in tokens):
            return ".".join(tokens)
        return " ".join(tokens)

    if len(tokens) == 1:
        return tokens[0]

    body, last = tokens[:-1], tokens[-1]
    if body and all(t.isdigit() for t in body) and last.isalpha():
        return ".".join(body) + " " + last
    if all(t.isdigit() for t in tokens):
        return ".".join(tokens)
    return " ".join(tokens)


def _tokenize_scalar_text(text: str) -> List[str]:
    """Split scalar string into word tokens; drop noise single-char tokens except N/E."""
    raw_tokens = re.split(r"[^\w]+", text)
    tokens: List[str] = []
    for t in raw_tokens:
        if not t:
            continue
        if len(t) == 1 and not t.isdigit() and t not in {"N", "E"}:
            continue
        tokens.append(t)
    return tokens


def _scalar_to_leaf_node(label: str, value: Any) -> TreeNode:
    """
    Build a leaf field node: label + merged string value (no token children).
    Complex dict/list values are handled by _build_subtree before this is used.
    """
    value_str = "" if value is None else str(value)

    # Meta / noisy URLs / timestamps: keep one string; still no token children.
    if label in _IGNORED_META_SCALAR_LABELS:
        return TreeNode(label=label, value=value_str)

    # UTC offsets: keep as one string on the node.
    if isinstance(value, str) and value.startswith("UTC"):
        return TreeNode(label=label, value=value_str)

    # Simple decimals like "0.496" — one value.
    if re.match(r"^\d+\.\d+$", value_str):
        return TreeNode(label=label, value=value_str)

    # Dates / timestamps: one string.
    if ("T" in value_str) or ("-" in value_str) or (":" in value_str):
        return TreeNode(label=label, value=value_str)

    tokens = _tokenize_scalar_text(value_str)
    merged = merge_tokens(tokens, label)
    return TreeNode(label=label, value=merged)


def _build_subtree(label: str, value: Any) -> TreeNode:
    """Recursively build a TreeNode tree with semantic scalar leaves (no token children)."""
    if isinstance(value, dict):
        node = TreeNode(label=label)
        primitive_keys: List[str] = []
        nested_keys: List[str] = []

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
        list_fields = set(SEMANTIC_FIELDS["list_fields"])

        if label in list_fields:
            for item in value:
                if isinstance(item, (dict, list)):
                    node.children.append(_build_subtree(f"{label}_item", item))
                elif item is None or (isinstance(item, str) and not str(item).strip()):
                    continue
                else:
                    # Primitive list entry: one leaf per item (e.g. language name).
                    item_str = str(item)
                    node.children.append(
                        TreeNode(label=f"{label}_item", value=item_str)
                    )
            return node

        for item in value:
            node.children.append(_build_subtree(f"{label}_item", item))
        return node

    # Scalar (including None)
    return _scalar_to_leaf_node(label, value)


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
