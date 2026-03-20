"""
Post-process document trees: tree → JSON/XML/infobox text.
Unified summarize and report for both Chawathe and NJ edit scripts.
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any, Dict, List, Tuple, Union

from domain.models.edit_script import NJTedResult, TedResult
from domain.models.tree import TreeNode


_XML_NAME_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


def _xml_safe_name(name: str) -> str:
    candidate = _XML_NAME_RE.sub("_", name.strip()) or "node"
    if not re.match(r"[A-Za-z_]", candidate[0]):
        candidate = f"n_{candidate}"
    return candidate


def tree_to_native_object(node: TreeNode) -> Any:
    """Convert TreeNode into a nested Python structure (dict/list/scalar)."""
    if not node.children:
        return node.value if node.value is not None else ""

    grouped: Dict[str, List[Any]] = {}
    for child in node.children:
        grouped.setdefault(child.label, []).append(tree_to_native_object(child))

    result: Dict[str, Any] = {}
    for label, values in grouped.items():
        result[label] = values[0] if len(values) == 1 else values

    if node.value is not None:
        result["_value"] = node.value

    return result


def tree_to_native_json_dict(root: TreeNode) -> Dict[str, Any]:
    return {root.label: tree_to_native_object(root)}


def tree_to_json_string(root: TreeNode, *, indent: int = 2) -> str:
    return json.dumps(tree_to_native_json_dict(root), ensure_ascii=False, indent=indent)


def _build_xml_element(node: TreeNode) -> ET.Element:
    element = ET.Element(_xml_safe_name(node.label))
    if node.value is not None:
        element.text = node.value
    for child in node.children:
        element.append(_build_xml_element(child))
    return element


def tree_to_xml_string(root: TreeNode) -> str:
    element = _build_xml_element(root)
    return ET.tostring(element, encoding="unicode")


def _flatten_infobox_rows(node: TreeNode, prefix: str = "") -> List[Tuple[str, str]]:
    key = f"{prefix}.{node.label}" if prefix else node.label
    if not node.children:
        return [(key, node.value or "")]

    rows: List[Tuple[str, str]] = []
    for child in node.children:
        rows.extend(_flatten_infobox_rows(child, prefix=key))
    return rows


def tree_to_infobox_text(root: TreeNode) -> str:
    """
    Render a normalized Wikipedia-style infobox text.
    """
    meta_node = next((child for child in root.children if child.label == "meta"), None)
    fields_node = next((child for child in root.children if child.label == "fields"), None)

    rows: List[Tuple[str, str]] = []
    title = root.label

    if meta_node is not None:
        meta_rows = dict(_flatten_infobox_rows(meta_node))
        title = meta_rows.get("meta.country_name") or meta_rows.get("meta.slug") or root.label

    if fields_node is not None:
        rows.extend(_flatten_infobox_rows(fields_node))
    else:
        rows.extend(_flatten_infobox_rows(root))

    out_lines = ["{{Infobox country", f"| name = {title}"]
    for key, value in rows:
        if not str(value).strip():  # Skip empty fields
            continue
        cleaned_key = key.removeprefix("fields.")
        out_lines.append(f"| {cleaned_key} = {value}")
    out_lines.append("}}")
    return "\n".join(out_lines)


def summarize_edit_script(ted_result: Union[TedResult, NJTedResult]) -> Dict[str, int]:
    """Summarize edit script operation counts. Works for both Chawathe and NJ results."""
    counts = Counter(op.op for op in ted_result.operations)
    if isinstance(ted_result, NJTedResult):
        return {
            "update": counts.get("update", 0),
            "insert_tree": counts.get("insert_tree", 0),
            "delete_tree": counts.get("delete_tree", 0),
            "total": sum(counts.values()),
        }
    return {
        "insert": counts.get("insert", 0),
        "delete": counts.get("delete", 0),
        "update": counts.get("update", 0),
        "total": sum(counts.values()),
    }


def render_comparison_report(
    source_slug: str,
    target_slug: str,
    ted_result: Union[TedResult, NJTedResult],
    patched_root: TreeNode,
) -> str:
    """Human-readable comparison report. Works for both Chawathe and NJ results."""
    summary = summarize_edit_script(ted_result)

    lines = [
        f"Comparison: {source_slug} -> {target_slug}",
        f"Algorithm: {ted_result.algorithm}",
        f"Distance: {ted_result.distance}",
        f"Similarity: {ted_result.similarity:.4f}",
        "",
        "Edit script summary:",
    ]
    for k, v in summary.items():
        if k != "total":
            lines.append(f"- {k}: {v}")
    lines.append(f"- total: {summary['total']}")

    if isinstance(ted_result, NJTedResult):
        lines.append("")
        lines.append("Operations:")
        for idx, op in enumerate(ted_result.operations, start=1):
            lines.append(f"  {idx}. {op.note or op.op}")

    lines.extend(["", "Patched infobox:", tree_to_infobox_text(patched_root)])
    return "\n".join(lines)
