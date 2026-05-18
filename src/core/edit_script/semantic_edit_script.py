"""
Post-process semantic (path-based) edit scripts: field matching, deduplication, meta cleanup.
Does not affect Chawathe LD-pair patching — only the clean/human-readable script.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence


def _leaf_base(segment: str) -> str:
    return segment.split("[", 1)[0]


def _path_with_renamed_leaf(path: Sequence[str], new_leaf_label: str) -> List[str]:
    """Rename the last segment, preserving a list suffix like [0] if present."""
    if not path:
        return [new_leaf_label]
    out = list(path)
    old_last = out[-1]
    if "[" in old_last:
        idx = old_last.index("[")
        out[-1] = _leaf_base(new_leaf_label) + old_last[idx:]
    else:
        out[-1] = new_leaf_label
    return out


def _extract_labels(op: Dict[str, Any]) -> tuple[str | None, str | None]:
    old_n = op.get("old_node")
    new_n = op.get("new_node")
    old_label = op.get("old_label")
    new_label = op.get("new_label")
    if isinstance(old_n, dict) and old_label is None:
        old_label = old_n.get("label")
    if isinstance(new_n, dict) and new_label is None:
        new_label = new_n.get("label")
    return (old_label if isinstance(old_label, str) else None, new_label if isinstance(new_label, str) else None)


def _extract_values_for_split(op: Dict[str, Any], old_n: Any, new_n: Any) -> tuple[Any, Any]:
    old_v = op.get("old_value")
    new_v = op.get("new_value")
    if isinstance(old_n, dict) and old_n.get("value") is not None:
        old_v = old_n.get("value")
    if isinstance(new_n, dict) and new_n.get("value") is not None:
        new_v = new_n.get("value")
    return old_v, new_v


def enforce_field_matching(ops: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Same field label → keep a single update. Different labels → delete + insert at correct paths.

    Handles optional hybrid keys (old_node/new_node) on semantic ops; strips those keys from
    plain updates.
    """
    out: List[Dict[str, Any]] = []
    for op in ops:
        if op.get("op") != "update":
            out.append(dict(op))
            continue

        path = op.get("path") or []
        old_label, new_label = _extract_labels(op)
        old_node = op.get("old_node")
        new_node = op.get("new_node")

        if old_label and new_label and old_label != new_label:
            old_path = list(path)
            new_path = _path_with_renamed_leaf(path, new_label)
            old_v, new_v = _extract_values_for_split(op, old_node, new_node)
            out.append({
                "op": "delete",
                "path": old_path,
                "old_value": old_v,
            })
            out.append({
                "op": "insert",
                "path": new_path,
                "value": new_v,
            })
            continue

        # Plain update: drop hybrid keys for a stable semantic shape
        clean = {k: v for k, v in op.items() if k not in ("old_node", "new_node", "old_label", "new_label")}
        out.append(clean)
    return out


def deduplicate_ops(ops: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate operations (same op, path, and value payloads)."""
    seen: set[tuple[Any, ...]] = set()
    result: List[Dict[str, Any]] = []
    for op in ops:
        key = (
            op.get("op"),
            tuple(op.get("path") or ()),
            json.dumps(op.get("old_value"), sort_keys=True, default=str),
            json.dumps(op.get("new_value"), sort_keys=True, default=str),
            json.dumps(op.get("value"), sort_keys=True, default=str),
        )
        if key not in seen:
            seen.add(key)
            result.append(dict(op))
    return result


def remove_meta_duplicates(ops: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Drop changes under meta.country_name — canonical field lives under fields.country_name.
    """
    filtered: List[Dict[str, Any]] = []
    for op in ops:
        path = op.get("path") or []
        if len(path) >= 2 and path[0] == "meta" and _leaf_base(path[1]) == "country_name":
            continue
        filtered.append(dict(op))
    return filtered


def postprocess_semantic_edit_script(ops: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply: field matching → deduplicate → drop meta country_name duplicate."""
    step1 = enforce_field_matching(ops)
    step2 = deduplicate_ops(step1)
    return remove_meta_duplicates(step2)
