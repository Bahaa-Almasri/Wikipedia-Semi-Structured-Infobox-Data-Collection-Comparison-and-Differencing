"""
Path-based patch: apply dot-notation updates onto a tree without replacing it.
Preserves all fields not mentioned in the patch (immutability via deep copy).

Feature-driven transformation:
- SOURCE tree = base (immutable)
- TARGET tree = only used to EXTRACT values
- Build PATCH = only selected features
- APPLY PATCH on SOURCE
"""
from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# When country_name is selected, update BOTH meta and fields for consistency
COUNTRY_NAME_PATHS = {"meta.country_name", "fields.country_name"}


def normalize_path(path: str) -> str:
    """
    Ensure path has explicit 'fields.' or 'meta.' prefix for tree traversal.
    Tree structure is Root -> meta | fields -> area, economy, etc.
    Paths like "area.total_km2" must become "fields.area.total_km2".
    """
    if not path or not path.strip():
        return path
    path = path.strip()
    if path.startswith("fields.") or path.startswith("meta."):
        return path
    return f"fields.{path}"


def path_exists_in_tree(tree: Dict[str, Any], path: str) -> bool:
    """Return True if the path exists in the tree (reaches a leaf node)."""
    if not tree or not isinstance(tree, dict):
        return False
    segments = _path_to_segments(path)
    if not segments:
        return False
    current = tree
    for i, seg in enumerate(segments):
        children = current.get("children")
        if children is None:
            return i == len(segments) - 1
        idx = _find_child_by_label(current, seg)
        if idx is None:
            return False
        current = children[idx]
    return True


def get_value_from_tree(tree: Dict[str, Any], path: str) -> Optional[Any]:
    """
    Traverse tree by dot-notation path and return the leaf value.
    Path is root-relative (e.g. "meta.country_name", "fields.area.total_km2").
    Returns None if path not found or node has no value.
    Handles nodes with direct value, or token/value children (semantic leaves).
    """
    if not tree or not isinstance(tree, dict):
        return None
    segments = _path_to_segments(path)
    if not segments:
        return None
    current = tree
    for i, seg in enumerate(segments):
        children = current.get("children")
        if children is None:
            if i == len(segments) - 1:
                return current.get("value")
            return None
        idx = _find_child_by_label(current, seg)
        if idx is None:
            return None
        current = children[idx]
    return _extract_leaf_value(current)


def get_values_from_tree(tree: Dict[str, Any], path: str) -> List[Any]:
    """
    Like get_value_from_tree but collects ALL values when path matches multiple
    children (e.g. internet_tld.internet_tld_item with items ["ch", "لبنان"]).
    Returns list of non-None values; empty list if path not found.
    """
    if not tree or not isinstance(tree, dict):
        return []
    segments = _path_to_segments(path)
    if not segments:
        return []
    current = tree
    for i, seg in enumerate(segments):
        children = current.get("children")
        if children is None:
            if i == len(segments) - 1:
                val = current.get("value")
                return [val] if val is not None else []
            return []
        if i == len(segments) - 1:
            # Last segment: collect all matching children
            base = seg.split("[", 1)[0]
            values: List[Any] = []
            for child in children:
                child_base = (child.get("label") or "").split("[", 1)[0]
                if child_base == base:
                    v = _extract_leaf_value(child)
                    if v is not None:
                        values.append(v)
            return values
        idx = _find_child_by_label(current, seg)
        if idx is None:
            return []
        current = children[idx]
    return []


def _extract_leaf_value(node: Dict[str, Any]) -> Optional[Any]:
    """
    Extract display value from a leaf node. Handles direct value, or token/value children.
    """
    val = node.get("value")
    if val is not None:
        return val
    children = node.get("children") or []
    if not children:
        return None
    child_labels = [c.get("label") for c in children]
    if all(lbl == "token" for lbl in child_labels):
        tokens = [str(c.get("value", "")).strip() for c in children if c.get("value")]
        return " ".join(tokens) if tokens else None
    if all(lbl == "value" for lbl in child_labels) and len(children) == 1:
        return children[0].get("value")
    return None


def build_patch_from_features(
    source_tree: Dict[str, Any],
    target_tree: Dict[str, Any],
    selected_features: List[str],
) -> Dict[str, Any]:
    """
    Build a patch from TARGET tree for ONLY the selected feature paths.
    SOURCE is immutable reference; TARGET is used only to EXTRACT values.
    Excluded paths must NOT be in selected_features.

    Returns patch dict: {path: value} for each selected path where target has a value.
    Paths are normalized (fields./meta. prefix) so traversal succeeds.
    Repeated keys (e.g. internet_tld_item) are stored as lists.
    """
    # Debug logging (temporary)
    print("SELECTED FEATURES:", selected_features)
    normalized_features = [normalize_path(f) for f in selected_features]
    print("NORMALIZED FEATURES:", normalized_features)

    selected_set: Set[str] = set(selected_features)
    patch: Dict[str, Any] = {}

    # Sync meta.country_name and fields.country_name when either is selected
    country_name_selected = selected_set & COUNTRY_NAME_PATHS
    if country_name_selected:
        val = get_value_from_tree(target_tree, "fields.country_name")
        if val is None:
            val = get_value_from_tree(target_tree, "meta.country_name")
        for p in COUNTRY_NAME_PATHS:
            if val is not None:
                patch[p] = _tree_value(val)
            elif path_exists_in_tree(source_tree, normalize_path(p)):
                patch[p] = None

    for path in selected_set:
        if path in COUNTRY_NAME_PATHS:
            continue  # Already handled above
        normalized = normalize_path(path)
        values = get_values_from_tree(target_tree, normalized)
        # Skip empty values (important)
        values = [v for v in values if v is not None and v != ""]
        if values:
            if len(values) > 1:
                patch[normalized] = [_tree_value(v) for v in values]
            else:
                patch[normalized] = _tree_value(values[0])
        else:
            if path_exists_in_tree(source_tree, normalized):
                patch[normalized] = None
            else:
                logger.debug("Path not found in target or source: %s (normalized: %s)", path, normalized)

    print("PATCH:", patch)
    return patch


def _path_to_segments(path: str) -> List[str]:
    """Split dot-notation path into segments."""
    return [s for s in path.split(".") if s]


def _find_child_by_label(node: Dict[str, Any], label: str) -> Optional[int]:
    """Return index of child with given label, or None. Prefer exact match."""
    children = node.get("children") or []
    for i, child in enumerate(children):
        if child.get("label") == label:
            return i
    # Fallback: match base (e.g. "languages_item" matches "languages_item[0]")
    base = label.split("[", 1)[0]
    for i, child in enumerate(children):
        child_base = (child.get("label") or "").split("[", 1)[0]
        if child_base == base:
            return i
    return None


def _ensure_path(
    node: Dict[str, Any],
    segments: List[str],
    root_label: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Traverse tree by label segments. Create intermediate nodes if missing.
    Returns the leaf node dict (or intermediate if path is partial), or None if
    an intermediate node is not a dict-like structure (has value, no children).
    """
    current = node
    # If root has a different structure, skip root label in path
    seg_start = 0
    if root_label and segments and current.get("label") == root_label:
        # Path is root-relative; we're already at root
        pass

    for i, seg in enumerate(segments):
        children = current.get("children")
        if children is None:
            # Leaf node - cannot traverse further
            if i < len(segments) - 1:
                logger.warning(
                    "Path %s: intermediate node at %s is a leaf (has value, no children); "
                    "cannot create nested path.",
                    ".".join(segments),
                    ".".join(segments[: i + 1]),
                )
                return None
            return current

        idx = _find_child_by_label(current, seg)
        if idx is not None:
            current = children[idx]
        else:
            # Create new child
            is_leaf = i == len(segments) - 1
            new_child: Dict[str, Any] = {
                "label": seg,
                "value": None if not is_leaf else "",
                "children": [] if not is_leaf else [],
            }
            children.append(new_child)
            current = new_child

    return current


def apply_patch_to_tree(
    original: Dict[str, Any],
    patch: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Applies a dot-notation patch onto a deep copy of the original tree.
    Only updates specified fields; all other fields remain unchanged.

    Immutability: original is never mutated. A deep copy is made first.

    Nested updates: For path "fields.economy.gdp_nominal_total", traverses
    root -> fields -> economy -> gdp_nominal_total and sets the leaf value.
    Intermediate nodes are created if they do not exist.
    Paths are normalized (fields./meta. prefix) before traversal.

    patch: Dict mapping dot-notation paths to values, e.g.:
        {"fields.economy.gdp_nominal_total": "18B"}
    List values (e.g. for internet_tld_item): replace all matching children.

    Patch value None: sets the leaf value to None (field remains, value cleared).
    """
    if not isinstance(original, dict) or "label" not in original:
        raise ValueError("original must be a TreeNode dict with 'label' key")

    result = deepcopy(original)
    if "children" not in result or result["children"] is None:
        result["children"] = []
    root_label = result.get("label")

    for path_str, value in patch.items():
        normalized_path = normalize_path(path_str)
        segments = _path_to_segments(normalized_path)
        if not segments:
            continue

        # Handle list values for repeated keys (e.g. internet_tld_item)
        if isinstance(value, list) and segments[-1].endswith("_item"):
            _apply_list_patch(result, segments, value, root_label)
            continue

        leaf = _ensure_path(result, segments, root_label)
        if leaf is not None:
            # Ensure value is JSON-serializable (TreeNode expects Optional[str])
            leaf["value"] = None if value is None else str(value)
        # If leaf is None, we logged a warning; skip this path

    return result


def _apply_list_patch(
    node: Dict[str, Any],
    segments: List[str],
    values: List[Any],
    root_label: Optional[str],
) -> None:
    """
    Apply patch for path ending in _item when value is a list.
    Replaces all matching children with new items from values.
    """
    if len(segments) < 2:
        return
    # Traverse to parent of last segment
    current = node
    for i, seg in enumerate(segments[:-1]):
        children = current.get("children")
        if children is None:
            return
        idx = _find_child_by_label(current, seg)
        if idx is None:
            return
        current = children[idx]
    parent = current
    children = parent.get("children") or []
    last_seg = segments[-1]
    base = last_seg.split("[", 1)[0]
    # Remove all children matching base
    new_children = [c for c in children if (c.get("label") or "").split("[", 1)[0] != base]
    # Add new children for each value
    for v in values:
        str_val = None if v is None else str(v)
        new_children.append({"label": base, "value": str_val, "children": []})
    parent["children"] = new_children


def _tree_value(value: Any) -> Optional[str]:
    """Normalize value for tree node (TreeNode.value is Optional[str]). Ensures all values are JSON-serializable strings."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return str(value)  # Complex values stringified for leaf
    return str(value)  # int, float, bool, str -> string for tree node


def edit_script_clean_to_patch(edit_script_clean: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert path-based edit script (update/insert/delete) to dot-notation patch.
    - update: path -> new_value
    - insert: path -> value
    - delete: path -> None (clears the field)
    """
    patch: Dict[str, Any] = {}
    for op in edit_script_clean:
        path = op.get("path")
        if path is None:
            path = []
        if not isinstance(path, (list, tuple)):
            path = [path]  # single segment
        if not path:
            continue
        path_str = ".".join(str(p) for p in path)

        kind = op.get("op")
        if kind == "update":
            patch[path_str] = _tree_value(op.get("new_value"))
        elif kind == "insert":
            patch[path_str] = _tree_value(op.get("value"))
        elif kind == "delete":
            patch[path_str] = None
    return patch
