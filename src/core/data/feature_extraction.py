"""
Feature extraction from tree documents.
Collects available field paths (dot notation) and prunes trees to selected features.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set


def _collect_paths_from_node(node: Dict[str, Any], prefix: str, skip_root: bool) -> List[str]:
    """
    Recursively collect dot-notation paths to all leaf nodes.
    Leaf = node with no children (or empty children).
    skip_root: if True, omit the root label from paths (for consistent schema across countries).
    """
    label = node.get("label", "")
    path = f"{prefix}.{label}" if prefix else label

    children = node.get("children") or []
    if not children:
        return [path]

    paths: List[str] = []
    for child in children:
        paths.extend(_collect_paths_from_node(child, path, skip_root))
    return paths


def collect_tree_paths(tree: Dict[str, Any], *, include_root: bool = False) -> List[str]:
    """
    Extract all dot-notation paths from a tree (e.g. meta.country_name, fields.area.total_km2).
    Returns sorted, unique list of paths to leaf nodes.
    By default, omits root label so paths are consistent across countries.
    """
    paths = _collect_paths_from_node(tree, "", skip_root=not include_root)
    if not include_root and tree.get("children"):
        # Strip root prefix (e.g. "lebanon.meta.country_name" -> "meta.country_name")
        root_label = tree.get("label", "")
        prefix = f"{root_label}."
        stripped = [p[len(prefix):] if p.startswith(prefix) else p for p in paths]
        paths = [p for p in stripped if p]
    return sorted(set(paths))


def collect_all_available_features(trees: List[Dict[str, Any]]) -> List[str]:
    """
    Merge paths from multiple trees to get the union of all available features.
    Returns sorted, unique list.
    """
    seen: Set[str] = set()
    for tree in trees:
        for path in collect_tree_paths(tree):
            seen.add(path)
    return sorted(seen)


def _path_to_segments(path: str) -> List[str]:
    """Split dot-notation path into segments."""
    return [s for s in path.split(".") if s]


def _path_matches_selected(
    path_segments: List[str],
    selected: Set[tuple],
    root_label: Optional[str] = None,
) -> bool:
    """
    Check if path is included by any selected feature.
    Selected features omit root (e.g. meta.country_name); path_segments may include root.
    """
    # Strip root for matching (selected paths are root-relative)
    segments = path_segments
    if root_label and path_segments and path_segments[0] == root_label:
        segments = path_segments[1:]
    path_tuple = tuple(segments)
    for sel in selected:
        if len(sel) <= len(path_tuple) and path_tuple[: len(sel)] == sel:
            return True
    return False


def _prune_node(
    node: Dict[str, Any],
    path_so_far: List[str],
    selected: Set[tuple],
    root_label: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Recursively prune tree to only include nodes on selected paths.
    Returns None if node and its subtree should be excluded.
    """
    label = node.get("label", "")
    current_path = path_so_far + [label]

    children = node.get("children") or []
    if not children:
        # Leaf node: include if path matches any selected
        if _path_matches_selected(current_path, selected, root_label):
            return {"label": label, "value": node.get("value"), "children": []}
        return None

    # Branch: recurse into children
    pruned_children: List[Dict[str, Any]] = []
    for child in children:
        pruned = _prune_node(child, current_path, selected, root_label)
        if pruned is not None:
            pruned_children.append(pruned)

    if not pruned_children:
        # No matching descendants - include if this exact path is selected (intermediate node)
        rel_path = current_path[1:] if root_label and current_path and current_path[0] == root_label else current_path
        if tuple(rel_path) in selected:
            return {"label": label, "value": node.get("value"), "children": []}
        return None

    return {
        "label": label,
        "value": node.get("value"),
        "children": pruned_children,
    }


def _path_matches_excluded(
    path_segments: List[str],
    excluded: Set[tuple],
    root_label: Optional[str] = None,
) -> bool:
    """True if path is excluded (matches or is under an excluded path)."""
    return _path_matches_selected(path_segments, excluded, root_label)


def _prune_node_exclude(
    node: Dict[str, Any],
    path_so_far: List[str],
    excluded: Set[tuple],
    root_label: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Recursively prune tree to exclude nodes on excluded paths.
    Returns None if node should be excluded.
    """
    label = node.get("label", "")
    current_path = path_so_far + [label]

    children = node.get("children") or []
    if not children:
        # Leaf: exclude if path matches
        if _path_matches_excluded(current_path, excluded, root_label):
            return None
        return {"label": label, "value": node.get("value"), "children": []}

    # Branch: recurse into children
    pruned_children: List[Dict[str, Any]] = []
    for child in children:
        pruned = _prune_node_exclude(child, current_path, excluded, root_label)
        if pruned is not None:
            pruned_children.append(pruned)

    # Exclude branch if all descendants were excluded
    if not pruned_children:
        return None

    return {
        "label": label,
        "value": node.get("value"),
        "children": pruned_children,
    }


def extract_selected_features(tree: Dict[str, Any], features: List[str]) -> Dict[str, Any]:
    """
    Traverse tree and extract only selected paths.
    Returns a pruned tree preserving structure for TED (root -> ... -> leaf).

    features: list of dot-notation paths, e.g. ["meta.country_name", "fields.area.total_km2"]
    """
    if not features:
        return tree

    selected_segments: Set[tuple] = set()
    for f in features:
        segs = _path_to_segments(f)
        if segs:
            selected_segments.add(tuple(segs))

    root_label = tree.get("label") or "root"
    pruned = _prune_node(tree, [], selected_segments, root_label)
    if pruned is None:
        # Root was pruned (shouldn't happen if features are valid)
        return {"label": tree.get("label", "root"), "value": None, "children": []}
    return pruned


def extract_excluding_features(tree: Dict[str, Any], features: List[str]) -> Dict[str, Any]:
    """
    Traverse tree and exclude selected paths. Keeps everything else.
    Returns a pruned tree preserving structure for TED.

    features: list of dot-notation paths to EXCLUDE, e.g. ["meta.retrieved_at", "fields.area.water_percent"]
    """
    if not features:
        return tree

    excluded_segments: Set[tuple] = set()
    for f in features:
        segs = _path_to_segments(f)
        if segs:
            excluded_segments.add(tuple(segs))

    root_label = tree.get("label") or "root"
    pruned = _prune_node_exclude(tree, [], excluded_segments, root_label)
    if pruned is None:
        return {"label": tree.get("label", "root"), "value": None, "children": []}
    return pruned
