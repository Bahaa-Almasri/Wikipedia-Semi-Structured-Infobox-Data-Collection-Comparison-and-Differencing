"""
Service layer for Wikipedia infobox data. Uses core storage (MongoDB only).
All data access and run operations go through this module; the API controller calls only this service.
"""
from __future__ import annotations

import logging
import random
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

from core.data.feature_extraction import (
    collect_all_available_features,
    collect_tree_paths,
    extract_excluding_features,
    extract_selected_features,
)
from core.data.storage import (
    list_slugs,
    read_all_json_documents,
    read_json_document,
    read_raw_html,
    read_tree_document,
    read_vsm_index,
    write_vsm_index,
)
from core.patch.patch import apply_patch_from_dict
from core.patch.path_patch import (
    apply_patch_to_tree,
    build_patch_from_features,
    edit_script_clean_to_patch,
    normalize_path,
)
from core.postprocess.postprocess import tree_to_infobox_text, tree_to_json_string, tree_to_xml_string
from core.edit_script.edit_script_ops_summary import (
    summarize_raw_edit_script_operations,
    summarize_semantic_diff_operations,
)
from core.edit_script.edit_script_normalize import (
    IGNORE_FIELDS,
    ignored_path,
    normalize_edit_script_for_algorithm,
)
from core.edit_script.semantic_edit_script import postprocess_semantic_edit_script
from core.preprocess.tree_builder import build_and_save_tree_for_slug, build_and_save_trees_for_all
from core.preprocess.pipeline import collect_all_countries
from utils.tree_utils import clone_tree, validate_tree
from core.similarity.clustering import (
    SUPPORTED_AGGLOMERATIVE_LINKAGES,
    SUPPORTED_CLUSTER_ALGORITHMS,
    SUPPORTED_CLUSTER_DISTANCES,
    build_ted_similarity_index,
    cluster_vsm_index,
)
from core.similarity.ted import compute_ted
from core.similarity.vsm import (
    SUPPORTED_VSM_METRICS,
    SUPPORTED_VSM_DOCUMENT_SOURCES,
    SUPPORTED_VSM_MODES,
    build_vsm_index,
    rank_document,
    rank_query,
    sparse_similarity,
    vsm_document_from_json,
)
from domain.models.tree import TreeNode
from domain.models.vsm import VSMIndex
from utils.compare import compare_country_slugs, compare_from_tree_dicts, load_tree_for_slug

# Alias for semantic diff filtering (same set as edit-script noise filter).
IGNORED_FIELDS = IGNORE_FIELDS

# Display placeholder for missing / empty scalar values in human-readable output.
EMPTY_VALUE_DISPLAY = "∅"

SUPPORTED_CLUSTER_SOURCES = {"vsm", "ted"}
SUPPORTED_TED_CLUSTER_ALGORITHMS = {"chawathe", "nj", "zhang_shasha"}

_MISSING = object()
_GAME_ROUNDS: Dict[str, str] = {}


def _is_empty_value(value: Any) -> bool:
    """True if value is None, empty string, or recursively empty (dict/list of empties)."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, dict):
        return all(_is_empty_value(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_is_empty_value(item) for item in value)
    return False


def _should_filter_empty_diff_op(op: Dict[str, Any]) -> bool:
    """Filter diff ops that only involve empty values (not meaningful differences)."""
    kind = op.get("op")
    if kind == "delete":
        return _is_empty_value(op.get("old_value"))
    if kind == "insert":
        return _is_empty_value(op.get("value"))
    if kind == "update":
        return _is_empty_value(op.get("old_value")) and _is_empty_value(op.get("new_value"))
    return False


SEMANTIC_FIELDS = {
    "capital",
    "country_name",
    "leader",
    "head_of_government",
    "head_of_state",
    "government",
    "currency",
    "economy",
    "gdp_nominal_total",
    "gdp_ppp_total",
}


def _is_ignored_path(path: Sequence[str]) -> bool:
    return ignored_path(path)


def _normalize_scalar(value: Optional[str]) -> Optional[Union[str, float, int]]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return ""
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def _leaf_base(segment: str) -> str:
    return str(segment).split("[", 1)[0]


def _node_to_semantic_value(node: TreeNode) -> Any:
    if not node.children:
        return _normalize_scalar(node.value)

    child_labels = [child.label for child in node.children]
    if all(label == "token" for label in child_labels):
        tokens = [str(child.value).strip() for child in node.children if child.value is not None]
        return " ".join(token for token in tokens if token)
    if all(label == "value" for label in child_labels):
        if len(node.children) == 1:
            return _normalize_scalar(node.children[0].value)
        return [_normalize_scalar(child.value) for child in node.children]

    grouped: Dict[str, List[Any]] = {}
    for child in node.children:
        grouped.setdefault(child.label, []).append(_node_to_semantic_value(child))

    result: Dict[str, Any] = {}
    for key, values in grouped.items():
        if key.endswith("_item"):
            normalized_key = key[:-5]
            existing = result.get(normalized_key, [])
            if not isinstance(existing, list):
                existing = [existing]
            existing.extend(values)
            result[normalized_key] = existing
        elif len(values) == 1:
            result[key] = values[0]
        else:
            result[key] = values
    return result


def _is_atomic_node(path: Sequence[str], node: TreeNode) -> bool:
    """
    True only for leaf-like nodes: no children, or only token/value leaves, or
    SEMANTIC_FIELDS with only token/value leaves.

    Structured subtrees (e.g. government, economy, coordinates with nested keys)
    are NOT atomic so we recurse and emit field-level ops.
    """
    label = path[-1] if path else node.label
    base = _leaf_base(str(label))
    if not node.children:
        return True
    child_labels = [child.label for child in node.children]
    only_tokens = all(lbl in {"token", "value"} for lbl in child_labels)
    if base in SEMANTIC_FIELDS and only_tokens:
        return True
    if only_tokens:
        return True
    return False


def _diff_dict_semantic(
    old_d: Dict[str, Any],
    new_d: Dict[str, Any],
    path: List[str],
) -> List[Dict[str, Any]]:
    """
    Field-level diff for two dict-shaped semantic values (same parent path).
    Recurses for nested dicts; does not emit a single blob update for the parent.
    """
    changes: List[Dict[str, Any]] = []
    keys = sorted(set(old_d.keys()) | set(new_d.keys()))
    for k in keys:
        child_path = [*path, k]
        ov = old_d.get(k, _MISSING)
        nv = new_d.get(k, _MISSING)
        if ov is _MISSING and nv is not _MISSING:
            if _is_ignored_path(child_path):
                continue
            changes.append({"op": "insert", "path": child_path, "value": nv})
            continue
        if nv is _MISSING and ov is not _MISSING:
            if _is_ignored_path(child_path):
                continue
            changes.append({"op": "delete", "path": child_path, "old_value": ov})
            continue
        # Both dicts have key k
        if isinstance(ov, dict) and isinstance(nv, dict):
            changes.extend(_diff_dict_semantic(ov, nv, child_path))
        elif ov != nv:
            if _is_ignored_path(child_path):
                continue
            changes.append({
                "op": "update",
                "path": child_path,
                "old_value": ov,
                "new_value": nv,
            })
    return changes


def _child_keyed(children: Sequence[TreeNode]) -> Dict[str, TreeNode]:
    grouped: Dict[str, List[TreeNode]] = {}
    for child in children:
        grouped.setdefault(child.label, []).append(child)

    keyed: Dict[str, TreeNode] = {}
    for label, nodes in grouped.items():
        if len(nodes) == 1:
            keyed[label] = nodes[0]
            continue
        for idx, node in enumerate(nodes):
            keyed[f"{label}[{idx}]"] = node
    return keyed


def _semantic_diff(
    source: Optional[TreeNode],
    target: Optional[TreeNode],
    path: List[str],
) -> List[Dict[str, Any]]:
    if _is_ignored_path(path):
        return []

    if source is None and target is None:
        return []
    if source is None and target is not None:
        return [{
            "op": "insert",
            "path": path,
            "value": _node_to_semantic_value(target),
        }]
    if target is None and source is not None:
        return [{
            "op": "delete",
            "path": path,
            "old_value": _node_to_semantic_value(source),
        }]

    assert source is not None
    assert target is not None

    if _is_atomic_node(path, source) or _is_atomic_node(path, target):
        old_value = _node_to_semantic_value(source)
        new_value = _node_to_semantic_value(target)
        if isinstance(old_value, dict) and isinstance(new_value, dict):
            return _diff_dict_semantic(old_value, new_value, path)
        if old_value != new_value:
            if _is_ignored_path(path):
                return []
            return [{
                "op": "update",
                "path": path,
                "old_value": old_value,
                "new_value": new_value,
            }]
        return []

    changes: List[Dict[str, Any]] = []
    source_children = _child_keyed(source.children)
    target_children = _child_keyed(target.children)
    all_keys = sorted(set(source_children.keys()) | set(target_children.keys()))
    for key in all_keys:
        changes.extend(_semantic_diff(source_children.get(key), target_children.get(key), [*path, key]))
    return changes


def clean_edit_script(
    source_tree: Dict[str, Any],
    target_tree: Dict[str, Any],
) -> List[Dict[str, Any]]:
    source_root = TreeNode.from_dict(source_tree)
    target_root = TreeNode.from_dict(target_tree)
    source_children = {child.label: child for child in source_root.children}
    target_children = {child.label: child for child in target_root.children}
    all_sections = sorted(set(source_children.keys()) | set(target_children.keys()))

    changes: List[Dict[str, Any]] = []
    for section in all_sections:
        changes.extend(
            _semantic_diff(
                source_children.get(section),
                target_children.get(section),
                [section],
            )
        )
    return postprocess_semantic_edit_script(changes)


def _path_to_label(path: Sequence[str]) -> str:
    if not path:
        return "Root"
    label = str(path[-1]).replace("_", " ")
    label = label.split("[", 1)[0]
    return label.title()


def _format_value(value: Any) -> str:
    if value is None:
        return EMPTY_VALUE_DISPLAY
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped == "None":
            return EMPTY_VALUE_DISPLAY
    if isinstance(value, dict):
        parts = []
        for key, val in value.items():
            key_label = str(key).replace("_", " ").title()
            parts.append(f"{key_label}: {_format_value(val)}")
        return ", ".join(parts)
    if isinstance(value, list):
        return ", ".join(_format_value(item) for item in value)
    return str(value)


def format_edit_script_human(edit_script: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for op in edit_script:
        kind = op.get("op")
        label = _path_to_label(op.get("path") or [])
        if kind == "update":
            lines.append(f"- {label}: {_format_value(op.get('old_value'))}")
            lines.append(f"+ {label}: {_format_value(op.get('new_value'))}")
        elif kind == "insert":
            lines.append(f"+ {label}: {_format_value(op.get('value'))}")
        elif kind == "delete":
            lines.append(f"- {label}: {_format_value(op.get('old_value'))}")
    return "\n".join(lines)


def get_country_index() -> List[Tuple[str, str]]:
    """
    Return a list of (slug, display_name) tuples.
    display_name is from meta.country_name when available, else slug title-cased.
    """
    slugs = list_slugs()
    result: List[Tuple[str, str]] = []
    for slug in slugs:
        doc = read_json_document(slug)
        if doc is None:
            result.append((slug, slug.replace("_", " ").title()))
            continue
        meta = (doc.get("meta") or {})
        name = meta.get("country_name") or slug.replace("_", " ").title()
        result.append((slug, name))
    return result


def get_json_document(slug: str) -> Optional[Dict[str, Any]]:
    """Return the full JSON document for a country, or None."""
    return read_json_document(slug)


def get_tree_document(slug: str) -> Optional[Dict[str, Any]]:
    """Return the tree document for a country, or None."""
    return read_tree_document(slug)


def get_raw_html(slug: str) -> Optional[str]:
    """Return the raw infobox HTML for a country, or None."""
    return read_raw_html(slug)


def get_available_features() -> List[str]:
    """
    Return all available feature paths (dot notation) from tree schema.
    Dynamically generated by collecting paths from all country trees.
    """
    slugs = list_slugs()
    trees: List[Dict[str, Any]] = []
    for slug in slugs[:50]:  # Sample up to 50 countries to get schema coverage
        tree = read_tree_document(slug)
        if tree is not None:
            trees.append(tree)
    if not trees:
        return []
    return collect_all_available_features(trees)


# --- Run operations ---


def run_collect_pipeline() -> List[str]:
    """
    Run the full collection pipeline: fetch UN member states, scrape infoboxes, store in MongoDB.
    Returns the list of slugs for which documents were written.
    """
    return collect_all_countries()


def run_build_trees(slug: Optional[str] = None) -> List[str]:
    """
    Build trees from JSON documents in MongoDB and write them back.
    If slug is given, build only for that country; otherwise build for all.
    Returns the list of slugs for which trees were written.
    """
    if slug is not None:
        result = build_and_save_tree_for_slug(slug)
        return [result] if result is not None else []
    return build_and_save_trees_for_all()


# --- TED: similarity, diff, patch, postprocess ---


def ted_similarity(
    source_slug: str,
    target_slug: str,
    *,
    algorithm: str = "chawathe",
    coerce_root_label: Optional[str] = None,
    cost_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute TED distance and similarity between two country trees."""
    source_root = load_tree_for_slug(source_slug)
    target_root = load_tree_for_slug(target_slug)
    al = (algorithm or "").lower()
    if al == "zhang_shasha":
        t0 = time.perf_counter()
        result = compute_ted(
            source_root, target_root,
            algorithm=algorithm,
            coerce_root_label=coerce_root_label,
            cost_model=cost_model,
        )
        runtime_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "source_slug": source_slug,
            "target_slug": target_slug,
            "algorithm": result.algorithm,
            "distance": result.distance,
            "similarity": result.similarity,
            "source_size": result.source_size,
            "target_size": result.target_size,
            "runtime_ms": runtime_ms,
        }
    result = compute_ted(
        source_root, target_root,
        algorithm=algorithm,
        coerce_root_label=coerce_root_label,
        cost_model=cost_model,
    )
    return {
        "source_slug": source_slug,
        "target_slug": target_slug,
        "algorithm": result.algorithm,
        "distance": result.distance,
        "similarity": result.similarity,
        "source_size": result.source_size,
        "target_size": result.target_size,
    }


def ted_diff(
    source_slug: str,
    target_slug: str,
    *,
    algorithm: str = "chawathe",
    coerce_root_label: Optional[str] = None,
    cost_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute full comparison (TED + edit script + patched tree + report)."""
    return compare_country_slugs(
        source_slug, target_slug,
        algorithm=algorithm,
        coerce_root_label=coerce_root_label,
        cost_model=cost_model,
    )


def ted_diff_from_trees(
    source_tree: Dict[str, Any],
    target_tree: Dict[str, Any],
    *,
    source_slug: str = "source",
    target_slug: str = "target",
    algorithm: str = "chawathe",
    coerce_root_label: Optional[str] = None,
    cost_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute full comparison from two tree dicts (e.g. from API)."""
    return compare_from_tree_dicts(
        source_tree, target_tree,
        source_slug=source_slug,
        target_slug=target_slug,
        algorithm=algorithm,
        coerce_root_label=coerce_root_label,
        cost_model=cost_model,
    )


def compare_countries(
    country_a: str,
    country_b: str,
    *,
    features: Optional[List[str]] = None,
    exclude: bool = False,
    algorithm: str = "chawathe",
    coerce_root_label: Optional[str] = None,
    cost_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare two countries by slug. Optionally restrict by features.
    If exclude=True, features are excluded; otherwise they are included.
    If features is None or empty, performs full tree comparison.
    Returns TED result plus source_tree, target_tree, and when features used:
    original_source_tree (full tree before pruning, for path-based patch).
    """
    source_tree = get_tree_document(country_a)
    target_tree = get_tree_document(country_b)
    if source_tree is None:
        raise ValueError(f"No tree for country: {country_a}")
    if target_tree is None:
        raise ValueError(f"No tree for country: {country_b}")

    original_source_tree: Optional[Dict[str, Any]] = None
    original_target_tree: Optional[Dict[str, Any]] = None
    if features:
        original_source_tree = source_tree  # Keep full trees for feature-driven patch
        original_target_tree = target_tree
        if exclude:
            source_tree = extract_excluding_features(source_tree, features)
            target_tree = extract_excluding_features(target_tree, features)
        else:
            source_tree = extract_selected_features(source_tree, features)
            target_tree = extract_selected_features(target_tree, features)

    result = ted_compute_from_trees(
        source_tree,
        target_tree,
        algorithm=algorithm,
        coerce_root_label=coerce_root_label,
        cost_model=cost_model,
    )
    result["source_tree"] = source_tree
    result["target_tree"] = target_tree
    if original_source_tree is not None:
        result["original_source_tree"] = original_source_tree
    if original_target_tree is not None:
        result["original_target_tree"] = original_target_tree
    return result


def ted_compute_from_trees(
    source_tree: Dict[str, Any],
    target_tree: Dict[str, Any],
    *,
    algorithm: str = "chawathe",
    coerce_root_label: Optional[str] = None,
    cost_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute TED metrics + edit script ONLY (no patch application).

    Returns edit_script as a list of serialized operations, so the streamlit can store it directly
    and apply it later via /ted/patch.
    """
    source_root = TreeNode.from_dict(source_tree)
    target_root = TreeNode.from_dict(target_tree)
    validate_tree(source_root)
    validate_tree(target_root)

    al = (algorithm or "").lower()
    if al == "zhang_shasha":
        t0 = time.perf_counter()
        ted_result = compute_ted(
            source_root,
            target_root,
            algorithm=algorithm,
            coerce_root_label=coerce_root_label,
            cost_model=cost_model,
        )
        runtime_ms = (time.perf_counter() - t0) * 1000.0
    else:
        ted_result = compute_ted(
            source_root,
            target_root,
            algorithm=algorithm,
            coerce_root_label=coerce_root_label,
            cost_model=cost_model,
        )
        runtime_ms = None

    # Same source shape as TED (coerced root label) for LD-pair replay / filtering.
    source_for_script = clone_tree(source_root)
    if coerce_root_label is not None:
        source_for_script.label = coerce_root_label

    edit_script_native = [op.to_dict() for op in ted_result.operations]
    zs_map = getattr(ted_result, "zhang_shasha_mappings", None)
    if al == "zhang_shasha" and zs_map:
        # TED native output is postorder node alignments, not LD-pair ops.
        edit_script_native = [
            {
                "op": "map",
                "source_id": m["source_id"],
                "target_id": m["target_id"],
            }
            for m in zs_map
        ]
    edit_script_display = normalize_edit_script_for_algorithm(
        ted_result.algorithm,
        source_for_script,
        edit_script_native,
    )
    edit_script_clean = clean_edit_script(source_tree, target_tree)
    edit_script_human = format_edit_script_human(edit_script_clean)
    raw_edit_script_summary = summarize_raw_edit_script_operations(edit_script_native)
    display_edit_script_summary = summarize_raw_edit_script_operations(edit_script_display)
    semantic_diff_summary = summarize_semantic_diff_operations(edit_script_clean)

    out: Dict[str, Any] = {
        "algorithm": ted_result.algorithm,
        "distance": ted_result.distance,
        "similarity": ted_result.similarity,
        "edit_script": edit_script_display,
        "edit_script_raw": edit_script_native,
        "edit_script_display": edit_script_display,
        "edit_script_display_normalized": edit_script_display,
        "raw_edit_script_summary": raw_edit_script_summary,
        "display_edit_script_summary": display_edit_script_summary,
        "semantic_diff_summary": semantic_diff_summary,
        # Backward-compatible aliases (semantic was previously exposed as edit_script_summary).
        "edit_script_summary": semantic_diff_summary,
        "edit_script_raw_summary": raw_edit_script_summary,
        "edit_script_display_summary": display_edit_script_summary,
        "edit_script_clean": edit_script_clean,
        "edit_script_human": edit_script_human,
        "source_size": ted_result.source_size,
        "target_size": ted_result.target_size,
    }
    if zs_map is not None:
        out["mappings"] = zs_map
    if runtime_ms is not None:
        out["runtime_ms"] = runtime_ms
    return out


def ted_patch(
    source_tree: Dict[str, Any],
    edit_script: Union[Dict[str, Any], List[Dict[str, Any]]],
    *,
    algorithm: str = "chawathe",
    original_tree: Optional[Dict[str, Any]] = None,
    edit_script_clean: Optional[List[Dict[str, Any]]] = None,
    target_tree: Optional[Dict[str, Any]] = None,
    excluded_features: Optional[List[str]] = None,
    mappings: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Apply edit script or feature-driven patch to source tree; returns patched tree as dict.

    Feature-driven (PREFERRED when feature selection used):
    - When original_tree + target_tree + excluded_features are provided:
      SOURCE = original_tree (base, immutable), TARGET = target_tree (extract values only).
      Build patch from selected features (all paths minus excluded), apply on SOURCE.
    - Non-destructive, feature-driven, preserves excluded fields from source.

    Legacy path-based (when no target_tree/excluded_features):
    - When original_tree and edit_script_clean are provided:
      Apply edit_script_clean as patch onto original_tree.

    LD-pair (fallback): Apply edit_script via Chawathe/NJ algorithm.

    Zhang–Shasha: requires ``target_tree`` and postorder ``mappings`` (from TED alignment),
    either in ``edit_script`` dict or via ``mappings=``.
    """
    if original_tree is not None and target_tree is not None and excluded_features is not None:
        # Feature-driven patch: SOURCE base, TARGET for values, only selected features
        try:
            # Use UNION of paths from both trees so we don't miss target-only paths
            source_paths = set(collect_tree_paths(original_tree))
            target_paths = set(collect_tree_paths(target_tree))
            all_paths = sorted(source_paths | target_paths)
            # Normalize excluded paths so "area.total_km2" matches "fields.area.total_km2"
            excluded_set = set(normalize_path(ex) for ex in excluded_features)

            def _is_excluded(path: str) -> bool:
                normalized_path = normalize_path(path)
                if normalized_path in excluded_set:
                    return True
                for ex in excluded_set:
                    if normalized_path.startswith(ex + ".") or normalized_path == ex:
                        return True
                return False

            selected_features = [p for p in all_paths if not _is_excluded(p)]
            logger.info(
                "Feature-driven patch: %d selected, %d excluded",
                len(selected_features),
                len(excluded_features),
            )
            patch = build_patch_from_features(original_tree, target_tree, selected_features)
            logger.info("Patch built: %d paths from target", len(patch))
            patched_dict = apply_patch_to_tree(original_tree, patch)
        except Exception as e:
            raise ValueError(
                f"Feature-driven patch failed: {e}. "
                "Ensure original_tree and target_tree have valid TreeNode structure."
            ) from e
    elif original_tree is not None and edit_script_clean:
        # Legacy path-based patch (edit script from diff)
        try:
            patch = edit_script_clean_to_patch(edit_script_clean)
            patched_dict = apply_patch_to_tree(original_tree, patch)
        except Exception as e:
            raise ValueError(
                f"Path-based patch failed: {e}. "
                "Ensure original_tree has valid TreeNode structure and edit_script_clean paths match."
            ) from e
    elif (algorithm or "").lower() == "zhang_shasha":
        if target_tree is None:
            raise ValueError(
                "Patching with algorithm 'zhang_shasha' requires 'target_tree' and postorder mappings."
            )
        if isinstance(edit_script, dict):
            edit_script_dict = dict(edit_script)
        else:
            edit_script_dict = {"operations": list(edit_script) if edit_script else []}
        if mappings is not None:
            edit_script_dict["mappings"] = mappings
        if not edit_script_dict.get("mappings"):
            raise ValueError(
                "Zhang–Shasha patch requires 'mappings' (list of {source_id, target_id}) "
                "in the request body or inside edit_script."
            )
        patched_dict = apply_patch_from_dict(
            source_tree,
            edit_script_dict,
            algorithm=algorithm,
            target_tree_dict=target_tree,
        )
    else:
        # LD-pair patch (existing behavior)
        if isinstance(edit_script, list):
            edit_script_dict = {"operations": edit_script}
        else:
            edit_script_dict = edit_script
        patched_dict = apply_patch_from_dict(
            source_tree, edit_script_dict,
            algorithm=algorithm,
        )

    root = TreeNode.from_dict(patched_dict)
    result: Dict[str, Any] = {
        "patched_tree": patched_dict,
        "patched_tree_json": tree_to_json_string(root),
        "patched_tree_xml": tree_to_xml_string(root),
        "patched_infobox_text": tree_to_infobox_text(root),
    }
    if target_tree is not None:
        try:
            # Semantic diff: patched vs target (what still differs)
            raw_diff = clean_edit_script(patched_dict, target_tree)
            # Filter out empty-value differences (patched has empty nodes target lacks — not meaningful)
            patch_validation_diff = [
                op for op in raw_diff
                if not _should_filter_empty_diff_op(op)
            ]
            result["patch_validation_diff"] = patch_validation_diff
            # Align metric with UI: no meaningful differences => 1.0 (structural score can lag semantics)
            if not patch_validation_diff:
                result["tree_similarity"] = 1.0
            else:
                target_root = TreeNode.from_dict(target_tree)
                result["tree_similarity"] = tree_similarity(root, target_root)
        except Exception as e:
            logger.warning("Could not compute tree_similarity or patch diff: %s", e)
    return result


def postprocess_tree(tree_dict: Dict[str, Any]) -> Dict[str, str]:
    """Convert tree dict to JSON string, XML string, and infobox text."""
    root = TreeNode.from_dict(tree_dict)
    return {
        "json": tree_to_json_string(root),
        "xml": tree_to_xml_string(root),
        "infobox_text": tree_to_infobox_text(root),
    }


def similarity_ranking(
    country: str,
    top_k: int = 5,
    *,
    algorithm: str = "chawathe",
    coerce_root_label: Optional[str] = "infobox",
) -> List[Dict[str, Any]]:
    """
    Return top_k countries most similar to the given country by TED similarity.
    Excludes the source country from results.
    """
    source_tree = get_tree_document(country)
    if source_tree is None:
        raise ValueError(f"No tree for country: {country}")

    slugs = list_slugs()
    results: List[Dict[str, Any]] = []

    for slug in slugs:
        if slug == country:
            continue
        target_tree = get_tree_document(slug)
        if target_tree is None:
            continue
        try:
            result = ted_compute_from_trees(
                source_tree,
                target_tree,
                algorithm=algorithm,
                coerce_root_label=coerce_root_label,
            )
            results.append({"country": slug, "score": result["similarity"]})
        except Exception as e:
            logger.debug("Skip %s: %s", slug, e)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def similarity_ranking_both(
    country: str,
    top_k: int = 5,
    *,
    coerce_root_label: Optional[str] = "infobox",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Return top_k countries most similar to the given country using BOTH TED algorithms.
    Returns {"chawathe": [...], "nj": [...]}.
    """
    return {
        "chawathe": similarity_ranking(country, top_k, algorithm="chawathe", coerce_root_label=coerce_root_label),
        "nj": similarity_ranking(country, top_k, algorithm="nj", coerce_root_label=coerce_root_label),
    }


# --- VSM / TF-IDF retrieval ---


def _validate_vsm_metric(metric: str) -> str:
    normalized = (metric or "cosine").strip().lower()
    if normalized not in SUPPORTED_VSM_METRICS:
        raise ValueError(f"Unsupported VSM metric '{metric}'.")
    return normalized


def _validate_vsm_mode(mode: str) -> str:
    normalized = (mode or "field").strip().lower()
    if normalized not in SUPPORTED_VSM_MODES:
        raise ValueError(f"Unsupported VSM mode '{mode}'.")
    return normalized


def _validate_cluster_algorithm(algorithm: str) -> str:
    normalized = (algorithm or "kmeans").strip().lower()
    if normalized not in SUPPORTED_CLUSTER_ALGORITHMS:
        raise ValueError(f"Unsupported clustering algorithm '{algorithm}'.")
    return normalized


def _validate_cluster_distance(distance: str) -> str:
    normalized = (distance or "euclidean").strip().lower()
    if normalized not in SUPPORTED_CLUSTER_DISTANCES:
        raise ValueError(f"Unsupported clustering distance '{distance}'.")
    return normalized


def _validate_linkage(linkage: str) -> str:
    normalized = (linkage or "average").strip().lower()
    if normalized not in SUPPORTED_AGGLOMERATIVE_LINKAGES:
        raise ValueError(f"Unsupported agglomerative linkage '{linkage}'.")
    return normalized


def _validate_cluster_source(source: str) -> str:
    normalized = (source or "vsm").strip().lower()
    if normalized not in SUPPORTED_CLUSTER_SOURCES:
        raise ValueError(f"Unsupported clustering vector source '{source}'.")
    return normalized


def _validate_vsm_document_source(source: str) -> str:
    normalized = (source or "all").strip().lower()
    if normalized not in SUPPORTED_VSM_DOCUMENT_SOURCES:
        raise ValueError(f"Unsupported VSM document source '{source}'.")
    return normalized


def _validate_ted_cluster_algorithm(algorithm: str) -> str:
    normalized = (algorithm or "chawathe").strip().lower()
    if normalized not in SUPPORTED_TED_CLUSTER_ALGORITHMS:
        raise ValueError(f"Unsupported TED clustering algorithm '{algorithm}'.")
    return normalized


def _vsm_index_name(mode: str) -> str:
    return f"tfidf:{mode}:all"


def _build_vsm_index(
    *,
    mode: str = "field",
    features: Optional[List[str]] = None,
    source: str = "all",
    max_df_ratio: Optional[float] = None,
) -> VSMIndex:
    mode = _validate_vsm_mode(mode)
    source = _validate_vsm_document_source(source)
    docs = read_all_json_documents()
    vsm_documents = [
        vsm_document_from_json(slug, doc, mode=mode, features=features, source=source)
        for slug, doc in sorted(docs.items())
    ]
    vsm_documents = [doc for doc in vsm_documents if doc.terms]
    return build_vsm_index(
        vsm_documents,
        mode=mode,
        max_df_ratio=max_df_ratio,
        metadata={
            "features": list(features or []),
            "document_source": source,
        },
    )


def _build_vsm_clustering_index(
    *,
    mode: str = "field",
    features: Optional[List[str]] = None,
    source: str = "comparison_fields",
    max_df_ratio: Optional[float] = 0.85,
) -> VSMIndex:
    """Build a non-persisted VSM index tuned for clustering, not keyword retrieval."""
    return _build_vsm_index(
        mode=mode,
        features=features,
        source=source,
        max_df_ratio=max_df_ratio,
    )


def _load_or_build_vsm_index(
    *,
    mode: str = "field",
    features: Optional[List[str]] = None,
) -> VSMIndex:
    mode = _validate_vsm_mode(mode)
    if features:
        return _build_vsm_index(mode=mode, features=features)

    stored = read_vsm_index(_vsm_index_name(mode))
    if stored is not None:
        return VSMIndex.from_dict(stored)
    return _build_vsm_index(mode=mode)


def _build_ted_similarity_index(
    *,
    ted_algorithm: str = "chawathe",
    features: Optional[List[str]] = None,
    cost_model: Optional[str] = None,
    coerce_root_label: Optional[str] = "infobox",
) -> VSMIndex:
    """Represent each country as its TED similarity profile against all countries."""
    ted_algorithm = _validate_ted_cluster_algorithm(ted_algorithm)
    docs = read_all_json_documents()
    tree_docs: Dict[str, Dict[str, Any]] = {}
    documents: Dict[str, str] = {}
    for slug, doc in sorted(docs.items()):
        tree = doc.get("tree")
        if tree is None:
            continue
        if features:
            tree = extract_selected_features(tree, features)
        tree_docs[slug] = tree
        meta = doc.get("meta") or {}
        documents[slug] = str(meta.get("country_name") or slug.replace("_", " ").title())

    index = build_ted_similarity_index(
        tree_docs,
        documents=documents,
        ted_algorithm=ted_algorithm,
        cost_model=cost_model,
        coerce_root_label=coerce_root_label,
    )
    index.metadata["features"] = list(features or [])
    return index


def run_vsm_preprocess(
    *,
    mode: str = "field",
    persist: bool = True,
) -> Dict[str, Any]:
    """Build a TF-IDF VSM index for all stored country documents."""
    mode = _validate_vsm_mode(mode)
    index = _build_vsm_index(mode=mode)
    if persist:
        write_vsm_index(_vsm_index_name(mode), index.to_dict())
    return {
        "status": "ok",
        "mode": mode,
        "persisted": persist,
        "document_count": len(index.doc_vectors),
        "vocabulary_size": len(index.vocabulary),
        "index_name": _vsm_index_name(mode) if persist else None,
    }


def vsm_query(
    query: str,
    *,
    top_k: int = 5,
    metric: str = "cosine",
    features: Optional[List[str]] = None,
    mode: str = "field",
) -> Dict[str, Any]:
    metric = _validate_vsm_metric(metric)
    mode = _validate_vsm_mode(mode)
    index = _load_or_build_vsm_index(mode=mode, features=features)
    results = rank_query(index, query, top_k=top_k, metric=metric)
    return {
        "query": query,
        "metric": metric,
        "mode": mode,
        "top_k": top_k,
        "features": list(features or []),
        "document_count": len(index.doc_vectors),
        "vocabulary_size": len(index.vocabulary),
        "results": [result.to_dict() for result in results],
    }


def vsm_similarity(
    source_slug: str,
    target_slug: str,
    *,
    metric: str = "cosine",
    features: Optional[List[str]] = None,
    mode: str = "field",
) -> Dict[str, Any]:
    metric = _validate_vsm_metric(metric)
    mode = _validate_vsm_mode(mode)
    index = _load_or_build_vsm_index(mode=mode, features=features)
    if source_slug not in index.doc_vectors:
        raise ValueError(f"No VSM vector for country: {source_slug}")
    if target_slug not in index.doc_vectors:
        raise ValueError(f"No VSM vector for country: {target_slug}")
    score = sparse_similarity(
        index.doc_vectors[source_slug],
        index.doc_vectors[target_slug],
        metric=metric,
    )
    return {
        "source_slug": source_slug,
        "source_display_name": index.documents.get(source_slug, source_slug),
        "target_slug": target_slug,
        "target_display_name": index.documents.get(target_slug, target_slug),
        "metric": metric,
        "mode": mode,
        "features": list(features or []),
        "score": score,
        "document_count": len(index.doc_vectors),
        "vocabulary_size": len(index.vocabulary),
    }


def vsm_similarity_ranking(
    country: str,
    *,
    top_k: int = 5,
    metric: str = "cosine",
    features: Optional[List[str]] = None,
    mode: str = "field",
) -> Dict[str, Any]:
    metric = _validate_vsm_metric(metric)
    mode = _validate_vsm_mode(mode)
    index = _load_or_build_vsm_index(mode=mode, features=features)
    results = rank_document(index, country, top_k=top_k, metric=metric)
    return {
        "country": country,
        "display_name": index.documents.get(country, country),
        "metric": metric,
        "mode": mode,
        "top_k": top_k,
        "features": list(features or []),
        "document_count": len(index.doc_vectors),
        "vocabulary_size": len(index.vocabulary),
        "results": [result.to_dict() for result in results],
    }


def vsm_cluster(
    *,
    vector_source: str = "vsm",
    algorithm: str = "kmeans",
    distance: str = "cosine",
    mode: str = "field",
    features: Optional[List[str]] = None,
    country: Optional[str] = None,
    k: int = 5,
    eps: float = 1.0,
    min_pts: int = 3,
    linkage: str = "average",
    ted_algorithm: str = "chawathe",
    cost_model: Optional[str] = None,
    vsm_document_source: str = "comparison_fields",
    max_df_ratio: Optional[float] = 0.85,
) -> Dict[str, Any]:
    vector_source = _validate_cluster_source(vector_source)
    algorithm = _validate_cluster_algorithm(algorithm)
    distance = _validate_cluster_distance(distance)
    mode = _validate_vsm_mode(mode)
    linkage = _validate_linkage(linkage)
    if vector_source == "ted":
        index = _build_ted_similarity_index(
            ted_algorithm=ted_algorithm,
            features=features,
            cost_model=cost_model,
        )
    else:
        index = _build_vsm_clustering_index(
            mode=mode,
            features=features,
            source=vsm_document_source,
            max_df_ratio=max_df_ratio,
        )
    selected = (country or "").strip().lower() or None
    result = cluster_vsm_index(
        index,
        algorithm=algorithm,
        distance=distance,
        selected_country=selected,
        k=k,
        eps=eps,
        min_pts=min_pts,
        linkage=linkage,
    )
    out = result.to_dict()
    out["vector_source"] = vector_source
    out["features"] = list(features or [])
    out["ted_algorithm"] = (
        _validate_ted_cluster_algorithm(ted_algorithm) if vector_source == "ted" else None
    )
    out["cost_model"] = cost_model if vector_source == "ted" else None
    out["vsm_document_source"] = (
        _validate_vsm_document_source(vsm_document_source) if vector_source == "vsm" else None
    )
    out["max_df_ratio"] = max_df_ratio if vector_source == "vsm" else None
    return out


# --- Public-facing product helpers for the React UI ---


def _country_display_name(slug: str, doc: Dict[str, Any]) -> str:
    meta = doc.get("meta") or {}
    return str(meta.get("country_name") or slug.replace("_", " ").title())


def _normalized_fields(doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    normalized = doc.get("normalized") or {}
    fields = normalized.get("fields") or {}
    return fields if isinstance(fields, dict) else {}


def _field_text(fields: Dict[str, Dict[str, Any]], key_fragment: str) -> str:
    parts: List[str] = []
    fragment = key_fragment.casefold()
    for key, value in fields.items():
        if fragment in str(key).casefold() and isinstance(value, dict):
            parts.append(str(value.get("text") or ""))
    return " ".join(parts).casefold()


def _field_numbers(fields: Dict[str, Dict[str, Any]], key_fragment: str) -> List[float]:
    numbers: List[float] = []
    fragment = key_fragment.casefold()
    for key, value in fields.items():
        if fragment not in str(key).casefold() or not isinstance(value, dict):
            continue
        for number in value.get("numbers") or []:
            try:
                numbers.append(float(number))
            except (TypeError, ValueError):
                continue
    return numbers


def _classify_size(area_km2: Optional[float]) -> str:
    if area_km2 is None:
        return "unknown"
    if area_km2 < 100_000:
        return "small"
    if area_km2 < 1_000_000:
        return "medium"
    return "large"


def _classify_population(population: Optional[float]) -> str:
    if population is None:
        return "unknown"
    if population < 5_000_000:
        return "less"
    if population < 50_000_000:
        return "moderate"
    return "high"


def _score_preference(
    *,
    actual: str,
    preferred: str,
    reason: str,
    weight: float,
) -> Tuple[float, List[str]]:
    if not preferred or preferred == "any" or preferred == "no_preference":
        return weight * 0.5, []
    if actual == preferred:
        return weight, [reason]
    if actual == "unknown":
        return weight * 0.25, []
    return 0.0, []


def recommend_country_matches(
    preferences: Dict[str, Any],
    *,
    limit: int = 8,
) -> Dict[str, Any]:
    docs = read_all_json_documents()
    scored: List[Dict[str, Any]] = []
    include_surprise = bool(preferences.get("include_surprising_matches"))

    for slug, doc in sorted(docs.items()):
        fields = _normalized_fields(doc)
        all_text = " ".join(
            str(value.get("text") or "")
            for value in fields.values()
            if isinstance(value, dict)
        ).casefold()
        reasons: List[str] = []
        score = 0.05

        area_values = _field_numbers(fields, "area")
        area = max(area_values) if area_values else None
        size_score, size_reasons = _score_preference(
            actual=_classify_size(area),
            preferred=str(preferences.get("size") or "any"),
            reason="Matches your preferred country scale",
            weight=0.18,
        )
        score += size_score
        reasons.extend(size_reasons)

        population_values = _field_numbers(fields, "population")
        population = max(population_values) if population_values else None
        population_score, population_reasons = _score_preference(
            actual=_classify_population(population),
            preferred=str(preferences.get("population") or "any"),
            reason="Fits your population preference",
            weight=0.18,
        )
        score += population_score
        reasons.extend(population_reasons)

        geography = str(preferences.get("geography") or "any")
        if geography not in {"any", "no_preference"}:
            geo_match = (
                (geography == "island" and "island" in all_text)
                or (geography == "coastal" and any(word in all_text for word in ["sea", "ocean", "coast", "gulf"]))
                or (geography == "landlocked" and "landlocked" in all_text)
            )
            if geo_match:
                score += 0.16
                reasons.append("Matches your geographic preference")

        government = str(preferences.get("government") or "any")
        government_text = _field_text(fields, "government")
        if government not in {"any", "no_preference"}:
            if government in government_text:
                score += 0.16
                reasons.append("Has a compatible government profile")

        language_profile = str(preferences.get("language_profile") or "any")
        language_text = _field_text(fields, "language")
        if language_profile == "multilingual" and ("," in language_text or " and " in language_text):
            score += 0.14
            reasons.append("Has a multilingual national profile")
        elif language_profile == "single" and language_text and "," not in language_text:
            score += 0.14
            reasons.append("Fits a single-language profile")
        elif language_profile in {"any", "no_preference"}:
            score += 0.07

        region = str(preferences.get("region") or "any").casefold()
        if region not in {"any", "no_preference", ""} and region in all_text:
            score += 0.08
            reasons.append("Connects with your selected regional flavor")

        if include_surprise:
            score += (hash(slug) % 10) / 200.0
        if not reasons:
            reasons.append("Shares several country-profile characteristics")

        scored.append({
            "country": slug,
            "display_name": _country_display_name(slug, doc),
            "score": min(score, 1.0),
            "reasons": reasons[:3],
        })

    scored.sort(key=lambda item: item["score"], reverse=True)
    recommendations = scored[:max(1, min(limit, 20))]
    top_match = recommendations[0] if recommendations else None
    return {
        "top_match": top_match,
        "recommendations": recommendations,
    }


def generate_guess_round(*, clue_count: int = 3, option_count: int = 4) -> Dict[str, Any]:
    countries = get_country_index()
    if len(countries) < option_count:
        raise ValueError("Not enough countries to generate a game round.")

    rng = random.SystemRandom()
    hidden_slug, hidden_name = rng.choice(countries)
    clues: List[Dict[str, str]] = []
    try:
        ranking = vsm_similarity_ranking(hidden_slug, top_k=max(clue_count, 3))
        for item in ranking.get("results", [])[:clue_count]:
            clues.append({
                "country": item["country"],
                "display_name": item.get("display_name") or item["country"],
            })
    except Exception as exc:
        logger.debug("Falling back to random game clues: %s", exc)

    if len(clues) < clue_count:
        pool = [(slug, name) for slug, name in countries if slug != hidden_slug]
        rng.shuffle(pool)
        for slug, name in pool:
            if len(clues) >= clue_count:
                break
            if slug not in {clue["country"] for clue in clues}:
                clues.append({"country": slug, "display_name": name})

    options = [{"country": hidden_slug, "display_name": hidden_name}]
    distractors = [(slug, name) for slug, name in countries if slug != hidden_slug]
    rng.shuffle(distractors)
    for slug, name in distractors:
        if len(options) >= option_count:
            break
        options.append({"country": slug, "display_name": name})
    rng.shuffle(options)

    round_id = uuid.uuid4().hex
    _GAME_ROUNDS[round_id] = hidden_slug
    return {
        "round_id": round_id,
        "clues": clues,
        "options": options,
    }


def submit_guess_answer(round_id: str, selected_country: str) -> Dict[str, Any]:
    correct_slug = _GAME_ROUNDS.pop(round_id, None)
    if correct_slug is None:
        raise ValueError("Unknown or expired round.")
    country_names = dict(get_country_index())
    correct = selected_country == correct_slug
    return {
        "correct": correct,
        "correct_answer": correct_slug,
        "correct_display_name": country_names.get(correct_slug, correct_slug),
        "explanation": (
            "Correct. The clue countries were selected from nearby country profiles."
            if correct
            else "Not quite. The clue countries were nearest neighbors of the hidden country."
        ),
    }
