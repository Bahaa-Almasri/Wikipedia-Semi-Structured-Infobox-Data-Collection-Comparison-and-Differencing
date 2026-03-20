"""
Service layer for Wikipedia infobox data. Uses core storage (MongoDB only).
All data access and run operations go through this module; the API controller calls only this service.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from core.data.storage import (
    list_slugs,
    read_json_document,
    read_raw_html,
    read_tree_document,
)
from core.patch.patch import apply_patch_from_dict
from core.postprocess.postprocess import tree_to_infobox_text, tree_to_json_string, tree_to_xml_string
from core.preprocess.tree_builder import build_and_save_tree_for_slug, build_and_save_trees_for_all
from core.preprocess.pipeline import collect_all_countries
from core.similarity.ted import compute_ted
from core.similarity.tree_validation import validate_tree
from domain.models.tree import TreeNode
from utils.compare import compare_country_slugs, compare_from_tree_dicts, load_tree_for_slug


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
) -> Dict[str, Any]:
    """Compute TED distance and similarity between two country trees."""
    source_root = load_tree_for_slug(source_slug)
    target_root = load_tree_for_slug(target_slug)
    result = compute_ted(
        source_root, target_root,
        algorithm=algorithm,
        coerce_root_label=coerce_root_label,
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
) -> Dict[str, Any]:
    """Compute full comparison (TED + edit script + patched tree + report)."""
    return compare_country_slugs(
        source_slug, target_slug,
        algorithm=algorithm,
        coerce_root_label=coerce_root_label,
    )


def ted_diff_from_trees(
    source_tree: Dict[str, Any],
    target_tree: Dict[str, Any],
    *,
    source_slug: str = "source",
    target_slug: str = "target",
    algorithm: str = "chawathe",
    coerce_root_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute full comparison from two tree dicts (e.g. from API)."""
    return compare_from_tree_dicts(
        source_tree, target_tree,
        source_slug=source_slug,
        target_slug=target_slug,
        algorithm=algorithm,
        coerce_root_label=coerce_root_label,
    )


def ted_compute_from_trees(
    source_tree: Dict[str, Any],
    target_tree: Dict[str, Any],
    *,
    algorithm: str = "chawathe",
    coerce_root_label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute TED metrics + edit script ONLY (no patch application).

    Returns edit_script as a list of serialized operations, so the frontend can store it directly
    and apply it later via /ted/patch.
    """
    source_root = TreeNode.from_dict(source_tree)
    target_root = TreeNode.from_dict(target_tree)
    validate_tree(source_root)
    validate_tree(target_root)

    ted_result = compute_ted(
        source_root,
        target_root,
        algorithm=algorithm,
        coerce_root_label=coerce_root_label,
    )

    # Both TedResult and NJTedResult expose `.operations` as typed operation objects.
    edit_script_ops = [op.to_dict() for op in ted_result.operations]

    return {
        "algorithm": ted_result.algorithm,
        "distance": ted_result.distance,
        "similarity": ted_result.similarity,
        "edit_script": edit_script_ops,
        "source_size": ted_result.source_size,
        "target_size": ted_result.target_size,
    }


def ted_patch(
    source_tree: Dict[str, Any],
    edit_script: Union[Dict[str, Any], List[Dict[str, Any]]],
    *,
    algorithm: str = "chawathe",
) -> Dict[str, Any]:
    """Apply edit script to source tree; returns patched tree as dict."""
    # Backward compatibility: older code passes the full TedResult.to_dict() (a dict with
    # "operations"). New compute endpoint may pass just the operations list.
    if isinstance(edit_script, list):
        edit_script_dict = {"operations": edit_script}
    else:
        edit_script_dict = edit_script

    patched_dict = apply_patch_from_dict(
        source_tree, edit_script_dict,
        algorithm=algorithm,
    )
    root = TreeNode.from_dict(patched_dict)
    return {
        "patched_tree": patched_dict,
        "patched_tree_json": tree_to_json_string(root),
        "patched_tree_xml": tree_to_xml_string(root),
        "patched_infobox_text": tree_to_infobox_text(root),
    }


def postprocess_tree(tree_dict: Dict[str, Any]) -> Dict[str, str]:
    """Convert tree dict to JSON string, XML string, and infobox text."""
    root = TreeNode.from_dict(tree_dict)
    return {
        "json": tree_to_json_string(root),
        "xml": tree_to_xml_string(root),
        "infobox_text": tree_to_infobox_text(root),
    }
