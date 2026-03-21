"""
Unified comparison pipeline: TED, edit script, patching, post-processing.
Higher-level orchestration; uses core.similarity, core.patch, core.postprocess, core.data.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from domain.models.tree import TreeNode

from core.data.storage import read_tree_document
from core.patch.patch import apply_patch, trees_equal
from core.postprocess.postprocess import (
    render_comparison_report,
    summarize_edit_script,
    tree_to_infobox_text,
    tree_to_json_string,
    tree_to_xml_string,
)
from core.similarity.common import clone_tree
from core.similarity.ted import (
    ALGORITHM_CHAWATHE,
    ALGORITHM_NJ,
    ALGORITHM_ZHANG_SHASHA,
    compute_ted,
)
from core.similarity.zhang_shasha import normalize_tree as zs_normalize_tree
from core.similarity.tree_validation import validate_tree


class ComparisonPipelineError(ValueError):
    pass


def load_tree_for_slug(slug: str) -> TreeNode:
    tree_doc = read_tree_document(slug)
    if tree_doc is None:
        raise ComparisonPipelineError(
            f"No tree document was found for slug '{slug}'. Run your tree-building pipeline first."
        )
    root = TreeNode.from_dict(tree_doc)
    validate_tree(root)
    return root


def compare_country_slugs(
    source_slug: str,
    target_slug: str,
    *,
    algorithm: str = ALGORITHM_CHAWATHE,
    coerce_root_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare two country trees by slug; returns distance, similarity, edit script, patched tree, and reports."""
    source_root = load_tree_for_slug(source_slug)
    target_root = load_tree_for_slug(target_slug)

    patch_source = clone_tree(source_root)
    patch_target = clone_tree(target_root)
    if coerce_root_label is not None:
        patch_source.label = coerce_root_label
        patch_target.label = coerce_root_label

    al = (algorithm or "").lower()
    if al == ALGORITHM_ZHANG_SHASHA:
        zs_normalize_tree(patch_source)
        zs_normalize_tree(patch_target)
        ted_result = compute_ted(
            patch_source,
            patch_target,
            algorithm=algorithm,
            coerce_root_label=None,
        )
    else:
        ted_result = compute_ted(
            source_root,
            target_root,
            algorithm=algorithm,
            coerce_root_label=coerce_root_label,
        )

    if al == ALGORITHM_ZHANG_SHASHA:
        patched_root = apply_patch(
            patch_source,
            ted_result,
            algorithm=algorithm,
            target_root=patch_target,
        )
        patch_matches_target = trees_equal(patched_root, patch_target, algorithm=ALGORITHM_NJ)
        report = render_comparison_report(source_slug, target_slug, ted_result, patched_root)
        report_text = report + "\n\n(Patched via Zhang–Shasha postorder alignment.)\n"
        return {
            "source_slug": source_slug,
            "target_slug": target_slug,
            "algorithm": algorithm,
            "distance": ted_result.distance,
            "similarity": ted_result.similarity,
            "edit_script": ted_result.to_dict(),
            "edit_script_summary": summarize_edit_script(ted_result),
            "patch_matches_target": patch_matches_target,
            "patched_tree": patched_root.to_dict(),
            "patched_tree_json": tree_to_json_string(patched_root),
            "patched_tree_xml": tree_to_xml_string(patched_root),
            "patched_infobox_text": tree_to_infobox_text(patched_root),
            "report_text": report_text,
        }

    patched_root = apply_patch(patch_source, ted_result, algorithm=algorithm)
    patch_matches_target = trees_equal(patched_root, patch_target, algorithm=algorithm)

    return {
        "source_slug": source_slug,
        "target_slug": target_slug,
        "algorithm": algorithm,
        "distance": ted_result.distance,
        "similarity": ted_result.similarity,
        "edit_script": ted_result.to_dict(),
        "edit_script_summary": summarize_edit_script(ted_result),
        "patch_matches_target": patch_matches_target,
        "patched_tree": patched_root.to_dict(),
        "patched_tree_json": tree_to_json_string(patched_root),
        "patched_tree_xml": tree_to_xml_string(patched_root),
        "patched_infobox_text": tree_to_infobox_text(patched_root),
        "report_text": render_comparison_report(
            source_slug, target_slug, ted_result, patched_root
        ),
    }


def compare_from_tree_dicts(
    source_tree: Dict[str, Any],
    target_tree: Dict[str, Any],
    *,
    source_slug: str = "source",
    target_slug: str = "target",
    algorithm: str = ALGORITHM_CHAWATHE,
    coerce_root_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare two trees given as dicts (e.g. from API); same shape as compare_country_slugs."""
    source_root = TreeNode.from_dict(source_tree)
    target_root = TreeNode.from_dict(target_tree)
    validate_tree(source_root)
    validate_tree(target_root)

    patch_source = clone_tree(source_root)
    patch_target = clone_tree(target_root)
    if coerce_root_label is not None:
        patch_source.label = coerce_root_label
        patch_target.label = coerce_root_label

    al = (algorithm or "").lower()
    if al == ALGORITHM_ZHANG_SHASHA:
        zs_normalize_tree(patch_source)
        zs_normalize_tree(patch_target)
        ted_result = compute_ted(
            patch_source,
            patch_target,
            algorithm=algorithm,
            coerce_root_label=None,
        )
    else:
        ted_result = compute_ted(
            source_root,
            target_root,
            algorithm=algorithm,
            coerce_root_label=coerce_root_label,
        )

    if al == ALGORITHM_ZHANG_SHASHA:
        patched_root = apply_patch(
            patch_source,
            ted_result,
            algorithm=algorithm,
            target_root=patch_target,
        )
        patch_matches_target = trees_equal(patched_root, patch_target, algorithm=ALGORITHM_NJ)
        report = render_comparison_report(source_slug, target_slug, ted_result, patched_root)
        report_text = report + "\n\n(Patched via Zhang–Shasha postorder alignment.)\n"
        return {
            "source_slug": source_slug,
            "target_slug": target_slug,
            "algorithm": algorithm,
            "distance": ted_result.distance,
            "similarity": ted_result.similarity,
            "edit_script": ted_result.to_dict(),
            "edit_script_summary": summarize_edit_script(ted_result),
            "patch_matches_target": patch_matches_target,
            "patched_tree": patched_root.to_dict(),
            "patched_tree_json": tree_to_json_string(patched_root),
            "patched_tree_xml": tree_to_xml_string(patched_root),
            "patched_infobox_text": tree_to_infobox_text(patched_root),
            "report_text": report_text,
        }

    patched_root = apply_patch(patch_source, ted_result, algorithm=algorithm)

    return {
        "source_slug": source_slug,
        "target_slug": target_slug,
        "algorithm": algorithm,
        "distance": ted_result.distance,
        "similarity": ted_result.similarity,
        "edit_script": ted_result.to_dict(),
        "edit_script_summary": summarize_edit_script(ted_result),
        "patch_matches_target": trees_equal(patched_root, patch_target, algorithm=algorithm),
        "patched_tree": patched_root.to_dict(),
        "patched_tree_json": tree_to_json_string(patched_root),
        "patched_tree_xml": tree_to_xml_string(patched_root),
        "patched_infobox_text": tree_to_infobox_text(patched_root),
        "report_text": render_comparison_report(
            source_slug, target_slug, ted_result, patched_root
        ),
    }
