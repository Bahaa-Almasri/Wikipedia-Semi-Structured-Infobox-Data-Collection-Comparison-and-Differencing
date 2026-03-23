"""
Wikipedia infobox API: countries list, JSON document, tree, raw HTML, download, and run (pipeline / build trees).
Prefix: /wikiinfobox
All data is provided by the wikiinfobox service (MongoDB only).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from application.services.wikiinfobox_service import (
    compare_countries,
    get_available_features,
    get_country_index,
    get_json_document,
    get_raw_html,
    get_tree_document,
    postprocess_tree,
    run_build_trees,
    run_collect_pipeline,
    similarity_ranking_both,
    ted_compute_from_trees,
    ted_diff,
    ted_diff_from_trees,
    ted_patch,
    ted_similarity,
)

router = APIRouter()


@router.get("/countries", response_model=List[Dict[str, str]])
def list_countries() -> List[Dict[str, str]]:
    """
    List all countries: slug and display name.
    Used by the frontend to build the country selector and search.
    """
    try:
        index = get_country_index()
        return [{"slug": slug, "display_name": name} for slug, name in index]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/features", response_model=List[str])
def list_features() -> List[str]:
    """
    Return all available feature paths (dot notation) from tree schema.
    Dynamically generated from country trees. Used for feature selection in comparison.
    """
    try:
        return get_available_features()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/countries/{slug}/json", response_model=Dict[str, Any])
def get_country_json(slug: str) -> Dict[str, Any]:
    """Return the full JSON document for a country (meta, raw, cleaned, normalized, tree if present)."""
    try:
        doc = get_json_document(slug)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"No document for slug: {slug}")
        return doc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/countries/{slug}/tree", response_model=Dict[str, Any])
def get_country_tree(slug: str) -> Dict[str, Any]:
    """Return the tree representation for a country."""
    try:
        tree = get_tree_document(slug)
        if tree is None:
            raise HTTPException(status_code=404, detail=f"No tree for slug: {slug}")
        return tree
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/countries/{slug}/html")
def get_country_html(slug: str) -> str:
    """Return the raw infobox HTML for a country. 404 if not found."""
    try:
        html = get_raw_html(slug)
        if html is None:
            raise HTTPException(status_code=404, detail=f"No HTML for slug: {slug}")
        return html
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/countries/{slug}/json/download")
def download_country_json(slug: str) -> Response:
    """
    Return the full JSON document as a downloadable file (on user request only).
    Response has Content-Disposition: attachment so the browser offers to save as {slug}.json.
    """
    try:
        doc = get_json_document(slug)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"No document for slug: {slug}")
        content = json.dumps(doc, ensure_ascii=False, indent=2)
        return Response(
            content=content,
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{slug}.json"',
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# --- Run ---


@router.post("/data/collect", response_model=Dict[str, Any])
def run_collect() -> Dict[str, Any]:
    """
    Run the collection pipeline: fetch UN member states, scrape infoboxes, store in MongoDB.
    Long-running; returns when complete with count and list of slugs written.
    """
    try:
        slugs = run_collect_pipeline()
        return {"status": "ok", "collected": len(slugs), "slugs": slugs}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/data/preprocess", response_model=Dict[str, Any])
def run_build_trees_endpoint(slug: Optional[str] = None) -> Dict[str, Any]:
    """
    Build trees from JSON documents in MongoDB and write them back.
    Optional query param slug: if set, build only for that country; otherwise build for all.
    """
    try:
        slugs = run_build_trees(slug=slug)
        return {"status": "ok", "trees_built": len(slugs), "slugs": slugs}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# --- TED: similarity, diff, patch, postprocess ---


@router.get("/ted/similarity", response_model=Dict[str, Any])
def get_ted_similarity(
    source_slug: str,
    target_slug: str,
    algorithm: str = "chawathe",
    coerce_root_label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tree Edit Distance similarity between two country trees.
    algorithm: "chawathe" (LD-pair), "nj" (Nierman & Jagadish), or "zhang_shasha".
    """
    try:
        return ted_similarity(
            source_slug, target_slug,
            algorithm=algorithm,
            coerce_root_label=coerce_root_label,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/ted/diff", response_model=Dict[str, Any])
def get_ted_diff(
    source_slug: str,
    target_slug: str,
    algorithm: str = "chawathe",
    coerce_root_label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full comparison: distance, similarity, edit script, patched tree, report.
    algorithm: "chawathe", "nj", or "zhang_shasha" (Zhang–Shasha is distance + postorder mappings; patch uses mappings + target tree, not LD-pair replay).
    """
    try:
        return ted_diff(
            source_slug, target_slug,
            algorithm=algorithm,
            coerce_root_label=coerce_root_label,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/ted/diff/trees", response_model=Dict[str, Any])
def post_ted_diff_trees(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Full comparison from two tree dicts (source_tree, target_tree).
    Body: { "source_tree": {...}, "target_tree": {...}, "source_slug": "optional", "target_slug": "optional", "algorithm": "chawathe"|"nj"|"zhang_shasha", "coerce_root_label": "optional" }.
    """
    try:
        source_tree = body["source_tree"]
        target_tree = body["target_tree"]
        return ted_diff_from_trees(
            source_tree, target_tree,
            source_slug=body.get("source_slug", "source"),
            target_slug=body.get("target_slug", "target"),
            algorithm=body.get("algorithm", "chawathe"),
            coerce_root_label=body.get("coerce_root_label"),
        )
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Missing key: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/compare", response_model=Dict[str, Any])
def post_compare(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two countries by slug. Optionally restrict by features.
    Body: { "country_a": str, "country_b": str, "features": Optional[List[str]], "exclude": bool, "algorithm": "chawathe"|"nj"|"zhang_shasha", "coerce_root_label": "optional" }.
    If exclude=True, features are excluded from comparison; otherwise they are included.
    If features is omitted or empty, performs full tree comparison.

    Response includes:
    - raw_edit_script_summary: counts from the normalized TED script (edit_script_length, inserts,
      deletes, updates; mappings separate for Zhang–Shasha).
    - semantic_diff_summary: path-level semantic diff between trees (independent of TED algorithm).
    - edit_script_summary / edit_script_raw_summary: legacy aliases (semantic vs raw metrics).
    """
    try:
        country_a = body["country_a"]
        country_b = body["country_b"]
        features = body.get("features")
        if features is not None and len(features) == 0:
            features = None
        return compare_countries(
            country_a,
            country_b,
            features=features,
            exclude=body.get("exclude", False),
            algorithm=body.get("algorithm", "chawathe"),
            coerce_root_label=body.get("coerce_root_label"),
        )
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Missing key: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/ted/compute", response_model=Dict[str, Any])
def post_ted_compute(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute TED metrics + edit script ONLY (no patching).
    Body: { "source_tree": {...}, "target_tree": {...}, "algorithm": "chawathe"|"nj"|"zhang_shasha", "coerce_root_label": "optional" }.

    Returns raw_edit_script_summary and semantic_diff_summary (same shape as /compare TED fields).
    """
    try:
        source_tree = body["source_tree"]
        target_tree = body["target_tree"]
        return ted_compute_from_trees(
            source_tree,
            target_tree,
            algorithm=body.get("algorithm", "chawathe"),
            coerce_root_label=body.get("coerce_root_label"),
        )
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Missing key: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/ted/patch", response_model=Dict[str, Any])
def post_ted_patch(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply edit script or feature-driven patch to source tree.
    Body: { "source_tree": {...}, "edit_script": {...}, "algorithm": "chawathe"|"nj"|"zhang_shasha",
           "original_tree": optional, "edit_script_clean": optional,
           "target_tree": optional, "excluded_features": optional, "mappings": optional }.
    When original_tree + target_tree + excluded_features are provided (feature selection),
    uses feature-driven patch: SOURCE base, TARGET for values, only selected features.
    For algorithm "zhang_shasha", applies node-mapping patch (postorder alignments from TED);
    ``target_tree`` and ``mappings`` (list of {source_id, target_id}) are required unless
    mappings are embedded in ``edit_script``.
    """
    try:
        source_tree = body.get("source_tree")
        edit_script = body.get("edit_script")
        if source_tree is None:
            raise HTTPException(status_code=400, detail="Missing required key: source_tree")
        if edit_script is None:
            raise HTTPException(status_code=400, detail="Missing required key: edit_script")
        return ted_patch(
            source_tree,
            edit_script,
            algorithm=body.get("algorithm", "chawathe"),
            original_tree=body.get("original_tree"),
            edit_script_clean=body.get("edit_script_clean"),
            target_tree=body.get("target_tree"),
            excluded_features=body.get("excluded_features"),
            mappings=body.get("mappings"),
        )
    except HTTPException:
        raise
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Missing key: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Patch failed: {exc}") from exc


@router.post("/ted/postprocess", response_model=Dict[str, str])
def post_ted_postprocess(body: Dict[str, Any]) -> Dict[str, str]:
    """
    Post-process tree to JSON string, XML string, and infobox text.
    Body: { "tree": {...} } (tree as TreeNode.to_dict() shape).
    """
    try:
        return postprocess_tree(body["tree"])
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Missing key: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/similarity-ranking", response_model=Dict[str, List[Dict[str, Any]]])
def post_similarity_ranking(body: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Return top_k countries most similar to the given country using BOTH TED algorithms.
    Body: { "country": str, "top_k": int (default 5) }.
    Response: { "chawathe": [...], "nj": [...] }.
    """
    try:
        country = body.get("country", "").strip().lower()
        if not country:
            raise HTTPException(status_code=400, detail="Missing or empty 'country'")
        top_k = body.get("top_k", 5)
        if not isinstance(top_k, int) or top_k < 1 or top_k > 50:
            top_k = 5
        return similarity_ranking_both(country, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
