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
    get_country_index,
    get_json_document,
    get_raw_html,
    get_tree_document,
    ted_compute_from_trees,
    postprocess_tree,
    run_build_trees,
    run_collect_pipeline,
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
    algorithm: "chawathe" (LD-pair) or "nj" (Nierman & Jagadish).
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
    algorithm: "chawathe" or "nj".
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
    Body: { "source_tree": {...}, "target_tree": {...}, "source_slug": "optional", "target_slug": "optional", "algorithm": "chawathe"|"nj", "coerce_root_label": "optional" }.
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


@router.post("/ted/compute", response_model=Dict[str, Any])
def post_ted_compute(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute TED metrics + edit script ONLY (no patching).
    Body: { "source_tree": {...}, "target_tree": {...}, "algorithm": "chawathe"|"nj", "coerce_root_label": "optional" }.
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
    Apply edit script to source tree. Returns patched tree (dict, JSON, XML, infobox text).
    Body: { "source_tree": {...}, "edit_script": {...}, "algorithm": "chawathe"|"nj" }.
    """
    try:
        return ted_patch(
            body["source_tree"],
            body["edit_script"],
            algorithm=body.get("algorithm", "chawathe"),
        )
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Missing key: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
