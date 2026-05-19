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
    generate_guess_round,
    get_available_features,
    get_country_index,
    get_json_document,
    get_raw_html,
    get_tree_document,
    postprocess_tree,
    recommend_country_matches,
    run_build_trees,
    run_collect_pipeline,
    run_ted_preprocess,
    run_vsm_preprocess,
    similarity_ranking_both,
    submit_guess_answer,
    ted_compute_from_trees,
    ted_diff,
    ted_diff_from_trees,
    ted_patch,
    ted_similarity,
    vsm_query,
    vsm_cluster,
    vsm_similarity,
    vsm_similarity_ranking,
)

router = APIRouter()


@router.get("/countries", response_model=List[Dict[str, str]])
def list_countries() -> List[Dict[str, str]]:
    """
    List all countries: slug and display name.
    Used by the streamlit to build the country selector and search.
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
    Body: { "country_a": str, "country_b": str, "features": Optional[List[str]], "exclude": bool, "algorithm": "chawathe"|"nj"|"zhang_shasha", "coerce_root_label": "optional", "cost_model": "value_aware"|"classic" }.
    If exclude=True, features are excluded from comparison; otherwise they are included.
    If features is omitted or empty, performs full tree comparison.

    Response includes:
    - raw_edit_script_summary: counts from the native TED script (edit_script_length, inserts,
      deletes, updates; mappings separate for Zhang–Shasha).
    - display_edit_script_summary: counts after display normalization/filtering.
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
            cost_model=body.get("cost_model"),
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
    Body: { "source_tree": {...}, "target_tree": {...}, "algorithm": "chawathe"|"nj"|"zhang_shasha", "coerce_root_label": "optional", "cost_model": "value_aware"|"classic" }.

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
            cost_model=body.get("cost_model"),
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


def _top_k_from_body(body: Dict[str, Any], default: int = 5) -> int:
    top_k = body.get("top_k", default)
    if not isinstance(top_k, int) or top_k < 1 or top_k > 50:
        return default
    return top_k


def _features_from_body(body: Dict[str, Any]) -> Optional[List[str]]:
    features = body.get("features")
    if not features:
        return None
    if not isinstance(features, list):
        raise HTTPException(status_code=400, detail="'features' must be a list of strings")
    return [str(feature) for feature in features if str(feature).strip()]


def _int_range_from_body(
    body: Dict[str, Any],
    key: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = body.get(key, default)
    if not isinstance(value, int):
        return default
    return max(minimum, min(value, maximum))


def _positive_float_from_body(body: Dict[str, Any], key: str, *, default: float) -> float:
    value = body.get(key, default)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed <= 0:
        return default
    return parsed


def _optional_positive_float_from_body(body: Dict[str, Any], key: str) -> Optional[float]:
    if key not in body or body.get(key) is None:
        return None
    try:
        parsed = float(body[key])
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _optional_unit_float_from_body(body: Dict[str, Any], key: str) -> Optional[float]:
    if key not in body or body.get(key) is None:
        return None
    try:
        parsed = float(body[key])
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, parsed))


@router.post("/ted/preprocess", response_model=Dict[str, Any])
def post_ted_preprocess(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build/persist TED similarity-profile indexes for clustering.
    Body: { "algorithms": ["chawathe", "nj"], "persist": bool, "cost_model": optional }.
    """
    try:
        algorithms = body.get("algorithms")
        if algorithms is not None and not isinstance(algorithms, list):
            raise HTTPException(status_code=400, detail="'algorithms' must be a list")
        return run_ted_preprocess(
            algorithms=algorithms,
            persist=bool(body.get("persist", True)),
            cost_model=body.get("cost_model"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/vsm/preprocess", response_model=Dict[str, Any])
def post_vsm_preprocess(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build/persist a TF-IDF VSM index.
    Body: { "persist": bool }. VSM indexing uses term-context units.
    """
    try:
        return run_vsm_preprocess(
            persist=bool(body.get("persist", True)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/vsm/query", response_model=Dict[str, Any])
def post_vsm_query(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rank countries against a free-text query using TF-IDF VSM.
    Body: { "query": str, "top_k": int, "metric": "cosine"|"pcc", "features": optional }.
    """
    try:
        query = str(body.get("query") or "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="Missing or empty 'query'")
        return vsm_query(
            query,
            top_k=_top_k_from_body(body),
            metric=body.get("metric", "cosine"),
            features=_features_from_body(body),
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/vsm/similarity", response_model=Dict[str, Any])
def post_vsm_similarity(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two countries using TF-IDF VSM.
    Body: { "source_slug": str, "target_slug": str, "metric": "cosine"|"pcc", "features": optional }.
    """
    try:
        source_slug = str(body.get("source_slug") or "").strip().lower()
        target_slug = str(body.get("target_slug") or "").strip().lower()
        if not source_slug or not target_slug:
            raise HTTPException(status_code=400, detail="Missing source_slug or target_slug")
        return vsm_similarity(
            source_slug,
            target_slug,
            metric=body.get("metric", "cosine"),
            features=_features_from_body(body),
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/vsm/similarity-ranking", response_model=Dict[str, Any])
def post_vsm_similarity_ranking(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return top-k countries most similar to one country using TF-IDF VSM.
    Body: { "country": str, "top_k": int, "metric": "cosine"|"pcc",
            "features": optional, "semantic_only": optional bool }.
    """
    try:
        country = str(body.get("country") or "").strip().lower()
        if not country:
            raise HTTPException(status_code=400, detail="Missing or empty 'country'")
        return vsm_similarity_ranking(
            country,
            top_k=_top_k_from_body(body),
            metric=body.get("metric", "cosine"),
            features=_features_from_body(body),
            semantic_only=bool(body.get("semantic_only", False)),
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/vsm/clustering", response_model=Dict[str, Any])
def post_vsm_clustering(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cluster all countries using VSM vectors.
    Body: { "vector_source": "vsm"|"ted", "algorithm": "kmeans"|"agglomerative",
            "distance": "cosine"|"euclidean"|"manhattan",
            "ted_algorithm": "chawathe"|"nj"|"zhang_shasha", "country": optional, "features": optional,
            "k": int, "linkage": "single"|"complete"|"average",
            "stopping_rule": "none"|"cluster_count"|"similarity_threshold",
            "similarity_threshold": optional float }.
    """
    try:
        return vsm_cluster(
            vector_source=body.get("vector_source", "vsm"),
            algorithm=body.get("algorithm", "kmeans"),
            distance=body.get("distance", "cosine"),
            features=_features_from_body(body),
            country=str(body.get("country") or "").strip().lower() or None,
            k=_int_range_from_body(body, "k", default=5, minimum=2, maximum=50),
            ted_algorithm=body.get("ted_algorithm", "chawathe"),
            cost_model=body.get("cost_model"),
            linkage=body.get("linkage", "average"),
            stopping_rule=body.get("stopping_rule", "none"),
            similarity_threshold=_optional_unit_float_from_body(body, "similarity_threshold"),
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/matchmaker/recommend", response_model=Dict[str, Any])
def post_matchmaker_recommend(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Public-facing recommendation layer for the React matchmaker.
    Body: preference answers plus optional limit.
    """
    try:
        limit = _int_range_from_body(body, "limit", default=8, minimum=1, maximum=20)
        return recommend_country_matches(body, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/game/guess-round", response_model=Dict[str, Any])
def get_guess_round(clue_count: int = 3, option_count: int = 4) -> Dict[str, Any]:
    """Generate a country guessing game round without exposing the answer."""
    try:
        clue_count = max(1, min(clue_count, 5))
        option_count = max(2, min(option_count, 8))
        return generate_guess_round(clue_count=clue_count, option_count=option_count)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/game/submit-answer", response_model=Dict[str, Any])
def post_guess_answer(body: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a submitted answer for a generated guessing round."""
    try:
        round_id = str(body.get("round_id") or "").strip()
        selected_country = str(body.get("selected_country") or "").strip().lower()
        if not round_id or not selected_country:
            raise HTTPException(status_code=400, detail="Missing round_id or selected_country")
        return submit_guess_answer(round_id, selected_country)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
