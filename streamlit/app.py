"""
WikiTreeDiff.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import xml.etree.ElementTree as ET

import plotly.express as px
import requests
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from domain.models.tree import format_draw_tree_dict  # noqa: E402

# Base URL of the Wikipedia Infobox API (no trailing slash)
API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
WIKIINFOBOX_PREFIX = f"{API_URL}/wikiinfobox"


def _get(path: str) -> requests.Response:
    """GET request to the API; returns the response for the caller to handle."""
    return requests.get(f"{WIKIINFOBOX_PREFIX}{path}", timeout=30)


def _post(path: str, *, timeout: int = 30) -> requests.Response:
    """POST request to the API. Use a large timeout for long-running endpoints (e.g. collect)."""
    return requests.post(f"{WIKIINFOBOX_PREFIX}{path}", timeout=timeout)


@st.cache_data(ttl=60)
def load_country_index() -> List[Tuple[str, str]]:
    """
    Fetch list of (slug, display_name) from the API.
    Returns empty list on error (caller should show an error message).
    Cached 60s to avoid repeated API calls.
    """
    try:
        r = _get("/countries")
        r.raise_for_status()
        data = r.json()
        return [(item["slug"], item["display_name"]) for item in data]
    except requests.RequestException:
        return []
    except (KeyError, TypeError):
        return []


@st.cache_data(ttl=120)
def load_features() -> List[str]:
    """
    Fetch list of available feature paths (dot notation) from the API.
    Returns empty list on error. Cached 120s.
    """
    try:
        r = _get("/features")
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []
    except requests.RequestException:
        return []
    except (KeyError, TypeError):
        return []


def compare_countries_api(
    country_a: str,
    country_b: str,
    *,
    features: Optional[List[str]] = None,
    exclude: bool = True,
    algorithm: str = "chawathe",
    coerce_root_label: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    POST /compare — compare two countries, optionally excluding selected features.
    Returns TED result with edit_script, distance, similarity, source_tree, target_tree.
    """
    try:
        payload: Dict[str, Any] = {
            "country_a": country_a,
            "country_b": country_b,
            "algorithm": algorithm,
        }
        if features:
            payload["features"] = features
            payload["exclude"] = exclude
        if coerce_root_label:
            payload["coerce_root_label"] = coerce_root_label
        r = requests.post(
            f"{WIKIINFOBOX_PREFIX}/compare",
            json=payload,
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def load_tree(slug: str) -> Optional[Dict[str, Any]]:
    """Fetch tree for a country from the API. Returns None on 404 or error."""
    try:
        r = _get(f"/countries/{slug}/tree")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def load_json(slug: str) -> Optional[Dict[str, Any]]:
    """Fetch JSON document for a country from the API. Returns None on 404 or error."""
    try:
        r = _get(f"/countries/{slug}/json")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def load_html(slug: str) -> Optional[str]:
    """Fetch raw HTML for a country from the API. Returns None on 404 or error."""
    try:
        r = _get(f"/countries/{slug}/html")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.text
    except requests.RequestException:
        return None


def api_available() -> bool:
    """Check if the API is reachable (health check)."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False


def run_collect() -> Optional[Dict[str, Any]]:
    """POST /data/collect — run collection pipeline. Long-running; use large timeout."""
    try:
        r = _post("/data/collect", timeout=3600)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def run_preprocess(slug: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """POST /data/preprocess — build trees. Optional slug for single country. Long-running for all."""
    try:
        path = "/data/preprocess" if slug is None else f"/data/preprocess?slug={slug}"
        r = _post(path, timeout=600)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


# --- TED: similarity, diff, patch, postprocess ---


def ted_similarity(
    source_slug: str,
    target_slug: str,
    algorithm: str = "chawathe",
    coerce_root_label: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """GET /ted/similarity — distance and similarity between two trees."""
    try:
        params: Dict[str, str] = {
            "source_slug": source_slug,
            "target_slug": target_slug,
            "algorithm": algorithm,
        }
        if coerce_root_label:
            params["coerce_root_label"] = coerce_root_label
        r = requests.get(
            f"{WIKIINFOBOX_PREFIX}/ted/similarity",
            params=params,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def ted_diff(
    source_slug: str,
    target_slug: str,
    algorithm: str = "chawathe",
    coerce_root_label: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """GET /ted/diff — full comparison: edit script, patched tree, report."""
    try:
        params = {
            "source_slug": source_slug,
            "target_slug": target_slug,
            "algorithm": algorithm,
        }
        if coerce_root_label:
            params["coerce_root_label"] = coerce_root_label
        r = requests.get(
            f"{WIKIINFOBOX_PREFIX}/ted/diff",
            params=params,
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def ted_compute(
    source_tree: Dict[str, Any],
    target_tree: Dict[str, Any],
    *,
    algorithm: str = "chawathe",
    coerce_root_label: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """POST /ted/compute — compute TED metrics + edit script ONLY."""
    try:
        payload: Dict[str, Any] = {
            "source_tree": source_tree,
            "target_tree": target_tree,
            "algorithm": algorithm,
        }
        if coerce_root_label:
            payload["coerce_root_label"] = coerce_root_label

        r = requests.post(
            f"{WIKIINFOBOX_PREFIX}/ted/compute",
            json=payload,
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def ted_patch(
    source_tree: Dict[str, Any],
    edit_script: Any,
    algorithm: str = "chawathe",
    *,
    original_tree: Optional[Dict[str, Any]] = None,
    edit_script_clean: Optional[List[Dict[str, Any]]] = None,
    target_tree: Optional[Dict[str, Any]] = None,
    excluded_features: Optional[List[str]] = None,
    mappings: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """POST /ted/patch — apply edit script or feature-driven patch. When original_tree + target_tree + excluded_features provided, uses feature-driven patch."""
    try:
        payload: Dict[str, Any] = {
            "source_tree": source_tree,
            "edit_script": edit_script,
            "algorithm": algorithm,
        }
        if mappings is not None and (algorithm or "").lower() == "zhang_shasha":
            payload["mappings"] = mappings
        if (
            edit_script_clean is not None
            and len(edit_script_clean) > 0
            and (algorithm or "").lower() == "zhang_shasha"
        ):
            payload["edit_script_clean"] = edit_script_clean
        if original_tree is not None and edit_script_clean:
            payload["original_tree"] = original_tree
            payload["edit_script_clean"] = edit_script_clean
        if target_tree is not None and excluded_features is not None:
            payload["original_tree"] = original_tree or source_tree
            payload["target_tree"] = target_tree
            payload["excluded_features"] = excluded_features
            payload.pop("edit_script_clean", None)
        elif target_tree is not None:
            # Pass target_tree for patch validation (tree_similarity) even when not feature-driven
            payload["target_tree"] = target_tree
        r = requests.post(
            f"{WIKIINFOBOX_PREFIX}/ted/patch",
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        if hasattr(e, "response") and e.response is not None:
            try:
                err_detail = e.response.json().get("detail", str(e))
            except Exception:
                err_detail = e.response.text or str(e)
            st.error(f"Patch failed: {err_detail}")
        return None


def ted_postprocess(tree: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """POST /ted/postprocess — tree to JSON/XML/infobox text."""
    try:
        r = requests.post(
            f"{WIKIINFOBOX_PREFIX}/ted/postprocess",
            json={"tree": tree},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def similarity_ranking_api(country: str, top_k: int = 5) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """
    POST /similarity-ranking — return top_k countries most similar by BOTH TED algorithms.
    Response: { "chawathe": [...], "nj": [...] }.
    """
    try:
        r = requests.post(
            f"{WIKIINFOBOX_PREFIX}/similarity-ranking",
            json={"country": country, "top_k": top_k},
            timeout=180,
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def run_vsm_preprocess(mode: str = "field", persist: bool = True) -> Optional[Dict[str, Any]]:
    """POST /vsm/preprocess — build a TF-IDF index for VSM retrieval."""
    try:
        r = requests.post(
            f"{WIKIINFOBOX_PREFIX}/vsm/preprocess",
            json={"mode": mode, "persist": persist},
            timeout=600,
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def vsm_query_api(
    query: str,
    *,
    top_k: int = 5,
    metric: str = "cosine",
    mode: str = "field",
    features: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """POST /vsm/query — rank countries against a free-text query."""
    try:
        payload: Dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "metric": metric,
            "mode": mode,
        }
        if features:
            payload["features"] = features
        r = requests.post(f"{WIKIINFOBOX_PREFIX}/vsm/query", json=payload, timeout=180)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def vsm_similarity_api(
    source_slug: str,
    target_slug: str,
    *,
    metric: str = "cosine",
    mode: str = "field",
    features: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """POST /vsm/similarity — compare two country vectors."""
    try:
        payload: Dict[str, Any] = {
            "source_slug": source_slug,
            "target_slug": target_slug,
            "metric": metric,
            "mode": mode,
        }
        if features:
            payload["features"] = features
        r = requests.post(f"{WIKIINFOBOX_PREFIX}/vsm/similarity", json=payload, timeout=180)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def vsm_similarity_ranking_api(
    country: str,
    *,
    top_k: int = 5,
    metric: str = "cosine",
    mode: str = "field",
    features: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """POST /vsm/similarity-ranking — rank country vectors by similarity."""
    try:
        payload: Dict[str, Any] = {
            "country": country,
            "top_k": top_k,
            "metric": metric,
            "mode": mode,
        }
        if features:
            payload["features"] = features
        r = requests.post(
            f"{WIKIINFOBOX_PREFIX}/vsm/similarity-ranking",
            json=payload,
            timeout=180,
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def vsm_clustering_api(
    *,
    vector_source: str = "vsm",
    algorithm: str = "kmeans",
    distance: str = "cosine",
    projection: str = "mds",
    mode: str = "field",
    country: Optional[str] = None,
    features: Optional[List[str]] = None,
    k: int = 5,
    eps: float = 1.0,
    min_pts: int = 3,
    linkage: str = "average",
    ted_algorithm: str = "chawathe",
    cost_model: Optional[str] = None,
    vsm_document_source: str = "comparison_fields",
    max_df_ratio: float = 0.85,
) -> Optional[Dict[str, Any]]:
    """POST /vsm/clustering — cluster country vectors and return Plotly-ready points."""
    try:
        payload: Dict[str, Any] = {
            "vector_source": vector_source,
            "algorithm": algorithm,
            "distance": distance,
            "projection": projection,
            "mode": mode,
            "k": k,
            "eps": eps,
            "min_pts": min_pts,
            "linkage": linkage,
            "ted_algorithm": ted_algorithm,
            "vsm_document_source": vsm_document_source,
            "max_df_ratio": max_df_ratio,
        }
        if cost_model:
            payload["cost_model"] = cost_model
        if country:
            payload["country"] = country
        if features:
            payload["features"] = features
        r = requests.post(f"{WIKIINFOBOX_PREFIX}/vsm/clustering", json=payload, timeout=300)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


# --- Semantic edit script (clean) display: structured diff + category summary ---


def path_to_label(path: List[str]) -> str:
    """Human-readable label from the last path segment (strips list suffixes)."""
    if not path:
        return "Root"
    leaf = path[-1]
    base = leaf.split("[", 1)[0]
    return base.replace("_", " ").title()


EMPTY_VALUE_DISPLAY = "∅"


def format_value(value: Any) -> str:
    if value is None:
        return EMPTY_VALUE_DISPLAY
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped == "None":
            return EMPTY_VALUE_DISPLAY
    if isinstance(value, dict):
        parts = [f"{k}: {format_value(v)}" for k, v in value.items()]
        return ", ".join(parts)
    if isinstance(value, (list, tuple)):
        return ", ".join(format_value(v) for v in value)
    return str(value)


def to_structured_diff(edit_script: List[Dict[str, Any]]) -> str:
    """Format clean (path-based) edit script as labeled UPDATE / INSERT / DELETE blocks."""
    lines: List[str] = []
    for op in edit_script:
        path = op.get("path") or []
        label = path_to_label(path)
        kind = op.get("op")

        if kind == "update":
            lines.append(f"🔄 UPDATE — {label}")
            lines.append(f"   - Old: {format_value(op.get('old_value'))}")
            lines.append(f"   + New: {format_value(op.get('new_value'))}")
            lines.append("")
        elif kind == "insert":
            lines.append(f"➕ INSERT — {label}")
            lines.append(f"   + Value: {format_value(op.get('value'))}")
            lines.append("")
        elif kind == "delete":
            lines.append(f"❌ DELETE — {label}")
            # API uses old_value for deletes; accept value for compatibility
            deleted = op.get("old_value")
            if deleted is None and "value" in op:
                deleted = op.get("value")
            lines.append(f"   - Value: {format_value(deleted)}")
            lines.append("")
        else:
            lines.append(f"❓ {kind} — {label}")
            lines.append(f"   {op}")
            lines.append("")

    return "\n".join(lines).rstrip()


def categorize(section: str) -> str:
    s = section.split("[", 1)[0].lower()
    if s == "economy":
        return "Economy"
    if s == "government":
        return "Government"
    if s == "population":
        return "Population"
    if s in ("area", "coordinates"):
        return "Geography"
    return "Others"


def build_summary(edit_script: List[Dict[str, Any]]) -> Tuple[int, Dict[str, int]]:
    """Count semantic diff ops per high-level category (Economy, Government, …)."""
    category_counts: Dict[str, int] = defaultdict(int)
    for op in edit_script:
        path = op.get("path") or []
        if len(path) > 1:
            section = path[1]
        else:
            section = "others"
        category_counts[categorize(section)] += 1
    total = len(edit_script)
    return total, dict(category_counts)


def summarize_raw_edit_script_ops_local(ops: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Same rules as backend summarize_raw_edit_script_operations (fallback for older APIs).
    """
    inserts = deletes = updates = mappings = 0
    for op in ops:
        kind = str(op.get("op") or "").lower()
        if kind in ("insert", "insert_tree"):
            inserts += 1
        elif kind in ("delete", "delete_tree"):
            deletes += 1
        elif kind == "update":
            updates += 1
        elif kind == "map":
            mappings += 1
    length = inserts + deletes + updates
    return {
        "edit_script_length": length,
        "inserts": inserts,
        "deletes": deletes,
        "updates": updates,
        "mappings": mappings,
        "operation_count_total": len(ops),
    }


def json_to_xml(data: Any, root_name: str = "root") -> str:
    """
    Convert JSON-like data (dict/list/scalar) to an XML string.
    Intended for visualization only in the streamlit (no backend calls).
    """

    def build_xml(element: ET.Element, value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                child = ET.SubElement(element, str(k))
                build_xml(child, v)
        elif isinstance(value, list):
            for item in value:
                child = ET.SubElement(element, "item")
                build_xml(child, item)
        else:
            # ElementTree uses element.text for leaf scalars.
            element.text = "" if value is None else str(value)

    root = ET.Element(root_name)
    build_xml(root, data)
    return ET.tostring(root, encoding="unicode")


def main() -> None:
    st.set_page_config(page_title="WikiTreeDiff", layout="wide")

    st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #f7f9fc;
}

/* Reduce harsh white containers */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    background-color: #f7f9fc;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #eef2f7;
}

/* Headings */
h1, h2, h3 {
    color: #1e3a5f;
    font-weight: 600;
}

/* Buttons */
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    border: none;
    transition: all 0.2s ease;
}

.stButton>button:hover {
    background-color: #1d4ed8;
    transform: translateY(-1px);
}

/* Metrics cards */
.stMetric {
    background-color: #ffffff;
    padding: 12px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}

/* Tabs */
button[role="tab"] {
    color: #6b7280;
    font-weight: 500;
}

button[role="tab"][aria-selected="true"] {
    color: #2563eb;
    border-bottom: 2px solid #2563eb;
}

/* Code blocks */
pre {
    background-color: #f1f5f9 !important;
    border-radius: 10px;
    padding: 10px;
    border: 1px solid #e2e8f0;
}

/* Text areas (XML viewer) */
textarea {
    background-color: #f1f5f9 !important;
    color: #1e293b !important;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
}

/* Expanders */
details {
    background-color: #ffffff;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
    padding: 6px;
}

/* Divider */
hr {
    border: 1px solid #e5e7eb;
}

/* Subtle card effect for sections */
section.main > div {
    border-radius: 12px;
}

</style>
    """, unsafe_allow_html=True)

    st.title("WikiTreeDiff")
    st.caption("JSON → Tree → TED Edit Script → Patch (demo-friendly, step-by-step).")

    # Sidebar: data pipeline
    with st.sidebar:
        st.header("Data pipeline")
        if st.button(
            "Run collect",
            help="Fetch UN member states, scrape infoboxes, store in MongoDB. Long-running.",
        ):
            with st.spinner("Running collect… (may take several minutes)"):
                result = run_collect()
            if result is not None:
                load_country_index.clear()
                st.success(
                    f"Collected {result.get('collected', 0)} countries. Refresh or re-run to see the list."
                )
                st.caption(
                    f"Slugs: {result.get('slugs', [])[:5]}{'…' if len(result.get('slugs', [])) > 5 else ''}"
                )
            else:
                st.error("Collect failed or timed out. Check API logs.")

        if st.button("Run preprocess (all)", help="Build trees for all countries in MongoDB."):
            with st.spinner("Running preprocess… (may take a minute)"):
                result = run_preprocess()
            if result is not None:
                st.success(f"Built trees for {result.get('trees_built', 0)} countries.")
                st.caption(
                    f"Slugs: {result.get('slugs', [])[:5]}{'…' if len(result.get('slugs', [])) > 5 else ''}"
                )
            else:
                st.error("Preprocess failed or timed out. Check API logs.")

        st.markdown("**VSM preprocessing**")
        vsm_index_mode = st.radio(
            "VSM index mode",
            options=["field", "flat"],
            index=0,
            horizontal=True,
            key="sidebar_vsm_mode",
            help="Field mode keeps semi-structured field context; flat mode concatenates all terms.",
        )
        if st.button("Build VSM index", help="Build and persist a TF-IDF vector index."):
            with st.spinner("Building VSM index…"):
                result = run_vsm_preprocess(mode=vsm_index_mode, persist=True)
            if result is not None:
                st.success(
                    f"Indexed {result.get('document_count', 0)} documents "
                    f"with {result.get('vocabulary_size', 0)} terms."
                )
            else:
                st.error("VSM index build failed or timed out. Check API logs.")

        st.divider()
        st.caption("TED and VSM operations run against the backend + MongoDB.")

    if not api_available():
        st.error(
            f"Cannot reach the API at {API_URL}. Start the API server (e.g. uvicorn src.app:app --host 0.0.0.0 --port 8000) "
            "and set API_URL if it runs elsewhere."
        )
        return

    index = load_country_index()
    if not index:
        st.error("No country data returned from the API. Run the pipeline to populate data.")
        return

    slug_for_name = {name: slug for slug, name in index}
    all_display_names = [name for _, name in index]

    main_tabs = st.tabs(["Comparison", "Country Similarity", "VSM Search", "Similarity Clustering"])

    # --- Tab 0: Comparison ---
    with main_tabs[0]:
        # --- Section 1 — Data Selection ---
        st.markdown("---")
        st.header("Step 1 — Data Selection")

        col_source, col_target = st.columns(2)
        with col_source:
            source_country = st.selectbox("Source Country", all_display_names, index=0, key="source_country")
        with col_target:
            target_country = st.selectbox(
                "Target Country",
                all_display_names,
                index=1 if len(all_display_names) > 1 else 0,
                key="target_country",
            )

        source_slug = slug_for_name.get(source_country)
        target_slug = slug_for_name.get(target_country)

        # Feature exclusion (optional — leave empty to compare everything)
        available_features = load_features()
        excluded_features = st.multiselect(
            "Select Features to Exclude",
            options=available_features,
            default=[],
            key="excluded_features",
            help="Select features to exclude from comparison (leave empty to compare everything). Supports nested paths like area.total_km2.",
            placeholder="Select features to exclude (leave empty to compare everything)",
        )

        if not source_slug or not target_slug:
            st.warning("Please select both countries.")
        else:
            # --- Section 2 — Data Views ---
            st.markdown("---")
            st.header("Step 2 — Data Views")

            source_json = load_json(source_slug)
            target_json = load_json(target_slug)
            source_tree = load_tree(source_slug)
            target_tree = load_tree(target_slug)
            source_html = load_html(source_slug)
            target_html = load_html(target_slug)

            tabs = st.tabs(["JSON", "XML", "Tree", "HTML Source", "HTML Preview"])

            with tabs[0]:
                st.subheader("Data Preview (JSON)")
                left, right = st.columns(2)
                with left:
                    st.markdown(f"**Source:** `{source_country}`")
                    if source_json is None:
                        st.warning("No JSON.")
                    else:
                        st.json(source_json)
                with right:
                    st.markdown(f"**Target:** `{target_country}`")
                    if target_json is None:
                        st.warning("No JSON.")
                    else:
                        st.json(target_json)

            with tabs[1]:
                st.subheader("Data Preview (XML)")
                left, right = st.columns(2)
                with left:
                    st.markdown(f"**Source:** `{source_country}`")
                    if source_tree is not None:
                        xml_output = json_to_xml(source_tree, root_name="tree")
                        st.text_area("XML Output", xml_output, height=700, disabled=True)
                    elif source_json is not None:
                        xml_output = json_to_xml(source_json, root_name="infobox_json")
                        st.text_area("XML Output", xml_output, height=700, disabled=True)
                    else:
                        st.info("No data available.")
                with right:
                    st.markdown(f"**Target:** `{target_country}`")
                    if target_tree is not None:
                        xml_output = json_to_xml(target_tree, root_name="tree")
                        st.text_area("XML Output", xml_output, height=700, disabled=True)
                    elif target_json is not None:
                        xml_output = json_to_xml(target_json, root_name="infobox_json")
                        st.text_area("XML Output", xml_output, height=700, disabled=True)
                    else:
                        st.info("No data available.")

            with tabs[2]:
                st.subheader("Tree Structure")
                left, right = st.columns(2)
                with left:
                    st.markdown(f"**Source:** `{source_country}`")
                    if source_tree is None:
                        st.warning("No tree found.")
                    else:
                        st.caption("Box-drawn tree (├── / └──); scroll in the code block for large trees.")
                        st.code(
                            format_draw_tree_dict(source_tree),
                            language="text",
                        )
                with right:
                    st.markdown(f"**Target:** `{target_country}`")
                    if target_tree is None:
                        st.warning("No tree found.")
                    else:
                        st.caption("Box-drawn tree (├── / └──); scroll in the code block for large trees.")
                        st.code(
                            format_draw_tree_dict(target_tree),
                            language="text",
                        )

            with tabs[3]:
                st.markdown("### HTML Source")
                left, right = st.columns(2)
                with left:
                    st.markdown(f"**Source:** `{source_country}`")
                    if source_html is None:
                        st.warning("No HTML.")
                    else:
                        st.code(source_html, language="html")
                with right:
                    st.markdown(f"**Target:** `{target_country}`")
                    if target_html is None:
                        st.warning("No HTML.")
                    else:
                        st.code(target_html, language="html")

            with tabs[4]:
                st.markdown("### HTML Preview")
                left, right = st.columns(2)
                with left:
                    st.markdown(f"**Source:** `{source_country}`")
                    if source_html is None:
                        st.warning("No HTML.")
                    else:
                        st.components.v1.html(source_html, height=600, scrolling=True)
                with right:
                    st.markdown(f"**Target:** `{target_country}`")
                    if target_html is None:
                        st.warning("No HTML.")
                    else:
                        st.components.v1.html(target_html, height=600, scrolling=True)

            # --- Section 3 — TED Comparison ---
            st.markdown("---")
            st.header("Step 3 — Tree Edit Distance Comparison")

            col_algo, col_coerce = st.columns(2)
            with col_algo:
                algorithm = st.radio(
                    "TED algorithm",
                    options=["chawathe", "nj", "zhang_shasha"],
                    index=0,
                    format_func=lambda x: (
                        "Chawathe (LD-pair)"
                        if x == "chawathe"
                        else "Nierman & Jagadish (NJ)"
                        if x == "nj"
                        else "Zhang–Shasha"
                    ),
                    key="ted_algorithm",
                )
            with col_coerce:
                coerce_root = st.checkbox(
                    "Coerce root label (compare content only)",
                    value=True,
                    key="ted_coerce_root",
                )
            coerce_root_label = "infobox" if coerce_root else None

            ui_key = (source_slug, target_slug, tuple(excluded_features), algorithm, coerce_root_label)
            if st.session_state.get("ted_ui_key") != ui_key:
                st.session_state["ted_ui_key"] = ui_key
                st.session_state.pop("ted_compute_result", None)
                st.session_state.pop("edit_script", None)
                st.session_state.pop("ted_source_tree_for_patch", None)
                st.session_state.pop("ted_original_tree_for_patch", None)
                st.session_state.pop("ted_target_tree_for_patch", None)
                st.session_state.pop("ted_edit_script_clean", None)
                st.session_state.pop("ted_target_tree_for_validation", None)
                st.session_state.pop("ted_patch_result", None)

            compare_disabled = source_tree is None or target_tree is None
            if st.button(
                "Compute Edit Script",
                key="btn_compute_edit_script",
                disabled=compare_disabled,
            ):
                if compare_disabled:
                    st.warning("Trees missing. Run preprocess first.")
                else:
                    with st.spinner("Computing TED edit script…"):
                        compute_result = compare_countries_api(
                            source_slug,
                            target_slug,
                            features=excluded_features if excluded_features else None,
                            exclude=True,
                            algorithm=algorithm,
                            coerce_root_label=coerce_root_label,
                        )

                    if compute_result is not None:
                        st.session_state["ted_compute_result"] = compute_result
                        st.session_state["edit_script"] = compute_result.get("edit_script", [])
                        # Semantic clean script (used by Zhang–Shasha patch path and viewers)
                        st.session_state["ted_edit_script_clean"] = compute_result.get("edit_script_clean") or []
                        # Use source_tree from compare result (pruned when features selected)
                        source_tree_for_patch = dict(compute_result.get("source_tree", source_tree or {}))
                        if coerce_root_label:
                            source_tree_for_patch["label"] = coerce_root_label
                        st.session_state["ted_source_tree_for_patch"] = source_tree_for_patch
                        # Target for patch validation (tree_similarity): always store
                        target_for_validation = compute_result.get("original_target_tree") or compute_result.get("target_tree")
                        if target_for_validation:
                            st.session_state["ted_target_tree_for_validation"] = dict(target_for_validation)
                        else:
                            st.session_state.pop("ted_target_tree_for_validation", None)
                        # When features used: store original trees for feature-driven patch
                        if excluded_features and compute_result.get("original_source_tree") is not None:
                            st.session_state["ted_original_tree_for_patch"] = dict(compute_result["original_source_tree"])
                            st.session_state["ted_target_tree_for_patch"] = dict(
                                compute_result.get("original_target_tree") or compute_result.get("target_tree", {})
                            )
                        else:
                            st.session_state.pop("ted_original_tree_for_patch", None)
                            st.session_state.pop("ted_target_tree_for_patch", None)
                        st.session_state.pop("ted_patch_result", None)
                        st.success("Edit script computed.")
                    else:
                        st.error("Compute edit script request failed. Check API.")

            compute_result = st.session_state.get("ted_compute_result")
            if isinstance(compute_result, dict):
                st.markdown("### Comparison Results")
                distance = compute_result.get("distance", "—")
                similarity = compute_result.get("similarity", 0.0)
                c1, c2 = st.columns(2)
                c1.metric("Distance", distance)
                c2.metric(
                    "Similarity",
                    f"{similarity:.4f}" if isinstance(similarity, (int, float)) else similarity,
                )

                st.markdown("### Edit Script Viewer")

                raw_script = compute_result.get("edit_script_raw")
                if raw_script is None:
                    raw_script = compute_result.get("edit_script") or []
                clean_script = compute_result.get("edit_script_clean") or []

                raw_metrics = compute_result.get("raw_edit_script_summary") or compute_result.get(
                    "edit_script_raw_summary"
                )
                if not raw_metrics and isinstance(raw_script, list):
                    raw_metrics = summarize_raw_edit_script_ops_local(raw_script)

                st.markdown("#### TED edit script (algorithm-native)")
                st.caption(
                    "These counts describe the script returned for the selected TED algorithm "
                    "(Chawathe / NJ / Zhang–Shasha). They can differ from semantic field-level "
                    "differences and from the distance/similarity score."
                )
                if raw_metrics:
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric(
                        "Edit script length",
                        raw_metrics.get("edit_script_length", 0),
                        help="Inserts + deletes + updates in this script (mappings excluded).",
                    )
                    r2.metric("Inserts", raw_metrics.get("inserts", 0))
                    r3.metric("Deletes", raw_metrics.get("deletes", 0))
                    r4.metric("Updates", raw_metrics.get("updates", 0))
                    maps_n = raw_metrics.get("mappings", 0)
                    if maps_n or (algorithm or "").lower() == "zhang_shasha":
                        st.metric(
                            "Mappings",
                            maps_n,
                            help="Zhang–Shasha postorder node alignments; not counted in edit script length.",
                        )
                    note = raw_metrics.get("summary_note")
                    if note:
                        with st.expander("About these metrics"):
                            st.markdown(note)

                sem = compute_result.get("semantic_diff_summary")
                if sem is None:
                    sem = compute_result.get("edit_script_summary") or {}
                if sem and "model" not in sem and "total" not in sem:
                    sem = {
                        "model": "semantic_path_diff",
                        "inserts": sem.get("inserts", 0),
                        "deletes": sem.get("deletes", 0),
                        "updates": sem.get("updates", 0),
                        "total": (
                            sem.get("inserts", 0)
                            + sem.get("deletes", 0)
                            + sem.get("updates", 0)
                        ),
                    }

                st.markdown("#### Semantic path differences (not tied to TED algorithm)")
                st.caption(
                    "Independent tree diff at dotted paths. Use this for human-readable “what changed” "
                    "between infoboxes; it does not depend on Chawathe vs NJ."
                )
                if clean_script:
                    sem_total = sem.get("total") if isinstance(sem, dict) else None
                    if sem_total is None and isinstance(sem, dict):
                        sem_total = (
                            sem.get("inserts", 0)
                            + sem.get("deletes", 0)
                            + sem.get("updates", 0)
                        )
                    if sem_total is None:
                        sem_total = len(clean_script)
                    st.markdown(
                        f"**Semantic differences (total):** {sem_total} — "
                        f"inserts {sem.get('inserts', 0)}, "
                        f"deletes {sem.get('deletes', 0)}, "
                        f"updates {sem.get('updates', 0)}"
                    )
                    total_sem, breakdown = build_summary(clean_script)
                    st.markdown("**By category (semantic):**")
                    order = ["Economy", "Government", "Population", "Geography", "Others"]
                    for cat in order:
                        if cat in breakdown and breakdown[cat]:
                            st.markdown(f"- **{cat}:** {breakdown[cat]} semantic op(s)")
                    for cat, n in sorted(breakdown.items()):
                        if cat not in order:
                            st.markdown(f"- **{cat}:** {n} semantic op(s)")
                else:
                    st.caption("No semantic path-level differences to summarize.")

                view_tabs = st.tabs(["Raw TED script", "Semantic (clean)", "Structured diff"])
                with view_tabs[0]:
                    st.json(raw_script)
                with view_tabs[1]:
                    st.json(clean_script)
                with view_tabs[2]:
                    if clean_script:
                        structured = to_structured_diff(clean_script)
                        st.download_button(
                            "Export Diff",
                            data=structured,
                            file_name=f"diff_{source_slug}_{target_slug}.txt",
                            mime="text/plain",
                            key="export_edit_diff",
                        )
                        with st.expander("View structured diff", expanded=True):
                            st.code(structured, language="text")
                    else:
                        st.info("No semantic differences to display.")

            # --- Section 4 — Patch Result ---
            st.markdown("---")
            st.header("Step 4 — Apply Patch & View Transformation Output")
            if algorithm == "zhang_shasha":
                st.caption(
                    "Zhang–Shasha Apply Patch replays the TED postorder node mapping "
                    "(plus target tree), not LD-pair operations."
                )

            edit_script = st.session_state.get("edit_script")
            if edit_script is None:
                st.info("Compute the edit script first to enable patching.")
                patch_result = None
            else:
                if st.button(
                    "Apply Patch",
                    key="btn_apply_patch",
                    disabled=st.session_state.get("ted_source_tree_for_patch") is None,
                ):
                    with st.spinner("Applying patch…"):
                        orig = st.session_state.get("ted_original_tree_for_patch")
                        tgt = st.session_state.get("ted_target_tree_for_patch")
                        tgt_validation = st.session_state.get("ted_target_tree_for_validation")
                        use_feature_driven = orig is not None and tgt is not None and excluded_features is not None
                        # Pass target_tree for feature-driven patch; pass target for validation (tree_similarity) when available
                        target_for_patch = tgt if use_feature_driven else tgt_validation
                        zs_maps = None
                        if (algorithm or "").lower() == "zhang_shasha":
                            cr = st.session_state.get("ted_compute_result")
                            if isinstance(cr, dict):
                                zs_maps = cr.get("mappings")
                        patch_result = ted_patch(
                            st.session_state["ted_source_tree_for_patch"],
                            edit_script,
                            algorithm=algorithm,
                            original_tree=orig,
                            edit_script_clean=st.session_state.get("ted_edit_script_clean") if not use_feature_driven else None,
                            target_tree=target_for_patch,
                            excluded_features=excluded_features if use_feature_driven else None,
                            mappings=zs_maps,
                        )
                    if patch_result is not None:
                        st.session_state["ted_patch_result"] = patch_result
                        st.success("Patch applied.")
                    else:
                        st.error("Patch request failed. Check API.")

            patch_result = st.session_state.get("ted_patch_result")
            if isinstance(patch_result, dict):
                # Patch validation: tree similarity and diff (patched vs target)
                tree_sim = patch_result.get("tree_similarity")
                if tree_sim is not None:
                    st.metric(
                        "Patch validation (tree similarity)",
                        f"{tree_sim:.4f}",
                        help="1.0 = perfect match, ~0.9+ = very close, lower = patch issues",
                    )
                    patch_diff = patch_result.get("patch_validation_diff") or []
                    if patch_diff:
                        st.markdown("#### Patched vs target — remaining differences")
                        structured_diff = to_structured_diff(patch_diff)
                        st.download_button(
                            "Export Diff",
                            data=structured_diff,
                            file_name=f"patch_validation_diff_{source_slug}_{target_slug}.txt",
                            mime="text/plain",
                            key="export_patch_diff",
                        )
                        with st.expander("View differences (patched → target)", expanded=True):
                            st.code(structured_diff, language="text")
                    else:
                        st.success("Patched tree matches target — no remaining differences.")
                patch_tabs = st.tabs(["Patched JSON", "Patched XML", "Infobox Text"])
                with patch_tabs[0]:
                    st.subheader("Transformation Output (JSON)")
                    st.code(patch_result.get("patched_tree_json", ""), language="json")
                with patch_tabs[1]:
                    st.subheader("Transformation Output (XML)")
                    st.code(patch_result.get("patched_tree_xml", ""), language="xml")
                with patch_tabs[2]:
                    st.subheader("Transformation Output (Infobox text)")
                    st.code(patch_result.get("patched_infobox_text", ""), language="text")

    # --- Tab 1: Country Similarity ---
    with main_tabs[1]:
        st.markdown("---")
        st.header("Country Similarity")
        st.caption("Find countries most similar to a selected country based on TED (Tree Edit Distance) similarity.")

        sim_selected = st.selectbox(
            "Select a country",
            options=[""] + all_display_names,
            index=0,
            key="sim_country_select",
            placeholder="Choose a country…",
        )
        sim_slug = slug_for_name.get(sim_selected) if sim_selected else ""

        top_k_options = [5, 10]
        top_k = st.radio("Show top", options=top_k_options, index=0, horizontal=True, key="sim_top_k")

        if st.button(
            "Find Similar Countries",
            key="btn_find_similar",
            disabled=not sim_slug,
            type="primary",
        ):
            if not sim_slug:
                st.warning("Please select a country first.")
            else:
                with st.spinner("Computing similarity ranking… (this may take a minute)"):
                    results = similarity_ranking_api(sim_slug, top_k=top_k)

                if results is not None:
                    st.session_state["sim_results"] = results
                    st.session_state["sim_query_country"] = sim_selected
                else:
                    st.session_state.pop("sim_results", None)
                    st.error("Failed to fetch similarity ranking. Check API logs.")

        # Display results (persisted in session state)
        results = st.session_state.get("sim_results")
        if results is not None:
            query_country = st.session_state.get("sim_query_country", "")
            display_name = query_country or slug_for_name.get(query_country, "")

            def _render_ranking(rank_list: List[Dict[str, Any]], label: str) -> None:
                st.markdown(f"**{label}**")
                if not rank_list:
                    st.caption("No results.")
                else:
                    for i, item in enumerate(rank_list, 1):
                        country_slug = item.get("country", "")
                        score = item.get("score", 0.0)
                        country_display = next(
                            (name for slug, name in index if slug == country_slug),
                            country_slug.replace("_", " ").title(),
                        )
                        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
                        st.markdown(f"{i}. {country_display} — {score_str}")
                st.markdown("")

            chawathe_list = results.get("chawathe") or []
            nj_list = results.get("nj") or []

            if not chawathe_list and not nj_list:
                st.info("No similar countries found.")
            else:
                st.markdown(f"**Most similar to {display_name}:**")
                st.markdown("")
                col_c, col_n = st.columns(2)
                with col_c:
                    _render_ranking(chawathe_list, "Chawathe (LD-pair)")
                with col_n:
                    _render_ranking(nj_list, "Nierman & Jagadish (NJ)")

        st.markdown("---")
        st.subheader("VSM Similarity")
        st.caption("Rank countries by TF-IDF vector similarity using cosine or Pearson correlation.")
        vsm_features = load_features()
        vsm_metric = st.radio(
            "VSM metric",
            options=["cosine", "pcc"],
            index=0,
            horizontal=True,
            key="vsm_similarity_metric",
        )
        vsm_mode = st.radio(
            "VSM mode",
            options=["field", "flat"],
            index=0,
            horizontal=True,
            key="vsm_similarity_mode",
        )
        vsm_selected_features = st.multiselect(
            "Restrict VSM to features (optional)",
            options=vsm_features,
            default=[],
            key="vsm_similarity_features",
        )
        if st.button(
            "Find Similar Countries with VSM",
            key="btn_find_similar_vsm",
            disabled=not sim_slug,
            type="secondary",
        ):
            if not sim_slug:
                st.warning("Please select a country first.")
            else:
                with st.spinner("Computing VSM similarity ranking…"):
                    vsm_results = vsm_similarity_ranking_api(
                        sim_slug,
                        top_k=top_k,
                        metric=vsm_metric,
                        mode=vsm_mode,
                        features=vsm_selected_features or None,
                    )
                if vsm_results is not None:
                    st.session_state["vsm_sim_results"] = vsm_results
                    st.session_state["vsm_sim_query_country"] = sim_selected
                else:
                    st.session_state.pop("vsm_sim_results", None)
                    st.error("Failed to fetch VSM ranking. Check API logs.")

        vsm_results = st.session_state.get("vsm_sim_results")
        if isinstance(vsm_results, dict):
            st.markdown(
                f"**VSM most similar to {st.session_state.get('vsm_sim_query_country', '')}:**"
            )
            rank_list = vsm_results.get("results") or []
            if not rank_list:
                st.info("No VSM results found.")
            else:
                for i, item in enumerate(rank_list, 1):
                    score = item.get("score", 0.0)
                    score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
                    display = item.get("display_name") or item.get("country", "")
                    terms = ", ".join(item.get("matched_terms") or [])
                    suffix = f" | matched: {terms}" if terms else ""
                    st.markdown(f"{i}. {display} — {score_str}{suffix}")

    # --- Tab 2: VSM Search ---
    with main_tabs[2]:
        st.markdown("---")
        st.header("VSM Search")
        st.caption("Run a free-text TF-IDF query over the collected infobox documents.")

        query_text = st.text_input(
            "Query",
            placeholder="e.g. capital Beirut, population density, GDP",
            key="vsm_query_text",
        )
        q_col1, q_col2, q_col3 = st.columns(3)
        with q_col1:
            query_top_k = st.radio(
                "Top results",
                options=[5, 10, 20],
                index=0,
                horizontal=True,
                key="vsm_query_top_k",
            )
        with q_col2:
            query_metric = st.radio(
                "Metric",
                options=["cosine", "pcc"],
                index=0,
                horizontal=True,
                key="vsm_query_metric",
            )
        with q_col3:
            query_mode = st.radio(
                "Mode",
                options=["field", "flat"],
                index=0,
                horizontal=True,
                key="vsm_query_mode",
            )

        query_features = st.multiselect(
            "Restrict query to features (optional)",
            options=load_features(),
            default=[],
            key="vsm_query_features",
        )

        if st.button(
            "Search Documents",
            key="btn_vsm_search",
            disabled=not query_text.strip(),
            type="primary",
        ):
            with st.spinner("Running VSM query…"):
                result = vsm_query_api(
                    query_text,
                    top_k=query_top_k,
                    metric=query_metric,
                    mode=query_mode,
                    features=query_features or None,
                )
            if result is not None:
                st.session_state["vsm_query_result"] = result
            else:
                st.session_state.pop("vsm_query_result", None)
                st.error("VSM query failed. Check API logs.")

        query_result = st.session_state.get("vsm_query_result")
        if isinstance(query_result, dict):
            st.markdown("### Ranked Results")
            st.caption(
                f"Documents: {query_result.get('document_count', 0)} | "
                f"Vocabulary: {query_result.get('vocabulary_size', 0)} | "
                f"Metric: {query_result.get('metric')} | Mode: {query_result.get('mode')}"
            )
            rows = query_result.get("results") or []
            if not rows:
                st.info("No matching documents found.")
            else:
                for i, item in enumerate(rows, 1):
                    score = item.get("score", 0.0)
                    score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
                    terms = ", ".join(item.get("matched_terms") or [])
                    st.markdown(
                        f"{i}. **{item.get('display_name') or item.get('country')}** — {score_str}"
                    )
                    if terms:
                        st.caption(f"Matched terms: {terms}")

    # --- Tab 3: Similarity Clustering ---
    with main_tabs[3]:
        st.markdown("---")
        st.header("Similarity Clustering")
        st.caption("Cluster all countries using VSM vectors or TED similarity profiles.")

        cluster_country_options = ["All countries"] + all_display_names
        selected_cluster_name = st.selectbox(
            "Highlight a country cluster",
            options=cluster_country_options,
            index=0,
            key="vsm_cluster_country",
        )
        selected_cluster_slug = (
            slug_for_name.get(selected_cluster_name)
            if selected_cluster_name != "All countries"
            else None
        )

        c_col0, c_col1, c_col2, c_col3 = st.columns(4)
        with c_col0:
            cluster_source = st.selectbox(
                "Vector source",
                options=["vsm", "ted"],
                index=0,
                key="vsm_cluster_source",
                help="TED computes all-pairs tree similarities first, so it can be slower.",
            )
        with c_col1:
            cluster_algorithm = st.selectbox(
                "Algorithm",
                options=["kmeans", "dbscan", "agglomerative"],
                index=0,
                key="vsm_cluster_algorithm",
            )
        with c_col2:
            cluster_distance = st.selectbox(
                "Distance",
                options=["cosine", "euclidean", "manhattan"],
                index=0,
                key="vsm_cluster_distance",
            )
        with c_col3:
            cluster_mode = st.selectbox(
                "VSM mode",
                options=["field", "flat"],
                index=0,
                key="vsm_cluster_mode",
                disabled=cluster_source == "ted",
            )

        proj_col, _proj_spacer = st.columns([1, 1])
        with proj_col:
            cluster_projection = st.selectbox(
                "2D projection",
                options=["mds", "pca"],
                format_func=lambda value: "MDS (distance map)" if value == "mds" else "PCA (variance axes)",
                index=0,
                key="vsm_cluster_projection",
                help="MDS preserves pairwise distances in the plot; PCA shows the top two principal components of the vectors.",
            )

        ted_col1, ted_col2 = st.columns(2)
        with ted_col1:
            cluster_ted_algorithm = st.selectbox(
                "TED algorithm",
                options=["chawathe", "nj", "zhang_shasha"],
                index=0,
                key="vsm_cluster_ted_algorithm",
                disabled=cluster_source != "ted",
            )
        with ted_col2:
            cluster_cost_model = st.selectbox(
                "TED cost model",
                options=["value_aware", "classic"],
                index=0,
                key="vsm_cluster_cost_model",
                disabled=cluster_source != "ted" or cluster_ted_algorithm == "zhang_shasha",
            )

        vsm_tuning_col1, vsm_tuning_col2 = st.columns(2)
        with vsm_tuning_col1:
            cluster_vsm_document_source = st.selectbox(
                "VSM document source",
                options=["comparison_fields", "fields", "all"],
                index=0,
                key="vsm_cluster_document_source",
                disabled=cluster_source == "ted",
                help="Use curated comparison fields by default to avoid schema/noise double-counting.",
            )
        with vsm_tuning_col2:
            cluster_max_df_ratio = st.slider(
                "Max document-frequency ratio",
                min_value=0.50,
                max_value=1.00,
                value=0.85,
                step=0.05,
                key="vsm_cluster_max_df_ratio",
                disabled=cluster_source == "ted",
                help="Terms appearing in more than this fraction of documents are pruned for VSM clustering.",
            )

        param_col1, param_col2, param_col3 = st.columns(3)
        with param_col1:
            cluster_k = st.number_input(
                "K",
                min_value=2,
                max_value=50,
                value=5,
                step=1,
                key="vsm_cluster_k",
                disabled=cluster_algorithm == "dbscan",
            )
        with param_col2:
            cluster_eps = st.number_input(
                "DBSCAN epsilon",
                min_value=0.01,
                value=1.0,
                step=0.1,
                key="vsm_cluster_eps",
                disabled=cluster_algorithm != "dbscan",
            )
        with param_col3:
            cluster_min_pts = st.number_input(
                "DBSCAN MinPts",
                min_value=1,
                max_value=50,
                value=3,
                step=1,
                key="vsm_cluster_min_pts",
                disabled=cluster_algorithm != "dbscan",
            )

        cluster_linkage = st.selectbox(
            "Agglomerative linkage",
            options=["single", "complete", "average"],
            index=2,
            key="vsm_cluster_linkage",
            disabled=cluster_algorithm != "agglomerative",
        )
        cluster_features = st.multiselect(
            "Restrict clustering to features (optional)",
            options=load_features(),
            default=[],
            key="vsm_cluster_features",
        )

        if st.button("Run Clustering", key="btn_vsm_clustering", type="primary"):
            spinner_label = (
                "Computing TED similarities and clustering…"
                if cluster_source == "ted"
                else "Clustering VSM vectors…"
            )
            with st.spinner(spinner_label):
                result = vsm_clustering_api(
                    vector_source=cluster_source,
                    algorithm=cluster_algorithm,
                    distance=cluster_distance,
                    projection=cluster_projection,
                    mode=cluster_mode,
                    country=selected_cluster_slug,
                    features=cluster_features or None,
                    k=int(cluster_k),
                    eps=float(cluster_eps),
                    min_pts=int(cluster_min_pts),
                    linkage=cluster_linkage,
                    ted_algorithm=cluster_ted_algorithm,
                    cost_model=cluster_cost_model if cluster_source == "ted" else None,
                    vsm_document_source=cluster_vsm_document_source,
                    max_df_ratio=float(cluster_max_df_ratio),
                )
            if result is not None:
                st.session_state["vsm_cluster_result"] = result
            else:
                st.session_state.pop("vsm_cluster_result", None)
                st.error("Clustering failed. Check API logs.")

        cluster_result = st.session_state.get("vsm_cluster_result")
        if isinstance(cluster_result, dict):
            metadata = cluster_result.get("metadata") or {}
            st.caption(
                f"Documents: {metadata.get('document_count', 0)} | "
                f"Vocabulary: {metadata.get('vocabulary_size', 0)} | "
                f"Source: {cluster_result.get('vector_source', 'vsm')} | "
                f"Algorithm: {cluster_result.get('algorithm')} | "
                f"Distance: {cluster_result.get('distance')} | "
                f"Mode: {cluster_result.get('mode')} | "
                f"Projection: {metadata.get('projection', metadata.get('layout', 'unknown'))}"
            )
            if metadata.get("projection_stress") is not None:
                st.caption(
                    "MDS stress: "
                    f"{float(metadata.get('projection_stress', 0.0)):.4f} "
                    "(lower means the 2D map preserves pairwise distances better)."
                )
            elif metadata.get("projection") == "pca":
                pc1 = metadata.get("explained_variance_ratio_pc1")
                pc2 = metadata.get("explained_variance_ratio_pc2")
                if pc1 is not None and pc2 is not None:
                    st.caption(
                        f"PCA explained variance: PC1 {float(pc1):.1%}, PC2 {float(pc2):.1%} "
                        "(fraction of total variance along each axis)."
                    )

            projection_name = metadata.get("projection", metadata.get("layout", "mds"))
            x_axis = "MDS dimension 1" if projection_name in {"mds", "classical_mds"} else "PC1"
            y_axis = "MDS dimension 2" if projection_name in {"mds", "classical_mds"} else "PC2"

            points = cluster_result.get("points") or []
            if not points:
                st.info("No cluster points returned.")
            else:
                plot_rows = []
                for point in points:
                    cluster_id = point.get("cluster_id")
                    plot_rows.append(
                        {
                            **point,
                            "cluster_label": "Noise" if point.get("is_noise") else f"Cluster {cluster_id}",
                            "marker_type": "Selected" if point.get("is_selected") else "Country",
                            "marker_size": 15 if point.get("is_selected") else 8,
                            "top_similar_display": ", ".join(
                                item.get("display_name", item.get("country", ""))
                                for item in (point.get("top_similar") or [])[:5]
                            ),
                        }
                    )
                fig = px.scatter(
                    plot_rows,
                    x="x",
                    y="y",
                    color="cluster_label",
                    symbol="marker_type",
                    size="marker_size",
                    hover_name="display_name",
                    hover_data={
                        "country": True,
                        "cluster_id": True,
                        "is_noise": True,
                        "top_similar_display": True,
                        "x": ":.4f",
                        "y": ":.4f",
                        "marker_size": False,
                        "marker_type": False,
                    },
                    title="Country Similarity Clusters",
                )
                fig.update_layout(
                    legend_title_text="Cluster",
                    xaxis_title=x_axis,
                    yaxis_title=y_axis,
                    xaxis={
                        "showgrid": False,
                        "zeroline": False,
                    },
                    yaxis={
                        "showgrid": False,
                        "zeroline": False,
                        "scaleanchor": "x",
                        "scaleratio": 1,
                    },
                )
                st.plotly_chart(fig, use_container_width=True)

            selected_cluster = cluster_result.get("selected_cluster")
            if selected_cluster:
                st.markdown("### Selected Country Cluster")
                names = selected_cluster.get("display_names") or selected_cluster.get("countries") or []
                st.write(", ".join(names))
                terms = selected_cluster.get("centroid_terms") or []
                if terms:
                    st.caption(f"Representative terms: {', '.join(terms)}")

            clusters = cluster_result.get("clusters") or []
            if clusters:
                st.markdown("### Cluster Summary")
                summary_rows = [
                    {
                        "cluster_id": cluster.get("cluster_id"),
                        "size": cluster.get("size"),
                        "is_noise": cluster.get("is_noise"),
                        "representative_terms": ", ".join(cluster.get("centroid_terms") or []),
                        "countries": ", ".join(cluster.get("display_names") or cluster.get("countries") or []),
                    }
                    for cluster in clusters
                ]
                st.dataframe(summary_rows, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
