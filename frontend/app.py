"""
Wikipedia Country Infobox Browser (Streamlit).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import xml.etree.ElementTree as ET

import requests
import streamlit as st

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
) -> Optional[Dict[str, Any]]:
    """POST /ted/patch — apply edit script to source tree."""
    try:
        r = requests.post(
            f"{WIKIINFOBOX_PREFIX}/ted/patch",
            json={"source_tree": source_tree, "edit_script": edit_script, "algorithm": algorithm},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
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


def json_to_xml(data: Any, root_name: str = "root") -> str:
    """
    Convert JSON-like data (dict/list/scalar) to an XML string.
    Intended for visualization only in the frontend (no backend calls).
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


def render_tree_branch_inline(node: Dict[str, Any], level: int = 0) -> None:
    """
    Recursively render a tree branch as indented text (no nested expanders).
    Expected node format: { "label": str, "value": Optional[str], "children": [ ... ] }
    """
    label = node.get("label", "")
    value = node.get("value")
    children = node.get("children") or []

    indent = "&nbsp;" * 4 * level
    if children:
        if value is not None:
            st.markdown(f"{indent}**{label}**: `{value}`", unsafe_allow_html=True)
        else:
            st.markdown(f"{indent}**{label}**", unsafe_allow_html=True)
        for child in children:
            render_tree_branch_inline(child, level=level + 1)
    else:
        if value is not None:
            st.markdown(f"{indent}**{label}** = `{value}`", unsafe_allow_html=True)
        else:
            st.markdown(f"{indent}**{label}**", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Wikipedia Country Infobox Browser", layout="wide")

    st.markdown(
        """
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
h1, h2, h3 {
    color: #1f4e79;
}
.stButton>button {
    background-color: #1f4e79;
    color: white;
    border-radius: 8px;
    padding: 0.5rem 1rem;
}
.stMetric {
    background-color: #f5f7fa;
    padding: 10px;
    border-radius: 10px;
}
</style>
""",
        unsafe_allow_html=True,
    )

    st.title("Wikipedia Country Infobox Browser")
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

        st.divider()
        st.caption("TED operations run against the backend + MongoDB.")

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

    if not source_slug or not target_slug:
        st.warning("Please select both countries.")
        return

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
                st.markdown(f"**Root:** `{source_tree.get('label', '')}`")
                for child in (source_tree.get("children") or []):
                    with st.expander(child.get("label", ""), expanded=False):
                        render_tree_branch_inline(child, level=1)
        with right:
            st.markdown(f"**Target:** `{target_country}`")
            if target_tree is None:
                st.warning("No tree found.")
            else:
                st.markdown(f"**Root:** `{target_tree.get('label', '')}`")
                for child in (target_tree.get("children") or []):
                    with st.expander(child.get("label", ""), expanded=False):
                        render_tree_branch_inline(child, level=1)

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
            options=["chawathe", "nj"],
            index=0,
            format_func=lambda x: "Chawathe (LD-pair)" if x == "chawathe" else "Nierman & Jagadish (NJ)",
            key="ted_algorithm",
        )
    with col_coerce:
        coerce_root = st.checkbox(
            "Coerce root label (compare content only)",
            value=True,
            key="ted_coerce_root",
        )
    coerce_root_label = "infobox" if coerce_root else None

    ui_key = (source_slug, target_slug, algorithm, coerce_root_label)
    if st.session_state.get("ted_ui_key") != ui_key:
        st.session_state["ted_ui_key"] = ui_key
        st.session_state.pop("ted_compute_result", None)
        st.session_state.pop("edit_script", None)
        st.session_state.pop("ted_source_tree_for_patch", None)
        st.session_state.pop("ted_patch_result", None)

    if st.button(
        "Compute Edit Script",
        key="btn_compute_edit_script",
        disabled=source_tree is None or target_tree is None,
    ):
        if source_tree is None or target_tree is None:
            st.warning("Trees missing. Run preprocess first.")
        else:
            with st.spinner("Computing TED edit script…"):
                compute_result = ted_compute(
                    source_tree,
                    target_tree,
                    algorithm=algorithm,
                    coerce_root_label=coerce_root_label,
                )

            if compute_result is not None:
                st.session_state["ted_compute_result"] = compute_result
                st.session_state["edit_script"] = compute_result.get("edit_script", [])
                source_tree_for_patch = dict(source_tree)  # top-level label only
                if coerce_root_label:
                    source_tree_for_patch["label"] = coerce_root_label
                st.session_state["ted_source_tree_for_patch"] = source_tree_for_patch
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
        with st.expander("View Edit Script", expanded=False):
            st.json(compute_result.get("edit_script") or [])

    # --- Section 4 — Patch Result ---
    st.markdown("---")
    st.header("Step 4 — Apply Patch & View Transformation Output")

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
                patch_result = ted_patch(
                    st.session_state["ted_source_tree_for_patch"],
                    edit_script,
                    algorithm=algorithm,
                )
            if patch_result is not None:
                st.session_state["ted_patch_result"] = patch_result
                st.success("Patch applied.")
            else:
                st.error("Patch request failed. Check API.")

    patch_result = st.session_state.get("ted_patch_result")
    if isinstance(patch_result, dict):
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


if __name__ == "__main__":
    main()
