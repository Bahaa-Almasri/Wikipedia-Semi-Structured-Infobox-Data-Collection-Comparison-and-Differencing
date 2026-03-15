"""
Wikipedia Country Infobox Browser (Streamlit).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

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


def ted_patch(
    source_tree: Dict[str, Any],
    edit_script: Dict[str, Any],
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
    st.set_page_config(
        page_title="Wikipedia Country Infobox Browser",
        layout="wide",
    )

    st.title("Wikipedia Country Infobox Browser")
    st.caption("Phase 1–2 data viewer: JSON documents, trees, and raw HTML (data via API).")

    # Sidebar: data pipeline (run collect / preprocess)
    with st.sidebar:
        st.header("Data pipeline")
        if st.button("Run collect", help="Fetch UN member states, scrape infoboxes, store in MongoDB. Long-running."):
            with st.spinner("Running collect… (may take several minutes)"):
                result = run_collect()
            if result is not None:
                load_country_index.clear()
                st.success(f"Collected {result.get('collected', 0)} countries. Refresh or re-run to see the list.")
                st.caption(f"Slugs: {result.get('slugs', [])[:5]}{'…' if len(result.get('slugs', [])) > 5 else ''}")
            else:
                st.error("Collect failed or timed out. Check API logs.")
        if st.button("Run preprocess (all)", help="Build trees for all countries in MongoDB."):
            with st.spinner("Running preprocess… (may take a minute)"):
                result = run_preprocess()
            if result is not None:
                st.success(f"Built trees for {result.get('trees_built', 0)} countries.")
                st.caption(f"Slugs: {result.get('slugs', [])[:5]}{'…' if len(result.get('slugs', [])) > 5 else ''}")
            else:
                st.error("Preprocess failed or timed out. Check API logs.")
        st.divider()

    if not api_available():
        st.error(
            f"Cannot reach the API at {API_URL}. Start the API server (e.g. uvicorn src.app:app --host 0.0.0.0 --port 8000) "
            "and set API_URL if it runs elsewhere."
        )
        return

    index = load_country_index()
    if not index:
        st.sidebar.error("No country data returned from the API. Run the pipeline to populate data.")
        return

    st.sidebar.header("Countries")
    search_query = st.sidebar.text_input("Search by name", "").strip().lower()

    filtered = [
        (slug, name)
        for slug, name in index
        if search_query in name.lower() or search_query in slug.lower()
    ]

    if not filtered:
        st.sidebar.info("No countries match this search.")
        return

    display_names = [name for _, name in filtered]
    default_idx = 0
    selected_name = st.sidebar.selectbox("Select a country", display_names, index=default_idx)
    selected_slug = next(slug for slug, name in filtered if name == selected_name)

    # Option to preprocess only current country
    with st.sidebar:
        if st.button("Run preprocess (this country)", key="preprocess_one", help=f"Build tree for {selected_name} only."):
            with st.spinner(f"Building tree for {selected_slug}…"):
                result = run_preprocess(slug=selected_slug)
            if result is not None:
                st.success(f"Built tree for {result.get('trees_built', 0)} country.")
            else:
                st.error("Preprocess failed. Check API logs.")
        st.divider()

    st.subheader(f"{selected_name}")
    st.caption(f"Slug: `{selected_slug}`")

    tree_data = load_tree(selected_slug)
    json_data = load_json(selected_slug)
    html_data = load_html(selected_slug)

    tabs = st.tabs(["Tree", "JSON", "HTML Source", "HTML Preview", "Compare Trees"])

    with tabs[0]:
        st.markdown("### Tree View")
        if tree_data is None:
            st.warning("No tree data found for this country. Build trees via the pipeline/tree_cli.")
        else:
            root_label = tree_data.get("label", "")
            st.markdown(f"**Root:** `{root_label}`")
            children = tree_data.get("children") or []
            for child in children:
                child_label = child.get("label", "")
                with st.expander(child_label, expanded=False):
                    render_tree_branch_inline(child, level=1)

    with tabs[1]:
        st.markdown("### JSON View")
        if json_data is None:
            st.warning("No JSON document found for this country.")
        else:
            st.json(json_data)

    with tabs[2]:
        st.markdown("### HTML Source")
        if html_data is None:
            st.warning("No raw HTML found for this country.")
        else:
            st.code(html_data, language="html")

    with tabs[3]:
        st.markdown("### HTML Preview")
        if html_data is None:
            st.warning("No raw HTML found for this country.")
        else:
            st.components.v1.html(html_data, height=600, scrolling=True)

    with tabs[4]:
        st.markdown("### Compare Trees (TED, diff, patch, postprocess)")
        st.caption(
            "Select two countries, choose Chawathe or Nierman & Jagadish (NJ), then compute similarity, "
            "edit script (diff), apply patch, or post-process a tree to JSON/XML/infobox."
        )

        all_index = load_country_index()
        if not all_index:
            st.warning("No country data found for comparison.")
        else:
            all_display_names = [name for _, name in all_index]
            try:
                default_a_idx = all_display_names.index(selected_name)
            except ValueError:
                default_a_idx = 0
            default_b_idx = 0 if default_a_idx != 0 else (1 if len(all_display_names) > 1 else 0)

            col_select_a, col_select_b = st.columns(2)
            with col_select_a:
                country_a_name = st.selectbox(
                    "Country A", all_display_names, index=default_a_idx, key="compare_country_a"
                )
            with col_select_b:
                country_b_name = st.selectbox(
                    "Country B", all_display_names, index=default_b_idx, key="compare_country_b"
                )

            slug_for_name = {name: slug for slug, name in all_index}
            slug_a = slug_for_name.get(country_a_name)
            slug_b = slug_for_name.get(country_b_name)

            algorithm = st.radio(
                "TED algorithm",
                options=["chawathe", "nj"],
                index=0,
                format_func=lambda x: "Chawathe (LD-pair)" if x == "chawathe" else "Nierman & Jagadish (NJ)",
                key="ted_algorithm",
            )
            coerce_root = st.checkbox(
                "Coerce root label (compare content only, ignore root name)",
                value=True,
                key="ted_coerce_root",
            )
            coerce_root_label = "infobox" if coerce_root else None

            # Actions: similarity, diff, patch, postprocess
            st.markdown("#### Actions")
            col_sim, col_diff, col_post = st.columns(3)

            with col_sim:
                if st.button("Compute similarity", key="btn_similarity"):
                    if slug_a and slug_b:
                        with st.spinner("Computing TED similarity…"):
                            result = ted_similarity(
                                slug_a, slug_b,
                                algorithm=algorithm,
                                coerce_root_label=coerce_root_label,
                            )
                        if result is not None:
                            st.success(
                                f"**Distance:** {result.get('distance', '—')}  \n"
                                f"**Similarity:** {result.get('similarity', 0):.4f}  \n"
                                f"**Source size:** {result.get('source_size', '—')}  \n"
                                f"**Target size:** {result.get('target_size', '—')}"
                            )
                        else:
                            st.error("Similarity request failed. Check API.")
                    else:
                        st.warning("Select both countries.")

            with col_diff:
                if st.button("Compute diff (edit script + patch + report)", key="btn_diff"):
                    if slug_a and slug_b:
                        with st.spinner("Computing diff…"):
                            result = ted_diff(
                                slug_a, slug_b,
                                algorithm=algorithm,
                                coerce_root_label=coerce_root_label,
                            )
                        if result is not None:
                            if "ted_diff_result" not in st.session_state:
                                st.session_state["ted_diff_result"] = {}
                            st.session_state["ted_diff_result"] = result
                            st.success(
                                f"**Distance:** {result.get('distance')}  **Similarity:** {result.get('similarity', 0):.4f}  \n"
                                f"Patch matches target: **{result.get('patch_matches_target', False)}**"
                            )
                        else:
                            st.error("Diff request failed. Check API.")
                    else:
                        st.warning("Select both countries.")

            # Show last diff result (edit script, patched tree, report, postprocess)
            diff_result = st.session_state.get("ted_diff_result")
            if diff_result is not None and isinstance(diff_result, dict):
                st.markdown("---")
                st.markdown("#### Last diff result")
                summary = diff_result.get("edit_script_summary") or {}
                st.json(summary)
                with st.expander("Full edit script (JSON)"):
                    st.json(diff_result.get("edit_script") or {})
                with st.expander("Patched tree (JSON)"):
                    st.json(diff_result.get("patched_tree") or {})
                st.markdown("**Report**")
                st.text(diff_result.get("report_text", ""))
                st.markdown("**Patched tree — JSON**")
                st.code(diff_result.get("patched_tree_json", ""), language="json")
                st.markdown("**Patched tree — XML**")
                st.code(diff_result.get("patched_tree_xml", ""), language="xml")
                st.markdown("**Patched tree — Infobox text**")
                st.code(diff_result.get("patched_infobox_text", ""), language="text")

            with col_post:
                st.markdown("Post-process **current** country tree to JSON/XML/infobox:")
                if st.button("Post-process this country tree", key="btn_postprocess"):
                    tree_for_post = load_tree(selected_slug)
                    if tree_for_post is not None:
                        with st.spinner("Post-processing…"):
                            out = ted_postprocess(tree_for_post)
                        if out is not None:
                            st.session_state["postprocess_out"] = out
                            st.success("Post-process done.")
                        else:
                            st.error("Postprocess request failed.")
                    else:
                        st.warning("No tree for current country.")

            if "postprocess_out" in st.session_state:
                po = st.session_state["postprocess_out"]
                st.markdown("---")
                st.markdown("#### Post-process output (current country tree)")
                st.markdown("**JSON**")
                st.code(po.get("json", ""), language="json")
                st.markdown("**XML**")
                st.code(po.get("xml", ""), language="xml")
                st.markdown("**Infobox text**")
                st.code(po.get("infobox_text", ""), language="text")

            st.markdown("---")
            st.markdown("#### Tree view (side by side)")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"##### {country_a_name}")
                if not slug_a:
                    st.warning("Could not resolve Country A slug.")
                else:
                    tree_a = load_tree(slug_a)
                    if tree_a is None:
                        st.warning(f"No tree found for `{slug_a}`.")
                    else:
                        root_label_a = tree_a.get("label", "")
                        st.markdown(f"**Root:** `{root_label_a}`")
                        for child in tree_a.get("children") or []:
                            with st.expander(child.get("label", ""), expanded=False):
                                render_tree_branch_inline(child, level=1)

            with col_b:
                st.markdown(f"##### {country_b_name}")
                if not slug_b:
                    st.warning("Could not resolve Country B slug.")
                else:
                    tree_b = load_tree(slug_b)
                    if tree_b is None:
                        st.warning(f"No tree found for `{slug_b}`.")
                    else:
                        root_label_b = tree_b.get("label", "")
                        st.markdown(f"**Root:** `{root_label_b}`")
                        for child in tree_b.get("children") or []:
                            with st.expander(child.get("label", ""), expanded=False):
                                render_tree_branch_inline(child, level=1)


if __name__ == "__main__":
    main()
