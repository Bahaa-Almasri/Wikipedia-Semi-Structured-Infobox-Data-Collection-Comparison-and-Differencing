from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_JSON_DIR = PROJECT_ROOT / "data" / "json"
DATA_TREES_DIR = PROJECT_ROOT / "data" / "trees"
DATA_HTML_DIR = PROJECT_ROOT / "data" / "raw_html"


def load_country_slugs() -> List[str]:
    if not DATA_JSON_DIR.exists():
        return []
    return sorted(p.stem for p in DATA_JSON_DIR.glob("*.json"))


@st.cache_data
def load_country_index() -> List[Tuple[str, str]]:
    """
    Return a list of (slug, display_name) tuples.

    display_name is taken from meta.country_name when available,
    otherwise falls back to the slug.
    """
    slugs = load_country_slugs()
    index: List[Tuple[str, str]] = []

    for slug in slugs:
        json_path = DATA_JSON_DIR / f"{slug}.json"
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            meta = data.get("meta", {}) or {}
            name = meta.get("country_name") or slug.replace("_", " ").title()
        except Exception:
            name = slug
        index.append((slug, name))

    return index


def load_tree(slug: str) -> Optional[Dict]:
    path = DATA_TREES_DIR / f"{slug}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_json(slug: str) -> Optional[Dict]:
    path = DATA_JSON_DIR / f"{slug}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_html(slug: str) -> Optional[str]:
    path = DATA_HTML_DIR / f"{slug}.html"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def render_tree_branch_inline(node: Dict, level: int = 0) -> None:
    """
    Recursively render a tree branch as indented text (no nested expanders).

    Expected node format matches TreeNode.to_dict():
    { "label": str, "value": Optional[str], "children": [ ... ] }
    """
    label = node.get("label", "")
    value = node.get("value")
    children = node.get("children") or []

    indent = "&nbsp;" * 4 * level
    if children:
        # Internal node
        if value is not None:
            st.markdown(f"{indent}**{label}**: `{value}`", unsafe_allow_html=True)
        else:
            st.markdown(f"{indent}**{label}**", unsafe_allow_html=True)
        for child in children:
            render_tree_branch_inline(child, level=level + 1)
    else:
        # Leaf node
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
    st.caption("Phase 1–2 data viewer: JSON documents, trees, and raw HTML.")

    # Sidebar: search & country list
    st.sidebar.header("Countries")
    index = load_country_index()
    if not index:
        st.sidebar.error("No country JSON files found in data/json.")
        return

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

    # Map back to slug
    selected_slug = next(slug for slug, name in filtered if name == selected_name)

    st.subheader(f"{selected_name}")
    st.caption(f"Slug: `{selected_slug}`")

    tree_data = load_tree(selected_slug)
    json_data = load_json(selected_slug)
    html_data = load_html(selected_slug)

    # Tabs for different views
    tabs = st.tabs(["Tree", "JSON", "HTML Source", "HTML Preview"])

    with tabs[0]:
        st.markdown("### Tree View")
        if tree_data is None:
            st.warning("No tree file found for this country in `data/trees`.")
        else:
            root_label = tree_data.get("label", "")
            st.markdown(f"**Root:** `{root_label}`")
            children = tree_data.get("children") or []
            # Use one level of expanders for top-level branches only
            for child in children:
                child_label = child.get("label", "")
                with st.expander(child_label, expanded=False):
                    render_tree_branch_inline(child, level=1)

    with tabs[1]:
        st.markdown("### JSON View")
        if json_data is None:
            st.warning("No JSON document found for this country in `data/json`.")
        else:
            st.json(json_data)

    with tabs[2]:
        st.markdown("### HTML Source")
        if html_data is None:
            st.warning("No raw HTML file found for this country in `data/raw_html`.")
        else:
            st.code(html_data, language="html")

    with tabs[3]:
        st.markdown("### HTML Preview")
        if html_data is None:
            st.warning("No raw HTML file found for this country in `data/raw_html`.")
        else:
            # Render inside a scrollable container
            st.components.v1.html(
                html_data,
                height=600,
                scrolling=True,
            )


if __name__ == "__main__":
    main()

