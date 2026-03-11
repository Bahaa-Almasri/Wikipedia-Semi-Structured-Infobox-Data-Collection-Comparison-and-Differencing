# Wikipedia Country Infobox Dataset (Phases 1–2)

This project collects and pre-processes Wikipedia infobox data for all UN-recognized countries, and stores each country as its own JSON document (document-oriented / NoSQL-style). It also converts each JSON document into a rooted, ordered, labeled tree for later tree edit distance experiments, and provides a small Streamlit frontend to browse the data.

## Folder structure

- `src/`
  - `wikinfobox/`
    - `__init__.py`
    - `config.py` – configuration and constants
    - `country_list.py` – fetches list of UN member states and their Wikipedia URLs
    - `fetch.py` – HTTP helpers with retries
    - `infobox_parser.py` – parses HTML infobox into raw row structures
    - `normalization.py` – normalization rules for keys and values
    - `storage.py` – JSON/tree file writing / path handling
    - `pipeline.py` – high-level orchestration pipeline (Phase 1)
    - `cli.py` – Phase 1 command-line entry point
    - `tree.py` – `TreeNode` dataclass and pretty-printer
    - `tree_builder.py` – JSON → tree conversion utilities
    - `tree_cli.py` – Phase 2 command-line entry point (build trees)
- `data/`
  - `raw_html/` – optional raw HTML snapshots of infobox tables
  - `json/` – final JSON documents, one file per country
  - `trees/` – tree JSON documents, one file per country
- `frontend/`
  - `app.py` – Streamlit frontend for browsing countries, trees, JSON, and HTML

## Backend quick start (pipeline + trees)

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the collection pipeline (Phase 1):

```bash
python -m src.wikinfobox.cli
```

By default this will:
- Fetch the list of UN member states from Wikipedia
- Download each country page and extract the infobox
- Normalize the fields
- Write one JSON document per country into `data/json/`

4. Build tree representations (Phase 2, still without TED/diff/patching):

```bash
python -m src.wikinfobox.tree_cli
```

This will:
- Read all JSON documents from `data/json/`
- Build rooted ordered labeled trees based on `meta` and `normalized.fields`
- Write one tree JSON per country into `data/trees/`

## Frontend: Streamlit data browser

A small Streamlit app is provided to visually browse:
- The country list (with search)
- Each country’s tree structure (expandable hierarchical view)
- Each country’s JSON document
- Each country’s raw HTML infobox

### Running the frontend

From the project root, after installing requirements and generating the data:

```bash
streamlit run frontend/app.py
```

Then open the URL printed by Streamlit in your browser (typically `http://localhost:8501`).

### Frontend layout

- **Sidebar**
  - Search input to filter countries by name or slug
  - Scrollable list of country names
- **Main panel**
  - Header showing the selected country name and slug
  - Tabs:
    - **Tree** – expandable/collapsible tree built from `data/trees/<slug>.json`
    - **JSON** – pretty-printed JSON from `data/json/<slug>.json`
    - **HTML Source** – raw HTML from `data/raw_html/<slug>.html`
    - **HTML Preview** – rendered HTML preview of the infobox

The Tree tab uses a recursive, expandable component so you can drill into nested nodes for debugging.

### Example layout (screenshot description)

Imagine a browser window split into:
- A narrow **left sidebar** with:
  - A text box labeled “Search by name” (e.g. you type “alb”).
  - A filtered list showing “Albania”, “Algeria”, etc. as clickable entries.
- A wide **main area** with:
  - A header: “Albania” and a smaller caption “Slug: `albania`”.
  - A row of tabs: “Tree | JSON | HTML Source | HTML Preview”.
  - In the **Tree** tab, a vertical list of expandable sections:
    - `meta` (expanded) showing simple leaves like `country_name = Albania`.
    - `fields` (expanded) containing nested expanders for `capital_and_largest_city`, `government`, `gdp_ppp`, etc., each with further children for `text`, `numbers`, `tokens`, and their leaf values.

