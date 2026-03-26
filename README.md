# Wikipedia Country Infobox Dataset (Phases 1–2)

This project collects and pre-processes Wikipedia infobox data for all UN-recognized countries, and stores each country as its own JSON document (document-oriented / NoSQL-style). It also converts each JSON document into a rooted, ordered, labeled tree for later tree edit distance experiments, and provides a small Streamlit frontend to browse the data.

## Folder structure

- `src/`
  - `app.py` – FastAPI application; mounts API routers
  - `api/controllers/` – API endpoints (health, wikiinfobox)
  - `application/services/` – service layer (data via core.data, orchestration via utils.compare)
  - `core/` – low-level logic by function
    - `data/` – config, storage, country list (MongoDB, env)
    - `preprocess/` – infobox_parser, normalization, pipeline, tree_builder
    - `similarity/` – TED (Chawathe, NJ), tree_validation, common helpers
    - `patch/` – apply edit scripts (Chawathe + NJ in one module)
    - `postprocess/` – tree → JSON/XML/infobox text, report
    - `edit/` – placeholder (edit script types in domain)
  - `domain/models/` – data structures
    - `tree.py` – TreeNode, `pretty_print`, `draw_tree`, `format_draw_tree_dict` (box-drawing text)
    - `country.py` – CountryInfo
    - `infobox.py` – InfoboxRow, ParsedInfobox
    - `normalized_field.py` – NormalizedField
  - `utils/` – higher-level helpers and entry points
    - `http_client.py` – HTTP GET with retries
    - `compare.py` – comparison pipeline (TED + diff + patch + report)
    - `cli.py` – Phase 1 CLI (collect)
    - `tree_cli.py` – Phase 2 CLI (build trees)
- `frontend/`
  - `app.py` – Streamlit UI; **only talks to the API** (no direct data or logic)

## Backend quick start (pipeline + trees)

1. Create and activate a virtual environment (recommended).
2. Install dependencies and set **MONGODB_URI** (required):

```bash
pip install -r requirements.txt
export MONGODB_URI=mongodb://localhost:27017
```

3. Run the collection pipeline (Phase 1):

```bash
python -m src.utils.cli
```

**Requires `MONGODB_URI`.** By default this will:
- Fetch the list of UN member states from Wikipedia
- Download each country page and extract the infobox
- Normalize the fields
- Write one JSON document per country into **MongoDB only** (no local files)

4. Build tree representations (Phase 2):

```bash
python -m src.utils.tree_cli
```

This will read from MongoDB, build trees from `meta` and `normalized.fields`, and write trees back to **MongoDB only**.

## API (FastAPI)

The **Wikipedia Country Infobox API** exposes all data used by the frontend. The Streamlit app **only** calls this API; it has no direct access to storage or business logic.

- **Base URL:** `http://localhost:8000` (or set `API_URL` for the frontend)
- **Docs:** `http://localhost:8000/docs`
- **Endpoints:** (all data from MongoDB)
  - `GET /health` – health check
  - `GET /wikiinfobox/countries` – list `{slug, display_name}` for all countries
  - `GET /wikiinfobox/countries/{slug}/json` – full JSON document
  - `GET /wikiinfobox/countries/{slug}/json/download` – download JSON as file (attachment; on user request)
  - `GET /wikiinfobox/countries/{slug}/tree` – tree representation
  - `GET /wikiinfobox/countries/{slug}/html` – raw infobox HTML
  - `POST /wikiinfobox/run/collect` – run collection pipeline (fetch infoboxes, store in MongoDB); long-running
  - `POST /wikiinfobox/run/build-trees` – build trees for all (or optional `?slug=...` for one) and store in MongoDB

Run the API locally (**MONGODB_URI required**):

```bash
export MONGODB_URI=mongodb://localhost:27017
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

Run from project root so that `src` is on `PYTHONPATH` (or set `PYTHONPATH=.`).

## Docker and MongoDB

**Storage is MongoDB-only** (no local data paths). The API and pipeline require `MONGODB_URI`. The frontend only calls the API.

### Run with Docker Compose

From the project root:

```bash
docker compose up --build
```

This starts:
- **MongoDB** on port `27017` (data persisted in a volume)
- **API** on `http://localhost:8000`
- **Streamlit frontend** on `http://localhost:8501` (calls the API at `http://api:8000`)

Populate the database and build trees via the API (or via CLI in the api container):

```bash
# Via API (recommended):
curl -X POST http://localhost:8000/wikiinfobox/run/collect
curl -X POST http://localhost:8000/wikiinfobox/run/build-trees

# Or via CLI in the api container:
docker compose run --rm api python -m src.utils.cli
docker compose run --rm api python -m src.utils.tree_cli
```

After that, refresh the Streamlit app in your browser; it will load all data via the API.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGODB_URI` | (required) | MongoDB connection string (e.g. `mongodb://mongodb:27017`). All storage is MongoDB-only. |
| `MONGODB_DATABASE` | `wikinfobox` | Database name |
| `MONGODB_COLLECTION` | `countries` | Collection name; each document has `_id` = country slug |
| `API_URL` | `http://localhost:8000` | Base URL of the API (used by the Streamlit frontend only) |

## Frontend: Streamlit data browser

The Streamlit app is a **thin client**: it only talks to the Wikipedia Infobox API. It has no direct access to MongoDB, files, or any business logic.

It lets you:
- Browse the country list (with search)
- View each country’s tree structure (expandable)
- View each country’s JSON document and raw HTML infobox

### Running the frontend

1. Start the API (see [API](#api-fastapi) above), e.g. `uvicorn src.app:app --host 0.0.0.0 --port 8000`.
2. Set `API_URL` if the API is not at `http://localhost:8000`.
3. Run the frontend:

```bash
streamlit run frontend/app.py
```

Then open the URL printed by Streamlit (typically `http://localhost:8501`).

### Frontend layout

- **Sidebar**
  - Search input to filter countries by name or slug
  - Scrollable list of country names
- **Main panel**
  - Header showing the selected country name and slug
  - Tabs:
    - **Tree** – expandable/collapsible tree (from API/MongoDB)
    - **JSON** – pretty-printed JSON (from API/MongoDB)
    - **HTML Source** – raw HTML (from API/MongoDB)
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

