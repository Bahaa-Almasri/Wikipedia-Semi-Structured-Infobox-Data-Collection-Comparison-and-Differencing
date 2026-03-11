# Wikipedia Country Infobox Dataset (Phase 1)

This project collects and pre-processes Wikipedia infobox data for all UN-recognized countries, and stores each country as its own JSON document (document-oriented / NoSQL-style).

Phase 1 covers:
- Data collection from Wikipedia
- Infobox scraping and parsing
- Light normalization
- JSON document storage (one file per country)

No tree conversion, tree edit distance, diff, or patching is implemented in this phase.

## Folder structure

- `src/`
  - `wikinfobox/`
    - `__init__.py`
    - `config.py` – configuration and constants
    - `country_list.py` – fetches list of UN member states and their Wikipedia URLs
    - `fetch.py` – HTTP helpers with retries
    - `infobox_parser.py` – parses HTML infobox into raw row structures
    - `normalization.py` – normalization rules for keys and values
    - `storage.py` – JSON file writing / path handling
    - `pipeline.py` – high-level orchestration pipeline
    - `cli.py` – command-line entry point
- `data/`
  - `raw_html/` – optional raw HTML snapshots of infobox tables
  - `json/` – final JSON documents, one file per country

## Quick start

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the collection pipeline:

```bash
python -m src.wikinfobox.cli
```

By default this will:
- Fetch the list of UN member states from Wikipedia
- Download each country page and extract the infobox
- Normalize the fields
- Write one JSON document per country into `data/json/`

## Phase 2 (not implemented yet)

Phase 2 will read these JSON documents and:
- Convert them into rooted ordered labeled trees
- Compute tree edit distance between two country trees
- Extract edit scripts (diffs)
- Apply patches and post-process back to XML/JSON/Wikipedia-like formats

See `src/wikinfobox/pipeline.py` and the comments in `src/wikinfobox/normalization.py` for phase 2 handoff notes.

