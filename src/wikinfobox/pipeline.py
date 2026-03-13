from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterable, List, Optional

from bs4 import BeautifulSoup
from tqdm import tqdm

from .config import WIKIPEDIA
from .country_list import CountryInfo, fetch_un_member_states
from .fetch import get
from .infobox_parser import ParsedInfobox, parse_infobox
from .normalization import build_comparison_fields, normalize_rows, normalized_fields_to_dict
from .storage import (
    ensure_data_dirs,
    iso_now,
    raw_html_path_for_slug,
    write_json_document,
)


def _raw_value_text_from_html(value_html: str) -> str:
    """
    Convert a value cell's HTML into a raw text approximation.

    This keeps citation markers (e.g. [1]) and does not attempt to
    clean nested structures; it's meant as a close textual view of
    the original HTML.
    """
    soup = BeautifulSoup(value_html, "html.parser")
    return soup.get_text(" ", strip=True)


def _build_document(
    country: CountryInfo,
    parsed: ParsedInfobox,
) -> Dict:
    normalized = normalize_rows(parsed.rows)
    normalized_dict = normalized_fields_to_dict(normalized)
    comparison_fields = build_comparison_fields(normalized, country_name=country.name)

    raw_rows = [
        {
            "label": row.label,
            "value_html": row.value_html,
            "value_text": _raw_value_text_from_html(row.value_html),
        }
        for row in parsed.rows
    ]

    cleaned_rows = [
        {
            "label": row.label,
            "value_text": row.value_text,
        }
        for row in parsed.rows
    ]

    document = {
        "meta": {
            "country_name": country.name,
            "slug": country.slug,
            "wikipedia_url": country.wikipedia_url,
            "retrieved_at": iso_now(),
            "source": "wikipedia-infobox",
        },
        "raw": {
            "infobox_html": parsed.html,
            "rows": raw_rows,
        },
        "cleaned": {
            "rows": cleaned_rows,
        },
        "normalized": {
            "fields": normalized_dict,
            "comparison_fields": comparison_fields,
        },
    }
    return document


def collect_single_country(country: CountryInfo, *, save_html: bool = True) -> Optional[Dict]:
    """
    Collect, parse, normalize, and persist one country's infobox.

    Returns the JSON-serializable document, or None if infobox not found.
    """
    html = get(country.wikipedia_url)
    parsed = parse_infobox(html)
    if parsed is None:
        return None

    if save_html:
        raw_path = raw_html_path_for_slug(country.slug)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(parsed.html, encoding="utf-8")

    doc = _build_document(country, parsed)
    write_json_document(country.slug, doc)
    return doc


def collect_all_countries(
    countries: Optional[Iterable[CountryInfo]] = None,
    *,
    save_html: bool = True,
) -> List[str]:
    """
    Collect data for all UN member states.

    Returns a list of slugs for which documents were successfully written.
    """
    ensure_data_dirs()

    if countries is None:
        countries = fetch_un_member_states()

    written: List[str] = []

    for country in tqdm(list(countries), desc="Collecting countries"):
        try:
            doc = collect_single_country(country, save_html=save_html)
            if doc is not None:
                written.append(country.slug)
        except Exception as exc:  # noqa: BLE001
            # For phase 1, we log to stderr and continue.
            # In a more robust setup, use proper logging.
            print(f"Failed to collect {country.name}: {exc}")

    return written

