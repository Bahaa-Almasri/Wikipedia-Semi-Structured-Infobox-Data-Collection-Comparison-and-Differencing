from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from bs4 import BeautifulSoup
from tqdm import tqdm

from core.data.config import WIKIPEDIA
from core.data.storage import iso_now, write_json_document
from core.data.country_list import fetch_un_member_states
from core.preprocess.infobox_parser import parse_infobox
from core.preprocess.normalization import build_comparison_fields, normalize_rows, normalized_fields_to_dict
from domain.models.country import CountryInfo
from domain.models.infobox import ParsedInfobox
from utils.http_client import get


def _raw_value_text_from_html(value_html: str) -> str:
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
        {"label": row.label, "value_text": row.value_text}
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
        "cleaned": {"rows": cleaned_rows},
        "normalized": {
            "fields": normalized_dict,
            "comparison_fields": comparison_fields,
        },
    }
    return document


def collect_single_country(country: CountryInfo, *, save_html: bool = True) -> Optional[Dict]:
    """Collect, parse, normalize, and persist one country's infobox to MongoDB. save_html is ignored (HTML is always stored in the document)."""
    html = get(
        country.wikipedia_url,
        timeout=WIKIPEDIA.request_timeout,
        headers={"User-Agent": WIKIPEDIA.user_agent},
        max_retries=WIKIPEDIA.max_retries,
        backoff_factor=WIKIPEDIA.backoff_factor,
    )
    parsed = parse_infobox(html)
    if parsed is None:
        return None

    doc = _build_document(country, parsed)
    write_json_document(country.slug, doc)
    return doc


def collect_all_countries(
    countries: Optional[Iterable[CountryInfo]] = None,
    *,
    save_html: bool = True,
) -> List[str]:
    """Collect data for all UN member states into MongoDB."""
    if countries is None:
        countries = fetch_un_member_states()

    written: List[str] = []
    for country in tqdm(list(countries), desc="Collecting countries"):
        try:
            doc = collect_single_country(country, save_html=save_html)
            if doc is not None:
                written.append(country.slug)
        except Exception as exc:
            print(f"Failed to collect {country.name}: {exc}")

    return written
