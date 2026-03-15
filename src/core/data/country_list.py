from __future__ import annotations

from typing import Iterable, List

from bs4 import BeautifulSoup

from core.data.config import WIKIPEDIA
from domain.models.country import CountryInfo
from utils.http_client import get


def _slugify(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
    )


def fetch_un_member_states() -> List[CountryInfo]:
    """
    Fetch the list of UN member states from Wikipedia.

    Returns a list of CountryInfo objects with:
    - human-readable name
    - absolute Wikipedia URL
    - simple slug to be used for filenames
    """
    html = get(
        WIKIPEDIA.un_member_states_url,
        timeout=WIKIPEDIA.request_timeout,
        headers={"User-Agent": WIKIPEDIA.user_agent},
        max_retries=WIKIPEDIA.max_retries,
        backoff_factor=WIKIPEDIA.backoff_factor,
    )
    soup = BeautifulSoup(html, "html.parser")

    tables: Iterable = soup.select("table.wikitable.sortable")
    countries: List[CountryInfo] = []

    for table in tables:
        for row in table.select("tr"):
            link = row.find("a")
            if not link or not link.get("href"):
                continue

            href = link["href"]
            if not href.startswith("/wiki/"):
                continue

            name = link.get_text(strip=True)
            if not name:
                continue

            url = f"{WIKIPEDIA.base_url}{href}"
            countries.append(
                CountryInfo(
                    name=name,
                    wikipedia_url=url,
                    slug=_slugify(name),
                )
            )

    seen = set()
    unique_countries: List[CountryInfo] = []
    for c in countries:
        if c.slug in seen:
            continue
        seen.add(c.slug)
        unique_countries.append(c)

    return unique_countries
