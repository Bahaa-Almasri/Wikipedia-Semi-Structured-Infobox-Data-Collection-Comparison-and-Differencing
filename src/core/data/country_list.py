from __future__ import annotations

from typing import Iterable, List, Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Tag

from core.data.config import WIKIPEDIA
from domain.schemas.country import CountryInfo
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


def _table_headers(table: Tag) -> List[str]:
    return [th.get_text(" ", strip=True).casefold() for th in table.select("th")]


def _is_current_members_table(table: Tag) -> bool:
    headers = _table_headers(table)
    return any("member state" in h for h in headers) and any(
        "date of admission" in h for h in headers
    )


def _candidate_member_tables(soup: BeautifulSoup) -> Iterable[Tag]:
    current_members = soup.find(id="Current_members")
    if current_members is not None:
        heading = current_members.find_parent(["h2", "h3"])
        if heading is not None:
            for sibling in heading.find_all_next():
                if sibling.name in {"h2", "h3"}:
                    break
                if sibling.name == "table" and "wikitable" in (sibling.get("class") or []):
                    yield sibling

    yield from soup.select("table.wikitable")


def _country_link_from_row(row: Tag) -> Optional[Tag]:
    first_cell = row.find(["th", "td"], recursive=False)
    if first_cell is None:
        return None

    for link in first_cell.find_all("a", href=True):
        href = str(link.get("href") or "")
        if _country_url_from_href(href) is None:
            continue
        if not link.get_text(strip=True):
            continue
        return link
    return None


def _country_url_from_href(href: str) -> Optional[str]:
    if href.startswith("/wiki/"):
        article_path = href
        url = f"{WIKIPEDIA.base_url}{href}"
    elif href.startswith("//") and "/wiki/" in href:
        article_path = urlparse(f"https:{href}").path
        url = f"https:{href}"
    elif href.startswith(("http://", "https://")) and "/wiki/" in href:
        article_path = urlparse(href).path
        url = href
    else:
        return None

    article = article_path.removeprefix("/wiki/")
    if not article or ":" in article:
        return None
    return url


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

    tables = [table for table in _candidate_member_tables(soup) if _is_current_members_table(table)]
    countries: List[CountryInfo] = []

    for table in tables:
        for row in table.select("tr"):
            link = _country_link_from_row(row)
            if link is None:
                continue

            url = _country_url_from_href(str(link["href"]))
            if url is None:
                continue
            name = link.get_text(strip=True)
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
