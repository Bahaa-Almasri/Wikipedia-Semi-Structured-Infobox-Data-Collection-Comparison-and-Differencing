from __future__ import annotations

from typing import List, Optional

from bs4 import BeautifulSoup, Tag

from domain.models.infobox import InfoboxRow, ParsedInfobox


def _extract_infobox(soup: BeautifulSoup) -> Optional[Tag]:
    infobox = soup.find("table", class_="infobox")
    return infobox


def _clean_cell_text(tag: Tag) -> str:
    for sup in tag.select("sup.reference"):
        sup.decompose()
    return tag.get_text(" ", strip=True)


def parse_infobox(html: str) -> Optional[ParsedInfobox]:
    """
    Parse the first infobox table from a Wikipedia country page HTML.

    Returns ParsedInfobox or None if no infobox is found.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = _extract_infobox(soup)
    if table is None:
        return None

    container = table.find("tbody") or table

    rows: List[InfoboxRow] = []
    current_section: Optional[str] = None
    current_section_source: Optional[str] = None

    for tr in container.find_all("tr", recursive=False):
        if tr.get("style") and "display:none" in tr.get("style", ""):
            continue

        th = tr.find("th")
        td = tr.find("td")

        if th is not None and td is None:
            th_classes = th.get("class") or []
            if "infobox-header" in th_classes:
                th_text = _clean_cell_text(th)
                if th_text:
                    current_section = th_text
                    current_section_source = "header"
            continue

        if th is None or td is None:
            continue

        value_html = str(td)
        label_text = _clean_cell_text(th)
        value_text = _clean_cell_text(td)

        if not label_text or not value_text:
            continue

        if label_text.startswith("GDP") or label_text == "Time zone":
            current_section = label_text
            current_section_source = "inline"

        if current_section and label_text.strip().startswith("•"):
            label_text = f"{current_section} {label_text}"
        elif current_section_source == "inline" and not label_text.strip().startswith("•"):
            current_section = None
            current_section_source = None

        rows.append(InfoboxRow(label=label_text, value_html=value_html, value_text=value_text))

    return ParsedInfobox(html=str(table), rows=rows)
