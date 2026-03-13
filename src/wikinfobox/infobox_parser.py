from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from bs4 import BeautifulSoup, Tag


@dataclass(frozen=True)
class InfoboxRow:
    """
    One logical label/value pair extracted from an infobox.

    - label: cleaned header text (citations stripped, whitespace normalized)
    - value_html: raw HTML of the value cell (for maximum fidelity)
    - value_text: cleaned text value (no HTML tags, citations stripped)
    """

    label: str
    value_html: str
    value_text: str


@dataclass(frozen=True)
class ParsedInfobox:
    html: str
    rows: List[InfoboxRow]


def _extract_infobox(soup: BeautifulSoup) -> Optional[Tag]:
    # Wikipedia infobox tables have class "infobox" (often with extra classes).
    infobox = soup.find("table", class_="infobox")
    return infobox


def _clean_cell_text(tag: Tag) -> str:
    """
    Convert a header/data cell into cleaned text:
    - strip citation markers (e.g. [1], [a])
    - remove all HTML tags
    - normalize whitespace
    """
    # Drop citation superscripts like <sup class="reference">[1]</sup>
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

    # Many country pages wrap rows in <tbody>, so look there first.
    container = table.find("tbody") or table

    rows: List[InfoboxRow] = []
    current_section: Optional[str] = None
    current_section_source: Optional[str] = None  # "header" | "inline"

    for tr in container.find_all("tr", recursive=False):
        # Ignore explicitly hidden rows
        if tr.get("style") and "display:none" in tr.get("style", ""):
            continue

        th = tr.find("th")
        td = tr.find("td")

        # Section header rows are typically <th class="infobox-header" colspan="2">...</th>
        # Avoid treating the top "infobox-above" title row as a section header.
        if th is not None and td is None:
            th_classes = th.get("class") or []
            if "infobox-header" in th_classes:
                th_text = _clean_cell_text(th)
                if th_text:
                    current_section = th_text
                    current_section_source = "header"
            continue

        if th is None or td is None:
            # Skip rows without a label+value pair (images, maps, etc.)
            continue

        value_html = str(td)
        label_text = _clean_cell_text(th)
        value_text = _clean_cell_text(td)

        if not label_text or not value_text:
            continue

        tr_classes = tr.get("class") or []

        # Some sections (GDP, Time zone) are represented as a mergedtoprow label+value row
        # followed by bullet sub-rows ("• Total", "• Per capita", "• Summer (DST)").
        if "mergedtoprow" in tr_classes and (label_text.startswith("GDP") or label_text == "Time zone"):
            current_section = label_text
            current_section_source = "inline"

        # Disambiguate bullet sub-rows like "• Total" by prefixing the current section.
        # This avoids key collisions across sections (Area/Population/GDP all have "Total").
        if current_section and label_text.strip().startswith("•"):
            label_text = f"{current_section} {label_text}"
        elif current_section_source == "inline" and label_text.strip().startswith("•") is False:
            # Inline sections (GDP, Time zone) should only apply to their immediate bullet rows.
            # Reset when we hit the next non-bullet row.
            current_section = None
            current_section_source = None

        rows.append(InfoboxRow(label=label_text, value_html=value_html, value_text=value_text))

    return ParsedInfobox(html=str(table), rows=rows)
