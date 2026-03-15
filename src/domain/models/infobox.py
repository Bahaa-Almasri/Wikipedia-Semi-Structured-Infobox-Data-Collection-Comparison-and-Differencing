from __future__ import annotations

from dataclasses import dataclass
from typing import List


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
