from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class NormalizedField:
    raw_label: str
    text: str
    tokens: List[str]
    numbers: List[float]
