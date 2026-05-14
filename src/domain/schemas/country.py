from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CountryInfo:
    name: str
    wikipedia_url: str
    slug: str
