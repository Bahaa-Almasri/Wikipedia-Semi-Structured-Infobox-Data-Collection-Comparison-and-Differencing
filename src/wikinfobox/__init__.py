"""
Wikipedia country infobox data collection and preprocessing (Phase 1).

This package:
- Fetches the list of UN member states from Wikipedia
- Scrapes each country's infobox
- Normalizes keys/values
- Stores each country as a standalone JSON document
"""

__all__ = [
    "config",
    "country_list",
    "fetch",
    "infobox_parser",
    "normalization",
    "storage",
    "pipeline",
]

