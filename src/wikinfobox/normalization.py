from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Mapping

from .infobox_parser import InfoboxRow


_key_non_alnum_re = re.compile(r"[^0-9a-zA-Z]+")
_number_re = re.compile(r"[-+]?(?:\d{1,3}(?:[,\u00A0]\d{3})+|\d+)(?:\.\d+)?")  # basic numbers


@dataclass(frozen=True)
class NormalizedField:
    raw_label: str
    text: str
    tokens: List[str]
    numbers: List[float]


def normalize_key(label: str) -> str:
    s = label.strip().strip(":")
    s = s.lower()
    s = _key_non_alnum_re.sub("_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s or "field"


def _tokenize(text: str) -> List[str]:
    # Split on any non-alphanumeric character.
    tokens = re.split(r"[^0-9A-Za-z]+", text)
    return [t for t in tokens if t]


def _parse_numbers(text: str) -> List[float]:
    results: List[float] = []
    for match in _number_re.finditer(text):
        token = match.group(0)
        cleaned = token.replace(",", "").replace("\u00A0", "")
        try:
            if "." in cleaned:
                results.append(float(cleaned))
            else:
                results.append(int(cleaned))
        except ValueError:
            continue
    return results


def normalize_rows(rows: List[InfoboxRow]) -> Mapping[str, NormalizedField]:
    """
    Convert raw infobox rows into a mapping of normalized_key -> NormalizedField.

    If multiple rows normalize to the same key, the last one wins for simplicity.
    """
    result: Dict[str, NormalizedField] = {}

    for row in rows:
        key = normalize_key(row.label)
        text = row.value_text.strip()
        tokens = _tokenize(text)
        numbers = _parse_numbers(text)

        result[key] = NormalizedField(
            raw_label=row.label,
            text=text,
            tokens=tokens,
            numbers=numbers,
        )

    return result


def normalized_fields_to_dict(fields: Mapping[str, NormalizedField]) -> Dict[str, dict]:
    """
    Helper to convert NormalizedField objects into JSON-serializable dicts.
    """
    return {k: asdict(v) for k, v in fields.items()}

