"""
Semi-structured VSM preprocessing over comparison_fields.

Before TF-IDF we treat each comparison attribute as an *indexing node* and build
term-context indexing units. Raw numeric tokens are not indexed; structured
numbers use magnitude / category bins so similar scales cluster together.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from core.preprocess.comparison_content import get_comparison_fields

TermCounts = Dict[str, int]

_TOKEN_RE = re.compile(r"[0-9A-Za-z]+")
_RAW_DIGITS_RE = re.compile(r"^\d+$")
_YEAR_RE = re.compile(r"^(19|20)\d{2}$")


def normalize_terms(text: Any) -> List[str]:
    """Lexical tokens for VSM (case-folded); excludes tokens that are only digits."""
    if text is None:
        return []
    tokens = [match.group(0).casefold() for match in _TOKEN_RE.finditer(str(text))]
    return [token for token in tokens if not _RAW_DIGITS_RE.match(token)]


def _feature_key(feature: str) -> str:
    clean = str(feature).strip().casefold()
    if clean.startswith("fields."):
        clean = clean.removeprefix("fields.")
    return clean.replace(".", "_")


def _path_label_terms(path: str) -> List[str]:
    pieces = re.split(r"[^0-9A-Za-z]+", path.replace(".", "_"))
    return [piece.casefold() for piece in pieces if piece and not _RAW_DIGITS_RE.match(piece)]


def _add_count(target: TermCounts, term: str, amount: int = 1) -> None:
    if not term or _RAW_DIGITS_RE.match(term):
        return
    target[term] = target.get(term, 0) + amount


def _merge_counts(target: TermCounts, source: Mapping[str, int]) -> None:
    for term, count in source.items():
        _add_count(target, term, int(count))


def _qualify_term(path: str, term: str, *, mode: str) -> str:
    if mode != "field":
        raise ValueError(f"Unsupported VSM preprocessing mode '{mode}'.")
    clean_path = _feature_key(path.replace(".", "_"))
    clean_term = term.casefold().strip()
    if not clean_term or _RAW_DIGITS_RE.match(clean_term):
        return ""
    return f"{clean_path}:{clean_term}"


def _magnitude_bin(value: float) -> str:
    """Order-of-magnitude bucket for comparable numeric fields (not raw digits)."""
    if value is None or value <= 0:
        return "none"
    if value < 1:
        return "frac"
    exponent = int(math.floor(math.log10(value)))
    return f"mag_{exponent}"


def _year_bin(value: int) -> str:
    decade = (int(value) // 10) * 10
    return f"year_{decade}s"


def _rank_bin(rank: int) -> str:
    if rank <= 25:
        return "rank_top25"
    if rank <= 100:
        return "rank_top100"
    return "rank_lower"


def _hdi_bin(value: float) -> str:
    if value >= 0.8:
        return "hdi_high"
    if value >= 0.7:
        return "hdi_medium"
    return "hdi_low"


def encode_comparison_value(path: str, value: Any, *, mode: str) -> TermCounts:
    """Turn one comparison_fields leaf into local TF counts for an indexing node."""
    counts: TermCounts = {}
    clean_path = path.strip().replace(".", "_")

    if value is None:
        return counts

    if isinstance(value, bool):
        _add_count(counts, _qualify_term(clean_path, str(value).lower(), mode=mode))
        return counts

    if isinstance(value, int):
        if _YEAR_RE.match(str(value)):
            term = _year_bin(value)
        elif "rank" in clean_path and value < 10_000:
            term = _rank_bin(value)
        else:
            term = _magnitude_bin(float(value))
        _add_count(counts, _qualify_term(clean_path, term, mode=mode))
        return counts

    if isinstance(value, float):
        if "hdi" in clean_path and 0 < value <= 1:
            term = _hdi_bin(value)
        else:
            term = _magnitude_bin(value)
        _add_count(counts, _qualify_term(clean_path, term, mode=mode))
        return counts

    if isinstance(value, list):
        for item in value:
            _merge_counts(counts, encode_comparison_value(path, item, mode=mode))
        return counts

    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{_feature_key(str(key))}" if path else _feature_key(str(key))
            _merge_counts(counts, encode_comparison_value(child_path, child, mode=mode))
        return counts

    text = str(value).strip()
    if not text:
        return counts

    for label in _path_label_terms(clean_path):
        _add_count(counts, _qualify_term(clean_path, label, mode=mode))

    for token in normalize_terms(text):
        _add_count(counts, _qualify_term(clean_path, token, mode=mode))

    return counts


def iter_comparison_indexing_nodes(
    value: Any,
    *,
    path: Tuple[str, ...] = (),
) -> Iterable[Tuple[str, Any]]:
    """Yield (context_path, leaf_value) pairs — one indexing node per leaf."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield from iter_comparison_indexing_nodes(child, path=(*path, str(key)))
        return
    if isinstance(value, list):
        if not value:
            yield (".".join(path), value)
            return
        for item in value:
            if isinstance(item, (dict, list)):
                yield from iter_comparison_indexing_nodes(item, path=path)
            else:
                yield (".".join(path), item)
        return
    yield (".".join(path), value)


def build_indexing_node_terms(
    document: Mapping[str, Any],
    *,
    mode: str,
    features: Optional[Sequence[str]] = None,
) -> Dict[str, TermCounts]:
    """
    Build indexing-node term counts from comparison_fields (same payload as TED trees).

    Returns field_terms[path] = local term counts for that node before document merge.
    """
    comparison_fields = get_comparison_fields(document)
    field_terms: Dict[str, TermCounts] = {}

    def _feature_matches(field_path: str) -> bool:
        if not features:
            return True
        field_key = _feature_key(field_path)
        for feature in features:
            feature_key = _feature_key(feature)
            if field_key == feature_key:
                return True
            if field_key.endswith(f"_{feature_key}") or feature_key.endswith(f"_{field_key}"):
                return True
            if field_key.split("_")[-1] == feature_key.split("_")[-1]:
                return True
        return False

    for node_path, leaf_value in iter_comparison_indexing_nodes(comparison_fields):
        if not node_path or not _feature_matches(node_path):
            continue
        counts = encode_comparison_value(node_path, leaf_value, mode=mode)
        existing = field_terms.setdefault(node_path, {})
        _merge_counts(existing, counts)

    return field_terms


def merge_indexing_nodes(
    field_terms: Mapping[str, TermCounts],
) -> TermCounts:
    """Multi-category document vector: sum TF across all indexing nodes."""
    merged: TermCounts = {}
    for counts in field_terms.values():
        _merge_counts(merged, counts)
    return merged
