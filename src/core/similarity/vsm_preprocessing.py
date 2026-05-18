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
# Match a single coordinate component such as 33°N or 35°12′E as one unit.
_COORD_COMPONENT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*°(?:\s*\d+(?:\.\d+)?\s*['′])?\s*([nsew]{1,2})",
    re.IGNORECASE,
)

# Values like "33°N" are split by the alphanumeric tokenizer into "33" (dropped)
# and "n"/"e", which appear in most northern/eastern countries and inflate cosine
# similarity. Coordinate paths are excluded here; see vsm_geo for distance ranking.
_COMPASS_TOKENS = frozenset({"n", "s", "e", "w", "ne", "nw", "se", "sw"})
_COORD_SYMBOL_TOKENS = frozenset(
    {"deg", "degree", "degrees", "min", "minute", "minutes", "sec", "second", "seconds"}
)
_MIN_VSM_TOKEN_LEN = 3
_SEMANTIC_BIN_PREFIXES = ("mag_", "year_", "rank_", "hdi_", "none", "frac")


def is_meaningful_vsm_token(token: str) -> bool:
    """Return True when a token should participate in VSM similarity."""
    clean = str(token).casefold().strip()
    if not clean or _RAW_DIGITS_RE.match(clean):
        return False
    if any(clean.startswith(prefix) for prefix in _SEMANTIC_BIN_PREFIXES):
        return True
    if clean in _COMPASS_TOKENS or clean in _COORD_SYMBOL_TOKENS:
        return False
    if len(clean) < _MIN_VSM_TOKEN_LEN:
        return False
    return True


def normalize_terms(text: Any) -> List[str]:
    """Lexical tokens for VSM (case-folded); excludes digits and low-signal tokens."""
    if text is None:
        return []
    tokens = [match.group(0).casefold() for match in _TOKEN_RE.finditer(str(text))]
    return [
        token
        for token in tokens
        if not _RAW_DIGITS_RE.match(token) and is_meaningful_vsm_token(token)
    ]


def _is_coordinate_field_path(path: str) -> bool:
    return is_coordinate_feature(path)


def is_coordinate_feature(feature: str) -> bool:
    """True when a feature path refers to geographic coordinates."""
    key = _feature_key(feature)
    return any(
        marker in key
        for marker in ("coordinate", "latitude", "longitude", "_lat", "_lon")
    )


def partition_features(
    features: Sequence[str],
) -> Tuple[List[str], List[str]]:
    """Split selected features into coordinate vs text fields for hybrid VSM."""
    coordinate_features: List[str] = []
    text_features: List[str] = []
    for feature in features:
        if is_coordinate_feature(feature):
            coordinate_features.append(feature)
        else:
            text_features.append(feature)
    return coordinate_features, text_features


def _feature_key(feature: str) -> str:
    clean = str(feature).strip().casefold()
    if clean.startswith("fields."):
        clean = clean.removeprefix("fields.")
    return clean.replace(".", "_")


def _path_label_terms(path: str) -> List[str]:
    pieces = re.split(r"[^0-9A-Za-z]+", path.replace(".", "_"))
    return [
        piece.casefold()
        for piece in pieces
        if piece and not _RAW_DIGITS_RE.match(piece) and is_meaningful_vsm_token(piece)
    ]


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
    if not is_meaningful_vsm_token(clean_term):
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

    # Coordinates are compared geographically (see vsm_geo), not via TF-IDF tokens.
    if _is_coordinate_field_path(clean_path):
        return counts

    for label in _path_label_terms(clean_path):
        _add_count(counts, _qualify_term(clean_path, label, mode=mode))

    for token in normalize_terms(text):
        _add_count(counts, _qualify_term(clean_path, token, mode=mode))

    return counts


def count_meaningful_index_terms(terms: Mapping[str, int]) -> int:
    """Count distinct meaningful leaf terms in a document term map."""
    meaningful: set[str] = set()
    for term in terms:
        leaf = term.split(":", 1)[-1] if ":" in term else term
        if is_meaningful_vsm_token(leaf):
            meaningful.add(leaf)
    return len(meaningful)


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
