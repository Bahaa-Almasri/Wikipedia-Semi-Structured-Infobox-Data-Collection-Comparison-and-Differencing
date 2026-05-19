"""
Semantic VSM preprocessing over Wikipedia comparison_fields.

Design goals
------------
Wikipedia infoboxes share a lot of *structural* markup (field names, years, ISO codes,
magnitude bins) that creates false TF-IDF similarity between unrelated countries.
We therefore:

1. Index **field:value** pairs built from cleaned infobox *content*, not bare field labels.
2. Drop noisy metadata paths (images, coordinates, reference years, codes).
3. Avoid abstract bins (mag_*, year_2020s, rank_*) that match template shape, not semantics.
4. Use coarse **semantic tiers** only for a few numeric domains (population, GDP, area, HDI).
5. Apply higher term weight on culturally/geopolitically meaningful fields.

Optional ``semantic_only`` mode restricts indexing to high-signal fields (languages,
religion, government, region, etc.) for exploratory queries.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from core.preprocess.comparison_content import get_comparison_fields

TermCounts = Dict[str, int]

_TOKEN_RE = re.compile(r"[0-9A-Za-z]+")
_RAW_DIGITS_RE = re.compile(r"^\d+$")
_YEAR_RE = re.compile(r"^(19|20)\d{2}$")
_WIKI_NOISE_RE = re.compile(
    r"\{\{|\}\}|<!--.*?-->|<[^>]+>|''+|&[a-z]+;|file:",
    re.IGNORECASE,
)
_WIKI_LINK_RE = re.compile(r"\[\[([^|\]]+)(?:\|[^\]]*)?\]\]")

# Coordinate strings are handled by vsm_geo (Haversine), not TF-IDF.
_COMPASS_TOKENS = frozenset({"n", "s", "e", "w", "ne", "nw", "se", "sw"})
_COORD_SYMBOL_TOKENS = frozenset(
    {"deg", "degree", "degrees", "min", "minute", "minutes", "sec", "second", "seconds"}
)

# Lexical noise common in infoboxes but not country semantics.
_VSM_STOP_TOKENS = frozenset(
    {
        "billion",
        "million",
        "thousand",
        "calling",
        "code",
        "coordinates",
        "coordinate",
        "dst",
        "demonym",
        "head",
        "state",
        "e",
        "n",
        "s",
        "w",
        "km",
        "sq",
        "utc",
        "iso",
        "url",
        "http",
        "https",
        "www",
        "org",
        "wiki",
        "wikipedia",
        "ref",
        "reference",
        "access",
        "archive",
        "web",
        "retrieved",
        "nominal",
        "per",
        "capita",
    }
)

# Values that repeat field/template vocabulary without semantic content.
_VALUE_BLACKLIST = frozenset(
    {
        "government",
        "currency",
        "state",
        "code",
        "rank",
        "lower",
        "upper",
        "total",
        "name",
        "value",
        "type",
        "list",
        "year",
        "years",
        "head",
        "demonym",
        "calling",
        "iso",
        "dst",
        "none",
        "frac",
        "true",
        "false",
    }
)

# Abstract bins removed from TF-IDF — they match infobox structure, not country identity.
_REJECTED_VALUE_PREFIXES = ("mag_", "year_", "rank_")

_MIN_VSM_TOKEN_LEN = 3
_FIELD_VALUE_WEIGHT = 3

# Paths we never index: weak metadata / presentation / citation scaffolding.
# Field names here are namespaces only — never tokenized as values (see path_segment_tokens).
_SKIP_PATH_MARKERS = (
    "image",
    "flag",
    "banner",
    "emblem",
    "coat_of_arms",
    "map",
    "locator",
    "symbol",
    "seal",
    "logo",
    "reference",
    "website",
    "url",
    "wikidata",
    "iso_",
    "_iso",
    "iso_code",
    "calling_code",
    "country_code",
    "internet_tld",
    "driving_side",
    "demonym",
    "coordinates",
    "coordinate",
    "latitude",
    "longitude",
    "_lat",
    "_lon",
    "time_zone",
    "timezone",
    "utc_offset",
    "utc",
    "dst",
    "footnotes",
    "empty_cell",
    "watermark",
    "commons",
    "signature",
)
_COORD_VALUE_RE = re.compile(r"^\d+[a-z]{1,2}$", re.IGNORECASE)

_YEAR_PATH_MARKERS = (
    "_year",
    ".year",
    "as_of",
    "established_date",
    "accessdate",
    "archive_date",
    "gdp_year",
    "gini_year",
    "hdi_year",
)

# High-signal fields for semantic_only mode.
_SEMANTIC_PATH_MARKERS = (
    "language",
    "religion",
    "ethnic",
    "government",
    "currency",
    "region",
    "continent",
    "capital",
    "neighbor",
    "neighbour",
    "economy",
    "gdp",
    "hdi",
    "gini",
    "population",
    "demographics",
)

# Term-frequency multipliers (higher = more influence on cosine similarity).
_FIELD_WEIGHTS: Dict[str, int] = {
    "languages_official": 4,
    "official_language": 4,
    "language": 4,
    "religion": 4,
    "ethnicity": 4,
    "ethnic_groups": 4,
    "government_type": 4,
    "government": 3,
    "region": 4,
    "continent": 4,
    "currency_name": 4,
    "currency": 2,
    "neighboring_countries": 4,
    "economy": 3,
    "gdp": 3,
    "hdi": 4,
    "gini": 2,
    "population": 2,
    "capital": 3,
    "area": 2,
}


def _feature_key(feature: str) -> str:
    clean = str(feature).strip().casefold()
    if clean.startswith("fields."):
        clean = clean.removeprefix("fields.")
    return clean.replace(".", "_")


def _path_key(path: str) -> str:
    return _feature_key(path.replace(".", "_"))


def should_skip_field_path(path: str, *, semantic_only: bool = False) -> bool:
    """Drop metadata/template paths before tokenization."""
    key = _path_key(path)
    if any(marker in key for marker in _SKIP_PATH_MARKERS):
        return True
    if key.endswith("_code") or key.split("_")[-1] == "code":
        return True
    if any(marker in key for marker in _YEAR_PATH_MARKERS):
        return True
    if "percent" in key and any(part in key for part in ("water", "area", "gini", "growth")):
        return True
    if semantic_only and not is_semantic_field_path(path):
        return True
    return False


def is_semantic_field_path(path: str) -> bool:
    """True for culturally/geopolitically meaningful comparison attributes."""
    key = _path_key(path)
    return any(marker in key for marker in _SEMANTIC_PATH_MARKERS)


def is_coordinate_feature(feature: str) -> bool:
    key = _feature_key(feature)
    return any(
        marker in key
        for marker in ("coordinate", "latitude", "longitude", "_lat", "_lon")
    )


def partition_features(
    features: Sequence[str],
) -> Tuple[List[str], List[str]]:
    coordinate_features: List[str] = []
    text_features: List[str] = []
    for feature in features:
        if is_coordinate_feature(feature):
            coordinate_features.append(feature)
        else:
            text_features.append(feature)
    return coordinate_features, text_features


def _is_coordinate_field_path(path: str) -> bool:
    return is_coordinate_feature(path)


def index_field_path(path: str) -> str:
    """
    Underscore field path used as the namespace in ``field_path:value`` tokens.

    The path is context only; it must never appear as the value side of a token.
    """
    normalized = str(path).strip().casefold()
    if normalized.startswith("fields."):
        normalized = normalized.removeprefix("fields.")
    parts = [part for part in re.split(r"[._]+", normalized) if part]
    if not parts:
        return normalized.replace(".", "_")
    return "_".join(parts)


def path_segment_tokens(path: str) -> frozenset[str]:
    """Lexical pieces of a field path — used to reject label-echo values."""
    field = index_field_path(path)
    return frozenset(piece for piece in field.split("_") if piece)


# Backward-compatible alias
semantic_field_name = index_field_path


def _value_echoes_field_path(value_token: str, path: str) -> bool:
    """
    True when the value is just the field name/label, not country content.

    Examples (path languages.official): discard ``languages``, ``official``,
    ``languages_official``, but keep ``arabic``.
    """
    clean = value_token.casefold().strip()
    if not clean:
        return True
    segments = path_segment_tokens(path)
    if clean in segments:
        return True
    parts = [part for part in clean.split("_") if part]
    if not parts:
        return True
    if all(part in segments for part in parts):
        return True
    if len(parts) == 1 and parts[0] in segments:
        return True
    field_key = index_field_path(path)
    if clean == field_key or clean.replace("_", "") == field_key.replace("_", ""):
        return True
    return False


def field_weight(path: str) -> int:
    name = index_field_path(path)
    best = 1
    for marker, weight in _FIELD_WEIGHTS.items():
        if marker in name:
            best = max(best, weight)
    return best


def _is_rejected_value_token(token: str) -> bool:
    clean = str(token).casefold().strip()
    if not clean:
        return True
    if clean in _VALUE_BLACKLIST:
        return True
    if any(clean.startswith(prefix) for prefix in _REJECTED_VALUE_PREFIXES):
        return True
    if clean in _COMPASS_TOKENS or clean in _COORD_SYMBOL_TOKENS:
        return True
    if clean in _VSM_STOP_TOKENS:
        return True
    if _COORD_VALUE_RE.match(clean):
        return True
    return False


def is_meaningful_vsm_token(token: str) -> bool:
    clean = str(token).casefold().strip()
    if not clean or _RAW_DIGITS_RE.match(clean):
        return False
    if _is_rejected_value_token(clean):
        return False
    if clean in {"high", "medium", "low"}:
        return True
    if len(clean) < _MIN_VSM_TOKEN_LEN:
        return False
    return True


def normalize_semantic_value(text: Any, *, field_path: str = "") -> str:
    """
    Collapse infobox *content* into one underscored value token.

    Strips wiki/template markup and drops words that only repeat the field path
    (e.g. path ``languages.official`` + value ``Official languages`` -> discarded).
    """
    if text is None:
        return ""
    raw = str(text)
    raw = _WIKI_LINK_RE.sub(r"\1", raw)
    raw = _WIKI_NOISE_RE.sub(" ", raw)
    raw = raw.replace("[[", " ").replace("]]", " ")
    path_segments = path_segment_tokens(field_path) if field_path else frozenset()
    parts = [
        match.group(0).casefold()
        for match in _TOKEN_RE.finditer(raw)
        if match.group(0) and not _RAW_DIGITS_RE.match(match.group(0))
    ]
    filtered: List[str] = []
    for part in parts:
        if part in path_segments:
            continue
        if _is_rejected_value_token(part):
            continue
        if is_meaningful_vsm_token(part):
            filtered.append(part)
    if not filtered:
        return ""
    value_token = "_".join(filtered)
    if _value_echoes_field_path(value_token, field_path):
        return ""
    return value_token


def is_meaningful_index_term(term: str) -> bool:
    """True when an indexing unit is a semantic field_path:value pair."""
    if ":" not in term:
        return False
    field, value = term.split(":", 1)
    if not field or not value:
        return False
    if value == "_field" or value == field or value == f"{field}s":
        return False
    if _is_rejected_value_token(value):
        return False
    if _value_echoes_field_path(value, field.replace("_", ".")):
        return False
    if value in {"high", "medium", "low"}:
        return True
    return is_meaningful_vsm_token(value)


def normalize_terms(text: Any) -> List[str]:
    """Tokenize free text queries (not used for infobox field indexing)."""
    if text is None:
        return []
    tokens = [match.group(0).casefold() for match in _TOKEN_RE.finditer(str(text))]
    return [
        token
        for token in tokens
        if not _RAW_DIGITS_RE.match(token) and is_meaningful_vsm_token(token)
    ]


def _add_count(target: TermCounts, term: str, amount: int = 1) -> None:
    if not term or _RAW_DIGITS_RE.match(term):
        return
    target[term] = target.get(term, 0) + amount


def _merge_counts(target: TermCounts, source: Mapping[str, int]) -> None:
    for term, count in source.items():
        _add_count(target, term, int(count))


def _qualify_term(path: str, value_token: str, *, mode: str) -> str:
    if mode != "field":
        raise ValueError(f"Unsupported VSM preprocessing mode '{mode}'.")
    if _is_rejected_value_token(value_token):
        return ""
    if _value_echoes_field_path(value_token, path):
        return ""
    field = index_field_path(path)
    return f"{field}:{value_token}"


def _add_index_term(
    counts: TermCounts,
    path: str,
    value_token: str,
    *,
    mode: str,
) -> None:
    qualified = _qualify_term(path, value_token, mode=mode)
    if not qualified:
        return
    weight = field_weight(path) * _FIELD_VALUE_WEIGHT
    _add_count(counts, qualified, weight)


def _hdi_tier(value: float) -> str:
    """Tier label only — field path already carries ``hdi`` context."""
    if value >= 0.8:
        return "high"
    if value >= 0.7:
        return "medium"
    return "low"


def _tier_label(value: float, thresholds: Sequence[Tuple[float, str]]) -> str:
    for bound, label in thresholds:
        if value >= bound:
            return label
    return thresholds[-1][1] if thresholds else "unknown"


def _population_tier(value: float) -> str:
    return _tier_label(
        value,
        (
            (100_000_000, "over_100m"),
            (10_000_000, "over_10m"),
            (1_000_000, "1m_to_10m"),
            (100_000, "100k_to_1m"),
            (0, "under_100k"),
        ),
    )


def _gdp_tier(value: float) -> str:
    return _tier_label(
        value,
        (
            (1_000_000_000_000, "over_1t"),
            (100_000_000_000, "over_100b"),
            (10_000_000_000, "over_10b"),
            (1_000_000_000, "over_1b"),
            (0, "under_1b"),
        ),
    )


def _area_tier(value: float) -> str:
    return _tier_label(
        value,
        (
            (1_000_000, "very_large"),
            (100_000, "large"),
            (10_000, "medium"),
            (0, "small"),
        ),
    )


def _numeric_semantic_token(path: str, value: float) -> Optional[str]:
    """
    Map selected numeric fields to coarse semantic tiers (not mag_* buckets).

    Years, ranks, and bare percentages are skipped — they dominate with template noise.
    """
    key = _path_key(path)
    if any(marker in key for marker in _YEAR_PATH_MARKERS) or key.endswith("_year"):
        return None
    if "rank" in key and "hdi" not in key:
        return None
    if "percent" in key:
        return None

    if "hdi" in key and 0 < value <= 1:
        return _hdi_tier(value)
    if "population" in key:
        return _population_tier(value)
    if "gdp" in key or ("economy" in key and "gdp" in key):
        return _gdp_tier(value)
    if "area" in key and "percent" not in key:
        return _area_tier(value)
    return None


def _add_semantic_value(
    counts: TermCounts,
    path: str,
    value: Any,
    *,
    mode: str,
) -> None:
    value_token = normalize_semantic_value(value, field_path=path)
    if value_token:
        _add_index_term(counts, path, value_token, mode=mode)


def encode_comparison_value(
    path: str,
    value: Any,
    *,
    mode: str,
    semantic_only: bool = False,
) -> TermCounts:
    """Turn one comparison_fields leaf into weighted field:value TF counts."""
    counts: TermCounts = {}
    field_path = path.strip()
    clean_path = field_path.replace(".", "_")

    if should_skip_field_path(field_path, semantic_only=semantic_only):
        return counts

    if value is None:
        return counts

    if isinstance(value, bool):
        return counts

    if isinstance(value, int):
        if _YEAR_RE.match(str(value)):
            return counts
        token = _numeric_semantic_token(field_path, float(value))
        if token:
            _add_index_term(counts, field_path, token, mode=mode)
        return counts

    if isinstance(value, float):
        token = _numeric_semantic_token(field_path, value)
        if token:
            _add_index_term(counts, field_path, token, mode=mode)
        return counts

    if isinstance(value, list):
        for item in value:
            _merge_counts(
                counts,
                encode_comparison_value(
                    field_path,
                    item,
                    mode=mode,
                    semantic_only=semantic_only,
                ),
            )
        return counts

    if isinstance(value, dict):
        for key, child in value.items():
            child_path = (
                f"{field_path}.{_feature_key(str(key))}" if field_path else _feature_key(str(key))
            )
            _merge_counts(
                counts,
                encode_comparison_value(
                    child_path,
                    child,
                    mode=mode,
                    semantic_only=semantic_only,
                ),
            )
        return counts

    text = str(value).strip()
    if not text:
        return counts

    if _is_coordinate_field_path(clean_path):
        return counts

    _add_semantic_value(counts, field_path, text, mode=mode)
    return counts


def count_meaningful_index_terms(terms: Mapping[str, int]) -> int:
    return sum(1 for term in terms if is_meaningful_index_term(term))


def iter_comparison_indexing_nodes(
    value: Any,
    *,
    path: Tuple[str, ...] = (),
) -> Iterable[Tuple[str, Any]]:
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
    semantic_only: bool = False,
) -> Dict[str, TermCounts]:
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
        counts = encode_comparison_value(
            node_path,
            leaf_value,
            mode=mode,
            semantic_only=semantic_only,
        )
        existing = field_terms.setdefault(node_path, {})
        _merge_counts(existing, counts)

    return field_terms


def merge_indexing_nodes(
    field_terms: Mapping[str, TermCounts],
) -> TermCounts:
    merged: TermCounts = {}
    for counts in field_terms.values():
        _merge_counts(merged, counts)
    return merged
