"""
Geographic similarity for VSM feature restriction on coordinate fields.

Coordinate strings are not tokenized for TF-IDF; instead we parse decimal
latitude/longitude and rank countries by great-circle distance.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from core.preprocess.comparison_content import get_comparison_fields
from core.similarity.vsm_preprocessing import (
    _feature_key,
    iter_comparison_indexing_nodes,
    is_coordinate_feature,
)
from domain.models.vsm import VSMSearchResult

_COORD_COMPONENT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*°(?:\s*(\d+(?:\.\d+)?)\s*['′])?\s*([nsew]{1,2})",
    re.IGNORECASE,
)

_EARTH_RADIUS_KM = 6371.0
_COORD_SIMILARITY_SCALE_KM = 1000.0

LatLon = Tuple[float, float]


def parse_coordinate_component(text: Any) -> Optional[float]:
    """Parse a coordinate component such as 33°N or 35°12′E into decimal degrees."""
    raw = str(text or "").strip()
    if not raw:
        return None
    match = _COORD_COMPONENT_RE.search(raw)
    if not match:
        return None
    degrees = float(match.group(1))
    minutes = float(match.group(2)) if match.group(2) else 0.0
    direction = match.group(3).casefold()
    decimal = degrees + minutes / 60.0
    if "s" in direction:
        decimal = -abs(decimal)
    if "w" in direction:
        decimal = -abs(decimal)
    return decimal


def _path_matches_coordinate_features(field_path: str, features: Optional[Sequence[str]]) -> bool:
    if not features:
        return _is_coordinate_path(field_path)
    field_key = _feature_key(field_path)
    for feature in features:
        if not is_coordinate_feature(feature):
            continue
        feature_key = _feature_key(feature)
        if field_key == feature_key:
            return True
        if field_key.endswith(f"_{feature_key}") or feature_key.endswith(f"_{field_key}"):
            return True
        if field_key.split("_")[-1] == feature_key.split("_")[-1]:
            return True
    return False


def _is_coordinate_path(path: str) -> bool:
    key = _feature_key(path)
    return any(
        marker in key
        for marker in ("coordinate", "latitude", "longitude", "_lat", "_lon")
    )


def _is_latitude_path(path: str) -> bool:
    key = _feature_key(path)
    return "latitude" in key or key.endswith("_lat")


def _is_longitude_path(path: str) -> bool:
    key = _feature_key(path)
    return "longitude" in key or key.endswith("_lon")


def extract_decimal_coordinates(
    document: Mapping[str, Any],
    *,
    coordinate_features: Optional[Sequence[str]] = None,
) -> Optional[LatLon]:
    """
    Extract decimal (latitude, longitude) from comparison_fields.

    When coordinate_features is set, only paths matching those selections are used.
    """
    comparison_fields = get_comparison_fields(document)
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    for node_path, leaf_value in iter_comparison_indexing_nodes(comparison_fields):
        if not node_path or not _is_coordinate_path(node_path):
            continue
        if coordinate_features and not _path_matches_coordinate_features(
            node_path, coordinate_features
        ):
            continue
        if _is_latitude_path(node_path):
            parsed = parse_coordinate_component(leaf_value)
            if parsed is not None:
                latitude = parsed
        elif _is_longitude_path(node_path):
            parsed = parse_coordinate_component(leaf_value)
            if parsed is not None:
                longitude = parsed

    if latitude is None or longitude is None:
        return None
    return latitude, longitude


def haversine_distance_km(left: LatLon, right: LatLon) -> float:
    """Great-circle distance between two decimal-degree points."""
    lat1, lon1 = (math.radians(v) for v in left)
    lat2, lon2 = (math.radians(v) for v in right)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return _EARTH_RADIUS_KM * c


def distance_to_similarity(distance_km: float) -> float:
    """
    Map geographic distance to a 0–1 similarity score.

    Linear decay: 0 km -> 1.0, 100 km -> 0.9, 1000+ km -> 0.0.
    Ranking still uses raw Haversine distance (closest first).
    """
    if distance_km < 0:
        distance_km = 0.0
    return max(0.0, 1.0 - distance_km / _COORD_SIMILARITY_SCALE_KM)


def build_country_coordinates(
    documents: Mapping[str, Mapping[str, Any]],
    *,
    coordinate_features: Optional[Sequence[str]] = None,
) -> Dict[str, LatLon]:
    """Build slug -> (lat, lon) for all documents with parseable coordinates."""
    coordinates: Dict[str, LatLon] = {}
    for slug, document in documents.items():
        point = extract_decimal_coordinates(
            document,
            coordinate_features=coordinate_features,
        )
        if point is not None:
            coordinates[slug] = point
    return coordinates


def _distance_label(distance_km: float) -> str:
    if distance_km < 1:
        return f"distance: {distance_km:.1f} km"
    return f"distance: {int(round(distance_km)):,} km"


def rank_by_geographic_proximity(
    coordinates: Mapping[str, LatLon],
    source_slug: str,
    *,
    top_k: int = 5,
    display_names: Optional[Mapping[str, str]] = None,
) -> List[VSMSearchResult]:
    """Rank countries by geographic proximity to the source country."""
    source_point = coordinates.get(source_slug)
    if source_point is None:
        raise ValueError(f"No parseable coordinates for country: {source_slug}")

    results: List[VSMSearchResult] = []
    for slug, point in coordinates.items():
        if slug == source_slug:
            continue
        distance_km = haversine_distance_km(source_point, point)
        score = distance_to_similarity(distance_km)
        label = _distance_label(distance_km)
        results.append(
            VSMSearchResult(
                slug=slug,
                display_name=(display_names or {}).get(
                    slug, slug.replace("_", " ").title()
                ),
                score=score,
                matched_terms=[f"{label}, coordinate proximity"],
            )
        )

    results.sort(key=lambda item: (-item.score, item.matched_terms[0]))
    return results[:top_k]


def merge_ranking_scores(
    *rankings: Sequence[VSMSearchResult],
    top_k: int,
) -> List[VSMSearchResult]:
    """
    Combine multiple ranked lists by averaging scores per country.

    matched_terms from each list are merged (geo distance labels first).
    """
    combined: Dict[str, Dict[str, Any]] = {}
    for ranking in rankings:
        for item in ranking:
            entry = combined.setdefault(
                item.slug,
                {
                    "display_name": item.display_name,
                    "scores": [],
                    "matched_terms": [],
                },
            )
            entry["scores"].append(item.score)
            for term in item.matched_terms:
                if term not in entry["matched_terms"]:
                    entry["matched_terms"].append(term)

    merged: List[VSMSearchResult] = []
    for slug, entry in combined.items():
        scores = entry["scores"]
        if not scores:
            continue
        merged.append(
            VSMSearchResult(
                slug=slug,
                display_name=entry["display_name"],
                score=sum(scores) / len(scores),
                matched_terms=entry["matched_terms"][:10],
            )
        )

    merged.sort(key=lambda item: item.score, reverse=True)
    return merged[:top_k]
