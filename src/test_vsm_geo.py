#!/usr/bin/env python
"""Diagnostics for geographic VSM coordinate ranking."""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

from core.similarity.vsm_geo import (  # noqa: E402
    build_country_coordinates,
    distance_to_similarity,
    extract_decimal_coordinates,
    haversine_distance_km,
    parse_coordinate_component,
    rank_by_geographic_proximity,
)


def _coord_doc(country_name: str, lat: str, lon: str) -> dict:
    return {
        "meta": {"country_name": country_name},
        "normalized": {
            "comparison_fields": {
                "coordinates": {"latitude": lat, "longitude": lon},
            }
        },
    }


def test_parse_coordinate_component_decimal_degrees():
    assert parse_coordinate_component("33°N") == 33.0
    assert parse_coordinate_component("35°12′E") == 35.2
    assert parse_coordinate_component("35°S") == -35.0


def test_extract_decimal_coordinates():
    doc = _coord_doc("Lebanon", "33°N", "35°E")
    assert extract_decimal_coordinates(doc) == (33.0, 35.0)


def test_distance_to_similarity_normalized_scale():
    assert distance_to_similarity(0) == 1.0
    assert distance_to_similarity(100) == 0.9
    assert distance_to_similarity(250) == 0.75
    assert distance_to_similarity(1000) == 0.0
    assert distance_to_similarity(5000) == 0.0


def test_lebanon_geographic_ranking_prefers_nearby_countries():
    docs = {
        "lebanon": _coord_doc("Lebanon", "33°N", "35°E"),
        "syria": _coord_doc("Syria", "34°N", "38°E"),
        "japan": _coord_doc("Japan", "36°N", "138°E"),
    }
    coordinates = build_country_coordinates(docs)
    results = rank_by_geographic_proximity(
        coordinates,
        "lebanon",
        top_k=2,
        display_names={"lebanon": "Lebanon", "syria": "Syria", "japan": "Japan"},
    )
    assert results[0].slug == "syria"
    assert "distance:" in results[0].matched_terms[0]
    assert "coordinate proximity" in results[0].matched_terms[0]
    assert haversine_distance_km(coordinates["lebanon"], coordinates["syria"]) < 500


if __name__ == "__main__":
    test_parse_coordinate_component_decimal_degrees()
    test_extract_decimal_coordinates()
    test_distance_to_similarity_normalized_scale()
    test_lebanon_geographic_ranking_prefers_nearby_countries()
    print("All VSM geographic diagnostic tests passed")
