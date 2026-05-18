#!/usr/bin/env python
"""Diagnostics for semi-structured VSM preprocessing."""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

from core.preprocess.comparison_content import get_comparison_fields  # noqa: E402
from core.similarity.vsm_preprocessing import (  # noqa: E402
    build_indexing_node_terms,
    encode_comparison_value,
    merge_indexing_nodes,
    normalize_terms,
)


def test_normalize_terms_skips_raw_digits():
    assert normalize_terms("Population 5,000,000 in 2024") == ["population", "in"]


def test_comparison_population_uses_magnitude_not_digits():
    counts = encode_comparison_value("population.total", 6_825_448, mode="field")
    assert "population_total:mag_6" in counts
    assert "6825448" not in counts
    assert "000" not in counts


def test_comparison_government_type_is_contextual():
    counts = encode_comparison_value(
        "government.type",
        "Federal parliamentary republic",
        mode="field",
    )
    assert "government_type:federal" in counts
    assert "government_type:republic" in counts


def test_field_mode_uses_path_colon_term():
    document = {
        "normalized": {
            "comparison_fields": {
                "population": {"total": 5_000_000, "year": 2020},
                "capital": "Beirut",
            }
        }
    }
    nodes = build_indexing_node_terms(document, mode="field")
    merged = merge_indexing_nodes(nodes)
    assert "population_total:mag_6" in merged or "population.total:mag_6" in merged
    assert "capital:beirut" in merged
    assert "5" not in merged


def test_get_comparison_fields_rebuilds_from_normalized_fields():
    document = {
        "meta": {"country_name": "Example"},
        "normalized": {
            "fields": {
                "capital": {
                    "raw_label": "Capital",
                    "text": "Beirut",
                    "tokens": ["Beirut"],
                    "numbers": [],
                }
            }
        },
    }
    payload = get_comparison_fields(document)
    assert payload.get("capital") == "Beirut"


def test_build_indexing_node_terms_rejects_non_context_mode():
    document = {
        "normalized": {
            "comparison_fields": {"capital": "Beirut"},
        }
    }
    try:
        build_indexing_node_terms(document, mode="flat")
    except ValueError as exc:
        assert "Unsupported VSM preprocessing mode" in str(exc)
    else:
        raise AssertionError("Expected flat preprocessing mode to be rejected")


if __name__ == "__main__":
    test_normalize_terms_skips_raw_digits()
    test_comparison_population_uses_magnitude_not_digits()
    test_comparison_government_type_is_contextual()
    test_field_mode_uses_path_colon_term()
    test_get_comparison_fields_rebuilds_from_normalized_fields()
    test_build_indexing_node_terms_rejects_non_context_mode()
    print("All VSM preprocessing diagnostic tests passed")
