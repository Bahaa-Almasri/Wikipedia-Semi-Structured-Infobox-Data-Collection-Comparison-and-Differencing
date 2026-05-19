#!/usr/bin/env python
"""Diagnostics for semantic VSM preprocessing."""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

from core.preprocess.comparison_content import get_comparison_fields  # noqa: E402
from core.similarity.vsm_preprocessing import (  # noqa: E402
    build_indexing_node_terms,
    encode_comparison_value,
    is_meaningful_index_term,
    is_meaningful_vsm_token,
    merge_indexing_nodes,
    normalize_semantic_value,
    normalize_terms,
    index_field_path,
    should_skip_field_path,
)


def test_normalize_terms_skips_raw_digits():
    assert normalize_terms("Population 5,000,000 in 2024") == ["population"]


def test_normalize_terms_filters_compass_direction_tokens():
    assert normalize_terms("33°N 35°E") == []
    assert not is_meaningful_vsm_token("n")
    assert not is_meaningful_vsm_token("e")


def test_coordinate_fields_are_excluded_from_tfidf_tokenization():
    counts = encode_comparison_value("coordinates.latitude", "33°N", mode="field")
    assert counts == {}


def test_year_and_mag_bins_are_not_indexed():
    counts = encode_comparison_value("economy.gdp.nominal.year", 2020, mode="field")
    assert counts == {}
    counts = encode_comparison_value("area.water.percent", 2.8, mode="field")
    assert counts == {}
    assert not is_meaningful_index_term("economy:mag_1")


def test_population_uses_semantic_tier_not_mag_bin():
    counts = encode_comparison_value("population.total", 6_825_448, mode="field")
    assert "population_total:1m_to_10m" in counts
    assert "population_total:mag_6" not in counts


def test_government_type_uses_full_field_value_pair():
    counts = encode_comparison_value(
        "government.type",
        "Unitary parliamentary republic",
        mode="field",
    )
    assert "government_type:unitary_parliamentary_republic" in counts
    assert "government_type:government" not in counts
    assert "government_type:unitary" not in counts or "parliamentary" in "unitary_parliamentary_republic"


def test_field_label_text_is_not_indexed_as_value():
    counts = encode_comparison_value(
        "languages.official",
        "Official languages",
        mode="field",
    )
    assert "languages_official:arabic" not in counts
    assert "languages_official:languages" not in counts
    assert "languages_official:official" not in counts
    assert not counts

    counts = encode_comparison_value("languages.official", "Arabic", mode="field")
    assert "languages_official:arabic" in counts


def test_time_zone_and_coordinate_paths_are_skipped():
    assert encode_comparison_value("time_zone.dst", "UTC+2", mode="field") == {}
    assert encode_comparison_value("coordinates.latitude", "33°N", mode="field") == {}
    assert normalize_semantic_value("33N", field_path="coordinates.latitude") == ""


def test_currency_name_field_preserves_semantic_value():
    lebanon = encode_comparison_value("currency.name", "Lebanese pound", mode="field")
    albania = encode_comparison_value("currency.name", "Albanian lek", mode="field")
    assert "currency_name:lebanese_pound" in lebanon
    assert "currency_name:albanian_lek" in albania
    assert not set(lebanon).intersection(albania)


def test_currency_code_path_is_skipped():
    counts = encode_comparison_value("currency.code", "LBP", mode="field")
    assert counts == {}


def test_normalize_semantic_value_strips_wiki_markup():
    assert normalize_semantic_value("[[Lebanese pound|LBP]]") == "lebanese_pound"
    assert normalize_semantic_value("Arabic") == "arabic"


def test_meaningful_index_term_rejects_template_tokens():
    assert not is_meaningful_index_term("currency:currency")
    assert not is_meaningful_index_term("economy:year_2020s")
    assert not is_meaningful_index_term("area:mag_0")
    assert is_meaningful_index_term("language:arabic")


def test_semantic_only_mode_filters_metadata_paths():
    document = {
        "normalized": {
            "comparison_fields": {
                "religion": "Islam",
                "image_flag": "Flag.svg",
                "economy_gdp_nominal_year": 2020,
            }
        }
    }
    full = merge_indexing_nodes(build_indexing_node_terms(document, mode="field"))
    semantic = merge_indexing_nodes(
        build_indexing_node_terms(document, mode="field", semantic_only=True)
    )
    assert "religion:islam" in semantic
    assert "image_flag" not in str(semantic)
    assert len(semantic) <= len(full)


def test_field_mode_document_indexing():
    document = {
        "normalized": {
            "comparison_fields": {
                "population": {"total": 5_000_000, "year": 2020},
                "capital": "Beirut",
                "official_language": "Arabic",
            }
        }
    }
    merged = merge_indexing_nodes(build_indexing_node_terms(document, mode="field"))
    assert "population_total:1m_to_10m" in merged
    assert "capital:beirut" in merged
    assert "official_language:arabic" in merged or "languages_official:arabic" in merged
    assert "population:mag_6" not in merged
    assert "capital:capital" not in merged


def test_index_field_path_preserves_full_path():
    assert index_field_path("government.type") == "government_type"
    assert index_field_path("languages.official") == "languages_official"
    assert index_field_path("currency.name") == "currency_name"


def test_hdi_uses_tier_value_not_field_echo():
    counts = encode_comparison_value("development.hdi.value", 0.75, mode="field")
    assert "development_hdi_value:medium" in counts
    assert "development_hdi_value:hdi" not in counts
    assert "development_hdi_value:value" not in counts


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


def test_skip_year_metadata_paths():
    assert should_skip_field_path("economy.gdp.nominal.year")


if __name__ == "__main__":
    test_normalize_terms_skips_raw_digits()
    test_normalize_terms_filters_compass_direction_tokens()
    test_coordinate_fields_are_excluded_from_tfidf_tokenization()
    test_year_and_mag_bins_are_not_indexed()
    test_population_uses_semantic_tier_not_mag_bin()
    test_government_type_uses_full_field_value_pair()
    test_field_label_text_is_not_indexed_as_value()
    test_time_zone_and_coordinate_paths_are_skipped()
    test_currency_name_field_preserves_semantic_value()
    test_currency_code_path_is_skipped()
    test_normalize_semantic_value_strips_wiki_markup()
    test_meaningful_index_term_rejects_template_tokens()
    test_semantic_only_mode_filters_metadata_paths()
    test_field_mode_document_indexing()
    test_index_field_path_preserves_full_path()
    test_hdi_uses_tier_value_not_field_echo()
    test_get_comparison_fields_rebuilds_from_normalized_fields()
    test_build_indexing_node_terms_rejects_non_context_mode()
    test_skip_year_metadata_paths()
    print("All VSM preprocessing diagnostic tests passed")
