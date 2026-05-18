#!/usr/bin/env python
"""Small diagnostics for TF-IDF VSM retrieval."""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import math

from core.similarity.vsm import (  # noqa: E402
    DEFAULT_VSM_MODE,
    SUPPORTED_VSM_MODES,
    build_vsm_index,
    cosine_similarity,
    inverse_document_frequency,
    normalize_terms,
    pcc_similarity,
    query_counts,
    rank_query,
    sparse_similarity,
    vsm_document_from_json,
)


def _doc(country_name: str, fields: dict) -> dict:
    normalized_fields = {}
    comparison_fields = {}
    for key, text in fields.items():
        normalized_fields[key] = {
            "raw_label": key.replace("_", " ").title(),
            "text": text,
            "tokens": normalize_terms(text),
            "numbers": [],
        }
        comparison_fields[key] = text
    return {
        "meta": {"country_name": country_name},
        "normalized": {
            "fields": normalized_fields,
            "comparison_fields": comparison_fields,
        },
    }


def _index():
    docs = [
        vsm_document_from_json(
            "lebanon",
            _doc("Lebanon", {"capital": "Beirut Beirut", "population": "5 million"}),
        ),
        vsm_document_from_json(
            "japan",
            _doc("Japan", {"capital": "Tokyo", "population": "125 million"}),
        ),
        vsm_document_from_json(
            "canada",
            _doc("Canada", {"capital": "Ottawa", "area": "large northern country"}),
        ),
    ]
    return build_vsm_index(docs)


def test_only_field_vsm_mode_supported():
    assert SUPPORTED_VSM_MODES == {"field"}
    assert DEFAULT_VSM_MODE == "field"


def test_tokenization_and_tf_counts():
    assert normalize_terms("Capital: Beirut, Beirut!") == ["capital", "beirut", "beirut"]
    counts = query_counts("Beirut Beirut capital")
    assert counts["beirut"] == 2
    assert counts["capital"] == 1


def test_idf_rewards_rarer_terms():
    index = _index()
    assert index.idf["capital:beirut"] > index.idf["population:million"]


def test_idf_matches_log_n_over_df():
    index = _index()
    assert index.idf["capital:beirut"] == math.log(3 / 1)
    assert index.idf["population:million"] == math.log(3 / 2)
    assert inverse_document_frequency(3, 3) == 0.0


def test_cosine_identity_and_disjoint():
    assert cosine_similarity({"a": 1.0}, {"a": 1.0}) == 1.0
    assert cosine_similarity({"a": 1.0}, {"b": 1.0}) == 0.0


def test_pcc_identity_and_empty():
    assert pcc_similarity({"a": 1.0, "b": 2.0}, {"a": 1.0, "b": 2.0}) == 1.0
    assert pcc_similarity({}, {"a": 1.0}) == 0.0


def test_query_ranks_matching_document_first():
    results = rank_query(_index(), "capital beirut", top_k=3, metric="cosine")
    assert results[0].slug == "lebanon"
    assert "beirut" in results[0].matched_terms


def test_feature_filtering_ignores_other_fields():
    source = _doc("Lebanon", {"capital": "Beirut", "population": "5 million"})
    filtered = vsm_document_from_json(
        "lebanon",
        source,
        features=["population"],
    )
    assert "population" in filtered.field_terms
    assert "capital" not in filtered.field_terms
    assert "capital:beirut" not in filtered.terms


def test_comparison_field_source_avoids_raw_field_double_counting():
    source = {
        "meta": {"country_name": "Example"},
        "normalized": {
            "fields": {
                "raw_only": {
                    "raw_label": "Raw Only",
                    "text": "rawtoken",
                    "tokens": ["rawtoken"],
                    "numbers": [],
                }
            },
            "comparison_fields": {"semantic_only": "semantic token"},
        },
    }
    document = vsm_document_from_json("example", source)
    assert "rawtoken" not in document.terms
    assert "semantic_only:semantic" in document.terms


def test_max_df_ratio_prunes_corpus_wide_terms():
    docs = [
        vsm_document_from_json("a", _doc("A", {"field": "common alpha"})),
        vsm_document_from_json("b", _doc("B", {"field": "common beta"})),
        vsm_document_from_json("c", _doc("C", {"field": "common gamma"})),
    ]
    index = build_vsm_index(docs, max_df_ratio=0.66)
    assert "field:common" not in index.vocabulary
    assert "field:alpha" in index.vocabulary


def test_empty_query_scores_zero():
    index = _index()
    results = rank_query(index, "", top_k=3, metric="cosine")
    assert all(result.score == 0.0 for result in results)


def test_pcc_metric_path_runs():
    index = _index()
    left = index.doc_vectors["lebanon"]
    right = index.doc_vectors["japan"]
    score = sparse_similarity(left, right, metric="pcc")
    assert isinstance(score, float)


if __name__ == "__main__":
    test_only_field_vsm_mode_supported()
    test_tokenization_and_tf_counts()
    test_idf_rewards_rarer_terms()
    test_idf_matches_log_n_over_df()
    test_cosine_identity_and_disjoint()
    test_pcc_identity_and_empty()
    test_query_ranks_matching_document_first()
    test_feature_filtering_ignores_other_fields()
    test_comparison_field_source_avoids_raw_field_double_counting()
    test_max_df_ratio_prunes_corpus_wide_terms()
    test_empty_query_scores_zero()
    test_pcc_metric_path_runs()
    print("All VSM diagnostic tests passed")
