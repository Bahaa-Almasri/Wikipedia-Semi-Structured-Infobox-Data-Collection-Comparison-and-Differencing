#!/usr/bin/env python
"""Small diagnostics for VSM clustering."""
from __future__ import annotations

import math
import sys
from collections import Counter

sys.path.insert(0, "src")

from core.similarity.clustering import (  # noqa: E402
    SUPPORTED_AGGLOMERATIVE_LINKAGES,
    SUPPORTED_AGGLOMERATIVE_STOPPING_RULES,
    SUPPORTED_CLUSTER_ALGORITHMS,
    SUPPORTED_CLUSTER_PROJECTIONS,
    agglomerative,
    build_ted_similarity_index,
    cluster_vsm_index,
    cosine_distance,
    euclidean,
    kmeans,
    manhattan,
    normalize_cluster_projection,
    project_cluster_coordinates,
    project_clusters_to_grid,
    project_2d,
    project_pca,
)
from domain.models.vsm import VSMIndex  # noqa: E402


def _vectors():
    return [
        {"warm": 1.0, "coast": 1.0},
        {"warm": 1.1, "coast": 0.9},
        {"cold": 1.0, "mountain": 1.0},
        {"cold": 1.1, "mountain": 0.9},
    ]


def _index() -> VSMIndex:
    slugs = ["alpha", "beta", "gamma", "delta", "outlier"]
    vectors = _vectors() + [{"island": 10.0}]
    return VSMIndex(
        vocabulary=["warm", "coast", "cold", "mountain", "island"],
        idf={term: 1.0 for term in ["warm", "coast", "cold", "mountain", "island"]},
        doc_vectors={slug: vector for slug, vector in zip(slugs, vectors)},
        doc_norms={slug: 1.0 for slug in slugs},
        documents={
            "alpha": "Alpha",
            "beta": "Beta",
            "gamma": "Gamma",
            "delta": "Delta",
            "outlier": "Outlier",
        },
        mode="field",
    )


def _tree(country_name: str, capital: str) -> dict:
    return {
        "label": "infobox",
        "children": [
            {"label": "country_name", "value": country_name, "children": []},
            {"label": "capital", "value": capital, "children": []},
        ],
    }


def _same_cluster(labels, left: int, right: int) -> bool:
    return labels[left] == labels[right]


def test_only_kmeans_algorithm_supported():
    assert SUPPORTED_CLUSTER_ALGORITHMS == {"kmeans", "agglomerative"}


def test_agglomerative_options_match_chapter_rules():
    assert SUPPORTED_AGGLOMERATIVE_LINKAGES == {"single", "complete", "average"}
    assert SUPPORTED_AGGLOMERATIVE_STOPPING_RULES == {
        "none",
        "cluster_count",
        "similarity_threshold",
    }


def test_only_pca_projection_supported():
    assert SUPPORTED_CLUSTER_PROJECTIONS == {"pca"}


def test_distances_on_tiny_vectors():
    assert euclidean({"x": 0.0, "y": 0.0}, {"x": 3.0, "y": 4.0}) == 5.0
    assert manhattan({"x": 0.0, "y": 0.0}, {"x": 3.0, "y": 4.0}) == 7.0
    assert cosine_distance({"x": 1.0}, {"x": 2.0}) == 0.0
    assert cosine_distance({"x": 1.0}, {"y": 1.0}) == 1.0


def test_kmeans_separates_obvious_groups():
    labels = kmeans(_vectors(), k=2, distance="euclidean", seed=4)
    assert _same_cluster(labels, 0, 1)
    assert _same_cluster(labels, 2, 3)
    assert labels[0] != labels[2]


def test_kmeans_is_deterministic_with_fixed_seed():
    first = kmeans(_vectors(), k=2, distance="manhattan", seed=7)
    second = kmeans(_vectors(), k=2, distance="manhattan", seed=7)
    assert first == second


def test_kmeans_supports_cosine_distance_with_l2_normalization():
    labels = kmeans(_vectors(), k=2, distance="cosine", seed=4)
    assert _same_cluster(labels, 0, 1)
    assert _same_cluster(labels, 2, 3)
    assert labels[0] != labels[2]


def test_agglomerative_average_link_builds_full_dendrogram():
    labels, merges, metadata = agglomerative(
        _vectors(),
        slugs=["alpha", "beta", "gamma", "delta"],
        distance="cosine",
        linkage="average",
    )
    assert len(set(labels)) == 1
    assert len(merges) == 3
    assert merges[0].similarity > merges[-1].similarity
    assert metadata["linkage"] == "average"
    assert metadata["stopping_rule"] == "none"


def test_agglomerative_cluster_count_stopping_returns_cut_labels():
    labels, merges, metadata = agglomerative(
        _vectors(),
        distance="cosine",
        linkage="complete",
        stopping_rule="cluster_count",
        target_clusters=2,
    )
    assert len(merges) == 3
    assert len(set(labels)) == 2
    assert _same_cluster(labels, 0, 1)
    assert _same_cluster(labels, 2, 3)
    assert labels[0] != labels[2]
    assert metadata["stopping_reason"] == "cluster_count"


def test_agglomerative_similarity_threshold_stops_before_weak_merge():
    labels, merges, metadata = agglomerative(
        _vectors(),
        distance="cosine",
        linkage="single",
        stopping_rule="similarity_threshold",
        similarity_threshold=0.5,
    )
    assert len(merges) == 3
    assert len(set(labels)) == 2
    assert metadata["stopping_reason"] == "similarity_threshold"


def _grouped_country_vectors(groups: int = 5, per_group: int = 20):
    vectors = []
    for group in range(groups):
        for _ in range(per_group):
            vector = {f"shared_schema_{idx}": 0.2 for idx in range(6)}
            for term_idx in range(6):
                vector[f"group_{group}_signal_{term_idx}"] = 1.0
            vectors.append(vector)
    return vectors


def test_manhattan_separates_distinct_sparse_groups():
    labels = kmeans(_grouped_country_vectors(), k=5, distance="manhattan", seed=13)
    sizes = Counter(labels)
    assert len(sizes) == 5
    assert max(sizes.values()) <= 25


def test_selected_country_returns_cluster_members():
    result = cluster_vsm_index(
        _index(),
        algorithm="kmeans",
        distance="euclidean",
        selected_country="alpha",
        k=2,
    )
    selected = result.selected_cluster
    assert selected is not None
    assert "alpha" in selected.countries
    assert "beta" in selected.countries


def test_cluster_metadata_records_kmeans_and_pca():
    result = cluster_vsm_index(_index(), algorithm="kmeans", distance="cosine", k=2)
    assert result.metadata["l2_normalized_kmeans"] is True
    assert result.metadata["layout"] == "pca"
    assert result.metadata["projection"] == "pca"
    assert result.metadata["k"] == 2
    assert result.metadata["explained_variance_ratio_pc1"] is not None


def test_cluster_metadata_skips_l2_normalization_for_manhattan():
    result = cluster_vsm_index(_index(), algorithm="kmeans", distance="manhattan", k=2)
    assert result.metadata["l2_normalized_kmeans"] is False


def test_projection_returns_finite_coordinates_for_every_document():
    index = _index()
    projection = project_2d(index, sorted(index.doc_vectors))
    assert set(projection) == set(index.doc_vectors)
    for x, y in projection.values():
        assert math.isfinite(x)
        assert math.isfinite(y)


def test_normalize_cluster_projection_accepts_pca_aliases():
    assert normalize_cluster_projection("pca") == "pca"
    assert normalize_cluster_projection("principal_components") == "pca"
    assert normalize_cluster_projection(None) == "pca"


def test_pca_projection_returns_finite_coordinates():
    index = _index()
    slugs = sorted(index.doc_vectors)
    vectors = [index.doc_vectors[slug] for slug in slugs]
    projection, meta = project_pca(slugs, vectors, index.vocabulary)
    assert meta["explained_variance_ratio_pc1"] >= 0.0
    assert set(projection) == set(slugs)
    for x, y in projection.values():
        assert math.isfinite(x)
        assert math.isfinite(y)


def test_project_cluster_coordinates_uses_pca():
    index = _index()
    slugs = sorted(index.doc_vectors)
    vectors = [index.doc_vectors[slug] for slug in slugs]
    coords, meta = project_cluster_coordinates(
        slugs,
        vectors,
        vocabulary=index.vocabulary,
    )
    assert meta["projection"] == "pca"
    assert meta["layout"] == "pca"
    assert set(coords) == set(slugs)


def test_cluster_points_include_top_similar_neighbors():
    result = cluster_vsm_index(_index(), algorithm="kmeans", distance="cosine", k=2)
    alpha = next(point for point in result.points if point.country == "alpha")
    assert alpha.top_similar
    assert alpha.top_similar[0]["country"] == "beta"


def test_cluster_result_includes_agglomerative_merges():
    result = cluster_vsm_index(
        _index(),
        algorithm="agglomerative",
        distance="cosine",
        linkage="average",
        stopping_rule="cluster_count",
        k=2,
        selected_country="alpha",
    )
    out = result.to_dict()
    assert len(out["merges"]) == len(_index().doc_vectors) - 1
    assert out["metadata"]["linkage"] == "average"
    assert out["metadata"]["stopping_rule"] == "cluster_count"
    assert out["selected_cluster"] is not None


def test_grid_projection_uses_minimum_ten_by_ten_layout():
    slugs = [f"country_{idx}" for idx in range(12)]
    labels = [0] * 6 + [1] * 6
    projection, grid_size = project_clusters_to_grid(slugs, labels)
    assert grid_size == 10
    assert set(projection) == set(slugs)
    assert all(1.0 <= x <= grid_size and 1.0 <= y <= grid_size for x, y in projection.values())


def test_grid_projection_grows_for_large_collections():
    slugs = [f"country_{idx}" for idx in range(121)]
    labels = [idx % 3 for idx in range(121)]
    _projection, grid_size = project_clusters_to_grid(slugs, labels)
    assert grid_size > 10


def test_ted_similarity_index_roundtrip_preserves_vectors():
    index = build_ted_similarity_index(
        {
            "alpha": _tree("Alpha", "Same City"),
            "beta": _tree("Beta", "Same City"),
        },
        documents={"alpha": "Alpha", "beta": "Beta"},
        ted_algorithm="nj",
    )
    index.metadata["country_slugs"] = ["alpha", "beta"]
    restored = VSMIndex.from_dict(index.to_dict())
    assert restored.doc_vectors["alpha"]["beta"] == index.doc_vectors["alpha"]["beta"]
    assert restored.metadata["ted_algorithm"] == "nj"


def test_ted_similarity_profiles_can_drive_clustering():
    index = build_ted_similarity_index(
        {
            "alpha": _tree("Alpha", "Same City"),
            "beta": _tree("Beta", "Same City"),
            "gamma": _tree("Gamma", "Different City"),
        },
        documents={"alpha": "Alpha", "beta": "Beta", "gamma": "Gamma"},
        ted_algorithm="chawathe",
    )

    assert index.metadata["source"] == "ted"
    assert index.doc_vectors["alpha"]["beta"] > index.doc_vectors["alpha"]["gamma"]
    result = cluster_vsm_index(index, algorithm="kmeans", k=2, selected_country="alpha")
    assert result.selected_cluster is not None
    assert "beta" in result.selected_cluster.countries


if __name__ == "__main__":
    test_only_kmeans_algorithm_supported()
    test_agglomerative_options_match_chapter_rules()
    test_only_pca_projection_supported()
    test_distances_on_tiny_vectors()
    test_kmeans_separates_obvious_groups()
    test_kmeans_is_deterministic_with_fixed_seed()
    test_kmeans_supports_cosine_distance_with_l2_normalization()
    test_agglomerative_average_link_builds_full_dendrogram()
    test_agglomerative_cluster_count_stopping_returns_cut_labels()
    test_agglomerative_similarity_threshold_stops_before_weak_merge()
    test_manhattan_separates_distinct_sparse_groups()
    test_selected_country_returns_cluster_members()
    test_cluster_metadata_skips_l2_normalization_for_manhattan()
    test_cluster_metadata_records_kmeans_and_pca()
    test_projection_returns_finite_coordinates_for_every_document()
    test_normalize_cluster_projection_accepts_pca_aliases()
    test_pca_projection_returns_finite_coordinates()
    test_project_cluster_coordinates_uses_pca()
    test_cluster_points_include_top_similar_neighbors()
    test_cluster_result_includes_agglomerative_merges()
    test_grid_projection_uses_minimum_ten_by_ten_layout()
    test_grid_projection_grows_for_large_collections()
    test_ted_similarity_index_roundtrip_preserves_vectors()
    test_ted_similarity_profiles_can_drive_clustering()
    print("All VSM clustering diagnostic tests passed")
