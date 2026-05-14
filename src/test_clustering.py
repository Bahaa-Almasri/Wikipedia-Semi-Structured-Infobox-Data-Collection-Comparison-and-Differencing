#!/usr/bin/env python
"""Small diagnostics for VSM clustering."""
from __future__ import annotations

import math
import sys

sys.path.insert(0, "src")

from core.similarity.clustering import (  # noqa: E402
    agglomerative,
    build_ted_similarity_index,
    cluster_vsm_index,
    cosine_distance,
    dbscan,
    euclidean,
    kmeans,
    manhattan,
    pairwise_distance_matrix,
    project_clusters_to_grid,
    project_2d,
    project_mds,
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


def test_dbscan_marks_isolated_point_as_noise():
    labels = dbscan(_vectors() + [{"far": 20.0}], eps=0.25, min_pts=2)
    assert _same_cluster(labels, 0, 1)
    assert _same_cluster(labels, 2, 3)
    assert labels[4] == -1


def test_agglomerative_linkages_return_requested_cluster_count():
    for linkage in ["single", "complete", "average"]:
        labels = agglomerative(_vectors(), k=2, distance="euclidean", linkage=linkage)
        assert len(set(labels)) == 2


def test_selected_country_returns_cluster_members():
    result = cluster_vsm_index(
        _index(),
        algorithm="dbscan",
        distance="euclidean",
        selected_country="alpha",
        eps=0.25,
        min_pts=2,
    )
    selected = result.selected_cluster
    assert selected is not None
    assert selected.countries == ["alpha", "beta"]


def test_cluster_metadata_records_kmeans_normalization():
    result = cluster_vsm_index(_index(), algorithm="kmeans", distance="cosine", k=2)
    assert result.metadata["l2_normalized_kmeans"] is True
    assert result.metadata["layout"] == "mds"
    assert result.metadata["projection"] == "classical_mds"


def test_projection_returns_finite_coordinates_for_every_document():
    index = _index()
    projection = project_2d(index, sorted(index.doc_vectors))
    assert set(projection) == set(index.doc_vectors)
    for x, y in projection.values():
        assert math.isfinite(x)
        assert math.isfinite(y)


def test_mds_places_similar_vectors_closer_than_dissimilar_vectors():
    vectors = _vectors()
    slugs = ["warm_a", "warm_b", "cold_a", "cold_b"]
    matrix = pairwise_distance_matrix(vectors, distance="cosine")
    projection, stress = project_mds(slugs, matrix)
    assert stress >= 0.0
    warm_distance = euclidean(
        {"x": projection["warm_a"][0], "y": projection["warm_a"][1]},
        {"x": projection["warm_b"][0], "y": projection["warm_b"][1]},
    )
    cross_distance = euclidean(
        {"x": projection["warm_a"][0], "y": projection["warm_a"][1]},
        {"x": projection["cold_a"][0], "y": projection["cold_a"][1]},
    )
    assert warm_distance < cross_distance


def test_cluster_points_include_top_similar_neighbors():
    result = cluster_vsm_index(_index(), algorithm="kmeans", distance="cosine", k=2)
    alpha = next(point for point in result.points if point.country == "alpha")
    assert alpha.top_similar
    assert alpha.top_similar[0]["country"] == "beta"


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
    test_distances_on_tiny_vectors()
    test_kmeans_separates_obvious_groups()
    test_kmeans_is_deterministic_with_fixed_seed()
    test_kmeans_supports_cosine_distance_with_l2_normalization()
    test_dbscan_marks_isolated_point_as_noise()
    test_agglomerative_linkages_return_requested_cluster_count()
    test_selected_country_returns_cluster_members()
    test_cluster_metadata_records_kmeans_normalization()
    test_projection_returns_finite_coordinates_for_every_document()
    test_mds_places_similar_vectors_closer_than_dissimilar_vectors()
    test_cluster_points_include_top_similar_neighbors()
    test_grid_projection_uses_minimum_ten_by_ten_layout()
    test_grid_projection_grows_for_large_collections()
    test_ted_similarity_profiles_can_drive_clustering()
    print("All VSM clustering diagnostic tests passed")
