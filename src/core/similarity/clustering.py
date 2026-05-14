from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from core.similarity.ted import compute_ted
from domain.models.tree import TreeNode
from domain.models.vsm import (
    SparseVector,
    VSMClusterPoint,
    VSMClusterSummary,
    VSMClusteringResult,
    VSMIndex,
)


SUPPORTED_CLUSTER_ALGORITHMS = {"kmeans", "dbscan", "agglomerative"}
SUPPORTED_CLUSTER_DISTANCES = {"euclidean", "manhattan", "cosine"}
SUPPORTED_AGGLOMERATIVE_LINKAGES = {"single", "complete", "average"}

DistanceFn = Callable[[Mapping[str, float], Mapping[str, float]], float]


def euclidean(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    terms = set(left) | set(right)
    return math.sqrt(
        sum((left.get(term, 0.0) - right.get(term, 0.0)) ** 2 for term in terms)
    )


def manhattan(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    terms = set(left) | set(right)
    return sum(abs(left.get(term, 0.0) - right.get(term, 0.0)) for term in terms)


def _vector_norm(vector: Mapping[str, float]) -> float:
    return math.sqrt(sum(weight * weight for weight in vector.values()))


def _l2_normalize_vector(vector: Mapping[str, float]) -> SparseVector:
    norm = _vector_norm(vector)
    if norm == 0.0:
        return {}
    return {term: weight / norm for term, weight in vector.items() if weight != 0.0}


def cosine_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    left_norm = _vector_norm(left)
    right_norm = _vector_norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    if len(left) > len(right):
        left, right = right, left
    dot = sum(weight * right.get(term, 0.0) for term, weight in left.items())
    similarity = max(-1.0, min(1.0, dot / (left_norm * right_norm)))
    return 1.0 - similarity


def distance_function(name: str) -> DistanceFn:
    if name == "euclidean":
        return euclidean
    if name == "manhattan":
        return manhattan
    if name == "cosine":
        return cosine_distance
    raise ValueError(f"Unsupported clustering distance '{name}'.")


def _mean_vector(vectors: Sequence[Mapping[str, float]]) -> SparseVector:
    if not vectors:
        return {}
    totals: Dict[str, float] = defaultdict(float)
    for vector in vectors:
        for term, weight in vector.items():
            totals[term] += weight
    denom = float(len(vectors))
    return {term: total / denom for term, total in totals.items() if total != 0.0}


def _closest_centroid(
    vector: Mapping[str, float],
    centroids: Sequence[Mapping[str, float]],
    dist: DistanceFn,
) -> int:
    best_idx = 0
    best_distance = float("inf")
    for idx, centroid in enumerate(centroids):
        current = dist(vector, centroid)
        if current < best_distance:
            best_idx = idx
            best_distance = current
    return best_idx


def kmeans(
    vectors: Sequence[SparseVector],
    *,
    k: int,
    distance: str = "cosine",
    max_iter: int = 100,
    seed: int = 13,
    normalize_vectors: bool = True,
) -> List[int]:
    if not vectors:
        return []
    working_vectors = [
        _l2_normalize_vector(vector) if normalize_vectors else dict(vector)
        for vector in vectors
    ]
    k = max(1, min(int(k), len(vectors)))
    dist = distance_function(distance)
    rng = random.Random(seed)
    initial = rng.sample(range(len(working_vectors)), k)
    centroids: List[SparseVector] = [dict(working_vectors[idx]) for idx in initial]
    labels = [-1] * len(vectors)

    for _ in range(max_iter):
        changed = False
        for idx, vector in enumerate(working_vectors):
            label = _closest_centroid(vector, centroids, dist)
            if labels[idx] != label:
                labels[idx] = label
                changed = True

        grouped: Dict[int, List[SparseVector]] = defaultdict(list)
        for label, vector in zip(labels, working_vectors):
            grouped[label].append(vector)

        new_centroids: List[SparseVector] = []
        for idx in range(k):
            if grouped[idx]:
                centroid = _mean_vector(grouped[idx])
                if normalize_vectors:
                    centroid = _l2_normalize_vector(centroid)
                new_centroids.append(centroid)
            else:
                new_centroids.append(centroids[idx])
        centroids = new_centroids
        if not changed:
            break
    return labels


def _region_query(
    vectors: Sequence[SparseVector],
    point_idx: int,
    *,
    eps: float,
    dist: DistanceFn,
) -> List[int]:
    point = vectors[point_idx]
    return [
        idx
        for idx, vector in enumerate(vectors)
        if dist(point, vector) <= eps
    ]


def dbscan(
    vectors: Sequence[SparseVector],
    *,
    eps: float,
    min_pts: int,
    distance: str = "euclidean",
) -> List[int]:
    if not vectors:
        return []
    dist = distance_function(distance)
    min_pts = max(1, int(min_pts))
    labels = [-99] * len(vectors)  # unvisited
    cluster_id = 0

    for point_idx in range(len(vectors)):
        if labels[point_idx] != -99:
            continue
        neighbors = _region_query(vectors, point_idx, eps=eps, dist=dist)
        if len(neighbors) < min_pts:
            labels[point_idx] = -1
            continue

        labels[point_idx] = cluster_id
        seeds = [idx for idx in neighbors if idx != point_idx]
        cursor = 0
        while cursor < len(seeds):
            neighbor_idx = seeds[cursor]
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
            if labels[neighbor_idx] != -99:
                cursor += 1
                continue

            labels[neighbor_idx] = cluster_id
            neighbor_neighbors = _region_query(vectors, neighbor_idx, eps=eps, dist=dist)
            if len(neighbor_neighbors) >= min_pts:
                for candidate in neighbor_neighbors:
                    if candidate not in seeds:
                        seeds.append(candidate)
            cursor += 1
        cluster_id += 1

    return labels


def _cluster_distance(
    left: Sequence[int],
    right: Sequence[int],
    pairwise: Mapping[Tuple[int, int], float],
    *,
    linkage: str,
) -> float:
    distances = [
        pairwise[(min(i, j), max(i, j))]
        for i in left
        for j in right
    ]
    if linkage == "single":
        return min(distances)
    if linkage == "complete":
        return max(distances)
    if linkage == "average":
        return sum(distances) / len(distances)
    raise ValueError(f"Unsupported agglomerative linkage '{linkage}'.")


def agglomerative(
    vectors: Sequence[SparseVector],
    *,
    k: int,
    distance: str = "euclidean",
    linkage: str = "average",
) -> List[int]:
    if not vectors:
        return []
    if linkage not in SUPPORTED_AGGLOMERATIVE_LINKAGES:
        raise ValueError(f"Unsupported agglomerative linkage '{linkage}'.")
    k = max(1, min(int(k), len(vectors)))
    dist = distance_function(distance)
    pairwise = {
        (i, j): dist(vectors[i], vectors[j])
        for i in range(len(vectors))
        for j in range(i, len(vectors))
    }
    clusters: List[List[int]] = [[idx] for idx in range(len(vectors))]

    while len(clusters) > k:
        best_pair = (0, 1)
        best_distance = float("inf")
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                current = _cluster_distance(
                    clusters[i],
                    clusters[j],
                    pairwise,
                    linkage=linkage,
                )
                if current < best_distance:
                    best_distance = current
                    best_pair = (i, j)

        left, right = best_pair
        clusters[left] = sorted([*clusters[left], *clusters[right]])
        del clusters[right]

    labels = [-1] * len(vectors)
    for cluster_id, members in enumerate(sorted(clusters, key=lambda c: c[0])):
        for idx in members:
            labels[idx] = cluster_id
    return labels


def _term_variances(index: VSMIndex, slugs: Sequence[str]) -> List[Tuple[float, str]]:
    variances: List[Tuple[float, str]] = []
    n = len(slugs)
    if n == 0:
        return []
    for term in index.vocabulary:
        values = [index.doc_vectors[slug].get(term, 0.0) for slug in slugs]
        mean = sum(values) / n
        variance = sum((value - mean) ** 2 for value in values) / n
        variances.append((variance, term))
    variances.sort(reverse=True)
    return variances


def _matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> List[float]:
    return [sum(value * vector[j] for j, value in enumerate(row)) for row in matrix]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _euclidean_coords(left: Tuple[float, float], right: Tuple[float, float]) -> float:
    return math.sqrt((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2)


def _top_eigenpair(
    matrix: Sequence[Sequence[float]],
    *,
    component: int,
    max_iter: int = 100,
    tolerance: float = 1e-10,
) -> Tuple[float, List[float]]:
    n = len(matrix)
    if n == 0:
        return 0.0, []

    vector = [math.sin((idx + 1) * (component + 1.37)) for idx in range(n)]
    norm = math.sqrt(_dot(vector, vector))
    if norm == 0.0:
        return 0.0, [0.0] * n
    vector = [value / norm for value in vector]

    for _ in range(max_iter):
        next_vector = _matvec(matrix, vector)
        next_norm = math.sqrt(_dot(next_vector, next_vector))
        if next_norm == 0.0:
            return 0.0, [0.0] * n
        next_vector = [value / next_norm for value in next_vector]
        delta = math.sqrt(sum((a - b) ** 2 for a, b in zip(next_vector, vector)))
        vector = next_vector
        if delta <= tolerance:
            break

    eigenvalue = _dot(vector, _matvec(matrix, vector))
    if eigenvalue <= 0.0:
        return 0.0, [0.0] * n
    return eigenvalue, vector


def _deflate(
    matrix: Sequence[Sequence[float]],
    eigenvalue: float,
    eigenvector: Sequence[float],
) -> List[List[float]]:
    return [
        [
            matrix[i][j] - eigenvalue * eigenvector[i] * eigenvector[j]
            for j in range(len(matrix))
        ]
        for i in range(len(matrix))
    ]


def pairwise_distance_matrix(
    vectors: Sequence[Mapping[str, float]],
    *,
    distance: str = "cosine",
) -> List[List[float]]:
    dist = distance_function(distance)
    n = len(vectors)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            value = dist(vectors[i], vectors[j])
            matrix[i][j] = value
            matrix[j][i] = value
    return matrix


def project_mds(
    slugs: Sequence[str],
    distance_matrix: Sequence[Sequence[float]],
) -> Tuple[Dict[str, Tuple[float, float]], float]:
    """
    Classical metric MDS over a pairwise distance matrix.

    Coordinates are chosen so Euclidean distance in the plot approximates the
    supplied country distances.
    """
    n = len(slugs)
    if n == 0:
        return {}, 0.0
    if n == 1:
        return {slugs[0]: (0.0, 0.0)}, 0.0

    squared = [[distance_matrix[i][j] ** 2 for j in range(n)] for i in range(n)]
    row_means = [sum(row) / n for row in squared]
    total_mean = sum(row_means) / n
    centered = [
        [
            -0.5 * (squared[i][j] - row_means[i] - row_means[j] + total_mean)
            for j in range(n)
        ]
        for i in range(n)
    ]

    first_value, first_vector = _top_eigenpair(centered, component=0)
    deflated = _deflate(centered, first_value, first_vector) if first_value > 0 else centered
    second_value, second_vector = _top_eigenpair(deflated, component=1)

    x_scale = math.sqrt(max(first_value, 0.0))
    y_scale = math.sqrt(max(second_value, 0.0))
    coordinates = {
        slug: (x_scale * first_vector[idx], y_scale * second_vector[idx])
        for idx, slug in enumerate(slugs)
    }

    numerator = 0.0
    denominator = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            original = distance_matrix[i][j]
            projected = _euclidean_coords(coordinates[slugs[i]], coordinates[slugs[j]])
            numerator += (original - projected) ** 2
            denominator += original ** 2
    stress = math.sqrt(numerator / denominator) if denominator > 0.0 else 0.0
    return coordinates, stress


def project_2d(
    index: VSMIndex,
    slugs: Sequence[str],
    *,
    distance: str = "cosine",
) -> Dict[str, Tuple[float, float]]:
    vectors = [index.doc_vectors[slug] for slug in slugs]
    matrix = pairwise_distance_matrix(vectors, distance=distance)
    projection, _stress = project_mds(slugs, matrix)
    return projection


def project_clusters_to_grid(
    slugs: Sequence[str],
    labels: Sequence[int],
    *,
    min_side: int = 10,
) -> Tuple[Dict[str, Tuple[float, float]], int]:
    """Place clustered documents on an integer grid, grouping cluster members together."""
    if len(slugs) != len(labels):
        raise ValueError("slugs and labels must have the same length.")
    groups: Dict[int, List[str]] = defaultdict(list)
    for slug, label in zip(slugs, labels):
        groups[label].append(slug)

    ordered_cells: List[Optional[str]] = []
    ordered_labels = sorted(groups, key=lambda label: (label == -1, label))
    for idx, label in enumerate(ordered_labels):
        ordered_cells.extend(groups[label])
        if idx < len(ordered_labels) - 1:
            ordered_cells.append(None)  # visual gap between clusters

    required_cells = max(len(ordered_cells), len(slugs), 1)
    side = max(int(min_side), math.ceil(math.sqrt(required_cells)))
    while side * side < required_cells:
        side += 1

    projection: Dict[str, Tuple[float, float]] = {}
    for idx, slug in enumerate(ordered_cells):
        if slug is None:
            continue
        row = idx // side
        col = idx % side
        projection[slug] = (float(col + 1), float(side - row))
    return projection, side


def build_ted_similarity_index(
    tree_docs: Mapping[str, Mapping[str, Any]],
    *,
    documents: Optional[Mapping[str, str]] = None,
    ted_algorithm: str = "chawathe",
    cost_model: Optional[str] = None,
    coerce_root_label: Optional[str] = "infobox",
) -> VSMIndex:
    """Represent each tree as a vector of its TED similarities to every other tree."""
    slugs = sorted(tree_docs)
    vectors: Dict[str, SparseVector] = {
        slug: {other_slug: 0.0 for other_slug in slugs}
        for slug in slugs
    }
    for slug in slugs:
        vectors[slug][slug] = 1.0

    for i, left_slug in enumerate(slugs):
        for right_slug in slugs[i + 1:]:
            result = compute_ted(
                TreeNode.from_dict(dict(tree_docs[left_slug])),
                TreeNode.from_dict(dict(tree_docs[right_slug])),
                algorithm=ted_algorithm,
                coerce_root_label=coerce_root_label,
                cost_model=cost_model,
            )
            similarity = float(result.similarity)
            vectors[left_slug][right_slug] = similarity
            vectors[right_slug][left_slug] = similarity

    doc_norms = {
        slug: sum(weight * weight for weight in vector.values()) ** 0.5
        for slug, vector in vectors.items()
    }
    display_names = dict(documents or {})
    return VSMIndex(
        vocabulary=slugs,
        idf={slug: 1.0 for slug in slugs},
        doc_vectors=vectors,
        doc_norms=doc_norms,
        documents={
            slug: display_names.get(slug, slug.replace("_", " ").title())
            for slug in slugs
        },
        mode=f"ted:{ted_algorithm}",
        metadata={
            "source": "ted",
            "ted_algorithm": ted_algorithm,
            "cost_model": cost_model,
        },
    )


def _cluster_members(labels: Sequence[int]) -> Dict[int, List[int]]:
    groups: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        groups[label].append(idx)
    return dict(groups)


def _centroid_terms(
    index: VSMIndex,
    slugs: Sequence[str],
    *,
    limit: int = 8,
) -> List[str]:
    centroid = _mean_vector([index.doc_vectors[slug] for slug in slugs])
    terms = [
        term
        for term, _weight in sorted(centroid.items(), key=lambda item: item[1], reverse=True)
        if ":" not in term
    ]
    return terms[:limit]


def _nearest_neighbors(
    slugs: Sequence[str],
    distance_matrix: Sequence[Sequence[float]],
    documents: Mapping[str, str],
    *,
    distance_name: str,
    limit: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    neighbors: Dict[str, List[Dict[str, Any]]] = {}
    use_similarity = distance_name == "cosine"
    for i, slug in enumerate(slugs):
        ranked = sorted(
            (
                (distance_matrix[i][j], other_slug)
                for j, other_slug in enumerate(slugs)
                if i != j
            ),
            key=lambda item: item[0],
        )
        items: List[Dict[str, Any]] = []
        for dist_value, other_slug in ranked[:limit]:
            item: Dict[str, Any] = {
                "country": other_slug,
                "display_name": documents.get(other_slug, other_slug.replace("_", " ").title()),
                "distance": dist_value,
            }
            if use_similarity:
                item["similarity"] = max(0.0, min(1.0, 1.0 - dist_value))
            items.append(item)
        neighbors[slug] = items
    return neighbors


def cluster_vsm_index(
    index: VSMIndex,
    *,
    algorithm: str = "kmeans",
    distance: str = "cosine",
    selected_country: Optional[str] = None,
    k: int = 5,
    eps: float = 1.0,
    min_pts: int = 3,
    linkage: str = "average",
    max_iter: int = 100,
    seed: int = 13,
    normalize_kmeans: bool = True,
) -> VSMClusteringResult:
    algorithm = (algorithm or "kmeans").strip().lower()
    distance = (distance or "cosine").strip().lower()
    if algorithm not in SUPPORTED_CLUSTER_ALGORITHMS:
        raise ValueError(f"Unsupported clustering algorithm '{algorithm}'.")
    if distance not in SUPPORTED_CLUSTER_DISTANCES:
        raise ValueError(f"Unsupported clustering distance '{distance}'.")

    slugs = sorted(index.doc_vectors)
    vectors = [index.doc_vectors[slug] for slug in slugs]
    clustering_vectors = [
        _l2_normalize_vector(vector) if algorithm == "kmeans" and normalize_kmeans else dict(vector)
        for vector in vectors
    ]
    if algorithm == "kmeans":
        labels = kmeans(
            vectors,
            k=k,
            distance=distance,
            max_iter=max_iter,
            seed=seed,
            normalize_vectors=normalize_kmeans,
        )
    elif algorithm == "dbscan":
        labels = dbscan(vectors, eps=eps, min_pts=min_pts, distance=distance)
    else:
        labels = agglomerative(vectors, k=k, distance=distance, linkage=linkage)

    distance_matrix = pairwise_distance_matrix(clustering_vectors, distance=distance)
    projection, projection_stress = project_mds(slugs, distance_matrix)
    top_similar = _nearest_neighbors(
        slugs,
        distance_matrix,
        index.documents,
        distance_name=distance,
    )
    selected_cluster_id: Optional[int] = None
    if selected_country in slugs:
        selected_cluster_id = labels[slugs.index(str(selected_country))]

    points: List[VSMClusterPoint] = []
    for slug, label in zip(slugs, labels):
        x, y = projection[slug]
        points.append(
            VSMClusterPoint(
                country=slug,
                display_name=index.documents.get(slug, slug.replace("_", " ").title()),
                cluster_id=label,
                x=x,
                y=y,
                top_similar=top_similar.get(slug, []),
                is_selected=slug == selected_country,
                is_noise=label == -1,
            )
        )

    clusters: List[VSMClusterSummary] = []
    for label, member_indices in sorted(_cluster_members(labels).items()):
        member_slugs = [slugs[idx] for idx in member_indices]
        clusters.append(
            VSMClusterSummary(
                cluster_id=label,
                size=len(member_slugs),
                countries=member_slugs,
                display_names=[
                    index.documents.get(slug, slug.replace("_", " ").title())
                    for slug in member_slugs
                ],
                centroid_terms=_centroid_terms(index, member_slugs),
                is_noise=label == -1,
            )
        )

    selected_cluster = next(
        (cluster for cluster in clusters if cluster.cluster_id == selected_cluster_id),
        None,
    )

    return VSMClusteringResult(
        algorithm=algorithm,
        distance=distance,
        mode=index.mode,
        points=points,
        clusters=clusters,
        selected_cluster=selected_cluster,
        metadata={
            "document_count": len(slugs),
            "vocabulary_size": len(index.vocabulary),
            "k": k,
            "eps": eps,
            "min_pts": min_pts,
            "linkage": linkage,
            "selected_country": selected_country,
            "l2_normalized_kmeans": normalize_kmeans if algorithm == "kmeans" else False,
            "layout": "mds",
            "projection": "classical_mds",
            "projection_distance": distance,
            "projection_stress": projection_stress,
            "nearest_neighbor_count": 5,
        },
    )
