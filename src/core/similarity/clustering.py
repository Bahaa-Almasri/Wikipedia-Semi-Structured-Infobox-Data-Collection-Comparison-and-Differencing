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
    VSMClusterMerge,
    VSMClusterSummary,
    VSMClusteringResult,
    VSMIndex,
)


SUPPORTED_CLUSTER_ALGORITHMS = {"kmeans", "agglomerative"}
SUPPORTED_CLUSTER_DISTANCES = {"euclidean", "manhattan", "cosine"}
SUPPORTED_AGGLOMERATIVE_LINKAGES = {"single", "complete", "average"}
SUPPORTED_AGGLOMERATIVE_STOPPING_RULES = {"none", "cluster_count", "similarity_threshold"}
SUPPORTED_CLUSTER_PROJECTIONS = {"pca"}

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


def normalize_cluster_projection(name: str) -> str:
    normalized = (name or "pca").strip().lower()
    aliases = {
        "pca": "pca",
        "principal_components": "pca",
    }
    mapped = aliases.get(normalized, normalized)
    if mapped not in SUPPORTED_CLUSTER_PROJECTIONS:
        raise ValueError(f"Unsupported cluster projection '{name}'.")
    return mapped


def _mean_vector(vectors: Sequence[Mapping[str, float]]) -> SparseVector:
    if not vectors:
        return {}
    totals: Dict[str, float] = defaultdict(float)
    for vector in vectors:
        for term, weight in vector.items():
            totals[term] += weight
    denom = float(len(vectors))
    return {term: total / denom for term, total in totals.items() if total != 0.0}


def _median_vector(vectors: Sequence[Mapping[str, float]]) -> SparseVector:
    if not vectors:
        return {}
    terms = set()
    for vector in vectors:
        terms.update(vector)
    median: SparseVector = {}
    count = len(vectors)
    for term in terms:
        values = sorted(vector.get(term, 0.0) for vector in vectors)
        if count % 2 == 1:
            median[term] = values[count // 2]
        else:
            median[term] = (values[count // 2 - 1] + values[count // 2]) / 2.0
    return {term: weight for term, weight in median.items() if weight != 0.0}


def _kmeans_normalize_vectors(distance: str, normalize_vectors: Optional[bool]) -> bool:
    if normalize_vectors is not None:
        return normalize_vectors
    return distance == "cosine"


def _cluster_centroid(
    members: Sequence[SparseVector],
    *,
    distance: str,
    normalize_vectors: bool,
) -> SparseVector:
    if not members:
        return {}
    if distance == "manhattan":
        return _median_vector(members)
    centroid = _mean_vector(members)
    if normalize_vectors:
        centroid = _l2_normalize_vector(centroid)
    return centroid


def _closest_centroid(
    vector: Mapping[str, float],
    centroids: Sequence[Mapping[str, float]],
    dist: DistanceFn,
    rng: random.Random,
) -> int:
    tied: List[int] = [0]
    best_distance = float("inf")
    for idx, centroid in enumerate(centroids):
        current = dist(vector, centroid)
        if current < best_distance - 1e-15:
            best_distance = current
            tied = [idx]
        elif math.isclose(current, best_distance, rel_tol=1e-12, abs_tol=1e-12):
            tied.append(idx)
    return tied[0] if len(tied) == 1 else rng.choice(tied)


def kmeans(
    vectors: Sequence[SparseVector],
    *,
    k: int,
    distance: str = "cosine",
    max_iter: int = 100,
    seed: int = 13,
    normalize_vectors: Optional[bool] = None,
) -> List[int]:
    if not vectors:
        return []
    distance = (distance or "cosine").strip().lower()
    normalize = _kmeans_normalize_vectors(distance, normalize_vectors)
    working_vectors = [
        _l2_normalize_vector(vector) if normalize else dict(vector)
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
            label = _closest_centroid(vector, centroids, dist, rng)
            if labels[idx] != label:
                labels[idx] = label
                changed = True

        grouped: Dict[int, List[SparseVector]] = defaultdict(list)
        for label, vector in zip(labels, working_vectors):
            grouped[label].append(vector)

        new_centroids: List[SparseVector] = []
        for idx in range(k):
            if grouped[idx]:
                centroid = _cluster_centroid(
                    grouped[idx],
                    distance=distance,
                    normalize_vectors=normalize,
                )
                new_centroids.append(centroid)
            else:
                new_centroids.append(centroids[idx])
        centroids = new_centroids
        if not changed:
            break
    return labels


def _distance_to_similarity(distance_value: float, distance_name: str) -> float:
    if distance_name == "cosine":
        return max(0.0, min(1.0, 1.0 - distance_value))
    return 1.0 / (1.0 + max(0.0, distance_value))


def _similarity_matrix_from_distances(
    distance_matrix: Sequence[Sequence[float]],
    *,
    distance_name: str,
) -> List[List[float]]:
    n = len(distance_matrix)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            value = _distance_to_similarity(distance_matrix[i][j], distance_name)
            matrix[i][j] = value
            matrix[j][i] = value
    return matrix


def _cluster_pair_similarity(
    left_members: Sequence[int],
    right_members: Sequence[int],
    similarity_matrix: Sequence[Sequence[float]],
    *,
    linkage: str,
) -> float:
    values = [
        similarity_matrix[left_idx][right_idx]
        for left_idx in left_members
        for right_idx in right_members
    ]
    if not values:
        return 0.0
    if linkage == "single":
        return max(values)
    if linkage == "complete":
        return min(values)
    if linkage == "average":
        return sum(values) / len(values)
    raise ValueError(f"Unsupported agglomerative linkage '{linkage}'.")


def _labels_from_active_clusters(
    active_ids: Sequence[int],
    clusters: Mapping[int, Sequence[int]],
    n: int,
) -> List[int]:
    labels = [-1] * n
    for label, cluster_id in enumerate(sorted(active_ids)):
        for member_idx in clusters[cluster_id]:
            labels[member_idx] = label
    return labels


def agglomerative(
    vectors: Sequence[SparseVector],
    *,
    slugs: Optional[Sequence[str]] = None,
    distance: str = "cosine",
    linkage: str = "average",
    stopping_rule: str = "none",
    target_clusters: Optional[int] = None,
    similarity_threshold: Optional[float] = None,
) -> Tuple[List[int], List[VSMClusterMerge], Dict[str, Any]]:
    """Bottom-up hierarchical clustering using the chapter's similarity link rules."""
    n = len(vectors)
    if n == 0:
        return [], [], {"stopping_reason": "empty"}

    distance = (distance or "cosine").strip().lower()
    linkage = (linkage or "average").strip().lower()
    stopping_rule = (stopping_rule or "none").strip().lower()
    if linkage not in SUPPORTED_AGGLOMERATIVE_LINKAGES:
        raise ValueError(f"Unsupported agglomerative linkage '{linkage}'.")
    if stopping_rule not in SUPPORTED_AGGLOMERATIVE_STOPPING_RULES:
        raise ValueError(f"Unsupported agglomerative stopping rule '{stopping_rule}'.")

    desired_clusters = None
    threshold = None
    if stopping_rule == "cluster_count":
        desired_clusters = max(1, min(int(target_clusters or 1), n))
    elif stopping_rule == "similarity_threshold":
        threshold = max(0.0, min(1.0, float(similarity_threshold or 0.0)))

    slug_list = list(slugs or [str(idx) for idx in range(n)])
    distance_matrix = pairwise_distance_matrix(vectors, distance=distance)
    similarity_matrix = _similarity_matrix_from_distances(
        distance_matrix,
        distance_name=distance,
    )

    clusters: Dict[int, List[int]] = {idx: [idx] for idx in range(n)}
    active = list(range(n))
    next_cluster_id = n
    merges: List[VSMClusterMerge] = []
    stopped_labels: Optional[List[int]] = None
    stopping_reason = "single_cluster"

    while len(active) > 1:
        best_pair: Optional[Tuple[int, int]] = None
        best_similarity = -1.0
        ordered_active = sorted(active)
        for pos, left_id in enumerate(ordered_active):
            for right_id in ordered_active[pos + 1:]:
                current = _cluster_pair_similarity(
                    clusters[left_id],
                    clusters[right_id],
                    similarity_matrix,
                    linkage=linkage,
                )
                if (
                    current > best_similarity + 1e-15
                    or (
                        math.isclose(current, best_similarity, rel_tol=1e-12, abs_tol=1e-12)
                        and (best_pair is None or (left_id, right_id) < best_pair)
                    )
                ):
                    best_similarity = current
                    best_pair = (left_id, right_id)

        if best_pair is None:
            break

        if stopped_labels is None:
            if desired_clusters is not None and len(active) <= desired_clusters:
                stopped_labels = _labels_from_active_clusters(active, clusters, n)
                stopping_reason = "cluster_count"
            elif threshold is not None and best_similarity < threshold:
                stopped_labels = _labels_from_active_clusters(active, clusters, n)
                stopping_reason = "similarity_threshold"

        left_id, right_id = best_pair
        merged_members = sorted(clusters[left_id] + clusters[right_id])
        clusters[next_cluster_id] = merged_members
        active = [
            cluster_id
            for cluster_id in active
            if cluster_id not in {left_id, right_id}
        ]
        active.append(next_cluster_id)
        merges.append(
            VSMClusterMerge(
                step=len(merges) + 1,
                left=left_id,
                right=right_id,
                new_cluster=next_cluster_id,
                similarity=best_similarity,
                distance=1.0 - best_similarity,
                size=len(merged_members),
                members=[slug_list[idx] for idx in merged_members],
            )
        )
        next_cluster_id += 1

    if stopped_labels is None:
        stopped_labels = _labels_from_active_clusters(active, clusters, n)

    metadata = {
        "linkage": linkage,
        "stopping_rule": stopping_rule,
        "stopping_reason": stopping_reason,
        "target_clusters": desired_clusters,
        "similarity_threshold": threshold,
        "merge_count": len(merges),
        "dendrogram_height": max((merge.distance for merge in merges), default=0.0),
    }
    return stopped_labels, merges, metadata



def _matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> List[float]:
    return [sum(value * vector[j] for j, value in enumerate(row)) for row in matrix]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


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

    # Shifted power iteration stabilizes the dominant eigenvector for PCA Gram matrices.
    spectral_bound = max((sum(abs(value) for value in row) for row in matrix), default=0.0)
    shift = spectral_bound + 1e-9
    vector = [math.sin((idx + 1) * (component + 1.37)) for idx in range(n)]
    norm = math.sqrt(_dot(vector, vector))
    if norm == 0.0:
        return 0.0, [0.0] * n
    vector = [value / norm for value in vector]

    for _ in range(max_iter):
        next_vector = [
            value + shift * vector[idx]
            for idx, value in enumerate(_matvec(matrix, vector))
        ]
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


def _centered_dense_rows(
    vectors: Sequence[Mapping[str, float]],
    vocabulary: Sequence[str],
) -> List[List[float]]:
    n = len(vectors)
    if n == 0:
        return []
    dimension = len(vocabulary)
    if dimension == 0:
        return [[0.0] * 0 for _ in range(n)]

    term_index = {term: idx for idx, term in enumerate(vocabulary)}
    dense = [[0.0] * dimension for _ in range(n)]
    for row_idx, vector in enumerate(vectors):
        for term, weight in vector.items():
            col = term_index.get(term)
            if col is not None:
                dense[row_idx][col] = weight

    means = [sum(row[col] for row in dense) / n for col in range(dimension)]
    return [[row[col] - means[col] for col in range(dimension)] for row in dense]


def project_pca(
    slugs: Sequence[str],
    vectors: Sequence[Mapping[str, float]],
    vocabulary: Sequence[str],
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float]]:
    """
    Two-dimensional PCA over document vectors.

    Uses the Gram matrix (n x n) when the vocabulary is larger than the
    document count, which is typical for country-scale VSM indices.
    """
    n = len(slugs)
    if n == 0:
        return {}, {
            "explained_variance_ratio_pc1": 0.0,
            "explained_variance_ratio_pc2": 0.0,
        }
    if n == 1:
        return {slugs[0]: (0.0, 0.0)}, {
            "explained_variance_ratio_pc1": 0.0,
            "explained_variance_ratio_pc2": 0.0,
        }

    centered = _centered_dense_rows(vectors, vocabulary)
    gram = [[_dot(centered[i], centered[j]) for j in range(n)] for i in range(n)]
    total_variance = sum(gram[i][i] for i in range(n))
    if total_variance <= 0.0:
        return {slug: (0.0, 0.0) for slug in slugs}, {
            "explained_variance_ratio_pc1": 0.0,
            "explained_variance_ratio_pc2": 0.0,
        }

    first_value, first_vector = _top_eigenpair(gram, component=0)
    deflated = _deflate(gram, first_value, first_vector) if first_value > 0 else gram
    second_value, second_vector = _top_eigenpair(deflated, component=1)

    x_scale = math.sqrt(max(first_value, 0.0))
    y_scale = math.sqrt(max(second_value, 0.0))
    coordinates = {
        slug: (x_scale * first_vector[idx], y_scale * second_vector[idx])
        for idx, slug in enumerate(slugs)
    }
    return coordinates, {
        "explained_variance_ratio_pc1": first_value / total_variance,
        "explained_variance_ratio_pc2": second_value / total_variance,
    }


def project_cluster_coordinates(
    slugs: Sequence[str],
    vectors: Sequence[Mapping[str, float]],
    *,
    vocabulary: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Any]]:
    vocab = list(vocabulary or [])
    if not vocab:
        terms = set()
        for vector in vectors:
            terms.update(vector)
        vocab = sorted(terms)
    coordinates, pca_meta = project_pca(slugs, vectors, vocab)
    metadata: Dict[str, Any] = {
        "layout": "pca",
        "projection": "pca",
        **pca_meta,
    }
    return coordinates, metadata


def project_2d(
    index: VSMIndex,
    slugs: Sequence[str],
) -> Dict[str, Tuple[float, float]]:
    vectors = [index.doc_vectors[slug] for slug in slugs]
    coordinates, _metadata = project_cluster_coordinates(
        slugs,
        vectors,
        vocabulary=index.vocabulary,
    )
    return coordinates


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
    max_iter: int = 100,
    seed: int = 13,
    normalize_kmeans: Optional[bool] = None,
    linkage: str = "average",
    stopping_rule: str = "none",
    similarity_threshold: Optional[float] = None,
) -> VSMClusteringResult:
    algorithm = (algorithm or "kmeans").strip().lower()
    distance = (distance or "cosine").strip().lower()
    if algorithm not in SUPPORTED_CLUSTER_ALGORITHMS:
        raise ValueError(f"Unsupported clustering algorithm '{algorithm}'.")
    if distance not in SUPPORTED_CLUSTER_DISTANCES:
        raise ValueError(f"Unsupported clustering distance '{distance}'.")

    slugs = sorted(index.doc_vectors)
    vectors = [index.doc_vectors[slug] for slug in slugs]
    merges: List[VSMClusterMerge] = []
    algorithm_metadata: Dict[str, Any] = {}
    if algorithm == "kmeans":
        l2_normalized = _kmeans_normalize_vectors(distance, normalize_kmeans)
        clustering_vectors = [
            _l2_normalize_vector(vector) if l2_normalized else dict(vector)
            for vector in vectors
        ]
        k_used = max(1, min(int(k), len(vectors)))
        labels = kmeans(
            vectors,
            k=k_used,
            distance=distance,
            max_iter=max_iter,
            seed=seed,
            normalize_vectors=l2_normalized,
        )
        algorithm_metadata = {
            "k": k_used,
            "l2_normalized_kmeans": l2_normalized,
        }
    elif algorithm == "agglomerative":
        clustering_vectors = [dict(vector) for vector in vectors]
        labels, merges, algorithm_metadata = agglomerative(
            clustering_vectors,
            slugs=slugs,
            distance=distance,
            linkage=linkage,
            stopping_rule=stopping_rule,
            target_clusters=k,
            similarity_threshold=similarity_threshold,
        )
    else:
        raise ValueError(f"Unsupported clustering algorithm '{algorithm}'.")

    distance_matrix = pairwise_distance_matrix(clustering_vectors, distance=distance)
    projection, projection_metadata = project_cluster_coordinates(
        slugs,
        clustering_vectors,
        vocabulary=index.vocabulary,
    )
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
    cluster_groups = _cluster_members(labels)
    non_noise_groups = {
        label: members for label, members in cluster_groups.items() if label >= 0
    }

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
            "cluster_count": len(non_noise_groups),
            "noise_count": len(cluster_groups.get(-1, [])),
            "singleton_cluster_count": sum(
                1 for members in non_noise_groups.values() if len(members) == 1
            ),
            "largest_cluster_size": max(
                (len(members) for members in non_noise_groups.values()),
                default=0,
            ),
            "selected_country": selected_country,
            "projection_distance": distance,
            "nearest_neighbor_count": 5,
            **algorithm_metadata,
            **projection_metadata,
        },
        merges=merges,
    )
