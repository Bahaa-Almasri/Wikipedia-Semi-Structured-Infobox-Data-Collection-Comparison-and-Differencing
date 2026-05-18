from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


SparseVector = Dict[str, float]
TermCounts = Dict[str, int]


@dataclass
class VSMDocument:
    """One corpus document prepared for vector-space indexing."""

    slug: str
    display_name: str
    terms: TermCounts = field(default_factory=dict)
    field_terms: Dict[str, TermCounts] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slug": self.slug,
            "display_name": self.display_name,
            "terms": dict(self.terms),
            "field_terms": {
                field: dict(terms) for field, terms in self.field_terms.items()
            },
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "VSMDocument":
        return VSMDocument(
            slug=str(data["slug"]),
            display_name=str(data.get("display_name") or data["slug"]),
            terms={str(k): int(v) for k, v in (data.get("terms") or {}).items()},
            field_terms={
                str(field): {str(k): int(v) for k, v in terms.items()}
                for field, terms in (data.get("field_terms") or {}).items()
            },
        )


@dataclass
class VSMIndex:
    """Sparse TF-IDF index for a country corpus."""

    vocabulary: List[str]
    idf: SparseVector
    doc_vectors: Dict[str, SparseVector]
    doc_norms: SparseVector
    documents: Dict[str, str]
    mode: str = "field"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vocabulary": list(self.vocabulary),
            "idf": dict(self.idf),
            "doc_vectors": {
                slug: dict(vector) for slug, vector in self.doc_vectors.items()
            },
            "doc_norms": dict(self.doc_norms),
            "documents": dict(self.documents),
            "mode": self.mode,
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "VSMIndex":
        return VSMIndex(
            vocabulary=[str(term) for term in data.get("vocabulary", [])],
            idf={str(k): float(v) for k, v in (data.get("idf") or {}).items()},
            doc_vectors={
                str(slug): {str(k): float(v) for k, v in vector.items()}
                for slug, vector in (data.get("doc_vectors") or {}).items()
            },
            doc_norms={
                str(k): float(v) for k, v in (data.get("doc_norms") or {}).items()
            },
            documents={
                str(slug): str(name)
                for slug, name in (data.get("documents") or {}).items()
            },
            mode=str(data.get("mode") or "field"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class VSMSearchResult:
    slug: str
    display_name: str
    score: float
    matched_terms: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "country": self.slug,
            "display_name": self.display_name,
            "score": self.score,
            "matched_terms": list(self.matched_terms),
        }


@dataclass(frozen=True)
class VSMClusterPoint:
    country: str
    display_name: str
    cluster_id: int
    x: float
    y: float
    top_similar: List[Dict[str, Any]] = field(default_factory=list)
    is_selected: bool = False
    is_noise: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "country": self.country,
            "display_name": self.display_name,
            "cluster_id": self.cluster_id,
            "x": self.x,
            "y": self.y,
            "top_similar": [dict(item) for item in self.top_similar],
            "is_selected": self.is_selected,
            "is_noise": self.is_noise,
        }


@dataclass(frozen=True)
class VSMClusterSummary:
    cluster_id: int
    size: int
    countries: List[str]
    display_names: List[str] = field(default_factory=list)
    centroid_terms: List[str] = field(default_factory=list)
    is_noise: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "size": self.size,
            "countries": list(self.countries),
            "display_names": list(self.display_names),
            "centroid_terms": list(self.centroid_terms),
            "is_noise": self.is_noise,
        }


@dataclass(frozen=True)
class VSMClusterMerge:
    step: int
    left: int
    right: int
    new_cluster: int
    similarity: float
    distance: float
    size: int
    members: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "left": self.left,
            "right": self.right,
            "new_cluster": self.new_cluster,
            "similarity": self.similarity,
            "distance": self.distance,
            "size": self.size,
            "members": list(self.members),
        }


@dataclass(frozen=True)
class VSMClusteringResult:
    algorithm: str
    distance: str
    mode: str
    points: List[VSMClusterPoint]
    clusters: List[VSMClusterSummary]
    selected_cluster: VSMClusterSummary | None = None
    merges: List[VSMClusterMerge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "distance": self.distance,
            "mode": self.mode,
            "points": [point.to_dict() for point in self.points],
            "clusters": [cluster.to_dict() for cluster in self.clusters],
            "selected_cluster": (
                self.selected_cluster.to_dict() if self.selected_cluster is not None else None
            ),
            "merges": [merge.to_dict() for merge in self.merges],
            "metadata": dict(self.metadata),
        }
