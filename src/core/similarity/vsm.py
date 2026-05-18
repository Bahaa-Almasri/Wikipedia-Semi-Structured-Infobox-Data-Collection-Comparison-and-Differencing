from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from core.similarity.vsm_preprocessing import (
    build_indexing_node_terms,
    is_meaningful_vsm_token,
    merge_indexing_nodes,
    normalize_terms,
)
from domain.models.vsm import SparseVector, TermCounts, VSMDocument, VSMIndex, VSMSearchResult


SUPPORTED_VSM_METRICS = {"cosine", "pcc"}
SUPPORTED_VSM_MODES = {"field"}
DEFAULT_VSM_MODE = "field"
COMPARISON_CONTENT_SOURCE = "comparison_fields"
MIN_SHARED_MEANINGFUL_TERMS = 3
INSUFFICIENT_TERMS_MESSAGE = (
    "Not enough meaningful terms after filtering. Try selecting broader features."
)


def _add_count(target: TermCounts, term: str, amount: int = 1) -> None:
    if not term:
        return
    target[term] = target.get(term, 0) + amount


def _merge_counts(target: TermCounts, source: Mapping[str, int]) -> None:
    for term, count in source.items():
        _add_count(target, term, int(count))


def _count_terms(terms: Iterable[str]) -> TermCounts:
    return {term: count for term, count in Counter(terms).items() if term}


def vsm_document_from_json(
    slug: str,
    document: Mapping[str, Any],
    *,
    mode: str = DEFAULT_VSM_MODE,
    features: Optional[Sequence[str]] = None,
) -> VSMDocument:
    """
    Build a VSM document from comparison_fields (same payload as TED country trees).

    Term-context mode keeps the comparison path on each term so values from
    different attributes remain distinct. Numeric values use semantic bins, not
    raw digits.
    """
    if mode not in SUPPORTED_VSM_MODES:
        raise ValueError(f"Unsupported VSM mode '{mode}'.")

    meta = document.get("meta") or {}
    display_name = str(meta.get("country_name") or slug.replace("_", " ").title())

    field_terms = build_indexing_node_terms(
        document,
        mode=mode,
        features=features,
    )
    all_terms = merge_indexing_nodes(field_terms)

    return VSMDocument(
        slug=slug,
        display_name=display_name,
        terms=all_terms,
        field_terms=field_terms,
    )


def inverse_document_frequency(doc_count: int, document_frequency: int) -> float:
    if doc_count <= 0 or document_frequency <= 0:
        return 0.0
    return math.log(doc_count / document_frequency)


def build_vsm_index(
    documents: Sequence[VSMDocument],
    *,
    mode: str = DEFAULT_VSM_MODE,
    max_df_ratio: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> VSMIndex:
    doc_count = len(documents)
    df: Counter[str] = Counter()
    for document in documents:
        for term in document.terms:
            df[term] += 1

    if max_df_ratio is not None:
        max_df_ratio = max(0.0, min(float(max_df_ratio), 1.0))
    pruned_terms = {
        term
        for term, frequency in df.items()
        if doc_count > 0 and max_df_ratio is not None and (frequency / doc_count) > max_df_ratio
    }
    vocabulary = sorted(term for term in df if term not in pruned_terms)
    idf: SparseVector = {
        term: inverse_document_frequency(doc_count, frequency)
        for term, frequency in df.items()
        if term not in pruned_terms
    }

    doc_vectors: Dict[str, SparseVector] = {}
    doc_norms: SparseVector = {}
    names: Dict[str, str] = {}
    for document in documents:
        vector = tfidf_vector(document.terms, idf)
        doc_vectors[document.slug] = vector
        doc_norms[document.slug] = vector_norm(vector)
        names[document.slug] = document.display_name

    return VSMIndex(
        vocabulary=vocabulary,
        idf=idf,
        doc_vectors=doc_vectors,
        doc_norms=doc_norms,
        documents=names,
        mode=mode,
        metadata={
            "document_count": doc_count,
            "vocabulary_size": len(vocabulary),
            "max_df_ratio": max_df_ratio,
            "pruned_high_df_terms": len(pruned_terms),
            "vsm_preprocessing": "indexing_nodes_term_context",
            **(metadata or {}),
        },
    )


def tfidf_vector(counts: Mapping[str, int], idf: Mapping[str, float]) -> SparseVector:
    return {
        term: float(count) * float(idf.get(term, 0.0))
        for term, count in counts.items()
        if count > 0 and idf.get(term, 0.0) > 0
    }


def vector_norm(vector: Mapping[str, float]) -> float:
    return math.sqrt(sum(weight * weight for weight in vector.values()))


def cosine_similarity(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    left_norm = vector_norm(left)
    right_norm = vector_norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    if len(left) > len(right):
        left, right = right, left
    dot = sum(weight * right.get(term, 0.0) for term, weight in left.items())
    return dot / (left_norm * right_norm)


def pcc_similarity(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if not left or not right:
        return 0.0
    if dict(left) == dict(right):
        return 1.0

    terms = sorted(set(left) | set(right))
    if not terms:
        return 0.0

    left_values = [left.get(term, 0.0) for term in terms]
    right_values = [right.get(term, 0.0) for term in terms]
    left_mean = sum(left_values) / len(left_values)
    right_mean = sum(right_values) / len(right_values)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left_values, right_values)
    )
    left_den = math.sqrt(sum((value - left_mean) ** 2 for value in left_values))
    right_den = math.sqrt(sum((value - right_mean) ** 2 for value in right_values))
    if left_den == 0.0 or right_den == 0.0:
        return 0.0
    return numerator / (left_den * right_den)


def sparse_similarity(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    metric: str = "cosine",
) -> float:
    if metric == "cosine":
        return cosine_similarity(left, right)
    if metric == "pcc":
        return pcc_similarity(left, right)
    raise ValueError(f"Unsupported VSM metric '{metric}'.")


def query_counts(query: str) -> TermCounts:
    return _count_terms(normalize_terms(query))


def _expand_query_to_index_units(
    counts: Mapping[str, int],
    index: VSMIndex,
) -> TermCounts:
    """Map free-text query terms onto field-mode (path:term) indexing units."""
    if index.mode != "field":
        return dict(counts)

    expanded: TermCounts = {}
    vocabulary = set(index.vocabulary) | set(index.idf)
    for term, count in counts.items():
        for vocab_term in vocabulary:
            if vocab_term == term or vocab_term.endswith(f":{term}"):
                expanded[vocab_term] = max(expanded.get(vocab_term, 0), int(count))
    return expanded


def query_vector(query: str, index: VSMIndex) -> SparseVector:
    counts = _expand_query_to_index_units(query_counts(query), index)
    return tfidf_vector(counts, index.idf)


def _display_term(term: str) -> str:
    return term.split(":", 1)[-1] if ":" in term else term


def matched_terms(left: Mapping[str, float], right: Mapping[str, float], *, limit: int = 10) -> List[str]:
    terms: List[str] = []
    for term in left:
        if term not in right:
            continue
        display = _display_term(term)
        if is_meaningful_vsm_token(display):
            terms.append(display)
    return sorted(set(terms))[:limit]


def count_shared_meaningful_terms(left: Mapping[str, float], right: Mapping[str, float]) -> int:
    return len(matched_terms(left, right, limit=10_000))


def rank_query(
    index: VSMIndex,
    query: str,
    *,
    top_k: int = 5,
    metric: str = "cosine",
) -> List[VSMSearchResult]:
    query_vec = query_vector(query, index)
    results: List[VSMSearchResult] = []
    for slug, vector in index.doc_vectors.items():
        score = sparse_similarity(query_vec, vector, metric=metric)
        results.append(
            VSMSearchResult(
                slug=slug,
                display_name=index.documents.get(slug, slug.replace("_", " ").title()),
                score=score,
                matched_terms=matched_terms(query_vec, vector),
            )
        )
    results.sort(key=lambda item: item.score, reverse=True)
    return results[:top_k]


def rank_document(
    index: VSMIndex,
    slug: str,
    *,
    top_k: int = 5,
    metric: str = "cosine",
    min_shared_meaningful_terms: int = MIN_SHARED_MEANINGFUL_TERMS,
) -> List[VSMSearchResult]:
    if slug not in index.doc_vectors:
        raise ValueError(f"No VSM vector for country: {slug}")

    source = index.doc_vectors[slug]
    results: List[VSMSearchResult] = []
    for other_slug, vector in index.doc_vectors.items():
        if other_slug == slug:
            continue
        shared_terms = matched_terms(source, vector, limit=10_000)
        if len(shared_terms) < min_shared_meaningful_terms:
            continue
        score = sparse_similarity(source, vector, metric=metric)
        results.append(
            VSMSearchResult(
                slug=other_slug,
                display_name=index.documents.get(other_slug, other_slug.replace("_", " ").title()),
                score=score,
                matched_terms=shared_terms[:10],
            )
        )
    results.sort(key=lambda item: item.score, reverse=True)
    return results[:top_k]
