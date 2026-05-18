from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from domain.models.vsm import SparseVector, TermCounts, VSMDocument, VSMIndex, VSMSearchResult


_TOKEN_RE = re.compile(r"[0-9A-Za-z]+")
SUPPORTED_VSM_METRICS = {"cosine", "pcc"}
SUPPORTED_VSM_MODES = {"flat", "field"}
SUPPORTED_VSM_DOCUMENT_SOURCES = {"all", "fields", "comparison_fields"}


def normalize_terms(text: Any) -> List[str]:
    """Tokenize text the same way normalized infobox rows are tokenized, but case-folded."""
    if text is None:
        return []
    return [match.group(0).casefold() for match in _TOKEN_RE.finditer(str(text))]


def _add_count(target: TermCounts, term: str, amount: int = 1) -> None:
    if not term:
        return
    target[term] = target.get(term, 0) + amount


def _merge_counts(target: TermCounts, source: Mapping[str, int]) -> None:
    for term, count in source.items():
        _add_count(target, term, int(count))


def _count_terms(terms: Iterable[str]) -> TermCounts:
    return {term: count for term, count in Counter(terms).items() if term}


def _field_label_terms(field_path: str) -> List[str]:
    pieces = re.split(r"[^0-9A-Za-z]+", field_path.replace(".", "_"))
    return [piece.casefold() for piece in pieces if piece]


def _feature_key(feature: str) -> str:
    clean = str(feature).strip().casefold()
    if clean.startswith("fields."):
        clean = clean.removeprefix("fields.")
    return clean.replace(".", "_")


def _feature_matches(field_path: str, features: Optional[Sequence[str]]) -> bool:
    if not features:
        return True
    field_key = _feature_key(field_path)
    for feature in features:
        feature_key = _feature_key(feature)
        if field_key == feature_key:
            return True
        if field_key.endswith(f"_{feature_key}") or feature_key.endswith(f"_{field_key}"):
            return True
        if field_key.split("_")[-1] == feature_key.split("_")[-1]:
            return True
    return False


def _contextualize(field_path: str, terms: TermCounts) -> TermCounts:
    contextual: TermCounts = {}
    clean_path = _feature_key(field_path)
    for term, count in terms.items():
        _add_count(contextual, term, count)
        _add_count(contextual, f"{clean_path}:{term}", count)
    for label_term in _field_label_terms(field_path):
        _add_count(contextual, label_term)
    return contextual


def _flatten_comparison_fields(
    value: Any,
    *,
    path: Tuple[str, ...] = (),
) -> Iterable[Tuple[str, List[str]]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _flatten_comparison_fields(child, path=(*path, str(key)))
        return
    if isinstance(value, list):
        for item in value:
            yield from _flatten_comparison_fields(item, path=path)
        return

    field_path = ".".join(path)
    terms = [*_field_label_terms(field_path), *normalize_terms(value)]
    yield field_path, terms


def vsm_document_from_json(
    slug: str,
    document: Mapping[str, Any],
    *,
    mode: str = "field",
    features: Optional[Sequence[str]] = None,
    source: str = "all",
) -> VSMDocument:
    if mode not in SUPPORTED_VSM_MODES:
        raise ValueError(f"Unsupported VSM mode '{mode}'.")
    if source not in SUPPORTED_VSM_DOCUMENT_SOURCES:
        raise ValueError(f"Unsupported VSM document source '{source}'.")

    meta = document.get("meta") or {}
    display_name = str(meta.get("country_name") or slug.replace("_", " ").title())
    normalized = document.get("normalized") or {}

    field_terms: Dict[str, TermCounts] = {}

    if source in {"all", "fields"}:
        fields = normalized.get("fields") or {}
        for field_key, field_data in fields.items():
            field_path = str(field_key)
            if not _feature_matches(field_path, features):
                continue
            tokens = field_data.get("tokens") if isinstance(field_data, Mapping) else None
            text = field_data.get("text") if isinstance(field_data, Mapping) else ""
            raw_terms = list(tokens or []) or normalize_terms(text)
            counts = _count_terms(
                [*_field_label_terms(field_path), *(str(t).casefold() for t in raw_terms)]
            )
            field_terms[field_path] = _contextualize(field_path, counts) if mode == "field" else counts

    if source in {"all", "comparison_fields"}:
        comparison_fields = normalized.get("comparison_fields") or {}
        for field_path, terms in _flatten_comparison_fields(comparison_fields):
            if not field_path or not _feature_matches(field_path, features):
                continue
            counts = _count_terms(terms)
            existing = field_terms.setdefault(field_path, {})
            _merge_counts(existing, _contextualize(field_path, counts) if mode == "field" else counts)

    all_terms: TermCounts = {}
    for counts in field_terms.values():
        _merge_counts(all_terms, counts)

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
    mode: str = "field",
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


def query_vector(query: str, index: VSMIndex) -> SparseVector:
    counts = query_counts(query)
    return tfidf_vector(counts, index.idf)


def matched_terms(left: Mapping[str, float], right: Mapping[str, float], *, limit: int = 10) -> List[str]:
    terms = [term for term in left if term in right and ":" not in term]
    return sorted(terms)[:limit]


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
) -> List[VSMSearchResult]:
    if slug not in index.doc_vectors:
        raise ValueError(f"No VSM vector for country: {slug}")

    source = index.doc_vectors[slug]
    results: List[VSMSearchResult] = []
    for other_slug, vector in index.doc_vectors.items():
        if other_slug == slug:
            continue
        score = sparse_similarity(source, vector, metric=metric)
        results.append(
            VSMSearchResult(
                slug=other_slug,
                display_name=index.documents.get(other_slug, other_slug.replace("_", " ").title()),
                score=score,
                matched_terms=matched_terms(source, vector),
            )
        )
    results.sort(key=lambda item: item.score, reverse=True)
    return results[:top_k]
