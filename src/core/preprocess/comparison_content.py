"""
Shared comparison payload for structure-aware TED and content-based VSM.

Both pipelines index the same curated ``comparison_fields`` tree produced during
country preprocessing (see ``build_comparison_fields`` in normalization.py).
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from core.preprocess.normalization import build_comparison_fields
from domain.schemas.normalized_field import NormalizedField


def normalized_fields_from_document(
    document: Mapping[str, Any],
) -> Dict[str, NormalizedField]:
    """Load ``normalized.fields`` as NormalizedField instances."""
    normalized = document.get("normalized") or {}
    raw = normalized.get("fields") or {}
    if not isinstance(raw, dict):
        return {}

    fields: Dict[str, NormalizedField] = {}
    for key, data in raw.items():
        if isinstance(data, NormalizedField):
            fields[str(key)] = data
        elif isinstance(data, dict):
            fields[str(key)] = NormalizedField(
                raw_label=str(data.get("raw_label") or key),
                text=str(data.get("text") or ""),
                tokens=list(data.get("tokens") or []),
                numbers=list(data.get("numbers") or []),
            )
    return fields


def get_comparison_fields(
    document: Mapping[str, Any],
    *,
    rebuild_if_missing: bool = True,
) -> Dict[str, Any]:
    """
    Return the comparison payload for a country document.

    TED trees and VSM indexing both use this structure. When ``comparison_fields``
    is absent on older documents, it can be rebuilt from ``normalized.fields``.
    """
    normalized = document.get("normalized") or {}
    existing = normalized.get("comparison_fields")
    if isinstance(existing, dict) and existing:
        return dict(existing)

    if not rebuild_if_missing:
        return {}

    fields = normalized_fields_from_document(document)
    if not fields:
        return {}

    meta = document.get("meta") or {}
    country_name = meta.get("country_name") if isinstance(meta, dict) else None
    return build_comparison_fields(fields, country_name=country_name)
