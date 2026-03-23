"""
Count operations in serialized edit scripts (list of op dicts).

Used for API and UI metrics that reflect the algorithm-native script (Chawathe, NJ, Zhang–Shasha),
separate from semantic path diffs between trees.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Union

from domain.models.edit_script import NJTedResult, TedResult


def summarize_raw_edit_script_operations(operations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize a raw TED edit script (as returned by the API after normalization).

    - insert / insert_tree -> inserts
    - delete / delete_tree -> deletes
    - update -> updates
    - map -> mappings (Zhang–Shasha alignments; not part of edit_script_length)

    edit_script_length = inserts + deletes + updates (mappings excluded by design).
    """
    inserts = 0
    deletes = 0
    updates = 0
    mappings = 0
    for op in operations:
        kind = str(op.get("op") or "").lower()
        if kind in ("insert", "insert_tree"):
            inserts += 1
        elif kind in ("delete", "delete_tree"):
            deletes += 1
        elif kind == "update":
            updates += 1
        elif kind == "map":
            mappings += 1

    length = inserts + deletes + updates
    return {
        "edit_script_length": length,
        "inserts": inserts,
        "deletes": deletes,
        "updates": updates,
        "mappings": mappings,
        "operation_count_total": len(operations),
        "summary_note": (
            "edit_script_length is inserts + deletes + updates in the algorithm script returned "
            "for this request. Mappings (Zhang–Shasha) are node alignments, not insert/delete/update "
            "steps, and are counted separately. This breakdown can differ from semantic path-level "
            "diffs and from TED distance."
        ),
    }


def summarize_semantic_diff_operations(operations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize semantic (path-based) diff ops: insert / delete / update on dotted paths."""
    inserts = 0
    deletes = 0
    updates = 0
    for op in operations:
        kind = str(op.get("op") or "").lower()
        if kind == "insert":
            inserts += 1
        elif kind == "delete":
            deletes += 1
        elif kind == "update":
            updates += 1

    total = len(operations)
    return {
        "model": "semantic_path_diff",
        "inserts": inserts,
        "deletes": deletes,
        "updates": updates,
        "total": total,
        "summary_note": (
            "Path-level semantic differences between the two trees; computed independently of "
            "the TED algorithm (Chawathe / NJ / Zhang–Shasha)."
        ),
    }


def raw_edit_ops_from_ted_result(ted_result: Union[TedResult, NJTedResult]) -> List[Dict[str, Any]]:
    """
    Serialize TED operations like the API native script. For Zhang–Shasha, the stored script is
    empty and alignments live in zhang_shasha_mappings — expose those as map ops for counting.
    """
    if isinstance(ted_result, NJTedResult):
        return [o.to_dict() for o in ted_result.operations]

    ops = [o.to_dict() for o in ted_result.operations]
    if ops:
        return ops
    zs = getattr(ted_result, "zhang_shasha_mappings", None)
    if zs:
        return [
            {"op": "map", "source_id": m["source_id"], "target_id": m["target_id"]}
            for m in zs
        ]
    return []
