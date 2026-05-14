from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from re import fullmatch
from typing import Any, Optional, Union


_NUMERIC_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"


def similarity_from_distance(distance: float, size_a: int, size_b: int) -> float:
    """Convert edit distance to similarity in [0, 1]: 1 - distance / (size_a + size_b)."""
    denom = size_a + size_b
    if denom == 0:
        return 1.0
    return max(0.0, 1.0 - (distance / denom))


def _normalize_text(value: Any, *, empty_is_none: bool = True) -> Optional[str]:
    if value is None:
        return None
    text = " ".join(str(value).strip().split())
    if empty_is_none and text == "":
        return None
    return text.casefold()


def _parse_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, Real):
        return float(value)
    if value is None:
        return None
    text = str(value).strip().replace(",", "").replace("\u00a0", "")
    if not fullmatch(_NUMERIC_RE, text):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _numeric_distance(left: float, right: float) -> float:
    if left == right:
        return 0.0
    denominator = max(abs(left), abs(right), 1.0)
    return min(1.0, abs(left - right) / denominator)


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _string_distance(left: str, right: str) -> float:
    if left == right:
        return 0.0
    denominator = max(len(left), len(right), 1)
    return _levenshtein_distance(left, right) / denominator


@dataclass(frozen=True)
class CostModel:
    """
    Configurable edit costs shared by Chawathe and Nierman-Jagadish TED.

    The classic model preserves unit update costs. The value-aware model keeps
    insert/delete as unit operations while making same-label value updates
    proportional to numeric or textual distance.
    """

    name: str = "value_aware"
    insert_delete_cost: float = 1.0
    label_mismatch_cost: float = 1.0
    value_mismatch_cost: float = 1.0
    normalize_values: bool = True
    empty_is_none: bool = True

    def insert_cost(self, node: Any) -> float:
        return self.insert_delete_cost

    def delete_cost(self, node: Any) -> float:
        return self.insert_delete_cost

    def nodes_equal(self, left: Any, right: Any) -> bool:
        return self.update_cost(left, right) == 0.0

    def update_cost(self, left: Any, right: Any) -> float:
        if left.label != right.label:
            return self.label_mismatch_cost

        left_value = getattr(left, "value", None)
        right_value = getattr(right, "value", None)
        if not self.normalize_values:
            return 0.0 if left_value == right_value else self.value_mismatch_cost

        left_text = _normalize_text(left_value, empty_is_none=self.empty_is_none)
        right_text = _normalize_text(right_value, empty_is_none=self.empty_is_none)
        if left_text == right_text:
            return 0.0
        if left_text is None or right_text is None:
            return self.value_mismatch_cost

        left_number = _parse_number(left_value)
        right_number = _parse_number(right_value)
        if left_number is not None and right_number is not None:
            return min(
                self.value_mismatch_cost,
                self.value_mismatch_cost * _numeric_distance(left_number, right_number),
            )

        return min(
            self.value_mismatch_cost,
            self.value_mismatch_cost * _string_distance(left_text, right_text),
        )


CLASSIC_COST_MODEL = CostModel(name="classic", normalize_values=False)
VALUE_AWARE_COST_MODEL = CostModel(name="value_aware")

CostModelInput = Union[CostModel, str, None]


def get_cost_model(cost_model: CostModelInput = None) -> CostModel:
    if cost_model is None:
        return VALUE_AWARE_COST_MODEL
    if isinstance(cost_model, CostModel):
        return cost_model

    key = str(cost_model).strip().lower().replace("-", "_")
    if key in {"classic", "unit", "lecture"}:
        return CLASSIC_COST_MODEL
    if key in {"value_aware", "value", "default"}:
        return VALUE_AWARE_COST_MODEL
    raise ValueError(f"Unknown TED cost model '{cost_model}'.")


def subtree_insert_cost(root: Any, cost_model: CostModel) -> float:
    return cost_model.insert_cost(root) + sum(
        subtree_insert_cost(child, cost_model) for child in getattr(root, "children", [])
    )


def subtree_delete_cost(root: Any, cost_model: CostModel) -> float:
    return cost_model.delete_cost(root) + sum(
        subtree_delete_cost(child, cost_model) for child in getattr(root, "children", [])
    )
