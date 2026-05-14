#!/usr/bin/env python
"""
Small diagnostic tests for value-aware TED costs.

These intentionally use one- and two-feature trees so cost-model regressions are
easy to isolate before running country-scale infobox comparisons.
"""
import sys

sys.path.insert(0, "src")

from core.patch.patch import apply_patch, trees_equal
from core.edit_script.edit_script_normalize import normalize_chawathe_edit_script
from core.similarity.ted import ALGORITHM_CHAWATHE, ALGORITHM_NJ, compute_ted
from domain.models.tree import TreeNode


ALGORITHMS = (ALGORITHM_CHAWATHE, ALGORITHM_NJ)


def leaf(label: str, value: str) -> TreeNode:
    return TreeNode(label=label, value=value)


def tree(*children: TreeNode) -> TreeNode:
    return TreeNode(label="root", children=list(children))


def op_kinds(result) -> list[str]:
    return [op.op for op in result.operations]


def assert_patch_matches(source: TreeNode, target: TreeNode, result, algorithm: str) -> None:
    patched = apply_patch(source, result, algorithm=algorithm)
    assert trees_equal(patched, target, algorithm=algorithm)


def test_identity_has_zero_distance_and_no_ops():
    for algorithm in ALGORITHMS:
        source = tree(leaf("population", "1"))
        target = tree(leaf("population", "1"))
        result = compute_ted(source, target, algorithm=algorithm)

        assert result.distance == 0
        assert result.similarity == 1
        assert result.operations == []


def test_numeric_costs_are_magnitude_sensitive():
    for algorithm in ALGORITHMS:
        small = compute_ted(
            tree(leaf("population", "1")),
            tree(leaf("population", "3")),
            algorithm=algorithm,
        )
        large = compute_ted(
            tree(leaf("population", "1")),
            tree(leaf("population", "30")),
            algorithm=algorithm,
        )

        assert small.distance < large.distance
        assert op_kinds(small) == ["update"]
        assert op_kinds(large) == ["update"]


def test_classic_model_keeps_unit_value_updates():
    for algorithm in ALGORITHMS:
        small = compute_ted(
            tree(leaf("population", "1")),
            tree(leaf("population", "3")),
            algorithm=algorithm,
            cost_model="classic",
        )
        large = compute_ted(
            tree(leaf("population", "1")),
            tree(leaf("population", "30")),
            algorithm=algorithm,
            cost_model="classic",
        )

        assert small.distance == 1
        assert large.distance == 1


def test_numeric_formatting_can_be_zero_cost_but_still_patch():
    for algorithm in ALGORITHMS:
        source = tree(leaf("population", "1"))
        target = tree(leaf("population", "1.0"))
        result = compute_ted(source, target, algorithm=algorithm)

        assert result.distance == 0
        assert op_kinds(result) == ["update"]
        assert_patch_matches(source, target, result, algorithm)


def test_string_typo_costs_less_than_different_string():
    for algorithm in ALGORITHMS:
        typo = compute_ted(
            tree(leaf("capital", "Beirutt")),
            tree(leaf("capital", "Beirut")),
            algorithm=algorithm,
        )
        different = compute_ted(
            tree(leaf("capital", "Beirut")),
            tree(leaf("capital", "Tokyo")),
            algorithm=algorithm,
        )

        assert typo.distance < different.distance
        assert op_kinds(typo) == ["update"]
        assert op_kinds(different) == ["update"]


def test_label_mismatch_costs_more_than_small_same_label_change():
    for algorithm in ALGORITHMS:
        same_label_change = compute_ted(
            tree(leaf("population", "1")),
            tree(leaf("population", "3")),
            algorithm=algorithm,
        )
        label_change = compute_ted(
            tree(leaf("population", "3")),
            tree(leaf("area", "3")),
            algorithm=algorithm,
        )

        assert same_label_change.distance < label_change.distance


def test_two_feature_tree_with_one_changed_value_has_one_update():
    for algorithm in ALGORITHMS:
        source = tree(leaf("population", "1"), leaf("area", "10"))
        target = tree(leaf("population", "3"), leaf("area", "10"))
        result = compute_ted(source, target, algorithm=algorithm)

        assert op_kinds(result) == ["update"]
        assert_patch_matches(source, target, result, algorithm)


def test_two_feature_missing_and_added_feature_shapes():
    for algorithm in ALGORITHMS:
        missing = compute_ted(
            tree(leaf("population", "1"), leaf("area", "10")),
            tree(leaf("population", "1")),
            algorithm=algorithm,
        )
        added = compute_ted(
            tree(leaf("population", "1")),
            tree(leaf("population", "1"), leaf("area", "10")),
            algorithm=algorithm,
        )

        assert op_kinds(missing) == (["delete"] if algorithm == ALGORITHM_CHAWATHE else ["delete_tree"])
        assert op_kinds(added) == (["insert"] if algorithm == ALGORITHM_CHAWATHE else ["insert_tree"])


def test_nj_repeated_subtree_insert_stays_unit_cost_in_classic_model():
    repeated = TreeNode(label="item", children=[leaf("name", "x")])
    source = tree(repeated)
    target = tree(repeated, TreeNode(label="item", children=[leaf("name", "x")]))

    result = compute_ted(source, target, algorithm=ALGORITHM_NJ, cost_model="classic")

    assert result.distance == 1
    assert op_kinds(result) == ["insert_tree"]


def test_chawathe_raw_label_update_is_split_only_for_display():
    source = tree(leaf("population", "3"))
    target = tree(leaf("area", "3"))
    result = compute_ted(source, target, algorithm=ALGORITHM_CHAWATHE)
    raw_ops = [op.to_dict() for op in result.operations]
    display_ops = normalize_chawathe_edit_script(source, raw_ops)

    assert op_kinds(result) == ["update"]
    assert [op["op"] for op in display_ops] == ["delete", "insert"]


if __name__ == "__main__":
    test_identity_has_zero_distance_and_no_ops()
    test_numeric_costs_are_magnitude_sensitive()
    test_classic_model_keeps_unit_value_updates()
    test_numeric_formatting_can_be_zero_cost_but_still_patch()
    test_string_typo_costs_less_than_different_string()
    test_label_mismatch_costs_more_than_small_same_label_change()
    test_two_feature_tree_with_one_changed_value_has_one_update()
    test_two_feature_missing_and_added_feature_shapes()
    test_nj_repeated_subtree_insert_stays_unit_cost_in_classic_model()
    test_chawathe_raw_label_update_is_split_only_for_display()
    print("All similarity cost-model diagnostic tests passed")
