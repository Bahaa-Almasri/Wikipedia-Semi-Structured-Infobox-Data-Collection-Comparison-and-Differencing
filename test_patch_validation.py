#!/usr/bin/env python
"""
Validation test: Lebanon -> Switzerland patch with area.total_km2 excluded.
Expected: ALL fields updated to Switzerland EXCEPT fields.area.total_km2.
"""
import sys
sys.path.insert(0, "src")

from core.patch.path_patch import (
    normalize_path,
    get_value_from_tree,
    get_values_from_tree,
    build_patch_from_features,
    apply_patch_to_tree,
)


def _make_tree(label: str, children: list) -> dict:
    return {"label": label, "value": None, "children": children}


def _make_leaf(label: str, value: str) -> dict:
    return {"label": label, "value": value, "children": []}


def test_normalize_path():
    assert normalize_path("area.total_km2") == "fields.area.total_km2"
    assert normalize_path("fields.area.total_km2") == "fields.area.total_km2"
    assert normalize_path("meta.country_name") == "meta.country_name"
    assert normalize_path("economy.gdp_nominal_total") == "fields.economy.gdp_nominal_total"
    print("✓ normalize_path works")


def test_patch_with_exclusion():
    # Lebanon-like source tree
    source = _make_tree("lebanon", [
        _make_tree("meta", [_make_leaf("country_name", "Lebanon")]),
        _make_tree("fields", [
            _make_tree("area", [
                _make_leaf("total_km2", "10452"),
                _make_leaf("water_percent", "1.8"),
            ]),
            _make_tree("economy", [_make_leaf("gdp_nominal_total", "18B")]),
        ]),
    ])

    # Switzerland-like target tree
    target = _make_tree("switzerland", [
        _make_tree("meta", [_make_leaf("country_name", "Switzerland")]),
        _make_tree("fields", [
            _make_tree("area", [
                _make_leaf("total_km2", "41285"),
                _make_leaf("water_percent", "4.2"),
            ]),
            _make_tree("economy", [_make_leaf("gdp_nominal_total", "818B")]),
        ]),
    ])

    # All paths minus area.total_km2
    all_paths = [
        "meta.country_name",
        "fields.area.total_km2",
        "fields.area.water_percent",
        "fields.economy.gdp_nominal_total",
    ]
    excluded = ["area.total_km2"]
    excluded_normalized = [normalize_path(e) for e in excluded]

    def _is_excluded(path: str) -> bool:
        norm = normalize_path(path)
        if norm in excluded_normalized:
            return True
        for ex in excluded_normalized:
            if norm.startswith(ex + ".") or norm == ex:
                return True
        return False

    selected = [p for p in all_paths if not _is_excluded(p)]
    assert "fields.area.total_km2" not in selected
    assert "fields.area.water_percent" in selected
    assert "fields.economy.gdp_nominal_total" in selected

    patch = build_patch_from_features(source, target, selected)
    patched = apply_patch_to_tree(source, patch)

    # Verify area.total_km2 stayed Lebanon
    val = get_value_from_tree(patched, "fields.area.total_km2")
    assert val == "10452", f"area.total_km2 should stay Lebanon (10452), got {val}"

    # Verify other fields updated to Switzerland
    assert get_value_from_tree(patched, "meta.country_name") == "Switzerland"
    assert get_value_from_tree(patched, "fields.area.water_percent") == "4.2"
    assert get_value_from_tree(patched, "fields.economy.gdp_nominal_total") == "818B"

    print("✓ Patch with exclusion: area.total_km2 preserved, others updated")


def test_short_path_normalization():
    """Paths without fields. prefix should be found after normalization."""
    tree = _make_tree("root", [
        _make_tree("fields", [
            _make_tree("area", [_make_leaf("total_km2", "10452")]),
        ]),
    ])
    # Without normalization, "area.total_km2" would fail
    val_bad = get_value_from_tree(tree, "area.total_km2")
    val_good = get_value_from_tree(tree, normalize_path("area.total_km2"))
    assert val_bad is None
    assert val_good == "10452"
    print("✓ Short paths normalized correctly")


if __name__ == "__main__":
    test_normalize_path()
    test_short_path_normalization()
    test_patch_with_exclusion()
    print("\n✅ All validation tests passed")
