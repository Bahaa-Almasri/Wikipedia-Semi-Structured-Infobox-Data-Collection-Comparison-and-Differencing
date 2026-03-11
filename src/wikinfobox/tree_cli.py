from __future__ import annotations

import argparse
import json

from .config import PATHS
from .storage import json_path_for_slug, tree_path_for_slug
from .tree import TreeNode, pretty_print
from .tree_builder import build_and_save_tree_for_slug, build_and_save_trees_for_all


def _describe_tree_source(slug: str) -> str:
    json_path = json_path_for_slug(slug)
    if not json_path.exists():
        return "unknown"

    data = json.loads(json_path.read_text(encoding="utf-8"))
    normalized = data.get("normalized", {}) or {}

    if normalized.get("comparison_fields"):
        return "normalized.comparison_fields"
    if normalized.get("fields"):
        return "normalized.fields (fallback)"
    return "no usable fields found"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build tree representations from Wikipedia country JSON documents.",
    )
    parser.add_argument(
        "--slug",
        help="If provided, only build the tree for this country slug.",
    )
    parser.add_argument(
        "--print",
        dest="print_slug",
        help="Pretty-print the tree for this slug (implies building it first).",
    )
    args = parser.parse_args()

    if args.print_slug:
        slug = args.print_slug
        path = build_and_save_tree_for_slug(slug)
        if path is None:
            print(f"No JSON document found for slug '{slug}'.")
            return

        source_used = _describe_tree_source(slug)
        print(f"Tree saved to {path}")
        print(f"Tree source: {source_used}")

        tree_json = json.loads(tree_path_for_slug(slug).read_text(encoding="utf-8"))
        tree_root = TreeNode.from_dict(tree_json)
        print("\nPretty-printed tree (truncated to depth 3):")
        pretty_print(tree_root, max_depth=3)
        return

    if args.slug:
        slug = args.slug
        path = build_and_save_tree_for_slug(slug)
        if path is None:
            print(f"No JSON document found for slug '{slug}'.")
        else:
            source_used = _describe_tree_source(slug)
            print(f"Tree saved to {path}")
            print(f"Tree source: {source_used}")
        return

    paths = build_and_save_trees_for_all()
    print(f"Built trees for {len(paths)} countries. Output directory: {PATHS.trees_dir}")


if __name__ == "__main__":
    main()

