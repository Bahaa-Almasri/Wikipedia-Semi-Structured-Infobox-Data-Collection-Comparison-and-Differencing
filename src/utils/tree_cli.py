from __future__ import annotations

import argparse

from core.data.storage import read_json_document, read_tree_document
from core.preprocess.tree_builder import build_and_save_tree_for_slug, build_and_save_trees_for_all
from domain.models.tree import TreeNode, pretty_print


def _describe_tree_source(slug: str) -> str:
    data = read_json_document(slug)
    if data is None:
        return "unknown"
    normalized = data.get("normalized", {}) or {}
    if normalized.get("comparison_fields"):
        return "normalized.comparison_fields"
    if normalized.get("fields"):
        return "normalized.fields (fallback)"
    return "no usable fields found"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build tree representations from Wikipedia country JSON documents (MongoDB).",
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
        result = build_and_save_tree_for_slug(slug)
        if result is None:
            print(f"No JSON document found for slug '{slug}'.")
            return

        source_used = _describe_tree_source(slug)
        print(f"Tree saved to MongoDB for slug: {result}")
        print(f"Tree source: {source_used}")

        tree_json = read_tree_document(slug)
        if tree_json is None:
            print("Could not load tree.")
            return
        tree_root = TreeNode.from_dict(tree_json)
        print("\nPretty-printed tree (truncated to depth 3):")
        pretty_print(tree_root, max_depth=3)
        return

    if args.slug:
        slug = args.slug
        result = build_and_save_tree_for_slug(slug)
        if result is None:
            print(f"No JSON document found for slug '{slug}'.")
        else:
            source_used = _describe_tree_source(slug)
            print(f"Tree saved to MongoDB for slug: {result}")
            print(f"Tree source: {source_used}")
        return

    written = build_and_save_trees_for_all()
    print(f"Built trees for {len(written)} countries. Stored in MongoDB.")


if __name__ == "__main__":
    main()
