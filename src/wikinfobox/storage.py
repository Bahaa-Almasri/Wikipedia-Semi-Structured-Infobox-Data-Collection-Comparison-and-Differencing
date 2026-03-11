from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from .config import PATHS


def ensure_data_dirs() -> None:
    PATHS.data_root.mkdir(parents=True, exist_ok=True)
    PATHS.raw_html_dir.mkdir(parents=True, exist_ok=True)
    PATHS.json_dir.mkdir(parents=True, exist_ok=True)
    PATHS.trees_dir.mkdir(parents=True, exist_ok=True)


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_path_for_slug(slug: str) -> Path:
    return PATHS.json_dir / f"{slug}.json"


def raw_html_path_for_slug(slug: str) -> Path:
    return PATHS.raw_html_dir / f"{slug}.html"


def tree_path_for_slug(slug: str) -> Path:
    return PATHS.trees_dir / f"{slug}.json"


def write_json_document(slug: str, document: Dict) -> None:
    path = json_path_for_slug(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)


def write_tree_document(slug: str, document: Dict) -> None:
    path = tree_path_for_slug(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)

