"""
MongoDB-only storage. All data is stored in and retrieved from MongoDB.
No local file paths or disk storage.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List, Optional

from core.data.config import MONGO

if TYPE_CHECKING:
    from pymongo.collection import Collection
    from pymongo.database import Database
    from pymongo.mongo_client import MongoClient


def _get_mongo_client() -> "MongoClient":
    from pymongo import MongoClient as _MongoClient
    if not MONGO.uri:
        raise RuntimeError("MONGODB_URI must be set. All storage is MongoDB-only.")
    return _MongoClient(MONGO.uri)


def _get_db() -> "Database":
    return _get_mongo_client()[MONGO.database]


def _get_collection() -> "Collection":
    return _get_db()[MONGO.collection]


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def list_slugs() -> List[str]:
    """Return sorted list of country slugs from MongoDB."""
    coll = _get_collection()
    return sorted(coll.distinct("_id"))


def read_json_document(slug: str) -> Optional[Dict]:
    """Load JSON document for a country by slug from MongoDB."""
    coll = _get_collection()
    doc = coll.find_one({"_id": slug})
    if doc is None:
        return None
    return {k: v for k, v in doc.items() if k != "_id"}


def read_tree_document(slug: str) -> Optional[Dict]:
    """Load tree document for a country by slug from MongoDB."""
    doc = read_json_document(slug)
    if doc is None:
        return None
    return doc.get("tree")


def read_raw_html(slug: str) -> Optional[str]:
    """Load raw infobox HTML for a country from MongoDB."""
    doc = read_json_document(slug)
    if doc is None:
        return None
    raw = doc.get("raw") or {}
    return raw.get("infobox_html")


def write_json_document(slug: str, document: Dict) -> None:
    """Upsert full JSON document for a country in MongoDB."""
    coll = _get_collection()
    doc_to_save = {"_id": slug, **document}
    coll.replace_one({"_id": slug}, doc_to_save, upsert=True)


def write_tree_document(slug: str, document: Dict) -> None:
    """Update the tree field for a country document in MongoDB."""
    coll = _get_collection()
    coll.update_one(
        {"_id": slug},
        {"$set": {"tree": document}},
        upsert=False,
    )
