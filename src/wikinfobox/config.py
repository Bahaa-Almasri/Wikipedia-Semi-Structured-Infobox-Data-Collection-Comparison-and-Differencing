from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    data_root: Path = PROJECT_ROOT / "data"
    raw_html_dir: Path = data_root / "raw_html"
    json_dir: Path = data_root / "json"
    trees_dir: Path = data_root / "trees"


@dataclass(frozen=True)
class WikipediaConfig:
    base_url: str = "https://en.wikipedia.org"
    un_member_states_url: str = (
        "https://en.wikipedia.org/wiki/Member_states_of_the_United_Nations"
    )
    request_timeout: int = 15
    max_retries: int = 3
    backoff_factor: float = 0.5
    user_agent: str = (
        "WikipediaInfoboxCollector/1.0 (student academic project)"
    )


PATHS = Paths()
WIKIPEDIA = WikipediaConfig()

