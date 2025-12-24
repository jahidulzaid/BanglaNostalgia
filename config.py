"""
Central configuration for the Bengali nostalgia dataset scraper.

Columns produced:
id, text, label, clean_text, reference_time
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # Load .env file if present



@dataclass(frozen=True)
class ScrapeConfig:
    # --- API ---
    # Prefer environment variable. Never hardcode keys in git.
    YOUTUBE_API_KEY: str = os.getenv("YOUTUBE_API_KEY", "").strip()

    # --- Bangla Trending Keywords ---
    QUERY: str = "শৈশবের দিনগুলো"
    SEARCH_QUERIES: tuple[str, ...] = (
        QUERY,

        "ছোটবেলার স্মৃতি",
        "স্কুল জীবনের দিন",
        "গ্রামের জীবন",
        "আগের দিনের জীবন",

        "পুরনো টিভি অনুষ্ঠান",
        "বাংলাদেশ বেতার",

        "মুক্তিযুদ্ধের স্মৃতি",
        "বাংলাদেশের ইতিহাস",
    )



    TARGET_ROWS: int = 200

    # --- Search breadth & per-video cap ---
    MAX_VIDEOS: int = 200                 # how many videos to scan
    MAX_COMMENTS_PER_VIDEO: int = 20     # max top-level comments per video

    # --- Filtering rules ---
    MIN_WORDS: int = 3
    MIN_BN_CHARS: int = 2                 # minimum Bengali unicode chars required

    # --- Politeness / throttling ---
    SLEEP_BETWEEN_VIDEOS_SEC: float = 0.2
    RETRY_MAX_ATTEMPTS: int = 6

    # --- Output ---
    OUTPUT_DIR: Path = Path("data")
    OUTPUT_FILE: str = "bengali_nostalgia_dataset_with_label.csv"

    # Placeholder label for later annotation:
    DEFAULT_LABEL: int = -1

    # --- Behavior toggles ---
    DEDUPE_BY_TEXT_HASH: bool = True      # additionally remove duplicates by normalized clean_text
    ORDER: str = "relevance"              # "relevance" or "time" for commentThreads order
    RELEVANCE_LANGUAGE: str = "bn"        # hints search to Bangla
    REGION_CODE: str | None = "BD"        # restrict search region; set to None to disable

    @property
    def out_csv_path(self) -> Path:
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return self.OUTPUT_DIR / self.OUTPUT_FILE

    @property
    def queries(self) -> tuple[str, ...]:
        # Keep a stable tuple for iteration; fall back to QUERY if SEARCH_QUERIES is empty.
        return tuple(q for q in self.SEARCH_QUERIES if q.strip()) or (self.QUERY,)


CONFIG = ScrapeConfig()
