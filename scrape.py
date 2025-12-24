"""
Scrape Bangla nostalgia comments from YouTube into a CSV dataset.

Dataset schema (strict order):
id, text, label, clean_text, reference_time
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import sys
import time
import unicodedata
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm import tqdm

from config import CONFIG, ScrapeConfig

CSV_FIELDS = ["id", "text", "label", "clean_text", "reference_time"]

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
BENGALI_CHAR_RE = re.compile(r"[ঀ-৿]")
WHITESPACE_RE = re.compile(r"\s+")


def build_youtube_client(api_key: str):
    if not api_key:
        sys.exit("Missing YOUTUBE_API_KEY env var. Export it before running.")
    return build("youtube", "v3", developerKey=api_key, cache_discovery=False)


def backoff_sleep(attempt: int) -> float:
    # Exponential backoff with a little jitter.
    return min(60.0, (2 ** attempt) + random.uniform(0, 0.5))


def execute_with_backoff(call_factory: Callable[[], Any], *, max_attempts: int, action: str):
    for attempt in range(max_attempts):
        try:
            return call_factory().execute()
        except HttpError as exc:
            if attempt == max_attempts - 1:
                raise
            time.sleep(backoff_sleep(attempt))
        except Exception:
            if attempt == max_attempts - 1:
                raise
            time.sleep(backoff_sleep(attempt))
    raise RuntimeError(f"Failed to {action} after {max_attempts} attempts.")


def iter_video_ids(youtube, config: ScrapeConfig) -> Iterator[str]:
    seen: set[str] = set()
    total_limit = config.MAX_VIDEOS
    for query in config.queries:
        if len(seen) >= total_limit:
            break
        next_page = None
        while len(seen) < total_limit:
            def make_request():
                params = {
                    "q": query,
                    "part": "id",
                    "type": "video",
                    "maxResults": 50,
                    "order": config.ORDER,
                }
                if config.RELEVANCE_LANGUAGE:
                    params["relevanceLanguage"] = config.RELEVANCE_LANGUAGE
                if config.REGION_CODE:
                    params["regionCode"] = config.REGION_CODE
                if next_page:
                    params["pageToken"] = next_page
                return youtube.search().list(**params)

            response = execute_with_backoff(
                make_request,
                max_attempts=config.RETRY_MAX_ATTEMPTS,
                action="search videos",
            )
            for item in response.get("items", []):
                vid = item.get("id", {}).get("videoId")
                if not vid or vid in seen:
                    continue
                seen.add(vid)
                yield vid
                if len(seen) >= total_limit:
                    break
            next_page = response.get("nextPageToken")
            if not next_page:
                break


def iter_comments(youtube, video_id: str, config: ScrapeConfig) -> Iterator[dict[str, Any]]:
    fetched = 0
    page_token = None
    while fetched < config.MAX_COMMENTS_PER_VIDEO:
        def make_request():
            params = {
                "part": "snippet",
                "videoId": video_id,
                "textFormat": "plainText",
                "order": config.ORDER,
                "maxResults": 100,
            }
            if page_token:
                params["pageToken"] = page_token
            return youtube.commentThreads().list(**params)

        try:
            response = execute_with_backoff(
                make_request,
                max_attempts=config.RETRY_MAX_ATTEMPTS,
                action="fetch comments",
            )
        except HttpError as exc:
            # Some videos block comments or have restricted access.
            return

        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            tlc = snippet.get("topLevelComment", {}).get("snippet", {})
            raw_text = tlc.get("textOriginal") or ""
            published = tlc.get("publishedAt") or ""
            comment_id = item.get("id") or tlc.get("id")
            if comment_id:
                yield {"id": comment_id, "text": raw_text, "publishedAt": published}
                fetched += 1
                if fetched >= config.MAX_COMMENTS_PER_VIDEO:
                    break
        page_token = response.get("nextPageToken")
        if not page_token:
            break


def strip_symbols(text: str) -> str:
    filtered_chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith(("So", "Co", "Cs")):
            continue
        filtered_chars.append(ch)
    return "".join(filtered_chars)


def normalize_comment(text: str) -> str:
    text = URL_RE.sub(" ", text)
    text = ZERO_WIDTH_RE.sub("", text)
    text = strip_symbols(text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def has_min_bengali_chars(text: str, minimum: int) -> bool:
    return len(BENGALI_CHAR_RE.findall(text)) >= minimum


def compute_clean_hash(clean_text: str) -> str:
    normalized = clean_text.lower().strip()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def load_existing(path: Path, config: ScrapeConfig) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    text_hashes: set[str] = set()
    if not path.exists() or path.stat().st_size == 0:
        return ids, text_hashes

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("id")
            clean_text = row.get("clean_text", "")
            if cid:
                ids.add(cid)
            if config.DEDUPE_BY_TEXT_HASH and clean_text:
                text_hashes.add(compute_clean_hash(clean_text))
    return ids, text_hashes


def should_keep(clean_text: str, config: ScrapeConfig) -> bool:
    if not clean_text:
        return False
    words = [w for w in clean_text.split(" ") if w]
    if len(words) < config.MIN_WORDS:
        return False
    if not has_min_bengali_chars(clean_text, config.MIN_BN_CHARS):
        return False
    return True


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Scrape YouTube comments into a labeled CSV.")
    parser.add_argument(
        "--label",
        type=int,
        help="Ground-truth label to assign to every scraped row. Defaults to config.DEFAULT_LABEL.",
    )
    return parser.parse_args(argv)


def main(label_override: int | None = None):
    config = CONFIG
    youtube = build_youtube_client(config.YOUTUBE_API_KEY)
    output_path = config.out_csv_path

    existing_ids, existing_hashes = load_existing(output_path, config)
    dataset_size = len(existing_ids)
    if dataset_size >= config.TARGET_ROWS:
        print(f"Dataset already has {dataset_size} rows; target reached.")
        return

    file_exists = output_path.exists() and output_path.stat().st_size > 0
    collected_this_run = 0

    with output_path.open("a", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()

        row_progress = tqdm(
            total=config.TARGET_ROWS,
            initial=dataset_size,
            desc="rows collected",
        )

        for video_id in iter_video_ids(youtube, config):
            if dataset_size >= config.TARGET_ROWS:
                break

            for comment in iter_comments(youtube, video_id, config):
                comment_id = comment["id"]
                if comment_id in existing_ids:
                    continue

                clean = normalize_comment(comment["text"])
                if not should_keep(clean, config):
                    continue

                text_hash = compute_clean_hash(clean)
                if config.DEDUPE_BY_TEXT_HASH and text_hash in existing_hashes:
                    continue

                label_value = config.DEFAULT_LABEL if label_override is None else label_override
                row = {
                    "id": comment_id,
                    "text": comment["text"],
                    "label": label_value,
                    "clean_text": clean,
                    "reference_time": comment["publishedAt"],
                }
                writer.writerow(row)
                csvfile.flush()

                existing_ids.add(comment_id)
                if config.DEDUPE_BY_TEXT_HASH:
                    existing_hashes.add(text_hash)
                collected_this_run += 1
                dataset_size += 1
                row_progress.update(1)

                if dataset_size >= config.TARGET_ROWS:
                    break

            time.sleep(config.SLEEP_BETWEEN_VIDEOS_SEC)

        row_progress.close()

    print(
        f"Finished. Added {collected_this_run} rows. "
        f"Current dataset size: {dataset_size}"
    )


if __name__ == "__main__":
    try:
        cli_args = parse_args()
        main(label_override=cli_args.label)
    except KeyboardInterrupt:
        print("\nInterrupted. Progress saved.")
