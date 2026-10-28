"""Paths the timing benchmarks share with nothing else.

The benchmarks read their own copies of feeds (``bench_feeds/``) and their own
Overpass cache (``timing_cache/overpass``), so a study sweep can never change what a
benchmark measured, and a fresh timing run never inherits responses fetched for the
study. ``fresh_start.py`` moves the cache aside before a new run.
"""

from pathlib import Path

from iduedu import config

PAPER_DIR = Path(__file__).resolve().parents[1]
BENCH_FEED_DIR = PAPER_DIR / "bench_feeds"
TIMING_CACHE_DIR = PAPER_DIR / "timing_cache"
OVERPASS_CACHE_DIR = TIMING_CACHE_DIR / "overpass"


def use_paper_cache() -> None:
    """Point IduEdu at the timing benchmarks' own Overpass cache."""
    OVERPASS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    config.set_overpass_cache(enabled=True, cache_dir=str(OVERPASS_CACHE_DIR))
