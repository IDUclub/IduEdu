import hashlib
import json
from pathlib import Path

from iduedu import config

logger = config.logger


def get_cache_dir() -> Path | None:
    """Return directory used for Overpass cache or None if caching is disabled."""
    enabled = config.overpass_cache_enabled
    cache_dir = config.overpass_cache_dir

    if not enabled or not cache_dir:
        return None

    p = Path(cache_dir)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception as e:  # pragma: no cover
        logger.warning(f"Failed to create cache dir {p}: {e}")
        return None
    return p


def cache_load(prefix: str, key_src: str):
    """Load a cached JSON object for a given prefix and key source, if it exists."""

    cache_dir = get_cache_dir()
    if cache_dir is None:
        return None

    digest = hashlib.sha1(key_src.encode("utf-8")).hexdigest()
    path = cache_dir / f"{prefix}_{digest}.json"
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # pragma: no cover
        logger.warning(f"Failed to load cache file {path}: {e}")
        return None


def cache_save(prefix: str, key_src: str, obj):
    """Save a JSON-serializable object to cache under a name derived from key source."""

    cache_dir = get_cache_dir()
    if cache_dir is None:
        return

    digest = hashlib.sha1(key_src.encode("utf-8")).hexdigest()
    path = cache_dir / f"{prefix}_{digest}.json"
    tmp = path.with_suffix(path.suffix + ".tmp")

    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
        tmp.replace(path)
    except Exception as e:  # pragma: no cover
        logger.warning(f"Failed to write cache file {path}: {e}")
