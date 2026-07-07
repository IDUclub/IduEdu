# pylint: disable=redefined-outer-name, unused-argument

import pytest

from iduedu.overpass import cache
from iduedu.overpass.cache import cache_load, cache_save, get_cache_dir

pytestmark = pytest.mark.unit


@pytest.fixture
def cache_config(monkeypatch, tmp_path):
    monkeypatch.setattr(cache.config, "overpass_cache_enabled", True, raising=False)
    monkeypatch.setattr(cache.config, "overpass_cache_dir", str(tmp_path), raising=False)
    return tmp_path


def test_get_cache_dir_none_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(cache.config, "overpass_cache_enabled", False, raising=False)
    monkeypatch.setattr(cache.config, "overpass_cache_dir", str(tmp_path), raising=False)
    assert get_cache_dir() is None


def test_get_cache_dir_none_when_dir_empty(monkeypatch):
    monkeypatch.setattr(cache.config, "overpass_cache_enabled", True, raising=False)
    monkeypatch.setattr(cache.config, "overpass_cache_dir", "", raising=False)
    assert get_cache_dir() is None


def test_cache_load_missing_returns_none(cache_config):
    assert cache_load("routes", "some-key-that-was-never-saved") is None


def test_cache_save_and_load_round_trip(cache_config):
    payload = {"elements": [{"type": "way", "id": 1}], "meta": "ok"}
    cache_save("network", "key-src", payload)

    loaded = cache_load("network", "key-src")
    assert loaded == payload


def test_cache_load_disabled_returns_none(monkeypatch, cache_config):
    cache_save("network", "key-src", {"a": 1})
    # once caching is disabled, load must not read the file back
    monkeypatch.setattr(cache.config, "overpass_cache_enabled", False, raising=False)
    assert cache_load("network", "key-src") is None
