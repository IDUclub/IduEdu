# pylint: disable=redefined-outer-name

import pytest
from loguru import logger

from iduedu._config import Config

# ---------- fixtures ----------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # wipe relevant env so defaults are deterministic per test
    for k in [
        "OVERPASS_URL",
        "OVERPASS_USER_AGENT",
        "OVERPASS_TIMEOUT",
        "OVERPASS_MIN_INTERVAL",
        "OVERPASS_MAX_RETRIES",
        "OVERPASS_BACKOFF_BASE",
        "OVERPASS_DATE",
        "ENABLE_TQDM",
        "LOG_LEVEL",
        "CI",
        "OVERPASS_CACHE_DIR",
        "OVERPASS_CACHE_ENABLED",
    ]:
        monkeypatch.delenv(k, raising=False)
    yield


# ---------- basic config behaviour ----------


def test_defaults_no_env_and_tty(monkeypatch):
    cfg = Config()

    assert cfg.overpass_url == "https://overpass-api.de/api/interpreter"
    assert cfg.timeout == 120
    assert cfg.overpass_min_interval == 2.0
    assert cfg.overpass_max_retries == 5
    assert cfg.overpass_retry_statuses == (429, 502, 503, 504)
    assert cfg.overpass_backoff_base == 2
    assert cfg.enable_tqdm_bar is True
    assert cfg.verify_ssl is True
    assert cfg.proxies is None
    assert cfg.overpass_date is None
    # tag sets are frozenset and contain expected keys
    assert isinstance(cfg.drive_useful_edges_attr, frozenset)
    assert "highway" in cfg.drive_useful_edges_attr
    assert isinstance(cfg.walk_useful_edges_attr, frozenset)
    assert isinstance(cfg.transport_useful_edges_attr, frozenset)
    # logger object is available
    assert cfg.logger is logger
    # cache defaults
    assert cfg.overpass_cache_dir == ".iduedu_cache"
    assert cfg.overpass_cache_enabled is True


def test_from_env_reads_values(monkeypatch):
    monkeypatch.setenv("OVERPASS_URL", "https://example.org/interpreter")
    monkeypatch.setenv("OVERPASS_TIMEOUT", "42")
    monkeypatch.setenv("OVERPASS_MIN_INTERVAL", "2.5")
    monkeypatch.setenv("OVERPASS_MAX_RETRIES", "7")
    monkeypatch.setenv("OVERPASS_BACKOFF_BASE", "0.25")
    monkeypatch.setenv("ENABLE_TQDM", "0")
    monkeypatch.setenv("OVERPASS_USER_AGENT", "test-agent/1.0")
    monkeypatch.setenv("OVERPASS_DATE", "2020-01-01")
    monkeypatch.setenv("OVERPASS_CACHE_DIR", "/tmp/overpass_cache")
    monkeypatch.setenv("OVERPASS_CACHE_ENABLED", "0")

    cfg = Config()  # __init__ reads env
    assert cfg.overpass_url == "https://example.org/interpreter"
    assert cfg.timeout == 42
    assert cfg.overpass_min_interval == 2.5
    assert cfg.overpass_max_retries == 7
    assert cfg.overpass_backoff_base == 0.25
    assert cfg.enable_tqdm_bar is False
    assert cfg.user_agent == "test-agent/1.0"
    assert cfg.overpass_date == "2020-01-01T00:00:00Z"
    assert cfg.overpass_cache_dir == "/tmp/overpass_cache"
    assert cfg.overpass_cache_enabled is False


def test_from_env_invalid_overpass_date(monkeypatch, caplog):
    monkeypatch.setenv("OVERPASS_DATE", "not-a-date")

    cfg = Config()
    # invalid env date should not crash config and should leave overpass_date as None
    assert cfg.overpass_date is None


def test_set_overpass_url_valid_and_invalid():
    cfg = Config()
    cfg.set_overpass_url("http://mirror.example/api/interpreter")
    assert cfg.overpass_url == "http://mirror.example/api/interpreter"
    with pytest.raises(ValueError):
        cfg.set_overpass_url("not-a-url")


def test_set_timeout_valid_and_invalid():
    cfg = Config()
    cfg.set_timeout(30)
    assert cfg.timeout == 30
    with pytest.raises(ValueError):
        cfg.set_timeout(0)
    with pytest.raises(ValueError):
        cfg.set_timeout(-5)


def test_set_enable_tqdm():
    cfg = Config()
    cfg.set_enable_tqdm(False)
    assert cfg.enable_tqdm_bar is False
    cfg.set_enable_tqdm(True)
    assert cfg.enable_tqdm_bar is True


def test_set_tag_sets_are_frozensets():
    cfg = Config()
    cfg.set_drive_useful_edges_attr(["a", "b"])
    cfg.set_walk_useful_edges_attr(("x", "y"))
    cfg.set_transport_useful_edges_attr({"m", "n"})
    assert isinstance(cfg.drive_useful_edges_attr, frozenset)
    assert cfg.drive_useful_edges_attr == frozenset({"a", "b"})
    assert isinstance(cfg.walk_useful_edges_attr, frozenset)
    assert cfg.walk_useful_edges_attr == frozenset({"x", "y"})
    assert isinstance(cfg.transport_useful_edges_attr, frozenset)
    assert cfg.transport_useful_edges_attr == frozenset({"m", "n"})


def test_set_rate_limit_updates_and_validation():
    cfg = Config()
    cfg.set_rate_limit(min_interval=2.0, max_retries=10, backoff_base=0.75)
    assert cfg.overpass_min_interval == 2.0
    assert cfg.overpass_max_retries == 10
    assert cfg.overpass_backoff_base == 0.75

    with pytest.raises(ValueError):
        cfg.set_rate_limit(min_interval=-1)
    with pytest.raises(ValueError):
        cfg.set_rate_limit(max_retries=-2)
    with pytest.raises(ValueError):
        cfg.set_rate_limit(backoff_base=0)


# ---------- overpass date / header ----------


def test_overpass_date_default_and_header_without_date():
    cfg = Config()
    assert cfg.overpass_date is None

    header = cfg.build_overpass_header()
    assert header.startswith("[out:json][timeout:120]")
    assert "[date:" not in header


def test_set_overpass_date_string_short_and_full():
    cfg = Config()

    # YYYY-MM-DD → YYYY-MM-DDT00:00:00Z
    cfg.set_overpass_date(date="2020-01-01")
    assert cfg.overpass_date == "2020-01-01T00:00:00Z"
    header = cfg.build_overpass_header(timeout=10)
    assert '[date:"2020-01-01T00:00:00Z"]' in header

    # full datetime with Z stays as normalized ISO
    cfg.set_overpass_date(date="2020-01-01T12:34:56Z")
    assert cfg.overpass_date == "2020-01-01T12:34:56Z"
    header = cfg.build_overpass_header(timeout=5)
    assert '[date:"2020-01-01T12:34:56Z"]' in header


def test_set_overpass_date_components_year_month_day():
    cfg = Config()

    cfg.set_overpass_date(year=2020)
    assert cfg.overpass_date == "2020-01-01T00:00:00Z"

    cfg.set_overpass_date(year=2020, month=5)
    assert cfg.overpass_date == "2020-05-01T00:00:00Z"

    cfg.set_overpass_date(year=2020, month=5, day=10)
    assert cfg.overpass_date == "2020-05-10T00:00:00Z"


def test_set_overpass_date_reset():
    cfg = Config()
    cfg.set_overpass_date(year=2020)
    assert cfg.overpass_date is not None

    cfg.set_overpass_date()  # reset
    assert cfg.overpass_date is None

    cfg.set_overpass_date(None)  # explicit reset
    assert cfg.overpass_date is None


def test_set_overpass_date_invalid_combinations_and_values():
    cfg = Config()

    # both date and components
    with pytest.raises(ValueError):
        cfg.set_overpass_date(date="2020-01-01", year=2020)

    # invalid short date
    with pytest.raises(ValueError):
        cfg.set_overpass_date(date="2020-13-01")

    # invalid full datetime
    with pytest.raises(ValueError):
        cfg.set_overpass_date(date="2020-01-01T99:00:00Z")

    # missing year when using components
    with pytest.raises(ValueError):
        cfg.set_overpass_date(month=1, day=1)

    # invalid calendar date
    with pytest.raises(ValueError):
        cfg.set_overpass_date(year=2020, month=2, day=30)


def test_build_overpass_header_custom_timeout_and_date_override():
    cfg = Config()
    cfg.set_timeout(60)
    cfg.set_overpass_date(date="2020-01-01")

    header = cfg.overpass_header
    assert header == '[out:json][timeout:60][date:"2020-01-01T00:00:00Z"];'

    header2 = cfg.build_overpass_header(timeout=10, date="2021-02-03T00:00:00Z")
    assert header2 == '[out:json][timeout:10][date:"2021-02-03T00:00:00Z"];'


def test_set_overpass_cache_mutator():
    cfg = Config()

    assert cfg.overpass_cache_dir == ".iduedu_cache"
    assert cfg.overpass_cache_enabled is True

    cfg.set_overpass_cache(enabled=False)
    assert cfg.overpass_cache_enabled is False

    assert cfg.overpass_cache_dir == ".iduedu_cache"

    cfg.set_overpass_cache(cache_dir="/var/cache/osm")
    assert cfg.overpass_cache_dir == "/var/cache/osm"

    cfg.set_overpass_cache(enabled=True)
    assert cfg.overpass_cache_enabled is True


def test_to_dict_roundtrip_structure():
    cfg = Config()
    cfg.set_overpass_url("https://mirror.test/api/interpreter")
    cfg.set_timeout(15)
    cfg.set_enable_tqdm(False)
    cfg.set_drive_useful_edges_attr(["hwy"])
    cfg.set_overpass_date(date="2020-01-01")
    cfg.set_overpass_cache(enabled=False, cache_dir="/tmp/overpass_cache")

    d = cfg.to_dict()
    # keys present
    for key in (
        "overpass_url",
        "timeout",
        "overpass_date",
        "enable_tqdm_bar",
        "drive_useful_edges_attr",
        "walk_useful_edges_attr",
        "transport_useful_edges_attr",
        "overpass_min_interval",
        "overpass_max_retries",
        "overpass_retry_statuses",
        "overpass_backoff_base",
        "user_agent",
        "proxies",
        "verify_ssl",
        "overpass_cache_dir",
        "overpass_cache_enabled",
    ):
        assert key in d
    # values are as expected
    assert d["overpass_url"] == "https://mirror.test/api/interpreter"
    assert d["timeout"] == 15
    assert d["enable_tqdm_bar"] is False
    assert d["drive_useful_edges_attr"] == ["hwy"]
    assert isinstance(d["walk_useful_edges_attr"], list)
    assert isinstance(d["transport_useful_edges_attr"], list)
    assert isinstance(d["overpass_retry_statuses"], tuple)
    assert d["overpass_date"] == "2020-01-01T00:00:00Z"
    assert d["overpass_cache_dir"] == "/tmp/overpass_cache"
    assert d["overpass_cache_enabled"] is False
