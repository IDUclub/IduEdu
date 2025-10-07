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
        "ENABLE_TQDM",
        "LOG_LEVEL",
        "CI",
    ]:
        monkeypatch.delenv(k, raising=False)
    yield


def test_defaults_no_env_and_tty(monkeypatch):
    cfg = Config()

    assert cfg.overpass_url == "https://overpass-api.de/api/interpreter"
    assert cfg.timeout == 120
    assert cfg.overpass_min_interval == 2.0
    assert cfg.overpass_max_retries == 3
    assert cfg.overpass_retry_statuses == (429, 502, 503, 504)
    assert cfg.overpass_backoff_base == 0.5
    assert cfg.enable_tqdm_bar is True
    assert cfg.verify_ssl is True
    assert cfg.proxies is None
    # tag sets are frozenset and contain expected keys
    assert isinstance(cfg.drive_useful_edges_attr, frozenset)
    assert "highway" in cfg.drive_useful_edges_attr
    assert isinstance(cfg.walk_useful_edges_attr, frozenset)
    assert isinstance(cfg.transport_useful_edges_attr, frozenset)
    # logger object is available
    assert cfg.logger is logger


def test_from_env_reads_values(monkeypatch):
    monkeypatch.setenv("OVERPASS_URL", "https://example.org/interpreter")
    monkeypatch.setenv("OVERPASS_TIMEOUT", "42")
    monkeypatch.setenv("OVERPASS_MIN_INTERVAL", "2.5")
    monkeypatch.setenv("OVERPASS_MAX_RETRIES", "7")
    monkeypatch.setenv("OVERPASS_BACKOFF_BASE", "0.25")
    monkeypatch.setenv("ENABLE_TQDM", "0")
    monkeypatch.setenv("OVERPASS_USER_AGENT", "test-agent/1.0")

    cfg = Config()  # __init__ reads env
    assert cfg.overpass_url == "https://example.org/interpreter"
    assert cfg.timeout == 42
    assert cfg.overpass_min_interval == 2.5
    assert cfg.overpass_max_retries == 7
    assert cfg.overpass_backoff_base == 0.25
    assert cfg.enable_tqdm_bar is False
    assert cfg.user_agent == "test-agent/1.0"


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


def test_to_dict_roundtrip_structure():
    cfg = Config()
    cfg.set_overpass_url("https://mirror.test/api/interpreter")
    cfg.set_timeout(15)
    cfg.set_enable_tqdm(False)
    cfg.set_drive_useful_edges_attr(["hwy"])
    d = cfg.to_dict()
    # keys present
    for key in (
        "overpass_url",
        "timeout",
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
