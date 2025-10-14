import os
import sys
import threading
from typing import Iterable, Literal, Optional

from loguru import logger

LogLevel = Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Config:
    """
    Global settings: Overpass endpoint(s), timeouts, logging, tag sets, and rate limiting.
    All setters validate inputs; logging can be configured to stderr and/or file.
    """

    def __init__(self):
        # --- Overpass endpoints ---
        self.overpass_url: str = os.getenv("OVERPASS_URL", "https://overpass-api.de/api/interpreter")
        self.user_agent: str = os.getenv("OVERPASS_USER_AGENT", "iduedu/0.1 (+https://example.org)")
        self.proxies: Optional[dict[str, str]] = None
        self.verify_ssl: bool = True

        # --- Networking / retries ---
        self.timeout: int = int(os.getenv("OVERPASS_TIMEOUT", "120"))
        self.overpass_min_interval: float = float(os.getenv("OVERPASS_MIN_INTERVAL", "2"))
        self.overpass_max_retries: int = int(os.getenv("OVERPASS_MAX_RETRIES", "5"))
        self.overpass_retry_statuses: tuple[int, ...] = (429, 502, 503, 504)
        self.overpass_backoff_base: float = float(os.getenv("OVERPASS_BACKOFF_BASE", "0.5"))

        # --- UX ---
        self.enable_tqdm_bar: bool = os.getenv("ENABLE_TQDM", "1") not in {"0", "false", "False"}

        # --- Tags to keep on edges ---
        self.drive_useful_edges_attr = frozenset({"highway", "name", "lanes"})
        self.walk_useful_edges_attr = frozenset({"highway", "name"})
        self.transport_useful_edges_attr = frozenset({"name"})

        # --- Logging ---
        self.logger = logger
        self.configure_logging(level=os.getenv("LOG_LEVEL", "INFO").upper())

        # --- Internals ---
        self._lock = threading.RLock()

    # ------------- Logging -------------
    def configure_logging(self, level: LogLevel = "INFO", *, to_file: str | None = None, json: bool = False):
        """
        Configure Loguru: stderr sink (+optional file sink), level, formatting.
        """
        self.logger.remove()
        fmt = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}" if not json else "{message}"
        self.logger.add(sys.stderr, level=level, backtrace=True, diagnose=False, format=fmt)
        if to_file:
            self.logger.add(
                to_file,
                level=level,
                rotation="20 MB",
                retention="14 days",
                compression="zip",
                enqueue=True,
                backtrace=True,
                diagnose=False,
                format=fmt,
            )

    # ------------- Overpass control -------------
    def set_overpass_url(self, url: str):
        """
        Set active Overpass URL (validated), keeps the pool intact.
        """
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ValueError(f"Invalid Overpass URL: {url!r}")
        with self._lock:
            self.overpass_url = url

    def set_timeout(self, timeout: int):
        if timeout <= 0:
            raise ValueError("timeout must be > 0")
        self.timeout = int(timeout)

    def set_enable_tqdm(self, enable: bool):
        self.enable_tqdm_bar = bool(enable)

    def set_drive_useful_edges_attr(self, attr: Iterable[str]):
        self.drive_useful_edges_attr = frozenset(attr)

    def set_walk_useful_edges_attr(self, attr: Iterable[str]):
        self.walk_useful_edges_attr = frozenset(attr)

    def set_transport_useful_edges_attr(self, attr: Iterable[str]):
        self.transport_useful_edges_attr = frozenset(attr)

    def set_rate_limit(
        self, *, min_interval: float | None = None, max_retries: int | None = None, backoff_base: float | None = None
    ):
        if min_interval is not None:
            if min_interval < 0:
                raise ValueError("min_interval must be >= 0")
            self.overpass_min_interval = float(min_interval)
        if max_retries is not None:
            if max_retries < 0:
                raise ValueError("max_retries must be >= 0")
            self.overpass_max_retries = int(max_retries)
        if backoff_base is not None:
            if backoff_base <= 0:
                raise ValueError("backoff_base must be > 0")
            self.overpass_backoff_base = float(backoff_base)

    def to_dict(self) -> dict:
        return {
            "overpass_url": self.overpass_url,
            "timeout": self.timeout,
            "enable_tqdm_bar": self.enable_tqdm_bar,
            "drive_useful_edges_attr": sorted(self.drive_useful_edges_attr),
            "walk_useful_edges_attr": sorted(self.walk_useful_edges_attr),
            "transport_useful_edges_attr": sorted(self.transport_useful_edges_attr),
            "overpass_min_interval": self.overpass_min_interval,
            "overpass_max_retries": self.overpass_max_retries,
            "overpass_retry_statuses": self.overpass_retry_statuses,
            "overpass_backoff_base": self.overpass_backoff_base,
            "user_agent": self.user_agent,
            "proxies": self.proxies,
            "verify_ssl": self.verify_ssl,
        }

    @classmethod
    def from_env(cls) -> "Config":
        """
        Construct Config using environment variables where available.
        """
        cfg = cls()
        return cfg


config = Config()
