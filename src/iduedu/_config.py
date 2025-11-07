import os
import sys
import threading
from datetime import date as _date
from datetime import datetime as _datetime
from typing import Iterable, Literal, Optional

from loguru import logger

LogLevel = Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Config:
    """
    Global settings: Overpass endpoint(s), timeouts, logging, tag sets, and rate limiting.
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
        self.overpass_backoff_base: float = float(os.getenv("OVERPASS_BACKOFF_BASE", "2"))

        # --- Overpass date / header control ---
        self.overpass_date: str | None = None
        env_date = os.getenv("OVERPASS_DATE")
        if env_date:
            try:
                self.set_overpass_date(date=env_date)
            except ValueError as exc:
                logger.warning(f"Invalid OVERPASS_DATE env value {env_date!r}: {exc}")

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

    def set_overpass_date(
        self,
        date: str | None = None,
        *,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
    ):
        """Set the snapshot date for Overpass queries.

        The date is stored in ISO 8601 UTC format: ``YYYY-MM-DDTHH:MM:SSZ`` and
        used in the Overpass header as ``[date:"..."]``. Calling this method
        allows you to fix the OSM data to a specific historical snapshot or
        reset back to the "current" data.

        Usage examples:
            * ``set_overpass_date()`` or ``set_overpass_date(None)`` —
              reset the date (no ``[date:...]`` in queries).
            * ``set_overpass_date(date="2020-01-01")`` →
              ``2020-01-01T00:00:00Z``.
            * ``set_overpass_date(date="2020-01-01T12:34:56Z")`` →
              stores the normalized value of this timestamp.

            * ``set_overpass_date(year=2020)`` →
              ``2020-01-01T00:00:00Z``.
            * ``set_overpass_date(year=2020, month=5)`` →
              ``2020-05-01T00:00:00Z``.
            * ``set_overpass_date(year=2020, month=5, day=10)`` →
              ``2020-05-10T00:00:00Z``.

        Args:
            date: A date string in ``YYYY-MM-DD`` or ``YYYY-MM-DDTHH:MM:SSZ``
                format. If provided, ``year``, ``month`` and ``day`` must all
                be ``None``.
            year: Year component of the date. If provided without ``date``,
                at least the year must be set; ``month`` and ``day`` are
                optional and default to January 1 if omitted.
            month: Month component of the date (1–12). Optional when used
                with ``year``. Defaults to 1 (January) if not provided.
            day: Day component of the date (1–31). Optional when used with
                ``year`` and ``month``. Defaults to 1 if not provided.

        Raises:
            ValueError: If both ``date`` and any of ``year``, ``month``,
                ``day`` are provided, if the date string is not in a supported
                format, or if the constructed date is invalid.
        """

        if date is None and year is None and month is None and day is None:
            self.overpass_date = None
            return

        if date is not None and any(v is not None for v in (year, month, day)):
            raise ValueError("Use either `date` or (`year`, `month`, `day`), not both")

        if date is not None:
            if len(date) == 10:
                try:
                    _date.fromisoformat(date)
                except ValueError as exc:
                    raise ValueError("overpass date must be in YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ format") from exc
                self.overpass_date = f"{date}T00:00:00Z"
                return
            else:
                try:
                    dt = _datetime.fromisoformat(date.replace("Z", "+00:00"))
                except ValueError as exc:
                    raise ValueError("overpass date must be in YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ format") from exc
                self.overpass_date = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                return

        if year is None:
            raise ValueError("year must be provided if `date` is not used")

        if month is None:
            month = 1
        if day is None:
            day = 1

        try:
            d = _date(year, month, day)
        except ValueError as exc:
            raise ValueError(f"Invalid date: {exc}") from exc

        self.overpass_date = f"{d.isoformat()}T00:00:00Z"

    def build_overpass_header(self, *, timeout: int | None = None, date: str | None = None) -> str:
        """
        Собрать заголовок Overpass, например:
            [out:json][timeout:120];
            [out:json][timeout:120][date:"2020-01-01T00:00:00Z"];
        """
        t = int(timeout) if timeout is not None else int(self.timeout)
        if t <= 0:
            raise ValueError("timeout must be > 0")

        effective_date = date if date is not None else self.overpass_date

        parts = [f"[out:json][timeout:{t}]"]
        if effective_date:
            parts.append(f'[date:"{effective_date}"]')
        return "".join(parts) + ";"

    @property
    def overpass_header(self) -> str:
        return self.build_overpass_header()

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
            "overpass_date": self.overpass_date,
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
