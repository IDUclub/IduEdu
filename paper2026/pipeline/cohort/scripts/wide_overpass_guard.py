"""A guard around every Overpass request a named run makes, shared by all processes.

The library already retries, but its pauses top out at a minute and each process
keeps its own rate limiter. A fresh snapshot is hundreds of heavy queries against
a public server, and a few stages (places, resolve, osm, routestops) all talk to
it. So for a named run (``PAPER_RUN`` set) this module wraps the library's lowest
HTTP call, ``iduedu.overpass.downloaders._overpass_http``, without editing the
library:

* **One request at a time across processes.** A lock file in the run's Overpass
  cache directory serialises every interpreter call, so two stages started by
  accident cannot hit the server in parallel.
* **A slower rate.** At least ``PAPER_OVERPASS_MIN_INTERVAL`` seconds (default 8)
  between the end of one interpreter request and the start of the next.
* **Long cool-downs.** Every ``FAIL_STREAK`` consecutive refusals (429, 5xx,
  connection errors) the guard stops all traffic for 10 minutes, then 20, 40,
  capped at 60, and resets after the first success.
* **A global stop.** When the cool-down accumulated *without a single successful
  answer in between* passes ``PAPER_OVERPASS_COOLDOWN_BUDGET_H`` hours (default 3)
  -- the server has stopped answering, not merely become noisy -- or the wall clock passes
  ``PAPER_OVERPASS_DEADLINE`` (ISO local time, optional), the next request raises
  :class:`OverpassBudgetExhausted`. It derives from ``SystemExit`` on purpose: the
  stages catch ``Exception`` per city, and a server that is down should stop the
  sweep, not be recorded as a hundred failed cities.

State (last request, failure streak, cool-down) lives in a JSON file next to the
lock, so it survives a restart and is shared by whatever stage runs next.
"""

from __future__ import annotations

import calendar
import json
import os
import time
from datetime import datetime
from pathlib import Path

import requests

from iduedu import config

#: Refusals in a row before traffic stops. During the run about half of all
#: requests to overpass-api.de came back 504 within seconds, at random; three in a
#: row then happens every eight requests and would spend the budget on noise.
#: Five in a row by chance at that rate is ~3%.
FAIL_STREAK = int(os.environ.get("PAPER_OVERPASS_FAIL_STREAK", "5"))
#: First pause; doubles on each further streak up to COOLDOWN_MAX_S. Shortened to 150 s for
#: the places stage on 12.10 (author): a 504 here is the dispatcher refusing in seconds, before any
#: work is done, so a short first pause costs the server almost nothing.
COOLDOWN_FIRST_S = float(os.environ.get("PAPER_OVERPASS_FIRST_PAUSE_S", "600"))
COOLDOWN_MAX_S = 3600.0
RETRY_STATUSES = {429, 500, 502, 503, 504}

logger = config.logger
_installed = False
#: ``osm3s.timestamp_osm_base`` of the last successful interpreter response: the
#: moment of the OSM database the answer reflects. Read by ``wide_osm`` per city.
LAST_OSM_BASE: str | None = None


class OverpassBudgetExhausted(SystemExit):
    """Raised instead of sending a request once the run's time budget is spent."""


def _min_interval() -> float:
    return float(os.environ.get("PAPER_OVERPASS_MIN_INTERVAL", "8"))


def _cooldown_budget_s() -> float:
    return float(os.environ.get("PAPER_OVERPASS_COOLDOWN_BUDGET_H", "3")) * 3600.0


def _deadline() -> float | None:
    raw = os.environ.get("PAPER_OVERPASS_DEADLINE", "").strip()
    return datetime.fromisoformat(raw).timestamp() if raw else None


class _FileLock:
    """An exclusive lock on one byte of a file, blocking, Windows and POSIX."""

    def __init__(self, path: Path):
        self.path = path
        self._fd: int | None = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT)
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    os.lseek(self._fd, 0, os.SEEK_SET)
                    msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.5)
        else:
            import fcntl

            fcntl.flock(self._fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        if os.name == "nt":
            import msvcrt

            os.lseek(self._fd, 0, os.SEEK_SET)
            msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._fd, fcntl.LOCK_UN)
        os.close(self._fd)
        self._fd = None


def _read_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _mirrors() -> list[str]:
    """Other Overpass interpreters to rotate to after a refusal (PAPER_OVERPASS_MIRRORS).

    Author, 15.10.2026: overpass-api.de had been at capacity for hours. The request is
    sent elsewhere, but the cache keeps the main server's key, so nothing cached is
    asked again and the run reads the same files whichever server answered; the host
    of every answer is in the request log.
    """
    return [m.strip() for m in os.environ.get("PAPER_OVERPASS_MIRRORS", "").split(",") if m.strip()]


def _routes() -> list[str]:
    """Network routes to rotate through after a refusal (PAPER_OVERPASS_ROUTES).

    15--16.10.2026 the proxy route and the direct one took turns failing for hours
    (TLS dropped through the proxy while the direct route answered, then the other
    way round). "direct" bypasses the proxy for the Overpass host only, for that one
    request; every other connection of the process keeps the proxy.
    """
    return [r.strip() for r in os.environ.get("PAPER_OVERPASS_ROUTES", "proxy").split(",") if r.strip()]


class _Route:
    """Bypass the proxy for one host during one request, if the route is direct."""

    def __init__(self, route: str, url: str):
        self.direct = route == "direct"
        self.host = str(url).split("/")[2]
        self.saved: dict[str, str | None] = {}

    def __enter__(self):
        if self.direct:
            # on Windows the environment is case-insensitive: one variable, not two
            for name in ("NO_PROXY",) if os.name == "nt" else ("NO_PROXY", "no_proxy"):
                self.saved[name] = os.environ.get(name)
                os.environ[name] = ",".join(filter(None, [self.saved[name], self.host]))
        return self

    def __exit__(self, *exc):
        for name, value in reversed(list(self.saved.items())):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


#: An answer whose data are older than this is refused, whichever server sent it.
def _max_lag_s() -> float:
    return float(os.environ.get("PAPER_OVERPASS_MAX_LAG_H", "6")) * 3600


class StaleAnswer(requests.ConnectionError):
    """A mirror answered from data older than PAPER_OVERPASS_MAX_LAG_H."""


def _osm_base_age_s(response) -> float | None:
    head = response.content[:600].decode("utf-8", errors="ignore")
    marker = '"timestamp_osm_base": "'
    start = head.find(marker)
    if start < 0:
        return None
    stamp = head[start + len(marker) : head.find('"', start + len(marker))]
    try:
        return time.time() - calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ"))
    except ValueError:
        return None


def _note_osm_base(response) -> None:
    global LAST_OSM_BASE
    head = response.content[:600].decode("utf-8", errors="ignore")
    marker = '"timestamp_osm_base": "'
    start = head.find(marker)
    if start >= 0:
        start += len(marker)
        LAST_OSM_BASE = head[start : head.find('"', start)]


def _query_kind(params, data) -> str:
    """A short label for what was asked: the statement after the header."""
    query = str((data or {}).get("data") or (params or {}).get("data") or "")
    body = query.split(";", 1)[1] if ";" in query else query
    body = " ".join(body.split())
    for marker, label in (
        ("is_in(", "is_in"),
        ('["place"', "settlements"),
        ("out count", "count"),
        ("relation(", "relation"),
        ("rel(", "relation"),
        ("route", "routes"),
        ("highway", "walk"),
    ):
        if marker in body:
            return label
    return body[:40]


def _server_error(response) -> str:
    text = response.text[:3000] if response.text else ""
    start = text.find("Error</strong>:")
    if start >= 0:
        return " ".join(text[start + 15 : start + 200].split()).split("</p>")[0][:160]
    return (response.reason or "")[:80]


def _log_request(path: Path, started: float, method, url, params, data, outcome) -> None:
    line = ",".join(
        [
            time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
            f"{time.time() - started:.1f}",
            _query_kind(params, data),
            str(outcome["status"]),
            '"' + str(outcome["error"]).replace('"', "'") + '"',
            str(url).split("/")[2],
        ]
    )
    new = not path.exists()
    with path.open("a", encoding="utf-8") as handle:
        if new:
            handle.write("started,elapsed_s,kind,status,error,host\n")
        handle.write(line + "\n")


def install(cache_dir: Path) -> None:
    """Wrap the library's HTTP call. Idempotent."""
    global _installed
    if _installed:
        return
    from iduedu.overpass import downloaders

    original = downloaders._overpass_http
    lock = _FileLock(Path(cache_dir) / "_overpass_guard.lock")
    state_path = Path(cache_dir) / "_overpass_guard_state.json"
    request_log = Path(cache_dir) / "_overpass_requests.csv"

    def guarded(method, url, *, params=None, data=None, timeout=None):
        # The /status probe the library sends before each request is cheap and
        # tells it how long to wait; it is serialised but not rate-limited or counted.
        is_probe = str(url).rstrip("/").endswith("/status")
        with lock:
            state = _read_state(state_path)
            deadline = _deadline()
            if deadline is not None and time.time() > deadline:
                raise OverpassBudgetExhausted(f"Overpass deadline {os.environ['PAPER_OVERPASS_DEADLINE']} passed")
            # A summed budget was tried first and was wrong: at one point about 57% of
            # answers were 504s at random, a 10-minute pause came every quarter hour,
            # and six hours of pauses would have stopped a healthy sweep half-way.
            if state.get("dry_cooldown_s", 0.0) > _cooldown_budget_s():
                raise OverpassBudgetExhausted(
                    f"Overpass cool-down budget spent ({state['dry_cooldown_s'] / 3600:.1f} h without a success); "
                    f"the server keeps refusing. Reset {state_path.name} to continue."
                )
            wait = state.get("cooldown_until", 0.0) - time.time()
            if wait > 0:
                logger.warning(f"Overpass guard: cooling down, {wait / 60:.1f} min left")
                time.sleep(wait)
            if not is_probe:
                gap = state.get("last_request_end", 0.0) + _min_interval() - time.time()
                if gap > 0:
                    time.sleep(gap)

            main = str(url).rsplit("/", 1)[0] + "/interpreter" if is_probe else str(url)
            servers = [(main, route) for route in _routes()] + [(mirror, "proxy") for mirror in _mirrors()]
            server, route = servers[state.get("server_index", 0) % len(servers)]
            target = server.rsplit("/", 1)[0] + "/status" if is_probe else server

            failed = False
            started = time.time()
            outcome = {"status": "", "error": ""}
            try:
                with _Route(route, target):
                    response = original(method, target, params=params, data=data, timeout=timeout)
                failed = response.status_code in RETRY_STATUSES
                outcome["status"] = response.status_code
                if response.status_code != 200:
                    outcome["error"] = _server_error(response)
                if response.status_code == 200 and not is_probe:
                    age = _osm_base_age_s(response)
                    if age is not None and age > _max_lag_s():
                        raise StaleAnswer(f"{target}: data {age / 3600:.1f} h old")
                    _note_osm_base(response)
                return response
            except requests.RequestException as exc:
                failed = True
                outcome["status"], outcome["error"] = "exception", type(exc).__name__
                raise
            finally:
                if not is_probe:
                    outcome["error"] = f"{outcome['error']} [{route}]".strip() if route != "proxy" else outcome["error"]
                    _log_request(request_log, started, method, target, params, data, outcome)
                    if failed and len(servers) > 1:
                        state["server_index"] = state.get("server_index", 0) + 1
                    state["last_request_end"] = time.time()
                    state["requests"] = state.get("requests", 0) + 1
                    if failed:
                        state["streak"] = state.get("streak", 0) + 1
                        state["failures"] = state.get("failures", 0) + 1
                        if state["streak"] % FAIL_STREAK == 0:
                            level = state.get("level", 0)
                            pause = min(COOLDOWN_MAX_S, COOLDOWN_FIRST_S * 2**level)
                            state["level"] = level + 1
                            state["cooldown_until"] = time.time() + pause
                            state["total_cooldown_s"] = state.get("total_cooldown_s", 0.0) + pause
                            state["dry_cooldown_s"] = state.get("dry_cooldown_s", 0.0) + pause
                            logger.warning(
                                f"Overpass guard: {state['streak']} refusals in a row, pausing all traffic "
                                f"for {pause / 60:.0f} min (cool-down so far {state['total_cooldown_s'] / 3600:.2f} h)"
                            )
                    else:
                        state["streak"] = 0
                        state["level"] = 0
                        state["dry_cooldown_s"] = 0.0
                    _write_state(state_path, state)

    downloaders._overpass_http = guarded
    _installed = True
    logger.warning(
        f"Overpass guard on: min interval {_min_interval():.0f} s, cool-down budget "
        f"{_cooldown_budget_s() / 3600:.1f} h, deadline {os.environ.get('PAPER_OVERPASS_DEADLINE') or 'none'}"
    )
