"""Download the selected GTFS archives, resume-safe, with a log of every failure.

Every attempt is recorded in ``results/wide_tier/downloads.csv``, successful or
not, because the failures are a result in their own right: a catalog entry that
returns a landing page instead of an archive, or a host that has stopped
answering, is a measurement of how available open transit data actually is.

Three lessons from the previous run are built in:

* ``requests`` timeouts bound a single read, not the whole transfer, so a slow
  host can hang for hours. Each archive gets a hard wall-clock budget.
* Catalogs rot. A 200 response is not enough -- the payload is accepted only if
  it is a zip that contains ``stops.txt``, which is what separates a real feed
  from an HTML "download our data" page.
* Some hosts fail under Python's TLS stack but work under curl, so curl is kept
  as a last resort rather than treating those feeds as dead.

Every stored archive gets a SHA-256 in the log. That is what makes the download
reproducible and any later substitution detectable, and it is required for the
feeds we are allowed to fetch only with certificate verification disabled.
"""

import argparse
import hashlib
import logging
import shutil
import subprocess
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

import pandas as pd
import requests
from bench_common import append_row
from wide_catalogs import USER_AGENT
from wide_paths import DOWNLOADS_CSV
from wide_paths import FEED_CACHE_DIR as FEED_CACHE
from wide_paths import SELECTION_CSV, WIDE_DIR

logger = logging.getLogger(__name__)

#: Wall-clock budget for one archive, including redirects and slow trickles.
ARCHIVE_TIMEOUT_S = 300
#: Feeds larger than this are national dumps rather than city networks.
MAX_ARCHIVE_BYTES = 600 * 1024 * 1024
CHUNK = 1 << 16

#: Hosts whose certificate is known to be invalid and that we were explicitly
#: allowed to fetch anyway. Verification is disabled per host, never globally,
#: and the SHA-256 of whatever arrives is recorded.
INSECURE_HOSTS: set[str] = {"git.digitaltransport4africa.org"}


def _encode_url(url: str) -> str:
    """Percent-encode a URL path that may contain non-ASCII characters."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, quote(parts.path, safe="/%:@&=+$,~"), parts.query, parts.fragment))


def _looks_like_gtfs(path: Path) -> tuple[bool, str]:
    if not zipfile.is_zipfile(path):
        return False, "not a zip archive"
    try:
        with zipfile.ZipFile(path) as archive:
            names = {Path(name).name.lower() for name in archive.namelist()}
    except zipfile.BadZipFile:
        return False, "corrupt zip archive"
    if "stops.txt" not in names:
        return False, "zip without stops.txt"
    return True, ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stream_to(url: str, destination: Path, verify: bool, deadline: float) -> tuple[bool, str]:
    try:
        response = requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=60, stream=True, verify=verify, allow_redirects=True
        )
    except requests.RequestException as exc:
        return False, f"{type(exc).__name__}"
    with response:
        if response.status_code != 200:
            return False, f"http {response.status_code}"
        written = 0
        with destination.open("wb") as handle:
            for chunk in response.iter_content(CHUNK):
                if not chunk:
                    continue
                handle.write(chunk)
                written += len(chunk)
                if written > MAX_ARCHIVE_BYTES:
                    return False, "larger than the size limit"
                if time.monotonic() > deadline:
                    return False, "exceeded the time budget"
    return True, ""


def _curl_to(url: str, destination: Path, verify: bool, remaining: float) -> tuple[bool, str]:
    """Last resort: curl succeeds on some TLS stacks where Python does not."""
    executable = shutil.which("curl")
    if not executable:
        return False, "curl unavailable"
    command = [executable, "-sSL", "--max-time", str(int(max(remaining, 30))), "-A", USER_AGENT, "-o", str(destination)]
    if not verify:
        command.append("-k")
    command.append(url)
    try:
        finished = subprocess.run(command, capture_output=True, timeout=max(remaining, 30) + 30, check=False)
    except subprocess.TimeoutExpired:
        return False, "curl timed out"
    if finished.returncode != 0:
        return False, f"curl exit {finished.returncode}"
    return True, ""


def fetch(feed_id: str, url: str) -> dict:
    """Download one archive and return the row describing the attempt."""
    started = time.monotonic()
    deadline = started + ARCHIVE_TIMEOUT_S
    destination = FEED_CACHE / f"{feed_id}.zip"
    partial = destination.with_suffix(".zip.part")
    FEED_CACHE.mkdir(parents=True, exist_ok=True)

    encoded = _encode_url(url)
    host = urlsplit(encoded).netloc.lower().split(":")[0]
    verify = host not in INSECURE_HOSTS

    ok, reason = _stream_to(encoded, partial, verify, deadline)
    if not ok:
        fallback_ok, fallback_reason = _curl_to(encoded, partial, verify, deadline - time.monotonic())
        if fallback_ok:
            ok, reason = True, ""
        else:
            reason = f"{reason}; curl: {fallback_reason}"

    row = {
        "feed_id": feed_id,
        "url": url,
        "verified_tls": verify,
        "elapsed_s": round(time.monotonic() - started, 1),
        "bytes": 0,
        "sha256": "",
        "status": "failed",
        "reason": reason,
    }

    if not ok or not partial.exists():
        partial.unlink(missing_ok=True)
        return row

    valid, why = _looks_like_gtfs(partial)
    row["bytes"] = partial.stat().st_size
    if not valid:
        row["status"] = "invalid"
        row["reason"] = why
        partial.unlink(missing_ok=True)
        return row

    row["sha256"] = _sha256(partial)
    partial.replace(destination)
    row["status"] = "ok"
    row["reason"] = ""
    return row


def run(limit: int | None = None, workers: int = 4, retry_failed: bool = False) -> pd.DataFrame:
    selection = pd.read_csv(SELECTION_CSV)
    WIDE_DIR.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if DOWNLOADS_CSV.exists():
        log = pd.read_csv(DOWNLOADS_CSV).drop_duplicates(subset=["feed_id"], keep="last")
        for feed_id, status in log.set_index("feed_id")["status"].to_dict().items():
            if status == "ok" and (FEED_CACHE / f"{feed_id}.zip").exists():
                done.add(str(feed_id))
            elif not retry_failed:
                done.add(str(feed_id))

    pending = [row for _, row in selection.iterrows() if str(row["feed_id"]) not in done]
    if limit:
        pending = pending[:limit]
    logger.info("%d feeds to fetch (%d already recorded)", len(pending), len(done))

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, str(row["feed_id"]), str(row["url"])): row for row in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            append_row(DOWNLOADS_CSV, row)
            results.append(row)
            logger.info(
                "[%d/%d] %-10s %-28s %s",
                index,
                len(pending),
                row["status"],
                row["feed_id"][:28],
                row["reason"] or f"{row['bytes'] / 1e6:.1f} MB",
            )
    return pd.DataFrame(results)


def _report() -> None:
    if not DOWNLOADS_CSV.exists():
        print("nothing downloaded yet")
        return
    log = pd.read_csv(DOWNLOADS_CSV).drop_duplicates(subset=["feed_id"], keep="last")
    print(f"attempts: {len(log)}")
    print("status: " + str(log["status"].value_counts().to_dict()))
    ok = log[log["status"] == "ok"]
    if not ok.empty:
        print(f"stored: {len(ok)} archives, {ok['bytes'].sum() / 1e9:.2f} GB")
    failures = log[log["status"] != "ok"]
    if not failures.empty:
        print("\nreasons:")
        for reason, count in failures["reason"].value_counts().head(15).items():
            print(f"  {count:>3}  {str(reason)[:90]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="fetch at most this many feeds")
    parser.add_argument("--workers", type=int, default=4, help="parallel downloads")
    parser.add_argument("--retry-failed", action="store_true", help="attempt feeds that failed before")
    parser.add_argument("--report", action="store_true", help="only summarise what has been downloaded")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    if arguments.report:
        _report()
        return
    run(arguments.limit, arguments.workers, arguments.retry_failed)
    _report()


if __name__ == "__main__":
    main()
