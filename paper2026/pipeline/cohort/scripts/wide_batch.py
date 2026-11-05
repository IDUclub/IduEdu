"""Many independent Overpass queries sent as one request, answers split back apart.

The pipeline asks Overpass the same small question about many objects -- which
areas contain this point, which settlements are near it, how many route
relations lie in this boundary. One request per object meant thousands of
requests against a server that refuses about half of them at random, and the
refusals, not the data, set the pace. Overpass runs any number of statements in
one query, so the statements are sent together, each preceded by a marker element
(``make marker idx="<i>"; out;``) that tells the answers apart.

``cached_batch`` additionally writes every answer into the IduEdu cache under the
key the single query would have had, so the code that asks one question at a time
finds the answer there and runs unchanged.
"""

import logging

from iduedu import config
from iduedu.overpass.cache import cache_load, cache_save
from iduedu.overpass.downloaders import _overpass_request

logger = logging.getLogger(__name__)

#: Statements per request. Small answers (tags, ids, a few nodes) at this size stay
#: far below the server's limits; it is the number of requests that costs.
BATCH = 25


def header() -> str:
    """The query header every single query of the pipeline uses."""
    return f"[out:json][timeout:{config.timeout}];"


def single_key(statement: str) -> str:
    """The cache key of ``statement`` sent alone with ``header()`` by POST."""
    return f"{config.overpass_url}\nPOST\n{header()}{statement}"


def split(payload: dict, n: int) -> list[list[dict]]:
    """Elements of a marked answer, grouped by the statement that produced them."""
    groups: list[list[dict]] = [[] for _ in range(n)]
    current = None
    for element in payload.get("elements", []):
        if element.get("type") == "marker":
            current = int(element.get("tags", {}).get("idx", -1))
            continue
        if current is not None and 0 <= current < n:
            groups[current].append(element)
    return groups


def run_batched(statements: list[str]) -> tuple[dict, list[list[dict]]]:
    """Send ``statements`` in one request; return the answer's metadata and the groups."""
    body = "".join(f'make marker idx="{i}";out;{statement}' for i, statement in enumerate(statements))
    payload = _overpass_request("POST", config.overpass_url, data={"data": header() + body}).json()
    meta = {key: value for key, value in payload.items() if key != "elements"}
    return meta, split(payload, len(statements))


def cached_batch(prefix: str, statements: list[str], batch: int = BATCH) -> dict[str, int]:
    """Answer every uncached statement in batches and cache each answer singly.

    A batch that fails is left alone: the stage asking one question at a time
    will fetch those answers itself, as it always did.
    """
    statements = list(dict.fromkeys(statements))
    missing = [s for s in statements if cache_load(prefix, single_key(s)) is None]
    done = failed = 0
    for start in range(0, len(missing), batch):
        chunk = missing[start : start + batch]
        try:
            meta, groups = run_batched(chunk)
        except Exception as exc:  # noqa: BLE001
            failed += len(chunk)
            logger.warning("%s: batch of %d failed (%s)", prefix, len(chunk), exc)
            continue
        for statement, elements in zip(chunk, groups):
            cache_save(prefix, single_key(statement), {**meta, "elements": elements})
            done += 1
        logger.info("%s: [%d/%d] cached", prefix, done, len(missing))
    return {"statements": len(statements), "missing": len(missing), "cached_now": done, "left": failed}


def cached_single(prefix: str, statement: str) -> list[dict]:
    """One statement's elements: from the cache if a batch (or an earlier run) put
    them there, else from its own request, which is then cached too."""
    key = single_key(statement)
    cached = cache_load(prefix, key)
    if cached is not None:
        return cached.get("elements", [])
    payload = _overpass_request("POST", config.overpass_url, data={"data": header() + statement}).json()
    cache_save(prefix, key, payload)
    return payload.get("elements", [])
