"""DigitalTransport4Africa as a third catalog source.

Neither Mobility Database nor Transitland Atlas covers Tropical Africa in any
useful way -- between them they offer eight feeds for the whole region, which is
the part of the world this study exists to say something about. DT4A publishes
field-mapped feeds for about two dozen African cities plus a few Latin American
ones, and they are public on gitlab.com.

Two things make this source different from a catalog:

* **A repository is not a feed.** One city can hold several archives -- different
  operators (Douala has SYSTRA and WhereIsMyTransport), different generations
  (Addis Ababa has three from 2018), or a raw and a cleaned version. Each archive
  becomes its own row, which is what the feed-as-unit design is for, and they are
  merged back into one city later with the merge count recorded.
* **Not every zip is a feed.** The repositories also hold GIS exports and
  workshop material. Nothing is filtered on the name here: the downloader already
  accepts an archive only if it is a zip containing ``stops.txt``, so a wrong
  guess is recorded as a rejection rather than becoming a silent bad feed.

Geometry provenance matters for these feeds -- several are known to be traced
from OSM -- but that is measured after download by the provenance test, not
assumed here.
"""

import argparse
import logging
from urllib.parse import quote

import pandas as pd
import requests
from wide_catalogs import USER_AGENT, normalise_url
from wide_paths import SELECTION_CSV
from wide_selection import merge_into_selection

logger = logging.getLogger(__name__)

GITLAB_API = "https://gitlab.com/api/v4"
GROUPS = ("digitaltransport/data/africa", "digitaltransport/data/latin-america")

#: Repositories that sit in the data group but hold something other than transit
#: feeds. "pedestrians-first" is the ITDP walkability indicator project: 65 city
#: archives of analysis rasters, none of them GTFS. Letting the validity check
#: reject them one by one would bury the real rejections under a single project.
EXCLUDED_REPOS = {"pedestrians-first"}


def _get(path: str, **params):
    response = requests.get(f"{GITLAB_API}/{path}", headers={"User-Agent": USER_AGENT}, params=params, timeout=90)
    response.raise_for_status()
    return response.json()


def archives() -> pd.DataFrame:
    """Every zip in every DT4A data repository, as catalog rows."""
    records: list[dict] = []
    for group in GROUPS:
        info = _get(f"groups/{quote(group, safe='')}")
        projects = _get(f"groups/{info['id']}/projects", per_page=100, include_subgroups="true", archived="false")
        logger.info("%s: %d repositories", group, len(projects))
        for project in projects:
            if project["path"] in EXCLUDED_REPOS:
                logger.info("  %s: skipped, not a transit data repository", project["path"])
                continue
            branch = project.get("default_branch") or "master"
            try:
                tree = _get(f"projects/{project['id']}/repository/tree", recursive="true", per_page=100, ref=branch)
            except requests.HTTPError as exc:
                logger.warning("%s: cannot list files (%s)", project["path"], exc)
                continue
            zips = [item["path"] for item in tree if item["path"].lower().endswith(".zip")]
            if not zips:
                logger.info("  %s: no archive", project["path"])
            for path in zips:
                stem = path.rsplit("/", 1)[-1][:-4]
                records.append(
                    {
                        "catalog": "dt4a",
                        "feed_id": f"dt4a-{project['path']}-{stem}".replace(" ", "_")[:80],
                        "provider": project.get("name", ""),
                        "name": stem,
                        "country_code": "",
                        "country_source": "",
                        "subdivision": "",
                        "municipality": project.get("name", ""),
                        # The raw endpoint needs the path percent-encoded whole, slashes included.
                        "url": (
                            f"{GITLAB_API}/projects/{project['id']}/repository/files/"
                            f"{quote(path, safe='')}/raw?ref={branch}"
                        ),
                        "license": f"https://gitlab.com/{project['path_with_namespace']}",
                        "lat": None,
                        "lon": None,
                        "min_lat": None,
                        "max_lat": None,
                        "min_lon": None,
                        "max_lon": None,
                        "country_name": "",
                        "region": "",
                        "income_group": "",
                    }
                )
    frame = pd.DataFrame.from_records(records)
    frame["url_key"] = frame["url"].map(normalise_url)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="list what would be added, change nothing")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    frame = archives()
    print(f"\n{len(frame)} archives in {frame['provider'].nunique()} repositories")
    print(frame.groupby("provider").size().sort_values(ascending=False).to_string())
    if arguments.dry_run:
        print("\nDry run: selection.csv untouched.")
        return
    added = merge_into_selection(frame)
    print(f"\nadded {added} feeds; selection.csv now holds {len(pd.read_csv(SELECTION_CSV))}")


if __name__ == "__main__":
    main()
