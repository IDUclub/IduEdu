"""Assemble everything the archive needs, so depositing is an upload and not a decision.

What goes in and what stays out is settled here rather than at upload time.

**Derived tables go in.** Every number in the paper comes from
``results/wide_tier/*.csv``, and those are our own measurements of third-party
data, not the data itself.

**Published feeds stay out.** Some of the archives carry licences that forbid
redistribution -- Santiago and Saint Petersburg among them -- and auditing 169
licences one by one to republish the rest would put the legal burden on the
deposit for very little gain. Instead the deposit carries the identifier, the
catalogue and the source URL of every feed used, so the collection can be
rebuilt from its sources.

**Code goes in**, because a table nobody can regenerate is not a result.

The output is a directory ready to upload, with a manifest of checksums.

    python make_deposit.py --run
"""

import argparse
import hashlib
import re
import shutil
from datetime import date
from pathlib import Path

import pandas as pd
from wide_cohort import measurable
from wide_paths import PAPER_DIR, SELECTION_CSV, WIDE_DIR

#: One deposit per paper. The study for Geographical Analysis measures the
#: library rather than the schedules, and mixing the two would leave a reader
#: of either unable to tell which files answer which question.
DEPOSIT_DIR = PAPER_DIR / "deposit_cus"

#: Result tables the paper draws on. Anything not listed is intermediate.
TABLES = [
    "METRICS.md",
    "selection.csv",
    "downloads.csv",
    "cities.csv",
    "city_feeds.csv",
    "indicators_coverage.csv",
    "indicators_waits.csv",
    "feed_provenance.csv",
    "matched_segments_summary.csv",
    "speed_by_length.csv",
    "kinematics_fit.csv",
    "kinematics_by_city.csv",
    "accessibility.csv",
    "route_stops.csv",
    "proxy_without_schedule.csv",
    "osm_graphs.csv",
    "gtfs_graphs.csv",
]

#: This paper's pipeline, module by module. ``benchmarks/`` also holds the
#: benchmark suite of the other paper (``bench_*.py``, ``run_all.py``), which
#: measures the library rather than the schedule s and belongs in that paper's
#: own deposit.
CODE = ["wide_*.py", "README.md"]

#: figures/ carries the artwork of two papers. Only the generators, their basemap
#: data and the eleven images this manuscript shows are deposited: copying the
#: directory whole put 22 MB into a 27 MB archive, and most of it was other work.
#: ``overview_figure.tex`` is here because the paper's first figure is drawn in
#: TikZ rather than measured: without its source that one image could not be
#: rebuilt from the deposit, and everything else could.
FIGURE_CODE = ["figure_common.py", "figures_cus.py", "overview_figure.tex", "data/**/*"]

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".venv*", "_*_preview", "*.log")


def manuscript_figures() -> list[str]:
    """Image files the manuscript asks for, read from its own text modules."""
    text = "".join(
        (PAPER_DIR / "manuscript_cus" / name).read_text(encoding="utf-8")
        for name in ("paper_en_head.py", "paper_en_body.py", "paper_en_tail.py")
        if (PAPER_DIR / "manuscript_cus" / name).exists()
    )
    return sorted(set(re.findall(r'page\.figure\(\s*"([^"]+\.png)"', text)))


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def feed_sources() -> pd.DataFrame:
    """Identifier, catalogue, licence and URL for every feed that entered a city.

    This is the table that replaces the archives themselves.
    """
    selection = pd.read_csv(SELECTION_CSV)
    city_feeds = pd.read_csv(WIDE_DIR / "city_feeds.csv").drop_duplicates("city_key", keep="last")
    city_feeds = city_feeds.loc[city_feeds["city_key"].map(str).isin(set(measurable()))]

    used = {}
    for city, feeds in zip(city_feeds["city_key"], city_feeds["feeds"].fillna("")):
        for identifier in re.split(r"[;+]", str(feeds)):
            if identifier:
                used[identifier] = city

    frame = selection.loc[selection["feed_id"].astype(str).isin(used)].copy()
    frame["city_key"] = frame["feed_id"].astype(str).map(used)
    columns = [
        "city_key",
        "feed_id",
        "catalog",
        "provider",
        "name",
        "country_code",
        "country_name",
        "region",
        "income_group",
        "license",
        "url",
    ]
    return frame[[c for c in columns if c in frame.columns]].sort_values(["city_key", "feed_id"])


README = """# Open transit schedules against OpenStreetMap: measurements for {n_cities} cities

This deposit accompanies the paper *Can OpenStreetMap stand in for a published
timetable? A two-directional comparison across 126 cities*. It holds the
measurements the paper reports, the code that produced them, and enough
provenance to rebuild the inputs.

Assembled {today}. Version 2 replaces every figure and the module that draws
them. The artwork is now drawn at the size it is printed, so its lettering and
its line widths are what the journal asks for; three generators whose images the
paper does not show were removed; and the two panels of the Delhi plate, which
had been labelled with the number of graph vertices, now carry the number of
stops the coverage table reports. No stage of the pipeline was rerun: every
measurement in `results/` is the one version 1 carried, and `MANIFEST.sha256`
lets a reader confirm it.

## What is here

| Path | What it is |
|---|---|
| `results/METRICS.md` | every headline number in the paper, generated from the tables beside it |
| `results/*.csv` | the measurements themselves, one row per city or per city and mode |
| `results/feed_sources.csv` | identifier, catalogue, licence and URL of every schedule used |
| `code/benchmarks/` | the pipeline, one module per stage |
| `code/figures/` | the figure generator and the eleven images the paper shows |
| `environment/` | dependency specifications |
| `MANIFEST.sha256` | checksum of every file above |

## What is deliberately not here

The published schedule archives themselves. Some carry licences that forbid
redistribution, and rather than republish some and not others we list the source
of every one in `results/feed_sources.csv`. Catalogues overwrite feeds in place
without versioning, so a rebuild from those URLs will not reproduce this study
exactly; that limitation is stated in the paper.

OpenStreetMap extracts are likewise not included. They are large, they change
daily, and the pipeline fetches them from the Overpass API on demand.

## Reproducing

Create the environment from `environment/environment-bench.yml`, then run the
stages in the order they appear in `code/benchmarks/README.md`. A full sweep over
the cohort costs about five and a half hours of computation once the
OpenStreetMap extracts are cached locally.

The graph library is IduEdu, an open-source Python package; the constants that
produced these results are identified in the paper by the digest
`{fingerprint}`.

## Citing

Cite the paper for the findings and this deposit for the measurements:
<https://doi.org/10.5281/zenodo.22232336>.

## Licence

Everything we made -- the measurements in `results/`, the pipeline in `code/`
and the figures -- is **CC BY 4.0**.

Two things inside are not ours to license, and neither restricts reuse of the
measurements. `code/figures/data/city_boundaries/` holds one administrative
boundary taken straight from OpenStreetMap; unlike the tables, that is not an
analysis result but a copy of the source geometry, so it stays under **ODbL
1.0**, attribution to OpenStreetMap contributors. The basemap beside it is
Natural Earth and is in the public domain.

The tables themselves are counts, medians and correlations computed from
OpenStreetMap and from published schedules. They are analysis output rather than
extracted data, so they are ours to license -- with attribution to OpenStreetMap
contributors, whose data they were computed from. The schedules remain under
their publishers' own licences, listed per feed in `results/feed_sources.csv`.
"""


def build() -> None:
    if DEPOSIT_DIR.exists():
        shutil.rmtree(DEPOSIT_DIR)
    (DEPOSIT_DIR / "results").mkdir(parents=True)
    (DEPOSIT_DIR / "environment").mkdir()

    missing = []
    for name in TABLES:
        source = WIDE_DIR / name
        if source.exists():
            shutil.copy2(source, DEPOSIT_DIR / "results" / name)
        else:
            missing.append(name)

    feed_sources().to_csv(DEPOSIT_DIR / "results" / "feed_sources.csv", index=False)

    pipeline = DEPOSIT_DIR / "code" / "benchmarks"
    pipeline.mkdir(parents=True, exist_ok=True)
    for pattern in CODE:
        for source in sorted((PAPER_DIR / "benchmarks").glob(pattern)):
            if source.is_file():
                shutil.copy2(source, pipeline / source.name)

    figures = DEPOSIT_DIR / "code" / "figures"
    for pattern in FIGURE_CODE:
        for source in sorted((PAPER_DIR / "figures").glob(pattern)):
            if source.is_file():
                target = figures / source.relative_to(PAPER_DIR / "figures")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    for name in manuscript_figures():
        source = PAPER_DIR / "figures" / name
        if source.exists():
            shutil.copy2(source, figures / name)

    for name in ("environment-bench.yml", "requirements-bench.txt"):
        source = PAPER_DIR / name
        if source.exists():
            shutil.copy2(source, DEPOSIT_DIR / "environment" / name)

    from wide_paths import registry_fingerprint

    from iduedu import DEFAULT_REGISTRY

    (DEPOSIT_DIR / "README.md").write_text(
        README.format(
            n_cities=len(measurable()),
            today=date.today().isoformat(),
            fingerprint=registry_fingerprint(DEFAULT_REGISTRY),
        ),
        encoding="utf-8",
    )

    lines, total = [], 0
    for path in sorted(DEPOSIT_DIR.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            lines.append(f"{_checksum(path)}  {path.relative_to(DEPOSIT_DIR).as_posix()}")
            total += path.stat().st_size
    (DEPOSIT_DIR / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"{DEPOSIT_DIR}: {len(lines)} files, {total / 1e6:.1f} MB")
    if missing:
        print("tables not found and therefore not deposited: " + ", ".join(missing))


def archive() -> Path:
    """Pack the deposit into one file, because a Zenodo record has no folders.

    The upload form takes files and lists them flat: two ``README.md`` in
    different directories would collide, and ``code/benchmarks/wide_osm.py``
    would arrive as ``wide_osm.py`` with nothing to say where it belonged.
    A single archive keeps the tree, and Zenodo previews its contents in the
    record, so a visitor still sees the structure without downloading.
    """
    target = DEPOSIT_DIR.with_suffix(".zip")
    if target.exists():
        target.unlink()
    shutil.make_archive(str(DEPOSIT_DIR), "zip", root_dir=DEPOSIT_DIR.parent, base_dir=DEPOSIT_DIR.name)
    print(f"{target.name}: {target.stat().st_size / 1e6:.1f} MB, ready to upload")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="without it, only what would be built is listed")
    arguments = parser.parse_args()
    if not arguments.run:
        print(f"would assemble {DEPOSIT_DIR}")
        print("  tables:      " + ", ".join(TABLES))
        print("  code:        " + ", ".join(CODE))
        print("  feed sources: identifier, catalogue, licence and URL per feed")
        print("Nothing was written. Re-run with --run.")
        return
    build()
    archive()


if __name__ == "__main__":
    main()
