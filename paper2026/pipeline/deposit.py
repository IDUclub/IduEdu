"""Assemble the study's Zenodo deposit in ``deposit_cus/``; nothing is uploaded.

The deposit is one archive and the README beside it. At its root the archive holds the
layout the pipeline expects, so the pipeline runs from the unpacked deposit as it runs here:

=============================  ======================================================
``pipeline/``                  the stages (without this script) and their README
``inputs/``                    the frozen inputs
``results/wide_tier/``         every result table and ``METRICS.md``
``figures/``                   the figures, the TikZ source of Figure 1, the basemap
``iduedu-*.tar.gz``            the IduEdu source
``requirements-pipeline.txt``  the packages the pipeline imports
=============================  ======================================================

Two inputs are not deposited. The GTFS archives: their licences differ, and
``inputs/feeds.csv`` gives each one's URL and sha256. The OpenStreetMap extracts of the
cities: their 1.2 GB kept failing to upload, and ``osm.py`` cuts them again from the
Geofabrik regions in ``inputs/geofabrik_regions.csv``.

    python pipeline/deposit.py
"""

import ast
import shutil
import subprocess
import sys
import tempfile
import zipfile
from importlib import metadata
from pathlib import Path

import pandas as pd
from common import FEEDS_CSV, INPUTS_DIR, PAPER_DIR, RESULTS_DIR, cities, regions, setup
from packaging.requirements import Requirement

DEPOSIT_DIR = PAPER_DIR / "deposit_cus"
PIPELINE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = PAPER_DIR / "figures"
REPOSITORY = PAPER_DIR.parent
#: The title of the Zenodo record; the paper's ``references.bib`` cites it under the same title.
TITLE = "OpenStreetMap against published timetables in 265 cities: derived tables, pipeline and feed provenance"
ARCHIVE = "osm_vs_gtfs_265_cities.zip"
#: Read and written through pandas, never imported: the parquet engine.
INDIRECT = ("pyarrow",)


def zip_tree(target: Path, members: list[tuple[Path, str]]) -> int:
    """Zip files and directories under the given archive names; returns the file count."""
    count = 0
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for source, name in members:
            paths = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file())
            for path in paths:
                if "__pycache__" in path.parts or path.suffix in {".pyc", ".log"}:
                    continue
                archive.write(path, name if source.is_file() else f"{name}/{path.relative_to(source).as_posix()}")
                count += 1
    return count


def code_files() -> list[tuple[Path, str]]:
    """The stages and their README; the scripts that drew the cohort are not part of the pipeline."""
    files = [path for path in sorted(PIPELINE_DIR.glob("*.py")) if path.name != Path(__file__).name]
    return [(path, f"pipeline/{path.name}") for path in files + [PIPELINE_DIR / "README.md"]]


def library_source(staging: Path) -> Path:
    """An sdist of IduEdu from the repository."""
    subprocess.run(["uv", "build", "--sdist", "-o", str(staging)], cwd=REPOSITORY, check=True, capture_output=True)
    return next(staging.glob("iduedu-*.tar.gz"))


def requirements(staging: Path) -> Path:
    """The packages the pipeline imports and everything they require, at the versions installed here."""
    local = {path.stem for path in PIPELINE_DIR.glob("*.py")}
    imported = set()
    for path, _ in code_files():
        if path.suffix != ".py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                imported.add(node.module.split(".")[0])
    providers = metadata.packages_distributions()
    queue = [d for name in imported - local - set(sys.stdlib_module_names) for d in providers.get(name, [])]
    queue = [(name, None) for name in queue + ["iduedu", *INDIRECT]]
    pinned, markers = {}, {}
    while queue:
        wanted, marker = queue.pop()
        try:
            distribution = metadata.distribution(wanted)
        except metadata.PackageNotFoundError:
            continue
        name = distribution.metadata["Name"].lower().replace("_", "-")
        # a package some path requires unconditionally loses the platform marker of another path
        markers[name] = None if name in markers and markers[name] is None else marker
        if name in pinned:
            continue
        pinned[name] = distribution.version
        for line in distribution.requires or []:
            requirement = Requirement(line)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                queue.append((requirement.name, str(requirement.marker) if requirement.marker else marker))
    pinned.pop("iduedu", None)  # installed from its own source in the deposit
    lines = [
        f"{name}=={version}" + (f"; {markers[name]}" if markers[name] else "")
        for name, version in sorted(pinned.items())
    ]
    header = [
        f"# Python {sys.version.split()[0]}",
        "# Figure 1 also needs pdflatex and pdftops (TeX), and pymupdf for its PNG",
    ]
    target = staging / "requirements-pipeline.txt"
    target.write_text("\n".join(header + lines) + "\n", encoding="utf-8")
    return target


def readme(library: str, commit: str) -> str:
    feeds = pd.read_csv(FEEDS_CSV, keep_default_na=False)
    plan = regions().drop_duplicates("region")
    return f"""# {TITLE}

Data and code of the paper "Can OpenStreetMap replace a published timetable? A two-directional
comparison of 265 cities" (Computational Urban Science, under review): {len(cities())} cities in
{cities()['country_code'].nunique()} countries, GTFS feeds as published in October 2026, OpenStreetMap as of
2026-10-15T20:21Z (Geofabrik).

## Contents

`{ARCHIVE}` holds:

| path | what |
|---|---|
| `pipeline/` | the study, stage by stage, described in `pipeline/README.md` |
| `inputs/` | the {len(cities())} cities and their boundaries, the {len(feeds)} feeds that serve them (catalogue, URL, licence, sha256), the Geofabrik region of each city, route-type fixes; `inputs/cohort/`: the sampling frame, the outcome of every download, the profile of every archive, the candidate cities of each country |
| `results/wide_tier/` | every result table, and `METRICS.md` with the numbers the paper reports |
| `figures/` | the figures of the paper, the TikZ source of Figure 1, and the Natural Earth basemap |
| `{library}` | the IduEdu library, release 2.3.0 (commit {commit[:10]}), the first to carry what this study added to it |
| `requirements-pipeline.txt` | the packages the pipeline imports, at the versions used |

The GTFS archives and the OpenStreetMap extracts of the cities are not deposited; the next
section explains how to obtain them.

## Reproducing

In the folder the archive is unpacked into:

    pip install "{library}[pbf]" -r requirements-pipeline.txt
    python pipeline/city_feeds.py --fetch    # the GTFS archives, checked against their sha256
    python pipeline/osm.py download          # the Geofabrik regions the cities are cut from
    python pipeline/run.py all               # every stage; results/wide_tier/METRICS.md at the end

A source that has changed since October 2026 cannot reproduce the study exactly, and
`--fetch` keeps such an archive aside as `<feed_id>.zip.changed`.

`osm.py download` fetches the {len(plan)} regions of `inputs/geofabrik_regions.csv`
({plan['bytes'].sum() / 1e9:.0f} GB in October 2026) as Geofabrik serves them on the day, so a
rerun builds its OpenStreetMap graphs from a later snapshot than the study's. The state of
2026-10-15T20:21Z can be recovered from the full history of OpenStreetMap, for instance with
`osmium time-filter`; regional files of that moment, saved as `osm/<region>.osm.pbf` with `_`
for `/` in the region name, take the place of the download. No step after these two
downloads queries a live service: `run.py` cuts one extract per city from the regions and
builds every OpenStreetMap graph from those extracts, without a request to Overpass.

Without the GTFS archives and the OpenStreetMap data, `python pipeline/report.py`
regenerates `METRICS.md` from the result tables, and `python pipeline/figures.py --charts`
redraws every figure except the two city maps, which need the graphs `run.py` builds.

## Licences

- Code (pipeline, IduEdu): BSD 3-Clause.
- Everything derived from OpenStreetMap: ODbL 1.0, © OpenStreetMap contributors.
- GTFS feeds: not redistributed; each feed's licence is in `inputs/feeds.csv`, column `license`.
- Result tables, figures: CC BY 4.0.
"""


def main() -> None:
    logger = setup("deposit")
    if DEPOSIT_DIR.exists():
        shutil.rmtree(DEPOSIT_DIR)
    DEPOSIT_DIR.mkdir(parents=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, capture_output=True, text=True).stdout.strip()

    with tempfile.TemporaryDirectory() as temporary:
        staging = Path(temporary)
        library = library_source(staging)
        # Unix line ends on every platform, so the deposit is the same bytes wherever it is built.
        (DEPOSIT_DIR / "README.md").write_text(readme(library.name, commit), encoding="utf-8", newline="\n")
        members = code_files() + [
            (FIGURES_DIR / "overview_figure.tex", "figures/overview_figure.tex"),
            (FIGURES_DIR / "data" / "naturalearth_lowres", "figures/data/naturalearth_lowres"),
            *[(path, f"figures/{path.name}") for path in sorted(FIGURES_DIR.glob("wide_*.png"))],
            (library, library.name), (requirements(staging), "requirements-pipeline.txt"),
            (INPUTS_DIR, "inputs"),
            *[(path, f"results/wide_tier/{path.name}") for path in sorted(RESULTS_DIR.glob("*")) if path.is_file()],
        ]  # fmt: skip
        logger.info("%s: %d files", ARCHIVE, zip_tree(DEPOSIT_DIR / ARCHIVE, members))

    total = sum(path.stat().st_size for path in DEPOSIT_DIR.iterdir()) / 1e6
    logger.info("deposit: %s, %.1f MB", DEPOSIT_DIR, total)


if __name__ == "__main__":
    main()
