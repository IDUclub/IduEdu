"""Drop the rows a rebuild has made stale, so a resumed sweep recomputes them.

Every measuring stage skips a city that already has a row, which is what makes a
sweep resumable. After the OSM side of some cities changes, those rows are the
old answer to a new question, and the stage will happily leave them in place.

Deleting them is not the same as ``--force``: the cities that did not change keep
their measurements, and only the affected ones are recomputed. The tables this
touches are the ones that read the OSM transit graph.

    python wide_invalidate.py --cities moscow tokyo --apply
"""

import argparse
import logging

import pandas as pd
from wide_paths import (
    ACCESSIBILITY_CSV,
    COVERAGE_CSV,
    PAIRS_CSV,
    PAIRS_SUMMARY_CSV,
    PROVENANCE_CSV,
)

logger = logging.getLogger(__name__)

#: Tables derived from the OSM transit graph. Waiting times come from the schedule
#: side and are left alone.
DERIVED = (COVERAGE_CSV, PROVENANCE_CSV, PAIRS_CSV, PAIRS_SUMMARY_CSV, ACCESSIBILITY_CSV)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", nargs="+", required=True)
    parser.add_argument("--apply", action="store_true", help="write; without it, only report")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    wanted = {str(key) for key in arguments.cities}
    for path in DERIVED:
        if not path.exists():
            logger.info("%-32s missing", path.name)
            continue
        frame = pd.read_csv(path)
        if "city_key" not in frame.columns:
            continue
        stale = frame["city_key"].map(str).isin(wanted)
        logger.info(
            "%-32s %6d rows, %5d stale (%d cities)",
            path.name,
            len(frame),
            int(stale.sum()),
            frame.loc[stale, "city_key"].nunique(),
        )
        if arguments.apply and stale.any():
            frame.loc[~stale].to_csv(path, index=False, encoding="utf-8")

    print("\napplied" if arguments.apply else "\ndry run; pass --apply to write")


if __name__ == "__main__":
    main()
