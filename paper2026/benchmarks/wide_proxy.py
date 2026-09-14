"""Can the substitution error be predicted without the schedule?

The headline result of the study is that the error of an OpenStreetMap-only
accessibility estimate follows *relative* completeness: which of the two sources
describes more of the city. That predictor is a difference between two matched
shares, and both shares need the timetable -- so the cities that most need the
prediction, the ones with no published feed, cannot compute it.

This module asks whether a quantity computable from OpenStreetMap alone stands in
for it. The candidate comes from the stop census: OpenStreetMap holds mapped stop
nodes, and it holds stops that a route relation actually reaches. A stop on no
route cannot be routed through, so the ratio between the two says how far the
mapping of a city got before it stopped -- positions collected, lines not drawn.
Nothing in that ratio touches a schedule.

Two candidates are scored, against the predictor they would replace and against
the error itself, and each is checked against city size: a "proxy" that is really
a measure of how big the city is would be worthless in the cities this study is
about.

    python wide_proxy.py
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from wide_cohort import measurable
from wide_paths import ACCESSIBILITY_CSV, CITIES_CSV, COVERAGE_CSV, WIDE_DIR

REPORT_CSV = WIDE_DIR / "proxy_without_schedule.csv"
CENSUS_CSV = WIDE_DIR / "route_stops.csv"

#: Every candidate is a ratio of two counts OpenStreetMap supplies on its own.
#:
#: ``routable_share`` divides the stops our transit graph actually holds by the
#: stop nodes mapped inside the same boundary. The numerator counts distinct
#: stops; a first version of this module used the census column ``stops_total``
#: instead, which counts stop-role *memberships* across relations, so a regional
#: route drags members from beyond the boundary into the count and Madison came
#: out at 156 "stops per stop". That column is left out of the candidates for
#: exactly that reason.
CANDIDATES = {
    "routable_share": "stops in the transit graph / stop nodes mapped in the boundary",
    "relations_with_stops": "route relations that list their stops / route relations",
}

TARGETS = {
    "relative_completeness": "the two-source predictor it would replace",
    "ratio_30": "the substitution error at 30 minutes",
    "ratio_45": "the substitution error at 45 minutes",
}


def _last(path, **kwargs) -> pd.DataFrame:
    """Result tables are append-only; the last row per city is the current one."""
    frame = pd.read_csv(path, **kwargs)
    return frame.drop_duplicates("city_key", keep="last")


def table() -> pd.DataFrame:
    census = _last(CENSUS_CSV)
    coverage = pd.read_csv(COVERAGE_CSV)
    coverage = coverage.loc[coverage["mode"].eq("*") & coverage["status"].eq("ok")]
    coverage = coverage.drop_duplicates("city_key", keep="last")
    access = _last(ACCESSIBILITY_CSV)
    access = access.loc[access["status"].eq("ok")]
    cities = _last(CITIES_CSV)

    frame = census.merge(
        coverage[["city_key", "feed_to_osm_100", "osm_to_feed_100", "n_osm_stops", "n_feed_stops"]],
        on="city_key",
    )
    frame = frame.merge(access[["city_key", "ratio_15", "ratio_30", "ratio_45"]], on="city_key", how="left")
    frame = frame.merge(cities[["city_key", "population"]], on="city_key", how="left")
    frame = frame.loc[frame["city_key"].map(str).isin(set(measurable()))]

    frame["relative_completeness"] = frame["feed_to_osm_100"] - frame["osm_to_feed_100"]
    mapped = frame["stop_nodes_in_boundary"].replace(0, np.nan)
    relations = frame["n_relations"].replace(0, np.nan)
    frame["routable_share"] = frame["n_osm_stops"] / mapped
    frame["relations_with_stops"] = frame["n_with_stops"] / relations
    return frame.replace([np.inf, -np.inf], np.nan)


def _partial_spearman(frame: pd.DataFrame, x: str, y: str, control: str) -> tuple[float, int]:
    """Spearman between x and y with the effect of ``control`` removed.

    Computed by inverting the rank correlation matrix, which is the standard
    partial-correlation identity applied to ranks.
    """
    subset = frame[[x, y, control]].dropna()
    ranks = subset.rank().to_numpy().T
    inverse = np.linalg.inv(np.corrcoef(ranks))
    return -inverse[0, 1] / np.sqrt(inverse[0, 0] * inverse[1, 1]), len(subset)


def correlations(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate in list(CANDIDATES) + ["feed_to_osm_100", "relative_completeness"]:
        row = {"predictor": candidate, "needs the schedule": candidate not in CANDIDATES}
        for target in TARGETS:
            if candidate == target:
                row[target] = None
                continue
            subset = frame[[candidate, target]].dropna()
            rho, p = spearmanr(subset[candidate], subset[target])
            row[target] = round(rho, 3)
            row[f"p_{target}"] = f"{p:.1e}"
            row["n"] = len(subset)
        rho, _ = _partial_spearman(frame, candidate, "ratio_30", "population")
        row["ratio_30 | population"] = round(rho, 3)
        rows.append(row)
    return pd.DataFrame(rows)


def size_check(frame: pd.DataFrame) -> pd.DataFrame:
    """A proxy that is really a size measure would not help a small city."""
    rows = []
    for candidate in CANDIDATES:
        for size, label in (("population", "population"), ("stop_nodes_in_boundary", "stops mapped")):
            subset = frame[[candidate, size]].dropna()
            rho, p = spearmanr(subset[candidate], subset[size])
            rows.append({"predictor": candidate, "against": label, "rho": round(rho, 3), "p": f"{p:.1e}"})
    return pd.DataFrame(rows)


def decision_rule(frame: pd.DataFrame, candidate: str = "relations_with_stops") -> pd.DataFrame:
    """What a practitioner would actually read off: terciles of the proxy."""
    subset = frame[[candidate, "ratio_30", "relative_completeness", "city_key"]].dropna()
    subset = subset.assign(band=pd.qcut(subset[candidate], 3, labels=["low", "middle", "high"]))
    rows = []
    for band, group in subset.groupby("band", observed=True):
        rows.append(
            {
                "band of " + candidate: str(band),
                "range": f"{group[candidate].min():.2f}-{group[candidate].max():.2f}",
                "cities": len(group),
                "median ratio 30 min": round(group["ratio_30"].median(), 3),
                "p10": round(group["ratio_30"].quantile(0.10), 3),
                "p90": round(group["ratio_30"].quantile(0.90), 3),
                "share within 20 per cent": round(float(group["ratio_30"].between(0.8, 1.2).mean()), 3),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    frame = table()
    print(f"{len(frame)} cities in the cohort with a stop census\n")
    print("Does an OpenStreetMap-only quantity predict what the two-source predictor predicts?\n")
    print(correlations(frame).to_string(index=False))
    print("\nIs the proxy really a measure of city size?\n")
    print(size_check(frame).to_string(index=False))
    print("\nWhat a practitioner reads off, terciles of relations_with_stops:\n")
    print(decision_rule(frame).to_string(index=False))

    keep = [
        "city_key",
        "n_relations",
        "stop_nodes_in_boundary",
        "n_osm_stops",
        *CANDIDATES,
        "feed_to_osm_100",
        "osm_to_feed_100",
        "relative_completeness",
        "ratio_15",
        "ratio_30",
        "ratio_45",
    ]
    frame[keep].to_csv(REPORT_CSV, index=False)
    print(f"\nwritten {REPORT_CSV}")


if __name__ == "__main__":
    main()
