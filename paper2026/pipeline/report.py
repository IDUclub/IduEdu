"""``METRICS.md``: the numbers the paper reports, regenerated from the inputs and the result tables.

One section per part of the paper. Nothing is measured here; the stages measure, and this
gathers what the text quotes, so the text can be checked against the data.

    python pipeline/report.py
"""

import pandas as pd
from common import INPUTS_DIR, RESULTS_DIR, THRESHOLDS_MIN, cities, last_rows, measurable, setup

METRICS_MD = RESULTS_DIR / "METRICS.md"
#: A feed whose geometry shares more vertices than this with OSM was traced from it.
DERIVED_THRESHOLD = 0.30
#: A timetable where more than this share of rides take zero minutes moves passengers at no cost.
ZERO_TIME_LIMIT = 0.2
MODE_ORDER = ["bus", "tram", "trolleybus", "subway", "train"]
#: The catalogues that link to the operator's server rather than keep a copy of the feed.
LINKING_CATALOGUES = ("mobility_database", "transitland_atlas", "dt4a")
#: The snapshot both sources are taken at.
SNAPSHOT = pd.Timestamp("2026-10-15")


def _last(name: str, keys: list[str] | None = None) -> pd.DataFrame | None:
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if "city_key" in frame:
        frame = frame[frame["city_key"].isin(measurable())]
    return frame.drop_duplicates(subset=keys, keep="last") if keys else frame


# --- metrics ----------------------------------------------------------------------------


def section(title: str) -> str:
    return f"\n## {title}\n\n"


def table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(map(str, frame.columns)) + " |\n|" + "|".join("---" for _ in frame.columns) + "|\n"
    return header + "".join(
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |\n" for row in frame.to_numpy()
    )


def _sample(out: list) -> None:
    cohort = INPUTS_DIR / "cohort"
    selection, downloads, profile = (
        pd.read_csv(cohort / n) for n in ("selection.csv", "downloads.csv", "feed_profile.csv")
    )
    feeds = last_rows(RESULTS_DIR / "city_feeds.csv", ["city_key"])
    osm, gtfs, access = (
        _last("osm_graphs.csv", ["city_key", "kind"]),
        _last("gtfs_graphs.csv", ["city_key"]),
        _last("accessibility.csv", ["city_key"]),
    )
    out.append(section("The sample, and what it cost to get it"))
    out.append(table(pd.DataFrame([
        ("selected from catalogues", selection["feed_id"].nunique()),
        ("downloaded successfully", int((downloads.drop_duplicates("feed_id", keep="last")["status"] == "ok").sum())),
        ("passed GTFS validation", int(profile.drop_duplicates("feed_id", keep="last")["ok"].astype(str).str.lower().isin(["true", "1"]).sum())),
        ("study cities (frozen inputs)", len(cities())),
        ("merged into a city feed", int((feeds["status"] == "ok").sum())),
        ("OSM transit graph built", int(((osm["kind"] == "pt") & (osm["status"] == "ok")).sum())),
        ("pedestrian graph built", int(((osm["kind"] == "walk") & (osm["status"] == "ok")).sum())),
        ("GTFS reference graph built", int((gtfs["status"] == "ok").sum())),
        ("accessibility compared", int((access["status"] == "ok").sum())),
    ], columns=["stage", "count"])))  # fmt: skip
    walk, transit = (
        osm[(osm["kind"] == "walk") & (osm["status"] == "ok")],
        osm[(osm["kind"] == "pt") & (osm["status"] == "ok")],
    )
    out.append(f"\nGraph sizes: pedestrian {int(walk['n_nodes'].sum()):,} nodes / {int(walk['n_edges'].sum()):,} edges; "
               f"OSM transit {int(transit['n_nodes'].sum()):,} / {int(transit['n_edges'].sum()):,}; "
               f"GTFS reference {int(gtfs['n_nodes'].fillna(0).sum()):,} / {int(gtfs['n_edges'].fillna(0).sum()):,}.\n")  # fmt: skip
    out.append(section("Defects in published feeds, repaired at merge time"))
    repairs = [("colliding_rows_dropped", "one identifier used for different objects"), ("repeated_rows_dropped", "byte-identical duplicate rows"),
               ("unusable_frequencies_dropped", "frequency rows with an empty window"),
               ("trips_dropped_bad_times", "trips whose times run backwards or never close"),
               ("orphan_rows_dropped", "rows referencing identifiers no table defines")]  # fmt: skip
    out.append(table(pd.DataFrame([{"defect": label, "rows or trips": int(feeds[c].fillna(0).sum()), "cities affected": int((feeds[c].fillna(0) > 0).sum())}
                                   for c, label in repairs if c in feeds])))  # fmt: skip
    broken = (
        gtfs.loc[gtfs["zero_time_share"].fillna(0) >= 1.0, "city_key"].tolist() if "zero_time_share" in gtfs else []
    )
    out.append(f"\nCities whose schedule carries no run time at all: {', '.join(broken) or 'none'}.\n")


def _frame(out: list) -> None:
    cohort = INPUTS_DIR / "cohort"
    outcomes = pd.read_csv(cohort / "feed_outcomes.csv", keep_default_na=False)
    profile = pd.read_csv(cohort / "feed_profile.csv", keep_default_na=False)
    out.append(section("The sampling frame: what each catalogue entry returned"))
    lost = outcomes[outcomes["outcome"] != "ok"]
    lost = lost.assign(reason=lost["reason"].where(lost["outcome"] == "invalid", "network failure"))
    out.append(table(lost.groupby(["outcome", "reason"]).size().rename("entries").reset_index()))
    by_catalogue = outcomes.groupby("catalog")["outcome"].agg(
        entries="size", retrieved=lambda s: int((s == "ok").sum())
    )
    out += ["\nBy catalogue:\n\n", table(by_catalogue.reset_index())]
    linking = outcomes[outcomes["catalog"].isin(LINKING_CATALOGUES)]
    by_income = linking.groupby("income_group")["outcome"].agg(
        entries="size", retrieved_share=lambda s: round(float((s == "ok").mean()), 3)
    )
    out += ["\nThe catalogues that link to the operator's server, by World Bank income group:\n\n",
            table(by_income.reset_index())]  # fmt: skip
    retrieved = linking[linking["outcome"] == "ok"].merge(profile[["feed_id", "service_to"]], on="feed_id", how="left")
    ends = pd.to_datetime(retrieved["service_to"].astype(str), format="%Y%m%d", errors="coerce")
    dated = retrieved.assign(lapsed=ends < SNAPSHOT)[ends.notna()]
    lapsed = dated.groupby("income_group")["lapsed"].agg(
        archives="size", lapsed_share=lambda s: round(float(s.mean()), 3)
    )
    out += [f"\nOf the archives they returned, those with a stated end of service before {SNAPSHOT.date()}:\n\n",
            table(lapsed.reset_index())]  # fmt: skip
    out += ["\nEntries of the frame by World Bank region:\n\n",
            table(outcomes.groupby("region").size().rename("entries").reset_index())]  # fmt: skip


def _coverage(out: list) -> None:
    from coverage import CASE_ORDER, NO_NETWORK, classify_bus

    coverage = _last("indicators_coverage.csv", ["city_key", "mode"])
    per_mode = coverage[
        (coverage["mode"] != "*")
        & (coverage["status"] == "ok")
        & coverage["fetchable_from_osm"].astype(str).str.lower().eq("true")
    ]
    out.append(section("Coverage: does the other source know this stop?"))
    rows, presence = [], []
    for mode in MODE_ORDER:
        m = per_mode[per_mode["mode"] == mode]
        both = m[m["n_feed_stops"].gt(0) & m["n_osm_stops"].gt(0)]
        if len(both) >= 2:
            rows.append({"mode": mode, "cities with both": len(both),
                         **{f"feed→OSM {r} m": round(both[f"feed_to_osm_{r}"].median(), 3) for r in (50, 100, 150)},
                         "IQR at 100 m": f"{both['feed_to_osm_100'].quantile(0.25):.2f}–{both['feed_to_osm_100'].quantile(0.75):.2f}",
                         "OSM→feed 100 m": round(both["osm_to_feed_100"].median(), 3)})  # fmt: skip
        presence.append({"mode": mode, "in the feed": int(m["n_feed_stops"].gt(0).sum()), "in OSM": int(m["n_osm_stops"].gt(0).sum()),
                         "in both": len(both), "OSM only": int((m["n_osm_stops"].gt(0) & m["n_feed_stops"].eq(0)).sum()),
                         "feed only": int((m["n_feed_stops"].gt(0) & m["n_osm_stops"].eq(0)).sum())})  # fmt: skip
    out += [table(pd.DataFrame(rows)), "\nWhich source runs the mode at all:\n\n", table(pd.DataFrame(presence))]
    bus = classify_bus(per_mode)
    cases = bus.groupby("case").agg(cities=("city_key", "size"), median_coverage=("feed_to_osm_100", "median"),
                                    median_distance_m=("nearest_median_m", "median")).round(3)  # fmt: skip
    out += [
        "\nLow coverage is not one phenomenon:\n\n",
        table(cases.reindex([c for c in CASE_ORDER if c in cases.index]).reset_index()),
    ]
    census = _last("route_stops.csv", ["city_key"])
    if census is not None and "share_relations_with_stops" in census:
        ok = census[census["status"] == "ok"]
        out.append(section("Route relations and stops in the OSM snapshot"))
        out.append(f"{len(census)} cities; {int(census['status'].eq('no relations').sum())} hold no route relation inside the boundary. "
                   f"Medians over the {len(ok)} that do: relations {int(ok['n_relations'].median())}, share listing a stop "
                   f"{ok['share_relations_with_stops'].median():.3f}, stop nodes mapped {int(ok['stop_nodes_in_boundary'].median())}, "
                   f"share of them on a route {ok['share_stops_on_a_route'].median():.3f}.\n\n")  # fmt: skip
        merged = bus.merge(
            census[["city_key", "stop_nodes_in_boundary", "n_relations", "share_stops_on_a_route"]],
            on="city_key",
            how="left",
        )
        out.append(table(pd.DataFrame([{
            "case": case, "cities": len(g), "stops mapped in OSM": int(g["stop_nodes_in_boundary"].median()),
            "share of them on a route": round(g["share_stops_on_a_route"].median(), 3), "stops in the graph": int(g["n_osm_stops"].median()),
            "route relations": int(g["n_relations"].median()), "stops in the feed": int(g["n_feed_stops"].median()),
        } for case in CASE_ORDER if len(g := merged[merged["case"] == case])])))  # fmt: skip
        mapped = merged[(merged["case"] == NO_NETWORK) & (merged["stop_nodes_in_boundary"].fillna(0) >= 20)]
        out.append(f"\nOf the {int((merged['case'] == NO_NETWORK).sum())} cities whose bus network OSM does not route, {len(mapped)} "
                   "have 20 or more stops mapped inside the boundary all the same: a stop on no route never enters a graph.\n")  # fmt: skip


def _times(out: list) -> None:
    waits = _last("waits_by_mode.csv")
    if waits is not None:
        out.append(section("Waiting time: each city's own, median across cities (registry constants)"))
        out.append(
            table(
                waits[
                    [
                        "mode",
                        "cities",
                        "library_constant_min",
                        "wait_p10",
                        "wait_p25",
                        "wait_median",
                        "wait_p75",
                        "wait_p90",
                    ]
                ]
            )
        )
    fit = _last("speed_fit.csv")
    if fit is not None:
        out.append(section("Run time by segment length, fitted on every timetable (registry constants)"))
        out.append("Shaped segments, faster-than-vmax removed; error |log(pred/obs)| weighted by minutes within a city, median "
                   "over cities; `cv_` scored on held-out cities (10 folds). The registry takes the growing free speed where "
                   "its held-out error is lower (tram, train).\n\n")  # fmt: skip
        columns = ["mode", "model", "cities", "segments", "removed_share", "dwell_min", "speed_kmh", "long_speed_kmh",
                   "half_length_m", "error_time", "bias_time", "cv_error_time", "cv_bias_time"]  # fmt: skip
        out.append(table(fit[[c for c in columns if c in fit]].round(4)))
        curves = _last("speed_fit_curves.csv")
        out += ["\nObserved median speed by length class (km/h) and each class's share of scheduled minutes:\n\n",
                table(curves[["mode", "length_class", "segments", "median_length_m", "observed_kmh", "minutes_share"]])]  # fmt: skip
    provenance = _last("feed_provenance.csv", ["city_key"])
    ok = provenance[provenance["status"] == "ok"]
    derived = ok[ok["coincident_share"] >= DERIVED_THRESHOLD].sort_values("coincident_share", ascending=False)
    out.append(section("Feed provenance: geometry copied from OSM"))
    out.append(f"{len(derived)} of {len(ok)} measurable feeds were traced from OSM (share of coincident vertices >= 0.30); "
               f"{int((ok['coincident_share'] < 0.05).sum())} are clearly surveyed (< 0.05).\n\n")  # fmt: skip
    out.append(table(derived[["city_key", "coincident_share", "median_distance_m"]].round(3).head(20)))


def _accessibility(out: list) -> None:
    access = _last("accessibility.csv", ["city_key"])
    coverage = _last("indicators_coverage.csv", ["city_key", "mode"])
    city = coverage[(coverage["mode"] == "*") & (coverage["status"] == "ok")]
    joined = access[access["status"] == "ok"].merge(
        city[["city_key", "feed_to_osm_100", "osm_to_feed_100"]], on="city_key", how="left"
    )
    out.append(section("Accessibility: OSM with the registry against the timetable (A/D)"))
    out.append(table(pd.DataFrame([{
        "threshold": f"{t} min", "cities": int(joined[f"ratio_{t}"].notna().sum()), "median": round(joined[f"ratio_{t}"].median(), 3),
        "p10": round(joined[f"ratio_{t}"].quantile(0.1), 3), "p90": round(joined[f"ratio_{t}"].quantile(0.9), 3),
        "within 10 %": round(float(joined[f"ratio_{t}"].between(0.9, 1.1).mean()), 3),
        "within 15 %": round(float(joined[f"ratio_{t}"].between(0.85, 1.15).mean()), 3),
        "within 25 %": round(float(joined[f"ratio_{t}"].between(0.75, 1.25).mean()), 3),
        "below 0.8": round(float((joined[f"ratio_{t}"] < 0.8).mean()), 3),
        "above 1.25": round(float((joined[f"ratio_{t}"] > 1.25).mean()), 3),
    } for t in THRESHOLDS_MIN])))  # fmt: skip
    missing = joined.loc[joined["feed_to_osm_100"].isna(), "city_key"]
    out.append(f"\nAll cities with both estimates; {len(missing)} of them share no mode between the sources and have "
               f"no completeness, so the rows below that use it count {len(joined) - len(missing)}.\n")  # fmt: skip
    bands = joined.assign(band=pd.cut(joined["feed_to_osm_100"], [0, 0.5, 0.8, 0.9, 0.95, 1.001], include_lowest=True))
    out += ["\nA/D by the share of feed stops found in OSM (30 minutes):\n\n", table(pd.DataFrame([{
        "feed stops found in OSM": str(band), "cities": len(g), "median A/D": round(g["ratio_30"].median(), 3),
        "p10": round(g["ratio_30"].quantile(0.1), 3), "p90": round(g["ratio_30"].quantile(0.9), 3),
        "within 15 %": round(float(g["ratio_30"].between(0.85, 1.15).mean()), 3),
    } for band, g in bands.groupby("band", observed=True)]))]  # fmt: skip
    extremes = pd.concat([joined.nsmallest(6, "ratio_30"), joined.nlargest(6, "ratio_30")])
    out += ["\nMost underestimated and most overestimated at 30 minutes:\n\n",
            table(extremes[["city_key", "feed_to_osm_100", "osm_to_feed_100", "ratio_30"]].round(3))]  # fmt: skip
    summary, decomposition = _last("accessibility_variants_summary.csv"), _last("accessibility_decomposition.csv")
    if summary is not None and decomposition is not None:
        out.append(section("Four variants: where the gap comes from"))
        out.append("D timetable; D1 its waits averaged per city and mode; A_w OSM with the city's waits and the "
                   "registry's run times; A OSM with the registry's constants (world median waits).\n\n")  # fmt: skip
        out += [
            table(summary),
            "\nShare of the spread of log(A/D) per step, cov(step, total) / var(total):\n\n",
            table(decomposition),
        ]


def _robustness(out: list) -> None:
    access = _last("accessibility.csv", ["city_key"])
    access = access[access["status"] == "ok"]
    provenance = _last("feed_provenance.csv", ["city_key"]).set_index("city_key")["coincident_share"]
    zero = _last("gtfs_graphs.csv", ["city_key"]).set_index("city_key")["zero_time_share"]
    derived = access["city_key"].map(provenance).fillna(0) >= DERIVED_THRESHOLD
    timeless = access["city_key"].map(zero) > ZERO_TIME_LIMIT
    out.append(section("The headline medians without two kinds of feed"))
    out.append(table(pd.DataFrame([{
        "cities": label, "count": len(part), **{f"median {t} min": round(part[f"ratio_{t}"].median(), 3) for t in THRESHOLDS_MIN},
    } for label, part in (("all", access), ("feeds not derived from OSM", access[~derived]),
                          ("under a fifth of timetable rides take zero minutes", access[~timeless]))])))  # fmt: skip
    proxy = _last("proxy_without_schedule.csv", ["city_key"])
    out.append(section("Route relations that list a stop"))
    rated = proxy.dropna(subset=["relations_with_stops", "ratio_30"])
    share = rated["relations_with_stops"]
    groups = {"not all": share < 1, "all": share >= 1}
    unlisted = sorted(proxy.loc[proxy["relations_with_stops"].isna() & proxy["ratio_30"].notna(), "city_key"])
    out.append(f"{len(unlisted)} cities with both estimates have no route relation at all: {', '.join(unlisted)}.\n\n")
    out += ["A/D by the share of route relations that list a stop:\n\n", table(pd.DataFrame([{
        "relations listing a stop": label, "cities": int(inside.sum()),
        "median A/D 30": round(rated.loc[inside, "ratio_30"].median(), 3),
        "p10": round(rated.loc[inside, "ratio_30"].quantile(0.1), 3),
        "p90": round(rated.loc[inside, "ratio_30"].quantile(0.9), 3),
        "within 15 %": round(float(rated.loc[inside, "ratio_30"].between(0.85, 1.15).mean()), 3),
        "below 0.8": round(float((rated.loc[inside, "ratio_30"] < 0.8).mean()), 3),
        "median A/D 45": round(rated.loc[inside, "ratio_45"].median(), 3),
    } for label, inside in groups.items()]))]  # fmt: skip


def metrics() -> str:
    out = [
        "# The numbers of the paper\n",
        "Generated by `pipeline/report.py` from `inputs/` and `results/wide_tier/`.\n",
    ]
    for part in (_sample, _frame, _coverage, _times, _accessibility, _robustness):
        part(out)
    return "".join(out)


def main() -> None:
    logger = setup("report")
    METRICS_MD.write_text(metrics(), encoding="utf-8", newline="\n")
    logger.info("written %s", METRICS_MD)


if __name__ == "__main__":
    main()
