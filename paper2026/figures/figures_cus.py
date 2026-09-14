"""Every figure the Computational Urban Science paper shows, and its journal export.

Three modules used to divide this work -- one for the charts, one for the city
maps, one for packaging -- and the division cost more than it saved: the palette
was defined twice under one name, the packaging step re-ran the chart module as a
subprocess, and a reader looking for "the figures of this paper" had to know
which of the three to open. The artwork of the Geographical Analysis paper lives
in the same directory but is not made here: its charts come out of the benchmark
runs and its two diagrams are standalone TikZ.

Written as a script rather than a notebook on purpose: eleven figures over eight
result tables are regenerated whenever a sweep is rerun, and a script is what a
reviewer can execute.

    python figures_cus.py            # every figure, as PNG
    python figures_cus.py --journal  # numbered EPS and TIFF for submission
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_common import (
    BODY,
    CASE_COLORS,
    CASE_ORDER,
    COLUMN_IN,
    COVERED,
    DPI,
    ELSEWHERE,
    HAIRLINE,
    HEAVY,
    LABEL,
    MODE_ORDER,
    NO_NETWORK,
    PANEL,
    PT_COLORS,
    RASTER_DPI,
    RULE,
    SPARSE,
    STROKE,
    VECTOR,
    C,
    canvas,
    city_coverage,
    classify_bus,
    load,
    per_mode_coverage,
    pt,
    save,
    style_ax,
)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter, ScalarFormatter
from PIL import Image
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from wide_cohort import measurable as cohort_cities  # noqa: E402
from wide_paths import ALL_OSM_MODES, gtfs_graph_path, osm_graph_path  # noqa: E402

from iduedu import read_urban_graph  # noqa: E402

HERE = Path(__file__).resolve().parent
MANUSCRIPT = HERE.parent / "manuscript_cus"
OUTPUT_DIR = HERE / "for_journal"

#: Maps use their own palette: there the two sources are the subject, not the modes.
MAP = {"osm": "#AF52DE", "gtfs": "#00A8FF", "walk": "#D8D8DC", "dark": "#3A3A3C", "gray": "#8E8E93"}

#: City plates draw a whole pedestrian network, so their vector form is hundreds
#: of thousands of line segments -- 33 MB and 74 MB against 0.1 MB for a bar
#: chart. The journal takes EPS or TIFF, and for these two TIFF at 300 dpi is the
#: honest choice: nobody zooms into an individual street, and a reviewer's PDF
#: reader should not have to draw them.
RASTER = {"wide_case_displacement_maps.png", "wide_two_sources.png"}


# ---------------------------------------------------------------------------
# Figure 5. The signature figure: completeness has two directions, not one
# ---------------------------------------------------------------------------
def figure_coverage_scatter() -> None:
    frame = city_coverage().dropna(subset=["feed_to_osm_100", "osm_to_feed_100"])
    fig, ax = plt.subplots(figsize=canvas(0.92), dpi=DPI)
    style_ax(ax, ygrid_only=False)

    ax.axhline(0.5, color="black", alpha=0.18, linewidth=RULE, zorder=2)
    ax.axvline(0.5, color="black", alpha=0.18, linewidth=RULE, zorder=2)
    # Captions sit in the middle of each quadrant, which is empty in every run, rather
    # than in the corners, which hold points; a quarter of the axes away from the lines.
    quadrants = [
        (0.26, 0.74, "OSM richer than the feed" + chr(10) + "(estimate runs high)", "center"),
        (0.74, 0.74, "both sources agree", "center"),
        (0.26, 0.26, "both sources thin", "center"),
        (0.74, 0.26, "feed richer than OSM" + chr(10) + "(estimate runs low)", "center"),
    ]
    for x, y, label, ha in quadrants:
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            fontsize=BODY,
            # Not grey: the journal wants 4.5:1 contrast, and #8E8E93 on white gives 3.0.
            color=C["dark"],
            ha=ha,
            va="center",
            style="italic",
            zorder=2,
            linespacing=1.3,
        )

    sizes = 14 + 120 * (frame["n_feed_stops"].rank(pct=True))
    ax.scatter(
        frame["feed_to_osm_100"],
        frame["osm_to_feed_100"],
        s=sizes,
        c=C["iduedu"],
        alpha=0.55,
        edgecolor="white",
        linewidth=HAIRLINE,
        zorder=3,
    )
    # Only the cities the text argues about, and each label pushed off its point in
    # the direction with room. Labels used to be placed identically and collided
    # wherever cities cluster, which is exactly where the interesting ones sit.
    labelled = {
        "bogota": (9, -4),
        "nairobi": (9, -4),
        "cairo": (9, -4),
        "curitiba": (9, 4),
        "ljubljana": (9, -4),
        "tokyo": (-9, 4),
        "sankt_peterburg": (-9, 4),
        "redland_city": (9, 4),
        "london": (9, -4),
    }
    placed: list[tuple[float, float]] = []
    for key, (dx, dy) in labelled.items():
        row = frame.loc[frame["city_key"].eq(key)]
        if row.empty:
            continue
        row = row.iloc[0]
        point = (row["feed_to_osm_100"], row["osm_to_feed_100"])
        # If two labelled cities land on the same spot, drop the later one instead
        # of stacking text on text.
        if any(abs(point[0] - x) < 0.04 and abs(point[1] - y) < 0.04 for x, y in placed):
            continue
        placed.append(point)
        ax.annotate(
            key.replace("_", " ").title().replace("Sankt Peterburg", "St Petersburg"),
            point,
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=BODY,
            ha="left" if dx > 0 else "right",
            color=C["dark"],
            path_effects=STROKE,
            zorder=5,
        )

    ax.set_xlabel("share of feed stops found in OSM", fontsize=LABEL)
    ax.set_ylabel("share of OSM stops found in the feed", fontsize=LABEL)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    save(fig, "wide_coverage_scatter.png")


# ---------------------------------------------------------------------------
# Figure 4. Coverage by mode, with the matching radius made visible
# ---------------------------------------------------------------------------
def figure_coverage_by_mode() -> None:
    frame = per_mode_coverage()
    # Both directions over the cities where both sources run the mode, matching the
    # metrics table. Giving each direction its own population made the reverse share
    # for rail collapse to zero, because it was then dominated by cities whose feed
    # has no rail to match; how many cities have the mode at all is a separate table.
    frame = frame.loc[frame["n_feed_stops"].gt(0) & frame["n_osm_stops"].gt(0)]
    modes = [mode for mode in MODE_ORDER if frame["mode"].eq(mode).sum() >= 3]

    fig, axes = plt.subplots(2, 1, figsize=canvas(0.68), dpi=DPI, sharex=True, sharey=True)
    for ax, (direction, denominator, title) in zip(
        axes,
        [
            ("feed_to_osm", "n_feed_stops", "(a) feed → OSM: does OSM know this stop?"),
            ("osm_to_feed", "n_osm_stops", "(b) OSM → feed: does the feed know this stop?"),
        ],
    ):
        style_ax(ax)
        width, gap = 0.24, 0.34
        cursor, centres = 0.0, []
        for mode in modes:
            served = frame.loc[frame["mode"].eq(mode)]
            left = cursor
            for radius, alpha in ((50, 0.40), (100, 0.75), (150, 1.0)):
                values = served[f"{direction}_{radius}"].dropna()
                if values.empty:
                    cursor += width
                    continue
                ax.bar(
                    cursor,
                    values.median(),
                    width=width,
                    align="edge",
                    color=PT_COLORS.get(mode, C["gray"]),
                    alpha=alpha,
                    edgecolor="white",
                    linewidth=HAIRLINE,
                    zorder=3,
                )
                if radius == 100:
                    upper = values.quantile(0.75)
                    ax.plot(
                        [cursor + width / 2] * 2,
                        [values.quantile(0.25), upper],
                        color="black",
                        alpha=0.45,
                        linewidth=RULE,
                        zorder=4,
                    )
                    # Above the whisker, not above the median: at the median the
                    # digits sat on the interquartile bar and were unreadable for
                    # every mode whose spread is wide.
                    ax.text(
                        cursor + width / 2,
                        min(upper, 1.0) + 0.022,
                        f"{values.median():.2f}",
                        ha="center",
                        fontsize=BODY,
                        color=C["dark"],
                        path_effects=STROKE,
                        zorder=5,
                    )
                cursor += width
            centres.append((mode, len(served), (left + cursor) / 2))
            cursor += gap

        ax.set_xticks([centre for _, _, centre in centres])
        ax.set_xticklabels([f"{mode}\n{count} cities" for mode, count, _ in centres], fontsize=BODY)
        ax.set_ylabel("median share across cities", fontsize=LABEL)
        ax.set_title(title, fontsize=PANEL, loc="left", color=C["dark"])

    # In the axes the legend landed on the bus whisker, which is the widest in the
    # figure; at figure level it has room of its own.
    fig.legend(
        handles=[
            Patch(facecolor=C["gray"], alpha=alpha, label=f"{radius} m")
            for radius, alpha in ((50, 0.40), (100, 0.75), (150, 1.0))
        ],
        title="matching radius",
        frameon=False,
        fontsize=BODY,
        title_fontsize=BODY,
        loc="upper right",
        ncols=3,
        handlelength=1.1,
        handletextpad=0.4,
        columnspacing=1.1,
        bbox_to_anchor=(0.995, 1.005),
    )
    for ax in axes:
        ax.set_ylim(0, 1.16)
    save(fig, "wide_coverage_by_mode.png")


# ---------------------------------------------------------------------------
# Figure 6. Low coverage is not one phenomenon
# ---------------------------------------------------------------------------
def figure_coverage_cases() -> None:
    bus = classify_bus(per_mode_coverage())
    order = CASE_ORDER

    fig = plt.figure(figsize=canvas(0.92), dpi=DPI, layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]

    ax = axes[0]
    style_ax(ax)
    counts = [int((bus["case"] == case).sum()) for case in order]
    ax.bar(
        range(len(order)),
        counts,
        color=[CASE_COLORS[case] for case in order],
        edgecolor="white",
        linewidth=HAIRLINE,
        width=0.68,
        zorder=3,
    )
    for index, value in enumerate(counts):
        ax.text(
            index,
            value + 0.6,
            str(value),
            ha="center",
            fontsize=BODY,
            fontweight="bold",
            color=C["dark"],
            path_effects=STROKE,
            zorder=5,
        )
    short = {
        COVERED: "covered",
        SPARSE: "sparse",
        ELSEWHERE: "elsewhere",
        NO_NETWORK: "no network",
    }
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([short[case] for case in order], fontsize=BODY, rotation=30, ha="right")
    ax.set_ylabel("cities with a bus feed", fontsize=LABEL)
    ax.set_title("(a)", fontsize=PANEL, loc="left", color=C["dark"])

    ax = axes[1]
    style_ax(ax)
    positions, labels = [], []
    for index, case in enumerate(order):
        values = bus.loc[bus["case"].eq(case), "nearest_median_m"].dropna()
        if values.empty:
            continue
        parts = ax.boxplot(values, positions=[index], widths=0.55, patch_artist=True, showfliers=False)
        for patch in parts["boxes"]:
            patch.set_facecolor(CASE_COLORS[case])
            patch.set_alpha(0.75)
            patch.set_edgecolor("white")
        for element in ("whiskers", "caps", "medians"):
            for line in parts[element]:
                line.set_color(C["dark"])
                line.set_linewidth(RULE)
        positions.append(index)
        labels.append(short[case])
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=BODY, rotation=30, ha="right")
    ax.set_ylabel("distance to the nearest OSM stop, m", fontsize=LABEL)
    ax.set_title("(b)", fontsize=PANEL, loc="left", color=C["dark"])

    ax = axes[2]
    style_ax(ax)
    displaced = bus.loc[bus["case"].eq(ELSEWHERE)].nlargest(12, "nearest_median_m")
    y = np.arange(len(displaced))
    ax.barh(
        y,
        displaced["nearest_median_m"] / 1000,
        color=CASE_COLORS[ELSEWHERE],
        edgecolor="white",
        linewidth=HAIRLINE,
        height=0.68,
        zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([key.replace("_", " ").title() for key in displaced["city_key"]], fontsize=BODY)
    ax.invert_yaxis()
    ax.set_xlabel("median distance from a feed stop to the nearest OSM stop, km", fontsize=LABEL)
    ax.set_title("(c)", fontsize=PANEL, loc="left", color=C["dark"])
    save(fig, "wide_coverage_cases.png")


# ---------------------------------------------------------------------------
# Figure 8. Speed against segment length: why one free-flow speed cannot work
# ---------------------------------------------------------------------------
def figure_speed_by_length() -> None:
    frame = load("speed_by_length.csv")
    if frame is None:
        return
    frame["left"] = frame["length_class"].str.extract(r"\((\d+(?:\.\d+)?),")[0].astype(float)
    frame["right"] = frame["length_class"].str.extract(r",\s*([\d.]+|inf)\]")[0].replace("inf", "8000").astype(float)
    frame["centre"] = np.sqrt(frame["left"].clip(lower=60) * frame["right"])

    fig, ax = plt.subplots(figsize=canvas(0.60), dpi=DPI)
    style_ax(ax, ygrid_only=False)
    markers = {"bus": "o", "tram": "s", "trolleybus": "^", "subway": "D", "train": "v"}
    for mode in MODE_ORDER:
        sub = frame.loc[frame["mode"].eq(mode) & frame["count"].ge(30)].sort_values("centre")
        if len(sub) < 3:
            continue
        ax.plot(
            sub["centre"],
            sub["median"],
            marker=markers.get(mode, "o"),
            markersize=pt(3.5),
            linewidth=HEAVY,
            color=PT_COLORS.get(mode, C["gray"]),
            label=f"{mode} ({int(sub['count'].sum()):,} segments)",
            zorder=4,
            markeredgecolor="white",
            markeredgewidth=HAIRLINE,
        )
    ax.set_xscale("log")
    ax.set_xlabel("segment length, m (log scale)", fontsize=LABEL)
    ax.set_ylabel("observed speed from the schedule, km/h", fontsize=LABEL)
    ax.legend(frameon=False, fontsize=BODY, loc="upper left")
    save(fig, "wide_speed_by_length.png")


# ---------------------------------------------------------------------------
# Figure 9. Provenance
# ---------------------------------------------------------------------------
def figure_provenance() -> None:
    frame = load("feed_provenance.csv", ["city_key"])
    if frame is None:
        return
    frame = frame.loc[frame["status"].eq("ok")].sort_values("coincident_share", ascending=False)
    colours = [C["red"] if value >= 0.30 else C["iduedu"] for value in frame["coincident_share"]]

    fig, axes = plt.subplots(1, 2, figsize=canvas(0.62), dpi=DPI, gridspec_kw={"width_ratios": [1.3, 1.0]})

    ax = axes[0]
    style_ax(ax)
    ax.bar(
        range(len(frame)),
        frame["coincident_share"],
        color=colours,
        width=1.0,
        edgecolor="white",
        linewidth=HAIRLINE,
        zorder=3,
    )
    ax.axhline(0.30, color=C["dark"], linestyle="--", linewidth=RULE, zorder=5)
    ax.axhline(0.05, color=C["dark"], linestyle=":", linewidth=RULE, zorder=5)
    ax.text(len(frame) * 0.30, 0.36, "copied from OSM", fontsize=BODY, color=C["dark"], path_effects=STROKE)
    ax.text(len(frame) * 0.30, 0.012, "surveyed", fontsize=BODY, color=C["dark"], path_effects=STROKE)
    # Symmetric log: the separation spans three orders of magnitude and a linear
    # axis pressed four fifths of the feeds into the baseline, which is exactly
    # the half of the distribution that carries the negative result.
    ax.set_yscale("symlog", linthresh=0.01, linscale=0.4)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.01, 0.05, 0.1, 0.3, 1.0])
    ax.set_yticklabels(["0", "0.01", "0.05", "0.1", "0.3", "1"])
    derived = int((frame["coincident_share"] >= 0.30).sum())
    ax.set_xticks([])
    ax.set_xlim(-1, len(frame))
    ax.set_ylabel("share of feed vertices on an OSM vertex", fontsize=LABEL)
    ax.set_title("(a)", fontsize=PANEL, loc="left", color=C["dark"])

    ax = axes[1]
    style_ax(ax)
    named = frame.head(12)
    y = np.arange(len(named))
    ax.barh(
        y,
        named["coincident_share"],
        color=C["red"],
        alpha=0.85,
        height=0.68,
        edgecolor="white",
        linewidth=HAIRLINE,
        zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([key.replace("_", " ").title() for key in named["city_key"]], fontsize=BODY)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("coincidence with OSM geometry", fontsize=LABEL)
    ax.set_title("(b)", fontsize=PANEL, loc="left", color=C["dark"])
    save(fig, "wide_provenance.png")


# ---------------------------------------------------------------------------
# Figures 10 and 11. Accessibility: the spread, and what predicts it
# ---------------------------------------------------------------------------
def accessibility_table() -> pd.DataFrame:
    access = load("accessibility.csv", ["city_key"])
    access = access.loc[access["status"].eq("ok")]
    coverage = city_coverage()
    table = access.merge(
        coverage[["city_key", "feed_to_osm_100", "osm_to_feed_100", "n_feed_stops"]], on="city_key", how="inner"
    )
    table["relative_completeness"] = table["feed_to_osm_100"] - table["osm_to_feed_100"]

    # The one predictor that needs no schedule: how many of a city's route
    # relations list their stops at all. Cities without a census drop out of that
    # panel only, which is why it is merged last and left as NaN.
    census = load("route_stops.csv", ["city_key"])
    if census is not None:
        census = census.assign(relations_with_stops=census["n_with_stops"] / census["n_relations"].replace(0, np.nan))
        table = table.merge(census[["city_key", "relations_with_stops"]], on="city_key", how="left")
    return table


def figure_accessibility_spread() -> None:
    table = accessibility_table()
    fig, axes = plt.subplots(2, 1, figsize=canvas(0.86), dpi=DPI)
    fig.subplots_adjust(left=0.17, right=0.83, top=0.955, bottom=0.075, hspace=0.36)
    fig.set_layout_engine("none")

    ax = axes[0]
    style_ax(ax, ygrid_only=False)
    # A cumulative curve answers the question the section asks -- how many cities
    # are badly wrong -- which a strip of jittered dots only hints at. Three
    # curves also make the point that the thresholds differ in spread and not in
    # centre, which three near-identical dot columns hid.
    styles = {15: (C["light"], ":"), 30: (C["iduedu"], "-"), 45: (C["pt"], "--")}
    for threshold, (colour, dash) in styles.items():
        values = table[f"ratio_{threshold}"].dropna().sort_values()
        share = np.arange(1, len(values) + 1) / len(values)
        ax.step(
            values,
            share,
            where="post",
            color=colour,
            linestyle=dash,
            linewidth=HEAVY,
            label=f"{threshold} min  (median {values.median():.2f})",
            zorder=3,
        )
    ax.axvline(1.0, color=C["red"], linestyle="--", linewidth=RULE, zorder=4)
    ax.text(1.03, 0.06, "agreement", fontsize=BODY, color=C["red"], path_effects=STROKE, zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(0.15, 3.2)
    ax.set_ylim(0, 1)
    # Matplotlib's default log labelling puts "2 x 10^-1" next to "3 x 10^-1"
    # and the two collide at this width; plain decimals at chosen stops read.
    ax.set_xticks([0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(NullFormatter())
    ax.set_xlabel("OSM estimate ÷ schedule reference", fontsize=LABEL)
    ax.set_ylabel("share of cities\nat or below", fontsize=LABEL)
    ax.legend(frameon=False, fontsize=BODY, loc="upper left")
    ax.set_title("(a)", fontsize=PANEL, loc="left", color=C["dark"])

    ax = axes[1]
    style_ax(ax)
    ordered = table.sort_values("ratio_30")
    y = np.arange(len(ordered))

    colours = [C["red"] if value < 0.8 else C["green"] if value > 1.2 else C["light"] for value in ordered["ratio_30"]]
    ax.barh(y, ordered["ratio_30"] - 1.0, left=1.0, color=colours, height=1.0, zorder=3)
    ax.axvline(1.0, color=C["dark"], linewidth=RULE, zorder=5)

    # Labels go out into the margin beside the panel, on the side its bar points
    # to. Placed at the bar's own x they sat on top of the bars, and the leader
    # lines ran diagonally across the chart and over each other; from the margin
    # every line is horizontal and none crosses another. Within a margin the
    # labels are spread apart from their true position, nearest first, so a line
    # is only as long as the crowding requires.
    # The cities the text names, and only those: the figure and the paragraph
    # beside it should point at the same places. They are also the extremes, so
    # every bar tip sits near a margin and no leader line has to cross the panel
    # -- a city near the centre line costs a rule the full width of the chart.
    named = []
    for key in ("gqeberha", "erzurum", "bogor", "london", "addis_ababa", "mississauga", "tallinn"):
        found = np.flatnonzero(ordered["city_key"].to_numpy() == key)
        if len(found):
            named.append((int(found[0]), float(ordered["ratio_30"].iloc[int(found[0])]), key))

    # Rows are thin next to the type, so a label needs a few of them to itself.
    span = len(ordered)
    gap = max(span * 0.075, 6.0)
    for side in (-1, 1):
        # Named cities are the extremes, so they crowd against one end of the
        # ranking. Labels are therefore pushed away from that end -- downwards
        # for the top group, upwards for the bottom one -- and the city closest
        # to the end is placed first, so it keeps its true height and the rest
        # give way. Pushing everything one way instead ran the top group off the
        # axis, where all three labels landed on the same line.
        upper = side > 0
        column = sorted((item for item in named if (item[1] >= 1.0) == upper), key=lambda item: item[0], reverse=upper)
        step = -gap if upper else gap
        taken: list[float] = []
        for position, value, key in column:
            slot = float(position)
            while any(abs(slot - other) < gap for other in taken):
                slot += step
            taken.append(slot)
            ax.annotate(
                key.replace("_", " ").title(),
                xy=(value, position),
                xycoords="data",
                xytext=(1.012 if upper else -0.012, slot),
                textcoords=("axes fraction", "data"),
                fontsize=BODY,
                va="center",
                ha="left" if side > 0 else "right",
                color=C["dark"],
                zorder=6,
                annotation_clip=False,
                arrowprops={"arrowstyle": "-", "linewidth": HAIRLINE, "color": C["dark"], "shrinkA": 1, "shrinkB": 1},
            )
    ax.set_yticks([])
    ax.set_xlabel("OSM estimate ÷ schedule reference, 30 minutes", fontsize=LABEL)
    ax.set_title("(b)", fontsize=PANEL, loc="left", color=C["dark"])
    save(fig, "wide_accessibility_spread.png")


def figure_predictor() -> None:
    table = accessibility_table()
    panels = [
        ("relative_completeness", "relative completeness", "needs the schedule"),
        ("feed_to_osm_100", "OSM completeness alone", "needs the schedule"),
        ("relations_with_stops", "relations listing stops", "needs nothing else"),
    ]
    panels = [panel for panel in panels if panel[0] in table.columns]
    fig, axes = plt.subplots(1, len(panels), figsize=canvas(0.82), dpi=DPI, sharey=True, layout="constrained")

    for letter, (ax, (column, label, provenance)) in zip("abc", zip(np.atleast_1d(axes), panels)):
        style_ax(ax, ygrid_only=False)
        subset = table[[column, "ratio_30"]].dropna()
        rho, p = spearmanr(subset[column], subset["ratio_30"])
        free = provenance == "needs nothing else"
        ax.scatter(
            subset[column],
            subset["ratio_30"],
            s=26,
            color=C["green"] if free else C["iduedu"],
            alpha=0.55,
            edgecolor="white",
            linewidth=HAIRLINE,
            zorder=3,
        )

        # Median of the outcome within quintiles of the predictor. A rank-based
        # summary belongs beside a rank-based statistic, and a straight least
        # squares line through a log-scaled cloud would claim more than Spearman
        # does.
        bands = pd.qcut(subset[column], 5, duplicates="drop")
        trend = subset.groupby(bands, observed=True).agg(x=(column, "median"), y=("ratio_30", "median"))
        ax.plot(
            trend["x"],
            trend["y"],
            color=C["dark"],
            linewidth=HEAVY,
            marker="o",
            markersize=pt(3.0),
            markerfacecolor="white",
            markeredgewidth=RULE,
            zorder=6,
        )

        ax.axhline(1.0, color=C["red"], linestyle="--", linewidth=RULE, zorder=4)
        ax.set_yscale("log")
        ax.set_yticks([0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
        ax.get_yaxis().set_major_formatter(ScalarFormatter())
        ax.get_yaxis().set_minor_formatter(NullFormatter())
        ax.set_xlabel(label, fontsize=LABEL)
        significance = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        ax.set_title(f"({letter}) {provenance}", fontsize=PANEL, loc="left", color=C["dark"])

        ax.text(
            0.04,
            0.04,
            f"ρ = {rho:+.2f}\n{significance}\nn = {len(subset)}",
            transform=ax.transAxes,
            fontsize=BODY,
            color=C["dark"],
            va="bottom",
            ha="left",
            linespacing=1.25,
            path_effects=STROKE,
            zorder=7,
        )

    np.atleast_1d(axes)[0].set_ylabel("OSM estimate ÷\nschedule reference, 30 min", fontsize=LABEL)
    save(fig, "wide_predictor.png")


# ---------------------------------------------------------------------------
# Figure 3. Where the sample is
# ---------------------------------------------------------------------------
#: Countries at 1:110m from Natural Earth (public domain), copied into the
#: repository. Returns ``None`` rather than failing: a missing basemap should cost
#: the map its background, not the whole figure run.
def _basemap():
    path = Path(__file__).resolve().parent / "data" / "naturalearth_lowres" / "naturalearth_lowres.shp"
    if not path.exists():
        print("[map] no basemap at figures/data/naturalearth_lowres; drawing without land")
        return None
    try:
        import geopandas as gpd

        return gpd.read_file(path)
    except Exception as error:  # noqa: BLE001 - the points matter more than the land
        print(f"[map] basemap unreadable ({type(error).__name__}); drawing without land")
        return None


def figure_world_map() -> None:
    profile = load("feed_profile.csv", ["feed_id"])
    members = load("city_members.csv")
    cities = load("cities.csv", ["city_key"])
    table = accessibility_table()

    centres = members.merge(profile[["feed_id", "centre_lat", "centre_lon"]], on="feed_id", how="left")
    centres = centres.groupby("city_key")[["centre_lat", "centre_lon"]].mean().reset_index()
    # The cohort, not every city the resolver named: the map has to show the same
    # sample the tables count.
    cohort = set(cohort_cities())
    frame = (
        cities.loc[cities["city_key"].map(str).isin(cohort), ["city_key"]]
        .merge(centres, on="city_key", how="left")
        .merge(table[["city_key", "ratio_30", "feed_to_osm_100"]], on="city_key", how="left")
        .dropna(subset=["centre_lat", "centre_lon"])
    )

    fig, ax = plt.subplots(figsize=canvas(0.56), dpi=DPI)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Land behind the points, or the figure reads as a scatter plot rather than a
    # map. Natural Earth 1:110m, public domain, kept in the repository so the
    # figure does not depend on a plotting library shipping its own basemap.
    land = _basemap()
    if land is not None:
        land.plot(ax=ax, color="#d7dde5", edgecolor="white", linewidth=HAIRLINE, zorder=1)

    for longitude in range(-180, 181, 30):
        ax.axvline(longitude, color="black", alpha=0.05, linewidth=HAIRLINE, zorder=2)
    for latitude in range(-60, 91, 30):
        ax.axhline(latitude, color="black", alpha=0.05, linewidth=HAIRLINE, zorder=2)

    measured = frame.dropna(subset=["ratio_30"])
    unmeasured = frame.loc[frame["ratio_30"].isna()]

    # Cities the study could not reach: a catalogue entry exists, the archive
    # behind it does not, and OpenStreetMap maps their transit anyway. Drawing
    # them puts the boundary of the sample on the same map as its contents, which
    # is the honest answer to "why is this city missing".
    unreachable = load("unpublished_cities.csv", ["feed_id"])
    if unreachable is not None:
        selection = load("selection.csv", ["feed_id"])
        unreachable = (
            unreachable.loc[unreachable["status"].eq("ok") & unreachable["n_routes_osm"].fillna(0).gt(0)]
            .merge(selection[["feed_id", "lat", "lon"]], on="feed_id", how="left")
            .dropna(subset=["lat", "lon"])
        )
    # Hollow, and in a colour the diverging scale never takes: filled grey sat exactly
    # where "equal" sits on coolwarm, so a city with no comparison looked like a city
    # whose two estimates agreed.
    ax.scatter(
        unmeasured["centre_lon"],
        unmeasured["centre_lat"],
        s=22,
        facecolor="none",
        edgecolor="#111111",
        linewidth=RULE,
        marker="s",
        zorder=6,
        label=f"no comparison ({len(unmeasured)})",
    )
    if unreachable is not None and not unreachable.empty:
        # An "x" draws with its edge colour, so facecolor="none" left the legend
        # entry blank and the marks nearly invisible on the land tint.
        ax.scatter(
            unreachable["lon"],
            unreachable["lat"],
            s=34,
            c="#1a1a1a",
            linewidths=RULE,
            marker="x",
            zorder=7,
            label=f"no schedule ({len(unreachable)})",
        )
    scatter = ax.scatter(
        measured["centre_lon"],
        measured["centre_lat"],
        # coolwarm puts a pale grey at the centre of the scale, which is also the
        # colour of land: a city whose two estimates agreed disappeared into the
        # continent it stood on. RdYlBu_r keeps blue low and red high but turns
        # the middle to yellow, which no part of the map uses; it is also the
        # colour-blind-safe choice of the two.
        c=np.log2(measured["ratio_30"].clip(0.2, 3.0)),
        s=26,
        cmap="RdYlBu_r",
        vmin=-1.4,
        vmax=1.4,
        alpha=0.95,
        edgecolor="#3a3a3a",
        linewidth=HAIRLINE,
        zorder=4,
    )
    bar = fig.colorbar(scatter, ax=ax, orientation="horizontal", fraction=0.055, pad=0.09, aspect=45)
    bar.set_ticks([-1, -0.5, 0, 0.5, 1])
    bar.set_ticklabels(["half", "0.7×", "same", "1.4×", "double"])
    bar.ax.tick_params(labelsize=BODY)
    # "OSM estimate / schedule, 30 min" told a reader nothing about what was
    # divided by what, or what the number means.
    bar.set_label("places reachable in 30 min: OSM network vs schedule network", fontsize=LABEL)
    bar.outline.set_visible(False)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-58, 78)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-45, 76, 30))
    ax.tick_params(colors=C["dark"], labelsize=BODY)
    ax.legend(frameon=False, fontsize=BODY, loc="lower left", labelcolor=C["dark"], handletextpad=0.5, borderpad=0.2)
    save(fig, "wide_world_map.png")


# --------------------------------------------------------------------------
# Data and methods: the design, the sample, the spread between cities.
# --------------------------------------------------------------------------


def figure_overview() -> None:
    """The introduction's figure: drawn in TikZ, not measured.

    Kept as LaTeX rather than matplotlib because it has to sit beside the
    library paper's workflow diagram and share its palette and its shapes; a
    reader who meets both should see one hand. The source is
    ``overview_figure.tex``, and this only drives the two conversions the
    manuscript needs: a PNG for the local build and vector EPS for the journal.
    """
    source = HERE / "overview_figure.tex"
    if not source.exists():
        print("[skip] overview_figure.tex not found")
        return

    binaries = Path.home() / "AppData/Local/Programs/MiKTeX/miktex/bin/x64"
    pdflatex = binaries / "pdflatex.exe"
    pdftops = binaries / "pdftops.exe"
    if not pdflatex.exists():
        print("[skip] no LaTeX to draw the overview figure")
        return

    subprocess.run([str(pdflatex), "-interaction=nonstopmode", source.name], cwd=HERE, check=True, capture_output=True)
    pdf = HERE / "overview_figure.pdf"

    import pymupdf

    # 400 dpi: the figure runs the full width of the page and its smallest type
    # is 6 pt, which at 300 dpi starts to break up.
    pymupdf.open(pdf)[0].get_pixmap(dpi=400).save(HERE / "wide_overview.png")
    print("saved wide_overview.png")

    if VECTOR and pdftops.exists():
        subprocess.run(
            [str(pdftops), "-eps", str(pdf), str(HERE / "wide_overview.eps")], check=True, capture_output=True
        )
        print("saved wide_overview.eps")


def figure_two_sources(city_key: str = "delhi", title: str = "Delhi") -> None:
    """One city, both sources, side by side and to the same scale.

    Section 3 described the two networks for five pages without showing either.
    Delhi is the ordinary case rather than a pathology: the feed covers the city
    densely, OpenStreetMap holds a part of it, and the part is not a random
    sample of the whole -- which is the asymmetry the rest of the paper measures.
    """
    walk = read_urban_graph(osm_graph_path(city_key, "walk"))
    osm = read_urban_graph(osm_graph_path(city_key, "pt", ALL_OSM_MODES))
    gtfs = read_urban_graph(gtfs_graph_path(city_key))
    crs = walk.nodes_gdf.crs

    fig, axes = plt.subplots(1, 2, figsize=canvas(0.55), dpi=DPI)
    minx, miny, maxx, maxy = walk.edges_gdf.total_bounds

    # Stops, not graph vertices: the graph keeps a vertex per boarding, and
    # len(nodes_gdf) gave Delhi 110,091 against 4,993 stops in the coverage table.
    counts = city_coverage()
    counts = counts.loc[counts["city_key"].astype(str).eq(city_key)]
    feed_stops = int(counts["n_feed_stops"].iloc[0]) if len(counts) else len(gtfs.nodes_gdf)
    osm_stops = int(counts["n_osm_stops"].iloc[0]) if len(counts) else len(osm.nodes_gdf)

    panels = [
        (gtfs, MAP["gtfs"], "published timetable", feed_stops),
        (osm, MAP["osm"], "OpenStreetMap", osm_stops),
    ]
    for letter, (ax, (graph, colour, label, stops)) in zip("ab", zip(axes, panels)):
        style_map(ax)
        # Background walk network in the thinnest line the journal accepts: 0.3 pt printed.
        walk.edges_gdf.plot(ax=ax, color="#e2e2e6", linewidth=pt(0.3), zorder=1)
        draw_transit(
            ax, graph.to_crs(crs) if hasattr(graph, "to_crs") else graph, colour, linewidth=pt(0.45), alpha=0.8
        )
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_title(f"({letter}) {label}\n{stops:,} stops", fontsize=PANEL, loc="left", color=colour, linespacing=1.3)

    scale_bar(axes[0], 10000)
    save(fig, "wide_two_sources.png", dpi=RASTER_DPI)


# --------------------------------------------------------------------------
# City maps: the cases worth looking at one by one.
# --------------------------------------------------------------------------

#: One city. Displacement is rare -- eight of 126 cities once the second criterion
#: is in the classification -- and only Redland City has what a map needs: both
#: networks substantial, clouds that do not overlap, a boundary that reads as one
#: city. Pune (nested clouds) and Truckee (112 km inside one convex hull) were
#: tried and rejected on the map.
DISPLACED = [("redland_city", "Redland City")]


#: City boundaries from their OSM relations, kept beside the figures so a rerun
#: does not depend on Overpass; without the file the map is drawn without one.
BOUNDARY_CACHE = Path(__file__).resolve().parent / "data" / "city_boundaries"


def city_boundary(city_key: str):
    """City boundary in EPSG:4326, or None if it could not be fetched."""
    import geopandas as gpd

    BOUNDARY_CACHE.mkdir(parents=True, exist_ok=True)
    cached = BOUNDARY_CACHE / f"{city_key}.geojson"
    if cached.exists():
        return gpd.read_file(cached)

    cities = pd.read_csv(Path(__file__).resolve().parents[1] / "results" / "wide_tier" / "cities.csv")
    row = cities.loc[cities["city_key"].astype(str).eq(city_key)]
    if row.empty or pd.isna(row["osm_id"].iloc[0]):
        return None
    try:
        from iduedu import get_4326_boundary

        polygon = get_4326_boundary(osm_id=int(row["osm_id"].iloc[0]))
    except Exception as error:  # noqa: BLE001 - the stops matter more than the outline
        print(f"[map] no boundary for {city_key} ({type(error).__name__}); drawing without it")
        return None
    frame = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
    frame.to_file(cached, driver="GeoJSON")
    return frame


def scale_bar(ax, length_m: float = 5000.0, height: float = 0.05) -> None:
    """Scale bar: without it a reader cannot tell a city from a district.

    ``height`` is the fraction of the axes height the bar sits at; the Redland City
    panel has its legend at the bottom.
    """
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    x = x0 + (x1 - x0) * 0.04
    y = y0 + (y1 - y0) * height
    ax.plot([x, x + length_m], [y, y], color=MAP["dark"], linewidth=HEAVY, solid_capstyle="butt", zorder=8)
    ax.text(
        x + length_m / 2,
        y + (y1 - y0) * 0.012,
        f"{length_m / 1000:.0f} km",
        ha="center",
        va="bottom",
        fontsize=BODY,
        color=MAP["dark"],
        path_effects=STROKE,
        zorder=8,
    )


def locator_inset(ax, lon: float, lat: float) -> None:
    """World map inset showing where the city is."""
    land_path = Path(__file__).resolve().parent / "data" / "naturalearth_lowres" / "naturalearth_lowres.shp"
    if not land_path.exists():
        return
    try:
        import geopandas as gpd

        land = gpd.read_file(land_path)
    except Exception:  # noqa: BLE001
        return
    inset = ax.inset_axes([0.72, 0.02, 0.26, 0.26])
    land.plot(ax=inset, color="#e4e8ee", edgecolor="white", linewidth=pt(0.3))
    inset.plot([lon], [lat], marker="o", markersize=pt(3.0), color=MAP["osm"], zorder=5)
    inset.set_xlim(-180, 180)
    inset.set_ylim(-58, 80)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_facecolor("white")
    for spine in inset.spines.values():
        spine.set_color(MAP["gray"])
        spine.set_linewidth(HAIRLINE)


def separation_km(city_key: str) -> float | None:
    """Distance between the cloud centroids from the classification table, in km."""
    path = Path(__file__).resolve().parents[1] / "results" / "wide_tier" / "displacement.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path).drop_duplicates("city_key", keep="last")
    row = table.loc[table["city_key"].astype(str).eq(city_key)]
    if row.empty or pd.isna(row["separation_m"].iloc[0]):
        return None
    return float(row["separation_m"].iloc[0]) / 1000.0


def style_map(ax) -> None:
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_facecolor("white")


def draw_transit(ax, graph, colour: str, linewidth: float = 0.9, alpha: float = 0.75) -> None:
    edges = graph.edges_gdf
    if edges.empty or "type" not in edges:
        return
    travel = edges.loc[~edges["type"].isin(["boarding", "alighting", "station_link", "walk"])]
    if travel.empty:
        return
    travel.plot(ax=ax, color=colour, linewidth=linewidth, alpha=alpha, zorder=3)


def figure_displacement() -> None:
    # Panel widths follow each city's own aspect ratio; equal panels would leave a
    # tall city floating in white space next to a wide one.
    loaded = []
    for city_key, title in DISPLACED:
        walk = read_urban_graph(osm_graph_path(city_key, "walk"))
        minx, miny, maxx, maxy = walk.edges_gdf.total_bounds
        loaded.append((city_key, title, walk, (maxx - minx) / max(maxy - miny, 1e-9)))
    ratios = [max(aspect, 0.35) for *_, aspect in loaded]

    # One row per city instead of side by side: two cities of different shape in one
    # row forced both into a strip a few centimetres tall, and the point of the
    # figure -- that the two sets of dots occupy different ground -- was invisible.
    fig, axes = plt.subplots(len(loaded), 1, figsize=canvas(0.66 * len(loaded)), dpi=DPI)
    for ax, (city_key, title, walk, _) in zip(np.atleast_1d(axes), loaded):
        style_map(ax)
        osm_pt = read_urban_graph(osm_graph_path(city_key, "pt", ALL_OSM_MODES))
        gtfs_pt = read_urban_graph(gtfs_graph_path(city_key))

        # Boundary beneath everything: without it the claim that both clouds lie in
        # one city has to be taken on trust. Streets are darker so the built-up area reads.
        outline = city_boundary(city_key)
        if outline is not None:
            outline.to_crs(walk.nodes_gdf.crs).plot(
                ax=ax, facecolor="#f2f4f7", edgecolor=MAP["gray"], linewidth=RULE, linestyle="--", alpha=0.9, zorder=0
            )
        walk.edges_gdf.plot(ax=ax, color="#c2c2c7", linewidth=pt(0.3), alpha=0.95, zorder=1)
        gtfs_nodes = gtfs_pt.nodes_gdf.to_crs(walk.nodes_gdf.crs)
        gtfs_nodes.plot(
            ax=ax, color=MAP["gtfs"], markersize=12, alpha=0.85, edgecolor="white", linewidth=pt(0.3), zorder=4
        )
        osm_pt.nodes_gdf.plot(
            ax=ax, color=MAP["osm"], markersize=12, alpha=0.85, edgecolor="white", linewidth=pt(0.3), zorder=5
        )

        # How far apart the two clouds actually are, stated on the panel rather than
        # left for the reader to estimate from dots. The number is read from the
        # classification table rather than recomputed here: the classification runs
        # on bus stops and this panel draws every mode, so recomputing gave 25 km
        # against the 24 the text quotes -- a figure and a sentence disagreeing
        # about the same city.
        gap_km = separation_km(city_key)
        if gap_km is None:
            gap_km = (
                gtfs_nodes.geometry.union_all().centroid.distance(osm_pt.nodes_gdf.geometry.union_all().centroid)
                / 1000.0
            )
        # The distance is a measurement and stays on the panel; the city name goes in the caption.
        ax.text(
            0.015,
            0.975,
            f"{gap_km:.0f} km between what the schedule serves and what OSM maps",
            transform=ax.transAxes,
            fontsize=BODY,
            color=MAP["dark"],
            va="top",
            ha="left",
            path_effects=STROKE,
            zorder=8,
        )
        scale_bar(ax, height=0.80)
        centre = walk.nodes_gdf.to_crs(4326).geometry.union_all().centroid
        locator_inset(ax, centre.x, centre.y)

    fig.legend(
        handles=[
            Line2D(
                [], [], marker="o", linestyle="none", color=MAP["osm"], markersize=pt(4), label="stops mapped in OSM"
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color=MAP["gtfs"],
                markersize=pt(4),
                label="stops in the GTFS feed",
            ),
            Line2D([], [], color="#c2c2c7", linewidth=HEAVY, label="pedestrian network"),
            Line2D([], [], color=MAP["gray"], linewidth=RULE, linestyle="--", label="city boundary"),
        ],
        frameon=False,
        fontsize=BODY,
        labelcolor=MAP["dark"],
        loc="lower left",
        bbox_to_anchor=(0.01, -0.01),
        ncol=2,
        handletextpad=0.5,
        columnspacing=1.4,
    )
    save(fig, "wide_case_displacement_maps.png", dpi=RASTER_DPI)


# --------------------------------------------------------------------------
# Packaging: numbered EPS and TIFF the way the journal wants them.
# --------------------------------------------------------------------------


def figure_order() -> list[str]:
    """Figure file names in the order Results shows them."""
    source = "".join(
        (MANUSCRIPT / name).read_text(encoding="utf-8")
        for name in ("paper_en_head.py", "paper_en_body.py", "paper_en_tail.py")
        if (MANUSCRIPT / name).exists()
    )
    seen, ordered = set(), []
    for name in re.findall(r'page\.figure\(\s*"([^"]+\.png)"', source):
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def regenerate() -> None:
    """Redraw the chart figures as vector, in a process that knows it from the start.

    ``figure_common`` decides PNG against EPS when it is imported, so the flag has
    to be in the environment before that happens -- hence a subprocess rather than
    a function call. Only the charts are redrawn: the city plates go out as TIFF,
    and asking them for vector spends two minutes producing 106 MB to delete.
    """
    environment = {**os.environ, "PAPER_VECTOR": "1"}
    print(f"redrawing charts as vector: {Path(__file__).name} --charts")
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--charts"], cwd=HERE, env=environment, check=True)


def collect(names: list[str]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    # Clear the whole directory: stale Fig11 and Fig12 from a longer figure list
    # would otherwise go into the submission.
    for stale in OUTPUT_DIR.glob("Fig*"):
        stale.unlink()
    for index, name in enumerate(names, start=1):
        if name in RASTER:
            source = HERE / name
            if not source.exists():
                print(f"Fig{index}: {name} missing")
                continue
            destination = OUTPUT_DIR / f"Fig{index}.tif"
            with Image.open(source) as image:
                # Resolution at printed size: pixels over the text width.
                printed = round(image.width / COLUMN_IN)
                image.convert("RGB").save(destination, format="TIFF", compression="tiff_lzw", dpi=(printed, printed))
                print(f"Fig{index}: {printed} dpi at final size")
        else:
            source = HERE / (Path(name).stem + ".eps")
            if not source.exists():
                print(f"Fig{index}: no vector for {name}")
                continue
            destination = OUTPUT_DIR / f"Fig{index}.eps"
            shutil.copy2(source, destination)
            # A PDF goes beside the EPS for the journal's LaTeX build; the TikZ figure
            # has its own from pdflatex.
            beside = HERE / (Path(name).stem + ".pdf")
            if "overview" in name and (HERE / "overview_figure.pdf").exists():
                beside = HERE / "overview_figure.pdf"
            if beside.exists():
                shutil.copy2(beside, OUTPUT_DIR / f"Fig{index}.pdf")
        print(f"{destination.name}  <-  {name}  ({destination.stat().st_size / 1e6:.1f} MB)")

    # The intermediate vector files sit beside the PNGs the manuscript reads; leaving
    # them there means the next figure run has to skip past a hundred megabytes
    # of stale art.
    for pattern in ("wide_*.eps", "wide_*.pdf"):
        for leftover in HERE.glob(pattern):
            leftover.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--charts", action="store_true", help="only the charts, skipping the city plate")
    parser.add_argument("--journal", action="store_true", help="collect numbered EPS and TIFF for submission")
    parser.add_argument("--run", action="store_true", help="with --journal: actually write the files")
    arguments = parser.parse_args()

    if arguments.journal:
        names = figure_order()
        if not arguments.run:
            for index, name in enumerate(names, start=1):
                print(f"Fig{index}.eps  <-  {name}")
            print(f"\n{len(names)} figures, in the order Results shows them.")
            print(f"Would regenerate them as vector and collect into {OUTPUT_DIR}.")
            print("Nothing was written; re-run with --run.")
            return
        regenerate()
        collect(names)
        print(f"\nwritten to {OUTPUT_DIR}")
        print("Captions stay in the manuscript: the journal wants them out of the image files.")
        return

    figure_overview()
    figure_two_sources()
    figure_coverage_scatter()
    figure_coverage_by_mode()
    figure_coverage_cases()
    figure_speed_by_length()
    figure_provenance()
    figure_accessibility_spread()
    figure_predictor()
    figure_world_map()
    if not arguments.charts:
        figure_displacement()


if __name__ == "__main__":
    main()
