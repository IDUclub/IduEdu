"""Every figure of the paper, drawn from the result tables.

Images go to ``paper2026/figures/`` under the names the manuscript reads (``wide_*.png``).
With ``PAPER_VECTOR=1`` every chart is also written as EPS and PDF.

    python pipeline/figures.py            # every figure
    python pipeline/figures.py --charts   # the charts only, not the two city plates
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import unicodedata
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import INPUTS_DIR, PAPER_DIR, RESULTS_DIR, boundary, gtfs_graph_path, measurable, osm_graph_path
from coverage import CASE_ORDER, COVERED, ELSEWHERE, NO_NETWORK, SPARSE, classify_bus
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FormatStrFormatter, NullFormatter, ScalarFormatter

from iduedu import read_urban_graph

# --- page geometry and style -----------------------------------------------------

HERE = PAPER_DIR / "figures"
BASEMAP = HERE / "data" / "naturalearth_lowres" / "naturalearth_lowres.shp"
DPI, RASTER_DPI = 300, 560
COLUMN_IN = 372 / 72.27  # the journal's text width
OVERSIZE = 1.35
FIG_W = COLUMN_IN * OVERSIZE
VECTOR = os.environ.get("PAPER_VECTOR") == "1"


def pt(printed: float) -> float:
    """Canvas size that prints as ``printed`` points."""
    return printed * OVERSIZE


def canvas(height_ratio: float) -> tuple[float, float]:
    return FIG_W, FIG_W * height_ratio


BODY, LABEL, PANEL = pt(8.5), pt(9.0), pt(9.5)
HAIRLINE, RULE, HEAVY = pt(0.5), pt(0.9), pt(1.6)
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": BODY,
    "axes.labelsize": LABEL, "axes.titlesize": PANEL, "xtick.labelsize": BODY, "ytick.labelsize": BODY,
    "legend.fontsize": BODY, "axes.linewidth": HAIRLINE, "xtick.major.width": HAIRLINE, "ytick.major.width": HAIRLINE,
    "lines.linewidth": RULE, "patch.linewidth": HAIRLINE, "ps.fonttype": 42, "pdf.fonttype": 42,
})  # fmt: skip

C = {"iduedu": "#00A8FF", "walk": "#FF9F0A", "pt": "#AF52DE", "green": "#32D74B", "red": "#FF375F",
     "gray": "#8E8E93", "light": "#B8B8B8", "dark": "#3A3A3C"}  # fmt: skip
PT_COLORS = {"bus": "#AF52DE", "tram": "#FF375F", "trolleybus": "#32D74B", "subway": "#007AFF", "train": "#5E5CE6",
             "ferry": "#00A8FF"}  # fmt: skip
CASE_COLORS = {COVERED: "#32D74B", SPARSE: "#FF9F0A", ELSEWHERE: "#5E5CE6", NO_NETWORK: "#FF375F"}
MODE_ORDER = ["bus", "tram", "trolleybus", "subway", "train"]
#: The data call the mode "subway"; the paper calls it "metro".
MODE_LABELS = {"subway": "metro"}


def mode_label(mode: str) -> str:
    return MODE_LABELS.get(mode, mode)


STROKE = [pe.Stroke(linewidth=pt(1.5), foreground="white"), pe.Normal()]
MAP = {"osm": "#AF52DE", "gtfs": "#00A8FF", "walk": "#D8D8DC", "dark": "#3A3A3C", "gray": "#8E8E93"}
MODE_MAP_COLORS = {"bus": "#0072B2", "tram": "#009E73", "trolleybus": "#D55E00", "subway": "#CC79A7",
                   "train": "#E69F00", "taxi": "#56B4E9", "ferry": "#7570B3", "monorail": "#8E8E93"}  # fmt: skip
NON_TRAVEL = ("boarding", "alighting", "walk", "station_link")


def style_ax(ax, ygrid_only: bool = True) -> None:
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    grid = {"color": "black", "alpha": 0.10, "linewidth": HAIRLINE, "zorder": 1}
    ax.yaxis.grid(True, **grid) if ygrid_only else ax.grid(True, which="both", **grid)
    ax.tick_params(axis="both", colors="black", labelsize=BODY)
    ax.tick_params(which="minor", length=0)


def save(fig, name: str, dpi: int = DPI) -> None:
    """PNG for the manuscript build; with ``PAPER_VECTOR=1`` also EPS and PDF from the figure itself."""
    fig.patch.set_facecolor("white")
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    fig.savefig(HERE / name, dpi=dpi, bbox_inches="tight", facecolor="white")
    if VECTOR:
        for suffix in ("eps", "pdf"):
            fig.savefig(HERE / f"{Path(name).stem}.{suffix}", format=suffix, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {name}")


# --- data ------------------------------------------------------------------------

#: The drawing's own tables live with the inputs.
COHORT_TABLES = {"selection.csv", "downloads.csv", "feed_profile.csv"}


def load(name: str, subset: list[str] | None = None, cohort: bool = True) -> pd.DataFrame | None:
    """A result table restricted to the study cities (``cohort``), one row per ``subset`` key."""
    if name in COHORT_TABLES:
        path = INPUTS_DIR / "cohort" / name
    elif name in {"cities.csv", "city_members.csv"}:
        path = INPUTS_DIR / name
    else:
        path = RESULTS_DIR / name
    if not path.exists():
        print(f"[skip] {name} not found")
        return None
    frame = pd.read_csv(path)
    if cohort and "city_key" in frame.columns:
        frame = frame.loc[frame["city_key"].map(str).isin(set(measurable()))]
    return frame.drop_duplicates(subset=subset, keep="last") if subset else frame


def per_mode_coverage() -> pd.DataFrame:
    """Coverage rows of the modes OSM can be asked for."""
    frame = load("indicators_coverage.csv", ["city_key", "mode"])
    mask = (
        ~frame["mode"].eq("*")
        & frame["status"].eq("ok")
        & frame["fetchable_from_osm"].astype(str).str.lower().eq("true")
    )
    return frame.loc[mask].copy()


def latin_part(text: str) -> str:
    """The Latin head of a name: OSM gives Tetouan as Latin, Arabic and Tifinagh at once,
    and the manuscript's font has glyphs for the first of the three only."""
    head = []
    for character in text:
        if ord(character) < 0x250 or character in "'-":
            head.append(character)
        else:
            break
    return " ".join("".join(head).split()).strip(" -/,;")


#: The names the text uses where it names a city by its administrative unit and not by its
#: settlement (Sections 4.6 and 4.7); Lower Hutt City is the unit of the feed placed at Wellington.
TEXT_NAMES = {
    "greater_manchester": "Greater Manchester",
    "aachen_district": "Aachen district",
    "mumbai_suburban_district": "Mumbai suburban district",
    "lower_hutt_city": "Lower Hutt",
}


def name(city_key: str) -> str:
    """Short name for a label: the settlement, unless the text names the city by its unit."""
    from common import cities

    if city_key in TEXT_NAMES:
        return TEXT_NAMES[city_key]
    short = str(cities().at[city_key, "settlement"]) if city_key in cities().index else ""
    # the text spells every name without diacritics (Zilina, Tetouan), and the labels follow it
    short = unicodedata.normalize("NFKD", latin_part(short)).encode("ascii", "ignore").decode()
    return {"Sankt-Peterburg": "St Petersburg"}.get(short, short) or city_key.replace("_", " ").title()


def city_coverage() -> pd.DataFrame:
    frame = load("indicators_coverage.csv", ["city_key", "mode"])
    return frame.loc[frame["mode"].eq("*") & frame["status"].eq("ok")].copy()


# --- figures -----------------------------------------------------------------------


def figure_coverage_scatter() -> None:
    frame = city_coverage().dropna(subset=["feed_to_osm_100", "osm_to_feed_100"])
    fig, ax = plt.subplots(figsize=canvas(0.92), dpi=DPI)
    style_ax(ax, ygrid_only=False)

    ax.axhline(0.5, color="black", alpha=0.18, linewidth=RULE, zorder=2)
    ax.axvline(0.5, color="black", alpha=0.18, linewidth=RULE, zorder=2)
    # on the diagonal both sources hold the same share of the other's stops
    ax.plot([0, 1], [0, 1], color="black", alpha=0.18, linewidth=RULE, linestyle="--", zorder=2)
    # the quadrant labels stand clear of the clusters of points and of the labelled cities
    quadrants = [
        (0.27, 0.615, "the timetable holds more stops", "center"),
        (0.80, 0.64, "both hold most stops", "center"),
        (0.31, 0.15, "both hold few", "center"),
        (0.74, 0.26, "OpenStreetMap holds more stops", "center"),
    ]
    for x, y, label, ha in quadrants:
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            fontsize=BODY,
            color=C["dark"],
            ha=ha,
            va="center",
            style="italic",
            zorder=6,
            linespacing=1.3,
            path_effects=STROKE,
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
    # the cities the text discusses, in every quadrant the points reach
    labelled = {
        "bogota_ciudad": (9, -4),
        "nairobi_county": (9, -4),
        "cairo": (9, -4),
        "ljubljana": (9, -4),
        "tokyo": (-9, 4),
        "sankt_peterburg": (-9, 4),
        "moscow": (9, 4),
        "delhi": (9, -4),
        "montevideo": (9, 3),
        "hyderabad": (9, -4),
        "zilina": (9, -4),
        "cochabamba": (-9, -12),
    }
    placed: list[tuple[float, float]] = []
    for key, (dx, dy) in labelled.items():
        row = frame.loc[frame["city_key"].eq(key)]
        if row.empty:
            continue
        row = row.iloc[0]
        point = (row["feed_to_osm_100"], row["osm_to_feed_100"])
        if any(abs(point[0] - x) < 0.04 and abs(point[1] - y) < 0.04 for x, y in placed):
            continue
        placed.append(point)
        ax.annotate(
            name(key),
            point,
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=BODY,
            ha="left" if dx > 0 else "right",
            color=C["dark"],
            path_effects=STROKE,
            zorder=5,
        )

    ax.set_xlabel("share of scheduled stops found in OpenStreetMap", fontsize=LABEL)
    ax.set_ylabel("share of mapped stops found in the timetable", fontsize=LABEL)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    save(fig, "wide_coverage_scatter.png")


def figure_coverage_by_mode() -> None:
    frame = per_mode_coverage()
    frame = frame.loc[frame["n_feed_stops"].gt(0) & frame["n_osm_stops"].gt(0)]
    modes = [mode for mode in MODE_ORDER if frame["mode"].eq(mode).sum() >= 3]

    fig, axes = plt.subplots(2, 1, figsize=canvas(0.68), dpi=DPI, sharex=True, sharey=True)
    for ax, (direction, denominator, title) in zip(
        axes,
        [
            ("feed_to_osm", "n_feed_stops", "(a) scheduled stops recorded in OpenStreetMap"),
            ("osm_to_feed", "n_osm_stops", "(b) mapped stops recorded in the timetable"),
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
        ax.set_xticklabels([f"{mode_label(mode)}\n{count} cities" for mode, count, _ in centres], fontsize=BODY)
        ax.set_title(title, fontsize=PANEL, loc="left", color=C["dark"])

    # below the panels: above them the legend sat on the value labels of the last bars
    fig.legend(
        handles=[
            Patch(facecolor=C["gray"], alpha=alpha, label=f"{radius} m")
            for radius, alpha in ((50, 0.40), (100, 0.75), (150, 1.0))
        ],
        title="matching radius",
        frameon=False,
        fontsize=BODY,
        title_fontsize=BODY,
        loc="upper center",
        ncols=3,
        handlelength=1.1,
        handletextpad=0.4,
        columnspacing=1.1,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.supylabel("median share across cities", fontsize=LABEL, x=0.015)
    for ax in axes:
        ax.set_ylim(0, 1.16)
    save(fig, "wide_coverage_by_mode.png")


def figure_coverage_cases() -> None:
    bus = classify_bus(per_mode_coverage())
    order = CASE_ORDER

    fig = plt.figure(figsize=canvas(0.92), dpi=DPI, layout="constrained")
    # the legend has a row of its own under (a) and (b), which it names
    grid = fig.add_gridspec(3, 2, height_ratios=[1.25, 0.14, 1.0])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[2, :])]
    legend_ax = fig.add_subplot(grid[1, :])
    legend_ax.axis("off")

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
        ELSEWHERE: "different areas",
        NO_NETWORK: "no network",
    }
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([short[case] for case in order], fontsize=BODY, rotation=30, ha="right")
    ax.set_ylabel("cities with a bus feed", fontsize=LABEL)
    ax.set_title("(a) cities in each case", fontsize=PANEL, loc="left", color=C["dark"])

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
    ax.set_ylabel("distance, m (log scale)", fontsize=LABEL)
    ax.set_title("(b) distance to the nearest mapped stop", fontsize=PANEL, loc="left", color=C["dark"])

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
    ax.set_yticklabels([name(key) for key in displaced["city_key"]], fontsize=BODY)
    ax.invert_yaxis()
    ax.set_xlabel("median distance from a scheduled stop to the nearest mapped stop, km", fontsize=LABEL)
    ax.set_title(
        "(c) the cities where the sources describe different areas", fontsize=PANEL, loc="left", color=C["dark"]
    )
    legend_ax.legend(
        handles=[Patch(facecolor=CASE_COLORS[case], edgecolor="white", label=case) for case in order],
        loc="center",
        ncol=2,
        frameon=False,
        fontsize=BODY,
        labelcolor=C["dark"],
        handlelength=1.1,
        handletextpad=0.5,
        labelspacing=0.3,
    )
    save(fig, "wide_coverage_cases.png")


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
    # the legend counts every segment of the mode, as Table 8 does; the line leaves out thin classes
    segments = frame.groupby("mode")["count"].sum()
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
            label=f"{mode_label(mode)} ({int(segments[mode]):,} segments)",
            zorder=4,
            markeredgecolor="white",
            markeredgewidth=HAIRLINE,
        )
    ax.set_xscale("log")
    ax.set_xlabel("segment length, m (log scale)", fontsize=LABEL)
    ax.set_ylabel("scheduled speed, km/h", fontsize=LABEL)
    ax.legend(frameon=False, fontsize=BODY, loc="upper left")
    save(fig, "wide_speed_by_length.png")


def figure_provenance() -> None:
    frame = load("feed_provenance.csv", ["city_key"])
    if frame is None:
        return
    frame = frame.loc[frame["status"].eq("ok")].sort_values("coincident_share", ascending=False)
    colours = [C["red"] if value >= 0.30 else C["iduedu"] for value in frame["coincident_share"]]

    fig, axes = plt.subplots(1, 2, figsize=canvas(0.62), dpi=DPI, gridspec_kw={"width_ratios": [1.3, 1.0]})

    ax = axes[0]
    style_ax(ax)
    # no white edge: at 238 bars an edge would cover the bars themselves
    ax.bar(
        range(len(frame)),
        frame["coincident_share"],
        color=colours,
        width=1.0,
        edgecolor="none",
        linewidth=0.0,
        zorder=3,
    )
    ax.axhline(0.30, color=C["dark"], linestyle="--", linewidth=RULE, zorder=5)
    ax.axhline(0.05, color=C["dark"], linestyle=":", linewidth=RULE, zorder=5)
    ax.text(len(frame) * 0.30, 0.36, "copied from the map", fontsize=BODY, color=C["dark"], path_effects=STROKE)
    ax.text(len(frame) * 0.30, 0.012, "surveyed", fontsize=BODY, color=C["dark"], path_effects=STROKE)
    ax.set_yscale("symlog", linthresh=0.01, linscale=0.4)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.01, 0.05, 0.1, 0.3, 1.0])
    ax.set_yticklabels(["0", "0.01", "0.05", "0.1", "0.3", "1"])
    ax.set_xticks([])
    ax.set_xlim(-1, len(frame))
    ax.set_ylabel("share of timetable shape vertices\ncoincident with an OpenStreetMap vertex", fontsize=LABEL)
    ax.set_title("(a) every comparable feed, ranked", fontsize=PANEL, loc="left", color=C["dark"])

    ax = axes[1]
    style_ax(ax)
    named = frame.head(12)
    y = np.arange(len(named))
    ax.barh(
        y,
        named["coincident_share"],
        # the same colours as panel (a): the twelfth share lies below the threshold
        color=[C["red"] if value >= 0.30 else C["iduedu"] for value in named["coincident_share"]],
        alpha=0.85,
        height=0.68,
        edgecolor="white",
        linewidth=HAIRLINE,
        zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([name(key) for key in named["city_key"]], fontsize=BODY)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("share of vertices coincident with OpenStreetMap", fontsize=LABEL)
    ax.set_title("(b) the twelve highest shares", fontsize=PANEL, loc="left", color=C["dark"])
    save(fig, "wide_provenance.png")


def accessibility_table() -> pd.DataFrame:
    access = load("accessibility.csv", ["city_key"])
    return access.loc[access["status"].eq("ok")]


#: Marked on Figure 11: the lowest and highest cities the text of Section 4.7 names, each with
#: where its label goes (ratio, share of cities). The low four sit within a tenth of each other
#: on the ratio, so their labels stand in a column above them, the one furthest left on top,
#: and no leader crosses another or a label; the high three take the empty corner under the curves.
NAMED_IN_TEXT = {
    "cochabamba": (0.24, 0.95),
    "nelson_mandela_bay_metropolitan_municipality": (0.265, 0.88),
    "santiago": (0.29, 0.81),
    "mumbai_suburban_district": (0.315, 0.74),
    "kyiv": (2.0, 0.86),
    "jelgava": (1.62, 0.78),
    "zilina": (1.62, 0.70),
}


def figure_accessibility_spread() -> None:
    table = accessibility_table()
    fig, ax = plt.subplots(figsize=canvas(0.52), dpi=DPI)
    style_ax(ax, ygrid_only=False)
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
            label=f"{threshold} min, median {values.median():.2f}",
            zorder=3,
        )
    ax.axvline(1.0, color=C["red"], linestyle="--", linewidth=RULE, zorder=4)
    ax.text(1.03, 0.45, "agreement", fontsize=BODY, color=C["red"], path_effects=STROKE, zorder=5)

    # A city sits on the 30-minute curve where the curve reaches its share of cities.
    ordered = table[["city_key", "ratio_30"]].dropna().sort_values("ratio_30")
    share = np.arange(1, len(ordered) + 1) / len(ordered)
    for key, label_at in NAMED_IN_TEXT.items():
        found = np.flatnonzero(ordered["city_key"].to_numpy() == key)
        if not len(found):
            continue
        point = (float(ordered["ratio_30"].iloc[found[0]]), float(share[found[0]]))
        ax.scatter(*point, s=14, color=C["iduedu"], edgecolor="white", linewidth=HAIRLINE, zorder=7, clip_on=False)
        ax.annotate(
            name(key),
            xy=point,
            xytext=label_at,
            fontsize=BODY,
            va="center",
            ha="left",
            color=C["dark"],
            zorder=8,
            annotation_clip=False,
            arrowprops={"arrowstyle": "-", "linewidth": HAIRLINE, "color": C["dark"], "shrinkA": 2, "shrinkB": 2,
                        "relpos": (0.0, 0.5)},  # fmt: skip
        )

    ax.set_xscale("log")
    ax.set_xlim(0.1, 3.2)
    ax.set_ylim(0, 1)
    ax.set_xticks([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(NullFormatter())
    ax.set_xlabel("places reached: OpenStreetMap network ÷ timetable's network", fontsize=LABEL)
    ax.set_ylabel("share of cities at or below", fontsize=LABEL)
    ax.legend(frameon=False, fontsize=BODY, loc="lower right", handlelength=1.4, handletextpad=0.5, borderaxespad=0.1)
    save(fig, "wide_accessibility_spread.png")


def _basemap():
    path = BASEMAP
    if not path.exists():
        print("[map] no basemap at figures/data/naturalearth_lowres; drawing without land")
        return None
    try:
        import geopandas as gpd

        return gpd.read_file(path)
    except Exception as error:  # noqa: BLE001 - the points matter more than the land
        print(f"[map] basemap unreadable ({type(error).__name__}); drawing without land")
        return None


WORLD_CRS = "ESRI:54030"


def _project(longitudes, latitudes):
    """Degrees to the world map projection."""
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", WORLD_CRS, always_xy=True)
    return transformer.transform(np.asarray(longitudes, dtype=float), np.asarray(latitudes, dtype=float))


def _world_axes(ratio: float = 0.56):
    """A world map with land behind the points and nothing else in the way."""
    fig, ax = plt.subplots(figsize=canvas(ratio), dpi=DPI)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)

    land = _basemap()
    if land is not None:
        land.to_crs(WORLD_CRS).plot(ax=ax, color="#e4e7ec", edgecolor="white", linewidth=pt(0.15), zorder=1)

    # The graticule stays faint and unlabelled: the Europe inset sits where the latitude
    # labels were, and the maps say nothing that depends on latitude.
    for longitude in range(-180, 181, 60):
        x, y = _project([longitude] * 61, np.linspace(-58, 84, 61))
        ax.plot(x, y, color="black", alpha=0.07, linewidth=HAIRLINE, zorder=2)
    for latitude in range(-45, 76, 30):
        x, y = _project(np.linspace(-180, 180, 181), [latitude] * 181)
        ax.plot(x, y, color="black", alpha=0.07, linewidth=HAIRLINE, zorder=2)

    edges_x, _ = _project([-180, 180], [0, 0])
    _, edges_y = _project([0, 0], [-58, 84])
    span = edges_x[1] - edges_x[0]
    ax.set_xlim(edges_x[0] - 0.01 * span, edges_x[1] + 0.01 * span)
    ax.set_ylim(edges_y[0], edges_y[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


#: Europe holds 150 of the 265 cities in a box the world map draws under two centimetres
#: wide, so both world maps repeat it enlarged.
EUROPE = (-10.5, 34.5, 40.5, 62.0)  # west, south, east, north, degrees
#: Where the enlargement goes, in fractions of the world axes: the South Pacific, which holds
#: no city of the study; its right edge stays west of Lima and Santiago.
EUROPE_INSET = (0.0, 0.0, 0.29, 0.5)


def _europe_inset(ax, draw) -> None:
    """Draw ``draw(inset)`` again for Europe in the corner of ``ax`` and outline the area on the world map."""
    west, south, east, north = EUROPE
    edge = np.linspace(0.0, 1.0, 40)
    lons = np.r_[west + (east - west) * edge, np.full(40, east), east - (east - west) * edge, np.full(40, west)]
    lats = np.r_[np.full(40, south), south + (north - south) * edge, np.full(40, north), north - (north - south) * edge]
    box_x, box_y = _project(lons, lats)

    inset = ax.inset_axes(EUROPE_INSET)
    inset.set_facecolor("white")
    draw(inset)
    inset.set_xlim(box_x.min(), box_x.max())
    inset.set_ylim(box_y.min(), box_y.max())
    inset.set_aspect("equal", adjustable="datalim")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color(C["dark"])
        spine.set_linewidth(HAIRLINE)
    inset.text(0.03, 0.97, "Europe", transform=inset.transAxes, ha="left", va="top", fontsize=BODY,
               color=C["dark"], path_effects=STROKE, zorder=9)  # fmt: skip

    ax.figure.canvas.draw()  # the aspect settles the limits only when drawn
    (x0, x1), (y0, y1) = inset.get_xlim(), inset.get_ylim()
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=C["dark"], linewidth=HAIRLINE, zorder=6))


def _city_centres() -> pd.DataFrame:
    """Where each city sits: the anchor the selection ranked it at.

    The mean of the centres of a city's feeds put a city served by a national feed
    hundreds of kilometres from itself -- Edmonton by 1,092 km -- because a city's
    feeds are every feed of its country with a stop inside the boundary. A point of
    the boundary will not do either: Tokyo's largest part is the sea around its
    islands. The anchor lies inside the city's own boundary for all 265 cities.
    """
    cities = load("cities.csv", ["city_key"], cohort=False)
    candidates = pd.read_csv(INPUTS_DIR / "cohort" / "pick5_candidates.csv")
    kept = candidates.loc[candidates["decision"].eq("keep") & candidates["boundary_osm"].notna()].copy()
    kept["osm_id"] = kept["boundary_osm"].str.split("/").str[1].astype("int64")
    anchors = cities[["city_key", "osm_id"]].merge(
        kept[["osm_id", "anchor_lat", "anchor_lon"]], on="osm_id", how="left"
    )
    return anchors.rename(columns={"anchor_lat": "centre_lat", "anchor_lon": "centre_lon"})[
        ["city_key", "centre_lat", "centre_lon"]
    ]


#: Countries shaded by how many cities of the study they hold (at most five each).
SAMPLE_CLASSES = [((1, 1), "one city", "#9ECAE1"), ((2, 4), "two to four", "#4292C6"), ((5, 5), "five", "#08519C")]
CITY_DOT = "#E8112D"
#: Natural Earth's country names where they differ from the World Bank's in ``cities.csv``.
NATURAL_EARTH_NAMES = {
    "BA": "Bosnia and Herz.", "CD": "Dem. Rep. Congo", "CI": "Côte d'Ivoire", "DO": "Dominican Rep.", "EG": "Egypt",
    "KR": "South Korea", "MK": "Macedonia", "PR": "Puerto Rico", "RU": "Russia", "SK": "Slovakia", "TR": "Turkey",
    "US": "United States of America", "VE": "Venezuela",
}  # fmt: skip


def figure_sample_map() -> None:
    """The countries and cities of the study, and nothing else.

    A country is shaded by how many cities it contributes; every city is a point,
    which also places the city states the basemap is too coarse to draw. What was
    lost on the way is counted in Table 2, not drawn.
    """
    cities = load("cities.csv", ["city_key"], cohort=False)
    per_country = cities.groupby("country_code").agg(country=("country_name", "first"), cities=("city_key", "size"))
    located = (
        cities[["city_key"]]
        .merge(_city_centres(), on="city_key", how="left")
        .dropna(subset=["centre_lat", "centre_lon"])
    )

    fig, ax = _world_axes(0.62)
    land = _basemap()
    shaded = None
    if land is not None:
        names = [NATURAL_EARTH_NAMES.get(code, row["country"]) for code, row in per_country.iterrows()]
        counts = dict(zip(names, per_country["cities"]))
        shaded = land.assign(cities=land["name"].map(counts)).dropna(subset=["cities"]).to_crs(WORLD_CRS)
    px, py = _project(located["centre_lon"], located["centre_lat"])

    def draw(target, dot: float) -> None:
        # Borders a quarter of a hairline: at a full one the small countries of Europe
        # were more border than colour. The cities are red, the one colour the shading lacks.
        if land is not None and target is not ax:
            land.to_crs(WORLD_CRS).plot(ax=target, color="#e4e7ec", edgecolor="white", linewidth=pt(0.15), zorder=1)
        if shaded is not None:
            for (low, high), _, colour in SAMPLE_CLASSES:
                part = shaded.loc[shaded["cities"].between(low, high)]
                part.plot(ax=target, color=colour, edgecolor="white", linewidth=pt(0.15), zorder=3)
        target.scatter(px, py, s=dot, c=CITY_DOT, linewidth=0, alpha=0.9, zorder=5)

    draw(ax, 1.2)
    _europe_inset(ax, lambda inset: draw(inset, 3.0))

    handles = [
        Patch(
            facecolor=colour,
            edgecolor="none",
            label=f"{label} ({int(per_country['cities'].between(low, high).sum())} countries)",
        )
        for (low, high), label, colour in SAMPLE_CLASSES
    ]
    handles += [
        Patch(facecolor="#e4e7ec", edgecolor="none", label="not in the study"),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=pt(1.6),
            markerfacecolor=CITY_DOT,
            markeredgewidth=0,
            label=f"a city under investigation ({len(located)})",
        ),
    ]
    ax.legend(
        handles=handles,
        title="cities under investigation in the country",
        title_fontsize=pt(7.5),
        frameon=False,
        fontsize=pt(7.5),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        labelcolor=C["dark"],
        handletextpad=0.5,
        columnspacing=1.4,
        ncol=3,
    )
    save(fig, "wide_sample_map.png")


#: Purple for less reach, green for more, and a mid grey for "the same": PRGn's white
#: midpoint would vanish against the land and the page. Grey, not a hue,
#: so that agreement does not read as a third category.
DISCREPANCY_CMAP = LinearSegmentedColormap.from_list(
    "discrepancy", ["#40004B", "#9970AB", "#9A9AA2", "#5AAE61", "#00441B"]
)
DISCREPANCY_SPAN = 1.4  # log2: from 0.38x to 2.6x


def figure_discrepancy_map() -> None:
    """The substitution error city by city, and nothing else.

    Only the cities where both estimates exist are drawn; the sample has its own map.
    """
    centres = _city_centres()
    table = accessibility_table()
    frame = (
        table.loc[table["ratio_30"].notna(), ["city_key", "ratio_30"]]
        .merge(centres, on="city_key", how="left")
        .dropna(subset=["centre_lat", "centre_lon"])
    )

    fig, ax = _world_axes(0.60)
    px, py = _project(frame["centre_lon"], frame["centre_lat"])
    colour = np.log2(frame["ratio_30"].clip(0.2, 3.0))

    def draw(target, dot: float):
        # no outline, so that neighbouring cities do not merge into one dark blot
        return target.scatter(px, py, c=colour, s=dot, cmap=DISCREPANCY_CMAP, vmin=-DISCREPANCY_SPAN,
                              vmax=DISCREPANCY_SPAN, alpha=0.95, linewidth=0, zorder=5)  # fmt: skip

    scatter = draw(ax, 9)

    def draw_inset(inset) -> None:
        land = _basemap()
        if land is not None:
            land.to_crs(WORLD_CRS).plot(ax=inset, color="#e4e7ec", edgecolor="white", linewidth=pt(0.15), zorder=1)
        draw(inset, 16)

    _europe_inset(ax, draw_inset)

    bar = fig.colorbar(scatter, ax=ax, orientation="horizontal", fraction=0.045, pad=0.07, aspect=48)
    bar.set_ticks([-1, -0.5, 0, 0.5, 1])
    bar.set_ticklabels(["0.5", "0.7", "1.0", "1.4", "2.0"])
    bar.ax.tick_params(labelsize=BODY)
    bar.set_label(
        "places reachable within 30 minutes: OpenStreetMap network ÷ timetable's network",
        fontsize=LABEL,
    )
    bar.outline.set_visible(False)
    save(fig, "wide_discrepancy_map.png")


def figure_overview() -> None:
    """The protocol diagram of the introduction, drawn in TikZ (``overview_figure.tex``).

    pdflatex compiles it, pymupdf renders the PDF to the PNG the manuscript uses, and with
    ``PAPER_VECTOR=1`` pdftops also writes it as EPS.
    """
    source = HERE / "overview_figure.tex"
    pdflatex, pdftops = shutil.which("pdflatex"), shutil.which("pdftops")
    if not source.exists() or pdflatex is None:
        print("[skip] wide_overview.png: needs overview_figure.tex and pdflatex")
        return
    subprocess.run([pdflatex, "-interaction=nonstopmode", source.name], cwd=HERE, check=True, capture_output=True)
    for leftover in (".aux", ".log"):
        source.with_suffix(leftover).unlink(missing_ok=True)
    pdf = source.with_suffix(".pdf")
    if VECTOR and pdftops:
        subprocess.run([pdftops, "-eps", str(pdf), str(HERE / "wide_overview.eps")], check=True, capture_output=True)
        print("saved wide_overview.eps")
    try:
        import pymupdf
    except ModuleNotFoundError:
        print("[skip] wide_overview.png: pymupdf is not installed (the PDF is built)")
        return
    pymupdf.open(pdf)[0].get_pixmap(dpi=400).save(HERE / "wide_overview.png")
    print("saved wide_overview.png")


def figure_two_sources(city_key: str = "zilina", title: str = "Žilina") -> None:
    """One city, both sources, side by side and to the same scale.

    Žilina: the feeds that cover it come from regional bus companies and the railway,
    while OpenStreetMap also maps the city's own buses and trolleybuses and holds five
    times as many stops.
    """
    walk = read_urban_graph(osm_graph_path(city_key, "walk"))
    osm = read_urban_graph(osm_graph_path(city_key, "pt"))
    gtfs = read_urban_graph(gtfs_graph_path(city_key))
    crs = walk.nodes_gdf.crs

    fig, axes = plt.subplots(1, 2, figsize=canvas(0.58), dpi=DPI, layout="constrained")
    minx, miny, maxx, maxy = walk.edges_gdf.total_bounds

    counts = city_coverage()
    counts = counts.loc[counts["city_key"].astype(str).eq(city_key)]
    feed_stops = int(counts["n_feed_stops"].iloc[0]) if len(counts) else len(gtfs.nodes_gdf)
    osm_stops = int(counts["n_osm_stops"].iloc[0]) if len(counts) else len(osm.nodes_gdf)

    panels = [
        (gtfs, "published timetable", feed_stops),
        (osm, "OpenStreetMap", osm_stops),
    ]
    drawn: list[str] = []
    for letter, (ax, (graph, label, stops)) in zip("ab", zip(axes, panels)):
        style_map(ax)
        walk.edges_gdf.plot(ax=ax, color="#e2e2e6", linewidth=pt(0.3), zorder=1)
        modes = draw_transit_by_mode(ax, graph.to_crs(crs) if hasattr(graph, "to_crs") else graph, linewidth=pt(0.45))
        drawn += [mode for mode in modes if mode not in drawn]
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_title(
            f"({letter}) {label}\n{stops:,} stops", fontsize=PANEL, loc="left", color=C["dark"], linespacing=1.3
        )

    naming = {"subway": "metro", "taxi": "shared taxi"}
    fig.legend(
        handles=[
            Line2D([], [], color=MODE_MAP_COLORS.get(mode, C["gray"]), linewidth=pt(1.6), label=naming.get(mode, mode))
            for mode in drawn
        ],
        loc="outside lower center",
        ncol=min(len(drawn), 5),
        frameon=False,
        fontsize=BODY,
        labelcolor=C["dark"],
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=1.4,
    )
    # the longest round length that takes no more than a quarter of the map's width
    width = maxx - minx
    scale_bar(axes[0], max([length for length in (1000, 2000, 5000, 10000, 20000) if length <= width / 4] or [1000]))
    save(fig, "wide_two_sources.png", dpi=RASTER_DPI)


def city_boundary(city_key: str):
    """The city's boundary as a GeoDataFrame in EPSG:4326."""
    import geopandas as gpd

    return gpd.GeoDataFrame(geometry=[boundary(city_key)], crs="EPSG:4326")


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
    land_path = BASEMAP
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
    path = RESULTS_DIR / "displacement.csv"
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


def _travel_edges(graph):
    """Travel edges: everything but boarding, alighting and station links."""
    edges = graph.edges_gdf
    if edges.empty or "type" not in edges:
        return None
    kind = edges["type"].astype(str)
    structural = kind.isin(NON_TRAVEL) | kind.str.contains("entrance|exit|station|transfer", regex=True)
    travel = edges.loc[~structural]
    return None if travel.empty else travel


def draw_transit_by_mode(ax, graph, linewidth: float = 0.9, alpha: float = 0.85) -> list[str]:
    """Travel edges coloured by mode, most frequent first so rare rail stays visible; returns the modes drawn."""
    travel = _travel_edges(graph)
    if travel is None:
        return []
    order = travel["type"].astype(str).value_counts().index.tolist()
    for mode in order:
        subset = travel.loc[travel["type"].astype(str).eq(mode)]
        subset.plot(ax=ax, color=MODE_MAP_COLORS.get(mode, C["gray"]), linewidth=linewidth, alpha=alpha, zorder=3)
    return order


def displaced_city() -> list[tuple[str, str]]:
    """The "different areas" case with the most stops on both sides."""
    bus = classify_bus(per_mode_coverage())
    elsewhere = bus[bus["case"] == ELSEWHERE].assign(smaller=lambda b: b[["n_feed_stops", "n_osm_stops"]].min(axis=1))
    if elsewhere.empty:
        return []
    key = str(elsewhere.nlargest(1, "smaller")["city_key"].iloc[0])
    return [(key, name(key))]


def figure_displacement() -> None:
    loaded = []
    for city_key, title in displaced_city():
        walk = read_urban_graph(osm_graph_path(city_key, "walk"))
        minx, miny, maxx, maxy = walk.edges_gdf.total_bounds
        loaded.append((city_key, title, walk, (maxx - minx) / max(maxy - miny, 1e-9)))

    fig, axes = plt.subplots(len(loaded), 1, figsize=canvas(0.66 * len(loaded)), dpi=DPI)
    for ax, (city_key, title, walk, _) in zip(np.atleast_1d(axes), loaded):
        style_map(ax)
        osm_pt = read_urban_graph(osm_graph_path(city_key, "pt"))
        gtfs_pt = read_urban_graph(gtfs_graph_path(city_key))

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

        gap_km = separation_km(city_key)
        if gap_km is None:
            gap_km = (
                gtfs_nodes.geometry.union_all().centroid.distance(osm_pt.nodes_gdf.geometry.union_all().centroid)
                / 1000.0
            )
        ax.text(
            0.015,
            0.975,
            f"the centres of the two stop clouds lie {gap_km:.1f} km apart",
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
                [], [], marker="o", linestyle="none", color=MAP["osm"], markersize=pt(4), label="stops in OpenStreetMap"
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color=MAP["gtfs"],
                markersize=pt(4),
                label="stops in the feed",
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
        columnspacing=0.8,
    )
    save(fig, "wide_case_displacement_maps.png", dpi=RASTER_DPI)


def figure_waits() -> None:
    """Each city's own wait per mode, and how far the registry constant misses it."""
    frame = load("waits_by_city_mode.csv")
    frame = frame[frame["mode"].isin(MODE_ORDER) & frame["wait_min"].notna()]
    fig, axes = plt.subplots(1, 2, figsize=canvas(0.52), dpi=DPI, gridspec_kw={"width_ratios": [1.25, 1.0]})
    modes = [m for m in ["subway", "tram", "trolleybus", "bus", "train"] if (frame["mode"] == m).sum() >= 5]
    for ax, column, label, title in (
        (axes[0], "wait_min", "mean wait per departure, min (log scale)", "(a) each city's wait"),
        (
            axes[1],
            "constant_over_own",
            "the model's constant ÷ the city's own wait (log scale)",
            "(b) the constant against each city",
        ),
    ):
        style_ax(ax)
        data = [frame.loc[frame["mode"] == m, column].dropna() for m in modes]
        parts = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False)
        for patch, mode in zip(parts["boxes"], modes):
            patch.set_facecolor(PT_COLORS.get(mode, C["gray"]))
            patch.set_alpha(0.75)
            patch.set_edgecolor("white")
        for element in ("whiskers", "caps", "medians"):
            for line in parts[element]:
                line.set_color(C["dark"])
                line.set_linewidth(RULE)
        ax.set_yscale("log")
        ax.get_yaxis().set_major_formatter(FormatStrFormatter("%g"))
        ax.get_yaxis().set_minor_formatter(NullFormatter())
        ax.set_xticks(range(1, len(modes) + 1))
        short = {"subway": "metro", "trolleybus": "trolley-\nbus"}
        ax.set_xticklabels([f"{short.get(m, m)}\n{len(d)}" for m, d in zip(modes, data)], fontsize=BODY)
        ax.set_ylabel(label, fontsize=LABEL)
        ax.set_title(title, fontsize=PANEL, loc="left", color=C["dark"])
    axes[0].set_yticks([1, 2, 5, 10, 20, 40])
    axes[1].set_yticks([0.25, 0.5, 1, 2, 4])
    axes[1].axhline(1.0, color=C["red"], linestyle="--", linewidth=RULE, zorder=4)
    save(fig, "wide_waits.png")


#: Three of the four estimates of each city, from its timetable to OpenStreetMap with the
#: registry; the first, the timetable itself, is the dashed line at 1. Named as Section 3.10
#: and Table 10 name them: each estimate differs from the one before it in the words that change.
CASES = [("D1", "timetable, one wait per mode", C["gray"]),
         ("A_w", "OpenStreetMap network, one wait per mode", C["pt"]),
         ("A", "OpenStreetMap network, median waits per mode", C["iduedu"])]  # fmt: skip


def figure_decomposition() -> None:
    """Each city's other three estimates as a share of what its timetable reaches (the dashed line).

    Boxes: quartiles over cities; whiskers: 10th and 90th percentiles; the number: the median.
    """
    frame = load("accessibility_variants.csv", ["city_key"])
    frame = frame[frame["status"] == "ok"]
    thresholds = (15, 30, 45)
    fig, axes = plt.subplots(1, len(thresholds), figsize=canvas(0.5), dpi=DPI, sharey=True)
    for ax, t in zip(axes, thresholds):
        style_ax(ax)
        data = [
            (frame[f"{name}_{t}"] / frame[f"D_{t}"]).replace([np.inf, -np.inf], np.nan).dropna() for name, _, _ in CASES
        ]
        parts = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False, whis=(10, 90))
        for patch, (_, _, colour) in zip(parts["boxes"], CASES):
            patch.set_facecolor(colour)
            patch.set_edgecolor("white")
        for element in ("whiskers", "caps", "medians"):
            for line in parts[element]:
                line.set_color(C["dark"])
                line.set_linewidth(RULE)
        for x, values in enumerate(data, start=1):
            ax.text(x, values.quantile(0.9) + 0.02, f"{values.median():.2f}", ha="center", va="bottom",
                    fontsize=BODY, color=C["dark"])  # fmt: skip
        ax.axhline(1.0, color=C["dark"], linestyle="--", linewidth=HAIRLINE, zorder=1)
        # numbered as Section 3.10 counts the estimates: the timetable is the first
        ax.set_xticks(range(1, len(CASES) + 1))
        ax.set_xticklabels([str(i) for i in range(2, len(CASES) + 2)], fontsize=BODY)
        ax.set_title(f"within {t} min", fontsize=PANEL, loc="left", color=C["dark"])
    axes[0].set_ylabel("places reached ÷ the timetable's", fontsize=LABEL)
    handles = [Line2D([0], [0], color=C["dark"], linestyle="--", linewidth=HAIRLINE, label="1  timetable")]
    handles += [
        Patch(facecolor=colour, edgecolor="white", label=f"{i}  {label}")
        for i, (_, label, colour) in enumerate(CASES, 2)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=BODY, bbox_to_anchor=(0.5, -0.13))
    save(fig, "wide_decomposition.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--charts", action="store_true", help="only the charts, not the city plates")
    arguments = parser.parse_args()
    charts = [figure_overview, figure_sample_map, figure_coverage_scatter, figure_coverage_by_mode,
              figure_coverage_cases, figure_speed_by_length, figure_provenance, figure_waits,
              figure_accessibility_spread, figure_discrepancy_map, figure_decomposition]  # fmt: skip
    plates = [figure_two_sources, figure_displacement]
    for draw in charts + ([] if arguments.charts else plates):
        draw()


if __name__ == "__main__":
    main()
