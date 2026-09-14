"""What every figure and the metrics table share: palette, loading, classification.

The classification lived in two copies -- one in the figure script, one in the
metrics script -- and the day the rule changed both had to be edited by hand in
step. A table and the figure of that same table are the one pair that must never
disagree, so the rule is declared once, here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from wide_cohort import measurable  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results" / "wide_tier"
FIGURES = Path(__file__).resolve().parent
DPI = 300

COLUMN_IN = 372 / 72.27

OVERSIZE = 1.35
FIG_W = COLUMN_IN * OVERSIZE


def pt(printed: float) -> float:
    """Canvas size that prints as ``printed`` points."""
    return printed * OVERSIZE


def canvas(height_ratio: float) -> tuple[float, float]:
    """Full-width canvas of the given aspect ratio.

    Printed height is the text width times the ratio, and that alone decides how
    much of the page the figure takes.
    """
    return FIG_W, FIG_W * height_ratio


BODY = pt(8.5)
LABEL = pt(9.0)
PANEL = pt(9.5)


HAIRLINE = pt(0.5)
RULE = pt(0.9)
HEAVY = pt(1.6)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": BODY,
        "axes.labelsize": LABEL,
        "axes.titlesize": PANEL,
        "xtick.labelsize": BODY,
        "ytick.labelsize": BODY,
        "legend.fontsize": BODY,
        "axes.linewidth": HAIRLINE,
        "xtick.major.width": HAIRLINE,
        "ytick.major.width": HAIRLINE,
        "lines.linewidth": RULE,
        "patch.linewidth": HAIRLINE,
        "ps.fonttype": 42,
        "pdf.fonttype": 42,
    }
)

#: The result tables record every city ever measured; the paper reports the study
#: cohort. Filtering here, once, is what keeps a figure from quoting a different
#: population than the table beside it -- which has now happened twice.
COHORT = set(measurable())

C = {
    "iduedu": "#00A8FF",
    "walk": "#FF9F0A",
    "pt": "#AF52DE",
    "green": "#32D74B",
    "red": "#FF375F",
    "gray": "#8E8E93",
    "light": "#B8B8B8",
    "dark": "#3A3A3C",
}
PT_COLORS = {
    "bus": "#AF52DE",
    "tram": "#FF375F",
    "trolleybus": "#32D74B",
    "subway": "#007AFF",
    "train": "#5E5CE6",
    "ferry": "#00A8FF",
}

#: Case names, in the order they are reported. Shared so the figure's bars and the
#: table's rows cannot drift apart.
COVERED = "covered"
SPARSE = "sparse in the same area"
ELSEWHERE = "OSM maps another area"
NO_NETWORK = "no bus network in OSM"
CASE_ORDER = [COVERED, SPARSE, ELSEWHERE, NO_NETWORK]
CASE_COLORS = {
    COVERED: "#32D74B",
    SPARSE: "#FF9F0A",
    ELSEWHERE: "#5E5CE6",
    NO_NETWORK: "#FF375F",
}

STROKE = [pe.Stroke(linewidth=pt(1.5), foreground="white"), pe.Normal()]

#: Modes the OSM builder can be asked for, so the only ones completeness applies to.
MODE_ORDER = ["bus", "tram", "trolleybus", "subway", "train"]

#: Waiting time needs no OSM counterpart, so ferries belong in that table alone.
WAIT_MODE_ORDER = MODE_ORDER + ["ferry"]

#: Below both thresholds OSM holds a handful of stops rather than a network, and
#: whether those few land near a feed stop or a kilometre away is an accident of
#: where someone happened to map. Testing for exactly zero, as an earlier version
#: did, sent Cairo -- two OSM stops against 2236 in the feed -- into the displaced
#: bucket and made displacement look four times as common as it is.
NO_NETWORK_STOPS = 10
NO_NETWORK_SHARE = 0.05

#: Beyond this the nearest counterpart is another part of town rather than a
#: missed match. Matches ``ELSEWHERE_M`` in ``benchmarks/wide_indicators.py``.
ELSEWHERE_SHARE = 0.5

#: A city counts as covered at this share of feed stops matched within 100 m.
COVERED_SHARE = 0.5


def load(name: str, subset: list[str] | None = None, cohort: bool = True) -> pd.DataFrame | None:
    path = RESULTS / name
    if not path.exists():
        print(f"[skip] {name} not found")
        return None
    frame = pd.read_csv(path)
    if cohort and "city_key" in frame.columns:
        frame = frame.loc[frame["city_key"].map(str).isin(COHORT)]
    if subset:
        frame = frame.drop_duplicates(subset=subset, keep="last")
    return frame


def per_mode_coverage() -> pd.DataFrame:
    """Coverage rows for real modes, excluding modes OSM was never asked for."""
    frame = load("indicators_coverage.csv", ["city_key", "mode"])
    mask = ~frame["mode"].eq("*") & frame["status"].eq("ok")
    mask &= frame["fetchable_from_osm"].astype(str).str.lower().eq("true")
    return frame.loc[mask].copy()


def city_coverage() -> pd.DataFrame:
    frame = load("indicators_coverage.csv", ["city_key", "mode"])
    return frame.loc[frame["mode"].eq("*") & frame["status"].eq("ok")].copy()


#: Distance between the centroids of the two stop clouds, divided by their own spread.
#: Above one the clouds sit apart; below one, one lies inside the other.
#: Computed by ``benchmarks/wide_displacement.py``.
DISPLACED_RATIO = 1.0


def _displacement() -> pd.DataFrame:
    """Cloud separation ratio per city, if it has been computed."""
    path = RESULTS / "displacement.csv"
    if not path.exists():
        return pd.DataFrame(columns=["city_key", "ratio"])
    frame = pd.read_csv(path).drop_duplicates("city_key", keep="last")
    return frame.loc[frame["status"].eq("ok"), ["city_key", "ratio"]]


def classify_bus(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a ``case`` column to the per-mode bus rows.

    The order of the tests is the argument: a source with no network cannot be
    said to map elsewhere, so substance is checked before position.

    Position needs two tests, not one. The share of feed stops further than a
    kilometre from any OSM stop rises both when the two networks sit apart and
    when OSM covers only the centre of a city whose feed covers the whole
    agglomeration -- Pune has 6,648 scheduled stops against 262 in OSM, a median
    8.7 km to the nearest counterpart, and an OSM cloud sitting squarely inside
    the feed's. So a city is only called displaced when its two clouds are also
    further apart than their own size.
    """
    bus = frame.loc[frame["mode"].eq("bus") & frame["n_feed_stops"].gt(0)].copy()
    ratios = dict(_displacement().itertuples(index=False, name=None))

    def case(row) -> str:
        if row["n_osm_stops"] < NO_NETWORK_STOPS and row["n_osm_stops"] < NO_NETWORK_SHARE * row["n_feed_stops"]:
            return NO_NETWORK
        if row["feed_to_osm_100"] >= COVERED_SHARE:
            return COVERED
        if row.get("share_elsewhere", 0) >= ELSEWHERE_SHARE:
            # Cities without a measured ratio stay where the old rule put them:
            # silently moving them to another group would be worse.
            ratio = ratios.get(row["city_key"])
            if ratio is None or ratio > DISPLACED_RATIO:
                return ELSEWHERE
        return SPARSE

    bus["case"] = bus.apply(case, axis=1)
    return bus


def style_ax(ax, ygrid_only: bool = True) -> None:
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    if ygrid_only:
        ax.yaxis.grid(True, color="black", alpha=0.10, linewidth=HAIRLINE, zorder=1)
    else:
        ax.grid(True, which="both", color="black", alpha=0.10, linewidth=HAIRLINE, zorder=1)
    ax.tick_params(axis="both", colors="black", labelsize=BODY)
    ax.tick_params(which="minor", length=0)


RASTER_DPI = 560


#: With ``PAPER_VECTOR=1`` every figure is also written as EPS, which is what the
#: journal asks for. It is off by default because vector files of a scatter with
#: a hundred thousand points are slow to write and slower to open, and nobody
#: reviewing a draft needs them.
VECTOR = os.environ.get("PAPER_VECTOR") == "1"


def save(fig, name: str, dpi: int = DPI) -> None:
    fig.patch.set_facecolor("white")
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=dpi, bbox_inches="tight", facecolor="white")
    if VECTOR:
        # Straight from the figure object, so the result is real vector art. A
        # first version re-wrapped the PNG into EPS and produced 535 MB of
        # bitmap in a vector container.
        #
        # Both formats, because they answer to different readers. The journal
        # asks for EPS in production; the journal's build server, handed the
        # LaTeX source, needs PDF -- pdflatex includes it without a converter.
        # Going EPS -> PDF through Ghostscript instead lost the font tables and
        # printed the charts with every label missing.
        stem = Path(name).stem
        for suffix in ("eps", "pdf"):
            fig.savefig(FIGURES / f"{stem}.{suffix}", format=suffix, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {name}" + (" and .eps/.pdf" if VECTOR else ""))
