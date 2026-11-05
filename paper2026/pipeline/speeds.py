"""Run times between stops: the world fit for the registry's speed constants.

The data: every timetable segment with its own geometry (``shapes.txt``; at most 4 000
per city and mode, faster-than-vmax removed). The model: the library's kinematic model
``t = dwell + K(L)/v`` (``K = 2 sqrt(L span)`` below the acceleration-plus-braking
span, ``L + span`` above it), with one free speed or a free speed growing with
length ``v(L) = v0 + (v_inf - v0) L/(L + L0)``. The error: ``|log(pred/obs)|``
weighted by the segment's scheduled minutes.

* ``fit`` -- the world: every city weighing the same, cross-validated on 10 folds of
  held-out cities. The registry takes the growing form where it lowers the held-out
  error (tram, train). -> ``speed_fit.csv``, ``speed_fit_curves.csv``,
  ``speed_by_length.csv``, ``speed_fit_segments.parquet``

    python pipeline/speeds.py fit
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from common import (
    GTFS_GRAPHS_CSV,
    RESULTS_DIR,
    TRANSIT_MODES,
    gtfs_graph_path,
    last_rows,
    measurable,
    setup,
)

from iduedu import read_urban_graph
from iduedu.constants.transport_specs import DEFAULT_REGISTRY_W_TRAIN, TransportSpec

MIN_SPEED_MPM = 60.0  # the builder's floor: 1 m/s
SEGMENTS = RESULTS_DIR / "speed_fit_segments.parquet"
PER_CITY, MIN_PER_CITY, FOLDS = 4000, 30, 10
CURVE_BINS = [0, 125, 250, 500, 950, 1350, 2500, 5000, 10000, np.inf]
GROWING = "kinematic, speed growing with length"
#: The classes of the speed-by-length figure.
FIGURE_BINS = [0, 125, 250, 500, 950, 1350, 2500, 5000, np.inf]


def span_and_vmax(spec: TransportSpec) -> tuple[float, float]:
    return max(spec.accel_dist_m, 0.0) + max(spec.brake_dist_m, 0.0), spec.vmax_tech_kmh * 1000 / 60


def k_term(length: np.ndarray, span: float) -> np.ndarray:
    return np.where(length < span, 2.0 * np.sqrt(length * span), length + span)


# --- fit: the world fit on every shaped segment -------------------------------------


def segments_of(city_key: str) -> pd.DataFrame:
    graph = read_urban_graph(gtfs_graph_path(city_key))
    edges, nodes = graph.edges_gdf, graph.nodes_gdf
    travel = edges[edges["type"].isin(TRANSIT_MODES)]
    if travel.empty:
        return pd.DataFrame()
    shape = nodes["shape_id"].astype(str) if "shape_id" in nodes else pd.Series("", index=nodes.index)
    frame = pd.DataFrame({
        "city_key": city_key, "mode": travel["type"].astype(str).to_numpy(),
        "length_m": pd.to_numeric(travel["length_meter"], errors="coerce").to_numpy(),
        "time_min": pd.to_numeric(travel["time_min"], errors="coerce").to_numpy(),
        "has_shape": travel["u"].map(shape).fillna("").ne("").to_numpy(),
    })  # fmt: skip
    frame = frame[frame["has_shape"] & (frame["length_m"] > 0) & (frame["time_min"] > 0)]
    return frame.sample(frac=1.0, random_state=0).groupby("mode").head(PER_CITY).drop(columns="has_shape")


def collect(workers: int) -> pd.DataFrame:
    graphs = last_rows(GTFS_GRAPHS_CSV, ["city_key"], keep_default_na=False)
    keys = [key for key in measurable() if key in set(graphs.loc[graphs["status"] == "ok", "city_key"])]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        table = pd.concat([part for part in pool.map(segments_of, keys) if not part.empty], ignore_index=True)
    counts = table.groupby(["city_key", "mode"]).size()
    table = table.set_index(["city_key", "mode"]).loc[counts[counts >= MIN_PER_CITY].index].reset_index()
    table.to_parquet(SEGMENTS, index=False)
    return table


def predict(length: np.ndarray, params: dict, span: float, vmax_mpm: float) -> np.ndarray:
    """``TransportSpec.travel_time_min`` vectorised (no road limit: timetable segments)."""
    speed = np.full(len(length), params["speed_kmh"], dtype=float)
    if params.get("long_speed_kmh") is not None:
        speed = speed + (params["long_speed_kmh"] - params["speed_kmh"]) * length / (length + params["half_length_m"])
    return params["dwell_min"] + k_term(length, span) / np.clip(speed * 1000 / 60, MIN_SPEED_MPM, vmax_mpm)


class Segments:
    """One mode's segments, arranged for city-weighted scoring."""

    def __init__(self, frame: pd.DataFrame):
        self.length = frame["length_m"].to_numpy(float)
        self.observed = frame["time_min"].to_numpy(float)
        self.codes, names = pd.factorize(frame["city_key"])
        self.n = len(names)
        self.city_time = np.bincount(self.codes, weights=self.observed, minlength=self.n)

    def time_weighted(self, signed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per city: |log error| and signed log error, each segment weighted by its minutes."""
        absolute = np.bincount(self.codes, weights=self.observed * np.abs(signed), minlength=self.n) / self.city_time
        bias = np.bincount(self.codes, weights=self.observed * signed, minlength=self.n) / self.city_time
        return absolute, bias

    def scores(self, predicted: np.ndarray) -> dict:
        signed = np.log(np.maximum(predicted, 1e-9)) - np.log(self.observed)
        absolute, bias = self.time_weighted(signed)
        per_segment = pd.DataFrame({"c": self.codes, "e": signed}).groupby("c")["e"]
        return {
            "error_time": float(np.median(absolute)),
            "bias_time": float(np.median(bias)),
            "error_segment": float(per_segment.apply(lambda e: np.median(np.abs(e))).median()),
            "bias_segment": float(per_segment.median().median()),
        }


def _params(x: np.ndarray, growing: bool) -> dict:
    params = {"dwell_min": max(float(x[0]), 0.0), "speed_kmh": float(np.exp(x[1]))}
    if growing:
        params.update(long_speed_kmh=float(np.exp(x[2])), half_length_m=float(np.exp(x[3])))
    return params


def _minimise(data: Segments, span, vmax_mpm, growing: bool, starts) -> np.ndarray:
    from scipy.optimize import minimize

    def loss(x):
        predicted = predict(data.length, _params(x, growing), span, vmax_mpm)
        return float(data.time_weighted(np.log(np.maximum(predicted, 1e-9)) - np.log(data.observed))[0].mean())

    results = [
        minimize(loss, np.array(s), method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 6000})
        for s in starts
    ]
    return min(results, key=lambda result: result.fun).x


def fit_world(data: Segments, span: float, vmax_mpm: float, growing: bool) -> dict:
    """Minimise the city mean of the time-weighted error from several starts.

    The growing model starts around the one-speed fit of the same cities."""
    one = _params(
        _minimise(data, span, vmax_mpm, False, [[d, np.log(v)] for d in (0.2, 0.6) for v in (25.0, 45.0, 80.0)]), False
    )
    if not growing:
        return {"dwell_min": round(one["dwell_min"], 4), "speed_kmh": round(one["speed_kmh"], 2)}
    v = one["speed_kmh"]
    starts = [
        [one["dwell_min"], np.log(lo * v), np.log(hi * v), np.log(half)]
        for lo in (0.7, 1.0)
        for hi in (1.3, 2.0)
        for half in (1000.0, 5000.0)
    ]
    p = _params(_minimise(data, span, vmax_mpm, True, starts), True)
    return {"dwell_min": round(p["dwell_min"], 4), "speed_kmh": round(p["speed_kmh"], 2),
            "long_speed_kmh": round(p["long_speed_kmh"], 2), "half_length_m": round(p["half_length_m"], 1)}  # fmt: skip


def cross_validate(frame: pd.DataFrame, span: float, vmax_mpm: float, growing: bool) -> dict:
    """Fit without a fold's cities, predict them; scored over all held-out cities."""
    shuffled = np.random.default_rng(20261019).permutation(np.unique(frame["city_key"]))
    predicted = np.empty(len(frame))
    for fold in (part for part in np.array_split(shuffled, min(FOLDS, len(shuffled))) if len(part)):
        test = frame["city_key"].isin(fold).to_numpy()
        params = fit_world(Segments(frame[~test]), span, vmax_mpm, growing)
        predicted[test] = predict(frame["length_m"].to_numpy(float)[test], params, span, vmax_mpm)
    return Segments(frame).scores(predicted)


def assert_matches_library(mode: str, params: dict) -> None:
    """``predict`` must be ``TransportSpec.travel_time_min``, so a fit can go into the registry."""
    from dataclasses import replace

    spec = replace(DEFAULT_REGISTRY_W_TRAIN.get(mode), dwell_min=params["dwell_min"], base_speed_kmh=params["speed_kmh"],
                   long_speed_kmh=params.get("long_speed_kmh"), speed_half_length_m=params.get("half_length_m"))  # fmt: skip
    span, vmax = span_and_vmax(spec)
    lengths = np.array([20.0, 80.0, 200.0, 600.0, 1500.0, 4000.0, 12000.0, 40000.0])
    library = np.array([spec.travel_time_min(float(length), min_speed_mpm=MIN_SPEED_MPM) for length in lengths])
    gap = float(np.max(np.abs(predict(lengths, params, span, vmax) - library)))
    assert gap < 1e-6, f"{mode}: predict and TransportSpec differ by {gap} min"


def fit_mode(frame: pd.DataFrame, mode: str) -> tuple[list[dict], list[dict]]:
    spec = DEFAULT_REGISTRY_W_TRAIN.get(mode)
    span, vmax = span_and_vmax(spec)
    impossible = frame["length_m"] / frame["time_min"] * 0.06 > spec.vmax_tech_kmh  # a timetable error
    frame = frame[~impossible].reset_index(drop=True)
    data = Segments(frame)
    base = {"mode": mode, "cities": data.n, "segments": len(frame), "removed_faster_than_vmax": int(impossible.sum()),
            "removed_share": round(float(impossible.mean()), 4), "span_m": span, "vmax_kmh": spec.vmax_tech_kmh}  # fmt: skip
    registry = {"dwell_min": spec.dwell_min, "speed_kmh": spec.base_speed_kmh}
    if spec.long_speed_kmh is not None:
        registry.update(long_speed_kmh=spec.long_speed_kmh, half_length_m=spec.speed_half_length_m)
    rows = [{**base, "model": "kinematic, registry constants", **registry,
             **{k: round(v, 4) for k, v in data.scores(predict(data.length, registry, span, vmax)).items()}}]  # fmt: skip
    models = {"registry": registry}
    for name, growing in (("kinematic", False), (GROWING, True)):
        params = fit_world(data, span, vmax, growing)
        assert_matches_library(mode, params)
        fitted = data.scores(predict(data.length, params, span, vmax))
        held = cross_validate(frame, span, vmax, growing)
        rows.append({**base, "model": name, **params, **{k: round(v, 4) for k, v in fitted.items()},
                     **{f"cv_{k}": round(v, 4) for k, v in held.items()}})  # fmt: skip
        models[name] = params

    curves = []
    for interval, group in frame.assign(length_class=pd.cut(frame["length_m"], CURVE_BINS, right=False)).groupby(
        "length_class", observed=True
    ):
        mid = float(group["length_m"].median())
        row = {"mode": mode, "length_class": str(interval), "segments": len(group), "cities": group["city_key"].nunique(),
               "median_length_m": round(mid, 1), "observed_kmh": round(float((group["length_m"] / group["time_min"] * 0.06).median()), 2),
               "minutes_share": round(float(group["time_min"].sum() / frame["time_min"].sum()), 4)}  # fmt: skip
        for name, params in models.items():
            row[f"{name} kmh"] = round(float(mid / predict(np.array([mid]), params, span, vmax)[0] * 0.06), 2)
        curves.append(row)
    return rows, curves


def run_fit(logger, workers: int, reuse: bool) -> None:
    table = pd.read_parquet(SEGMENTS) if reuse and SEGMENTS.exists() else collect(workers)
    logger.info("%d segments with a feed geometry in %d cities", len(table), table["city_key"].nunique())
    results, curves = [], []
    for mode in TRANSIT_MODES:
        frame = table[table["mode"] == mode].reset_index(drop=True)
        if frame["city_key"].nunique() < 3:
            continue
        rows, mode_curves = fit_mode(frame, mode)
        results += rows
        curves += mode_curves
        logger.info("%s: %s", mode, rows[-1])
        pd.DataFrame(results).to_csv(RESULTS_DIR / "speed_fit.csv", index=False)
        pd.DataFrame(curves).to_csv(RESULTS_DIR / "speed_fit_curves.csv", index=False)
    speed_by_length(table).to_csv(RESULTS_DIR / "speed_by_length.csv", index=False)


def possible(table: pd.DataFrame) -> pd.DataFrame:
    """Segments no faster than their mode's technical maximum (a faster one is a timetable error)."""
    vmax = table["mode"].map(lambda mode: DEFAULT_REGISTRY_W_TRAIN.get(mode).vmax_tech_kmh)
    return table[table["length_m"] / table["time_min"] * 0.06 <= vmax]


def speed_by_length(table: pd.DataFrame) -> pd.DataFrame:
    """Median scheduled speed by length class: the shape the model must reproduce."""
    frame = possible(table).assign(speed_kmh=lambda f: f["length_m"] / f["time_min"] * 0.06)
    frame["length_class"] = pd.cut(frame["length_m"], bins=FIGURE_BINS)
    result = frame.groupby(["mode", "length_class"], observed=True)["speed_kmh"].agg(["count", "median"]).reset_index()
    result["median"] = result["median"].round(1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["fit"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reuse", action="store_true", help="fit: reuse speed_fit_segments.parquet")
    arguments = parser.parse_args()
    logger = setup("speeds")
    run_fit(logger, arguments.workers, arguments.reuse)


if __name__ == "__main__":
    main()
