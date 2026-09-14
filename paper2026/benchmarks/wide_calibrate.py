"""Refit the kinematic constants against schedules, and check they generalise.

The model is
    v_free = clip(min(base_speed_kmh, road_limit), floor, vmax_tech)
    t      = dwell_min + run(L, v_free, span)
and ``run`` collapses to ``K / v`` where ``K`` depends only on the segment length
and the mode's span. Precomputing ``K`` turns the fit into a search over two
numbers, cheap enough to enumerate exhaustively rather than trust a local method.

Free speed used to be affine in the road limit, ``base + traffic_coef * limit``.
Fitted here, the two terms proved not separately identifiable -- for buses a fast
base with a weak coefficient and a slow base with a strong one differ by under a
percent -- and dropping the limit term improved accuracy for **every** mode. The
limit survives only as a ceiling, which binds on slow streets and never on rail.

Two rules govern the fit, and both change its answer:

* **Cities are the unit, not segments.** A single city with a dense feed can
  contribute a fifth of all pairs and would otherwise decide the constants for
  everyone. Pairs are capped per city and the error is aggregated across cities.
* **``span`` is fixed at its physical value, not fitted.** Left free it drifted
  to 550 m for a trolleybus -- 275 m of acceleration, which is not a vehicle but
  a place for the rest of the model's error to hide.

Leave-one-city-out is reported next to the fit. A gap between them is the honest
measure of how much of the fit is the world and how much is this sample.
"""

import argparse
import logging

import numpy as np
import pandas as pd
from wide_cohort import measurable
from wide_paths import KINEMATICS_BY_CITY_CSV as PER_CITY_CSV
from wide_paths import KINEMATICS_FIT_CSV as RESULT_CSV
from wide_paths import (
    PAIRS_CSV,
    PROVENANCE_CSV,
)

from iduedu.constants.transport_specs import DEFAULT_REGISTRY_W_TRAIN

logger = logging.getLogger(__name__)


MIN_SPEED_MPM = 60.0
MAX_PAIRS_PER_CITY = 4000
MIN_PAIRS_PER_CITY = 30
MIN_CITIES = 3

#: Share of feed vertices sitting exactly on an OSM vertex, above which the feed's
#: geometry was generated from OSM (``wide_provenance.py``). The controls separate
#: by three orders of magnitude, so the threshold is not a delicate choice.
DERIVED_THRESHOLD = 0.30

DWELL_GRID = np.round(np.arange(0.0, 2.01, 0.05), 2)
BASE_GRID = np.round(np.arange(2.0, 90.1, 1.0), 1)


def _relative_error(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Symmetric error in ratio space: |log(predicted / observed)|.

    The obvious measure, ``|predicted - observed| / observed``, is **not**
    symmetric: a model twice too fast scores 0.5 while one twice too slow scores
    1.0, so minimising it quietly rewards optimism. Fitted that way the bus came
    out 12.5% fast overall and 26-31% fast on the 500-1350 m segments that carry
    most urban service -- which then inflated intermodal accessibility by nearly
    half. In log space both directions cost the same and the fit has no reason to
    prefer either.
    """
    ratio = np.maximum(predicted, 1e-9) / np.maximum(observed, 1e-9)
    return np.abs(np.log(ratio))


def prepare(pairs: pd.DataFrame, mode: str, span_m: float) -> pd.DataFrame:
    """Rows usable for fitting one mode, with the length term precomputed.

    A segment whose road limit could not be recovered because the limit never bound
    -- the mode was already slower than the road -- is kept, with the limit set
    beyond any speed the fit can reach. Its prediction is the mode's free speed,
    which is exactly what the model says, and no information is missing.

    Dropping those rows, as an earlier version did, restricted the fit to segments
    on roads slower than the mode's free speed. That biases the answer downwards
    and feeds back on itself: a higher fitted speed excludes more fast streets,
    which pulls the next fit lower still. Raising the bus from 37 to 41 km/h cut
    the usable bus sample from 100,303 segments to 86,824 before this was fixed.
    """
    usable = pairs["osm_limit_kmh"].notna() | pairs["limit_recovery"].eq("limit_above_free_speed")
    frame = pairs.loc[pairs["mode"].eq(mode) & usable].copy()
    frame = frame.loc[frame["gtfs_time_min"].gt(0) & frame["gtfs_length_m"].gt(0)]
    if frame.empty:
        return frame
    length = frame["gtfs_length_m"].to_numpy(dtype=float)
    # Below the span the vehicle never reaches cruising speed and the profile is
    # acceleration then braking; above it there is a cruising phase. Both reduce
    # to K / v, so the branch is decided once here instead of inside the search.
    frame["k_term"] = np.where(length < span_m, 2.0 * np.sqrt(length * span_m), length + span_m)
    # A non-binding limit is infinite for the purposes of ``min(base, limit)``.
    frame["limit_mpm"] = frame["osm_limit_kmh"].to_numpy(dtype=float) * 1000.0 / 60.0
    frame["limit_mpm"] = frame["limit_mpm"].fillna(np.inf)
    frame["observed_min"] = frame["gtfs_time_min"].to_numpy(dtype=float)
    return frame


def cap_by_city(frame: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """One city must not outvote the rest by sheer feed density."""
    counts = frame.groupby("city_key").size()
    keep = counts[counts >= MIN_PAIRS_PER_CITY].index
    frame = frame.loc[frame["city_key"].isin(keep)]
    if frame.empty:
        return frame
    chosen = []
    generator = np.random.default_rng(seed)
    for _, positions in frame.groupby("city_key", sort=False).indices.items():
        if len(positions) > MAX_PAIRS_PER_CITY:
            positions = generator.choice(positions, MAX_PAIRS_PER_CITY, replace=False)
        chosen.append(positions)
    return frame.iloc[np.sort(np.concatenate(chosen))].reset_index(drop=True)


class Sample:
    """A mode's pairs, arranged so a parameter set can be scored in microseconds.

    The search evaluates tens of thousands of combinations, and a pandas groupby
    per combination would dominate the run. Sorting by city once turns the
    city-weighted score into a walk over contiguous slices.
    """

    def __init__(self, frame: pd.DataFrame, vmax_mpm: float):
        ordered = frame.sort_values("city_key", kind="stable")
        self.cities = ordered["city_key"].to_numpy()
        self.k_term = ordered["k_term"].to_numpy(dtype=float)
        self.limit_mpm = ordered["limit_mpm"].to_numpy(dtype=float)
        self.observed = ordered["observed_min"].to_numpy(dtype=float)
        self.vmax_mpm = vmax_mpm
        boundaries = np.flatnonzero(self.cities[1:] != self.cities[:-1]) + 1
        self.slices = np.split(np.arange(len(self.cities)), boundaries)
        self.city_names = [self.cities[chunk[0]] for chunk in self.slices]

    def city_errors(self, dwell_min: float, base_speed_kmh: float) -> np.ndarray:
        velocity = np.clip(np.minimum(base_speed_kmh * 1000.0 / 60.0, self.limit_mpm), MIN_SPEED_MPM, self.vmax_mpm)
        predicted = dwell_min + self.k_term / velocity
        error = _relative_error(predicted, self.observed)
        return np.array([np.median(error[chunk]) for chunk in self.slices])

    def score(self, dwell_min: float, base_speed_kmh: float) -> float:
        return float(np.median(self.city_errors(dwell_min, base_speed_kmh)))


#: How much dispersion may be traded for an unbiased model. The error surface has a
#: long flat valley -- for buses, (19.5, 0.375) and (39.5, 0.05) score within 0.002
#: of each other -- so the parameters are barely identified and the choice inside
#: the valley is nearly free. Spending that freedom on removing systematic bias is
#: worth far more than the third decimal of the median error: accessibility follows
#: the bias directly, and a model 10% fast everywhere overstates reach everywhere.
BIAS_TOLERANCE = 0.05


def _search(sample: Sample, dwells, bases) -> tuple[dict, float]:
    """Best dispersion first, then the least biased point among the near-best."""
    candidates: list[tuple[float, float, dict]] = []
    for base_kmh in bases:
        # Free speed belongs to the mode, capped by the road only where the road is
        # slower. The affine term this replaced was not identifiable and removing it
        # improved accuracy for every mode; see ``wide_match`` and the plan.
        velocity = np.clip(np.minimum(base_kmh * 1000.0 / 60.0, sample.limit_mpm), MIN_SPEED_MPM, sample.vmax_mpm)
        moving = sample.k_term / velocity
        for dwell in dwells:
            predicted = dwell + moving
            error = _relative_error(predicted, sample.observed)
            score = float(np.median([np.median(error[chunk]) for chunk in sample.slices]))
            signed = np.log(np.maximum(predicted, 1e-9) / np.maximum(sample.observed, 1e-9))
            bias = abs(float(np.median([np.median(signed[chunk]) for chunk in sample.slices])))
            candidates.append((score, bias, {"dwell_min": float(dwell), "base_speed_kmh": float(base_kmh)}))
    if not candidates:
        return None, np.inf
    best_score = min(score for score, _, _ in candidates)
    within = [item for item in candidates if item[0] <= best_score * (1.0 + BIAS_TOLERANCE)]
    score, _, constants = min(within, key=lambda item: (item[1], item[0]))
    return constants, score


def fit(sample: Sample) -> tuple[dict, float]:
    """Exhaustive search over the whole grid.

    This used to be coarse-to-fine, which was a concession to a third parameter
    that no longer exists. With two it is cheap to enumerate everything, and
    enumerating matters here: the surface has a long flat valley, so a two-stage
    search can settle in a different part of it than the exhaustive one and return
    constants *worse* than the ones already in the registry -- which is exactly
    what it did.
    """
    return _search(
        sample,
        np.round(np.arange(0.0, 2.01, 0.025), 3),
        np.round(np.arange(2.0, 90.1, 0.5), 1),
    )


def leave_one_city_out(frame: pd.DataFrame, vmax_mpm: float) -> float | None:
    """Fit without a city, score on it, and report the median of those scores."""
    cities = sorted(frame["city_key"].unique())
    if len(cities) < MIN_CITIES:
        return None
    scores = []
    for city in cities:
        others = frame.loc[frame["city_key"].ne(city)]
        held = frame.loc[frame["city_key"].eq(city)]
        if others.empty or held.empty:
            continue
        constants, _ = fit(Sample(others, vmax_mpm))
        scores.append(float(Sample(held, vmax_mpm).city_errors(**constants)[0]))
    return float(np.median(scores)) if scores else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="*", help="modes to fit; default is every mode with enough cities")
    parser.add_argument("--skip-loco", action="store_true", help="skip leave-one-city-out, which refits once per city")
    parser.add_argument("--exclude", nargs="*", default=[], help="city keys to leave out")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    pairs = pd.read_csv(PAIRS_CSV)
    # Cities outside the cohort do not calibrate the model. A feed with a handful of
    # stops carries a handful of segments, and those segments are one operator's
    # shuttle rather than a sample of the city's service.
    cohort = set(measurable())
    pairs = pairs.loc[pairs["city_key"].map(str).isin(cohort)]
    if arguments.exclude:
        pairs = pairs.loc[~pairs["city_key"].isin(arguments.exclude)]

    # Feeds generated from OSM are marked, never dropped (decision @user). Their
    # run times may have been modelled rather than observed, so every mode is fitted
    # twice -- on everything, and on surveyed feeds alone. The gap between the two
    # measures how much the derived feeds move the answer, which is a result; dropping
    # them silently would assume it.
    derived: set[str] = set()
    if PROVENANCE_CSV.exists():
        provenance = pd.read_csv(PROVENANCE_CSV).drop_duplicates(subset=["city_key"], keep="last")
        derived = set(provenance.loc[provenance["coincident_share"].fillna(0) >= DERIVED_THRESHOLD, "city_key"])
        present = sorted(derived & set(pairs["city_key"]))
        logger.info(f"{len(present)} feeds generated from OSM, marked: {', '.join(present)}")
    logger.info(f"{len(pairs)} pairs, {pairs['city_key'].nunique()} cities")

    modes = arguments.modes or sorted(pairs["mode"].unique())
    results, per_city = [], []
    for mode in modes:
        spec = DEFAULT_REGISTRY_W_TRAIN.try_get(mode)
        if spec is None:
            continue
        span = max(spec.accel_dist_m, 0.0) + max(spec.brake_dist_m, 0.0)
        usable = prepare(pairs, mode, span)
        current = {"dwell_min": spec.dwell_min, "base_speed_kmh": spec.base_speed_kmh}
        vmax_mpm = spec.vmax_tech_kmh * 1000.0 / 60.0

        for scope in ("all", "surveyed_only"):
            subset = usable if scope == "all" else usable.loc[~usable["city_key"].isin(derived)]
            frame = cap_by_city(subset)
            if frame.empty or frame["city_key"].nunique() < MIN_CITIES:
                logger.info(f"{mode} [{scope}]: too few cities, skipped")
                continue

            sample = Sample(frame, vmax_mpm)
            before = sample.score(**current)
            constants, after = fit(sample)
            loco = None if arguments.skip_loco else leave_one_city_out(frame, vmax_mpm)

            row = {
                "mode": mode,
                "scope": scope,
                "cities": int(frame["city_key"].nunique()),
                "derived_cities": int(frame["city_key"].isin(derived).groupby(frame["city_key"]).first().sum()),
                "pairs": len(frame),
                "span_m": round(span, 1),
                **{f"current_{key}": value for key, value in current.items()},
                "current_error": round(before, 4),
                **{f"fitted_{key}": value for key, value in constants.items()},
                "fitted_error": round(after, 4),
                "loco_error": None if loco is None else round(loco, 4),
            }
            results.append(row)
            logger.info(
                f"{mode} [{scope}]: {row['cities']} cities ({row['derived_cities']} derived), {row['pairs']} pairs | "
                f"current err {before:.3f} -> fitted {constants} err {after:.3f} loco {loco}"
            )

            if scope == "all":
                for city, value in zip(sample.city_names, sample.city_errors(**constants)):
                    per_city.append(
                        {"mode": mode, "city_key": city, "error": round(float(value), 4), "derived": city in derived}
                    )

    if results:
        pd.DataFrame(results).to_csv(RESULT_CSV, index=False)
        pd.DataFrame(per_city).to_csv(PER_CITY_CSV, index=False)
        print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
