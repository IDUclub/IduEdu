import math

import pytest

from iduedu.constants.transport_specs import (
    DEFAULT_REGISTRY,
    DEFAULT_REGISTRY_W_TRAIN,
    TransportRegistry,
    TransportSpec,
)

pytestmark = pytest.mark.unit


def approx(a: float, b: float, rel: float = 1e-9, abs_: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)


@pytest.mark.parametrize(
    "spec",
    [
        TransportSpec(name="", vmax_tech_kmh=10, accel_dist_m=1, brake_dist_m=1, base_speed_kmh=20),
        TransportSpec(name="   ", vmax_tech_kmh=10, accel_dist_m=1, brake_dist_m=1, base_speed_kmh=20),
    ],
)
def test_transport_spec_validate_name(spec):
    with pytest.raises(ValueError):
        spec.validate()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(name="bus", vmax_tech_kmh=0, accel_dist_m=1, brake_dist_m=1, base_speed_kmh=20),
        dict(name="bus", vmax_tech_kmh=-1, accel_dist_m=1, brake_dist_m=1, base_speed_kmh=20),
        dict(name="bus", vmax_tech_kmh=10, accel_dist_m=-1, brake_dist_m=1, base_speed_kmh=20),
        dict(name="bus", vmax_tech_kmh=10, accel_dist_m=1, brake_dist_m=-1, base_speed_kmh=20),
        dict(name="bus", vmax_tech_kmh=10, accel_dist_m=1, brake_dist_m=1, base_speed_kmh=0),
        dict(name="bus", vmax_tech_kmh=10, accel_dist_m=1, brake_dist_m=1, base_speed_kmh=-5),
        dict(name="bus", vmax_tech_kmh=10, accel_dist_m=1, brake_dist_m=1, base_speed_kmh=20, avg_wait_time_min=-1),
    ],
)
def test_transport_spec_validate_ranges(kwargs):
    spec = TransportSpec(**kwargs)
    with pytest.raises(ValueError):
        spec.validate()


def test_transport_spec_validate_str():
    spec = TransportSpec(name="bus", vmax_tech_kmh="50", accel_dist_m="10", brake_dist_m="10", base_speed_kmh="20")
    with pytest.raises(ValueError):
        spec.validate()

    bad = TransportSpec(name="bus", vmax_tech_kmh="fast", accel_dist_m=10, brake_dist_m=10, base_speed_kmh=20)
    with pytest.raises(ValueError):
        bad.validate()

    bad_wait = TransportSpec(
        name="bus",
        vmax_tech_kmh=50,
        accel_dist_m=10,
        brake_dist_m=10,
        base_speed_kmh=20,
        avg_wait_time_min="slow",
    )
    with pytest.raises(ValueError):
        bad_wait.validate()


def test_travel_time_zero_or_negative_length():
    spec = TransportSpec("bus", 60, 100, 100, 1.0)
    assert spec.travel_time_min(0) == 0.0
    assert spec.travel_time_min(-5) == 0.0


def test_travel_time_long_segment_with_cruise():
    spec = TransportSpec("bus", vmax_tech_kmh=90, accel_dist_m=100, brake_dist_m=50, base_speed_kmh=60)
    L = 1000.0
    V = 60 * 1000.0 / 60.0  # free speed of the mode, 1000 m/min
    span = 150.0
    expected = (2 * 100 / V) + (2 * 50 / V) + ((L - span) / V)
    got = spec.travel_time_min(L)
    assert approx(got, expected, rel=1e-12, abs_=1e-12)


def test_travel_time_short_segment_no_cruise_uses_v_peak():
    spec = TransportSpec("bus", vmax_tech_kmh=90, accel_dist_m=100, brake_dist_m=100, base_speed_kmh=60)
    L = 50.0
    V = 60 * 1000.0 / 60.0  # free speed of the mode, 1000 m/min
    span = 200.0
    v_peak = V * math.sqrt(L / span)  # 500 m/min
    expected = (2 * L) / v_peak
    got = spec.travel_time_min(L)
    assert approx(got, expected, rel=1e-12, abs_=1e-12)


def test_travel_time_grows_with_length_on_short_segments():
    """Peak speed scaling linearly in L made every sub-span segment cost the same."""
    spec = TransportSpec("bus", vmax_tech_kmh=90, accel_dist_m=700, brake_dist_m=650, base_speed_kmh=40)
    times = [spec.travel_time_min(L, speed_limit_mpm=50 * 1000 / 60) for L in (150, 400, 800, 1200)]
    assert times == sorted(times)
    assert times[-1] > times[0] * 2


def test_travel_time_branches_meet_at_span():
    spec = TransportSpec("bus", vmax_tech_kmh=60, accel_dist_m=100, brake_dist_m=100, base_speed_kmh=20)
    span = 200.0
    below = spec.travel_time_min(span - 1e-6)
    above = spec.travel_time_min(span + 1e-6)
    assert approx(below, above, rel=1e-6, abs_=1e-6)


def test_travel_time_respects_speed_limit():
    spec = TransportSpec("bus", vmax_tech_kmh=80, accel_dist_m=0, brake_dist_m=0, base_speed_kmh=60)
    L = 1000.0

    # no limit: the mode's free-flow speed, 60 km/h -> 1000 m/min
    t_free = spec.travel_time_min(L)

    # a 600 m/min (36 km/h) limit is below free flow, so it binds
    t_limited = spec.travel_time_min(L, speed_limit_mpm=600.0)
    assert t_limited > t_free
    assert approx(t_limited, L / 600.0, rel=1e-12, abs_=1e-12)


def test_a_slower_mode_takes_longer():
    quick = TransportSpec("bus", vmax_tech_kmh=60, accel_dist_m=0, brake_dist_m=0, base_speed_kmh=40)
    slow = TransportSpec("bus", vmax_tech_kmh=60, accel_dist_m=0, brake_dist_m=0, base_speed_kmh=20)
    L = 1200.0
    assert slow.travel_time_min(L) > quick.travel_time_min(L)


def test_travel_time_min_speed_floor_on_short_segment():
    spec = TransportSpec("bus", vmax_tech_kmh=60, accel_dist_m=1000, brake_dist_m=1000, base_speed_kmh=20)
    L = 1.0
    min_speed = 60.0
    got = spec.travel_time_min(L, min_speed_mpm=min_speed)
    expected = (2 * L) / min_speed
    assert approx(got, expected, rel=1e-12, abs_=1e-12)


def test_a_limit_above_the_free_speed_changes_nothing():
    """The road limit binds only where it is lower than what the mode does anyway.

    Free speed used to be affine in the limit. Fitted against published timetables
    the two terms turned out not to be separately identifiable, and dropping the
    limit term improved accuracy for every mode, so a faster road no longer makes
    a bus faster.
    """
    spec = TransportSpec("bus", vmax_tech_kmh=90, accel_dist_m=0, brake_dist_m=0, base_speed_kmh=36)
    L = 3000.0
    free = spec.travel_time_min(L)
    for limit_kmh in (50, 90, 130):
        assert approx(spec.travel_time_min(L, speed_limit_mpm=limit_kmh * 1000 / 60), free, rel=1e-12, abs_=1e-12)
    slow_street = spec.travel_time_min(L, speed_limit_mpm=20 * 1000 / 60)
    assert slow_street > free


def test_free_speed_is_capped_by_technical_maximum():
    spec = TransportSpec("tram", vmax_tech_kmh=40, accel_dist_m=0, brake_dist_m=0, base_speed_kmh=70)
    L = 2000.0
    got = spec.travel_time_min(L, speed_limit_mpm=90 * 1000 / 60)
    assert approx(got, L / (40 * 1000 / 60), rel=1e-12, abs_=1e-12)


def test_dwell_is_added_once_on_both_branches():
    kwargs = dict(vmax_tech_kmh=60, accel_dist_m=100, brake_dist_m=100, base_speed_kmh=20)
    plain = TransportSpec("bus", **kwargs)
    with_dwell = TransportSpec("bus", **kwargs, dwell_min=0.5)
    for L in (50.0, 1000.0):  # below and above span
        assert approx(with_dwell.travel_time_min(L), plain.travel_time_min(L) + 0.5, rel=1e-12, abs_=1e-12)


def test_free_speed_is_the_mode_speed_when_no_limit_is_given():
    spec = TransportSpec("bus", vmax_tech_kmh=60, accel_dist_m=0, brake_dist_m=0, base_speed_kmh=42)
    L = 1000.0
    assert approx(spec.travel_time_min(L), L / (42 * 1000 / 60), rel=1e-12, abs_=1e-12)


def test_registry_add_and_get_normalizes_key():
    reg = TransportRegistry()
    reg.add(TransportSpec("  BuS  ", 60, 1, 1, 1.0))
    assert reg.get("bus").name == "bus"
    assert reg.get(" BUS ").name == "bus"


def test_registry_add_duplicate_without_overwrite_raises():
    reg = TransportRegistry()
    reg.add(TransportSpec("bus", 60, 1, 1, 1.0))
    with pytest.raises(ValueError):
        reg.add(TransportSpec("bus", 70, 1, 1, 1.0))


def test_registry_add_duplicate_with_overwrite():
    reg = TransportRegistry()
    reg.add(TransportSpec("bus", 60, 1, 1, 1.0))
    reg.add(TransportSpec("bus", 70, 1, 1, 1.0), overwrite=True)
    assert reg.get("bus").vmax_tech_kmh == 70


def test_registry_try_get_returns_none():
    reg = TransportRegistry()
    assert reg.try_get("unknown") is None


def test_registry_remove():
    reg = TransportRegistry()
    reg.add(TransportSpec("bus", 60, 1, 1, 1.0))
    reg.remove("bus")
    with pytest.raises(KeyError):
        reg.get("bus")


def test_registry_update_field():
    reg = TransportRegistry()
    reg.add(TransportSpec("bus", 60, 1, 1, 1.0))
    updated = reg.update("bus", base_speed_kmh=42)
    assert updated.base_speed_kmh == 42
    assert reg.get("bus").base_speed_kmh == 42


def test_registry_update_wait_time():
    reg = TransportRegistry()
    reg.add(TransportSpec("bus", 60, 1, 1, 1.0))
    updated = reg.update("bus", avg_wait_time_min=4.5)
    assert updated.avg_wait_time_min == 4.5
    assert reg.get("bus").avg_wait_time_min == 4.5


def test_registry_update_rename_moves_key():
    reg = TransportRegistry()
    reg.add(TransportSpec("bus", 60, 1, 1, 1.0))
    reg.update("bus", name="express_bus")
    with pytest.raises(KeyError):
        reg.get("bus")
    assert reg.get("express_bus").name == "express_bus"


def test_registry_update_rename_to_existing_raises():
    reg = TransportRegistry()
    reg.add(TransportSpec("bus", 60, 1, 1, 1.0))
    reg.add(TransportSpec("tram", 50, 1, 1, 1.0))
    with pytest.raises(ValueError):
        reg.update("bus", name="tram")


def test_registry_ensure_existing_returns_same():
    reg = TransportRegistry()
    reg.add(TransportSpec("bus", 60, 1, 1, 1.0))
    spec = reg.ensure("bus")
    assert spec.name == "bus"
    assert reg.get("bus") is spec


def test_registry_ensure_creates_default_when_missing():
    reg = TransportRegistry()
    spec = reg.ensure("ferry")
    assert spec.name == "ferry"
    # defaults filled in by ensure()
    assert spec.vmax_tech_kmh > 0
    assert reg.get("ferry").name == "ferry"


def test_registry_ensure_uses_provided_defaults():
    reg = TransportRegistry()
    defaults = TransportSpec("funicular", 30, 10, 10, 0.9)
    spec = reg.ensure("funicular", defaults=defaults)
    assert reg.get("funicular").vmax_tech_kmh == 30
    assert spec.base_speed_kmh == 0.9


def test_registry_list_types_contains_added():
    reg = TransportRegistry()
    reg.add(TransportSpec("bus", 60, 1, 1, 1.0))
    reg.add(TransportSpec("tram", 50, 1, 1, 1.0))
    types = reg.list_types()
    assert "bus" in types
    assert "tram" in types


def test_default_registry_has_expected_keys():
    types = set(DEFAULT_REGISTRY.list_types())
    assert {"bus", "tram", "trolleybus", "subway"} <= types
    assert "train" not in types


def test_default_registry_has_mode_specific_wait_times():
    """Fitted against observed headways, not chosen.

    Each value is the median across cities of the harmonic-mean wait measured on
    graphs built from those cities' own timetables, over regular services only.
    """
    assert DEFAULT_REGISTRY.get("bus").avg_wait_time_min == 8.2
    assert DEFAULT_REGISTRY.get("trolleybus").avg_wait_time_min == 5.92
    assert DEFAULT_REGISTRY.get("tram").avg_wait_time_min == 4.95
    assert DEFAULT_REGISTRY.get("subway").avg_wait_time_min == 3.02
    assert DEFAULT_REGISTRY_W_TRAIN.get("train").avg_wait_time_min == 7.71


def test_default_registry_keeps_wait_times_distinct_by_mode():
    """The point of per-mode waits disappears if they collapse into one value.

    This used to assert subway < tram < bus <= trolleybus < train, an ordering
    taken from intuition. Measurement overturned one of its links: trolleybuses
    are waited for less than buses, not more. What survives is what the data
    support -- the metro is waited for least of all -- and the property the test
    existed to protect, that the modes do not share one number.
    """
    waits = {t: DEFAULT_REGISTRY_W_TRAIN.get(t).avg_wait_time_min for t in DEFAULT_REGISTRY_W_TRAIN.list_types()}
    fitted = {mode: waits[mode] for mode in ("bus", "trolleybus", "tram", "subway", "train")}
    assert len(set(fitted.values())) == len(fitted)
    assert fitted["subway"] == min(fitted.values())


def test_default_registry_ignores_road_class_above_the_mode_speed():
    """A motorway does not make a city bus faster, and the data say so.

    Free speed was affine in the posted limit until it was fitted against 541 315
    segments matched between OSM and published timetables: the base speed and the
    limit coefficient turned out to be unidentifiable, and removing the limit term
    improved accuracy for every mode.
    """
    bus = DEFAULT_REGISTRY.get("bus")
    L = 3000.0
    city = bus.travel_time_min(L, speed_limit_mpm=50 * 1000 / 60)
    highway = bus.travel_time_min(L, speed_limit_mpm=90 * 1000 / 60)
    assert approx(highway, city, rel=1e-12, abs_=1e-12)


def test_default_registry_slow_streets_still_slow_every_mode_down():
    """The limit is kept as a ceiling: nothing is modelled faster than the road."""
    L = 3000.0
    for mode in ("bus", "tram", "subway"):
        spec = DEFAULT_REGISTRY.get(mode)
        assert spec.travel_time_min(L, speed_limit_mpm=15 * 1000 / 60) > spec.travel_time_min(L)


def test_default_registry_with_train_has_expected_keys():
    types = set(DEFAULT_REGISTRY_W_TRAIN.list_types())
    assert {"bus", "tram", "trolleybus", "subway", "train"} <= types


def test_default_registry_specs_are_valid():
    for t in DEFAULT_REGISTRY.list_types():
        DEFAULT_REGISTRY.get(t).validate()
    for t in DEFAULT_REGISTRY_W_TRAIN.list_types():
        DEFAULT_REGISTRY_W_TRAIN.get(t).validate()


def test_default_registries_are_independent():
    assert DEFAULT_REGISTRY_W_TRAIN is not DEFAULT_REGISTRY
    assert DEFAULT_REGISTRY.try_get("train") is None
    assert DEFAULT_REGISTRY_W_TRAIN.try_get("train") is not None


def test_osm_route_aliases_follow_gtfs_vocabulary():
    """OSM names some services differently; the mapping mirrors GTFS route types.

    GTFS codes tram and light rail as one type (0), and keeps monorail (12, 405)
    and shared taxi (1501) as their own. The registry follows that, so the two
    sides of an OSM-against-schedule comparison speak the same vocabulary.
    """
    from iduedu.constants.transport_specs import canonical_transport_type, osm_route_values

    assert canonical_transport_type("light_rail") == "tram"
    assert canonical_transport_type("share_taxi") == "taxi"
    assert canonical_transport_type("monorail") == "monorail"
    assert canonical_transport_type("bus") == "bus", "a name the registry already uses is left alone"

    assert osm_route_values("tram") == ["light_rail", "tram"]
    assert osm_route_values("taxi") == ["share_taxi", "taxi"]
    assert osm_route_values("bus") == ["bus"]


def test_inherited_specs_are_declared_and_valid():
    """Monorail and shared taxi have no observed run times, so they inherit."""
    subway = DEFAULT_REGISTRY.get("subway")
    monorail = DEFAULT_REGISTRY.get("monorail")
    assert (monorail.base_speed_kmh, monorail.dwell_min) == (subway.base_speed_kmh, subway.dwell_min)

    bus = DEFAULT_REGISTRY.get("bus")
    taxi = DEFAULT_REGISTRY.get("taxi")
    assert (taxi.base_speed_kmh, taxi.dwell_min) == (bus.base_speed_kmh, bus.dwell_min)

    monorail.validate()
    taxi.validate()


def test_every_registry_mode_can_reach_a_graph():
    """A mode the registry declares must not be dropped between download and build.

    ``get_public_transport_graph`` used to select ground routes from a hand-written
    set of four names, so ``taxi`` and ``monorail`` were fetched from Overpass and
    then discarded without a word -- which hid 524 shared-taxi routes in Addis Ababa
    and 40 in Saint Petersburg.
    """
    import inspect

    from iduedu.graph_builders import public_transport_builders

    source = inspect.getsource(public_transport_builders)
    assert (
        '{"bus", "tram", "trolleybus", "train"} & set(transport_types)' not in source
    ), "ground modes are filtered against a hard-coded list again"
    assert 'set(transport_types) - {"subway"}' in source
