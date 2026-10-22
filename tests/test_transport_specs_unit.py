import math

import pytest

from iduedu.constants.transport_specs import (
    DEFAULT_REGISTRY,
    DEFAULT_REGISTRY_W_TRAIN,
    TransportRegistry,
    TransportSpec,
)

pytestmark = pytest.mark.unit


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
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_travel_time_short_segment_no_cruise_uses_v_peak():
    spec = TransportSpec("bus", vmax_tech_kmh=90, accel_dist_m=100, brake_dist_m=100, base_speed_kmh=60)
    L = 50.0
    V = 60 * 1000.0 / 60.0  # free speed of the mode, 1000 m/min
    span = 200.0
    v_peak = V * math.sqrt(L / span)  # 500 m/min
    expected = (2 * L) / v_peak
    got = spec.travel_time_min(L)
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)


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
    assert below == pytest.approx(above, rel=1e-6, abs=1e-6)


def test_travel_time_respects_speed_limit():
    spec = TransportSpec("bus", vmax_tech_kmh=80, accel_dist_m=0, brake_dist_m=0, base_speed_kmh=60)
    L = 1000.0

    # no limit: the mode's free-flow speed, 60 km/h -> 1000 m/min
    t_free = spec.travel_time_min(L)

    # a 600 m/min (36 km/h) limit is below free flow, so it binds
    t_limited = spec.travel_time_min(L, speed_limit_mpm=600.0)
    assert t_limited > t_free
    assert t_limited == pytest.approx(L / 600.0, rel=1e-12, abs=1e-12)


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
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)


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
        assert spec.travel_time_min(L, speed_limit_mpm=limit_kmh * 1000 / 60) == pytest.approx(
            free, rel=1e-12, abs=1e-12
        )
    slow_street = spec.travel_time_min(L, speed_limit_mpm=20 * 1000 / 60)
    assert slow_street > free


def test_free_speed_is_capped_by_technical_maximum():
    spec = TransportSpec("tram", vmax_tech_kmh=40, accel_dist_m=0, brake_dist_m=0, base_speed_kmh=70)
    L = 2000.0
    got = spec.travel_time_min(L, speed_limit_mpm=90 * 1000 / 60)
    assert got == pytest.approx(L / (40 * 1000 / 60), rel=1e-12, abs=1e-12)


def test_dwell_is_added_once_on_both_branches():
    kwargs = dict(vmax_tech_kmh=60, accel_dist_m=100, brake_dist_m=100, base_speed_kmh=20)
    plain = TransportSpec("bus", **kwargs)
    with_dwell = TransportSpec("bus", **kwargs, dwell_min=0.5)
    for L in (50.0, 1000.0):  # below and above span
        assert with_dwell.travel_time_min(L) == pytest.approx(plain.travel_time_min(L) + 0.5, rel=1e-12, abs=1e-12)


def test_free_speed_is_the_mode_speed_when_no_limit_is_given():
    spec = TransportSpec("bus", vmax_tech_kmh=60, accel_dist_m=0, brake_dist_m=0, base_speed_kmh=42)
    L = 1000.0
    assert spec.travel_time_min(L) == pytest.approx(L / (42 * 1000 / 60), rel=1e-12, abs=1e-12)


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


def test_default_registry_has_mode_specific_wait_times():
    """Fitted against observed headways, not chosen.

    Each value is the median across cities of the harmonic-mean wait measured on
    graphs built from those cities' own timetables, over regular services only.
    """
    assert DEFAULT_REGISTRY.get("bus").avg_wait_time_min == 12.24
    assert DEFAULT_REGISTRY.get("trolleybus").avg_wait_time_min == 8.02
    assert DEFAULT_REGISTRY.get("tram").avg_wait_time_min == 5.22
    assert DEFAULT_REGISTRY.get("subway").avg_wait_time_min == 3.26
    assert DEFAULT_REGISTRY_W_TRAIN.get("train").avg_wait_time_min == 21.5


def test_default_registry_grows_free_speed_only_where_it_validates():
    """Cross-validated by city: a growing free speed for tram and train only."""
    assert DEFAULT_REGISTRY.get("tram").long_speed_kmh is not None
    assert DEFAULT_REGISTRY_W_TRAIN.get("train").long_speed_kmh is not None
    for mode in ("bus", "trolleybus", "subway"):
        assert DEFAULT_REGISTRY.get(mode).long_speed_kmh is None
    tram = DEFAULT_REGISTRY.get("tram")
    assert tram.free_speed_kmh(5000.0) > tram.free_speed_kmh(500.0)


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
    assert highway == pytest.approx(city, rel=1e-12, abs=1e-12)


def test_default_registry_slow_streets_still_slow_every_mode_down():
    """The limit is kept as a ceiling: nothing is modelled faster than the road."""
    L = 3000.0
    for mode in ("bus", "tram", "subway"):
        spec = DEFAULT_REGISTRY.get(mode)
        assert spec.travel_time_min(L, speed_limit_mpm=15 * 1000 / 60) > spec.travel_time_min(L)


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


def test_free_speed_unchanged_without_long_speed():
    spec = TransportSpec("bus", vmax_tech_kmh=90, accel_dist_m=100, brake_dist_m=50, base_speed_kmh=25)
    same = TransportSpec(
        "bus",
        vmax_tech_kmh=90,
        accel_dist_m=100,
        brake_dist_m=50,
        base_speed_kmh=25,
        long_speed_kmh=25,
        speed_half_length_m=1000,
    )
    for length in (50.0, 400.0, 3000.0, 20000.0):
        assert spec.free_speed_kmh(length) == 25.0
        assert spec.travel_time_min(length) == pytest.approx(same.travel_time_min(length), rel=1e-9, abs=1e-9)


def test_free_speed_grows_with_length():
    spec = TransportSpec(
        "bus",
        vmax_tech_kmh=90,
        accel_dist_m=100,
        brake_dist_m=50,
        base_speed_kmh=25,
        long_speed_kmh=40,
        speed_half_length_m=5000,
    )
    spec.validate()
    assert spec.free_speed_kmh(5000.0) == pytest.approx(32.5, rel=1e-9, abs=1e-9)
    speeds = [spec.free_speed_kmh(length) for length in (100.0, 1000.0, 5000.0, 50000.0)]
    assert speeds == sorted(speeds) and 25 < speeds[0] < speeds[-1] < 40

    length = 5000.0
    expected = (length + 150.0) / (32.5 * 1000.0 / 60.0)
    assert spec.travel_time_min(length) == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_growing_free_speed_still_capped_by_limits():
    spec = TransportSpec(
        "bus",
        vmax_tech_kmh=50,
        accel_dist_m=100,
        brake_dist_m=50,
        base_speed_kmh=25,
        long_speed_kmh=80,
        speed_half_length_m=100,
    )
    length = 20000.0
    assert spec.travel_time_min(length) == pytest.approx((length + 150.0) / (50 * 1000.0 / 60.0), rel=1e-9, abs=1e-9)
    road = 30 * 1000.0 / 60.0
    assert spec.travel_time_min(length, speed_limit_mpm=road) == pytest.approx(
        (length + 150.0) / road, rel=1e-9, abs=1e-9
    )


@pytest.mark.parametrize(
    "extra",
    [
        dict(long_speed_kmh=40),
        dict(speed_half_length_m=1000),
        dict(long_speed_kmh=0, speed_half_length_m=1000),
        dict(long_speed_kmh=40, speed_half_length_m=0),
        dict(long_speed_kmh="40", speed_half_length_m=1000),
    ],
)
def test_long_speed_validation(extra):
    spec = TransportSpec("bus", vmax_tech_kmh=90, accel_dist_m=100, brake_dist_m=50, base_speed_kmh=25, **extra)
    with pytest.raises(ValueError):
        spec.validate()


def test_registry_update_accepts_long_speed():
    registry = TransportRegistry({"bus": DEFAULT_REGISTRY.get("bus")})
    spec = registry.update("bus", long_speed_kmh=40.0, speed_half_length_m=5000.0)
    assert spec.long_speed_kmh == 40.0 and registry.try_get("bus").speed_half_length_m == 5000.0


def test_registry_add_lookup_overwrite_and_remove():
    registry = TransportRegistry()
    registry.add(TransportSpec("  BuS  ", 60, 1, 1, 1.0))
    assert registry.get(" BUS ").name == "bus"
    with pytest.raises(ValueError):
        registry.add(TransportSpec("bus", 70, 1, 1, 1.0))
    registry.add(TransportSpec("bus", 70, 1, 1, 1.0), overwrite=True)
    registry.add(TransportSpec("tram", 50, 1, 1, 1.0))
    assert registry.get("bus").vmax_tech_kmh == 70
    assert set(registry.list_types()) == {"bus", "tram"}
    registry.remove("bus")
    assert registry.try_get("bus") is None
    with pytest.raises(KeyError):
        registry.get("bus")


def test_registry_updates_fields_together():
    registry = TransportRegistry({"bus": DEFAULT_REGISTRY.get("bus")})
    updated = registry.update("bus", base_speed_kmh=42, avg_wait_time_min=4.5)
    assert updated.base_speed_kmh == registry.get("bus").base_speed_kmh == 42
    assert updated.avg_wait_time_min == registry.get("bus").avg_wait_time_min == 4.5


def test_registry_ensure_reuses_existing_and_applies_defaults():
    registry = TransportRegistry({"bus": DEFAULT_REGISTRY.get("bus")})
    assert registry.ensure("bus") is registry.get("bus")
    ferry = registry.ensure("ferry")
    assert ferry.name == "ferry" and ferry.vmax_tech_kmh > 0
    assert registry.get("ferry") is ferry
    defaults = TransportSpec("funicular", 30, 10, 10, 0.9)
    spec = registry.ensure("funicular", defaults=defaults)
    assert spec.vmax_tech_kmh == 30 and spec.base_speed_kmh == 0.9
    assert registry.get("funicular") is spec


def test_default_registries_have_valid_modes_and_are_independent():
    assert DEFAULT_REGISTRY is not DEFAULT_REGISTRY_W_TRAIN
    assert {"bus", "tram", "trolleybus", "subway"} <= set(DEFAULT_REGISTRY.list_types())
    assert {"bus", "tram", "trolleybus", "subway", "train"} <= set(DEFAULT_REGISTRY_W_TRAIN.list_types())
    assert DEFAULT_REGISTRY.try_get("train") is None
    assert DEFAULT_REGISTRY_W_TRAIN.try_get("train") is not None
    for registry in [DEFAULT_REGISTRY, DEFAULT_REGISTRY_W_TRAIN]:
        for name in registry.list_types():
            registry.get(name).validate()
