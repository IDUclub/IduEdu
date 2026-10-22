from dataclasses import dataclass, replace
from math import sqrt


@dataclass(frozen=True, slots=True)
class TransportSpec:  # pylint: disable=too-many-instance-attributes
    """
    Configuration of a single public-transport mode used to estimate travel time on graph edges.

    Each transport specification defines technical and operational characteristics of a mode
    (e.g. bus, tram, subway) that are used to compute per-edge travel time based on segment length,
    road speed limits, acceleration/braking behavior, and traffic conditions.

    Attributes:
        name (str):
            Transport type identifier, usually matching the OSM ``route=*`` value
            (e.g. ``"bus"``, ``"tram"``, ``"subway"``).
        vmax_tech_kmh (float):
            Technical maximum speed of the vehicle in kilometers per hour.
        accel_dist_m (float):
            Typical distance (meters) required to accelerate from standstill to cruising speed.
        brake_dist_m (float):
            Typical distance (meters) required to decelerate from cruising speed to standstill.
        base_speed_kmh (float):
            Free-flow speed of the mode in kilometers per hour: the speed it holds
            between stops once it is up to speed.

            A road speed limit does **not** scale this value. Earlier versions made
            free speed affine in the limit, ``base + traffic_coef * limit``, and
            fitting that form against published timetables showed the two terms are
            not separately identifiable: for buses, a fast base with a weak
            coefficient and a slow base with a strong one differ by less than a
            percent in error. Dropping the term outright improved accuracy for
            every mode measured (bus, tram, trolleybus, subway, commuter rail),
            because OSM's ``maxspeed`` describes what a car may do, not what a bus
            with passengers, stops and traffic actually does.

            The limit is still honoured where it binds: a vehicle is not modelled
            faster than the road allows.
        dwell_min (float):
            Time lost standing at a stop, in minutes. Added once per segment. Defaults to ``0``.
        long_speed_kmh (float | None):
            Free speed approached on long segments, in km/h. With ``speed_half_length_m``
            the free speed grows with the segment length ``L`` from ``base_speed_kmh``
            towards it::

                v(L) = base_speed_kmh + (long_speed_kmh - base_speed_kmh) * L / (L + speed_half_length_m)

            Timetables show it for every mode: a long stretch between stops runs on roads
            and tracks built for speed, which a single free speed cannot express -- buses
            make about 22 km/h over 500-950 m and 36 km/h over 5-10 km, of which
            acceleration and dwell explain only part. Both ``None`` (default): one free
            speed, ``base_speed_kmh``, on every segment.
        speed_half_length_m (float | None):
            Segment length at which the free speed is halfway from ``base_speed_kmh`` to
            ``long_speed_kmh``, in metres. Set together with ``long_speed_kmh``.
        avg_wait_time_min (float):
            Average passenger waiting time in minutes. It is assigned to directed
            ``boarding`` edges built from OSM data.

    See also:
        https://iduclub.github.io/IduEdu/examples/transport_registry.html
    """

    name: str
    vmax_tech_kmh: float
    accel_dist_m: float
    brake_dist_m: float
    base_speed_kmh: float
    avg_wait_time_min: float = 1.0
    dwell_min: float = 0.0
    long_speed_kmh: float | None = None
    speed_half_length_m: float | None = None

    def validate(self) -> None:
        """Validate transport specification fields.

        Raises:
            ValueError: If a required field is missing or outside the supported
            range.
        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("TransportSpec.name must be a non-empty string")

        for field in (
            "vmax_tech_kmh",
            "accel_dist_m",
            "brake_dist_m",
            "avg_wait_time_min",
            "base_speed_kmh",
            "dwell_min",
        ):
            v = getattr(self, field)
            if v is None:
                raise ValueError(f"{field} must not be None")
            if not isinstance(v, (float, int)):
                raise ValueError(f"{field} must be numeric, got {type(getattr(self, field))}")

        if self.vmax_tech_kmh <= 0:
            raise ValueError("vmax_tech_kmh must be > 0")
        if self.accel_dist_m < 0 or self.brake_dist_m < 0:
            raise ValueError("accel_dist_m and brake_dist_m must be >= 0")
        if self.avg_wait_time_min < 0:
            raise ValueError("avg_wait_time_min must be >= 0")
        if self.base_speed_kmh <= 0:
            raise ValueError("base_speed_kmh must be > 0")
        if self.dwell_min < 0:
            raise ValueError("dwell_min must be >= 0")
        if (self.long_speed_kmh is None) != (self.speed_half_length_m is None):
            raise ValueError("long_speed_kmh and speed_half_length_m must be set together")
        if self.long_speed_kmh is not None:
            for field in ("long_speed_kmh", "speed_half_length_m"):
                if not isinstance(getattr(self, field), (float, int)):
                    raise ValueError(f"{field} must be numeric, got {type(getattr(self, field))}")
            if self.long_speed_kmh <= 0:
                raise ValueError("long_speed_kmh must be > 0")
            if self.speed_half_length_m <= 0:
                raise ValueError("speed_half_length_m must be > 0")

    def free_speed_kmh(self, segment_len_m: float) -> float:
        """Free speed on a segment of this length: ``base_speed_kmh``, or growing with
        the length towards ``long_speed_kmh`` when that is set."""
        base = float(self.base_speed_kmh)
        if self.long_speed_kmh is None or segment_len_m <= 0:
            return base
        share = float(segment_len_m) / (float(segment_len_m) + float(self.speed_half_length_m))
        return base + (float(self.long_speed_kmh) - base) * share

    def travel_time_min(
        self,
        segment_len_m: float,
        *,
        speed_limit_mpm: float | None = None,
        min_speed_mpm: float = 60.0,  # 60 m/min = 1 m/s = 3.6 km/h
    ) -> float:
        """
        Compute travel time (minutes) for a single graph segment.

        The method estimates traversal time using a simplified kinematic model that accounts for:
        - the transport mode technical maximum speed;
        - a free speed that may grow with the segment length (``long_speed_kmh``);
        - an optional road speed limit, entering the free-flow speed affinely;
        - time lost standing at a stop (``dwell_min``);
        - time lost on acceleration and braking.

        For short segments where the vehicle cannot reach cruising speed, a reduced peak speed
        is assumed and the segment is traversed using an acceleration-deceleration profile
        without a cruising phase. The peak speed scales as ``sqrt(L / span)`` because under
        constant acceleration the distance covered grows with the square of the speed reached;
        the two branches meet continuously at ``L == span``.

        Parameters:
            segment_len_m (float):
                Segment length in meters. Must be positive.
            speed_limit_mpm (float | None):
                Optional road speed limit in meters per minute. If provided, the effective speed
                will not exceed this value.
            min_speed_mpm (float):
                Lower bound for effective speed (meters per minute), used to avoid unrealistically
                large travel times on very short segments.

        Returns:
            float:
                Estimated travel time for the segment in minutes.
        """
        segment_len_m = float(segment_len_m)
        if segment_len_m <= 0:
            return 0.0

        vmax = float(self.vmax_tech_kmh) * 1000.0 / 60.0

        # Free speed is a property of the mode, not of the road: see ``base_speed_kmh``.
        # A posted limit only matters where it is *below* what the mode would do
        # anyway, which is rare on transit routes and never on rail.
        velocity = min(self.free_speed_kmh(segment_len_m) * 1000.0 / 60.0, vmax)
        if speed_limit_mpm is not None and float(speed_limit_mpm) > 0:
            velocity = min(velocity, float(speed_limit_mpm))
        velocity = max(velocity, float(min_speed_mpm))  # avoid zero speed

        d_acc = max(float(self.accel_dist_m), 0.0)
        d_brk = max(float(self.brake_dist_m), 0.0)

        span = d_acc + d_brk
        dwell = float(self.dwell_min)

        if span > 1e-9 and segment_len_m < span:
            v_peak = velocity * sqrt(segment_len_m / span)
            v_peak = max(v_peak, float(min_speed_mpm))

            return dwell + (2.0 * segment_len_m) / v_peak

        # acceleration and braking together cost the same as covering `span` twice,
        # so the whole segment reduces to (L + span) / v
        return dwell + (segment_len_m + span) / velocity


class TransportRegistry:
    """
    Registry of available public-transport modes and their specifications.

    The registry stores ``TransportSpec`` objects indexed by normalized transport type names
    (lowercase). It provides utilities for validating transport types, updating parameters,
    and ensuring that unknown types encountered during parsing are assigned reasonable defaults.

    The registry is used throughout graph construction to compute per-edge travel times
    consistently across different transport modes.

    See also:
        https://iduclub.github.io/IduEdu/examples/transport_registry.html
    """

    def __init__(self, specs: dict[str, TransportSpec] | None = None):
        """Initialize the registry with optional transport specifications."""
        self._specs: dict[str, TransportSpec] = {}
        if specs:
            for k, v in specs.items():
                self.add(v if isinstance(v, TransportSpec) else TransportSpec(**v))

    @staticmethod
    def _norm_key(name: str) -> str:
        return name.strip().lower()

    def get(self, name: str) -> TransportSpec:
        """Return a transport specification by name.

        Args:
            name: Transport type name. Matching is case-insensitive and ignores
                surrounding whitespace.

        Raises:
            KeyError: If the transport type is unknown.
        """
        key = self._norm_key(name)
        try:
            return self._specs[key]
        except KeyError as e:
            raise KeyError(f"Unknown transport type: {name!r}") from e

    def try_get(self, name: str) -> TransportSpec | None:
        """Return a transport specification, or ``None`` if it is unknown."""
        return self._specs.get(self._norm_key(name))

    def add(self, spec: TransportSpec, *, overwrite: bool = False) -> None:
        """Add a transport specification to the registry.

        Args:
            spec: Specification to add.
            overwrite: If true, replace an existing specification with the
                same normalized name.

        Raises:
            ValueError: If ``spec`` is invalid or already exists.
        """
        spec = replace(spec, name=self._norm_key(spec.name))
        spec.validate()
        if (spec.name in self._specs) and not overwrite:
            raise ValueError(f"Transport {spec.name!r} already exists (use overwrite=True)")
        self._specs[spec.name] = spec

    def update(self, transport_type: str, **fields) -> TransportSpec:
        """Update fields of an existing transport specification.

        Args:
            transport_type: Existing transport type name.
            **fields: Dataclass fields to update.

        Returns:
            Updated transport specification.

        Raises:
            KeyError: If ``transport_type`` is unknown.
            ValueError: If updated fields are invalid or rename conflicts.
        """
        key = self._norm_key(transport_type)
        cur = self.get(key)
        if "name" in fields:
            fields["name"] = self._norm_key(fields["name"])
        nxt = replace(cur, **fields)
        nxt.validate()

        if nxt.name != key:
            if nxt.name in self._specs:
                raise ValueError(f"Cannot rename to {nxt.name!r}: already exists")
            del self._specs[key]
        self._specs[nxt.name] = nxt
        return nxt

    def remove(self, name: str) -> None:
        """Remove a transport specification by name."""
        key = self._norm_key(name)
        del self._specs[key]

    def ensure(self, name: str, *, defaults: TransportSpec | None = None) -> TransportSpec:
        """Return an existing spec or create one from defaults.

        Args:
            name: Transport type name.
            defaults: Optional specification used when the name is missing.

        Returns:
            Existing or newly registered transport specification.
        """
        key = self._norm_key(name)
        spec = self._specs.get(key)
        if spec:
            return spec

        if defaults is None:
            defaults = TransportSpec(
                name=key,
                vmax_tech_kmh=25.0,
                accel_dist_m=500.0,
                brake_dist_m=500.0,
                base_speed_kmh=20.0,
            )
        self.add(defaults, overwrite=False)
        return self._specs[key]

    def list_types(self):
        """Return registered transport type names."""
        return list(self._specs.keys())


# Speed parameters are fitted against observed GTFS run times: segments are matched to OSM
# travel edges by both endpoints, which yields triples of (length, scheduled time, road limit).
# Waiting times are harmonic means of boarding-edge times, taken per city and then aggregated
# by the median across cities: within a city routing takes the minimum over available
# departures, between cities no such minimum exists.
#: OpenStreetMap tags some services under names the registry does not use, and a
#: request for ``tram`` must find them or the mode goes missing from the graph.
#: The mapping follows GTFS, which codes light rail and tram as one type (0) and
#: keeps monorail (12, 405) and shared taxi (1501) apart from their relatives.
#:
#: Measured over 126 cities: 14 hold ``light_rail`` routes, 9 hold ``share_taxi``
#: -- among them Moscow, Saint Petersburg, Kampala and Jakarta, whose informal
#: networks were invisible before this -- and 8 hold ``monorail``.
OSM_ROUTE_ALIASES: dict[str, str] = {
    "light_rail": "tram",
    "monorail": "monorail",
    "share_taxi": "taxi",
}


def osm_route_values(transport_type: str) -> list[str]:
    """Every ``route`` tag value that stands for this transport type in OSM."""
    values = [transport_type]
    values.extend(
        tag for tag, canonical in OSM_ROUTE_ALIASES.items() if canonical == transport_type and tag != transport_type
    )
    return sorted(set(values))


def canonical_transport_type(osm_route_value: str) -> str:
    """The registry's name for an OSM ``route`` tag value."""
    return OSM_ROUTE_ALIASES.get(osm_route_value, osm_route_value)


_DEFAULT_TRANSPORT_SPECS = {
    "bus": TransportSpec(
        "bus",
        vmax_tech_kmh=90,
        accel_dist_m=25.9,
        brake_dist_m=24.1,
        avg_wait_time_min=12.24,
        base_speed_kmh=29.5,
        dwell_min=0.387,
    ),
    "trolleybus": TransportSpec(
        "trolleybus",
        vmax_tech_kmh=70,
        accel_dist_m=25.9,
        brake_dist_m=24.1,
        avg_wait_time_min=8.02,
        base_speed_kmh=20.0,
        dwell_min=0.159,
    ),
    "tram": TransportSpec(
        "tram",
        vmax_tech_kmh=75,
        accel_dist_m=52.6,
        brake_dist_m=47.4,
        avg_wait_time_min=5.22,
        base_speed_kmh=11.5,
        long_speed_kmh=74.1,
        speed_half_length_m=3506.0,
        dwell_min=0.0,
    ),
    "subway": TransportSpec(
        "subway",
        vmax_tech_kmh=80,
        accel_dist_m=25.0,
        brake_dist_m=25.0,
        avg_wait_time_min=3.26,
        base_speed_kmh=56.7,
        dwell_min=0.694,
    ),
    "monorail": TransportSpec(
        "monorail",
        vmax_tech_kmh=80,
        accel_dist_m=25.0,
        brake_dist_m=25.0,
        avg_wait_time_min=3.26,
        base_speed_kmh=56.7,
        dwell_min=0.694,
    ),
    "taxi": TransportSpec(
        "taxi",
        vmax_tech_kmh=90,
        accel_dist_m=25.9,
        brake_dist_m=24.1,
        avg_wait_time_min=12.24,
        base_speed_kmh=29.5,
        dwell_min=0.387,
    ),
}
_TRAIN_SPEC = TransportSpec(
    "train",
    vmax_tech_kmh=140,
    accel_dist_m=114.3,
    brake_dist_m=85.7,
    avg_wait_time_min=21.5,
    base_speed_kmh=33.5,
    long_speed_kmh=78.9,
    speed_half_length_m=5270.0,
    dwell_min=0.365,
)

DEFAULT_REGISTRY = TransportRegistry(_DEFAULT_TRANSPORT_SPECS)
DEFAULT_REGISTRY_W_TRAIN = TransportRegistry({**_DEFAULT_TRANSPORT_SPECS, "train": _TRAIN_SPEC})
