from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class TransportSpec:
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
        traffic_coef (float):
            Traffic slowdown coefficient. Values below 1.0 reduce effective speed due to congestion,
            values close to 1.0 indicate free-flow or priority operation.
    """

    name: str
    vmax_tech_kmh: float
    accel_dist_m: float
    brake_dist_m: float
    traffic_coef: float = 1.0

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("TransportSpec.name must be a non-empty string")

        for field in ("vmax_tech_kmh", "accel_dist_m", "brake_dist_m", "traffic_coef"):
            v = getattr(self, field)
            if v is None:
                raise ValueError(f"{field} must not be None")
            if not isinstance(v, (float, int)):
                raise ValueError(f"{field} must be numeric, got {type(getattr(self, field))}")

        if self.vmax_tech_kmh <= 0:
            raise ValueError("vmax_tech_kmh must be > 0")
        if self.accel_dist_m < 0 or self.brake_dist_m < 0:
            raise ValueError("accel_dist_m and brake_dist_m must be >= 0")
        if not (0 < self.traffic_coef <= 1.5):
            raise ValueError("traffic_coef must be in (0, 1.5]")

    def travel_time_min(
        self,
        segment_len_m: float,
        *,
        speed_limit_mpm: float | None = None,
        min_speed_mpm: float = 60.0,  # 60 м/мин = 1 м/с = 3.6 км/ч
    ) -> float:
        """
        Compute travel time (minutes) for a single graph segment.

        The method estimates traversal time using a simplified kinematic model that accounts for:
        - the transport mode technical maximum speed;
        - an optional road speed limit;
        - traffic slowdown coefficient;
        - time lost on acceleration and braking.

        For short segments where the vehicle cannot reach cruising speed, a reduced peak speed
        is assumed and the segment is traversed using an acceleration–deceleration profile
        without a cruising phase.

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

        velocity = float(self.vmax_tech_kmh) * 1000.0 / 60.0

        if speed_limit_mpm is not None and float(speed_limit_mpm) > 0:
            velocity = min(velocity, float(speed_limit_mpm))

        velocity *= float(self.traffic_coef)
        velocity = max(velocity, float(min_speed_mpm))  # защита от нулей

        d_acc = max(float(self.accel_dist_m), 0.0)
        d_brk = max(float(self.brake_dist_m), 0.0)

        span = d_acc + d_brk
        if span > 1e-9 and segment_len_m < span:
            V_peak = velocity * (segment_len_m / span)
            V_peak = max(V_peak, float(min_speed_mpm))

            return (2.0 * segment_len_m) / V_peak

        d_cruise = max(segment_len_m - span, 0.0)

        # время в минутах (на accel/brake средняя скорость ~ V/2)
        t_acc = (2.0 * d_acc) / velocity
        t_brk = (2.0 * d_brk) / velocity
        t_cruise = d_cruise / velocity

        return t_acc + t_brk + t_cruise


class TransportRegistry:
    """
    Registry of available public-transport modes and their specifications.

    The registry stores ``TransportSpec`` objects indexed by normalized transport type names
    (lowercase). It provides utilities for validating transport types, updating parameters,
    and ensuring that unknown types encountered during parsing are assigned reasonable defaults.

    The registry is used throughout graph construction to compute per-edge travel times
    consistently across different transport modes.
    """

    def __init__(self, specs: dict[str, TransportSpec] | None = None):
        self._specs: dict[str, TransportSpec] = {}
        if specs:
            for k, v in specs.items():
                self.add(v if isinstance(v, TransportSpec) else TransportSpec(**v))

    @staticmethod
    def _norm_key(name: str) -> str:
        return name.strip().lower()

    def get(self, name: str) -> TransportSpec:
        key = self._norm_key(name)
        try:
            return self._specs[key]
        except KeyError as e:
            raise KeyError(f"Unknown transport type: {name!r}") from e

    def try_get(self, name: str) -> TransportSpec | None:
        return self._specs.get(self._norm_key(name))

    def add(self, spec: TransportSpec, *, overwrite: bool = False) -> None:
        spec = replace(spec, name=self._norm_key(spec.name))
        spec.validate()
        if (spec.name in self._specs) and not overwrite:
            raise ValueError(f"Transport {spec.name!r} already exists (use overwrite=True)")
        self._specs[spec.name] = spec

    def update(self, transport_type: str, **fields) -> TransportSpec:
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
        key = self._norm_key(name)
        del self._specs[key]

    def ensure(self, name: str, *, defaults: TransportSpec | None = None) -> TransportSpec:
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
                traffic_coef=0.8,
            )
        self.add(defaults, overwrite=False)
        return self._specs[key]

    def list_types(self):
        return list(self._specs.keys())


DEFAULT_REGISTRY = TransportRegistry(
    {
        "bus": TransportSpec("bus", vmax_tech_kmh=90, accel_dist_m=700, brake_dist_m=650, traffic_coef=0.7),
        "trolleybus": TransportSpec(
            "trolleybus", vmax_tech_kmh=70, accel_dist_m=750, brake_dist_m=700, traffic_coef=0.7
        ),
        "tram": TransportSpec("tram", vmax_tech_kmh=75, accel_dist_m=500, brake_dist_m=450, traffic_coef=0.8),
        "subway": TransportSpec("subway", vmax_tech_kmh=80, accel_dist_m=450, brake_dist_m=450, traffic_coef=0.9),
        "train": TransportSpec("train", vmax_tech_kmh=140, accel_dist_m=600, brake_dist_m=450, traffic_coef=0.97),
    }
)
