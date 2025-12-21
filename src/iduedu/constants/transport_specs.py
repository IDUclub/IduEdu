from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class TransportSpec:
    """
    name: ключ типа транспорта (как в OSM route=*)
    vmax_tech_kmh: тех. максимум, км/ч
    accel_dist_m: дистанция разгона до vmax, м
    brake_dist_m: дистанция торможения с vmax, м
    traffic_coef: [0..1], где 1 = без пробок, 0.7 = среднее замедление и т.п.
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
            try:
                v = float(v)
            except Exception as e:
                raise ValueError(f"{field} must be numeric, got {type(getattr(self, field))}") from e

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
        Время прохождения сегмента в МИНУТАХ.

        Учитывает:
          - tech vmax транспорта (self.vmax_tech_kmh)
          - ограничение дороги (speed_limit_mpm), если есть
          - traffic_coef (пробки/приоритет)
          - время на разгон/торможение по accel_dist_m + brake_dist_m

        segment_len_m: длина сегмента (метры) — ОБЯЗАТЕЛЬНО
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

    def update(self, name: str, **fields) -> TransportSpec:
        key = self._norm_key(name)
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
                accel_dist_m=120.0,
                brake_dist_m=80.0,
                traffic_coef=0.8,
            )
        self.add(defaults, overwrite=False)
        return self._specs[key]

    def list_types(self):
        return list(self._specs.keys())


DEFAULT_REGISTRY = TransportRegistry(
    {
        "bus": TransportSpec("bus", vmax_tech_kmh=90, accel_dist_m=220, brake_dist_m=140, traffic_coef=0.6),
        "trolleybus": TransportSpec(
            "trolleybus", vmax_tech_kmh=70, accel_dist_m=200, brake_dist_m=130, traffic_coef=0.65
        ),
        "tram": TransportSpec("tram", vmax_tech_kmh=75, accel_dist_m=180, brake_dist_m=120, traffic_coef=0.8),
        "subway": TransportSpec("subway", vmax_tech_kmh=80, accel_dist_m=350, brake_dist_m=250, traffic_coef=0.8),
        "train": TransportSpec("train", vmax_tech_kmh=140, accel_dist_m=600, brake_dist_m=450, traffic_coef=0.97),
    }
)
