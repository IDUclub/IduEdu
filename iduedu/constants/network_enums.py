from enum import Enum

#: Node types of a public-transport graph that the walk graph is joined to: platforms (a subway station
#: without mapped entrances is one too) and subway entrances and exits.
WALK_ACCESS_NODE_TYPES = frozenset(
    {"platform", "station_entry_exit", "subway_entry_exit", "subway_entry", "subway_exit"}
)


class Network(Enum):
    """Preset OpenStreetMap network filters used by graph builders."""

    DRIVE = "drive"
    DRIVE_SERVICE = "drive_service"
    WALK = "walk"
    BIKE = "bike"

    @property
    def filter(self) -> str:
        """Return the Overpass way filter associated with this network type."""
        speeds = {
            Network.DRIVE: (
                f'["highway"]["area"!~"yes"]["access"!~"private"]'
                f'["highway"!~"abandoned|bridleway|bus_guideway|construction|corridor|cycleway|elevator|'
                f"escalator|footway|no|path|pedestrian|planned|platform|proposed|raceway|razed|service|"
                f'steps|track"]'
                f'["motor_vehicle"!~"no"]["motorcar"!~"no"]'
                f'["service"!~"alley|driveway|emergency_access|parking|parking_aisle|private"]'
            ),
            Network.WALK: (
                f'["highway"]["area"!~"yes"]["access"!~"private"]'
                f'["highway"!~"abandoned|bus_guideway|construction|cycleway|motor|no|planned|platform|'
                f'proposed|raceway|razed"]'
                f'["foot"!~"no"]["service"!~"private"]'
                f'["sidewalk"!~"separate"]["sidewalk:both"!~"separate"]'
                f'["sidewalk:left"!~"separate"]["sidewalk:right"!~"separate"]'
            ),
            Network.BIKE: (
                f'["highway"]["area"!~"yes"]["access"!~"private"]'
                f'["highway"!~"abandoned|bus_guideway|construction|corridor|elevator|escalator|motor|no|'
                f'planned|platform|proposed|raceway|razed|steps"]'
                f'["bicycle"!~"no"]["service"!~"private"]'
            ),
            Network.DRIVE_SERVICE: (
                f'["highway"]["area"!~"yes"]["access"!~"private"]'
                f'["highway"!~"abandoned|bridleway|bus_guideway|construction|corridor|cycleway|elevator|'
                f"escalator|footway|no|path|pedestrian|planned|platform|proposed|raceway|razed|steps|"
                f'track"]'
                f'["motor_vehicle"!~"no"]["motorcar"!~"no"]'
                f'["service"!~"emergency_access|parking|parking_aisle|private"]'
            ),
        }
        return speeds[self]
