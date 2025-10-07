from enum import Enum


class Network(Enum):

    DRIVE = "drive"
    DRIVE_SERVICE = "drive_service"
    WALK = "walk"

    @property
    def filter(self) -> str:
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
