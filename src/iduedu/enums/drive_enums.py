from enum import Enum


class HighwayType(Enum):
    MOTORWAY = "motorway"
    TRUNK = "trunk"
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    UNCLASSIFIED = "unclassified"
    RESIDENTIAL = "residential"
    MOTORWAY_LINK = "motorway_link"
    TRUNK_LINK = "trunk_link"
    PRIMARY_LINK = "primary_link"
    SECONDARY_LINK = "secondary_link"
    TERTIARY_LINK = "tertiary_link"
    LIVING_STREET = "living_street"

    @property
    def reg_status(self):
        reg_status = {  # 1 - federal , 2 - regional 3 - local
            HighwayType.MOTORWAY: 1,
            HighwayType.TRUNK: 1,
            HighwayType.PRIMARY: 2,
            HighwayType.SECONDARY: 2,
            HighwayType.TERTIARY: 3,
            HighwayType.UNCLASSIFIED: 3,
            HighwayType.RESIDENTIAL: 3,
            HighwayType.MOTORWAY_LINK: 1,
            HighwayType.TRUNK_LINK: 1,
            HighwayType.PRIMARY_LINK: 2,
            HighwayType.SECONDARY_LINK: 2,
            HighwayType.TERTIARY_LINK: 3,
            HighwayType.LIVING_STREET: 3,
        }
        return reg_status[self]

    @property
    def max_speed(self):
        speeds = {  # km/h
            HighwayType.MOTORWAY: 110,
            HighwayType.TRUNK: 90,
            HighwayType.PRIMARY: 60,
            HighwayType.SECONDARY: 60,
            HighwayType.TERTIARY: 60,
            HighwayType.UNCLASSIFIED: 40,
            HighwayType.RESIDENTIAL: 40,
            HighwayType.MOTORWAY_LINK: 90,
            HighwayType.TRUNK_LINK: 90,
            HighwayType.PRIMARY_LINK: 60,
            HighwayType.SECONDARY_LINK: 60,
            HighwayType.TERTIARY_LINK: 60,
            HighwayType.LIVING_STREET: 20,
        }
        return speeds[self] * 1000 / 60  # metres per minute
