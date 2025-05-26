from enum import Enum
from typing import Literal


class RegistrationStatus(Enum):
    """
    Enum for registration status of the road.
    """

    FEDERAL = 1
    REGIONAL = 2
    LOCAL = 3


class HighwayType(Enum):
    """
    Enum of highway types. Properties contain registration status & max speeds
    """

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
    def reg_status(self) -> Literal[1, 2, 3]:
        reg_status = {
            HighwayType.MOTORWAY: RegistrationStatus.FEDERAL,
            HighwayType.TRUNK: RegistrationStatus.FEDERAL,
            HighwayType.MOTORWAY_LINK: RegistrationStatus.FEDERAL,
            HighwayType.TRUNK_LINK: RegistrationStatus.FEDERAL,
            HighwayType.PRIMARY: RegistrationStatus.REGIONAL,
            HighwayType.SECONDARY: RegistrationStatus.REGIONAL,
            HighwayType.PRIMARY_LINK: RegistrationStatus.REGIONAL,
            HighwayType.SECONDARY_LINK: RegistrationStatus.REGIONAL,
            HighwayType.TERTIARY_LINK: RegistrationStatus.LOCAL,
            HighwayType.LIVING_STREET: RegistrationStatus.LOCAL,
            HighwayType.TERTIARY: RegistrationStatus.LOCAL,
            HighwayType.UNCLASSIFIED: RegistrationStatus.LOCAL,
            HighwayType.RESIDENTIAL: RegistrationStatus.LOCAL,
        }
        return reg_status[self].value

    @property
    def max_speed(self) -> float:
        """
        Average speed in m/min.
        """
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
