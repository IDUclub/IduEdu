from enum import Enum


class PublicTrasport(Enum):
    """
    Enumeration class for edge types in graphs.
    """

    SUBWAY = "subway"
    BUS = "bus"
    TRAM = "tram"
    TROLLEYBUS = "trolleybus"
    TRAIN = "train"

    @property
    def avg_speed(self) -> float:
        """
        Average speed in m/min.
        """
        speeds = {  # km/h
            PublicTrasport.SUBWAY: 40,
            PublicTrasport.BUS: 20,
            PublicTrasport.TRAM: 20,
            PublicTrasport.TROLLEYBUS: 18,
            PublicTrasport.TRAIN: 60,
        }
        return speeds[self] * 1000 / 60
