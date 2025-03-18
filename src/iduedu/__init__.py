"""
IduEdu
========

IduEdu is a Python package for the creation and manipulation of complex city networks from OpenStreetMap.

Homepage https://github.com/DDonnyy/IduEdu.
"""

from ._config import config
from ._api import *
from ._version import VERSION as __version__

from .enums import HighwayType, PublicTrasport
