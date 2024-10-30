# IduEdu

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://img.shields.io/pypi/v/iduedu.svg)](https://pypi.org/project/iduedu/)

### IduEdu is an open-source Python library for the creation and manipulation of complex city networks from OpenStreetMap.

## Features and how to use

1. **[Graphs from OSM/Polygon/Name](./examples/get_any_graph.ipynb)** - Functions for building a graph of roads,
   pedestrians and public transport based on OpenStreetMap (OSM), as well as creating an intermodal (public transport +
   pedestrians) graph.
2. **[Adjacency matrix](./examples/calc_adj_matrix.ipynb)** - Calculate adjacency matrix based on the provided graph and
   edge weight type (_time_min_ or _length_meter_). 

## Installation

**IduEdu** can be installed with ``pip``:

```
pip install IduEdu
```

### Configuration changes

```python
from iduedu import config

config.set_timeout(10)  # Timeout for overpass queries
config.change_logger_lvl('INFO')  # To mute all debug msgs
config.set_enable_tqdm(False)  # To mute all tqdm's progress bars
config.set_overpass_url('http://your.overpass-api.de/interpreter/URL')
```
