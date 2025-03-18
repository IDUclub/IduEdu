# IduEdu

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://img.shields.io/pypi/v/iduedu.svg)](https://pypi.org/project/iduedu/)
[![CI](https://github.com/DDonnyy/IduEdu/actions/workflows/ci_pipeline.yml/badge.svg)](https://github.com/DDonnyy/IduEdu/actions/workflows/ci_pipeline.yml)
[![Coverage](https://codecov.io/gh/DDonnyy/IduEdu/graph/badge.svg?token=VN8CBP8ZW3)](https://codecov.io/gh/DDonnyy/IduEdu)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/MIT)
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

## Acknowledgments

Реализовано при финансовой поддержке Фонда поддержки проектов Национальной технологической инициативы в рамках реализации "дорожной карты" развития высокотехнологичного направления "Искусственный интеллект" на период до 2030 года (Договор № 70-2021-00187)

This research is financially supported by the Foundation for National Technology Initiative's Projects Support as a part of the roadmap implementation for the development of the high-tech field of Artificial Intelligence for the period up to 2030 (agreement 70-2021-00187)
