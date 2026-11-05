# pipeline: from the frozen inputs to every table and figure of the paper

Can a public transport network built from OpenStreetMap replace a published timetable? For
265 cities in 91 countries the pipeline builds both networks inside one boundary, compares
them stop by stop, and compares the accessibility each gives within 15, 30 and 45 minutes.

## Inputs (`../inputs/`)

| file | what |
|---|---|
| `cities.csv` | the 265 cities: country, World Bank region and income group, population, boundary relation and its area |
| `city_members.csv`, `feeds.csv` | which of the 626 feeds serve which city; each feed's catalogue, URL, licence and sha256 |
| `boundaries.geojson` | each city's boundary (EPSG:4326), which clips both sources |
| `geofabrik_regions.csv` | the Geofabrik extract each city is cut from |
| `route_type_fixes.csv` | route types a source codes by a scheme of its own (the national feed of South Korea) |
| `cohort/` | the sampling frame (`selection.csv`), every download (`downloads.csv`) and its outcome (`feed_outcomes.csv`), the profile of every archive (`feed_profile.csv`), the candidate cities of each country (`pick5_candidates.csv`) |

The pipeline also reads the GTFS archives, `../feed_cache/<feed_id>.zip`
(`city_feeds.py --fetch` downloads them), and the OpenStreetMap extract of each city,
`../osm/cities/<city>.osm.pbf` (`PAPER_PBF_DIR` points elsewhere). Every OpenStreetMap
graph is built from these files; nothing is requested from Overpass. The Geofabrik regional
extracts the cities were cut from are read only by `osm.py split` and by the count of mapped
stop nodes in `osm_relations.py census`, which keeps the rows it has for a region whose
extract is absent.

## How the cities were drawn

The sampling frame holds 2,457 entries of four open catalogues: the Mobility Database,
the Transitland Atlas, DigitalTransport4Africa and Transitous, with the feeds of Moscow and
Saint Petersburg from archived copies. Every archive was downloaded, read as GTFS and placed
by its own stops: each feed goes to the settlement its stops serve. A city's boundary is its
own OpenStreetMap administrative unit of at most 3,000 km² (Tokyo, whose unit takes in the
sea, and the city-governorates of Cairo and Alexandria are larger): the unit that bears the
city's name, or, where no unit of city size does, the smallest unit holding 70 per cent of
the stops the country's feeds publish within 5 km of the point where the city's feed was
placed, else the unit holding most of them. The candidate cities of a country are ranked by
the number of published stops within 10 km of that point. At most five are taken, at least 25 km
apart, and a city after the first needs at least 20 such stops. Every feed of the country
with a stop inside a city's boundary serves that city. `cohort/pick5_candidates.csv` records
every candidate: its count of stops, the unit chosen for it, the share of the stops that
unit holds where no unit bears the city's name, and why the candidate was or was not taken.
The drawing queried live
catalogues and OpenStreetMap, so it does not rerun to the same result; its result is the
tables above.

## Running

    python pipeline/run.py list        # the stages and the command behind each
    python pipeline/run.py all         # everything; each stage resumes where it stopped
    python pipeline/run.py from waits  # a stage and all after it

Outputs go to `../results/wide_tier/` (tables, `METRICS.md`, `logs/`), `../city_feeds/`,
`../osm_graphs/`, `../gtfs_graphs/` and `../osm_graphs_variants/`; `PAPER_RUN=<name>` sends
them to copies suffixed `_<name>`. The environment is Python with IduEdu (the `pbf` extra)
and the packages of `requirements-pipeline.txt`.

## Stages

| stage | module | makes | tables |
|---|---|---|---|
| city_feeds | `city_feeds.py` | one archive per city: its feeds merged, clipped to the boundary, deduplicated, repaired | `city_feeds.csv` |
| osm_split, osm_pt, osm_walk | `osm.py` | one .osm.pbf per city; its OpenStreetMap transit graph (every mode, the registry's constants) and walk graph | `osm_graphs.csv` |
| gtfs | `gtfs.py` | the timetable graph of each city, with the waits of the average day (reference week 12–18 October 2026, 06:00–22:00, per line) | `gtfs_graphs.csv` |
| coverage | `coverage.py` | stop coverage in both directions, the provenance of each feed's geometry, the displacement of the two stop clouds | `indicators_coverage.csv`, `feed_provenance.csv`, `displacement.csv` |
| relations | `osm_relations.py` | route relations and stops in each city's extract | `route_stops.csv` |
| waits | `waits.py` | each city's wait per mode, and its distribution over cities | `indicators_waits.csv`, `waits_by_city_mode.csv`, `waits_by_mode.csv` |
| speed_fit | `speeds.py` | run time by segment length, fitted on every timetable segment with its own geometry (`--reuse` refits from the saved segments) | `speed_fit.csv`, `speed_fit_curves.csv`, `speed_by_length.csv`, `speed_fit_segments.parquet` |
| variants | `access.py` | accessibility in four variants (D, D1, A_w, A) and the decomposition of A/D into three steps | `accessibility_variants.csv`, `accessibility.csv`, `accessibility_variants_summary.csv`, `accessibility_decomposition.csv`, `osm_graphs_variants.csv` |
| report | `osm_relations.py proxy`, `report.py` | the share of each city's route relations that list a stop, beside the discrepancy; the numbers the paper reports | `proxy_without_schedule.csv`, `METRICS.md` |
| figures | `figures.py` | every figure, into `../figures/wide_*.png` | |

`common.py` holds the paths, the inputs, the cohort and the resumable sweep every stage uses.

## The registry's constants

The waiting times and speeds IduEdu uses for a city without a timetable
(`iduedu/constants/transport_specs.py`) are outputs of this pipeline: `waits_by_mode.csv`
(the median over cities) and `speed_fit.csv` (the cross-validated fit). Each transit graph
records the fingerprint of the constants it was built with, and `osm.py graphs` rebuilds a
graph built under other constants.
