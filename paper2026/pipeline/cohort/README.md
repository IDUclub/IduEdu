# cohort — how the 265 study cities were drawn (October 2026)

Not part of the pipeline. The drawing queried live catalogues and Overpass; neither
returns the same answers twice, so it cannot be rerun to the same result. What it
produced is frozen in `../../inputs/` by `freeze_inputs.py`, and the pipeline starts
there. The scripts that did the drawing are kept unchanged in `scripts/`, in the
state they ran in, as the record of the method.

## The drawing, in order

1. **Catalogues** (`wide_catalogs.py`, `wide_selection.py`, `wide_dt4a.py`,
   `wide_transitous.py`): Mobility Database, Transitland Atlas, DigitalTransport4Africa
   and all of Transitous — 2 457 entries selected.
2. **Download and profile** (`wide_feeds.py`, `wide_profile.py`): 2 374 archives
   downloaded (sha256 in `inputs/feeds.csv`), 2 372 readable as GTFS. Each archive is
   placed by its own stops (median position, spread of 90 % of stops), not by its
   catalogue entry.
3. **Places and boundaries** (`wide_places.py`, `wide_prefetch.py`, `wide_batch.py`,
   `wide_resolve.py`, Overpass through `wide_overpass_guard.py`): the settlement and
   administrative units at a feed's median stop, or at its densest cells for a
   regional feed. A city's boundary is its own `boundary=administrative` unit of at
   most 3 000 km² — a city, not a province.
4. **Five cities per country** (`wide_pick5.py`): in each country the candidates are
   ranked by completeness — distinct stops of the country's feeds within 10 km of the
   city's anchor, so a city served by a regional network (Berlin by VBB) counts as
   fully as one with its own feed — and taken in that order, at most five, at least
   25 km apart, skipping a boundary already taken or inside one. A city's feeds are
   every feed of its country with a stop inside its boundary.
5. **Decisions by hand** (author, 14–17 October): Moscow and Saint Petersburg always
   in (from archived copies of their feeds: the live sources need a token or fail
   TLS); Tokyo (its polygon takes in the sea and islands), Cairo and Alexandria
   (city-governorates) kept above 3 000 km²; Armenia, Kosovo and China kept although
   their feeds are small.
6. **Country, region, income** (`wide_country.py`, `wide_failed.py`): the country of a
   city is that of its boundary; World Bank region and income group by country.

Result: 265 cities in 91 countries, 1 024 feed-to-city links over 626 feeds
(`inputs/cities.csv`, `inputs/city_members.csv`).

## Freezing

    python pipeline/cohort/freeze_inputs.py

writes `inputs/` from the drawing's tables in `results/wide_tier/` and the Overpass
cache: cities, members, feeds, boundary polygons, Geofabrik regions, route-type
fixes, and the drawing's own tables in `inputs/cohort/` (sampling frame, downloads,
profiles, feed outcomes) for the attrition counts.
