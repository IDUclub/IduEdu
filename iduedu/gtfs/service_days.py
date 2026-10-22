"""Service days of a GTFS feed: which service runs on which calendar day.

Used by the GTFS graph builder to average boarding waits over the days a feed runs:
a gap between departures is only a gap between departures of the same day.
"""

from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd

from iduedu.gtfs.reader import GTFSFeed

#: Without ``service_period`` a feed is averaged over its own calendar, but at most over
#: this many days from its first one: calendars running to 2099 would otherwise build
#: a matrix of decades for what is a weekly cycle.
MAX_AVERAGED_DAYS = 366

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def service_days(
    feed: GTFSFeed,
    service_ids: set[str],
    first: date | None = None,
    last: date | None = None,
    dates: Iterable[date] | None = None,
) -> tuple[pd.DatetimeIndex, list[str], np.ndarray] | None:
    """Which service runs on which day: ``(days, services, matrix[service, day])``.

    The days are ``dates`` when given; else the feed's own calendar span (``calendar``
    and ``calendar_dates``), clipped to ``first``..``last`` when given, else to
    ``MAX_AVERAGED_DAYS`` from its start. ``None`` when the feed carries no calendar.
    """
    calendar = feed.get("calendar")
    exceptions = feed.get("calendar_dates")
    bounds: list[pd.Timestamp] = []
    if calendar is not None and not calendar.empty:
        bounds.append(pd.to_datetime(calendar["start_date"], format="%Y%m%d", errors="coerce").min())
        bounds.append(pd.to_datetime(calendar["end_date"], format="%Y%m%d", errors="coerce").max())
    if exceptions is not None and not exceptions.empty:
        # only added days define the span; a removal is no day of service
        added = exceptions.loc[exceptions["exception_type"].astype(str).eq("1"), "date"]
        exception_days = pd.to_datetime(added, format="%Y%m%d", errors="coerce")
        bounds += [exception_days.min(), exception_days.max()]
    bounds = [bound for bound in bounds if pd.notna(bound)]
    if not bounds:
        return None
    services = sorted(service_ids)
    if dates is not None:
        days = pd.DatetimeIndex(sorted({pd.Timestamp(day) for day in dates}))
    else:
        start = pd.Timestamp(first) if first is not None else min(bounds)
        if last is not None:
            end = pd.Timestamp(last)
        else:
            end = min(max(bounds), start + pd.Timedelta(days=MAX_AVERAGED_DAYS - 1))
        if end < start:
            return pd.DatetimeIndex([]), services, np.zeros((len(services), 0), dtype=bool)
        days = pd.date_range(start, end, freq="D")
    position = {service: index for index, service in enumerate(services)}
    matrix = np.zeros((len(services), len(days)), dtype=bool)
    if calendar is not None and not calendar.empty:
        rows = calendar[calendar["service_id"].astype(str).isin(position)]
        starts = pd.to_datetime(rows["start_date"], format="%Y%m%d", errors="coerce")
        ends = pd.to_datetime(rows["end_date"], format="%Y%m%d", errors="coerce")
        # Monday = 0; 1970-01-01 was a Thursday
        day_of_week = (days.to_numpy().astype("datetime64[D]").astype(np.int64) + 3) % 7
        flags = np.column_stack(
            [
                rows[name].astype(str).eq("1").to_numpy() if name in rows else np.zeros(len(rows), bool)
                for name in _WEEKDAYS
            ]
        )
        for index, (service, row_start, row_end) in enumerate(zip(rows["service_id"].astype(str), starts, ends)):
            if pd.isna(row_start) or pd.isna(row_end):
                continue
            active = flags[index][day_of_week] & (days >= row_start) & (days <= row_end)
            matrix[position[service]] |= active
    if exceptions is not None and not exceptions.empty:
        rows = exceptions[exceptions["service_id"].astype(str).isin(position)]
        exception_days = pd.to_datetime(rows["date"], format="%Y%m%d", errors="coerce")
        column = pd.Series(np.arange(len(days)), index=days)
        for service, when, kind in zip(
            rows["service_id"].astype(str), exception_days, rows["exception_type"].astype(str)
        ):
            if pd.notna(when) and when in column.index:
                matrix[position[service], column[when]] = kind == "1"
    return days, services, matrix


def day_types(
    feed: GTFSFeed, trips: pd.DataFrame, period: tuple[date, date] | None = None, dates: Iterable[date] | None = None
) -> list[tuple[frozenset[str], int]] | None:
    """The feed's distinct service days as ``(active services, number of such days)``.

    Days with the same set of active services have the same timetable, so each set is
    computed once and weighted by its days. ``None`` when the feed has no calendar.
    """
    first, last = period if period is not None else (None, None)
    found = service_days(feed, set(trips["service_id"].astype(str)), first, last, dates)
    if found is None:
        return None
    _, services, matrix = found
    if matrix.shape[1] == 0 or not services:
        return []
    keys, counts = np.unique(np.packbits(matrix, axis=0).T, axis=0, return_counts=True)
    names = np.array(services)
    types = []
    for key, count in zip(keys, counts):
        active = np.unpackbits(key)[: len(services)].astype(bool)
        if active.any():
            types.append((frozenset(names[active]), int(count)))
    return types
