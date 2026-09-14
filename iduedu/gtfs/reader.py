from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from zipfile import BadZipFile, ZipFile

import pandas as pd

REQUIRED_GTFS_TABLES = ("agency", "stops", "routes", "trips", "stop_times")
OPTIONAL_GTFS_TABLES = (
    "calendar",
    "calendar_dates",
    "frequencies",
    "shapes",
    "pathways",
    "levels",
    "feed_info",
)


@dataclass(frozen=True, slots=True)
class GTFSFeed:
    """In-memory GTFS Schedule tables loaded from a directory or ZIP archive."""

    source: Path
    tables: dict[str, pd.DataFrame]

    def __contains__(self, table: str) -> bool:
        return table in self.tables

    def __getitem__(self, table: str) -> pd.DataFrame:
        return self.tables[table]

    def get(self, table: str) -> pd.DataFrame | None:
        return self.tables.get(table)


def _normalize_table_names(table_names: Iterable[str] | None) -> tuple[str, ...]:
    if table_names is None:
        return REQUIRED_GTFS_TABLES + OPTIONAL_GTFS_TABLES
    return tuple(dict.fromkeys(str(name).removesuffix(".txt") for name in table_names))


def _read_zip(
    source: Path,
    table_names: tuple[str, ...],
    usecols: dict[str, list[str]] | None,
) -> dict[str, pd.DataFrame]:
    try:
        with ZipFile(source) as archive:
            members: dict[str, str] = {}
            for member in archive.namelist():
                if member.endswith("/"):
                    continue
                name = Path(member).name
                if not name.lower().endswith(".txt"):
                    continue
                table = name[:-4].lower()
                if table in members:
                    raise ValueError(f"GTFS ZIP contains more than one {name!r}")
                members[table] = member

            tables: dict[str, pd.DataFrame] = {}
            for table in table_names:
                member = members.get(table)
                if member is None:
                    continue
                with archive.open(member) as stream:
                    tables[table] = pd.read_csv(
                        stream,
                        dtype=str,
                        encoding="utf-8-sig",
                        keep_default_na=False,
                        na_filter=False,
                        usecols=None if usecols is None else usecols.get(table),
                        low_memory=False,
                    )
            return tables
    except BadZipFile as exc:
        raise ValueError(f"Invalid GTFS ZIP archive: {source}") from exc


def read_gtfs_feed(
    source: str | Path,
    *,
    table_names: Iterable[str] | None = None,
    usecols: dict[str, list[str]] | None = None,
) -> GTFSFeed:
    """Read GTFS Schedule text tables from a directory or ZIP archive.

    All fields are retained as strings. Numeric, date, time and geometry
    conversion belongs to the validation and graph-building layers.
    """

    source = Path(source).expanduser()
    names = _normalize_table_names(table_names)
    if source.is_dir():
        tables: dict[str, pd.DataFrame] = {}
        for table in names:
            path = source / f"{table}.txt"
            if path.is_file():
                tables[table] = pd.read_csv(
                    path,
                    dtype=str,
                    encoding="utf-8-sig",
                    keep_default_na=False,
                    na_filter=False,
                    usecols=None if usecols is None else usecols.get(table),
                    low_memory=False,
                )
    elif source.is_file():
        tables = _read_zip(source, names, usecols)
    else:
        raise FileNotFoundError(f"GTFS source does not exist: {source}")
    return GTFSFeed(source=source, tables=tables)
