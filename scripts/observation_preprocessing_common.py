"""Shared time and domain utilities for observation preprocessing."""

from __future__ import annotations

from datetime import datetime


STRICT_CONUS = (24.0, 50.0, -125.0, -66.0)


def seasonally_distributed_24_times(year: int) -> list[datetime]:
    """Return 00 UTC on the 1st and 15th of each month."""
    return [
        datetime(year, month, day, 0)
        for month in range(1, 13)
        for day in (1, 15)
    ]


def six_hour_index(dt: datetime) -> int:
    """Return the zero-based 6-hour index within ``dt.year``."""
    start = datetime(dt.year, 1, 1)
    delta = dt - start
    seconds = int(delta.total_seconds())
    if seconds % (6 * 3600):
        raise ValueError(f"Timestamp is not on a 6-hour boundary: {dt!r}")
    return seconds // (6 * 3600)


def lon_to_180(values):
    return ((values + 180.0) % 360.0) - 180.0
