"""Shared coordinate projection utilities for Solar Shade.

Uses `pyproj` (the PROJ library) for coordinate transformations, supporting
any EPSG code — UTM, national grids, Lambert-93, CH1903+, etc.

The ``latlon_to_epsg`` / ``epsg_to_latlon`` pair works for *any* EPSG code
supported by PROJ.  UTM helpers compute the zone number automatically.
"""

from __future__ import annotations

from functools import lru_cache

from pyproj import Transformer


# ── Internal helpers ─────────────────────────────────────────────────────

@lru_cache(maxsize=64)
def _transformer(from_epsg: int, to_epsg: int) -> Transformer:
    """Return a cached Transformer for the given EPSG pair."""
    return Transformer.from_crs(from_epsg, to_epsg, always_xy=True)


_WGS84 = 4326


def _forward(lat: float, lon: float, epsg: int) -> tuple[float, float]:
    """WGS84 lat/lon → projected (easting, northing)."""
    t = _transformer(_WGS84, epsg)
    # always_xy=True means input is (lon, lat), output is (easting, northing)
    e, n = t.transform(lon, lat)
    return float(e), float(n)


def _inverse(easting: float, northing: float, epsg: int) -> tuple[float, float]:
    """Projected (easting, northing) → WGS84 (latitude, longitude)."""
    t = _transformer(epsg, _WGS84)
    lon, lat = t.transform(easting, northing)
    return float(lat), float(lon)


# ── Public UTM API (unchanged interface) ─────────────────────────────────

def latlon_to_utm(latitude: float, longitude: float) -> tuple[int, float, float]:
    """Convert lat/lon to UTM easting/northing.

    Returns (zone, easting, northing).
    """
    zone = int((longitude + 180) / 6) + 1
    northern = latitude >= 0
    epsg = (32600 + zone) if northern else (32700 + zone)
    e, n = _forward(latitude, longitude, epsg)
    return zone, e, n


def utm_to_latlon(
    zone: int, easting: float, northing: float, northern: bool = True,
) -> tuple[float, float]:
    """Convert UTM easting/northing to lat/lon.

    Returns (latitude, longitude).
    """
    epsg = (32600 + zone) if northern else (32700 + zone)
    return _inverse(easting, northing, epsg)


# ── Generic EPSG-based conversion ───────────────────────────────────────

def latlon_to_epsg(latitude: float, longitude: float, epsg: int) -> tuple[float, float]:
    """Convert WGS84 lat/lon to any projected CRS by EPSG code.

    Returns (easting, northing).
    Works for any EPSG code supported by the PROJ library.
    """
    return _forward(latitude, longitude, epsg)


def epsg_to_latlon(easting: float, northing: float, epsg: int) -> tuple[float, float]:
    """Convert projected CRS coordinates back to WGS84 lat/lon.

    Returns (latitude, longitude).
    Works for any EPSG code supported by the PROJ library.
    """
    return _inverse(easting, northing, epsg)
