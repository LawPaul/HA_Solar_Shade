"""Shared coordinate projection utilities for Solar Shade."""

from __future__ import annotations

import math


def latlon_to_utm(latitude: float, longitude: float) -> tuple[int, float, float]:
    """Convert lat/lon to UTM easting/northing.

    Returns (zone, easting, northing).
    """
    zone = int((longitude + 180) / 6) + 1
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = 2 * f - f * f
    e_prime2 = e2 / (1 - e2)
    k0 = 0.9996
    lat_rad = math.radians(latitude)
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    n = a / math.sqrt(1 - e2 * math.sin(lat_rad) ** 2)
    t = math.tan(lat_rad) ** 2
    c = e_prime2 * math.cos(lat_rad) ** 2
    a_coeff = math.cos(lat_rad) * (math.radians(longitude) - lon0)
    m = a * (
        (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * lat_rad
        - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * math.sin(2 * lat_rad)
        + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * math.sin(4 * lat_rad)
        - (35 * e2**3 / 3072) * math.sin(6 * lat_rad)
    )
    easting = k0 * n * (
        a_coeff + (1 - t + c) * a_coeff**3 / 6
        + (5 - 18 * t + t**2 + 72 * c - 58 * e_prime2) * a_coeff**5 / 120
    ) + 500000.0
    northing = k0 * (
        m + n * math.tan(lat_rad) * (
            a_coeff**2 / 2
            + (5 - t + 9 * c + 4 * c**2) * a_coeff**4 / 24
            + (61 - 58 * t + t**2 + 600 * c - 330 * e_prime2) * a_coeff**6 / 720
        )
    )
    if latitude < 0:
        northing += 10000000.0
    return zone, easting, northing


def utm_to_latlon(
    zone: int, easting: float, northing: float, northern: bool = True,
) -> tuple[float, float]:
    """Convert UTM easting/northing to lat/lon.

    Returns (latitude, longitude).
    """
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = 2 * f - f * f
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    k0 = 0.9996
    x = easting - 500000.0
    y = northing if northern else northing - 10000000.0
    m = y / k0
    mu = m / (a * (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256))
    phi1 = (
        mu
        + (3 * e1 / 2 - 27 * e1**3 / 32) * math.sin(2 * mu)
        + (21 * e1**2 / 16 - 55 * e1**4 / 32) * math.sin(4 * mu)
        + (151 * e1**3 / 96) * math.sin(6 * mu)
        + (1097 * e1**4 / 512) * math.sin(8 * mu)
    )
    n1 = a / math.sqrt(1 - e2 * math.sin(phi1) ** 2)
    r1 = a * (1 - e2) / (1 - e2 * math.sin(phi1) ** 2) ** 1.5
    t1 = math.tan(phi1) ** 2
    e_prime2 = e2 / (1 - e2)
    c1 = e_prime2 * math.cos(phi1) ** 2
    d = x / (n1 * k0)
    lat = phi1 - (n1 * math.tan(phi1) / r1) * (
        d**2 / 2
        - (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * e_prime2) * d**4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * e_prime2 - 3 * c1**2)
        * d**6
        / 720
    )
    lon = (
        d
        - (1 + 2 * t1 + c1) * d**3 / 6
        + (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * e_prime2 + 24 * t1**2)
        * d**5
        / 120
    ) / math.cos(phi1)
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    return math.degrees(lat), math.degrees(lon + lon0)
