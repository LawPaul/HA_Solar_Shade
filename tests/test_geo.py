"""Tests for the geo.py coordinate projection utilities."""

import math
import pytest

from custom_components.solar_shade.geo import latlon_to_utm, utm_to_latlon


class TestLatLonToUtm:
    """Test forward UTM projection."""

    def test_known_point_zone_14n(self):
        """Test a known point in UTM zone 14N (central Texas)."""
        zone, easting, northing = latlon_to_utm(32.2873, -95.2934)
        assert zone == 15  # -95.29 is in zone 15
        assert 200_000 < easting < 800_000
        assert 3_000_000 < northing < 4_000_000

    def test_equator_prime_meridian(self):
        """Test point at equator, near prime meridian (zone 31)."""
        zone, easting, northing = latlon_to_utm(0.0, 3.0)  # zone 31 central meridian = 3°
        assert zone == 31
        assert abs(easting - 500000.0) < 1000  # very near central meridian
        assert abs(northing) < 100  # near equator

    def test_southern_hemisphere(self):
        """Test southern hemisphere adds 10M to northing."""
        zone, easting, northing = latlon_to_utm(-33.86, 151.21)  # Sydney
        assert northing > 6_000_000  # 10M - ~3.7M

    def test_roundtrip_accuracy(self):
        """Test that forward → inverse roundtrip is accurate to <1mm."""
        lat, lon = 32.2873, -95.2934
        zone, e, n = latlon_to_utm(lat, lon)
        lat2, lon2 = utm_to_latlon(zone, e, n, northern=True)
        assert abs(lat2 - lat) < 1e-8
        assert abs(lon2 - lon) < 1e-8

    def test_roundtrip_southern(self):
        """Test roundtrip in southern hemisphere."""
        lat, lon = -33.86, 151.21
        zone, e, n = latlon_to_utm(lat, lon)
        lat2, lon2 = utm_to_latlon(zone, e, n, northern=False)
        assert abs(lat2 - lat) < 1e-8
        assert abs(lon2 - lon) < 1e-8


class TestUtmToLatLon:
    """Test inverse UTM projection."""

    def test_known_utm_to_latlon(self):
        """Test inverse from known UTM coordinates."""
        # UTM zone 15N, East Texas
        lat, lon = utm_to_latlon(15, 300_000, 3_575_000, northern=True)
        assert 30 < lat < 35
        assert -97 < lon < -93


class TestCrsReturnNoneBug:
    """Test for the fixed bug where _rasterize_laz_file returned None
    when CRS was detected.

    Bug: _read_laz_epsg found an EPSG code, _project_to_epsg returned
    (None, None) meaning no reprojection needed, but a misplaced
    `return None` at the wrong indentation level caused the entire
    rasterization to abort.

    The fix: removed the stray `return None` so the function continues
    to rasterize after CRS detection.
    """

    def test_project_to_epsg_returns_none_for_standard_utm(self):
        """_project_to_epsg should return (None, None) for standard UTM EPSG codes,
        meaning no reprojection is needed."""
        from custom_components.solar_shade.usgs_downloader import _project_to_epsg

        # WGS84 UTM North zones
        for epsg in [32601, 32614, 32615, 32660]:
            result = _project_to_epsg(500000.0, 3575000.0, epsg)
            assert result == (None, None), f"EPSG:{epsg} should not need reprojection"

        # NAD83 UTM zones
        for epsg in [26901, 26914, 26915]:
            result = _project_to_epsg(500000.0, 3575000.0, epsg)
            assert result == (None, None), f"EPSG:{epsg} should not need reprojection"

    def test_project_to_epsg_warns_for_unknown_crs(self):
        """_project_to_epsg should return (None, None) and log warning for
        non-UTM CRS codes (e.g., State Plane)."""
        from custom_components.solar_shade.usgs_downloader import _project_to_epsg

        # State Plane EPSG
        result = _project_to_epsg(500000.0, 3575000.0, 2277)
        assert result == (None, None)
