"""Tests for the USGS downloader module."""

import numpy as np
import pytest

from custom_components.solar_shade.usgs_downloader import (
    _latlon_to_utm,
)


class TestLatLonToUTM:
    """Test lat/lon to UTM conversion for tile matching."""

    def test_east_texas(self):
        """User's location in East Texas should be UTM zone 15."""
        zone, easting, northing, is_northern = _latlon_to_utm(32.28, -95.28)
        assert zone == 15
        assert is_northern is True
        # Easting should be near 300000-400000 for western part of zone 15
        assert 200000 < easting < 800000
        # Northing should be around 3.5 million for ~32° latitude
        assert 3500000 < northing < 3700000

    def test_southern_hemisphere(self):
        zone, easting, northing, is_northern = _latlon_to_utm(-33.86, 151.21)
        assert is_northern is False
        assert northing > 0  # includes 10M offset

    def test_equator(self):
        zone, easting, northing, is_northern = _latlon_to_utm(0.0, -80.0)
        assert is_northern is True
        assert -1 < northing < 100000  # At equator, northing ≈ 0


class TestDTMTileNaming:
    """Test DTM download parameters."""

    def test_bbox_computation(self):
        """Verify bbox is computed correctly for ImageServer request."""
        import math

        lat, lon = 32.28, -95.28
        radius_m = 150.0
        m_per_deg_lat = 111320.0
        m_per_deg_lng = 111320.0 * math.cos(math.radians(lat))
        margin = radius_m * 1.2
        dlat = margin / m_per_deg_lat
        dlng = margin / m_per_deg_lng

        bbox_str = (
            f"{lon - dlng:.6f},{lat - dlat:.6f},"
            f"{lon + dlng:.6f},{lat + dlat:.6f}"
        )

        parts = [float(x) for x in bbox_str.split(",")]
        # West < East, South < North
        assert parts[0] < parts[2]
        assert parts[1] < parts[3]
        # Extent should be about 360m in each direction (150 * 1.2 * 2)
        extent_lat_m = (parts[3] - parts[1]) * m_per_deg_lat
        extent_lon_m = (parts[2] - parts[0]) * m_per_deg_lng
        assert 300 < extent_lat_m < 400
        assert 300 < extent_lon_m < 400
