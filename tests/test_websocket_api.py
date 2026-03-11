"""Tests for Solar Shade websocket API utility functions."""

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from custom_components.solar_shade.shadow_engine import SiteModel
from custom_components.solar_shade.websocket_api import (
    _calc_sun_position,
    _site_bounds_latlng,
)


class TestSiteBoundsLatlng:
    """Test the flat-earth site bounds calculation."""

    def _make_site(self, lat=40.0, lng=-105.0, x_range=(-50, 50), y_range=(-50, 50)):
        dsm = np.zeros((10, 10), dtype=np.float32)
        site = SiteModel(
            dsm=dsm, resolution=1.0, latitude=lat, longitude=lng,
        )
        site.x_min_m = x_range[0]
        site.x_max_m = x_range[1]
        site.y_min_m = y_range[0]
        site.y_max_m = y_range[1]
        return site

    def test_returns_dict_with_four_keys(self):
        site = self._make_site()
        bounds = _site_bounds_latlng(site)
        assert set(bounds.keys()) == {"south", "north", "west", "east"}

    def test_symmetric_bounds(self):
        site = self._make_site(lat=40.0, lng=-105.0, x_range=(-100, 100), y_range=(-100, 100))
        bounds = _site_bounds_latlng(site)
        assert bounds["south"] < 40.0
        assert bounds["north"] > 40.0
        assert bounds["west"] < -105.0
        assert bounds["east"] > -105.0

    def test_north_south_symmetry(self):
        site = self._make_site(lat=45.0, lng=0.0, x_range=(-100, 100), y_range=(-100, 100))
        bounds = _site_bounds_latlng(site)
        offset_south = 45.0 - bounds["south"]
        offset_north = bounds["north"] - 45.0
        assert abs(offset_south - offset_north) < 1e-10

    def test_east_west_wider_at_equator(self):
        equator = self._make_site(lat=0.0, lng=0.0, x_range=(-100, 100), y_range=(-100, 100))
        high_lat = self._make_site(lat=60.0, lng=0.0, x_range=(-100, 100), y_range=(-100, 100))
        eq_bounds = _site_bounds_latlng(equator)
        hi_bounds = _site_bounds_latlng(high_lat)
        eq_width = eq_bounds["east"] - eq_bounds["west"]
        hi_width = hi_bounds["east"] - hi_bounds["west"]
        # At higher latitude, same meter distance spans more degrees
        assert hi_width > eq_width

    def test_zero_range_returns_point(self):
        site = self._make_site(lat=40.0, lng=-105.0, x_range=(0, 0), y_range=(0, 0))
        bounds = _site_bounds_latlng(site)
        assert bounds["south"] == pytest.approx(40.0, abs=1e-6)
        assert bounds["north"] == pytest.approx(40.0, abs=1e-6)
        assert bounds["west"] == pytest.approx(-105.0, abs=1e-6)
        assert bounds["east"] == pytest.approx(-105.0, abs=1e-6)


class TestCalcSunPosition:
    """Test the sun position calculation."""

    def test_returns_tuple_of_two_floats(self):
        when = datetime(2025, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
        az, el = _calc_sun_position(40.0, -105.0, when)
        assert isinstance(az, float)
        assert isinstance(el, float)

    def test_noon_summer_solstice_high_elevation(self):
        """At solar noon on summer solstice, sun should be high."""
        # Denver, CO at ~18:00 UTC (noon local MDT)
        when = datetime(2025, 6, 21, 18, 0, 0, tzinfo=timezone.utc)
        az, el = _calc_sun_position(40.0, -105.0, when)
        assert el > 50  # Should be well above horizon at solar noon in summer

    def test_midnight_low_elevation(self):
        """At midnight, sun should be below horizon."""
        when = datetime(2025, 6, 21, 6, 0, 0, tzinfo=timezone.utc)  # midnight MDT
        az, el = _calc_sun_position(40.0, -105.0, when)
        assert el < 0

    def test_azimuth_range(self):
        """Azimuth should be in 0-360 range."""
        when = datetime(2025, 6, 21, 18, 0, 0, tzinfo=timezone.utc)
        az, el = _calc_sun_position(40.0, -105.0, when)
        assert 0 <= az <= 360

    def test_different_locations_different_results(self):
        """Two distant locations should have different sun positions."""
        when = datetime(2025, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
        az1, el1 = _calc_sun_position(40.0, -105.0, when)
        az2, el2 = _calc_sun_position(-33.9, 18.4, when)  # Cape Town
        # At least one should differ significantly
        assert abs(az1 - az2) > 10 or abs(el1 - el2) > 10
