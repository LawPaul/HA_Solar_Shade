"""Tests for Solar Shade coordinator."""

import pytest
from unittest.mock import MagicMock

from custom_components.solar_shade.coordinator import _get_sun_position


class TestGetSunPosition:
    """Test the _get_sun_position helper."""

    def test_returns_sun_attributes(self):
        hass = MagicMock()
        sun = MagicMock()
        sun.attributes = {"azimuth": 180.5, "elevation": 45.2}
        hass.states.get.return_value = sun
        az, el = _get_sun_position(hass)
        assert az == 180.5
        assert el == 45.2
        hass.states.get.assert_called_once_with("sun.sun")

    def test_returns_defaults_when_sun_missing(self):
        hass = MagicMock()
        hass.states.get.return_value = None
        az, el = _get_sun_position(hass)
        assert az == 0.0
        assert el == -90.0

    def test_handles_missing_attributes(self):
        hass = MagicMock()
        sun = MagicMock()
        sun.attributes = {}
        hass.states.get.return_value = sun
        az, el = _get_sun_position(hass)
        assert az == 0.0
        assert el == -90.0

