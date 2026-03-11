"""Tests for Solar Shade __init__ module."""

import pytest
from unittest.mock import MagicMock

from custom_components.solar_shade import _get_location
from custom_components.solar_shade.const import CONF_LATITUDE, CONF_LONGITUDE


class TestGetLocation:
    """Test the _get_location helper."""

    def _make_hass(self, lat=40.0, lng=-105.0):
        hass = MagicMock()
        hass.config.latitude = lat
        hass.config.longitude = lng
        return hass

    def _make_entry(self, options=None):
        entry = MagicMock()
        entry.options = options or {}
        return entry

    def test_uses_ha_defaults_when_no_override(self):
        hass = self._make_hass(40.0, -105.0)
        entry = self._make_entry()
        lat, lng = _get_location(hass, entry)
        assert lat == 40.0
        assert lng == -105.0

    def test_uses_override_latitude(self):
        hass = self._make_hass(40.0, -105.0)
        entry = self._make_entry({CONF_LATITUDE: 35.5})
        lat, lng = _get_location(hass, entry)
        assert lat == 35.5
        assert lng == -105.0

    def test_uses_override_longitude(self):
        hass = self._make_hass(40.0, -105.0)
        entry = self._make_entry({CONF_LONGITUDE: -110.2})
        lat, lng = _get_location(hass, entry)
        assert lat == 40.0
        assert lng == -110.2

    def test_uses_both_overrides(self):
        hass = self._make_hass(40.0, -105.0)
        entry = self._make_entry({CONF_LATITUDE: 35.5, CONF_LONGITUDE: -110.2})
        lat, lng = _get_location(hass, entry)
        assert lat == 35.5
        assert lng == -110.2

    def test_empty_string_override_falls_back_to_ha(self):
        """Empty string should be falsy, fall back to HA config."""
        hass = self._make_hass(40.0, -105.0)
        entry = self._make_entry({CONF_LATITUDE: "", CONF_LONGITUDE: ""})
        lat, lng = _get_location(hass, entry)
        assert lat == 40.0
        assert lng == -105.0

    def test_zero_override_preserved(self):
        """Zero is a valid coordinate (equator / prime meridian).

        Explicit 0 should be kept, not treated as missing.
        """
        hass = self._make_hass(40.0, -105.0)
        entry = self._make_entry({CONF_LATITUDE: 0, CONF_LONGITUDE: 0})
        lat, lng = _get_location(hass, entry)
        assert lat == 0.0
        assert lng == 0.0

    def test_returns_floats(self):
        hass = self._make_hass(40, -105)
        entry = self._make_entry({CONF_LATITUDE: "35.5"})
        lat, lng = _get_location(hass, entry)
        assert isinstance(lat, float)
        assert isinstance(lng, float)
