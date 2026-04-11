"""Tests for Solar Shade sensor platform.

Verifies that sensors are created from config options (not site.zones),
and that sensor values behave correctly with and without coordinator data.
"""

import asyncio
from unittest.mock import MagicMock, AsyncMock

import pytest

from custom_components.solar_shade.sensor import (
    SolarShadeSensor,
    SolarShadeSunSensor,
    async_setup_entry,
)
from custom_components.solar_shade.const import CONF_ZONES, DOMAIN


def _make_coordinator(data=None):
    """Create a mock coordinator with optional data."""
    coord = MagicMock()
    coord.data = data
    return coord


def _make_hass_and_entry(zones, coordinator, site_zones=None):
    """Build mock hass and entry objects.

    zones: list of dicts (config options format, e.g. [{"id": "z1", "name": "Zone 1"}])
    site_zones: list of ZoneDef-like objects on the site (defaults to empty)
    """
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.options = {CONF_ZONES: zones}

    site = MagicMock()
    site.zones = site_zones or []

    hass = MagicMock()
    hass.data = {
        DOMAIN: {
            entry.entry_id: {
                "coordinator": coordinator,
                "site": site,
            }
        }
    }
    return hass, entry


class TestAsyncSetupEntry:
    """Test that async_setup_entry creates sensors from config options."""

    def test_creates_sensors_from_config_zones(self):
        """Sensors should be created from entry.options zones, not site.zones."""
        coordinator = _make_coordinator()
        zones = [
            {"id": "z1", "name": "Front Yard", "polygon": []},
            {"id": "z2", "name": "Back Yard", "polygon": []},
        ]
        hass, entry = _make_hass_and_entry(zones, coordinator)

        added_entities = []
        async_add_entities = MagicMock(side_effect=lambda e: added_entities.extend(e))

        asyncio.run(async_setup_entry(hass, entry, async_add_entities))

        async_add_entities.assert_called_once()
        # 2 radiation + 2 sun = 4 entities
        assert len(added_entities) == 4
        # Check types: first 2 are SolarShadeSensor, next 2 are SolarShadeSunSensor
        assert isinstance(added_entities[0], SolarShadeSensor)
        assert isinstance(added_entities[1], SolarShadeSensor)
        assert isinstance(added_entities[2], SolarShadeSunSensor)
        assert isinstance(added_entities[3], SolarShadeSunSensor)

    def test_creates_sensors_with_empty_site_zones(self):
        """Sensors should still be created even if site.zones is empty (placeholder site)."""
        coordinator = _make_coordinator()
        zones = [{"id": "z1", "name": "Garden", "polygon": []}]
        # site has NO zones (placeholder), but config has one
        hass, entry = _make_hass_and_entry(zones, coordinator, site_zones=[])

        added_entities = []
        async_add_entities = MagicMock(side_effect=lambda e: added_entities.extend(e))

        asyncio.run(async_setup_entry(hass, entry, async_add_entities))

        assert len(added_entities) == 2  # 1 radiation + 1 sun

    def test_no_zones_creates_no_sensors(self):
        """No zones in config → no zone sensors (only Open-Meteo if enabled)."""
        coordinator = _make_coordinator()
        hass, entry = _make_hass_and_entry([], coordinator)

        added_entities = []
        async_add_entities = MagicMock(side_effect=lambda e: added_entities.extend(e))

        asyncio.run(async_setup_entry(hass, entry, async_add_entities))

        assert len(added_entities) == 0

    def test_sensor_zone_ids_match_config(self):
        """Sensor zone IDs should come from config options, not site model."""
        coordinator = _make_coordinator()
        zones = [
            {"id": "abc123", "name": "Patio", "polygon": []},
            {"id": "def456", "name": "Driveway", "polygon": []},
        ]
        hass, entry = _make_hass_and_entry(zones, coordinator)

        added_entities = []
        async_add_entities = MagicMock(side_effect=lambda e: added_entities.extend(e))

        asyncio.run(async_setup_entry(hass, entry, async_add_entities))

        radiation_sensors = [e for e in added_entities if isinstance(e, SolarShadeSensor)]
        assert radiation_sensors[0]._zone_id == "abc123"
        assert radiation_sensors[1]._zone_id == "def456"


class TestSolarShadeSensor:
    """Test SolarShadeSensor value behavior."""

    def test_native_value_returns_none_without_data(self):
        coordinator = _make_coordinator(data=None)
        sensor = SolarShadeSensor(coordinator, "z1", "Zone 1")
        assert sensor.native_value is None

    def test_native_value_returns_none_for_missing_zone(self):
        coordinator = _make_coordinator(data={"other_zone": {}})
        sensor = SolarShadeSensor(coordinator, "z1", "Zone 1")
        assert sensor.native_value is None

    def test_native_value_returns_adjusted_radiation(self):
        coordinator = _make_coordinator(data={
            "z1": {
                "adjusted_radiation": 450.0,
                "raw_radiation": 600.0,
                "shade_fraction": 0.25,
                "zone_name": "Zone 1",
            }
        })
        sensor = SolarShadeSensor(coordinator, "z1", "Zone 1")
        assert sensor.native_value == 450.0

    def test_extra_state_attributes_with_data(self):
        coordinator = _make_coordinator(data={
            "z1": {
                "adjusted_radiation": 450.0,
                "raw_radiation": 600.0,
                "shade_fraction": 0.25,
                "zone_name": "Zone 1",
                "sun_azimuth": 180.0,
                "sun_elevation": 45.0,
                "shade_average": 0.25,
                "shade_sunniest": 0.1,
                "shade_shadiest": 0.4,
                "diffuse_fraction": 0.3,
            }
        })
        sensor = SolarShadeSensor(coordinator, "z1", "Zone 1")
        attrs = sensor.extra_state_attributes
        assert attrs is not None
        assert attrs["shade_fraction"] == 0.25
        assert attrs["raw_radiation"] == 600.0

    def test_extra_state_attributes_none_without_data(self):
        coordinator = _make_coordinator(data=None)
        sensor = SolarShadeSensor(coordinator, "z1", "Zone 1")
        assert sensor.extra_state_attributes is None


class TestSolarShadeSunSensor:
    """Test SolarShadeSunSensor percentage calculation."""

    def test_full_sun_returns_100(self):
        coordinator = _make_coordinator(data={
            "z1": {"shade_fraction": 0.0, "zone_name": "Zone 1"}
        })
        sensor = SolarShadeSunSensor(coordinator, "z1", "Zone 1")
        assert sensor.native_value == 100

    def test_full_shade_returns_0(self):
        coordinator = _make_coordinator(data={
            "z1": {"shade_fraction": 1.0, "zone_name": "Zone 1"}
        })
        sensor = SolarShadeSunSensor(coordinator, "z1", "Zone 1")
        assert sensor.native_value == 0

    def test_half_shade_returns_50(self):
        coordinator = _make_coordinator(data={
            "z1": {"shade_fraction": 0.5, "zone_name": "Zone 1"}
        })
        sensor = SolarShadeSunSensor(coordinator, "z1", "Zone 1")
        assert sensor.native_value == 50

    def test_native_value_returns_none_without_data(self):
        coordinator = _make_coordinator(data=None)
        sensor = SolarShadeSunSensor(coordinator, "z1", "Zone 1")
        assert sensor.native_value is None

    def test_unique_id_has_sun_pct_suffix(self):
        coordinator = _make_coordinator()
        sensor = SolarShadeSunSensor(coordinator, "z1", "Zone 1")
        assert sensor._attr_unique_id == f"{DOMAIN}_z1_sun_pct"
