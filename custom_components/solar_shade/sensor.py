"""Sensor platform for Solar Shade integration.

Creates one sensor per zone, outputting shadow-adjusted solar radiation (W/m²).
"""

from __future__ import annotations

import logging

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    ATTR_RAW_RADIATION,
    ATTR_SHADE_FRACTION,
    ATTR_SUN_AZIMUTH,
    ATTR_SUN_ELEVATION,
    ATTR_ZONE_NAME,
    CONF_ZONES,
    DOMAIN,
)
from .coordinator import SolarShadeCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    """Set up Solar Shade sensors from a config entry."""
    coordinator: SolarShadeCoordinator = hass.data[DOMAIN][entry.entry_id][
        "coordinator"
    ]

    # Build sensors from config options (always available) rather than
    # site.zones, which may be empty when using a placeholder site during
    # background LiDAR download.
    zones = entry.options.get(CONF_ZONES, [])

    entities = []
    # Radiation sensors — always created for each zone.
    # They report shadow-adjusted W/m² and return unavailable when
    # no radiation source is available.
    entities += [
        SolarShadeSensor(coordinator, zone["id"], zone["name"])
        for zone in zones
    ]
    # Sun percentage sensors — always created
    entities += [
        SolarShadeSunSensor(coordinator, zone["id"], zone["name"])
        for zone in zones
    ]

    # Add Open-Meteo radiation sensors if enabled
    om_coordinator = hass.data[DOMAIN][entry.entry_id].get("open_meteo")
    if om_coordinator:
        entities.append(OpenMeteoShortwaveSensor(om_coordinator))
        entities.append(OpenMeteoDiffuseSensor(om_coordinator))

    async_add_entities(entities)


class SolarShadeSensor(CoordinatorEntity, SensorEntity):
    """Sensor that reports shadow-adjusted solar radiation for a zone."""

    _attr_device_class = SensorDeviceClass.IRRADIANCE
    _attr_native_unit_of_measurement = "W/m²"
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(
        self,
        coordinator: SolarShadeCoordinator,
        zone_id: str,
        zone_name: str,
    ) -> None:
        super().__init__(coordinator)
        self._zone_id = zone_id
        self._attr_name = f"Solar Shade {zone_name}"
        self._attr_unique_id = f"{DOMAIN}_{zone_id}"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, "solar_shade")},
            name="Solar Shade",
            manufacturer="Solar Shade",
            model="Shadow Analysis",
        )

    @property
    def native_value(self) -> float | None:
        """Return the adjusted radiation for this zone."""
        if self.coordinator.data and self._zone_id in self.coordinator.data:
            return self.coordinator.data[self._zone_id]["adjusted_radiation"]
        return None

    @property
    def extra_state_attributes(self) -> dict | None:
        """Return additional state attributes."""
        if self.coordinator.data and self._zone_id in self.coordinator.data:
            data = self.coordinator.data[self._zone_id]
            return {
                ATTR_ZONE_NAME: data["zone_name"],
                ATTR_SHADE_FRACTION: data["shade_fraction"],
                ATTR_RAW_RADIATION: data["raw_radiation"],
                ATTR_SUN_AZIMUTH: data.get("sun_azimuth"),
                ATTR_SUN_ELEVATION: data.get("sun_elevation"),
                "shade_average": data.get("shade_average"),
                "shade_sunniest": data.get("shade_sunniest"),
                "shade_shadiest": data.get("shade_shadiest"),
                "diffuse_fraction": data.get("diffuse_fraction"),
            }
        return None


class SolarShadeSunSensor(CoordinatorEntity, SensorEntity):
    """Sensor that reports sun exposure percentage (0-100%) for a zone.

    100% = full sun, 0% = complete shade.
    Useful for blind automations, garden planning, or any automation
    that needs a simple sun/shade value without radiation units.
    """

    _attr_native_unit_of_measurement = "%"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:white-balance-sunny"

    def __init__(
        self,
        coordinator: SolarShadeCoordinator,
        zone_id: str,
        zone_name: str,
    ) -> None:
        super().__init__(coordinator)
        self._zone_id = zone_id
        self._attr_name = f"Solar Shade {zone_name} Sun"
        self._attr_unique_id = f"{DOMAIN}_{zone_id}_sun_pct"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, "solar_shade")},
            name="Solar Shade",
            manufacturer="Solar Shade",
            model="Shadow Analysis",
        )

    @property
    def native_value(self) -> int | None:
        """Return sun exposure percentage (100 = full sun, 0 = full shade)."""
        if self.coordinator.data and self._zone_id in self.coordinator.data:
            shade = self.coordinator.data[self._zone_id]["shade_fraction"]
            return round((1.0 - shade) * 100)
        return None

    @property
    def extra_state_attributes(self) -> dict | None:
        """Return additional state attributes."""
        if self.coordinator.data and self._zone_id in self.coordinator.data:
            data = self.coordinator.data[self._zone_id]
            return {
                ATTR_ZONE_NAME: data["zone_name"],
                ATTR_SHADE_FRACTION: data["shade_fraction"],
                ATTR_SUN_AZIMUTH: data.get("sun_azimuth"),
                ATTR_SUN_ELEVATION: data.get("sun_elevation"),
            }
        return None


class OpenMeteoShortwaveSensor(CoordinatorEntity, SensorEntity):
    """Shortwave (total) solar radiation from Open-Meteo."""

    _attr_device_class = SensorDeviceClass.IRRADIANCE
    _attr_native_unit_of_measurement = "W/m²"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:weather-sunny"
    _attr_name = "Solar Shade Shortwave Radiation"
    _attr_unique_id = f"{DOMAIN}_open_meteo_shortwave"
    _attr_device_info = DeviceInfo(
        identifiers={(DOMAIN, "solar_shade")},
        name="Solar Shade",
        manufacturer="Solar Shade",
        model="Shadow Analysis",
    )

    @property
    def native_value(self) -> float | None:
        if self.coordinator.data:
            return self.coordinator.data.get("shortwave", 0)
        return None


class OpenMeteoDiffuseSensor(CoordinatorEntity, SensorEntity):
    """Diffuse solar radiation from Open-Meteo."""

    _attr_device_class = SensorDeviceClass.IRRADIANCE
    _attr_native_unit_of_measurement = "W/m²"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:weather-partly-cloudy"
    _attr_name = "Solar Shade Diffuse Radiation"
    _attr_unique_id = f"{DOMAIN}_open_meteo_diffuse"
    _attr_device_info = DeviceInfo(
        identifiers={(DOMAIN, "solar_shade")},
        name="Solar Shade",
        manufacturer="Solar Shade",
        model="Shadow Analysis",
    )

    @property
    def native_value(self) -> float | None:
        if self.coordinator.data:
            return self.coordinator.data.get("diffuse", 0)
        return None
