"""Data update coordinator for Solar Shade integration."""

from __future__ import annotations

from datetime import timedelta
import logging

from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from .const import (
    CONF_DIFFUSE_ENTITY,
    CONF_DIFFUSE_FRACTION,
    CONF_ENABLE_OPEN_METEO,
    CONF_MIN_SHADOW_HEIGHT,
    CONF_RADIATION_ENTITY,
    CONF_SHADE_METHOD,
    CONF_UPDATE_INTERVAL,
    DEFAULT_DIFFUSE_ENTITY,
    DEFAULT_DIFFUSE_FRACTION,
    DEFAULT_ENABLE_OPEN_METEO,
    DEFAULT_MIN_SHADOW_HEIGHT,
    DEFAULT_RADIATION_ENTITY,
    DEFAULT_SHADE_METHOD,
    DEFAULT_UPDATE_INTERVAL,
    DOMAIN,
)
from .shadow_engine import SiteModel, compute_adjusted_radiation, compute_zone_shade_fractions

_LOGGER = logging.getLogger(__name__)


def _get_sun_position(hass: HomeAssistant) -> tuple[float, float]:
    """Get current sun azimuth and elevation from HA's sun tracking."""
    sun_state = hass.states.get("sun.sun")
    if sun_state is None:
        return 0.0, -90.0
    attrs = sun_state.attributes
    return float(attrs.get("azimuth", 0)), float(attrs.get("elevation", -90))


class SolarShadeCoordinator(DataUpdateCoordinator):
    """Coordinator that computes per-zone adjusted solar radiation live."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        site: SiteModel,
    ) -> None:
        interval = entry.options.get(CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL)
        super().__init__(
            hass, _LOGGER, name=DOMAIN, update_interval=timedelta(minutes=interval)
        )
        self._radiation_entity: str | None = entry.options.get(CONF_RADIATION_ENTITY)
        self._diffuse_entity: str | None = entry.options.get(CONF_DIFFUSE_ENTITY)

        # Auto-wire Open-Meteo entities when enabled and no custom entity is set
        open_meteo_enabled = entry.options.get(CONF_ENABLE_OPEN_METEO, DEFAULT_ENABLE_OPEN_METEO)
        if open_meteo_enabled:
            if not self._radiation_entity:
                self._radiation_entity = DEFAULT_RADIATION_ENTITY
            if not self._diffuse_entity:
                self._diffuse_entity = DEFAULT_DIFFUSE_ENTITY
        self._diffuse: float = entry.options.get(
            CONF_DIFFUSE_FRACTION, DEFAULT_DIFFUSE_FRACTION
        )
        self._min_shadow_height: float = entry.options.get(
            CONF_MIN_SHADOW_HEIGHT, DEFAULT_MIN_SHADOW_HEIGHT
        )
        self._shade_method: str = entry.options.get(
            CONF_SHADE_METHOD, DEFAULT_SHADE_METHOD
        )
        self.site = site

    async def _async_update_data(self) -> dict[str, dict]:
        """Read radiation source, compute live shadows, return adjusted values."""
        return await self._do_update()

    async def _do_update(self) -> dict[str, dict]:
        raw = 0.0
        if self._radiation_entity:
            state = self.hass.states.get(self._radiation_entity)
            if state is not None and state.state not in ("unknown", "unavailable"):
                try:
                    raw = float(state.state)
                except (ValueError, TypeError):
                    raw = 0.0

        # Compute dynamic diffuse fraction from entity if available
        diffuse = self._diffuse  # static fallback
        if self._diffuse_entity and raw > 0:
            diff_state = self.hass.states.get(self._diffuse_entity)
            if diff_state is None:
                if not getattr(self, '_diffuse_warned', False):
                    _LOGGER.warning(
                        "Diffuse radiation entity '%s' not found — using static fraction %.0f%%",
                        self._diffuse_entity, self._diffuse * 100,
                    )
                    self._diffuse_warned = True
            elif diff_state.state in ("unknown", "unavailable"):
                pass  # sensor temporarily unavailable, use static fallback
            else:
                try:
                    diffuse_rad = float(diff_state.state)
                    # diffuse_fraction = diffuse_radiation / total_radiation
                    diffuse = min(1.0, max(0.0, diffuse_rad / raw))
                except (ValueError, TypeError, ZeroDivisionError):
                    _LOGGER.debug(
                        "Could not parse diffuse entity '%s' state '%s'",
                        self._diffuse_entity, diff_state.state,
                    )

        azimuth, elevation = _get_sun_position(self.hass)

        from homeassistant.util import dt as dt_util
        day_of_year = dt_util.now().timetuple().tm_yday

        from .const import CONF_CANOPY_MODEL, DEFAULT_CANOPY_MODEL
        canopy_model = self.config_entry.options.get(CONF_CANOPY_MODEL, DEFAULT_CANOPY_MODEL)

        shade_stats = await self.hass.async_add_executor_job(
            compute_zone_shade_fractions, self.site, azimuth, elevation,
            self._min_shadow_height, day_of_year, canopy_model,
        )

        result: dict[str, dict] = {}
        for zone in self.site.zones:
            stats = shade_stats.get(zone.zone_id, {"average": 0.0, "sunniest": 0.0, "shadiest": 0.0})
            shade = stats.get(self._shade_method, stats["average"])
            adjusted = compute_adjusted_radiation(raw, shade, diffuse)
            result[zone.zone_id] = {
                "adjusted_radiation": adjusted,
                "raw_radiation": raw,
                "shade_fraction": shade,
                "shade_average": stats["average"],
                "shade_sunniest": stats["sunniest"],
                "shade_shadiest": stats["shadiest"],
                "diffuse_fraction": round(diffuse, 3),
                "zone_name": zone.zone_name,
                "sun_azimuth": round(azimuth, 1),
                "sun_elevation": round(elevation, 1),
            }

        return result
