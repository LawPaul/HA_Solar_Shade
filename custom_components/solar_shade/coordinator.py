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
    CONF_UPDATE_INTERVAL,
    DEFAULT_DIFFUSE_ENTITY,
    DEFAULT_DIFFUSE_FRACTION,
    DEFAULT_ENABLE_OPEN_METEO,
    DEFAULT_MIN_SHADOW_HEIGHT,
    DEFAULT_RADIATION_ENTITY,
    DEFAULT_UPDATE_INTERVAL,
    DOMAIN,
)
from .shadow_engine import SiteModel, compute_adjusted_radiation, compute_zone_shade_fractions, compute_zone_spot_windows

_LOGGER = logging.getLogger(__name__)


def _get_sun_position(hass: HomeAssistant) -> tuple[float, float]:
    """Get current sun azimuth and elevation from HA's sun tracking."""
    sun_state = hass.states.get("sun.sun")
    if sun_state is None:
        return 0.0, -90.0
    attrs = sun_state.attributes
    return float(attrs.get("azimuth", 0)), float(attrs.get("elevation", -90))


def _sample_day_sun_path(lat: float, lng: float, when, step_minutes: int = 60) -> list[tuple[float, float, float]]:
    """Sample (azimuth, elevation, weight) across the daylight hours of a day.

    Weight tracks clear-sky radiation on a horizontal surface (sin of elevation)
    so midday dominates, matching how solar exposure actually accumulates.
    """
    import math
    from datetime import datetime, timedelta
    from astral import Observer
    from astral.sun import azimuth as astral_az, elevation as astral_el

    obs = Observer(latitude=lat, longitude=lng, elevation=0)
    tz = when.tzinfo
    samples: list[tuple[float, float, float]] = []
    t = datetime(when.year, when.month, when.day, 0, 0, tzinfo=tz)
    end = t + timedelta(days=1)
    while t < end:
        try:
            el = float(astral_el(obs, t))
            az = float(astral_az(obs, t))
        except ValueError:
            t += timedelta(minutes=step_minutes)
            continue
        if el > 2.0:
            samples.append((az, el, math.sin(math.radians(el))))
        t += timedelta(minutes=step_minutes)
    return samples



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
        self.site = site
        self._spot_windows: dict[str, dict] | None = None
        self._spot_windows_day: int | None = None

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

        spot_windows = await self._get_spot_windows(day_of_year, canopy_model)

        shade_stats = await self.hass.async_add_executor_job(
            compute_zone_shade_fractions, self.site, azimuth, elevation,
            self._min_shadow_height, day_of_year, canopy_model, spot_windows,
        )

        result: dict[str, dict] = {}
        for zone in self.site.zones:
            stats = shade_stats.get(zone.zone_id, {"average": 0.0, "sunniest": 0.0, "shadiest": 0.0})
            method = getattr(zone, "shade_method", "average")
            shade = stats.get(method, stats["average"])
            adjusted = compute_adjusted_radiation(raw, shade, diffuse)
            result[zone.zone_id] = {
                "adjusted_radiation": adjusted,
                "raw_radiation": raw,
                "shade_fraction": shade,
                "shade_average": stats["average"],
                "shade_sunniest": stats["sunniest"],
                "shade_shadiest": stats["shadiest"],
                "shade_method": method,
                "spot_area": getattr(zone, "spot_area", 1.0),
                "diffuse_fraction": round(diffuse, 3),
                "zone_name": zone.zone_name,
                "sun_azimuth": round(azimuth, 1),
                "sun_elevation": round(elevation, 1),
            }

        return result

    async def _get_spot_windows(self, day_of_year: int, canopy_model: str) -> dict[str, dict] | None:
        """Fixed sunniest/shadiest patches, recomputed once per day from sun path."""
        needs_spots = any(
            getattr(z, "shade_method", "average") in ("sunniest", "shadiest")
            for z in self.site.zones
        )
        if not needs_spots:
            return None
        if self._spot_windows is not None and self._spot_windows_day == day_of_year:
            return self._spot_windows

        from homeassistant.util import dt as dt_util
        when = dt_util.now()
        samples = _sample_day_sun_path(self.site.latitude, self.site.longitude, when)
        windows = await self.hass.async_add_executor_job(
            compute_zone_spot_windows, self.site, samples,
            self._min_shadow_height, day_of_year, canopy_model,
        )
        self._spot_windows = windows
        self._spot_windows_day = day_of_year
        return windows
