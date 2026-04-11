"""Open-Meteo solar radiation data coordinator.

Polls the Open-Meteo free API for current shortwave and diffuse radiation.
Creates two sensor entities that can be used as the radiation source for
the Solar Shade integration — no separate integration or YAML needed.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

import aiohttp

from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from .const import DOMAIN, OPEN_METEO_UPDATE_INTERVAL

_LOGGER = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


class OpenMeteoCoordinator(DataUpdateCoordinator):
    """Fetch current solar radiation from Open-Meteo API."""

    def __init__(self, hass: HomeAssistant, entry) -> None:
        super().__init__(
            hass,
            _LOGGER,
            name=f"{DOMAIN}_open_meteo",
            update_interval=timedelta(minutes=OPEN_METEO_UPDATE_INTERVAL),
        )
        from .const import CONF_LATITUDE, CONF_LONGITUDE
        self._lat = entry.options.get(CONF_LATITUDE) or hass.config.latitude
        self._lng = entry.options.get(CONF_LONGITUDE) or hass.config.longitude

    async def _async_update_data(self) -> dict[str, float]:
        """Fetch shortwave and diffuse radiation from Open-Meteo."""
        params = {
            "latitude": self._lat,
            "longitude": self._lng,
            "current": "shortwave_radiation,diffuse_radiation",
        }

        try:
            from homeassistant.helpers.aiohttp_client import async_get_clientsession
            session = async_get_clientsession(self.hass)
            async with session.get(
                OPEN_METEO_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    _LOGGER.warning("Open-Meteo API returned HTTP %d", resp.status)
                    return self.data or {"shortwave": 0.0, "diffuse": 0.0}

                data = await resp.json()
                current = data.get("current", {})
                return {
                    "shortwave": float(current.get("shortwave_radiation", 0)),
                    "diffuse": float(current.get("diffuse_radiation", 0)),
                }
        except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError) as err:
            _LOGGER.warning("Open-Meteo fetch failed: %s", err)
            return self.data or {"shortwave": 0.0, "diffuse": 0.0}
