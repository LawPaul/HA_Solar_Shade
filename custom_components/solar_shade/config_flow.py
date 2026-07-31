"""Config flow for Solar Shade integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers import selector

from .const import (
    CONF_DIFFUSE_ENTITY,
    CONF_DIFFUSE_FRACTION,
    CONF_DOWNLOAD_RADIUS,
    CONF_DSM_PROVIDER,
    CONF_DSM_SOURCE,
    CONF_DSM_GAP_FILL,
    CONF_ENABLE_OPEN_METEO,
    CONF_LATITUDE,
    CONF_LIDAR_FILE,
    CONF_LIDAR_PROJECT,
    CONF_LONGITUDE,
    CONF_MANUAL_EPSG,
    CONF_MIN_SHADOW_HEIGHT,
    CONF_CANOPY_MODEL,
    CONF_MIN_CELL_SIZE,
    CONF_RADIATION_ENTITY,

    CONF_UPDATE_INTERVAL,
    CONF_ZONES,
    DATA_DIR,
    DEFAULT_CANOPY_MODEL,
    DEFAULT_DIFFUSE_FRACTION,
    DEFAULT_DOWNLOAD_RADIUS,
    DEFAULT_DSM_GAP_FILL,
    DEFAULT_DSM_PROVIDER,
    DEFAULT_ENABLE_OPEN_METEO,
    DEFAULT_MANUAL_EPSG,
    DEFAULT_MIN_CELL_SIZE,
    DEFAULT_MIN_SHADOW_HEIGHT,
    DEFAULT_UPDATE_INTERVAL,
    DOMAIN,
    DSM_PROVIDER_AUTO,
    DSM_PROVIDER_IGN,
    DSM_PROVIDER_NRW,
    DSM_PROVIDER_PDOK,
    DSM_PROVIDER_SWISSTOPO,
    DSM_PROVIDER_USGS,
    DSM_SOURCE_AUTO,
    DSM_SOURCE_LAZ,
    CANOPY_MODEL_RAISED,
    CANOPY_MODEL_SOLID,
)


class SolarShadeConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Solar Shade."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        """Single-step setup: just create the entry with LiDAR mode."""
        # Only allow one instance
        if self._async_current_entries():
            return self.async_abort(reason="single_instance_allowed")

        if user_input is not None:
            return self.async_create_entry(
                title="Solar Shade",
                data={},
                options={
                    CONF_ZONES: [],
                },
            )

        # No form needed — confirm and go
        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({}),
            description_placeholders={
                "info": "Solar Shade will download LiDAR elevation data for your property and create shadow sensors. All settings can be configured from the Solar Shade panel after setup."
            },
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        """Get the options flow handler."""
        return SolarShadeOptionsFlow(config_entry)


class SolarShadeOptionsFlow(config_entries.OptionsFlow):
    """Handle options for Solar Shade."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        self._config_entry = config_entry

    async def async_step_init(self, user_input=None):
        """Single settings pane for all options."""
        return await self.async_step_settings(user_input)

    async def async_step_select_laz(self, user_input=None):
        """Select a manually placed LAZ file from config/solar_shade/."""
        from .shadow_engine import find_lidar_files

        data_dir = self.hass.config.path(DATA_DIR)
        files = await self.hass.async_add_executor_job(find_lidar_files, data_dir)

        if not files:
            return self.async_abort(reason="no_laz_files")

        if user_input is not None:
            # Delete cached DSM to force reprocessing
            npz = Path(data_dir) / "site_dsm.npz"
            if npz.exists():
                npz.unlink()
            # Merge with pending settings if any
            merged = getattr(self, '_pending_settings', {})
            merged[CONF_DSM_SOURCE] = DSM_SOURCE_LAZ
            merged[CONF_LIDAR_FILE] = user_input[CONF_LIDAR_FILE]
            self._pending_settings = None
            return self._save_options(merged)

        current = self._config_entry.options.get(CONF_LIDAR_FILE, "")

        schema = vol.Schema(
            {
                vol.Required(
                    CONF_LIDAR_FILE,
                    default=current if current in files else files[0],
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            selector.SelectOptionDict(value=f, label=f)
                            for f in files
                        ],
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
            }
        )

        return self.async_show_form(step_id="select_laz", data_schema=schema)

    # ── Settings ─────────────────────────────────────────────────────────

    async def async_step_settings(self, user_input=None):
        """Edit general settings."""
        errors = {}
        if user_input is not None:
            # Convert lat/long text to float, remove if blank
            for key, lo, hi in [
                (CONF_LATITUDE, -90, 90),
                (CONF_LONGITUDE, -180, 180),
            ]:
                val = user_input.get(key, "")
                if isinstance(val, str):
                    val = val.strip()
                if val:
                    try:
                        fval = float(val)
                        if not lo <= fval <= hi:
                            errors[key] = "invalid_coordinates"
                        else:
                            user_input[key] = fval
                    except ValueError:
                        errors[key] = "invalid_coordinates"
                else:
                    user_input.pop(key, None)
            if not errors:
                # If DSM source is LAZ, redirect to file selection
                if user_input.get(CONF_DSM_SOURCE) == DSM_SOURCE_LAZ:
                    # Save other settings first, then go to LAZ file picker
                    self._pending_settings = user_input
                    return await self.async_step_select_laz()
                return self._save_options(user_input)

        current = self._config_entry.options

        schema = vol.Schema(
            {
                vol.Optional(
                    CONF_ENABLE_OPEN_METEO,
                    default=current.get(CONF_ENABLE_OPEN_METEO, DEFAULT_ENABLE_OPEN_METEO),
                ): selector.BooleanSelector(),
                vol.Optional(
                    CONF_RADIATION_ENTITY,
                    description={"suggested_value": current.get(CONF_RADIATION_ENTITY, "")},
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain="sensor"),
                ),
                vol.Optional(
                    CONF_DIFFUSE_ENTITY,
                    description={"suggested_value": current.get(CONF_DIFFUSE_ENTITY, "")},
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain="sensor"),
                ),
                vol.Optional(
                    CONF_LATITUDE,
                    description={"suggested_value": current.get(CONF_LATITUDE, "")},
                ): selector.TextSelector(
                    selector.TextSelectorConfig(
                        type=selector.TextSelectorType.TEXT,
                    )
                ),
                vol.Optional(
                    CONF_LONGITUDE,
                    description={"suggested_value": current.get(CONF_LONGITUDE, "")},
                ): selector.TextSelector(
                    selector.TextSelectorConfig(
                        type=selector.TextSelectorType.TEXT,
                    )
                ),
                vol.Optional(
                    CONF_DIFFUSE_FRACTION,
                    default=current.get(CONF_DIFFUSE_FRACTION, DEFAULT_DIFFUSE_FRACTION),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0.0, max=1.0, step=0.01, mode="slider"
                    )
                ),
                vol.Optional(
                    CONF_MIN_CELL_SIZE,
                    default=current.get(CONF_MIN_CELL_SIZE, DEFAULT_MIN_CELL_SIZE),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0.5, max=5.0, step=0.5,
                        unit_of_measurement="m",
                        mode="box",
                    )
                ),
                vol.Optional(
                    CONF_UPDATE_INTERVAL,
                    default=current.get(CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=1, max=60, step=1,
                        unit_of_measurement="min",
                        mode="box",
                    )
                ),
                vol.Optional(
                    CONF_DOWNLOAD_RADIUS,
                    default=current.get(CONF_DOWNLOAD_RADIUS, DEFAULT_DOWNLOAD_RADIUS),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=50, max=500, step=25,
                        unit_of_measurement="m",
                        mode="box",
                    )
                ),
                vol.Optional(
                    CONF_MIN_SHADOW_HEIGHT,
                    default=current.get(CONF_MIN_SHADOW_HEIGHT, DEFAULT_MIN_SHADOW_HEIGHT),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0.1, max=5.0, step=0.1,
                        unit_of_measurement="m",
                        mode="box",
                    )
                ),
                vol.Optional(
                    CONF_CANOPY_MODEL,
                    default=current.get(CONF_CANOPY_MODEL, DEFAULT_CANOPY_MODEL),
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            selector.SelectOptionDict(value=CANOPY_MODEL_SOLID, label="Solid column (simple, treats trees as walls)"),
                            selector.SelectOptionDict(value=CANOPY_MODEL_RAISED, label="Raised canopy (sun passes under tree trunks)"),
                        ],
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Optional(
                    CONF_DSM_GAP_FILL,
                    default=current.get(CONF_DSM_GAP_FILL, DEFAULT_DSM_GAP_FILL),
                ): selector.BooleanSelector(),
                vol.Optional(
                    CONF_DSM_PROVIDER,
                    default=current.get(CONF_DSM_PROVIDER, DEFAULT_DSM_PROVIDER),
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            selector.SelectOptionDict(value=DSM_PROVIDER_AUTO, label="Auto-detect from location"),
                            selector.SelectOptionDict(value=DSM_PROVIDER_USGS, label="USGS 3DEP (United States)"),
                            selector.SelectOptionDict(value=DSM_PROVIDER_IGN, label="IGN LiDAR HD (France)"),
                            selector.SelectOptionDict(value=DSM_PROVIDER_SWISSTOPO, label="swisstopo swissSURFACE3D (Switzerland)"),
                            selector.SelectOptionDict(value=DSM_PROVIDER_NRW, label="NRW 3D-Messdaten (Germany)"),
                            selector.SelectOptionDict(value=DSM_PROVIDER_PDOK, label="PDOK 3D (Netherlands)"),
                        ],
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Optional(
                    CONF_DSM_SOURCE,
                    default=current.get(CONF_DSM_SOURCE, DSM_SOURCE_AUTO),
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            selector.SelectOptionDict(value=DSM_SOURCE_AUTO, label="Auto-download"),
                            selector.SelectOptionDict(value=DSM_SOURCE_LAZ, label="Manual LAZ file"),
                        ],
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Optional(
                    CONF_LIDAR_PROJECT,
                    description={"suggested_value": current.get(CONF_LIDAR_PROJECT, "")},
                ): selector.TextSelector(
                    selector.TextSelectorConfig(
                        type=selector.TextSelectorType.TEXT,
                    )
                ),
                vol.Optional(
                    CONF_MANUAL_EPSG,
                    default=current.get(CONF_MANUAL_EPSG, DEFAULT_MANUAL_EPSG),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, max=99999, step=1,
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
            }
        )

        return self.async_show_form(
            step_id="settings", data_schema=schema, errors=errors,
        )

    # ── Helper ───────────────────────────────────────────────────────────

    def _save_options(self, updates: dict[str, Any]):
        """Merge updates into current options and save."""
        new_options = {**self._config_entry.options, **updates}
        return self.async_create_entry(title="", data=new_options)
