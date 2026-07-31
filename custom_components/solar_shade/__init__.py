"""Solar Shade integration for Home Assistant.

Provides per-zone shadow-adjusted solar radiation sensors by ray-tracing
a DSM (Digital Surface Model) against the live sun position.

Auto-downloads DSM from USGS 3DEP if available, or reads a manually
provided LAZ file.
"""

from __future__ import annotations

import logging
from pathlib import Path

from homeassistant.config_entries import ConfigEntry
from homeassistant.components import persistent_notification
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv

CONFIG_SCHEMA = cv.config_entry_only_config_schema("solar_shade")

from .const import (
    CONF_DOWNLOAD_RADIUS,
    CONF_DSM_GAP_FILL,
    CONF_DSM_PROVIDER,
    CONF_DSM_SOURCE,
    CONF_ENABLE_OPEN_METEO,
    CONF_LATITUDE,
    CONF_LIDAR_FILE,
    CONF_LIDAR_PROJECT,
    CONF_LONGITUDE,
    CONF_MANUAL_EPSG,
    CONF_MIN_CELL_SIZE,
    CONF_ZONES,
    DATA_DIR,
    DEFAULT_DOWNLOAD_RADIUS,
    DEFAULT_DSM_GAP_FILL,
    DEFAULT_DSM_PROVIDER,
    DEFAULT_ENABLE_OPEN_METEO,
    DEFAULT_MANUAL_EPSG,
    DEFAULT_MIN_CELL_SIZE,
    DEFAULT_RESOLUTION,
    DOMAIN,
    DSM_PROVIDER_AUTO,
    DSM_SOURCE_AUTO,
    DSM_SOURCE_LAZ,
    SERVICE_RELOAD_SITE,
)
from .coordinator import SolarShadeCoordinator
from .shadow_engine import (
    SiteModel,
    apply_eraser_to_site,
    apply_zones_to_site,
    load_eraser_mask,
    load_site_model,
    save_processed_dsm,
)

_LOGGER = logging.getLogger(__name__)

PLATFORMS = ["sensor"]


def _get_location(hass: HomeAssistant, entry: ConfigEntry) -> tuple[float, float]:
    """Return (latitude, longitude), preferring manual overrides."""
    lat = entry.options.get(CONF_LATITUDE)
    lng = entry.options.get(CONF_LONGITUDE)
    if lat is None or lat == "":
        lat = hass.config.latitude
    if lng is None or lng == "":
        lng = hass.config.longitude
    return float(lat), float(lng)


async def _build_site_LiDAR(hass: HomeAssistant, entry: ConfigEntry) -> SiteModel:
    """Build site for LiDAR mode.

    Priority:
    1. Load cached DSM from disk (instant restart)
    2. Return placeholder immediately; download starts in background
    3. Process manual LAZ file (if source is 'laz_file')
    4. Empty fallback
    """
    import numpy as np

    zones = entry.options.get(CONF_ZONES, [])
    latitude, longitude = _get_location(hass, entry)
    data_dir = hass.config.path(DATA_DIR)
    dsm_source = entry.options.get(CONF_DSM_SOURCE, DSM_SOURCE_AUTO)

    # 1. Check for cached DSM first (works for both auto and manual)
    cached = await hass.async_add_executor_job(load_site_model, data_dir)
    if cached is not None and cached.rows > 1:
        apply_zones_to_site(cached, zones)
        # Apply eraser mask if one exists
        eraser_mask = await hass.async_add_executor_job(load_eraser_mask, data_dir)
        if eraser_mask is not None:
            apply_eraser_to_site(cached, eraser_mask)
        _LOGGER.info(
            "Loaded cached DSM (%dx%d), applied %d zones%s",
            cached.rows, cached.cols, len(cached.zones),
            ", eraser mask applied" if eraser_mask is not None else "",
        )
        return cached

    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # 3. Manual LAZ file (small, can process inline)
    if dsm_source == DSM_SOURCE_LAZ:
        lidar_file = entry.options.get(CONF_LIDAR_FILE, "")
        if lidar_file:
            manual_epsg = entry.options.get(CONF_MANUAL_EPSG, DEFAULT_MANUAL_EPSG)
            site = await _try_laz_file(
                hass, lidar_file, latitude, longitude, data_dir, zones,
                epsg_override=manual_epsg,
            )
            if site is not None:
                return site

    # 2 / 4. Return placeholder — download will happen in background
    _LOGGER.info("No cached LiDAR data. Download will start in the background.")
    return SiteModel(dsm=np.zeros((1, 1)), resolution=DEFAULT_RESOLUTION, is_placeholder=True)


async def _background_download(
    hass: HomeAssistant,
    entry: ConfigEntry,
    coordinator,
) -> None:
    """Download LiDAR data in the background without blocking HA startup."""
    import asyncio

    zones = entry.options.get(CONF_ZONES, [])
    latitude, longitude = _get_location(hass, entry)
    data_dir = hass.config.path(DATA_DIR)
    download_radius = entry.options.get(CONF_DOWNLOAD_RADIUS, DEFAULT_DOWNLOAD_RADIUS)
    lidar_project = entry.options.get(CONF_LIDAR_PROJECT, "")
    min_cell_size = entry.options.get(CONF_MIN_CELL_SIZE, DEFAULT_MIN_CELL_SIZE)
    dsm_gap_fill = entry.options.get(CONF_DSM_GAP_FILL, DEFAULT_DSM_GAP_FILL)
    dsm_provider = entry.options.get(CONF_DSM_PROVIDER, DEFAULT_DSM_PROVIDER)

    _LOGGER.info(
        "Background LiDAR download starting (radius %dm)...",
        download_radius,
    )

    try:
        persistent_notification.async_create(
            hass,
            f"Downloading LiDAR data ({download_radius}m radius). "
            "This may take a few minutes...",
            title="Solar Shade: Downloading",
            notification_id="solar_shade_download_progress",
        )
    except (TypeError, RuntimeError, AttributeError):
        _LOGGER.debug("Could not create download notification (non-fatal)")

    # Retry up to 3 times with increasing delay for transient network failures
    max_retries = 3
    site = None
    for attempt in range(1, max_retries + 1):
        try:
            _LOGGER.debug("Download attempt %d/%d...", attempt, max_retries)
            site = await _try_download(
                hass, latitude, longitude, data_dir, zones,
                download_radius, lidar_project=lidar_project,
                min_cell_size=min_cell_size,
                dsm_gap_fill=dsm_gap_fill,
                dsm_provider=dsm_provider,
            )
            if site is not None:
                break
            _LOGGER.warning("Download attempt %d returned no data", attempt)
        except Exception:
            _LOGGER.exception("Download attempt %d failed", attempt)

        if attempt < max_retries:
            delay = 30 * attempt
            _LOGGER.debug("Retrying in %ds...", delay)
            await asyncio.sleep(delay)

    if site is None:
        _LOGGER.warning(
            "Background download found no data. "
            "Try Configure → Change DSM source"
        )
        try:
            persistent_notification.async_dismiss(
                hass, "solar_shade_download_progress",
            )
            persistent_notification.async_create(
                hass,
                "No LiDAR data found for your location. "
                "Try Configure → Change DSM source to manually specify a LiDAR project.",
                title="Solar Shade: No Data Found",
                notification_id="solar_shade_no_data",
            )
        except (TypeError, RuntimeError, AttributeError):
            _LOGGER.debug("Notification error (non-critical)", exc_info=True)
        return

    # Swap the real site into the coordinator
    coordinator.site = site
    hass.data[DOMAIN][entry.entry_id]["site"] = site
    _LOGGER.info(
        "Background download complete — DSM %dx%d now active",
        site.rows, site.cols,
    )
    try:
        persistent_notification.async_dismiss(
            hass, "solar_shade_download_progress",
        )
        persistent_notification.async_create(
            hass,
            f"LiDAR data ready! {site.rows}×{site.cols} grid loaded. "
            "Open the Solar Shade panel to view your terrain and draw zones.",
            title="Solar Shade: Ready",
            notification_id="solar_shade_ready",
        )
    except (TypeError, RuntimeError, AttributeError):
        _LOGGER.debug("Notification error (non-critical)", exc_info=True)
    await coordinator.async_request_refresh()


async def _try_download(
    hass, latitude, longitude, data_dir, zones, radius_m=150.0,
    lidar_project: str = "", min_cell_size: float = 0.5,
    dsm_gap_fill: bool = False, dsm_provider: str = "auto",
) -> SiteModel | None:
    """Try to auto-download DSM from the configured elevation provider."""
    from .elevation_provider import detect_provider, get_provider

    # Resolve provider
    if dsm_provider == DSM_PROVIDER_AUTO:
        provider_id = detect_provider(hass.config.country)
        _LOGGER.info(
            "Auto-detected elevation provider '%s' for country '%s'",
            provider_id, hass.config.country,
        )
    else:
        provider_id = dsm_provider

    # USGS path: use existing download_usgs_dsm for full feature parity
    # (lidar_project filtering, etc.)
    if provider_id == "usgs":
        return await _try_usgs_download(
            hass, latitude, longitude, data_dir, zones,
            radius_m, lidar_project=lidar_project,
            min_cell_size=min_cell_size,
            dsm_gap_fill=dsm_gap_fill,
        )

    # International providers: use the provider abstraction
    provider = get_provider(provider_id)
    _LOGGER.info(
        "Downloading DSM via %s for %.4f, %.4f (radius %dm)...",
        provider.PROVIDER_NAME, latitude, longitude, radius_m,
    )

    result = await provider.download_elevation(
        latitude=latitude,
        longitude=longitude,
        radius_m=radius_m,
        min_cell_size=min_cell_size,
        dsm_gap_fill=dsm_gap_fill,
    )

    if result is None:
        return None

    dsm, dtm, cls_grid, canopy_base, x_min, y_min, x_max, y_max, resolution = result

    extent_x = (x_max - x_min) / 2
    extent_y = (y_max - y_min) / 2

    site = SiteModel(
        dsm=dsm,
        resolution=resolution,
        latitude=latitude,
        longitude=longitude,
        dtm=dtm,
        classification=cls_grid,
        canopy_base=canopy_base,
        x_min_m=-extent_x,
        y_min_m=-extent_y,
        x_max_m=extent_x,
        y_max_m=extent_y,
        native_epsg=provider.NATIVE_EPSG,
    )

    await hass.async_add_executor_job(save_processed_dsm, site, data_dir)
    apply_zones_to_site(site, zones)
    _LOGGER.info(
        "%s data ready: DSM %dx%d, DTM %s",
        provider.PROVIDER_NAME,
        site.rows, site.cols,
        f"{dtm.shape[0]}x{dtm.shape[1]}" if dtm is not None else "not available",
    )
    return site


async def _try_usgs_download(
    hass, latitude, longitude, data_dir, zones, radius_m=150.0,
    lidar_project: str = "", min_cell_size: float = 0.5,
    dsm_gap_fill: bool = False,
) -> SiteModel | None:
    """Try to auto-download DSM GeoTIFF from USGS."""
    from .usgs_downloader import download_usgs_dsm

    if lidar_project:
        _LOGGER.info(
            "Downloading DSM from USGS S3 for project '%s' near %.4f, %.4f (radius %dm)...",
            lidar_project, latitude, longitude, radius_m,
        )
    else:
        _LOGGER.info(
            "Auto-downloading DSM from USGS for %.4f, %.4f (radius %dm)...",
            latitude, longitude, radius_m,
        )

    result = await download_usgs_dsm(
        latitude=latitude,
        longitude=longitude,
        radius_m=radius_m,
        lidar_project=lidar_project,
        min_cell_size=min_cell_size,
        dsm_gap_fill=dsm_gap_fill,
    )

    if result is None:
        return None

    dsm, dtm, cls_grid, canopy_base, x_min, y_min, x_max, y_max, resolution = result

    extent_x = (x_max - x_min) / 2
    extent_y = (y_max - y_min) / 2

    site = SiteModel(
        dsm=dsm,
        resolution=resolution,
        latitude=latitude,
        longitude=longitude,
        dtm=dtm,
        classification=cls_grid,
        canopy_base=canopy_base,
        x_min_m=-extent_x,
        y_min_m=-extent_y,
        x_max_m=extent_x,
        y_max_m=extent_y,
    )

    await hass.async_add_executor_job(save_processed_dsm, site, data_dir)
    apply_zones_to_site(site, zones)
    _LOGGER.info(
        "USGS data ready: DSM %dx%d, DTM %s",
        site.rows, site.cols,
        f"{dtm.shape[0]}x{dtm.shape[1]}" if dtm is not None else "not available",
    )
    return site


async def _try_laz_file(
    hass, lidar_file, latitude, longitude, data_dir, zones,
    epsg_override: int = 0,
) -> SiteModel | None:
    """Try to process a manually placed LAZ file."""
    from .shadow_engine import process_lidar_file
    from functools import partial

    lidar_path = Path(data_dir) / lidar_file
    if not lidar_path.exists():
        _LOGGER.warning("LAZ file not found: %s", lidar_path)
        return None

    _LOGGER.info("Processing LAZ file: %s", lidar_path)
    site = await hass.async_add_executor_job(
        partial(
            process_lidar_file,
            str(lidar_path), latitude, longitude,
            epsg_override=epsg_override,
        )
    )
    await hass.async_add_executor_job(save_processed_dsm, site, data_dir)
    apply_zones_to_site(site, zones)
    _LOGGER.info("LAZ DSM ready: %dx%d pixels", site.rows, site.cols)
    return site


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Set up the Solar Shade component."""
    _LOGGER.debug("Solar Shade: async_setup starting")
    try:
        from .websocket_api import async_register_websocket_api
        async_register_websocket_api(hass)
    except (ImportError, AttributeError, RuntimeError):
        _LOGGER.exception("Solar Shade: failed to register websocket API")
        return False

    try:
        from homeassistant.components.http import StaticPathConfig
        from homeassistant.components.frontend import async_register_built_in_panel

        await hass.http.async_register_static_paths(
            [StaticPathConfig(
                "/solar_shade/panel",
                hass.config.path("custom_components/solar_shade/frontend"),
                cache_headers=False,
            )]
        )

        # Cache-bust the iframe so a restart always loads the latest panel.html
        # (HA embeds the panel in an iframe whose HTML the browser caches
        # aggressively; a changing query string forces a fresh fetch).
        import time
        panel_url = f"/solar_shade/panel/panel.html?v={int(time.time())}"

        async_register_built_in_panel(
            hass,
            component_name="iframe",
            sidebar_title="Solar Shade",
            sidebar_icon="mdi:sun-angle",
            frontend_url_path="solar-shade",
            config={"url": panel_url},
            require_admin=True,
            update=True,
        )
        _LOGGER.info("Solar Shade: panel registered in sidebar")
    except (ImportError, AttributeError, OSError, RuntimeError):
        _LOGGER.exception("Solar Shade: failed to register panel")

    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Solar Shade from a config entry."""
    try:
        return await _async_setup_entry_inner(hass, entry)
    except Exception:
        _LOGGER.exception("Solar Shade: async_setup_entry failed")
        raise


async def _async_setup_entry_inner(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Solar Shade from a config entry."""
    _LOGGER.debug("Solar Shade: async_setup_entry starting (entry_id=%s)", entry.entry_id)

    _LOGGER.debug("Solar Shade: building site...")
    site = await _build_site_LiDAR(hass, entry)
    _LOGGER.debug("Solar Shade: site built (%dx%d)", site.rows, site.cols)

    coordinator = SolarShadeCoordinator(hass, entry, site)
    _LOGGER.debug("Solar Shade: running first coordinator refresh...")
    await coordinator.async_config_entry_first_refresh()
    _LOGGER.info("Solar Shade: coordinator ready")

    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = {
        "coordinator": coordinator,
        "site": site,
    }

    # Optionally set up Open-Meteo radiation sensors
    if entry.options.get(CONF_ENABLE_OPEN_METEO, DEFAULT_ENABLE_OPEN_METEO):
        from .open_meteo import OpenMeteoCoordinator
        om_coordinator = OpenMeteoCoordinator(hass, entry)
        await om_coordinator.async_config_entry_first_refresh()
        hass.data[DOMAIN][entry.entry_id]["open_meteo"] = om_coordinator

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # If site is a placeholder (no cached data), start background download
    if site.is_placeholder:
        dsm_source = entry.options.get(CONF_DSM_SOURCE, DSM_SOURCE_AUTO)
        _LOGGER.info(
            "Solar Shade: No cached LiDAR data (site.rows=%d, "
            "dsm_source=%s). Starting background download...",
            site.rows, dsm_source,
        )
        if dsm_source != DSM_SOURCE_LAZ:
            hass.async_create_task(
                _background_download(hass, entry, coordinator),
                "solar_shade_lidar_download",
            )
        else:
            _LOGGER.debug("DSM source is LAZ file — skipping auto-download")
    else:
        _LOGGER.info(
            "Using cached LiDAR data: %dx%d grid", site.rows, site.cols,
        )

    # Register reload service
    async def handle_reload(call: ServiceCall) -> None:
        """Rebuild site model from current config/files."""
        new_site = await _build_site_LiDAR(hass, entry)
        coordinator.site = new_site
        _LOGGER.info("Rebuilt site model with %d zones", len(new_site.zones))
        await coordinator.async_request_refresh()

    hass.services.async_register(DOMAIN, SERVICE_RELOAD_SITE, handle_reload)

    # Listen for options updates (obstacles/zones changed in UI)
    entry.async_on_unload(entry.add_update_listener(_async_options_updated))

    return True


async def _async_options_updated(
    hass: HomeAssistant, entry: ConfigEntry
) -> None:
    """Rebuild DSM when options (obstacles, zones, settings) change.

    Skip reload if LiDAR data hasn't loaded yet (background download in progress).
    """
    domain_data = hass.data.get(DOMAIN, {})
    entry_data = domain_data.get(entry.entry_id, {})
    site = entry_data.get("site")

    # A WebSocket handler (e.g. save_zones) may already be performing the
    # reload itself and awaiting it; skip to avoid a duplicate reload.
    from .websocket_api import SKIP_OPTIONS_RELOAD
    if entry.entry_id in SKIP_OPTIONS_RELOAD:
        _LOGGER.debug("Options update reload handled by caller, skipping listener reload")
        return

    # Don't reload if we're still waiting for background download
    if site is not None and site.is_placeholder:
        _LOGGER.info(
            "Options updated but LiDAR download in progress — skipping reload"
        )
        return

    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id)
        if not hass.data[DOMAIN]:
            hass.services.async_remove(DOMAIN, SERVICE_RELOAD_SITE)
    return unload_ok


async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Remove a config entry — clean up panel and cached data."""
    import shutil

    try:
        from homeassistant.components.frontend import async_remove_panel
        async_remove_panel(hass, "solar-shade")
    except (ImportError, KeyError):
        _LOGGER.debug("Panel removal error (non-critical)", exc_info=True)

    # Remove cached LiDAR data
    data_dir = hass.config.path(DATA_DIR)
    try:
        await hass.async_add_executor_job(shutil.rmtree, data_dir)
        _LOGGER.info("Removed cached data directory: %s", data_dir)
    except FileNotFoundError:
        pass
    except OSError:
        _LOGGER.warning("Could not remove cache directory: %s", data_dir)
