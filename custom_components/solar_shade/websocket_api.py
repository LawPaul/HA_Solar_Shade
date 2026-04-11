"""WebSocket API for the Solar Shade zone editor panel."""

from __future__ import annotations

import base64
import io
import logging
from datetime import timedelta
from typing import Any

import aiohttp
import asyncio
import numpy as np
import voluptuous as vol

from homeassistant.components import websocket_api
from homeassistant.core import HomeAssistant, callback
from homeassistant.util import dt as dt_util

from .const import (
    CONF_DIFFUSE_ENTITY,
    CONF_DIFFUSE_FRACTION,
    CONF_DOWNLOAD_RADIUS,
    CONF_ENABLE_OPEN_METEO,
    CONF_LATITUDE,
    CONF_LONGITUDE,
    CONF_RADIATION_ENTITY,
    CONF_ZONES,
    DEFAULT_DOWNLOAD_RADIUS,
    DEFAULT_DIFFUSE_FRACTION,
    DEFAULT_ENABLE_OPEN_METEO,
    DOMAIN,
)
from .geo import latlon_to_utm

_LOGGER = logging.getLogger(__name__)


@callback
def async_register_websocket_api(hass: HomeAssistant) -> None:
    """Register the websocket commands."""
    websocket_api.async_register_command(hass, ws_get_config)
    websocket_api.async_register_command(hass, ws_save_zones)
    websocket_api.async_register_command(hass, ws_update_radius)
    websocket_api.async_register_command(hass, ws_clear_cache)
    websocket_api.async_register_command(hass, ws_get_dsm_image)
    websocket_api.async_register_command(hass, ws_get_shadow_preview)
    websocket_api.async_register_command(hass, ws_get_dsm_data)
    websocket_api.async_register_command(hass, ws_get_shade_timeline)
    websocket_api.async_register_command(hass, ws_get_satellite_image)
    websocket_api.async_register_command(hass, ws_get_surface_type_image)


@websocket_api.websocket_command(
    {vol.Required("type"): "solar_shade/get_config"}
)
@callback
def ws_get_config(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return current config, zones, and DSM info for the panel."""
    result: dict[str, Any] = {
        "latitude": hass.config.latitude,
        "longitude": hass.config.longitude,
        "zones": [],
        "site_extent": None,
        "dsm_overlay": None,
        "entry_id": None,
        "download_radius": DEFAULT_DOWNLOAD_RADIUS,
    }

    # Find the first (usually only) solar_shade config entry
    entries = hass.config_entries.async_entries(DOMAIN)
    if not entries:
        connection.send_result(msg["id"], result)
        return

    entry = entries[0]
    result["entry_id"] = entry.entry_id

    # Use lat/long override if set
    lat_override = entry.options.get(CONF_LATITUDE)
    lng_override = entry.options.get(CONF_LONGITUDE)
    result["latitude"] = lat_override if lat_override is not None and lat_override != "" else hass.config.latitude
    result["longitude"] = lng_override if lng_override is not None and lng_override != "" else hass.config.longitude

    # Download radius from options
    result["download_radius"] = entry.options.get(
        CONF_DOWNLOAD_RADIUS, DEFAULT_DOWNLOAD_RADIUS
    )

    # Entity settings
    result["radiation_entity"] = entry.options.get(CONF_RADIATION_ENTITY, "")
    result["diffuse_entity"] = entry.options.get(CONF_DIFFUSE_ENTITY, "")
    result["diffuse_fraction"] = entry.options.get(CONF_DIFFUSE_FRACTION, DEFAULT_DIFFUSE_FRACTION)
    result["latitude_override"] = entry.options.get(CONF_LATITUDE, "")
    result["longitude_override"] = entry.options.get(CONF_LONGITUDE, "")
    result["enable_open_meteo"] = entry.options.get(CONF_ENABLE_OPEN_METEO, DEFAULT_ENABLE_OPEN_METEO)

    # Zones from options
    result["zones"] = entry.options.get(CONF_ZONES, [])

    # Site extent from loaded model
    domain_data = hass.data.get(DOMAIN, {})
    entry_data = domain_data.get(entry.entry_id, {})
    site = entry_data.get("site")

    if site and site.rows > 1:
        result["site_extent"] = {
            "x_min_m": site.x_min_m,
            "y_min_m": site.y_min_m,
            "x_max_m": site.x_max_m,
            "y_max_m": site.y_max_m,
        }

    connection.send_result(msg["id"], result)


@websocket_api.websocket_command(
    {
        vol.Required("type"): "solar_shade/save_zones",
        vol.Required("zones"): [
            {
                vol.Required("id"): str,
                vol.Required("name"): str,
                vol.Required("polygon"): [[vol.Coerce(float)]],
                vol.Optional("color"): str,
                vol.Optional("is_point", default=False): bool,
                vol.Optional("surface", default="ground"): vol.In(["ground", "dsm"]),
            }
        ],
    }
)
@websocket_api.async_response
async def ws_save_zones(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Save zones from the map panel to the config entry options."""
    entries = hass.config_entries.async_entries(DOMAIN)
    if not entries:
        connection.send_error(msg["id"], "not_configured", "No Solar Shade config entry found")
        return

    entry = entries[0]
    zones = msg["zones"]

    # Validate polygon data
    for zone in zones:
        is_point = zone.get("is_point", False)
        min_vertices = 1 if is_point else 3
        if len(zone["polygon"]) < min_vertices:
            connection.send_error(
                msg["id"], "invalid_polygon",
                f"Zone '{zone['name']}' needs at least {min_vertices} vertices"
            )
            return
        for point in zone["polygon"]:
            if len(point) != 2:
                connection.send_error(
                    msg["id"], "invalid_point",
                    f"Zone '{zone['name']}' has a point with {len(point)} coordinates (expected 2)"
                )
                return
            lat_val, lng_val = point[0], point[1]
            if not (-90 <= lat_val <= 90) or not (-180 <= lng_val <= 180):
                connection.send_error(
                    msg["id"], "invalid_coordinates",
                    f"Zone '{zone['name']}' has out-of-range coordinates: ({lat_val}, {lng_val})"
                )
                return

    # Update options (this triggers reload via update listener)
    new_options = {**entry.options, CONF_ZONES: zones}
    hass.config_entries.async_update_entry(entry, options=new_options)

    _LOGGER.info("Saved %d zones from map panel", len(zones))
    connection.send_result(msg["id"], {"saved": len(zones)})


@websocket_api.websocket_command(
    {
        vol.Required("type"): "solar_shade/update_radius",
        vol.Required("radius"): vol.All(vol.Coerce(int), vol.Range(min=50, max=500)),
    }
)
@websocket_api.async_response
async def ws_update_radius(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Update the download radius and trigger DSM re-download."""
    from pathlib import Path
    from .const import DATA_DIR

    entries = hass.config_entries.async_entries(DOMAIN)
    if not entries:
        connection.send_error(msg["id"], "not_configured", "No config entry found")
        return

    entry = entries[0]
    new_radius = msg["radius"]

    # Delete cached DSM to force re-download with new radius
    data_dir = hass.config.path(DATA_DIR)
    npz = Path(data_dir) / "site_dsm.npz"
    if npz.exists():
        npz.unlink()
        _LOGGER.info("Deleted cached DSM for radius change")

    # Update options (triggers reload)
    new_options = {**entry.options, CONF_DOWNLOAD_RADIUS: new_radius}
    hass.config_entries.async_update_entry(entry, options=new_options)

    _LOGGER.info("Download radius updated to %dm — reloading", new_radius)
    connection.send_result(msg["id"], {"radius": new_radius})

@websocket_api.websocket_command(
    {vol.Required("type"): "solar_shade/clear_cache"}
)
@websocket_api.async_response
async def ws_clear_cache(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Delete cached DSM and trigger a fresh LiDAR download."""
    from pathlib import Path
    from .const import DATA_DIR

    entries = hass.config_entries.async_entries(DOMAIN)
    if not entries:
        connection.send_error(msg["id"], "not_configured", "No config entry found")
        return

    entry = entries[0]
    data_dir = hass.config.path(DATA_DIR)
    npz = Path(data_dir) / "site_dsm.npz"
    deleted = False
    if npz.exists():
        npz.unlink()
        deleted = True
        _LOGGER.info("Cleared cached DSM — will re-download on reload")

    # Trigger reload to kick off fresh download
    await hass.config_entries.async_reload(entry.entry_id)

    connection.send_result(msg["id"], {"cleared": deleted})

def _get_site(hass: HomeAssistant):
    """Get the current SiteModel from integration data."""
    entries = hass.config_entries.async_entries(DOMAIN)
    if not entries:
        return None
    entry_data = hass.data.get(DOMAIN, {}).get(entries[0].entry_id, {})
    return entry_data.get("site")


def _site_bounds_latlng(site):
    """Get site bounds as lat/lng using proper CRS inverse projection."""
    lat, lng = site.latitude, site.longitude

    if site.native_epsg:
        from .geo import latlon_to_epsg, epsg_to_latlon
        center_e, center_n = latlon_to_epsg(lat, lng, site.native_epsg)
        epsg = site.native_epsg
    else:
        center_e, center_n = latlon_to_utm(lat, lng)[1:3]
        zone = int((lng + 180) / 6) + 1
        epsg = (32600 + zone) if lat >= 0 else (32700 + zone)
        from .geo import epsg_to_latlon

    s_lat, s_lon = epsg_to_latlon(center_e + site.x_min_m, center_n + site.y_min_m, epsg)
    n_lat, n_lon = epsg_to_latlon(center_e + site.x_max_m, center_n + site.y_max_m, epsg)
    return {
        "south": s_lat,
        "north": n_lat,
        "west": s_lon,
        "east": n_lon,
    }



def _calc_sun_position(lat: float, lng: float, when) -> tuple[float, float]:
    """Calculate solar azimuth and elevation using the astral library.

    Uses the same algorithm as HA's built-in sun tracking (US Naval Observatory),
    accurate to ~0.01°. Handles timezone, refraction, and equation of time correctly.

    Args:
        lat: Latitude in degrees.
        lng: Longitude in degrees.
        when: Timezone-aware datetime (use dt_util.now()).

    Returns (azimuth_degrees, elevation_degrees).
    """
    from astral import Observer
    from astral.sun import azimuth as astral_azimuth, elevation as astral_elevation

    observer = Observer(latitude=lat, longitude=lng, elevation=0)

    # Ensure timezone-aware
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt_util.now().tzinfo)

    try:
        az = astral_azimuth(observer, when)
        el = astral_elevation(observer, when)
    except ValueError:
        # Fallback for edge cases (e.g., midnight sun / polar night)
        az = 180.0
        el = -90.0

    return float(az), float(el)


def _array_to_png_data_url(rgba: np.ndarray) -> str:
    """Encode an RGBA numpy array as a base64 PNG data URL."""
    try:
        from PIL import Image
        img = Image.fromarray(rgba, "RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except ImportError:
        # Fallback: minimal PNG encoder for RGBA data
        return _minimal_png_encode(rgba)


def _minimal_png_encode(rgba: np.ndarray) -> str:
    """Encode RGBA array as PNG without PIL (minimal implementation)."""
    import struct
    import zlib

    h, w, _ = rgba.shape
    raw = b""
    for row in rgba:
        raw += b"\x00" + row.tobytes()  # filter byte + RGBA

    compressed = zlib.compress(raw)

    def chunk(ctype, data):
        c = ctype + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))  # 8-bit RGBA
    png += chunk(b"IDAT", compressed)
    png += chunk(b"IEND", b"")

    b64 = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _dsm_to_rgba(site) -> np.ndarray:
    """Render DSM heights as an RGBA image with a color ramp.

    Colormap:
      Transparent: 0-1.5m (ground, noise, grass)
      Green:       1.5-3m (low shrubs, fences)
      Yellow:      3-6m (small trees, single-story)
      Orange:      6-12m (medium trees, two-story buildings)
      Red:         12m+ (tall trees, large buildings)
    """
    dsm = site.dsm

    if site.dtm is not None:
        height_above_ground = dsm - site.dtm
        above_1_5 = int(np.sum(height_above_ground > 1.5))
        _LOGGER.info(
            "Height overlay: DSM %.1f-%.1f, DTM %.1f-%.1f, diff %.1f-%.1f, "
            "pixels above 1.5m: %d/%d",
            float(dsm.min()), float(dsm.max()),
            float(site.dtm.min()), float(site.dtm.max()),
            float(height_above_ground.min()), float(height_above_ground.max()),
            above_1_5, height_above_ground.size,
        )
        # If DTM doesn't add useful info (no features above ground), fall back
        if above_1_5 == 0:
            _LOGGER.warning(
                "DTM matches DSM everywhere — OPR tiles may be bare-earth DEMs. "
                "Falling back to relative height mode."
            )
            height_above_ground = dsm - float(dsm.min())
    else:
        height_above_ground = dsm - float(dsm.min())

    above_thresh = int(np.sum(height_above_ground > 1.5))
    _LOGGER.info(
        "Height overlay: final range %.1f-%.1fm, %d/%d pixels visible",
        float(height_above_ground.min()), float(height_above_ground.max()),
        above_thresh, height_above_ground.size,
    )

    h = np.clip(height_above_ground, 0, 15)  # cap at 15m

    # Color ramp using height thresholds
    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    # 1.5-3m: green (0,180,0) → yellow-green
    mask = (h >= 1.5) & (h < 3)
    t = (h - 1.5) / 1.5  # 0→1 within range
    r = np.where(mask, t * 0.6, r)
    g = np.where(mask, 0.7, g)

    # 3-6m: yellow (200,200,0) → orange
    mask = (h >= 3) & (h < 6)
    t = (h - 3) / 3
    r = np.where(mask, 0.6 + t * 0.4, r)
    g = np.where(mask, 0.7 - t * 0.3, g)

    # 6-12m: orange (255,100,0) → red
    mask = (h >= 6) & (h < 12)
    t = (h - 6) / 6
    r = np.where(mask, 1.0, r)
    g = np.where(mask, 0.4 - t * 0.4, g)

    # 12m+: deep red
    mask = h >= 12
    r = np.where(mask, 0.9, r)
    g = np.where(mask, 0.0, g)

    # Alpha: transparent below 1.5m, solid for features
    a = np.where(h < 1.5, 0.0, 0.75)
    # Fade in from 1.5-2.5m
    fade = (h >= 1.5) & (h < 2.5)
    a = np.where(fade, 0.4 + (h - 1.5) * 0.35, a)

    rgba = np.stack([
        (r * 255).astype(np.uint8),
        (g * 255).astype(np.uint8),
        (b * 255).astype(np.uint8),
        (a * 255).astype(np.uint8),
    ], axis=-1)

    return rgba


def _shadow_to_rgba(shadow_map: np.ndarray) -> np.ndarray:
    """Render shadow map as semi-transparent dark overlay.

    Supports both boolean (legacy) and float (0-1 opacity) shadow maps.
    """
    h, w = shadow_map.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    if shadow_map.dtype == bool:
        # Legacy binary shadow
        rgba[shadow_map, 0] = 30
        rgba[shadow_map, 1] = 30
        rgba[shadow_map, 2] = 80
        rgba[shadow_map, 3] = 140
    else:
        # Float opacity map — darker = more shadow
        opacity = np.clip(shadow_map, 0, 1)
        rgba[:, :, 0] = (30 * opacity).astype(np.uint8)
        rgba[:, :, 1] = (30 * opacity).astype(np.uint8)
        rgba[:, :, 2] = (80 * opacity).astype(np.uint8)
        rgba[:, :, 3] = (160 * opacity).astype(np.uint8)

    return rgba


@websocket_api.websocket_command(
    {vol.Required("type"): "solar_shade/get_dsm_image"}
)
@websocket_api.async_response
async def ws_get_dsm_image(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return DSM height visualization as a PNG data URL with geo bounds."""
    site = _get_site(hass)
    if site is None or site.is_placeholder:
        connection.send_result(msg["id"], {"available": False})
        return

    def render():
        rgba = _dsm_to_rgba(site)
        return _array_to_png_data_url(rgba)

    image_url = await hass.async_add_executor_job(render)
    bounds = _site_bounds_latlng(site)

    connection.send_result(msg["id"], {
        "available": True,
        "image_url": image_url,
        "bounds": bounds,
        "rows": site.rows,
        "cols": site.cols,
        "has_dtm": site.dtm is not None,
    })


@websocket_api.websocket_command(
    {
        vol.Required("type"): "solar_shade/get_shadow_preview",
        vol.Optional("hour_offset", default=0): vol.Coerce(float),
        vol.Optional("sun_only", default=False): bool,
    }
)
@websocket_api.async_response
async def ws_get_shadow_preview(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return shadow map as a PNG overlay for a given time.

    hour_offset: hours from now (e.g. -2 = 2 hours ago, +3 = 3 hours from now, 0 = now).
    sun_only: if True, skip raytraced render and return only sun position (for 3D view).
    """
    site = _get_site(hass)
    if site is None or site.is_placeholder:
        connection.send_result(msg["id"], {"available": False})
        return

    hour_offset = msg.get("hour_offset", 0)

    # Compute sun position at the requested time
    lat = site.latitude if site.latitude is not None else hass.config.latitude
    lng = site.longitude if site.longitude is not None else hass.config.longitude

    if hour_offset == 0:
        # Use HA's live sun position for zero offset
        sun_state = hass.states.get("sun.sun")
        if sun_state is None:
            connection.send_result(msg["id"], {"available": False, "reason": "no_sun"})
            return
        try:
            azimuth = float(sun_state.attributes.get("azimuth") or 0)
            elevation = float(sun_state.attributes.get("elevation") or -90)
        except (TypeError, ValueError) as err:
            _LOGGER.warning("Invalid sun.sun attributes: %s", err)
            connection.send_result(msg["id"], {"available": False, "reason": "no_sun"})
            return
        target_time = dt_util.now()
    else:
        # Calculate sun position at offset time
        target_time = dt_util.now() + timedelta(hours=hour_offset)
        azimuth, elevation = _calc_sun_position(lat, lng, target_time)

    time_label = target_time.strftime("%I:%M %p").lstrip("0")

    if elevation <= 2:
        connection.send_result(msg["id"], {
            "available": True,
            "image_url": None,
            "sun_below_horizon": True,
            "azimuth": round(azimuth, 1),
            "elevation": round(elevation, 1),
            "time_label": time_label,
        })
        return

    # sun_only: return just sun position without expensive raytrace (for 3D view)
    if msg.get("sun_only", False):
        connection.send_result(msg["id"], {
            "available": True,
            "image_url": None,
            "sun_below_horizon": False,
            "azimuth": round(azimuth, 1),
            "elevation": round(elevation, 1),
            "time_label": time_label,
        })
        return

    from .shadow_engine import compute_shadow_map, build_transmittance_grid
    from .const import CONF_CANOPY_MODEL, DEFAULT_CANOPY_MODEL, CANOPY_MODEL_RAISED

    # Read canopy model setting before entering executor thread
    entries = hass.config_entries.async_entries(DOMAIN)
    canopy_model = entries[0].options.get(CONF_CANOPY_MODEL, DEFAULT_CANOPY_MODEL) if entries else DEFAULT_CANOPY_MODEL

    def render():
        day_of_year = dt_util.now().timetuple().tm_yday
        trans = build_transmittance_grid(site, day_of_year)
        canopy = site.canopy_base if canopy_model == CANOPY_MODEL_RAISED else None
        shadow = compute_shadow_map(
            site.dsm, azimuth, elevation, site.resolution,
            ground=site.ground,
            transmittance=trans,
            canopy_base=canopy,
        )
        rgba = _shadow_to_rgba(shadow)
        return _array_to_png_data_url(rgba), float(shadow.mean())

    try:
        image_url, shade_fraction = await hass.async_add_executor_job(render)
    except (ValueError, RuntimeError, TypeError):
        _LOGGER.exception("Failed to render shadow preview")
        connection.send_result(msg["id"], {"available": False, "reason": "render_error"})
        return

    bounds = _site_bounds_latlng(site)

    connection.send_result(msg["id"], {
        "available": True,
        "image_url": image_url,
        "bounds": bounds,
        "azimuth": round(azimuth, 1),
        "elevation": round(elevation, 1),
        "shade_fraction": round(shade_fraction, 3),
        "sun_below_horizon": False,
        "time_label": time_label,
    })


@websocket_api.websocket_command(
    {
        vol.Required("type"): "solar_shade/get_dsm_data",
        vol.Optional("hour_offset", default=0): vol.Coerce(float),
    }
)
@websocket_api.async_response
async def ws_get_dsm_data(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return raw DSM/DTM height data for 3D rendering."""
    site = _get_site(hass)
    if site is None or site.is_placeholder:
        connection.send_result(msg["id"], {"available": False})
        return

    def prepare():
        from .shadow_engine import build_transmittance_grid

        dsm = site.dsm
        ground = site.ground
        cls_grid = site.classification
        rows, cols = dsm.shape

        # Build transmittance from classification at runtime
        day_of_year = dt_util.now().timetuple().tm_yday
        trans = build_transmittance_grid(site, day_of_year)

        # Normalize heights relative to ground minimum
        base = float(ground.min())
        dsm_norm = (dsm - base).astype(np.float32)
        gnd_norm = (ground - base).astype(np.float32)

        # Encode as base64 binary for compact transfer
        dsm_b64 = base64.b64encode(dsm_norm.tobytes()).decode("ascii")
        gnd_b64 = base64.b64encode(gnd_norm.tobytes()).decode("ascii")

        result = {
            "dsm_b64": dsm_b64,
            "ground_b64": gnd_b64,
            "rows": rows,
            "cols": cols,
            "pixel_m": site.resolution,
            "base_elevation": round(base, 1),
            "site_lat": site.latitude,
            "site_lng": site.longitude,
            "x_min_m": site.x_min_m,
            "y_min_m": site.y_min_m,
            "x_max_m": site.x_max_m,
            "y_max_m": site.y_max_m,
        }
        trans_b64 = base64.b64encode(
            np.round(trans, 2).astype(np.float32).tobytes()
        ).decode("ascii")
        result["transmittance_b64"] = trans_b64
        if cls_grid is not None:
            cls_b64 = base64.b64encode(
                cls_grid.astype(np.uint8).tobytes()
            ).decode("ascii")
            result["classification_b64"] = cls_b64

        # Include canopy base for raised canopy 3D rendering
        if site.canopy_base is not None:
            cb_norm = (site.canopy_base - base).astype(np.float32)
            result["canopy_base_b64"] = base64.b64encode(cb_norm.tobytes()).decode("ascii")

        return result

    data = await hass.async_add_executor_job(prepare)

    # Add sun position (with optional time offset)
    hour_offset = msg.get("hour_offset", 0)
    lat = site.latitude if site.latitude is not None else hass.config.latitude
    lng = site.longitude if site.longitude is not None else hass.config.longitude

    if hour_offset == 0:
        sun_state = hass.states.get("sun.sun")
        if sun_state:
            data["sun_azimuth"] = float(sun_state.attributes.get("azimuth", 180))
            data["sun_elevation"] = float(sun_state.attributes.get("elevation", 45))
        else:
            data["sun_azimuth"] = 180.0
            data["sun_elevation"] = 45.0
        target_time = dt_util.now()
    else:
        target_time = dt_util.now() + timedelta(hours=hour_offset)
        az, el = _calc_sun_position(lat, lng, target_time)
        data["sun_azimuth"] = round(az, 1)
        data["sun_elevation"] = round(max(el, 0), 1)

    data["time_label"] = target_time.strftime("%I:%M %p").lstrip("0")

    # Add zone polygons in relative coordinates
    zones_3d = []
    for z in site.zones:
        if z.polygon_latlng:
            zones_3d.append({
                "id": z.zone_id,
                "name": z.zone_name,
                "color": z.color or "#2196F3",
                "polygon": z.polygon_latlng,
                "surface": z.surface,
            })
    data["zones"] = zones_3d
    data["available"] = True

    connection.send_result(msg["id"], data)


@websocket_api.websocket_command(
    {vol.Required("type"): "solar_shade/get_shade_timeline"}
)
@websocket_api.async_response
async def ws_get_shade_timeline(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Compute hourly shade fractions for each zone across today.

    Returns per-zone shade fraction for each hour from sunrise to sunset
    (computed via astral), enabling a daily shade forecast chart.
    """
    site = _get_site(hass)
    if site is None or site.is_placeholder or not site.zones:
        connection.send_result(msg["id"], {"available": False})
        return

    from .shadow_engine import compute_zone_shade_fractions

    lat = site.latitude if site.latitude is not None else hass.config.latitude
    lng = site.longitude if site.longitude is not None else hass.config.longitude

    from .const import CONF_CANOPY_MODEL, DEFAULT_CANOPY_MODEL
    entries = hass.config_entries.async_entries(DOMAIN)
    _canopy_model = entries[0].options.get(CONF_CANOPY_MODEL, DEFAULT_CANOPY_MODEL) if entries else DEFAULT_CANOPY_MODEL

    def compute_timeline():
        """Compute shade for each hour from sunrise to sunset using astral."""
        from datetime import datetime
        from astral import Observer
        from astral.sun import sun as astral_sun

        observer = Observer(latitude=lat, longitude=lng, elevation=0)
        now = dt_util.now()
        today = now.date()
        tz = now.tzinfo

        # Get sunrise and sunset from astral
        try:
            sun_times = astral_sun(observer, today, tzinfo=tz)
            sunrise_hour = sun_times["sunrise"].hour
            sunset_hour = sun_times["sunset"].hour + 1  # include sunset hour
        except ValueError:
            # Fallback if astral can't compute (e.g., polar regions)
            sunrise_hour = 6
            sunset_hour = 20

        # Clamp to reasonable range
        sunrise_hour = max(0, min(sunrise_hour, 12))
        sunset_hour = max(sunrise_hour + 1, min(sunset_hour, 23))

        hours = list(range(sunrise_hour, sunset_hour + 1))
        timeline: dict[str, list] = {z.zone_id: [] for z in site.zones}
        hour_labels = []

        for hour in hours:
            when = datetime(today.year, today.month, today.day, hour, 0, 0, tzinfo=tz)
            azimuth, elevation = _calc_sun_position(lat, lng, when)

            hour_labels.append(f"{hour}:00")

            if elevation <= 2:
                for z in site.zones:
                    timeline[z.zone_id].append(0.0)
                continue

            canopy_model = _canopy_model
            fracs = compute_zone_shade_fractions(site, azimuth, elevation, canopy_model=canopy_model)
            for z in site.zones:
                timeline[z.zone_id].append(round(fracs.get(z.zone_id, {}).get("average", 0.0), 2))

        return hour_labels, timeline

    hour_labels, timeline = await hass.async_add_executor_job(compute_timeline)

    zones_info = []
    for z in site.zones:
        shades = timeline[z.zone_id]
        avg = round(sum(shades) / len(shades), 2) if shades else 0
        zones_info.append({
            "id": z.zone_id,
            "name": z.zone_name,
            "color": z.color or "#2196F3",
            "hourly_shade": shades,
            "avg_shade": avg,
        })

    connection.send_result(msg["id"], {
        "available": True,
        "hours": hour_labels,
        "zones": zones_info,
        "current_hour": dt_util.now().hour,
    })


@websocket_api.websocket_command(
    {vol.Required("type"): "solar_shade/get_satellite_image"}
)
@websocket_api.async_response
async def ws_get_satellite_image(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Fetch satellite imagery for the site extent from Esri World Imagery.

    Returns a base64 PNG data URL sized to the DSM grid dimensions.
    Used as a ground texture in the 3D viewer.
    """
    site = _get_site(hass)
    if site is None or site.is_placeholder:
        connection.send_result(msg["id"], {"available": False})
        return

    # Use a projected CRS for the satellite bbox so pixels stay square.
    # If the site was built from a national CRS (SWEREF99 TM, D96/TM, …)
    # use that CRS directly; otherwise fall back to UTM.
    if site.native_epsg:
        from .geo import latlon_to_epsg
        center_e, center_n = latlon_to_epsg(
            site.latitude, site.longitude, site.native_epsg,
        )
        proj_epsg = site.native_epsg
    else:
        zone, center_e, center_n = latlon_to_utm(site.latitude, site.longitude)
        northern = site.latitude >= 0
        proj_epsg = (32600 + zone) if northern else (32700 + zone)

    proj_x_min = center_e + site.x_min_m
    proj_y_min = center_n + site.y_min_m
    proj_x_max = center_e + site.x_max_m
    proj_y_max = center_n + site.y_max_m

    # Esri World Imagery export endpoint
    export_url = (
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/export"
    )

    # Image size matches grid dimensions (1 pixel per cell)
    img_w = min(site.cols, 1024)
    img_h = min(site.rows, 1024)

    params = {
        "bbox": f"{proj_x_min},{proj_y_min},{proj_x_max},{proj_y_max}",
        "bboxSR": str(proj_epsg),
        "imageSR": str(proj_epsg),
        "size": f"{img_w},{img_h}",
        "format": "png",
        "f": "image",
    }

    from homeassistant.helpers.aiohttp_client import async_get_clientsession

    try:
        session = async_get_clientsession(hass)
        async with session.get(
            export_url, params=params,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
                if resp.status != 200:
                    _LOGGER.warning("Satellite image fetch failed: HTTP %d", resp.status)
                    connection.send_result(msg["id"], {"available": False})
                    return

                image_bytes = await resp.read()
                b64 = base64.b64encode(image_bytes).decode("ascii")
                data_url = f"data:image/png;base64,{b64}"

    except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError) as err:
        _LOGGER.warning("Failed to fetch satellite imagery: %s", err)
        connection.send_result(msg["id"], {"available": False})
        return

    connection.send_result(msg["id"], {
        "available": True,
        "image_url": data_url,
        "width": img_w,
        "height": img_h,
    })


@websocket_api.websocket_command(
    {vol.Required("type"): "solar_shade/get_surface_type_image"}
)
@websocket_api.async_response
async def ws_get_surface_type_image(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return surface type classification as a colored PNG overlay.

    Colors match the 3D view: tan=ground, green=vegetation, red=buildings.
    """
    site = _get_site(hass)
    if site is None or site.is_placeholder or site.classification is None:
        connection.send_result(msg["id"], {"available": False})
        return

    def render():
        cls = site.classification
        hab = site.dsm - site.ground if site.dtm is not None else np.zeros_like(site.dsm)
        rows, cols = cls.shape
        rgba = np.zeros((rows, cols, 4), dtype=np.uint8)

        # Color map matching 3D view
        colors = {
            2: (194, 178, 128, 120),   # ground — tan (semi-transparent)
            3: (140, 204, 77, 160),     # low veg — lime
            4: (77, 166, 51, 180),      # med veg — green
            5: (26, 115, 31, 200),      # high veg — dark green
            6: (191, 51, 38, 200),      # building — red
            9: (64, 128, 200, 180),     # water — blue
        }

        for cls_code, (r, g, b, a) in colors.items():
            mask = cls == cls_code
            if cls_code == 2:
                # Ground only visible if not flat (hills, etc.)
                mask = mask & (hab > 0.3)
            rgba[mask, 0] = r
            rgba[mask, 1] = g
            rgba[mask, 2] = b
            rgba[mask, 3] = a

        # Unclassified elevated features
        known = np.isin(cls, list(colors.keys()))
        unknown = ~known & (hab > 0.3)
        rgba[unknown] = [153, 140, 128, 160]  # gray

        return _array_to_png_data_url(rgba)

    image_url = await hass.async_add_executor_job(render)
    bounds = _site_bounds_latlng(site)

    connection.send_result(msg["id"], {
        "available": True,
        "image_url": image_url,
        "bounds": bounds,
    })


