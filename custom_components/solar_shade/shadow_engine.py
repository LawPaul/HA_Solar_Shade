"""Shadow computation engine for Solar Shade integration.

Loads a DSM (Digital Surface Model) and zone definitions, then computes
shadow fractions on-the-fly by ray-tracing the current sun position.

Supports building a DSM from:
- Obstacle definitions (buildings, trees, fences entered in the UI)
- LiDAR point cloud files (LAS/LAZ placed in config/solar_shade/)
- Pre-built .npz files (from companion tools)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_LOGGER = logging.getLogger(__name__)

# Sun elevation below this angle is treated as no direct radiation
MIN_SUN_ELEVATION = 2.0

# LiDAR classification constants (must match usgs_downloader.py)
CLASS_GROUND = 2
CLASS_LOW_VEG = 3
CLASS_MED_VEG = 4
CLASS_HIGH_VEG = 5
CLASS_BUILDING = 6

# Transmittance bounds for seasonal variation (high vegetation only)
TRANSMITTANCE_LEAF_ON = 0.15   # Dense summer canopy
TRANSMITTANCE_LEAF_OFF = 0.65  # Bare deciduous branches

# Base transmittance by LiDAR classification.
# HIGH_VEG is overridden at runtime by apply_seasonal_transmittance().
BASE_TRANSMITTANCE_MAP = {
    CLASS_GROUND: 0.0,
    CLASS_LOW_VEG: 0.6,
    CLASS_MED_VEG: 0.4,
    CLASS_HIGH_VEG: TRANSMITTANCE_LEAF_ON,
    CLASS_BUILDING: 0.0,
}
DEFAULT_TRANSMITTANCE = 0.0


def seasonal_high_veg_transmittance(day_of_year: int, latitude: float) -> float:
    """Compute high-vegetation transmittance adjusted for season and latitude.

    Uses a sinusoidal curve tied to day-of-year, with latitude-based blending:
    - |lat| < 23: tropical, leaf-on year-round (0.15)
    - |lat| > 35: full seasonal curve (0.15 summer, 0.65 winter)
    - 23-35: linear blend between tropical and full seasonal
    - Southern hemisphere: curve is flipped (winter in June)
    """
    abs_lat = abs(latitude)

    if abs_lat < 23.0:
        return TRANSMITTANCE_LEAF_ON

    # Sinusoidal seasonal factor: 1.0 at summer solstice, 0.0 at winter solstice
    # Northern hemisphere: summer solstice ~ day 172
    angle = 2.0 * math.pi * (day_of_year - 172) / 365.0
    if latitude < 0:
        angle += math.pi  # flip for southern hemisphere
    leaf_factor = 0.5 * (1.0 + math.cos(angle))  # 1.0 = full leaf, 0.0 = bare

    # Blend between leaf-on and leaf-off
    transmittance = TRANSMITTANCE_LEAF_OFF + leaf_factor * (TRANSMITTANCE_LEAF_ON - TRANSMITTANCE_LEAF_OFF)

    if abs_lat < 35.0:
        # Subtropical blend: linearly reduce seasonality toward tropics
        blend = (abs_lat - 23.0) / 12.0  # 0 at 23°, 1 at 35°
        transmittance = TRANSMITTANCE_LEAF_ON + blend * (transmittance - TRANSMITTANCE_LEAF_ON)

    return transmittance


def build_transmittance_grid(site, day_of_year: int) -> np.ndarray:
    """Build a transmittance grid from classification + seasonal adjustment.

    Uses the classification grid to look up base transmittance values,
    then applies seasonal variation for high-vegetation pixels.
    If no classification is available, returns all zeros (fully opaque)
    so unclassified obstacles cast full shadows.
    """
    if site.classification is None:
        return np.zeros(site.dsm.shape, dtype=np.float32)

    cls = site.classification
    t = np.full(cls.shape, DEFAULT_TRANSMITTANCE, dtype=np.float32)
    for cls_code, trans_val in BASE_TRANSMITTANCE_MAP.items():
        t[cls == cls_code] = trans_val
    # Apply seasonal adjustment for high-veg pixels
    seasonal_t = seasonal_high_veg_transmittance(day_of_year, site.latitude)
    t[cls == CLASS_HIGH_VEG] = seasonal_t
    return t


@dataclass
class ZoneDef:
    """Definition of an irrigation zone on the DSM grid.

    Supports polygon masks (from map panel) or bounding-box regions (legacy).
    """

    zone_id: str
    zone_name: str
    # Bounding box on DSM grid (always set, used for fast clipping)
    row_start: int = 0
    row_end: int = 0
    col_start: int = 0
    col_end: int = 0
    # Polygon mask within the bounding box (None = full bbox)
    mask: np.ndarray | None = None
    # Original polygon as lat/lng pairs (for serialization back to panel)
    polygon_latlng: list | None = None
    color: str | None = None
    # Surface type: "ground" = shade measured at DTM level (default)
    #               "dsm" = shade measured at DSM level (rooftops, elevated surfaces)
    surface: str = "ground"


@dataclass
class SiteModel:
    """A DSM + DTM grid with zone definitions.

    dsm: Digital Surface Model — highest point per pixel (trees, buildings, ground).
         Used as the shadow-casting surface.
    dtm: Digital Terrain Model — bare ground elevation per pixel.
         Used as the shadow-receiving surface for zones.
         If None, dsm is used for both (legacy/obstacle mode).
    """

    dsm: np.ndarray  # 2D float array, meters — shadow casting surface
    resolution: float  # meters per pixel
    zones: list[ZoneDef] = field(default_factory=list)
    latitude: float = 0.0
    longitude: float = 0.0
    dtm: np.ndarray | None = None  # ground-only surface — shadow receiving
    classification: np.ndarray | None = None  # uint8 per pixel, ASPRS LAS class of top point
    canopy_base: np.ndarray | None = None  # lowest canopy return per pixel (for raised canopy model)
    # DSM extent in meters from the center point
    x_min_m: float = 0.0
    y_min_m: float = 0.0
    x_max_m: float = 0.0
    y_max_m: float = 0.0

    @property
    def rows(self) -> int:
        return self.dsm.shape[0]

    @property
    def cols(self) -> int:
        return self.dsm.shape[1]

    @property
    def ground(self) -> np.ndarray:
        """The ground surface — DTM if available, otherwise DSM."""
        return self.dtm if self.dtm is not None else self.dsm


def find_lidar_files(data_dir: str) -> list[str]:
    """Find all LAS/LAZ files in a directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    files = []
    for ext in ("*.las", "*.laz", "*.LAS", "*.LAZ"):
        files.extend(str(f.name) for f in data_path.glob(ext))
    return sorted(files)


def process_lidar_file(
    filepath: str,
    latitude: float,
    longitude: float,
) -> SiteModel:
    """Process a LAS/LAZ file into a SiteModel with DSM.

    The DSM is centered on the center of the point cloud.
    Zone coordinates are expressed as meters from this center point.
    """
    try:
        import laspy
    except ImportError:
        _LOGGER.error(
            "laspy is required for LAZ file processing. "
            "Install it with: pip install laspy[lazrs]"
        )
        raise

    _LOGGER.info("Processing LiDAR file: %s", filepath)
    las = laspy.read(filepath)
    x = np.array(las.x, dtype=np.float64)
    y = np.array(las.y, dtype=np.float64)
    z = np.array(las.z, dtype=np.float32)

    _LOGGER.info(
        "Loaded %d points, X: %.1f-%.1f, Y: %.1f-%.1f, Z: %.1f-%.1f",
        len(x), x.min(), x.max(), y.min(), y.max(), z.min(), z.max(),
    )

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    # Auto-calculate resolution from point density (target ~4 pts/cell).
    area_m2 = max((x_max - x_min) * (y_max - y_min), 1.0)
    density = len(x) / area_m2
    auto_res = max(0.5, 2.0 / max(density ** 0.5, 0.01))
    resolution = round(auto_res * 2) / 2  # snap to 0.5m increments
    resolution = max(0.5, min(resolution, 5.0))
    _LOGGER.info("Auto resolution: %.1fm (density: %.1f pts/m²)", resolution, density)

    cols = int(np.ceil((x_max - x_min) / resolution)) + 1
    rows = int(np.ceil((y_max - y_min) / resolution)) + 1

    col_idx = ((x - x_min) / resolution).astype(int)
    row_idx = ((y_max - y) / resolution).astype(int)  # row 0 = north

    col_idx = np.clip(col_idx, 0, cols - 1)
    row_idx = np.clip(row_idx, 0, rows - 1)

    # DSM = max height per cell
    dsm = np.full((rows, cols), np.nan, dtype=np.float32)
    for i in range(len(z)):
        r, c = int(row_idx[i]), int(col_idx[i])
        if np.isnan(dsm[r, c]) or z[i] > dsm[r, c]:
            dsm[r, c] = z[i]

    # Fill gaps with minimum (ground estimate)
    ground = float(np.nanmin(dsm))
    dsm = np.where(np.isnan(dsm), ground, dsm)

    _LOGGER.info("DSM: %dx%d pixels at %.1fm resolution", rows, cols, resolution)

    # Calculate extents in meters from center
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    extent_x = (x_max - x_min) / 2
    extent_y = (y_max - y_min) / 2

    return SiteModel(
        dsm=dsm,
        resolution=resolution,
        latitude=latitude,
        longitude=longitude,
        x_min_m=-extent_x,
        y_min_m=-extent_y,
        x_max_m=extent_x,
        y_max_m=extent_y,
    )


def save_processed_dsm(site: SiteModel, data_dir: str) -> None:
    """Save a processed DSM (and DTM if available) to disk."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    npz_path = data_path / "site_dsm.npz"

    save_data = {
        "dsm": site.dsm,
        "resolution": np.array(site.resolution),
        "latitude": np.array(site.latitude),
        "longitude": np.array(site.longitude),
        "x_min_m": np.array(site.x_min_m),
        "y_min_m": np.array(site.y_min_m),
        "x_max_m": np.array(site.x_max_m),
        "y_max_m": np.array(site.y_max_m),
    }
    if site.dtm is not None:
        save_data["dtm"] = site.dtm
    if site.classification is not None:
        save_data["classification"] = site.classification
    if site.canopy_base is not None:
        save_data["canopy_base"] = site.canopy_base

    np.savez_compressed(str(npz_path), **save_data)
    _LOGGER.info(
        "Saved DSM%s%s%s to %s",
        " + DTM" if site.dtm is not None else "",
        " + classification" if site.classification is not None else "",
        " + canopy_base" if site.canopy_base is not None else "",
        npz_path,
    )


def load_site_model(data_dir: str) -> SiteModel | None:
    """Load a site model from a .npz file.

    Expected files in data_dir:
        - site_dsm.npz: contains 'dsm', 'resolution', 'latitude', 'longitude', extent fields
    """
    data_path = Path(data_dir)
    npz_file = data_path / "site_dsm.npz"

    if not npz_file.exists():
        _LOGGER.warning("No site_dsm.npz found in %s", data_dir)
        return None

    data = np.load(str(npz_file))
    dsm = data["dsm"]
    resolution = float(data["resolution"])
    latitude = float(data["latitude"])
    longitude = float(data["longitude"])

    x_min_m = float(data["x_min_m"]) if "x_min_m" in data else 0.0
    y_min_m = float(data["y_min_m"]) if "y_min_m" in data else 0.0
    x_max_m = float(data["x_max_m"]) if "x_max_m" in data else 0.0
    y_max_m = float(data["y_max_m"]) if "y_max_m" in data else 0.0

    dtm = data["dtm"] if "dtm" in data else None
    classification = data["classification"] if "classification" in data else None
    canopy_base = data["canopy_base"] if "canopy_base" in data else None

    _LOGGER.info(
        "Loaded DSM: %dx%d at %.1fm, extent: %.0fm x %.0fm, DTM: %s, classification: %s",
        dsm.shape[0], dsm.shape[1], resolution,
        x_max_m - x_min_m, y_max_m - y_min_m,
        f"{dtm.shape[0]}x{dtm.shape[1]}" if dtm is not None else "none",
        "yes" if classification is not None else "none",
    )

    return SiteModel(
        dsm=dsm,
        resolution=resolution,
        latitude=latitude,
        longitude=longitude,
        dtm=dtm,
        classification=classification,
        canopy_base=canopy_base,
        x_min_m=x_min_m,
        y_min_m=y_min_m,
        x_max_m=x_max_m,
        y_max_m=y_max_m,
    )


def compute_shadow_map(
    dsm: np.ndarray,
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
    pixel_size_m: float,
    ground: np.ndarray | None = None,
    min_shadow_height: float = 1.5,
    transmittance: np.ndarray | None = None,
    canopy_base: np.ndarray | None = None,
) -> np.ndarray:
    """Compute shadow opacity map using scanline shadow propagation.

    Returns a float array (0.0 = full sun, 1.0 = full shadow) allowing
    partial shade through vegetation canopy.

    When a transmittance grid is provided, shadow intensity is attenuated
    as it passes through semi-transparent cells (vegetation). Buildings
    (transmittance=0) cast full shadow; tree canopy (transmittance=0.25)
    casts 75% shadow.

    Args:
        dsm: Digital Surface Model — shadow casting heights.
        ground: Ground receiving surface. If None, uses dsm.
        transmittance: Per-pixel light transmittance (0=opaque, 1=transparent).

    Returns float array 0.0-1.0 where 1.0 = fully shaded.
    """
    rows, cols = dsm.shape

    if sun_elevation_deg <= MIN_SUN_ELEVATION:
        return np.zeros((rows, cols), dtype=np.float32)

    # The receiving surface
    receive = ground if ground is not None else dsm

    # Filter out DSM noise: only features above min_shadow_height cast shadows
    if ground is not None:
        height_above_ground = dsm - ground
        cast = np.where(height_above_ground > min_shadow_height, dsm, ground)
    else:
        cast = dsm

    # Opacity of each DSM cell (1 - transmittance). Default = fully opaque.
    if transmittance is not None:
        opacity = 1.0 - transmittance.astype(np.float32)
        # Ground-level cells don't cast shadow
        if ground is not None:
            opacity = np.where(height_above_ground > min_shadow_height, opacity, 0.0)
    else:
        opacity = None  # binary mode

    az = math.radians(sun_azimuth_deg)
    el = math.radians(sun_elevation_deg)
    tan_el = math.tan(el)

    light_dx = math.sin(az)
    light_dy = -math.cos(az)

    step_scale = math.sqrt(light_dx**2 + light_dy**2)
    if step_scale < 1e-10:
        return np.zeros((rows, cols), dtype=np.float32)
    height_drop_per_pixel = pixel_size_m * step_scale * tan_el

    shadow = np.zeros((rows, cols), dtype=np.float32)

    # Ray-march approach: for each pixel, trace toward the sun checking
    # for obstructors. No axis-dependent switching, no quantization artifacts.
    shadow = _ray_march_shadow(
        rows, cols, cast, receive, opacity,
        light_dx, light_dy, pixel_size_m, el,
        canopy_base=canopy_base,
    )

    return shadow


def _ray_march_shadow(
    rows: int, cols: int,
    cast: np.ndarray,
    receive: np.ndarray,
    opacity: np.ndarray | None,
    light_dx: float,
    light_dy: float,
    pixel_size_m: float,
    sun_elevation_rad: float,
    canopy_base: np.ndarray | None = None,
) -> np.ndarray:
    """Compute shadow via per-pixel ray marching toward the sun.

    For each ground pixel, march along the light direction (toward the sun)
    checking DSM pixels along the ray. If any DSM pixel along the ray is
    tall enough to block the sun at that distance, the ground pixel is shaded.

    This approach has no axis-dependent artifacts because it traces the
    exact diagonal ray in floating point, sampling the nearest DSM cell.

    Complexity: O(rows * cols * max_ray_steps). With numpy vectorization,
    each step processes all pixels simultaneously.
    """
    tan_el = math.tan(sun_elevation_rad)
    if tan_el <= 0:
        return np.zeros((rows, cols), dtype=np.float32)

    shadow = np.zeros((rows, cols), dtype=np.float32)

    # Normalize the step direction: step by 1 pixel in the dominant axis
    abs_dx = abs(light_dx)
    abs_dy = abs(light_dy)
    dominant = max(abs_dx, abs_dy)
    if dominant < 1e-10:
        return shadow

    # Step size in row/col per march step (each step ≈ 1 pixel)
    step_dr = light_dy / dominant  # row change per step (toward sun)
    step_dc = light_dx / dominant  # col change per step (toward sun)
    # Real-world distance per step
    step_dist_m = pixel_size_m * math.sqrt(step_dr ** 2 + step_dc ** 2)

    # Maximum ray length (pixels) — diagonal of the grid
    max_steps = int(math.sqrt(rows ** 2 + cols ** 2)) + 1

    shadow_opacity = np.zeros((rows, cols), dtype=np.float32)
    fully_done = np.zeros((rows, cols), dtype=bool)

    # Pre-check: find the maximum feature height to bound ray length.
    # No shadow can be cast further than max_height / tan(el) / pixel_size.
    max_feature_height = float(cast.max() - receive.min())
    if max_feature_height <= 0:
        return shadow_opacity
    max_shadow_reach = int(max_feature_height / (step_dist_m * tan_el)) + 2
    max_steps = min(max_steps, max_shadow_reach)

    # Precompute per-step height threshold increment (scalar, not array)
    height_per_step = step_dist_m * tan_el

    # Track previous offset to skip duplicate steps.
    # When step_dr/step_dc are small, consecutive d values can round to
    # the same integer offset — the second pass would be identical work.
    prev_offset = None

    has_transmittance = opacity is not None

    for d in range(1, max_steps):
        # Source cell: integer offset from each pixel toward the sun.
        dr_int = int(round(d * step_dr))
        dc_int = int(round(d * step_dc))

        # Skip if same offset as previous step (identical computation)
        offset = (dr_int, dc_int)
        if offset == prev_offset:
            continue
        prev_offset = offset

        # Compute valid row/col ranges (avoids full-grid bounds check)
        r_lo = max(0, -dr_int)
        r_hi = min(rows, rows - dr_int)
        c_lo = max(0, -dc_int)
        c_hi = min(cols, cols - dc_int)

        if r_lo >= r_hi or c_lo >= c_hi:
            break

        # Slice the source and receiver grids to only valid pixels
        src_height = cast[r_lo + dr_int:r_hi + dr_int, c_lo + dc_int:c_hi + dc_int]
        recv_height = receive[r_lo:r_hi, c_lo:c_hi]
        done_slice = fully_done[r_lo:r_hi, c_lo:c_hi]

        # Height threshold: receiver + distance * tan(elevation)
        min_height_scalar = d * height_per_step

        # Does this source block the sun? (skip already-done pixels)
        blocks = ~done_slice & (src_height > recv_height + min_height_scalar)

        # Raised canopy model: rays passing under the canopy base
        # go through the open trunk zone unobstructed.
        # Only applies when there's actual clearance (canopy_base < DSM).
        if canopy_base is not None and blocks.any():
            src_canopy_base = canopy_base[r_lo + dr_int:r_hi + dr_int, c_lo + dc_int:c_hi + dc_int]
            # Ray height at the source cell = receiver ground + d * step * tan(el)
            ray_height = recv_height + min_height_scalar
            # Only consider "under canopy" when there's actual trunk clearance
            # (canopy_base significantly below DSM)
            has_clearance = src_canopy_base < (src_height - 0.5)
            under_canopy = has_clearance & (ray_height < src_canopy_base)
            blocks = blocks & ~under_canopy

        if not blocks.any():
            continue

        shad_slice = shadow_opacity[r_lo:r_hi, c_lo:c_hi]

        if has_transmittance:
            src_opac = opacity[r_lo + dr_int:r_hi + dr_int, c_lo + dc_int:c_hi + dc_int]
            update = blocks & (src_opac > shad_slice)
            if update.any():
                shad_slice[update] = src_opac[update]
        else:
            shad_slice[blocks] = 1.0

        done_slice |= (shad_slice >= 0.99)

    return shadow_opacity


def compute_zone_shade_fractions(
    site: SiteModel,
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
    min_shadow_height: float = 1.5,
    day_of_year: int = 172,
    canopy_model: str = "solid",
) -> dict[str, dict]:
    """Compute shade stats for each zone given current sun position.

    Supports two zone surface types:
    - "ground" zones: shadow computed with DTM as receive surface (default)
    - "dsm" zones: shadow computed with DSM as receive surface (rooftops)

    day_of_year is used to adjust high-veg transmittance for seasonal leaf cover.

    Returns per zone: {
        "average": float,   # mean shade fraction (0-1)
        "sunniest": float,  # shade at the sunniest spot (lowest shade)
        "shadiest": float,  # shade at the shadiest spot (highest shade)
    }
    """
    if sun_elevation_deg <= MIN_SUN_ELEVATION:
        return {z.zone_id: {"average": 1.0, "sunniest": 1.0, "shadiest": 1.0}
                for z in site.zones}

    # Build transmittance from classification + seasonal adjustment
    transmittance = build_transmittance_grid(site, day_of_year)

    # Check if we need DSM-surface shadow map
    has_dsm_zones = any(z.surface == "dsm" for z in site.zones)
    has_ground_zones = any(z.surface != "dsm" for z in site.zones)

    shadow_map_ground = None
    shadow_map_dsm = None

    canopy = site.canopy_base if canopy_model == "raised" else None

    if has_ground_zones:
        shadow_map_ground = compute_shadow_map(
            site.dsm, sun_azimuth_deg, sun_elevation_deg, site.resolution,
            ground=site.ground,
            min_shadow_height=min_shadow_height,
            transmittance=transmittance,
            canopy_base=canopy,
        )

    if has_dsm_zones:
        shadow_map_dsm = compute_shadow_map(
            site.dsm, sun_azimuth_deg, sun_elevation_deg, site.resolution,
            ground=site.dsm,
            min_shadow_height=0.0,
            transmittance=transmittance,
            canopy_base=canopy,
        )

    results: dict[str, dict] = {}
    for zone in site.zones:
        shadow_map = shadow_map_dsm if zone.surface == "dsm" else shadow_map_ground
        zone_shadow = shadow_map[
            zone.row_start : zone.row_end, zone.col_start : zone.col_end
        ]

        if zone.mask is not None:
            pixels = zone_shadow[zone.mask]
        else:
            pixels = zone_shadow.ravel()

        total = len(pixels)
        if total == 0:
            results[zone.zone_id] = {"average": 0.0, "sunniest": 0.0, "shadiest": 0.0}
        else:
            avg = round(float(pixels.sum()) / total, 3)
            # For spatial variation, compute shade in sub-regions
            # sunniest = minimum local shade, shadiest = maximum local shade
            # Split zone into a small grid and compute local shade per cell
            if total >= 9 and zone.mask is None:
                rows_z = zone.row_end - zone.row_start
                cols_z = zone.col_end - zone.col_start
                sr = max(1, rows_z // 3)
                sc = max(1, cols_z // 3)
                local_shades = []
                for r in range(0, rows_z, sr):
                    for c in range(0, cols_z, sc):
                        block = zone_shadow[r:r+sr, c:c+sc]
                        if block.size > 0:
                            local_shades.append(float(block.mean()))
                sunniest = round(min(local_shades), 3) if local_shades else avg
                shadiest = round(max(local_shades), 3) if local_shades else avg
            else:
                sunniest = 0.0 if avg < 1.0 else 1.0
                shadiest = 1.0 if avg > 0.0 else 0.0

            results[zone.zone_id] = {
                "average": avg,
                "sunniest": sunniest,
                "shadiest": shadiest,
            }

    return results


def compute_adjusted_radiation(
    raw_radiation: float,
    shade_fraction: float,
    diffuse_fraction: float = 0.15,
) -> float:
    """Compute shadow-adjusted solar radiation.

    In full shade, only diffuse radiation reaches the ground.
    adjusted = raw * (1 - shade_fraction * (1 - diffuse_fraction))
    """
    if raw_radiation <= 0:
        return 0.0

    adjustment = 1.0 - shade_fraction * (1.0 - diffuse_fraction)
    return round(raw_radiation * max(0.0, adjustment), 1)


def _zones_to_defs(
    zones: list[dict],
    dsm_shape: tuple[int, int],
    resolution: float,
    x_min: float,
    y_max: float,
) -> list[ZoneDef]:
    """Convert zone dicts (center + radius in meters) to pixel-based ZoneDefs."""
    rows, cols = dsm_shape
    result: list[ZoneDef] = []

    for z in zones:
        cx = z["center_x"]
        cy = z["center_y"]
        r = z.get("radius", 5)
        zone_id = z["id"]
        zone_name = z["name"]

        col_start = max(0, int((cx - r - x_min) / resolution))
        col_end = min(cols, int((cx + r - x_min) / resolution) + 1)
        row_start = max(0, int((y_max - cy - r) / resolution))
        row_end = min(rows, int((y_max - cy + r) / resolution) + 1)

        result.append(ZoneDef(
            zone_id=zone_id,
            zone_name=zone_name,
            row_start=row_start,
            row_end=row_end,
            col_start=col_start,
            col_end=col_end,
        ))

    return result


def apply_zones_to_site(site: SiteModel, zones: list[dict]) -> None:
    """Apply zone definitions to a SiteModel (works for both modes).

    Zones can be defined as:
    - Polygon lat/lng (from the map panel): has a "polygon" key with [[lat,lng],...]
    - Center + radius in meters (from the config flow): has center_x, center_y, radius

    This mutates the site in place.
    """
    rows, cols = site.dsm.shape

    new_zones: list[ZoneDef] = []

    for z in zones:
        if "polygon" in z and z["polygon"]:
            # Polygon mode (from map panel)
            zone_def = _polygon_to_zone_def(
                zone_id=z["id"],
                zone_name=z["name"],
                polygon_latlng=z["polygon"],
                color=z.get("color"),
                site=site,
                surface=z.get("surface", "ground"),
            )
            if zone_def:
                new_zones.append(zone_def)
        else:
            # Center + radius mode (from config flow)
            x_min = site.x_min_m
            y_max = site.y_max_m
            defs = _zones_to_defs([z], (rows, cols), site.resolution, x_min, y_max)
            new_zones.extend(defs)

    site.zones = new_zones
    _LOGGER.info("Applied %d zones to site model", len(new_zones))


def _polygon_to_zone_def(
    zone_id: str,
    zone_name: str,
    polygon_latlng: list[list[float]],
    color: str | None,
    site: SiteModel,
    surface: str = "ground",
) -> ZoneDef | None:
    """Convert a lat/lng polygon into a pixel-masked ZoneDef.

    Uses point-in-polygon (ray casting) to create a boolean mask on the DSM grid.
    """
    if len(polygon_latlng) < 3:
        _LOGGER.warning("Zone '%s' has fewer than 3 vertices, skipping", zone_name)
        return None

    rows, cols = site.dsm.shape

    # Convert lat/lng to DSM pixel coordinates
    # The DSM center corresponds to (site.latitude, site.longitude)
    ref_lat = site.latitude
    ref_lng = site.longitude
    m_per_deg_lat = 111320.0
    m_per_deg_lng = 111320.0 * math.cos(math.radians(ref_lat))

    pixel_coords = []
    for lat, lng in polygon_latlng:
        # Meters from center
        mx = (lng - ref_lng) * m_per_deg_lng
        my = (lat - ref_lat) * m_per_deg_lat

        # Pixel coordinates (row 0 = north = y_max)
        col = (mx - site.x_min_m) / site.resolution
        row = (site.y_max_m - my) / site.resolution
        pixel_coords.append((row, col))

    # Bounding box
    prows = [p[0] for p in pixel_coords]
    pcols = [p[1] for p in pixel_coords]

    row_start = max(0, int(math.floor(min(prows))))
    row_end = min(rows, int(math.ceil(max(prows))) + 1)
    col_start = max(0, int(math.floor(min(pcols))))
    col_end = min(cols, int(math.ceil(max(pcols))) + 1)

    if row_end <= row_start or col_end <= col_start:
        _LOGGER.warning("Zone '%s' is outside the DSM bounds", zone_name)
        return None

    # Build polygon mask using vectorized ray casting
    mask_h = row_end - row_start
    mask_w = col_end - col_start

    # Translate polygon to local mask coordinates
    local_poly = [(r - row_start, c - col_start) for r, c in pixel_coords]

    mask = _rasterize_polygon(mask_h, mask_w, local_poly)

    n_pixels = int(mask.sum())
    _LOGGER.info(
        "Zone '%s': bbox [%d:%d, %d:%d], %d pixels in polygon",
        zone_name, row_start, row_end, col_start, col_end, n_pixels,
    )

    return ZoneDef(
        zone_id=zone_id,
        zone_name=zone_name,
        row_start=row_start,
        row_end=row_end,
        col_start=col_start,
        col_end=col_end,
        mask=mask,
        polygon_latlng=polygon_latlng,
        color=color,
        surface=surface,
    )


def _rasterize_polygon(
    height: int, width: int, poly: list[tuple[float, float]]
) -> np.ndarray:
    """Rasterize a polygon into a boolean mask using vectorized ray casting.

    ~100x faster than per-pixel _point_in_polygon for large zones.
    Uses the even-odd rule: for each row, find all edge crossings,
    then fill between pairs of crossings.
    """
    mask = np.zeros((height, width), dtype=bool)
    n = len(poly)
    if n < 3:
        return mask

    # For each row, scan all edges and find x-intersections
    for r in range(height):
        y = r + 0.5  # pixel center
        crossings = []
        j = n - 1
        for i in range(n):
            yi, xi = poly[i]
            yj, xj = poly[j]
            if (yi > y) != (yj > y):
                # Edge crosses this row — compute x intersection
                x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
                crossings.append(x_cross)
            j = i

        if not crossings:
            continue

        # Sort crossings and fill between pairs
        crossings.sort()
        for k in range(0, len(crossings) - 1, 2):
            c_start = max(0, int(math.floor(crossings[k])))
            c_end = min(width, int(math.ceil(crossings[k + 1])))
            if c_start < c_end:
                # Refine: check pixel centers are actually inside
                cols = np.arange(c_start, c_end) + 0.5
                inside = (cols >= crossings[k]) & (cols < crossings[k + 1])
                mask[r, c_start:c_end] = inside

    return mask
