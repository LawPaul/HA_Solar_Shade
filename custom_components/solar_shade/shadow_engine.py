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
    # Shade aggregation method for this zone: "average" | "sunniest" | "shadiest"
    shade_method: str = "average"
    # Spot size in m² used by the "sunniest"/"shadiest" methods: the shade value
    # reported is the min/max average over a sliding window of this area.
    spot_area: float = 1.0


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
    # EPSG code of the projected CRS used for grid coordinates (0 = UTM)
    native_epsg: int = 0
    # True while background download is in progress (no real data yet)
    is_placeholder: bool = False

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
    min_cell_size: float = 0.5,
    dsm_gap_fill: bool = False,
    epsg_override: int = 0,
) -> SiteModel:
    """Process a LAS/LAZ file into a SiteModel.

    For manually-placed files, always rasterizes using the file's own
    centroid and full extent — the user's lat/lon is metadata for sun
    position calculations, not a clipping anchor.

    CRS detection order:
    1. epsg_override (user-configured, if non-zero)
    2. VLR metadata (parse_crs, WKT, GeoKeys)
    3. Filename EPSG extraction
    4. Falls back to 0 (unknown) — still works, just no reprojection.
    """
    from .usgs_downloader import (
        _read_laz_epsg,
        _rasterize_laz_file,
    )

    try:
        import laspy
    except ImportError:
        _LOGGER.error(
            "laspy is required for LAZ file processing. "
            "Install it with: pip install laspy[lazrs]"
        )
        raise

    _LOGGER.info("Processing LiDAR file: %s", filepath)

    # ── Detect the file's CRS ────────────────────────────────────
    if epsg_override:
        native_epsg = epsg_override
        _LOGGER.info("Using user-configured EPSG override: %d", native_epsg)
    else:
        detected = _read_laz_epsg(filepath, laspy)
        native_epsg = detected or 0
        if native_epsg:
            _LOGGER.info("Auto-detected CRS: EPSG:%d", native_epsg)
        else:
            _LOGGER.info(
                "No CRS detected from file metadata — "
                "rasterizing with raw coordinates."
            )

    # ── Use file centroid and full extent for rasterization ───────
    # A manually-placed file should be processed in its entirety.
    # The user's lat/lon may project outside this tile's bounds
    # (e.g. the API returned a neighboring tile), so we always use
    # the file's own centroid as the rasterization center.
    try:
        with laspy.open(filepath) as reader:
            mins = reader.header.mins
            maxs = reader.header.maxs
            center_e = (mins[0] + maxs[0]) / 2
            center_n = (mins[1] + maxs[1]) / 2
            radius_m = max(maxs[0] - mins[0], maxs[1] - mins[1]) / 2 + 10
    except (OSError, ValueError, IndexError):
        _LOGGER.error("Could not read LAZ header to determine extent")
        raise

    _LOGGER.info(
        "Rasterizing manual LAZ: centroid E%.1f N%.1f, "
        "radius %.0fm, EPSG:%d",
        center_e, center_n, radius_m, native_epsg,
    )

    result = _rasterize_laz_file(
        laz_path=filepath,
        center_easting=center_e,
        center_northing=center_n,
        radius_m=radius_m,
        min_cell_size=min_cell_size,
        dsm_gap_fill=dsm_gap_fill,
        expected_epsg=native_epsg,
    )

    if result is None:
        # Rasterization failed — build a minimal DSM from raw points
        _LOGGER.warning(
            "Full rasterization pipeline returned None for %s. "
            "Falling back to simple max-height DSM.",
            filepath,
        )
        return _process_lidar_file_simple(filepath, latitude, longitude)

    dsm, dtm, cls_grid, canopy_base, x_min, y_min, x_max, y_max, resolution = result

    extent_x = (x_max - x_min) / 2
    extent_y = (y_max - y_min) / 2

    return SiteModel(
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
        native_epsg=native_epsg,
    )


def _process_lidar_file_simple(
    filepath: str,
    latitude: float,
    longitude: float,
) -> SiteModel:
    """Fallback: simple max-height DSM when full pipeline cannot process the file."""
    import laspy

    las = laspy.read(filepath)
    x = np.array(las.x, dtype=np.float64)
    y = np.array(las.y, dtype=np.float64)
    z = np.array(las.z, dtype=np.float32)

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    area_m2 = max((x_max - x_min) * (y_max - y_min), 1.0)
    density = len(x) / area_m2
    resolution = round(max(0.5, 2.0 / max(density ** 0.5, 0.01)) * 2) / 2
    resolution = max(0.5, min(resolution, 5.0))

    cols = int(np.ceil((x_max - x_min) / resolution)) + 1
    rows = int(np.ceil((y_max - y_min) / resolution)) + 1

    col_idx = np.clip(((x - x_min) / resolution).astype(int), 0, cols - 1)
    row_idx = np.clip(((y_max - y) / resolution).astype(int), 0, rows - 1)
    flat_idx = row_idx * cols + col_idx

    dsm = np.full(rows * cols, -np.inf, dtype=np.float32)
    np.maximum.at(dsm, flat_idx, z)
    dsm = dsm.reshape(rows, cols)
    dsm[dsm == -np.inf] = np.nan

    ground = float(np.nanmin(dsm))
    dsm = np.where(np.isnan(dsm), ground, dsm)

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
    if site.native_epsg:
        save_data["native_epsg"] = np.array(site.native_epsg)

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
    native_epsg = int(data["native_epsg"]) if "native_epsg" in data else 0

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
        native_epsg=native_epsg,
    )


# ── Eraser support ──────────────────────────────────────────────────


def save_eraser_mask(mask: np.ndarray, data_dir: str) -> None:
    """Save the eraser mask to a separate file alongside the DSM."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    mask_path = data_path / "eraser_mask.npy"
    np.save(str(mask_path), mask.astype(np.bool_))
    _LOGGER.info("Saved eraser mask (%d erased pixels) to %s",
                 int(mask.sum()), mask_path)


def load_eraser_mask(data_dir: str) -> np.ndarray | None:
    """Load eraser mask from disk, or None if not present."""
    mask_path = Path(data_dir) / "eraser_mask.npy"
    if not mask_path.exists():
        return None
    try:
        mask = np.load(str(mask_path))
        _LOGGER.info("Loaded eraser mask: %d erased pixels", int(mask.sum()))
        return mask.astype(np.bool_)
    except (OSError, ValueError) as err:
        _LOGGER.warning("Could not load eraser mask: %s", err)
        return None


def delete_eraser_mask(data_dir: str) -> bool:
    """Delete the eraser mask file. Returns True if a file was deleted."""
    mask_path = Path(data_dir) / "eraser_mask.npy"
    if mask_path.exists():
        mask_path.unlink()
        _LOGGER.info("Deleted eraser mask")
        return True
    return False


def apply_eraser_to_site(site: SiteModel, mask: np.ndarray) -> None:
    """Apply an eraser mask to a site model in-place.

    Erased pixels get flattened to ground level:
    - DSM set to DTM (ground) value
    - Classification set to CLASS_GROUND
    - Canopy base set to ground value
    """
    if mask.shape != site.dsm.shape:
        _LOGGER.warning(
            "Eraser mask shape %s doesn't match DSM %s — skipping",
            mask.shape, site.dsm.shape,
        )
        return

    ground = site.ground  # DTM if available, else DSM
    site.dsm[mask] = ground[mask]
    if site.classification is not None:
        site.classification[mask] = CLASS_GROUND
    if site.canopy_base is not None:
        site.canopy_base[mask] = ground[mask]


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

    light_dx = math.sin(az)
    light_dy = -math.cos(az)

    step_scale = math.sqrt(light_dx**2 + light_dy**2)
    if step_scale < 1e-10:
        return np.zeros((rows, cols), dtype=np.float32)

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
    if not (tan_el > 0):  # catches <= 0 and NaN
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
    max_feature_height = float(np.nanmax(cast) - np.nanmin(receive))
    if not (max_feature_height > 0):  # catches <= 0 and NaN
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
    spot_windows: dict[str, dict] | None = None,
) -> dict[str, dict]:
    """Compute shade stats for each zone given current sun position.

    Supports two zone surface types:
    - "ground" zones: shadow computed with DTM as receive surface (default)
    - "dsm" zones: shadow computed with DSM as receive surface (rooftops)

    day_of_year is used to adjust high-veg transmittance for seasonal leaf cover.

    ``spot_windows`` (from :func:`compute_zone_spot_windows`) pins the sunniest
    and shadiest sub-patches to fixed locations chosen by cumulative daily solar
    exposure, so they represent a stable piece of ground rather than wherever is
    momentarily brightest/darkest. When omitted, sunniest/shadiest fall back to
    the instantaneous min/max over a sliding spot.

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

    shadow_map_ground, shadow_map_dsm = _shadow_maps_for_sun(
        site, sun_azimuth_deg, sun_elevation_deg, min_shadow_height,
        transmittance, canopy_model, has_ground_zones, has_dsm_zones,
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
            win = spot_windows.get(zone.zone_id) if spot_windows else None
            if win is not None:
                # Fixed representative patches from cumulative daily exposure.
                sunniest = round(_window_mean_at(zone_shadow, zone.mask, *win["sunniest"]), 3)
                shadiest = round(_window_mean_at(zone_shadow, zone.mask, *win["shadiest"]), 3)
            else:
                # Fallback: instantaneous extremes over a sliding spot.
                sunniest, shadiest = _spot_extremes(
                    zone_shadow, zone.mask, site.resolution,
                    getattr(zone, "spot_area", 1.0), avg,
                )

            results[zone.zone_id] = {
                "average": avg,
                "sunniest": sunniest,
                "shadiest": shadiest,
            }

    return results


def _shadow_maps_for_sun(
    site: SiteModel,
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
    min_shadow_height: float,
    transmittance: np.ndarray,
    canopy_model: str,
    need_ground: bool,
    need_dsm: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Compute the ground and/or DSM-surface shadow maps for one sun position."""
    canopy = site.canopy_base if canopy_model == "raised" else None
    ground_map = None
    dsm_map = None
    if need_ground:
        ground_map = compute_shadow_map(
            site.dsm, sun_azimuth_deg, sun_elevation_deg, site.resolution,
            ground=site.ground,
            min_shadow_height=min_shadow_height,
            transmittance=transmittance,
            canopy_base=canopy,
        )
    if need_dsm:
        dsm_map = compute_shadow_map(
            site.dsm, sun_azimuth_deg, sun_elevation_deg, site.resolution,
            ground=site.dsm,
            min_shadow_height=0.0,
            transmittance=transmittance,
            canopy_base=canopy,
        )
    return ground_map, dsm_map


def compute_zone_spot_windows(
    site: SiteModel,
    sun_samples,
    min_shadow_height: float = 1.5,
    day_of_year: int = 172,
    canopy_model: str = "solid",
) -> dict[str, dict]:
    """Pick a FIXED sunniest/shadiest patch per zone from daily solar exposure.

    The sunniest and shadiest spots should describe a stable piece of ground,
    not wherever happens to be brightest at this instant, which roams across the
    zone as the sun moves. This integrates shade over a day's sun path and picks
    the window with the lowest (sunniest) / highest (shadiest) cumulative shade.

    ``sun_samples`` is an iterable of ``(azimuth_deg, elevation_deg, weight)``;
    weight should track solar radiation (e.g. ``sin(elevation)``) so midday
    counts more. Samples at/below the horizon are skipped.

    Returns ``{zone_id: {"sunniest": (r0, c0, w), "shadiest": (r0, c0, w)}}``
    where coordinates are top-left offsets in the zone sub-array. Zones with no
    daylight are omitted.
    """
    if not site.zones:
        return {}

    transmittance = build_transmittance_grid(site, day_of_year)
    need_ground = any(z.surface != "dsm" for z in site.zones)
    need_dsm = any(z.surface == "dsm" for z in site.zones)

    accum: dict[str, np.ndarray] = {}
    total_w = 0.0
    for az, el, w in sun_samples:
        if el <= MIN_SUN_ELEVATION or w <= 0:
            continue
        gmap, dmap = _shadow_maps_for_sun(
            site, az, el, min_shadow_height, transmittance,
            canopy_model, need_ground, need_dsm,
        )
        total_w += w
        for zone in site.zones:
            smap = dmap if zone.surface == "dsm" else gmap
            sub = smap[zone.row_start:zone.row_end, zone.col_start:zone.col_end]
            if zone.zone_id not in accum:
                accum[zone.zone_id] = np.zeros(sub.shape, dtype=np.float64)
            accum[zone.zone_id] += sub.astype(np.float64) * w

    if total_w <= 0:
        return {}

    result: dict[str, dict] = {}
    for zone in site.zones:
        daily = accum.get(zone.zone_id)
        if daily is None:
            continue
        daily = daily / total_w  # daily-mean shade per pixel
        rows, cols = daily.shape
        if rows == 0 or cols == 0:
            continue
        w = _window_size(rows, cols, site.resolution, getattr(zone, "spot_area", 1.0))
        means, fully_valid, cnt = _window_mean_grid(daily, zone.mask, w)
        sun_pos = _argbest(means, fully_valid, cnt, want_min=True)
        sha_pos = _argbest(means, fully_valid, cnt, want_min=False)
        if sun_pos is None or sha_pos is None:
            continue
        result[zone.zone_id] = {
            "sunniest": (int(sun_pos[0]), int(sun_pos[1]), w),
            "shadiest": (int(sha_pos[0]), int(sha_pos[1]), w),
        }
    return result



def _integral_image(a: np.ndarray) -> np.ndarray:
    """Return a zero-padded summed-area table for fast window sums."""
    ii = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    ii[1:, 1:] = a.cumsum(axis=0).cumsum(axis=1)
    return ii


def _window_size(rows: int, cols: int, resolution: float, spot_area_m2: float) -> int:
    """Side length (pixels) of a square window of ``spot_area_m2``, clamped."""
    side_m = math.sqrt(max(spot_area_m2, 0.0))
    w = max(1, int(round(side_m / resolution))) if resolution > 0 else 1
    return min(w, rows, cols)


def _window_mean_grid(
    values: np.ndarray, mask: np.ndarray | None, w: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean of ``values`` over every w×w window (masked pixels only).

    Returns (means, fully_valid, counts): means[i,j] is the window mean,
    fully_valid[i,j] is True when the window lies entirely in the mask, and
    counts[i,j] is the number of masked pixels in that window.
    """
    rows, cols = values.shape
    mask_f = np.ones((rows, cols), dtype=np.float64) if mask is None else mask.astype(np.float64)
    vals = np.where(mask_f > 0, values.astype(np.float64), 0.0)

    ii_v = _integral_image(vals)
    ii_m = _integral_image(mask_f)

    def _win(ii: np.ndarray) -> np.ndarray:
        return ii[w:, w:] - ii[:-w, w:] - ii[w:, :-w] + ii[:-w, :-w]

    sum_v = _win(ii_v)
    cnt = _win(ii_m)
    means = np.where(cnt > 0, sum_v / np.where(cnt > 0, cnt, 1.0), 0.0)
    fully_valid = cnt >= (w * w) - 1e-6
    return means, fully_valid, cnt


def _argbest(
    means: np.ndarray, fully_valid: np.ndarray, cnt: np.ndarray, want_min: bool
) -> tuple[int, int] | None:
    """Top-left of the min/max window, preferring fully-masked windows."""
    allowed = fully_valid if fully_valid.any() else (cnt > 0)
    if not allowed.any():
        return None
    fill = np.inf if want_min else -np.inf
    m = np.where(allowed, means, fill)
    flat = int(np.argmin(m) if want_min else np.argmax(m))
    return np.unravel_index(flat, m.shape)


def _window_mean_at(
    values: np.ndarray, mask: np.ndarray | None, r0: int, c0: int, w: int
) -> float:
    """Mean of ``values`` over a fixed w×w window (masked pixels only)."""
    sub = values[r0:r0 + w, c0:c0 + w]
    if sub.size == 0:
        return float(values.mean()) if values.size else 0.0
    if mask is not None:
        m = mask[r0:r0 + w, c0:c0 + w]
        if m.any():
            return float(sub[m].mean())
    return float(sub.mean())


def _spot_extremes(
    values: np.ndarray,
    mask: np.ndarray | None,
    resolution: float,
    spot_area_m2: float,
    avg: float,
) -> tuple[float, float]:
    """Instantaneous min (sunniest) / max (shadiest) mean shade over a spot.

    The spot is a square window whose area is ``spot_area_m2`` m². For masked
    zones only windows lying fully inside the mask are considered; if none fit,
    it falls back to the best partially-covered window, then to ``avg``.
    """
    rows, cols = values.shape
    if rows == 0 or cols == 0:
        return avg, avg

    w = _window_size(rows, cols, resolution, spot_area_m2)
    if w <= 0:
        return avg, avg

    means, fully_valid, cnt = _window_mean_grid(values, mask, w)
    allowed = fully_valid if fully_valid.any() else (cnt > 0)
    if not allowed.any():
        return avg, avg
    vals = means[allowed]
    return round(float(vals.min()), 3), round(float(vals.max()), 3)




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
            shade_method=z.get("shade_method", "average"),
            spot_area=float(z.get("spot_area", 1.0)),
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
                shade_method=z.get("shade_method", "average"),
                spot_area=float(z.get("spot_area", 1.0)),
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
    shade_method: str = "average",
    spot_area: float = 1.0,
) -> ZoneDef | None:
    """Convert a lat/lng polygon into a pixel-masked ZoneDef.

    Uses point-in-polygon (ray casting) to create a boolean mask on the DSM grid.
    """
    if len(polygon_latlng) < 3:
        _LOGGER.warning("Zone '%s' has fewer than 3 vertices, skipping", zone_name)
        return None

    rows, cols = site.dsm.shape

    # Convert lat/lng to DSM pixel coordinates using proper CRS projection
    from .geo import latlon_to_epsg, latlon_to_utm

    if site.native_epsg:
        center_e, center_n = latlon_to_epsg(site.latitude, site.longitude, site.native_epsg)
        proj_epsg = site.native_epsg
    else:
        zone, center_e, center_n = latlon_to_utm(site.latitude, site.longitude)
        proj_epsg = (32600 + zone) if site.latitude >= 0 else (32700 + zone)

    pixel_coords = []
    for lat, lng in polygon_latlng:
        if site.native_epsg:
            px_e, px_n = latlon_to_epsg(lat, lng, proj_epsg)
        else:
            _, px_e, px_n = latlon_to_utm(lat, lng)
        mx = px_e - center_e
        my = px_n - center_n

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
        shade_method=shade_method,
        spot_area=spot_area,
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
