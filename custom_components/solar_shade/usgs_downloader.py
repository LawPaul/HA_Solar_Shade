"""Auto-download LiDAR data from USGS The National Map.

Download strategy (LAZ-first):
1. Query TNM API for LAZ tiles covering a lat/lon → get real download URLs
2. Download classified LAZ point cloud tile from rockyweb.usgs.gov
3. Rasterize ground points (class 2) → DTM, first-return points → DSM
4. Fallback: ImageServer DTM

LAZ point clouds contain per-point classification (ground, vegetation,
buildings) and return number, giving true DSM with trees/buildings for
accurate shadow casting.
"""

from __future__ import annotations

import logging
import math

import aiohttp
import numpy as np

_LOGGER = logging.getLogger(__name__)

TNM_API_URL = "https://tnmaccess.nationalmap.gov/api/v1/products"

# ASPRS LAS classification codes
CLASS_GROUND = 2
CLASS_LOW_VEG = 3
CLASS_MED_VEG = 4
CLASS_HIGH_VEG = 5
CLASS_BUILDING = 6


async def find_laz_urls(
    latitude: float,
    longitude: float,
    session: aiohttp.ClientSession,
) -> list[dict]:
    """Find LAZ tile download URLs covering a lat/lon via the USGS TNM API.

    Returns a list of dicts with 'url', 'title', 'date', 'size' sorted by
    publication date (newest first), or an empty list if no coverage.
    """
    delta = 0.002  # ~200m bbox
    params = {
        "datasets": "Lidar Point Cloud (LPC)",
        "bbox": f"{longitude - delta},{latitude - delta},{longitude + delta},{latitude + delta}",
        "max": "10",
        "outputFormat": "JSON",
    }

    import asyncio
    import json as _json

    data = None
    for attempt in range(3):
        try:
            _LOGGER.debug(
                "Querying TNM API for LAZ tiles at %.4f, %.4f (attempt %d)",
                latitude, longitude, attempt + 1,
            )
            async with session.get(
                TNM_API_URL, params=params,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                body = await resp.text()
                # TNM API returns non-JSON error bodies when its backend
                # (sciencebase.gov) is overloaded — detect and retry.
                try:
                    data = _json.loads(body)
                except _json.JSONDecodeError:
                    raise RuntimeError(
                        f"TNM API returned non-JSON (HTTP {resp.status}): "
                        f"{body[:120]}"
                    )
                if "errorMessage" in data and "items" not in data:
                    raise RuntimeError(
                        f"TNM API error: {str(data.get('errorMessage', ''))[:120]}"
                    )
                break
        except Exception as err:
            _LOGGER.warning("TNM API attempt %d failed: %s", attempt + 1, err)
            data = None
            if attempt < 2:
                await asyncio.sleep(5 * (attempt + 1))

    if data is None:
        _LOGGER.warning("TNM API unavailable after 3 attempts")
        return []

    items = data.get("items", [])
    if not items:
        _LOGGER.warning("No LAZ tiles found at %.4f, %.4f", latitude, longitude)
        return []

    results = []
    for item in items:
        url = item.get("downloadURL") or item.get("downloadLazURL") or ""
        if not url or not url.lower().endswith((".laz", ".las")):
            continue
        results.append({
            "url": url,
            "title": item.get("title", ""),
            "date": item.get("publicationDate", ""),
            "size": item.get("sizeInBytes", 0),
        })

    # Sort by publication date descending (newest first)
    results.sort(key=lambda x: x.get("date", ""), reverse=True)

    _LOGGER.info(
        "Found %d LAZ tile(s) at %.4f, %.4f: %s",
        len(results), latitude, longitude,
        ", ".join(r["url"].rsplit("/", 1)[-1] for r in results[:3]),
    )
    return results


async def download_usgs_dsm(
    latitude: float,
    longitude: float,
    radius_m: float = 150.0,
    lidar_project: str = "",
    min_cell_size: float = 0.5,
    dsm_gap_fill: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, float, float, float, float, float] | None:
    """Auto-download DSM and DTM from USGS for the area around lat/lng.

    Download strategy (LAZ-first):
    1. Query TNM API for LAZ tiles → get real download URL
    2. Download LAZ point cloud tile → rasterize to DSM + DTM
    3. Fallback: ImageServer DTM

    Returns:
        (dsm, dtm, x_min, y_min, x_max, y_max), or None if no data.
        dtm may be None if bare-earth data isn't available.
    """
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # ── Discover LAZ tile URL via TNM API ─────────────────────
        laz_urls = await find_laz_urls(latitude, longitude, session)

        if not laz_urls:
            _LOGGER.warning(
                "No LAZ tiles found at %.4f, %.4f. "
                "Check LiDAR coverage at https://apps.nationalmap.gov/lidar-explorer/",
                latitude, longitude,
            )
            return None

        # ── Download and rasterize LAZ (try each URL) ─────────────
        for tile_info in laz_urls:
            laz_url = tile_info["url"]
            _LOGGER.info(
                "Attempting LAZ download: %s", laz_url.rsplit("/", 1)[-1]
            )
            laz_result = await _download_and_rasterize_laz(
                latitude, longitude, radius_m, session, laz_url,
                min_cell_size=min_cell_size,
                dsm_gap_fill=dsm_gap_fill,
            )
            if laz_result is not None:
                dsm, dtm, cls_grid, canopy_base, x_min, y_min, x_max, y_max, eff_res = laz_result
                _LOGGER.info(
                    "LAZ pipeline complete: DSM %dx%d (%.1fm res), "
                    "DTM %dx%d, height range %.1f-%.1fm above ground",
                    dsm.shape[0], dsm.shape[1], eff_res,
                    dtm.shape[0], dtm.shape[1],
                    float((dsm - dtm).min()), float((dsm - dtm).max()),
                )
                return dsm, dtm, cls_grid, canopy_base, x_min, y_min, x_max, y_max, eff_res

        _LOGGER.warning("All LAZ download attempts failed")
        return None


# ── LAZ Point Cloud Download & Rasterization ─────────────────────────────


async def _download_and_rasterize_laz(
    latitude: float,
    longitude: float,
    radius_m: float,
    session: aiohttp.ClientSession,
    laz_url: str,
    min_cell_size: float = 0.5,
    dsm_gap_fill: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float, float] | None:
    """Download a LAZ file from a URL and rasterize to DSM + DTM grids."""
    zone, easting, northing, is_northern = _latlon_to_utm(latitude, longitude)

    filename = laz_url.rsplit("/", 1)[-1]
    _LOGGER.info("LAZ: downloading %s", filename)

    import tempfile
    import os
    try:
        async with session.get(
            laz_url, timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            if resp.status == 404:
                _LOGGER.info("LAZ: tile URL returned 404: %s", laz_url)
                return None
            resp.raise_for_status()

            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".laz")
            try:
                total_bytes = 0
                with os.fdopen(tmp_fd, 'wb') as tmp_file:
                    async for chunk in resp.content.iter_chunked(1024 * 1024):
                        tmp_file.write(chunk)
                        total_bytes += len(chunk)
                _LOGGER.info("LAZ: downloaded %.1f MB", total_bytes / 1024 / 1024)
            except Exception:
                os.unlink(tmp_path)
                raise
    except Exception as err:
        _LOGGER.warning("LAZ download failed: %s", err)
        return None

    # Rasterize in executor (CPU-bound)
    import asyncio
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            _rasterize_laz_file,
            tmp_path, easting, northing, radius_m, min_cell_size,
            dsm_gap_fill,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return result


def _read_laz_epsg(laz_path: str, laspy_module) -> int | None:
    """Extract EPSG code from a LAZ file's VLR metadata.

    Checks WKT VLR (LAS 1.4) and GeoTIFF GeoKeyDirectory VLR (LAS 1.2+).
    Returns the EPSG code or None if not found.
    """
    import re as _re
    try:
        with laspy_module.open(laz_path) as reader:
            for vlr in reader.header.vlrs:
                # WKT CRS (LAS 1.4, record_id 2112)
                if vlr.user_id == "LASF_Projection" and vlr.record_id == 2112:
                    try:
                        wkt = vlr.record_data.decode("utf-8", errors="ignore").rstrip("\x00")
                        # Extract EPSG from AUTHORITY["EPSG","XXXXX"]
                        match = _re.search(r'AUTHORITY\s*\[\s*"EPSG"\s*,\s*"(\d+)"\s*\]', wkt)
                        if match:
                            epsg = int(match.group(1))
                            _LOGGER.info("LAZ WKT VLR: found EPSG:%d", epsg)
                            return epsg
                        # Also try ID["EPSG",XXXXX] (WKT2 style)
                        match = _re.search(r'ID\s*\[\s*"EPSG"\s*,\s*(\d+)\s*\]', wkt)
                        if match:
                            epsg = int(match.group(1))
                            _LOGGER.info("LAZ WKT2 VLR: found EPSG:%d", epsg)
                            return epsg
                    except Exception as e:
                        _LOGGER.debug("Failed to parse WKT VLR: %s", e)

                # GeoTIFF GeoKeyDirectory (record_id 34735)
                if vlr.user_id == "LASF_Projection" and vlr.record_id == 34735:
                    try:
                        data = vlr.record_data
                        if len(data) >= 8:
                            import struct
                            # Parse GeoKey directory header
                            key_dir_version, key_revision, minor_rev, num_keys = \
                                struct.unpack('<HHHH', data[:8])
                            for i in range(num_keys):
                                offset = 8 + i * 8
                                if offset + 8 > len(data):
                                    break
                                key_id, tif_tag, count, value = \
                                    struct.unpack('<HHHH', data[offset:offset + 8])
                                # Key 3072 = ProjectedCSTypeGeoKey (EPSG code for projected CRS)
                                if key_id == 3072 and tif_tag == 0 and value > 0:
                                    _LOGGER.info("LAZ GeoKey VLR: found EPSG:%d (ProjectedCSType)", value)
                                    return value
                                # Key 2048 = GeographicTypeGeoKey
                                if key_id == 2048 and tif_tag == 0 and value > 0:
                                    _LOGGER.info("LAZ GeoKey VLR: found EPSG:%d (GeographicType)", value)
                                    return value
                    except Exception as e:
                        _LOGGER.debug("Failed to parse GeoKey VLR: %s", e)
    except Exception as e:
        _LOGGER.debug("Failed to read LAZ VLRs: %s", e)
    return None


def _project_to_epsg(
    utm_easting: float, utm_northing: float, target_epsg: int,
) -> tuple[float | None, float | None]:
    """Reproject UTM coordinates if the target EPSG is a different UTM zone.

    For USGS 3DEP data, the file CRS is typically UTM NAD83 (EPSG 269xx or
    326xx). If the file's UTM zone matches our computed zone, no reprojection
    is needed. If it differs (e.g., near zone boundaries), we reproject.

    Returns (easting, northing) in the target CRS, or (None, None) if
    reprojection is not needed or not possible.
    """
    # Extract UTM zone from EPSG code
    # EPSG 326xx = WGS84 UTM North zone xx
    # EPSG 327xx = WGS84 UTM South zone xx
    # EPSG 269xx = NAD83 UTM zone xx (NAD83 ≈ WGS84 for our purposes)
    if 32601 <= target_epsg <= 32660:
        # WGS84 UTM North — same datum, just different zone possible
        return None, None  # Our UTM calc uses WGS84, should match
    if 32701 <= target_epsg <= 32760:
        return None, None  # WGS84 UTM South
    if 26901 <= target_epsg <= 26999:
        # NAD83 UTM — effectively identical to WGS84 UTM for our scale
        # NAD83 and WGS84 differ by ~1-2m in CONUS, which is within noise
        return None, None

    # For other CRS (State Plane, etc.), we can't easily reproject without
    # pyproj. Log a warning so the user knows.
    _LOGGER.warning(
        "LAZ file uses EPSG:%d which is not a standard UTM zone. "
        "Coordinate alignment may be imprecise. Consider installing pyproj "
        "for accurate reprojection.",
        target_epsg,
    )
    return None, None


def _rasterize_laz_file(
    laz_path: str,
    center_easting: float,
    center_northing: float,
    radius_m: float,
    min_cell_size: float = 0.5,
    dsm_gap_fill: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float, float] | None:
    """Read a LAZ file and rasterize classified points into DSM and DTM grids.

    Resolution is auto-calculated from point density (target: ~4 pts/cell,
    floored at *min_cell_size*, snapped to 0.5m increments).

    Reads in chunks to keep memory low — only points within the bounding box
    are kept. Peak memory is proportional to radius, not tile size.

    DSM: maximum Z from first-return points per cell.
    DTM: mean Z from ground-classified (class 2) points per cell.
    """
    try:
        import laspy
    except ImportError:
        _LOGGER.error(
            "laspy not installed — cannot process LAZ point clouds. "
            "Install with: pip install laspy laszip"
        )
        return None

    # Try to read CRS from the LAZ file and reproject center if needed.
    # USGS 3DEP files include WKT or GeoTIFF VLRs describing their CRS.
    # If the file's CRS differs from our computed UTM zone, using the
    # wrong center coordinates would shift the entire grid by meters.
    file_epsg = _read_laz_epsg(laz_path, laspy)
    if file_epsg is not None:
        file_center_e, file_center_n = _project_to_epsg(
            center_easting, center_northing, file_epsg,
        )
        if file_center_e is not None:
            _LOGGER.info(
                "LAZ CRS: EPSG:%d — reprojected center from UTM (%.1f, %.1f) "
                "to file CRS (%.1f, %.1f), shift: %.1fm E, %.1fm N",
                file_epsg, center_easting, center_northing,
                file_center_e, file_center_n,
                file_center_e - center_easting,
                file_center_n - center_northing,
            )
            center_easting = file_center_e
            center_northing = file_center_n

    # Bounding box for clipping — only keep points within radius
    x_min_clip = center_easting - radius_m
    x_max_clip = center_easting + radius_m
    y_min_clip = center_northing - radius_m
    y_max_clip = center_northing + radius_m

    # Read LAZ in chunks to avoid loading entire tile into memory.
    # A full 1500m tile can be 20M+ points (~1GB as arrays). With a
    # 200m radius we only need points in a 400m box — typically <5%
    # of the tile. Chunked reading keeps peak memory under 100MB.
    CHUNK_SIZE = 500_000
    x_list, y_list, z_list, cls_list, ret_list = [], [], [], [], []
    total_points = 0
    kept_points = 0

    try:
        with laspy.open(laz_path) as reader:
            _LOGGER.info(
                "LAZ: %d total points, point format %s, reading in chunks of %d",
                reader.header.point_count, reader.header.point_format.id,
                CHUNK_SIZE,
            )
            for chunk in reader.chunk_iterator(CHUNK_SIZE):
                total_points += len(chunk)

                # Extract and clip immediately — discard out-of-bounds
                cx = np.asarray(chunk.x, dtype=np.float64)
                cy = np.asarray(chunk.y, dtype=np.float64)

                in_bounds = (
                    (cx >= x_min_clip) & (cx <= x_max_clip) &
                    (cy >= y_min_clip) & (cy <= y_max_clip)
                )

                if not np.any(in_bounds):
                    continue

                x_list.append(cx[in_bounds])
                y_list.append(cy[in_bounds])
                z_list.append(np.asarray(chunk.z, dtype=np.float32)[in_bounds])
                cls_list.append(
                    np.asarray(chunk.classification, dtype=np.uint8)[in_bounds]
                )
                if hasattr(chunk, 'return_number'):
                    ret_list.append(
                        np.asarray(chunk.return_number, dtype=np.uint8)[in_bounds]
                    )
                else:
                    ret_list.append(np.ones(int(np.sum(in_bounds)), dtype=np.uint8))

                kept_points += int(np.sum(in_bounds))

    except Exception as err:
        _LOGGER.error("Failed to read LAZ data: %s", err)
        return None

    if kept_points < 100:
        _LOGGER.warning(
            "LAZ: only %d points within %.0fm radius (of %d total) "
            "— insufficient data",
            kept_points, radius_m, total_points,
        )
        return None

    # Concatenate kept chunks
    x = np.concatenate(x_list)
    y = np.concatenate(y_list)
    z = np.concatenate(z_list)
    classification = np.concatenate(cls_list)
    return_number = np.concatenate(ret_list)
    # Free chunk lists
    del x_list, y_list, z_list, cls_list, ret_list

    _LOGGER.info(
        "LAZ: kept %d / %d points (%.1f%%) within %.0fm radius",
        kept_points, total_points,
        100 * kept_points / total_points if total_points else 0,
        radius_m,
    )

    # Log classification distribution
    classes, counts = np.unique(classification, return_counts=True)
    class_names = {
        0: "never_classified", 1: "unassigned", 2: "ground",
        3: "low_veg", 4: "med_veg", 5: "high_veg", 6: "building",
        7: "low_noise", 9: "water", 17: "bridge", 18: "high_noise",
    }
    for cls, cnt in zip(classes, counts):
        name = class_names.get(int(cls), f"class_{cls}")
        _LOGGER.info("  class %d (%s): %d points (%.1f%%)",
                     cls, name, cnt, 100 * cnt / len(x))

    # Auto-calculate resolution from point density.
    # Target: ~4 points per cell.  resolution = 2 / sqrt(density).
    # Snap to 0.5m increments, clamp to [0.5, 5.0].
    area_m2 = (x_max_clip - x_min_clip) * (y_max_clip - y_min_clip)
    density = kept_points / area_m2 if area_m2 > 0 else 0
    if density > 0:
        auto_res = 2.0 / max(density ** 0.5, 0.01)
        resolution = round(auto_res * 2) / 2  # snap to 0.5m
        resolution = max(min_cell_size, min(resolution, 5.0))
    else:
        resolution = max(min_cell_size, 1.0)
    _LOGGER.info(
        "LAZ: point density %.1f pts/m² → resolution %.1fm",
        density, resolution,
    )

    # Define output grid
    grid_x_min = x_min_clip
    grid_y_min = y_min_clip
    grid_x_max = x_max_clip
    grid_y_max = y_max_clip

    cols = int(round((grid_x_max - grid_x_min) / resolution))
    rows = int(round((grid_y_max - grid_y_min) / resolution))
    if cols < 2 or rows < 2:
        _LOGGER.warning("LAZ: grid too small (%dx%d)", rows, cols)
        return None

    # Compute cell indices for each point
    col_idx = np.clip(((x - grid_x_min) / resolution).astype(np.int32), 0, cols - 1)
    # Y axis is flipped: row 0 = north (max Y)
    row_idx = np.clip(((grid_y_max - y) / resolution).astype(np.int32), 0, rows - 1)
    cell_idx = row_idx * cols + col_idx

    # ── Build DTM from ground points (class 2) ────────────────────
    ground_mask = classification == CLASS_GROUND
    dtm = np.full(rows * cols, np.nan, dtype=np.float32)

    if np.sum(ground_mask) > 0:
        ground_cells = cell_idx[ground_mask]
        ground_z = z[ground_mask]
        # Accumulate sum and count per cell for mean calculation
        dtm_sum = np.zeros(rows * cols, dtype=np.float64)
        dtm_count = np.zeros(rows * cols, dtype=np.int32)
        np.add.at(dtm_sum, ground_cells, ground_z.astype(np.float64))
        np.add.at(dtm_count, ground_cells, 1)
        valid = dtm_count > 0
        dtm[valid] = (dtm_sum[valid] / dtm_count[valid]).astype(np.float32)
        _LOGGER.info(
            "DTM: %d ground points → %d/%d cells filled (%.0f%%)",
            np.sum(ground_mask), np.sum(valid), rows * cols,
            100 * np.sum(valid) / (rows * cols),
        )
    else:
        _LOGGER.warning(
            "LAZ: no ground-classified points; estimating ground from "
            "lowest 5th-percentile Z per cell"
        )
        # Estimate ground from lowest points when no classification exists.
        # For each cell, use minimum Z as the ground estimate; then smooth
        # by replacing outliers with the global 5th-percentile.
        dtm_min = np.full(rows * cols, np.inf, dtype=np.float32)
        np.minimum.at(dtm_min, cell_idx, z)
        has_points = dtm_min < np.inf
        if np.any(has_points):
            p5 = float(np.percentile(z, 5))
            # Cells with points: use minimum Z, clamped to p5 floor
            dtm[has_points] = np.maximum(dtm_min[has_points], p5)
            _LOGGER.info(
                "DTM (estimated): 5th-percentile=%.1fm, %d/%d cells filled",
                p5, int(np.sum(has_points)), rows * cols,
            )

    # Fill DTM gaps with nearest-neighbor interpolation
    dtm = dtm.reshape(rows, cols)
    dtm = _fill_nan_nearest(dtm)

    # ── Build DSM from first-return / non-ground points ──────────
    # DSM = maximum Z per cell from all first-return points
    # (first returns hit the highest surface: tree canopy, roof, etc.)
    dsm_mask = (return_number == 1) | (return_number == 0)
    if np.sum(dsm_mask) < len(x) * 0.1:
        # If very few first returns, use all points
        dsm_mask = np.ones(len(x), dtype=bool)

    dsm = np.full(rows * cols, -np.inf, dtype=np.float32)
    dsm_cells = cell_idx[dsm_mask]
    dsm_z = z[dsm_mask]
    # Maximum Z per cell (captures tree tops, rooftops)
    np.maximum.at(dsm, dsm_cells, dsm_z)

    # Cells with no points remain -inf → replace with NaN
    dsm[dsm == -np.inf] = np.nan
    dsm = dsm.reshape(rows, cols)

    # ── Build per-cell classification (needed for gap-filling) ───
    # For each cell, use the classification of the highest point
    cell_max_z = np.full(rows * cols, -np.inf, dtype=np.float32)
    cell_top_cls = np.full(rows * cols, CLASS_GROUND, dtype=np.uint8)

    all_cells = cell_idx
    all_z = z
    all_cls = classification
    for i in range(len(all_cells)):
        c = all_cells[i]
        if all_z[i] > cell_max_z[c]:
            cell_max_z[c] = all_z[i]
            cell_top_cls[c] = all_cls[i]

    cls_grid = cell_top_cls.reshape(rows, cols)

    # ── Build canopy base grid (bottom of tree canopy) ───────────
    # For veg cells: lowest non-ground return >2m above DTM (canopy base).
    # For non-veg cells with data: same as DSM (solid column).
    # For cells with no data: NaN (will be gap-filled just like DSM).
    TRUNK_MIN_HEIGHT = 2.0
    canopy_base = np.full_like(dsm, np.nan)

    # Non-veg cells with DSM data: canopy_base = DSM (solid)
    VEG_CLASSES_SET = {CLASS_LOW_VEG, CLASS_MED_VEG, CLASS_HIGH_VEG}
    has_dsm = ~np.isnan(dsm)
    is_veg_raw = np.isin(cls_grid, list(VEG_CLASSES_SET))
    canopy_base[has_dsm & ~is_veg_raw] = dsm[has_dsm & ~is_veg_raw]

    # Veg cells: lowest non-ground return above trunk threshold
    dtm_flat = dtm.reshape(-1)
    non_ground = classification != CLASS_GROUND
    non_ground_cells = cell_idx[non_ground]
    non_ground_z = z[non_ground]
    above_trunk = non_ground_z > (dtm_flat[non_ground_cells] + TRUNK_MIN_HEIGHT)
    trunk_cells = non_ground_cells[above_trunk]
    trunk_z = non_ground_z[above_trunk]

    min_canopy_z = np.full(rows * cols, np.inf, dtype=np.float32)
    if len(trunk_cells) > 0:
        np.minimum.at(min_canopy_z, trunk_cells, trunk_z)

    min_canopy_z = min_canopy_z.reshape(rows, cols)
    has_canopy = min_canopy_z < np.inf
    canopy_base[has_canopy & is_veg_raw] = min_canopy_z[has_canopy & is_veg_raw]
    # Veg cells with DSM data but no canopy returns: canopy_base = DSM (solid)
    canopy_base[has_dsm & is_veg_raw & ~has_canopy] = dsm[has_dsm & is_veg_raw & ~has_canopy]

    # ── Classification-aware gap-filling (optional) ────────────
    if dsm_gap_fill:
        dsm = _fill_dsm_classification_aware(dsm, dtm, cls_grid, resolution)
    # Fill remaining canopy_base NaN with DSM (solid column)
    cb_nan = np.isnan(canopy_base)
    canopy_base[cb_nan] = dsm[cb_nan]

    # Ensure DSM >= DTM everywhere
    dsm = np.maximum(dsm, dtm)

    classification_grid = cls_grid  # already reshaped above

    # ── Morphological closing for classification ────────────────
    # Fill holes in classification grid to match the
    # already gap-filled DSM (done by _fill_dsm_classification_aware above).
    _fill_classification_holes(dsm, dtm, classification_grid, rows, cols)

    # Finalize canopy_base: clip and ensure non-veg cells are solid
    is_veg = np.isin(classification_grid, list(VEG_CLASSES_SET))
    # Cells reclassified to veg by morph closing have no real canopy data → solid
    newly_veg = is_veg & ~is_veg_raw
    canopy_base[newly_veg] = dsm[newly_veg]
    canopy_base[~is_veg] = dsm[~is_veg]  # non-veg: solid column
    canopy_base = np.clip(canopy_base, dtm, dsm)

    veg_cells_with_trunk = int(np.sum(is_veg & (canopy_base < dsm - 0.5)))
    _LOGGER.info(
        "Canopy base: %d vegetation cells have trunk clearance (avg %.1fm)",
        veg_cells_with_trunk,
        float((dsm[is_veg & (canopy_base < dsm - 0.5)] - canopy_base[is_veg & (canopy_base < dsm - 0.5)]).mean())
        if veg_cells_with_trunk > 0 else 0.0,
    )

    _LOGGER.info(
        "Rasterized: DSM %dx%d (%.2fm res), "
        "ground %.1f-%.1fm, canopy height 0-%.1fm",
        rows, cols, resolution,
        float(dtm.min()), float(dtm.max()),
        float((dsm - dtm).max()),
    )

    return dsm, dtm, classification_grid, canopy_base, grid_x_min, grid_y_min, grid_x_max, grid_y_max, resolution


def _fill_dsm_classification_aware(
    dsm: np.ndarray,
    dtm: np.ndarray,
    cls_grid: np.ndarray,
    resolution: float = 0.5,
) -> np.ndarray:
    """Fill DSM NaN gaps using classification-aware interpolation.

    For each empty cell, examines the classification of surrounding filled
    cells to choose the best fill strategy:
      - Building neighbors (class 6): use MAX of neighbors (preserves flat roofs)
      - Vegetation neighbors (class 3-5): use MEDIAN of neighbors (realistic canopy)
      - Ground/other: use DTM or nearest neighbor

    Also updates cls_grid in-place so filled cells get the correct classification
    (needed for 3D mesh building and transmittance).

    Pass count scales with resolution to fill ~2m gaps regardless of cell size.
    """
    rows, cols = dsm.shape
    filled = dsm.copy()
    original_nan = np.isnan(dsm)  # Track which cells were originally empty
    nan_count_start = int(np.sum(original_nan))

    if nan_count_start == 0:
        return filled

    BUILDING_CLASSES = {CLASS_BUILDING}
    VEG_CLASSES = {CLASS_LOW_VEG, CLASS_MED_VEG, CLASS_HIGH_VEG}
    ELEVATED_CLASSES = BUILDING_CLASSES | VEG_CLASSES
    # Threshold (meters) above DTM that marks a point as "elevated"
    # even when unclassified — used to connect sparse unclassified features.
    UNCLASSIFIED_ELEV_THRESH = 2.0

    # Scale passes with resolution: fill ~2m gaps regardless of cell size
    # With auto-resolution targeting ~4 pts/cell, gaps are small (1-2 cells).
    # 2 passes fills interior gaps without expanding building footprints.
    max_passes = 2
    for iteration in range(max_passes):
        nan_mask = np.isnan(filled)
        if not np.any(nan_mask):
            break

        new_filled = filled.copy()
        any_change = False

        nan_rows, nan_cols = np.where(nan_mask)
        for idx in range(len(nan_rows)):
            r, c = int(nan_rows[idx]), int(nan_cols[idx])

            # Collect valid neighbors in a 3x3 window
            r_lo = max(0, r - 1)
            r_hi = min(rows, r + 2)
            c_lo = max(0, c - 1)
            c_hi = min(cols, c + 2)

            neighbor_vals = []
            building_vals = []
            veg_vals = []
            elevated_unclass_vals = []
            # Count original (non-filled) neighbors for classification decision
            orig_building = 0
            orig_veg = 0
            orig_elevated_unclass = 0

            for nr in range(r_lo, r_hi):
                for nc in range(c_lo, c_hi):
                    if nr == r and nc == c:
                        continue
                    if not np.isnan(filled[nr, nc]):
                        v = filled[nr, nc]
                        neighbor_vals.append(v)
                        ncls = int(cls_grid[nr, nc])
                        if ncls in BUILDING_CLASSES:
                            building_vals.append(v)
                            if not original_nan[nr, nc]:
                                orig_building += 1
                        elif ncls in VEG_CLASSES:
                            veg_vals.append(v)
                            if not original_nan[nr, nc]:
                                orig_veg += 1
                        elif ncls not in ELEVATED_CLASSES:
                            # Unclassified / never-classified — check if elevated
                            dtm_here = dtm[nr, nc]
                            if (not np.isnan(dtm_here)
                                    and v > dtm_here + UNCLASSIFIED_ELEV_THRESH):
                                elevated_unclass_vals.append(v)
                                if not original_nan[nr, nc]:
                                    orig_elevated_unclass += 1

            if not neighbor_vals:
                continue  # no valid neighbors yet, try next pass

            # Only fill as elevated if original elevated data exists on
            # multiple sides — meaning this cell is INSIDE the feature,
            # not on its edge. Check 4 cardinal directions for original
            # elevated neighbors (including unclassified elevated points).
            sides_with_orig_elev = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr2, nc2 = r + dr, c + dc
                if 0 <= nr2 < rows and 0 <= nc2 < cols:
                    if not original_nan[nr2, nc2] and not np.isnan(filled[nr2, nc2]):
                        ncls2 = int(cls_grid[nr2, nc2])
                        if ncls2 in ELEVATED_CLASSES:
                            sides_with_orig_elev += 1
                        elif (ncls2 not in ELEVATED_CLASSES
                              and not np.isnan(dtm[nr2, nc2])
                              and filled[nr2, nc2] > dtm[nr2, nc2] + UNCLASSIFIED_ELEV_THRESH):
                            sides_with_orig_elev += 1

            n_total = len(neighbor_vals)
            majority = n_total // 2 + 1
            # Combine building + elevated unclassified for gap decisions
            all_building_like = building_vals + elevated_unclass_vals
            if sides_with_orig_elev >= 3 and len(all_building_like) >= majority:
                new_filled[r, c] = max(all_building_like)
                cls_grid[r, c] = CLASS_BUILDING
            elif sides_with_orig_elev >= 2 and len(veg_vals) >= majority:
                sorted_v = sorted(veg_vals)
                new_filled[r, c] = sorted_v[len(sorted_v) // 2]
                cls_grid[r, c] = CLASS_HIGH_VEG
            elif sides_with_orig_elev >= 3 and len(all_building_like) + len(veg_vals) >= majority:
                elev = all_building_like + veg_vals
                new_filled[r, c] = sum(elev) / len(elev)
                cls_grid[r, c] = CLASS_BUILDING if len(building_vals) > len(veg_vals) else CLASS_HIGH_VEG
            else:
                new_filled[r, c] = dtm[r, c]

            any_change = True

        filled = new_filled
        if not any_change:
            break

    # Final fallback for any remaining NaN
    remaining = np.isnan(filled)
    if np.any(remaining):
        filled[remaining] = dtm[remaining]
        still_nan = np.isnan(filled)
        if np.any(still_nan):
            filled = _fill_nan_nearest(filled)

    nan_count_end = int(np.sum(np.isnan(filled)))
    _LOGGER.info(
        "Classification-aware gap fill: %d → %d NaN cells (%d filled)",
        nan_count_start, nan_count_end, nan_count_start - nan_count_end,
    )
    return filled


def _fill_nan_nearest(arr: np.ndarray) -> np.ndarray:
    """Fill NaN values with nearest valid neighbor (simple iterative dilation)."""
    if not np.any(np.isnan(arr)):
        return arr

    filled = arr.copy()
    # Iterate until no NaN remains (or max 50 iterations for safety)
    for _ in range(50):
        nan_mask = np.isnan(filled)
        if not np.any(nan_mask):
            break
        # Shift in 4 directions and take the first non-NaN
        rows, cols = filled.shape
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = np.full_like(filled, np.nan)
            src_r = slice(max(0, -dr), rows - max(0, dr))
            dst_r = slice(max(0, dr), rows - max(0, -dr))
            src_c = slice(max(0, -dc), cols - max(0, dc))
            dst_c = slice(max(0, dc), cols - max(0, -dc))
            shifted[dst_r, dst_c] = filled[src_r, src_c]
            fill_mask = nan_mask & ~np.isnan(shifted)
            filled[fill_mask] = shifted[fill_mask]

    # Final fallback: fill any remaining NaN with global min
    if np.any(np.isnan(filled)):
        min_val = np.nanmin(filled) if not np.all(np.isnan(filled)) else 0.0
        filled = np.where(np.isnan(filled), float(min_val), filled)

    return filled


def _fill_classification_holes(
    dsm: np.ndarray,
    dtm: np.ndarray,
    classification: np.ndarray,
    rows: int,
    cols: int,
    passes: int = 5,
    min_neighbors: int = 3,
) -> None:
    """Fill holes in elevated features using morphological closing (in-place).

    At fine resolution, some cells within a building or tree canopy lack
    LiDAR returns and default to ground classification. These appear as
    holes in the DSM surface. This fills them by checking if a ground-level
    cell is surrounded by elevated features of the same type, and if so,
    reclassifying it and interpolating the DSM height.

    Modifies dsm and classification arrays in-place.
    """
    ELEVATED_CLASSES = {CLASS_BUILDING, CLASS_HIGH_VEG, CLASS_MED_VEG}
    TRANSMITTANCE_MAP = {
        CLASS_BUILDING: 0.0,
        CLASS_HIGH_VEG: 0.15,
        CLASS_MED_VEG: 0.4,
    }

    total_filled = 0
    for _ in range(passes):
        height_above = dsm - dtm
        fill_count = 0

        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                # Only consider cells significantly below their neighbors.
                # A "hole" is a cell at ground level when neighbors are elevated.
                if height_above[r, c] > 1.0:
                    continue

                # Count elevated neighbors by class
                neighbor_cls = {}
                elev_count = 0
                sum_h = 0.0
                cnt_h = 0

                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        ncls = int(classification[nr, nc])
                        if ncls in ELEVATED_CLASSES and height_above[nr, nc] > 1.0:
                            elev_count += 1
                            neighbor_cls[ncls] = neighbor_cls.get(ncls, 0) + 1
                            sum_h += dsm[nr, nc]
                            cnt_h += 1

                if elev_count < min_neighbors:
                    continue

                # Pick the dominant elevated neighbor class
                best_cls = max(neighbor_cls, key=neighbor_cls.get)
                interp_h = sum_h / cnt_h if cnt_h > 0 else dsm[r, c]

                # Fill the hole
                dsm[r, c] = interp_h
                classification[r, c] = best_cls
                fill_count += 1

        total_filled += fill_count
        if fill_count == 0:
            break  # no more holes to fill

    if total_filled > 0:
        _LOGGER.info(
            "Morphological closing: filled %d holes in elevated features (%d passes)",
            total_filled, passes,
        )


def _latlon_to_utm(latitude: float, longitude: float) -> tuple[int, float, float, bool]:
    """Convert lat/lon to UTM easting/northing.

    Returns (zone_number, easting, northing, is_northern).
    """
    zone = int((longitude + 180) / 6) + 1

    # UTM projection constants
    a = 6378137.0  # WGS84 semi-major axis
    f = 1 / 298.257223563
    e2 = 2 * f - f * f
    e_prime2 = e2 / (1 - e2)
    k0 = 0.9996

    lat_rad = math.radians(latitude)
    lon_rad = math.radians(longitude)
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)

    n = a / math.sqrt(1 - e2 * math.sin(lat_rad) ** 2)
    t = math.tan(lat_rad) ** 2
    c = e_prime2 * math.cos(lat_rad) ** 2
    a_coeff = math.cos(lat_rad) * (lon_rad - lon0)

    # Meridional arc
    m = a * (
        (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * lat_rad
        - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * math.sin(2 * lat_rad)
        + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * math.sin(4 * lat_rad)
        - (35 * e2**3 / 3072) * math.sin(6 * lat_rad)
    )

    easting = k0 * n * (
        a_coeff
        + (1 - t + c) * a_coeff**3 / 6
        + (5 - 18 * t + t**2 + 72 * c - 58 * e_prime2) * a_coeff**5 / 120
    ) + 500000.0

    northing = k0 * (
        m
        + n * math.tan(lat_rad) * (
            a_coeff**2 / 2
            + (5 - t + 9 * c + 4 * c**2) * a_coeff**4 / 24
            + (61 - 58 * t + t**2 + 600 * c - 330 * e_prime2) * a_coeff**6 / 720
        )
    )

    is_northern = latitude >= 0
    if not is_northern:
        northing += 10000000.0

    return zone, easting, northing, is_northern
