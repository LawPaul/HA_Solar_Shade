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
        except (RuntimeError, aiohttp.ClientError, asyncio.TimeoutError) as err:
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
        zone, easting, northing, _is_northern = _latlon_to_utm(latitude, longitude)
        for tile_info in laz_urls:
            laz_url = tile_info["url"]
            _LOGGER.info(
                "Attempting LAZ download: %s", laz_url.rsplit("/", 1)[-1]
            )
            laz_result = await download_and_rasterize_laz(
                center_easting=easting,
                center_northing=northing,
                radius_m=radius_m,
                session=session,
                laz_url=laz_url,
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


async def download_and_rasterize_laz(
    center_easting: float,
    center_northing: float,
    radius_m: float,
    session: aiohttp.ClientSession,
    laz_url: str,
    min_cell_size: float = 0.5,
    dsm_gap_fill: bool = False,
    expected_epsg: int = 0,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float, float] | None:
    """Download a LAZ file from a URL and rasterize to DSM + DTM grids.

    center_easting/center_northing must be in the same CRS as the LAZ data.
    expected_epsg is used to skip unnecessary reprojection when the LAZ
    file's CRS matches the caller's coordinate system.
    """
    easting = center_easting
    northing = center_northing

    filename = laz_url.rsplit("/", 1)[-1]
    _LOGGER.info("LAZ: downloading %s", filename)

    import tempfile
    import os
    import asyncio
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
            except (OSError, aiohttp.ClientError):
                os.unlink(tmp_path)
                raise
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as err:
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
            dsm_gap_fill, expected_epsg,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return result


def _read_laz_epsg(laz_path: str, laspy_module) -> int | None:
    """Extract EPSG code from a LAZ file's VLR metadata.

    Detection strategy (tried in order):
    1. laspy ``parse_crs()`` — works for both WKT (record 2112) and
       GeoKeyDirectory (record 34735) VLRs via laspy's parsed classes.
    2. WKT string regex — fallback for the ``.string`` attribute on
       ``WktCoordinateSystemVlr`` or raw ``record_data`` bytes.
    3. GeoKey raw-byte parsing — fallback for older laspy versions that
       don't have ``parse_crs()``.
    4. Filename EPSG extraction — if no CRS metadata exists in the file
       (e.g. swisstopo LAS 1.2 files stripped by LAStools), try to
       extract an EPSG code from the filename or parent path.
    5. Coordinate-range heuristic — match the point cloud bounding box
       against known national projected CRS coordinate ranges.

    Returns the EPSG code or None if detection fails.
    """
    import re as _re

    epsg_from_vlr = _read_laz_epsg_from_vlrs(laz_path, laspy_module, _re)
    if epsg_from_vlr:
        return epsg_from_vlr

    # No CRS in VLRs — try extracting EPSG from filename/path
    epsg_from_name = _epsg_from_filename(laz_path)
    if epsg_from_name:
        return epsg_from_name

    # Last resort — infer from coordinate ranges in the file header
    return _epsg_from_coordinate_ranges(laz_path, laspy_module)


def _read_laz_epsg_from_vlrs(
    laz_path: str, laspy_module, _re,
) -> int | None:
    """Try to extract EPSG from VLR/EVLR metadata."""
    try:
        with laspy_module.open(laz_path) as reader:
            # Collect all VLRs + EVLRs
            all_vlrs = list(reader.header.vlrs)
            try:
                evlrs = reader.evlrs
                if evlrs:
                    all_vlrs.extend(evlrs)
            except (AttributeError, ValueError):
                pass

            for vlr in all_vlrs:
                if getattr(vlr, "user_id", "") != "LASF_Projection":
                    continue

                # ── Strategy 1: laspy parse_crs() ────────────────────
                if hasattr(vlr, "parse_crs"):
                    try:
                        crs = vlr.parse_crs()
                        if crs is not None:
                            epsg = crs.to_epsg()
                            if epsg:
                                _LOGGER.info(
                                    "LAZ parse_crs(): EPSG:%d", epsg
                                )
                                return epsg
                    except (ValueError, RuntimeError) as e:
                        _LOGGER.debug("parse_crs() failed: %s", e)

                # ── Strategy 2: WKT string (record 2112) ─────────────
                if vlr.record_id == 2112:
                    wkt = _extract_wkt_string(vlr)
                    if wkt:
                        epsg = _epsg_from_wkt(wkt, _re)
                        if epsg:
                            _LOGGER.info("LAZ WKT regex: EPSG:%d", epsg)
                            return epsg

                # ── Strategy 3: GeoKey raw bytes (record 34735) ───────
                if vlr.record_id == 34735:
                    epsg = _epsg_from_geokey_bytes(vlr)
                    if epsg:
                        _LOGGER.info("LAZ GeoKey raw: EPSG:%d", epsg)
                        return epsg
    except (OSError, ValueError, IndexError) as e:
        _LOGGER.debug("Failed to read LAZ VLRs: %s", e)
    return None


def _extract_wkt_string(vlr) -> str:
    """Get the WKT string from a VLR, handling both raw and parsed types."""
    # laspy >= 2.x WktCoordinateSystemVlr stores WKT in .string
    if hasattr(vlr, "string") and vlr.string:
        return vlr.string
    # Raw bytes fallback
    raw = getattr(vlr, "record_data", None)
    if isinstance(raw, (bytes, bytearray)) and len(raw) > 0:
        return raw.decode("utf-8", errors="ignore").rstrip("\x00")
    # record_data_bytes() fallback
    if hasattr(vlr, "record_data_bytes"):
        try:
            raw = vlr.record_data_bytes()
            if isinstance(raw, bytes) and len(raw) > 0:
                return raw.decode("utf-8", errors="ignore").rstrip("\x00")
        except (AttributeError, ValueError, OSError):
            pass
    return ""


def _epsg_from_wkt(wkt: str, _re) -> int | None:
    """Extract EPSG code from a WKT CRS string."""
    if not wkt:
        return None
    # WKT1: AUTHORITY["EPSG","XXXX"]
    match = _re.search(r'AUTHORITY\s*\[\s*"EPSG"\s*,\s*"(\d+)"\s*\]', wkt)
    if match:
        return int(match.group(1))
    # WKT2: ID["EPSG",XXXX]
    match = _re.search(r'ID\s*\[\s*"EPSG"\s*,\s*(\d+)\s*\]', wkt)
    if match:
        return int(match.group(1))
    return None


def _epsg_from_geokey_bytes(vlr) -> int | None:
    """Extract EPSG from GeoKeyDirectory raw bytes."""
    import struct

    # laspy parsed .geo_keys
    if hasattr(vlr, "geo_keys"):
        for gk in vlr.geo_keys:
            key_id = getattr(gk, "id", 0)
            val = getattr(gk, "value_offset", getattr(gk, "offset", 0))
            if key_id == 3072 and val > 0:  # ProjectedCSTypeGeoKey
                return val
            if key_id == 2048 and val > 0:  # GeographicTypeGeoKey
                return val

    # Raw bytes fallback
    data = (
        vlr.record_data_bytes()
        if hasattr(vlr, "record_data_bytes")
        else getattr(vlr, "record_data", b"")
    )
    if not isinstance(data, (bytes, bytearray)) or len(data) < 8:
        return None
    _, _, _, num_keys = struct.unpack("<HHHH", data[:8])
    for i in range(num_keys):
        off = 8 + i * 8
        if off + 8 > len(data):
            break
        key_id, tif_tag, _, value = struct.unpack("<HHHH", data[off : off + 8])
        if key_id == 3072 and tif_tag == 0 and value > 0:
            return value
        if key_id == 2048 and tif_tag == 0 and value > 0:
            return value
    return None


# Regex patterns to extract EPSG codes from filenames or parent paths.
# Tried in order; first valid match wins.
_FILENAME_EPSG_PATTERNS: list[tuple[str, int]] = [
    # swisstopo: swisssurface3d_2018_2682-1246_2056_5728.las.zip
    #   → two trailing EPSG codes (horizontal_vertical) before extension
    (r"_(\d{4,5})_\d{4,5}\.(?:las|laz)", 1),
    # Generic: any "_EPSG2056" or "_epsg3006" segment
    (r"[_-][Ee][Pp][Ss][Gg](\d{4,6})", 1),
]


def _epsg_from_filename(laz_path: str) -> int | None:
    """Try to extract a valid EPSG code from the filename or parent path.

    Many national LiDAR datasets encode the CRS in the filename
    (e.g. swisstopo ``*_2056_5728.las.zip``).  This is more reliable
    than coordinate-range heuristics which suffer from ambiguity
    between overlapping national grids.
    """
    import os
    import re

    # Search the full path (parent dirs may contain EPSG info too)
    search_text = os.path.basename(laz_path)
    # Also include immediate parent directory name
    parent = os.path.basename(os.path.dirname(laz_path))
    if parent:
        search_text = parent + "/" + search_text

    for pattern, group_idx in _FILENAME_EPSG_PATTERNS:
        m = re.search(pattern, search_text)
        if not m:
            continue
        try:
            candidate = int(m.group(group_idx))
        except (ValueError, IndexError):
            continue

        # Validate: must be a real projected CRS (not a year, tile index, etc.)
        if candidate < 1024 or candidate > 99999:
            continue

        try:
            from pyproj import CRS
            crs = CRS.from_epsg(candidate)
            if crs.is_projected:
                _LOGGER.info(
                    "LAZ filename fallback: EPSG:%d from %r",
                    candidate, os.path.basename(laz_path),
                )
                return candidate
        except (ValueError, RuntimeError):
            continue

    return None


# Coordinate ranges for national projected CRS codes.
# Each entry: (EPSG, (x_min, x_max), (y_min, y_max))
# Only CRS systems with truly distinctive, non-overlapping coordinate
# ranges are included here.  Many national grids share similar value
# ranges (e.g. UTM zones, Lambert-93 vs SWEREF99 TM), so we only list
# CRS codes whose bounding boxes are unambiguous.
_KNOWN_CRS_RANGES: list[tuple[int, tuple[float, float], tuple[float, float]]] = [
    # Switzerland LV95 (EPSG:2056) — X always starts with 2_4xx_xxx to 2_8xx_xxx
    # No other national CRS uses coordinates in the 2.4M-2.9M range.
    (2056, (2_480_000, 2_840_000), (1_070_000, 1_300_000)),
    # Netherlands RD New (EPSG:28992) — X: 0-300k, Y: 300k-625k
    # The Y range starting at 300k with X under 300k is unique.
    (28992, (0, 300_000), (300_000, 625_000)),
]


def _epsg_from_coordinate_ranges(laz_path: str, laspy_module) -> int | None:
    """Infer EPSG from the file's bounding box coordinates.

    This is a last-resort heuristic for files with no VLR metadata and
    no EPSG in the filename (e.g. swisstopo LAS 1.2 stripped by LAStools).

    Only matches if the entire bounding box falls within a single known
    CRS range. Returns None if zero or multiple ranges match (ambiguous).
    """
    try:
        with laspy_module.open(laz_path) as reader:
            mins = reader.header.mins
            maxs = reader.header.maxs
    except (OSError, ValueError, IndexError):
        return None

    x_min, y_min = mins[0], mins[1]
    x_max, y_max = maxs[0], maxs[1]

    matches = []
    for epsg, (rx_min, rx_max), (ry_min, ry_max) in _KNOWN_CRS_RANGES:
        if rx_min <= x_min and x_max <= rx_max and ry_min <= y_min and y_max <= ry_max:
            matches.append(epsg)

    if len(matches) == 1:
        _LOGGER.info(
            "LAZ coordinate-range heuristic: EPSG:%d "
            "(bbox X:%.0f–%.0f Y:%.0f–%.0f)",
            matches[0], x_min, x_max, y_min, y_max,
        )
        return matches[0]

    if len(matches) > 1:
        _LOGGER.debug(
            "LAZ coordinate-range heuristic: ambiguous — %d matches (%s) "
            "for bbox X:%.0f–%.0f Y:%.0f–%.0f",
            len(matches), matches, x_min, x_max, y_min, y_max,
        )

    return None


def _project_to_epsg(
    center_easting: float, center_northing: float,
    target_epsg: int, expected_epsg: int = 0,
) -> tuple[float | None, float | None]:
    """Reproject center coordinates if the LAZ file's CRS differs.

    If expected_epsg is set and matches target_epsg, no reprojection is
    needed (the caller already provided coordinates in the right CRS).

    For USGS 3DEP data, the file CRS is typically UTM NAD83 (EPSG 269xx or
    326xx). If the file's UTM zone matches our computed zone, no reprojection
    is needed. If it differs (e.g., near zone boundaries), we reproject.

    Also recognises SWEREF99 TM (3006) and D96/TM (3794) as valid CRS codes.

    Returns (easting, northing) in the target CRS, or (None, None) if
    reprojection is not needed or not possible.
    """
    # If the caller told us what CRS the center is in and it matches, done.
    if expected_epsg and expected_epsg == target_epsg:
        return None, None

    # Standard UTM zones — WGS84 and NAD83 (≈ WGS84 at our scale)
    if 32601 <= target_epsg <= 32660:
        return None, None
    if 32701 <= target_epsg <= 32760:
        return None, None
    if 26901 <= target_epsg <= 26999:
        return None, None

    # Any other projected CRS — pyproj handles it via geo module.
    # No coordinate reprojection here; the caller (process_lidar_file or
    # download_and_rasterize_laz) already computed the center in the
    # file's native CRS.
    if expected_epsg == 0:
        _LOGGER.warning(
            "LAZ file uses EPSG:%d but center is in UTM. "
            "Coordinate alignment may be imprecise. "
            "Select the correct DSM provider for best results.",
            target_epsg,
        )
    return None, None


# ── Rasterization helpers ────────────────────────────────────────
# Extracted from _rasterize_laz_file for readability.


def _read_laz_points_in_radius(
    laz_path: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    laspy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int] | None:
    """Read LAZ in chunks, keeping only points within the bounding box.

    Returns (x, y, z, classification, return_number, total_points) or None.
    """
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
                cx = np.asarray(chunk.x, dtype=np.float64)
                cy = np.asarray(chunk.y, dtype=np.float64)
                in_bounds = (
                    (cx >= x_min) & (cx <= x_max) &
                    (cy >= y_min) & (cy <= y_max)
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
    except (OSError, ValueError, IndexError) as err:
        _LOGGER.error("Failed to read LAZ data: %s", err)
        return None

    if kept_points < 100:
        _LOGGER.warning(
            "LAZ: only %d points within bounding box (of %d total) "
            "— insufficient data",
            kept_points, total_points,
        )
        return None

    return (
        np.concatenate(x_list),
        np.concatenate(y_list),
        np.concatenate(z_list),
        np.concatenate(cls_list),
        np.concatenate(ret_list),
        total_points,
    )


def _build_dtm(
    cell_idx: np.ndarray,
    z: np.ndarray,
    classification: np.ndarray,
    n_cells: int,
    rows: int,
    cols: int,
) -> np.ndarray:
    """Build DTM (digital terrain model) from ground-classified points.

    Returns gap-filled 2-D array (rows, cols).
    """
    ground_mask = classification == CLASS_GROUND
    dtm = np.full(n_cells, np.nan, dtype=np.float32)

    if np.sum(ground_mask) > 0:
        ground_cells = cell_idx[ground_mask]
        ground_z = z[ground_mask]
        dtm_sum = np.zeros(n_cells, dtype=np.float64)
        dtm_count = np.zeros(n_cells, dtype=np.int32)
        np.add.at(dtm_sum, ground_cells, ground_z.astype(np.float64))
        np.add.at(dtm_count, ground_cells, 1)
        valid = dtm_count > 0
        dtm[valid] = (dtm_sum[valid] / dtm_count[valid]).astype(np.float32)
        _LOGGER.info(
            "DTM: %d ground points → %d/%d cells filled (%.0f%%)",
            np.sum(ground_mask), np.sum(valid), n_cells,
            100 * np.sum(valid) / n_cells,
        )
    else:
        _LOGGER.warning(
            "LAZ: no ground-classified points; estimating ground from "
            "lowest 5th-percentile Z per cell"
        )
        dtm_min = np.full(n_cells, np.inf, dtype=np.float32)
        np.minimum.at(dtm_min, cell_idx, z)
        has_points = dtm_min < np.inf
        if np.any(has_points):
            p5 = float(np.percentile(z, 5))
            dtm[has_points] = np.maximum(dtm_min[has_points], p5)
            _LOGGER.info(
                "DTM (estimated): 5th-percentile=%.1fm, %d/%d cells filled",
                p5, int(np.sum(has_points)), n_cells,
            )

    dtm = dtm.reshape(rows, cols)
    return _fill_nan_nearest(dtm)


def _build_dsm(
    cell_idx: np.ndarray,
    z: np.ndarray,
    return_number: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    """Build DSM (digital surface model) from first-return points.

    Returns 2-D array (rows, cols) with NaN for empty cells.
    """
    n_cells = rows * cols
    dsm_mask = (return_number == 1) | (return_number == 0)
    if np.sum(dsm_mask) < len(z) * 0.1:
        dsm_mask = np.ones(len(z), dtype=bool)

    dsm = np.full(n_cells, -np.inf, dtype=np.float32)
    np.maximum.at(dsm, cell_idx[dsm_mask], z[dsm_mask])
    dsm[dsm == -np.inf] = np.nan
    return dsm.reshape(rows, cols)


def _build_classification_grid(
    cell_idx: np.ndarray,
    z: np.ndarray,
    classification: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    """Assign each cell the classification of its highest point.

    Returns 2-D uint8 array (rows, cols).
    """
    n_cells = rows * cols
    # Sort ascending by z — last write per cell wins (= highest point)
    order = np.argsort(z)
    sorted_cells = cell_idx[order]
    sorted_cls = classification[order]
    cell_top_cls = np.full(n_cells, CLASS_GROUND, dtype=np.uint8)
    cell_top_cls[sorted_cells] = sorted_cls
    return cell_top_cls.reshape(rows, cols)


def _build_canopy_base(
    dsm: np.ndarray,
    dtm: np.ndarray,
    cls_grid: np.ndarray,
    cell_idx: np.ndarray,
    z: np.ndarray,
    classification: np.ndarray,
    rows: int,
    cols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build canopy-base grid and the raw-vegetation mask.

    Returns (canopy_base, is_veg_raw) — both 2-D arrays (rows, cols).
    """
    TRUNK_MIN_HEIGHT = 2.0
    VEG_CLASSES_SET = {CLASS_LOW_VEG, CLASS_MED_VEG, CLASS_HIGH_VEG}

    canopy_base = np.full_like(dsm, np.nan)
    has_dsm = ~np.isnan(dsm)
    is_veg_raw = np.isin(cls_grid, list(VEG_CLASSES_SET))

    # Non-veg cells with DSM data: solid column
    canopy_base[has_dsm & ~is_veg_raw] = dsm[has_dsm & ~is_veg_raw]

    # Veg cells: lowest non-ground return above trunk threshold
    dtm_flat = dtm.reshape(-1)
    non_ground = classification != CLASS_GROUND
    non_ground_cells = cell_idx[non_ground]
    non_ground_z = z[non_ground]
    above_trunk = non_ground_z > (dtm_flat[non_ground_cells] + TRUNK_MIN_HEIGHT)
    trunk_cells = non_ground_cells[above_trunk]
    trunk_z = non_ground_z[above_trunk]

    n_cells = rows * cols
    min_canopy_z = np.full(n_cells, np.inf, dtype=np.float32)
    if len(trunk_cells) > 0:
        np.minimum.at(min_canopy_z, trunk_cells, trunk_z)

    min_canopy_z = min_canopy_z.reshape(rows, cols)
    has_canopy = min_canopy_z < np.inf
    canopy_base[has_canopy & is_veg_raw] = min_canopy_z[has_canopy & is_veg_raw]
    canopy_base[has_dsm & is_veg_raw & ~has_canopy] = dsm[has_dsm & is_veg_raw & ~has_canopy]

    return canopy_base, is_veg_raw


def _rasterize_laz_file(
    laz_path: str,
    center_easting: float,
    center_northing: float,
    radius_m: float,
    min_cell_size: float = 0.5,
    dsm_gap_fill: bool = False,
    expected_epsg: int = 0,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float, float] | None:
    """Read a LAZ file and rasterize classified points into DSM and DTM grids.

    Resolution is auto-calculated from point density (target: ~4 pts/cell,
    floored at *min_cell_size*, snapped to 0.5m increments).

    Reads in chunks to keep memory low — only points within the bounding box
    are kept. Peak memory is proportional to radius, not tile size.

    expected_epsg: if set, skips reprojection when the LAZ file's embedded
    EPSG matches this value (the caller already supplied center coordinates
    in the correct CRS).
    """
    try:
        import laspy
    except ImportError:
        _LOGGER.error(
            "laspy not installed — cannot process LAZ point clouds. "
            "Install with: pip install laspy laszip"
        )
        return None

    # Reproject center if file CRS differs from expected
    file_epsg = _read_laz_epsg(laz_path, laspy)
    if file_epsg is not None:
        file_center_e, file_center_n = _project_to_epsg(
            center_easting, center_northing, file_epsg,
            expected_epsg=expected_epsg,
        )
        if file_center_e is not None:
            _LOGGER.info(
                "LAZ CRS: EPSG:%d — reprojected center (%.1f, %.1f) → "
                "(%.1f, %.1f), shift: %.1fm E, %.1fm N",
                file_epsg, center_easting, center_northing,
                file_center_e, file_center_n,
                file_center_e - center_easting,
                file_center_n - center_northing,
            )
            center_easting = file_center_e
            center_northing = file_center_n

    # Bounding box
    grid_x_min = center_easting - radius_m
    grid_x_max = center_easting + radius_m
    grid_y_min = center_northing - radius_m
    grid_y_max = center_northing + radius_m

    # Read and clip points
    result = _read_laz_points_in_radius(
        laz_path, grid_x_min, grid_x_max, grid_y_min, grid_y_max, laspy,
    )
    if result is None:
        return None
    x, y, z, classification, return_number, total_points = result
    kept_points = len(x)

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
                     cls, name, cnt, 100 * cnt / kept_points)

    # Auto-calculate resolution from point density
    area_m2 = (grid_x_max - grid_x_min) * (grid_y_max - grid_y_min)
    density = kept_points / area_m2 if area_m2 > 0 else 0
    if density > 0:
        auto_res = 2.0 / max(density ** 0.5, 0.01)
        resolution = round(auto_res * 2) / 2  # snap to 0.5m
        resolution = max(min_cell_size, min(resolution, 5.0))
    else:
        resolution = max(min_cell_size, 1.0)
    _LOGGER.info("LAZ: point density %.1f pts/m² → resolution %.1fm", density, resolution)

    cols = int(round((grid_x_max - grid_x_min) / resolution))
    rows = int(round((grid_y_max - grid_y_min) / resolution))
    if cols < 2 or rows < 2:
        _LOGGER.warning("LAZ: grid too small (%dx%d)", rows, cols)
        return None

    # Compute cell indices (row 0 = north / max Y)
    col_idx = np.clip(((x - grid_x_min) / resolution).astype(np.int32), 0, cols - 1)
    row_idx = np.clip(((grid_y_max - y) / resolution).astype(np.int32), 0, rows - 1)
    cell_idx = row_idx * cols + col_idx
    n_cells = rows * cols

    # Build grids via helpers
    dtm = _build_dtm(cell_idx, z, classification, n_cells, rows, cols)
    dsm = _build_dsm(cell_idx, z, return_number, rows, cols)
    cls_grid = _build_classification_grid(cell_idx, z, classification, rows, cols)
    canopy_base, is_veg_raw = _build_canopy_base(
        dsm, dtm, cls_grid, cell_idx, z, classification, rows, cols,
    )

    # Gap-filling and morphological closing
    if dsm_gap_fill:
        dsm = _fill_dsm_classification_aware(dsm, dtm, cls_grid, resolution)
    cb_nan = np.isnan(canopy_base)
    canopy_base[cb_nan] = dsm[cb_nan]
    dsm = np.maximum(dsm, dtm)

    _fill_classification_holes(dsm, dtm, cls_grid, rows, cols)

    # Finalize canopy_base
    VEG_CLASSES_SET = {CLASS_LOW_VEG, CLASS_MED_VEG, CLASS_HIGH_VEG}
    is_veg = np.isin(cls_grid, list(VEG_CLASSES_SET))
    canopy_base[is_veg & ~is_veg_raw] = dsm[is_veg & ~is_veg_raw]
    canopy_base[~is_veg] = dsm[~is_veg]
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

    return dsm, dtm, cls_grid, canopy_base, grid_x_min, grid_y_min, grid_x_max, grid_y_max, resolution


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
    ELEVATED_CLASSES = np.array([CLASS_BUILDING, CLASS_HIGH_VEG, CLASS_MED_VEG], dtype=np.uint8)

    total_filled = 0
    for _ in range(passes):
        height_above = dsm - dtm
        # Candidate cells: ground-level (height <= 1m), interior (not border)
        is_hole = np.zeros((rows, cols), dtype=bool)
        is_hole[1:-1, 1:-1] = height_above[1:-1, 1:-1] <= 1.0

        # Is-elevated mask for neighbor counting
        is_elevated = np.isin(classification, ELEVATED_CLASSES) & (height_above > 1.0)

        # Count elevated neighbors in 3x3 window using convolution
        # Shift in 8 directions and sum
        elev_count = np.zeros((rows, cols), dtype=np.int32)
        dsm_sum = np.zeros((rows, cols), dtype=np.float64)
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                shifted_elev = np.zeros((rows, cols), dtype=bool)
                shifted_dsm = np.zeros((rows, cols), dtype=np.float64)
                shifted_cls = np.zeros((rows, cols), dtype=np.uint8)
                # Source/destination slices for the shift
                sr = slice(max(0, -dr), rows - max(0, dr))
                dr_s = slice(max(0, dr), rows - max(0, -dr))
                sc = slice(max(0, -dc), cols - max(0, dc))
                dc_s = slice(max(0, dc), cols - max(0, -dc))
                shifted_elev[dr_s, dc_s] = is_elevated[sr, sc]
                shifted_dsm[dr_s, dc_s] = dsm[sr, sc]
                shifted_cls[dr_s, dc_s] = classification[sr, sc]
                elev_count += shifted_elev.astype(np.int32)
                dsm_sum += np.where(shifted_elev, shifted_dsm, 0.0)

        # Cells to fill: holes with enough elevated neighbors
        fill_mask = is_hole & (elev_count >= min_neighbors)
        fill_count = int(fill_mask.sum())
        if fill_count == 0:
            break

        # Determine dominant elevated neighbor class per cell
        # Count each elevated class from shifted arrays
        cls_counts = {}
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                shifted_cls = np.zeros((rows, cols), dtype=np.uint8)
                shifted_elev = np.zeros((rows, cols), dtype=bool)
                sr = slice(max(0, -dr), rows - max(0, dr))
                dr_s = slice(max(0, dr), rows - max(0, -dr))
                sc = slice(max(0, -dc), cols - max(0, dc))
                dc_s = slice(max(0, dc), cols - max(0, -dc))
                shifted_cls[dr_s, dc_s] = classification[sr, sc]
                shifted_elev[dr_s, dc_s] = is_elevated[sr, sc]
                for ec in ELEVATED_CLASSES:
                    key = int(ec)
                    match = shifted_elev & (shifted_cls == ec)
                    if key not in cls_counts:
                        cls_counts[key] = np.zeros((rows, cols), dtype=np.int32)
                    cls_counts[key] += match.astype(np.int32)

        # Pick the class with the highest count per cell
        best_cls = np.full((rows, cols), CLASS_GROUND, dtype=np.uint8)
        best_count = np.zeros((rows, cols), dtype=np.int32)
        for ec_int, counts in cls_counts.items():
            better = counts > best_count
            best_cls[better] = ec_int
            best_count[better] = counts[better]

        # Apply fills
        safe_count = np.maximum(elev_count, 1)
        interp_h = np.where(elev_count > 0, dsm_sum / safe_count, dsm)
        dsm[fill_mask] = interp_h[fill_mask]
        classification[fill_mask] = best_cls[fill_mask]

        total_filled += fill_count

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


# ── Provider class (self-registers via __init_subclass__) ────────────────

from .elevation_provider import ElevationProvider  # noqa: E402


class USGSProvider(ElevationProvider):
    """USGS 3DEP LiDAR data provider (United States)."""

    PROVIDER_ID = "usgs"
    PROVIDER_NAME = "USGS 3DEP (United States)"
    NATIVE_EPSG = 0  # UTM zone varies by location
    COUNTRY_CODES = ("US", "PR", "VI", "GU", "AS", "MP")  # US + territories

    def latlon_to_native(self, latitude: float, longitude: float) -> tuple[float, float]:
        _zone, easting, northing, _is_n = _latlon_to_utm(latitude, longitude)
        return easting, northing

    async def find_tiles(self, latitude, longitude, session):
        return await find_laz_urls(latitude, longitude, session)
