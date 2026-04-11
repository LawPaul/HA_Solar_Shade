"""Manual test: Real Slovenian GKOT LAZ file through the full pipeline.

Tests the manual file upload workflow with a real GKOT file downloaded
from https://clss.si/ — verifies CRS detection, point cloud processing,
DSM/DTM generation, classification handling, and shadow computation.

Run:  python tests/manual_test_real_si.py
"""
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# ── Stub out Home Assistant imports for standalone use ────────────────
import types
for mod_name in [
    "homeassistant", "homeassistant.config_entries",
    "homeassistant.core", "homeassistant.helpers",
    "homeassistant.helpers.aiohttp_client",
    "homeassistant.helpers.config_validation",
    "homeassistant.components",
    "homeassistant.components.persistent_notification",
    "homeassistant.helpers.update_coordinator",
    "homeassistant.helpers.entity",
    "homeassistant.helpers.device_registry",
    "homeassistant.components.sensor",
]:
    if mod_name not in sys.modules:
        m = types.ModuleType(mod_name)
        m.ConfigEntry = type("ConfigEntry", (), {})
        m.HomeAssistant = type("HomeAssistant", (), {})
        m.ServiceCall = type("ServiceCall", (), {})
        m.DataUpdateCoordinator = type("DataUpdateCoordinator", (), {})
        m.SensorEntity = type("SensorEntity", (), {})
        m.SensorDeviceClass = type("SensorDeviceClass", (), {})
        m.SensorStateClass = type("SensorStateClass", (), {})
        m.config_entry_only_config_schema = lambda *a, **kw: None
        sys.modules[mod_name] = m


def find_gkot_file():
    """Find the GKOT LAZ file in the Downloads folder."""
    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    for f in os.listdir(downloads):
        if f.startswith("GKOT_") and f.endswith(".laz"):
            return os.path.join(downloads, f)
    return None


def test_real_slovenian_file():
    """Full pipeline test with real Slovenian GKOT data."""
    laz_path = find_gkot_file()
    if not laz_path:
        print("SKIP: No GKOT_*.laz file found in Downloads")
        return False

    fname = os.path.basename(laz_path)
    fsize_mb = os.path.getsize(laz_path) / (1024 * 1024)
    print(f"File: {fname} ({fsize_mb:.1f} MB)")

    # ── Step 1: CRS detection ──────────────────────────────────────────
    print("\n=== Step 1: CRS Detection ===")
    import laspy
    from custom_components.solar_shade.usgs_downloader import _read_laz_epsg

    t0 = time.time()
    epsg = _read_laz_epsg(laz_path, laspy)
    t_crs = time.time() - t0
    print(f"  Detected EPSG: {epsg} (took {t_crs:.2f}s)")
    assert epsg == 3794, f"Expected EPSG:3794, got {epsg}"
    print("  PASS: EPSG:3794 (Slovenia 1996 / Slovene National Grid)")

    # ── Step 2: Full pipeline — process_lidar_file() ───────────────────
    print("\n=== Step 2: process_lidar_file() ===")
    from custom_components.solar_shade.shadow_engine import process_lidar_file

    # Use center of tile as the reference point
    with laspy.open(laz_path) as reader:
        mins = reader.header.mins
        maxs = reader.header.maxs
        cx = (mins[0] + maxs[0]) / 2
        cy = (mins[1] + maxs[1]) / 2
        print(f"  Tile center (native): E={cx:.1f}, N={cy:.1f}")
        print(f"  Tile extent: X={mins[0]:.0f}-{maxs[0]:.0f}, Y={mins[1]:.0f}-{maxs[1]:.0f}")
        print(f"  Z range: {mins[2]:.1f} - {maxs[2]:.1f}")

    # Convert native coords to lat/lon for the pipeline
    import pyproj
    transformer = pyproj.Transformer.from_crs(
        f"EPSG:3794", "EPSG:4326", always_xy=True
    )
    lon, lat = transformer.transform(cx, cy)
    print(f"  Center (WGS84): lat={lat:.6f}, lon={lon:.6f}")

    # Process with a 250m radius around center
    t0 = time.time()
    site = process_lidar_file(
        filepath=laz_path,
        latitude=lat,
        longitude=lon,
        min_cell_size=1.0,
    )
    t_proc = time.time() - t0
    print(f"  Processing took {t_proc:.1f}s")

    # ── Step 3: Verify SiteModel ───────────────────────────────────────
    print("\n=== Step 3: SiteModel Verification ===")
    print(f"  native_epsg: {site.native_epsg}")
    assert site.native_epsg == 3794, f"Expected native_epsg=3794, got {site.native_epsg}"

    print(f"  DSM shape: {site.dsm.shape}")
    print(f"  DTM shape: {site.dtm.shape}")
    assert site.dsm.shape == site.dtm.shape, "DSM/DTM shape mismatch"
    assert site.dsm.shape[0] > 50, f"DSM too small: {site.dsm.shape}"
    assert site.dsm.shape[1] > 50, f"DSM too small: {site.dsm.shape}"

    print(f"  Resolution: {site.resolution}m")
    assert 0.5 <= site.resolution <= 2.0, f"Unexpected resolution: {site.resolution}"

    print(f"  Lat/Lon: {site.latitude:.6f}, {site.longitude:.6f}")
    assert abs(site.latitude - lat) < 0.01, f"Latitude mismatch: {site.latitude} vs {lat}"
    assert abs(site.longitude - lon) < 0.01, f"Longitude mismatch: {site.longitude} vs {lon}"

    # ── Step 4: Height analysis ─────────────────────────────────────────
    print("\n=== Step 4: Height Analysis ===")
    height_diff = site.dsm - site.dtm
    max_height = np.nanmax(height_diff)
    mean_height = np.nanmean(height_diff)
    above_2m = np.nansum(height_diff > 2.0)
    above_5m = np.nansum(height_diff > 5.0)
    print(f"  Max height above ground: {max_height:.1f}m")
    print(f"  Mean height above ground: {mean_height:.1f}m")
    print(f"  Cells > 2m above ground: {above_2m:,}")
    print(f"  Cells > 5m above ground: {above_5m:,}")
    assert max_height > 3.0, f"No structures detected (max height {max_height:.1f}m)"

    # ── Step 5: Classification analysis ─────────────────────────────────
    print("\n=== Step 5: Classification ===")
    if site.classification is not None:
        unique, counts = np.unique(site.classification, return_counts=True)
        total = site.classification.size
        for c, n in zip(unique, counts):
            CLASS_NAMES = {
                0: 'Never classified', 1: 'Unclassified', 2: 'Ground',
                3: 'Low veg', 4: 'Medium veg', 5: 'High veg', 6: 'Building'
            }
            name = CLASS_NAMES.get(int(c), f'Class {c}')
            print(f"  {c:3d} ({name}): {n:>8,} cells ({n/total*100:.1f}%)")
        print(f"  PASS: Classification grid present")
    else:
        print("  WARNING: No classification grid (still OK for shadow calc)")

    # ── Step 6: Shadow computation ──────────────────────────────────────
    print("\n=== Step 6: Shadow Computation ===")
    from custom_components.solar_shade.shadow_engine import compute_shadow_map

    # Midday summer sun for Slovenia
    sun_altitude = 60.0
    sun_azimuth = 180.0

    t0 = time.time()
    shadow = compute_shadow_map(
        site.dsm, sun_azimuth, sun_altitude, site.resolution,
        ground=site.dtm,
    )
    t_shadow = time.time() - t0
    print(f"  Shadow map shape: {shadow.shape}")
    print(f"  Shadow map dtype: {shadow.dtype}")
    print(f"  Shadow range: {np.nanmin(shadow):.3f} - {np.nanmax(shadow):.3f}")
    print(f"  Mean shadow opacity: {np.nanmean(shadow):.3f}")
    print(f"  Shadow computation took {t_shadow:.3f}s")

    shadowed_pct = np.nanmean(shadow > 0.1) * 100
    print(f"  Shadowed area (>10% opacity): {shadowed_pct:.1f}%")

    # ── Step 7: Seasonal transmittance ──────────────────────────────────
    print("\n=== Step 7: Seasonal Transmittance ===")
    from custom_components.solar_shade.shadow_engine import seasonal_high_veg_transmittance

    for month in [1, 4, 7, 10]:
        trans = seasonal_high_veg_transmittance(lat, month)
        season = {1: "Winter", 4: "Spring", 7: "Summer", 10: "Autumn"}[month]
        print(f"  {season} (month {month:2d}): transmittance = {trans:.3f}")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print(f"  File: {fname}")
    print(f"  CRS: EPSG:3794 (detected in {t_crs:.2f}s)")
    print(f"  Pipeline: {t_proc:.1f}s")
    print(f"  DSM: {site.dsm.shape[0]}x{site.dsm.shape[1]} @ {site.resolution}m")
    print(f"  Max structure height: {max_height:.1f}m")
    print(f"  Shadow: {shadowed_pct:.1f}% shadowed at solar alt={sun_altitude}°")
    print("=" * 60)
    return True


if __name__ == "__main__":
    ok = test_real_slovenian_file()
    sys.exit(0 if ok else 1)
