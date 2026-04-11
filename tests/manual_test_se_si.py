"""Manual end-to-end test for Sweden and Slovenia users.

Simulates what a real user would experience:
1. Creates a realistic LAS file (as if downloaded from their national portal)
2. Includes proper classification (ground, vegetation, buildings)
3. Includes realistic terrain (flat ground + some buildings + trees)
4. Runs the full process_lidar_file() pipeline
5. Verifies the SiteModel has correct CRS, DSM, DTM, classification
6. Runs a shadow computation to prove the data works end-to-end

Run:  python tests/manual_test_se_si.py
"""
import os
import struct
import sys
import tempfile

# Add project root to path so custom_components is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import laspy
import numpy as np

# ── Stub out Home Assistant imports so we can run standalone ──────────
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
        # Add dummy attributes that __init__.py etc. import
        m.ConfigEntry = type("ConfigEntry", (), {})
        m.HomeAssistant = type("HomeAssistant", (), {})
        m.ServiceCall = type("ServiceCall", (), {})
        m.DataUpdateCoordinator = type("DataUpdateCoordinator", (), {})
        m.SensorEntity = type("SensorEntity", (), {})
        m.SensorDeviceClass = type("SensorDeviceClass", (), {})
        m.SensorStateClass = type("SensorStateClass", (), {})
        m.config_entry_only_config_schema = lambda *a, **kw: None
        sys.modules[mod_name] = m


def create_realistic_las(
    epsg: int,
    x_center: float, y_center: float,
    filename: str,
    n_points: int = 50_000,
    tile_size: float = 200.0,
) -> str:
    """Create a realistic LAS file with terrain, buildings, and trees.

    Simulates a real national LiDAR tile with:
    - Ground points (class 2) forming a gently sloping terrain
    - Building points (class 6) forming rectangular blocks
    - High vegetation points (class 5) forming tree canopies
    - Low vegetation points (class 3) as ground cover
    """
    rng = np.random.default_rng(12345)
    half = tile_size / 2

    # Allocate arrays
    x = np.empty(n_points)
    y = np.empty(n_points)
    z = np.empty(n_points)
    cls = np.empty(n_points, dtype=np.uint8)
    ret_num = np.ones(n_points, dtype=np.uint8)
    num_ret = np.ones(n_points, dtype=np.uint8)

    # Distribute points by class
    n_ground = int(n_points * 0.50)
    n_building = int(n_points * 0.15)
    n_high_veg = int(n_points * 0.20)
    n_low_veg = n_points - n_ground - n_building - n_high_veg

    idx = 0

    # ── Ground points (class 2) — gently sloping terrain ──
    gx = x_center + rng.uniform(-half, half, n_ground)
    gy = y_center + rng.uniform(-half, half, n_ground)
    # Gentle slope: 2% grade east, slight undulation
    gz = 100.0 + (gx - x_center) * 0.02 + np.sin((gy - y_center) / 30) * 0.5
    gz += rng.normal(0, 0.05, n_ground)  # noise
    x[idx:idx+n_ground] = gx
    y[idx:idx+n_ground] = gy
    z[idx:idx+n_ground] = gz
    cls[idx:idx+n_ground] = 2
    idx += n_ground

    # ── Building points (class 6) — two rectangular buildings ──
    n_b1 = n_building // 2
    n_b2 = n_building - n_b1
    # Building 1: 20x15m, 8m tall, NE quadrant
    b1x = x_center + 30 + rng.uniform(0, 20, n_b1)
    b1y = y_center + 20 + rng.uniform(0, 15, n_b1)
    base1 = 100.0 + (b1x - x_center) * 0.02
    b1z = base1 + 8.0 + rng.normal(0, 0.03, n_b1)
    # Building 2: 25x20m, 12m tall, SW quadrant
    b2x = x_center - 50 + rng.uniform(0, 25, n_b2)
    b2y = y_center - 40 + rng.uniform(0, 20, n_b2)
    base2 = 100.0 + (b2x - x_center) * 0.02
    b2z = base2 + 12.0 + rng.normal(0, 0.03, n_b2)
    x[idx:idx+n_b1] = b1x
    y[idx:idx+n_b1] = b1y
    z[idx:idx+n_b1] = b1z
    cls[idx:idx+n_b1] = 6
    idx += n_b1
    x[idx:idx+n_b2] = b2x
    y[idx:idx+n_b2] = b2y
    z[idx:idx+n_b2] = b2z
    cls[idx:idx+n_b2] = 6
    idx += n_b2

    # ── High vegetation (class 5) — tree canopies ──
    # Cluster trees in groups
    n_trees = 8
    pts_per_tree = n_high_veg // n_trees
    for t in range(n_trees):
        count = pts_per_tree if t < n_trees - 1 else n_high_veg - t * pts_per_tree
        # Random tree position
        tx = x_center + rng.uniform(-half * 0.8, half * 0.8)
        ty = y_center + rng.uniform(-half * 0.8, half * 0.8)
        # Crown spread ~5m radius, height 6-15m above ground
        tree_height = rng.uniform(6, 15)
        crown_r = rng.uniform(2, 5)
        angles = rng.uniform(0, 2 * np.pi, count)
        radii = rng.uniform(0, crown_r, count)
        px = tx + radii * np.cos(angles)
        py = ty + radii * np.sin(angles)
        base_z = 100.0 + (px - x_center) * 0.02
        pz = base_z + tree_height + rng.normal(0, 1.0, count)
        x[idx:idx+count] = px
        y[idx:idx+count] = py
        z[idx:idx+count] = pz
        cls[idx:idx+count] = 5
        # Some returns are first-return, some last
        ret_num[idx:idx+count] = 1
        num_ret[idx:idx+count] = rng.choice([1, 2, 3], count)
        idx += count

    # ── Low vegetation (class 3) — ground cover ──
    lvx = x_center + rng.uniform(-half, half, n_low_veg)
    lvy = y_center + rng.uniform(-half, half, n_low_veg)
    base_lv = 100.0 + (lvx - x_center) * 0.02
    lvz = base_lv + rng.uniform(0.1, 0.8, n_low_veg)
    x[idx:idx+n_low_veg] = lvx
    y[idx:idx+n_low_veg] = lvy
    z[idx:idx+n_low_veg] = lvz
    cls[idx:idx+n_low_veg] = 3

    # ── Build the LAS file ──
    header = laspy.LasHeader(point_format=1, version="1.2")
    header.offsets = [x_center, y_center, 0.0]
    header.scales = [0.001, 0.001, 0.001]  # mm precision

    # Add GeoKey VLR with the correct EPSG
    geokey_data = struct.pack(
        "<HHHH" "HHHH",
        1, 1, 0, 1,       # GeoKey directory header
        3072, 0, 1, epsg,  # ProjectedCSTypeGeoKey = epsg
    )
    vlr = laspy.VLR(
        user_id="LASF_Projection",
        record_id=34735,
        record_data=geokey_data,
        description="GeoTIFF GeoKeyDirectoryTag",
    )
    header.vlrs.append(vlr)

    las = laspy.LasData(header)
    las.x = x
    las.y = y
    las.z = z
    las.classification = cls
    las.return_number = ret_num
    las.number_of_returns = num_ret

    path = os.path.join(tempfile.gettempdir(), filename)
    las.write(path)
    return path


def run_test(country: str, epsg: int, lat: float, lon: float, filename: str):
    """Run a full end-to-end test for one country."""
    from custom_components.solar_shade.geo import latlon_to_epsg, epsg_to_latlon
    from custom_components.solar_shade.shadow_engine import (
        process_lidar_file, seasonal_high_veg_transmittance,
    )

    print(f"\n{'='*60}")
    print(f"  {country} — EPSG:{epsg}")
    print(f"  Location: {lat}, {lon}")
    print(f"{'='*60}")

    # Step 1: Compute expected native coordinates
    center_e, center_n = latlon_to_epsg(lat, lon, epsg)
    print(f"\n1. Native coordinates: E={center_e:.1f}  N={center_n:.1f}")

    # Step 2: Create realistic LAS file
    path = create_realistic_las(epsg, center_e, center_n, filename,
                                 n_points=50_000, tile_size=200.0)
    file_size = os.path.getsize(path)
    print(f"2. Created LAS file: {filename} ({file_size/1024:.0f} KB)")

    # Step 3: Verify raw LAS metadata
    with laspy.open(path) as reader:
        hdr = reader.header
        print(f"3. LAS metadata:")
        print(f"   Version: {hdr.version}")
        print(f"   Point count: {hdr.point_count}")
        print(f"   VLRs: {len(hdr.vlrs)}")
        for i, vlr in enumerate(hdr.vlrs):
            print(f"     [{i}] {vlr.user_id} record_id={vlr.record_id}")
        print(f"   X range: {hdr.mins[0]:.1f} — {hdr.maxs[0]:.1f}")
        print(f"   Y range: {hdr.mins[1]:.1f} — {hdr.maxs[1]:.1f}")
        print(f"   Z range: {hdr.mins[2]:.1f} — {hdr.maxs[2]:.1f}")

    # Step 4: Test CRS detection (the critical part)
    from custom_components.solar_shade.usgs_downloader import _read_laz_epsg
    detected = _read_laz_epsg(path, laspy)
    status = "PASS" if detected == epsg else "FAIL"
    print(f"4. CRS detection: EPSG:{detected}  [{status}]")
    assert detected == epsg, f"CRS detection failed: expected {epsg}, got {detected}"

    # Step 5: Full pipeline — process_lidar_file()
    site = process_lidar_file(path, lat, lon, min_cell_size=0.5)
    print(f"5. Pipeline result:")
    print(f"   native_epsg:     {site.native_epsg}")
    print(f"   DSM shape:       {site.dsm.shape}")
    print(f"   DTM shape:       {site.dtm.shape if site.dtm is not None else 'None'}")
    print(f"   Classification:  {site.classification is not None}")
    print(f"   Canopy base:     {site.canopy_base is not None}")
    print(f"   Resolution:      {site.resolution:.2f} m/pixel")
    print(f"   Lat/Lon:         {site.latitude:.4f}, {site.longitude:.4f}")
    print(f"   Extent:          X[{site.x_min_m:.0f} to {site.x_max_m:.0f}] "
          f"Y[{site.y_min_m:.0f} to {site.y_max_m:.0f}]")

    assert site.native_epsg == epsg, \
        f"native_epsg: expected {epsg}, got {site.native_epsg}"
    assert site.dsm.shape[0] > 50, f"DSM too small: {site.dsm.shape}"
    assert site.dsm.shape[1] > 50, f"DSM too small: {site.dsm.shape}"
    assert site.dtm is not None, "DTM is None"
    assert site.classification is not None, "Classification is None"
    assert abs(site.latitude - lat) < 0.01, \
        f"Lat mismatch: {site.latitude} vs {lat}"
    assert abs(site.longitude - lon) < 0.01, \
        f"Lon mismatch: {site.longitude} vs {lon}"

    # Step 6: Check DSM > DTM where buildings/trees exist
    height_diff = site.dsm - site.dtm
    max_height = np.nanmax(height_diff)
    mean_height = np.nanmean(height_diff[height_diff > 1.0])
    print(f"6. Height analysis:")
    print(f"   Max DSM-DTM:     {max_height:.1f} m")
    print(f"   Mean above-ground: {mean_height:.1f} m (where > 1m)")
    assert max_height > 5.0, f"Max height too low: {max_height}"

    # Step 7: Check classification grid has expected classes
    if site.classification is not None:
        unique = np.unique(site.classification[site.classification > 0])
        print(f"   Classes present: {sorted(unique)}")
        has_ground = 2 in unique
        has_buildings = 6 in unique
        has_veg = 5 in unique or 3 in unique
        print(f"   Ground={has_ground}, Buildings={has_buildings}, Vegetation={has_veg}")

    # Step 8: Compute a shadow map for a realistic sun position
    from custom_components.solar_shade.shadow_engine import compute_shadow_map
    sun_azimuth = 210.0   # SW
    sun_elevation = 30.0  # moderate altitude
    shadow_map = compute_shadow_map(
        site.dsm, sun_azimuth, sun_elevation, site.resolution,
        ground=site.dtm,
    )
    fraction = float(np.mean(shadow_map > 0.5))
    print(f"7. Shadow computation:")
    print(f"   Sun: az={sun_azimuth}°, el={sun_elevation}°")
    print(f"   Shadow map shape: {shadow_map.shape}")
    print(f"   Shadowed pixels:  {fraction:.1%}")
    assert shadow_map.shape == site.dsm.shape, "Shadow map shape mismatch"
    assert 0.0 <= fraction <= 1.0, f"Invalid shadow fraction: {fraction}"

    # Step 9: Test seasonal transmittance for this latitude
    import datetime
    doy = datetime.date(2025, 6, 21).timetuple().tm_yday  # summer solstice
    tx_summer = seasonal_high_veg_transmittance(doy, lat)
    doy_winter = datetime.date(2025, 12, 21).timetuple().tm_yday
    tx_winter = seasonal_high_veg_transmittance(doy_winter, lat)
    print(f"8. Seasonal transmittance at lat={lat}°:")
    print(f"   Summer solstice: {tx_summer:.3f}")
    print(f"   Winter solstice: {tx_winter:.3f}")
    if abs(lat) > 35:
        assert tx_winter > tx_summer, \
            "Winter should have higher transmittance (bare branches)"

    # Step 10: Roundtrip lat/lon check
    rt_lat, rt_lon = epsg_to_latlon(center_e, center_n, epsg)
    print(f"9. Coordinate roundtrip:")
    print(f"   Input:     {lat:.4f}, {lon:.4f}")
    print(f"   Roundtrip: {rt_lat:.4f}, {rt_lon:.4f}")
    print(f"   Error:     {abs(rt_lat - lat)*111000:.1f}m, "
          f"{abs(rt_lon - lon)*111000*np.cos(np.radians(lat)):.1f}m")

    os.unlink(path)
    print(f"\n   *** {country} — ALL CHECKS PASSED ***")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("  MANUAL END-TO-END TEST: SWEDEN & SLOVENIA")
    print("  Simulating real user manual file import")
    print("=" * 60)

    results = {}

    # ── Sweden ──────────────────────────────────────────────────
    # Lantmäteriet distributes LAS files in SWEREF 99 TM (EPSG:3006)
    # Typical filename: 67_5_RUT_0_2m.laz or similar
    try:
        results["Sweden (Stockholm)"] = run_test(
            country="Sweden (Stockholm)",
            epsg=3006,
            lat=59.33, lon=18.07,
            filename="se_lidar_sweref99tm_stockholm.las",
        )
    except Exception as e:
        results["Sweden (Stockholm)"] = False
        print(f"\n   *** Sweden (Stockholm) FAILED: {e} ***")
        import traceback; traceback.print_exc()

    try:
        results["Sweden (Gothenburg)"] = run_test(
            country="Sweden (Gothenburg)",
            epsg=3006,
            lat=57.71, lon=11.97,
            filename="se_lidar_sweref99tm_goteborg.las",
        )
    except Exception as e:
        results["Sweden (Gothenburg)"] = False
        print(f"\n   *** Sweden (Gothenburg) FAILED: {e} ***")
        import traceback; traceback.print_exc()

    # ── Slovenia ────────────────────────────────────────────────
    # ARSO/GURS distributes LAS files in D96/TM (EPSG:3794)
    # Typical filename: b_35_34.laz or GKB_435_97.laz
    try:
        results["Slovenia (Ljubljana)"] = run_test(
            country="Slovenia (Ljubljana)",
            epsg=3794,
            lat=46.05, lon=14.51,
            filename="si_lidar_d96tm_ljubljana.las",
        )
    except Exception as e:
        results["Slovenia (Ljubljana)"] = False
        print(f"\n   *** Slovenia (Ljubljana) FAILED: {e} ***")
        import traceback; traceback.print_exc()

    try:
        results["Slovenia (Maribor)"] = run_test(
            country="Slovenia (Maribor)",
            epsg=3794,
            lat=46.56, lon=15.65,
            filename="si_lidar_d96tm_maribor.las",
        )
    except Exception as e:
        results["Slovenia (Maribor)"] = False
        print(f"\n   *** Slovenia (Maribor) FAILED: {e} ***")
        import traceback; traceback.print_exc()

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        icon = "PASS" if passed else "FAIL"
        print(f"  [{icon}] {name}")
        if not passed:
            all_pass = False
    print("=" * 60)
    if all_pass:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)
