"""Tests for international elevation data providers and CRS transforms."""

import math

import pytest

from custom_components.solar_shade.geo import (
    latlon_to_epsg,
    epsg_to_latlon,
    latlon_to_utm,
    utm_to_latlon,
)


# ── Generic EPSG conversion across multiple national CRS ────────────────

class TestEPSGForwardTransform:
    """Test latlon_to_epsg produces correct coordinates for various CRS."""

    # SWEREF99 TM (EPSG:3006) — Sweden

    def test_stockholm_sweref99tm(self):
        e, n = latlon_to_epsg(59.33, 18.07, 3006)
        assert 650_000 < e < 700_000
        assert 6_500_000 < n < 6_700_000

    def test_gothenburg_sweref99tm(self):
        e, n = latlon_to_epsg(57.71, 11.97, 3006)
        assert 300_000 < e < 500_000
        assert 6_300_000 < n < 6_500_000

    def test_kiruna_sweref99tm(self):
        e, n = latlon_to_epsg(67.86, 20.23, 3006)
        assert 400_000 < e < 800_000
        assert 7_500_000 < n < 7_600_000

    # D96/TM (EPSG:3794) — Slovenia

    def test_ljubljana_d96tm(self):
        e, n = latlon_to_epsg(46.05, 14.51, 3794)
        assert 440_000 < e < 480_000
        assert 80_000 < n < 120_000

    def test_maribor_d96tm(self):
        e, n = latlon_to_epsg(46.56, 15.65, 3794)
        assert 540_000 < e < 570_000
        assert 130_000 < n < 170_000

    def test_triglav_d96tm(self):
        e, n = latlon_to_epsg(46.38, 13.84, 3794)
        assert 400_000 < e < 440_000
        assert 110_000 < n < 150_000

    # Lambert-93 (EPSG:2154) — France

    def test_paris_lambert93(self):
        e, n = latlon_to_epsg(48.86, 2.35, 2154)
        assert 640_000 < e < 660_000
        assert 6_850_000 < n < 6_870_000

    def test_lyon_lambert93(self):
        e, n = latlon_to_epsg(45.76, 4.83, 2154)
        assert 830_000 < e < 850_000
        assert 6_510_000 < n < 6_530_000

    def test_marseille_lambert93(self):
        e, n = latlon_to_epsg(43.30, 5.37, 2154)
        assert 880_000 < e < 900_000
        assert 6_230_000 < n < 6_260_000

    # CH1903+/LV95 (EPSG:2056) — Switzerland

    def test_zurich_lv95(self):
        e, n = latlon_to_epsg(47.37, 8.54, 2056)
        assert 2_680_000 < e < 2_690_000
        assert 1_240_000 < n < 1_260_000

    def test_bern_lv95(self):
        e, n = latlon_to_epsg(46.95, 7.45, 2056)
        assert 2_590_000 < e < 2_610_000
        assert 1_190_000 < n < 1_210_000

    def test_geneva_lv95(self):
        e, n = latlon_to_epsg(46.20, 6.14, 2056)
        assert 2_490_000 < e < 2_510_000
        assert 1_110_000 < n < 1_130_000


class TestEPSGRoundtrip:
    """Test forward → inverse roundtrip accuracy for multiple EPSG codes."""

    @pytest.mark.parametrize("lat, lon, epsg", [
        # SWEREF99 TM
        (55.60, 12.99, 3006),   # Malmö
        (57.71, 11.97, 3006),   # Gothenburg
        (59.33, 18.07, 3006),   # Stockholm
        (63.83, 20.26, 3006),   # Umeå
        (67.86, 20.23, 3006),   # Kiruna
        # D96/TM
        (45.55, 13.73, 3794),   # Koper
        (46.05, 14.51, 3794),   # Ljubljana
        (46.56, 15.65, 3794),   # Maribor
        (46.38, 13.84, 3794),   # Triglav
        (46.23, 15.27, 3794),   # Celje
        # Lambert-93
        (48.86, 2.35, 2154),    # Paris
        (45.76, 4.83, 2154),    # Lyon
        (43.30, 5.37, 2154),    # Marseille
        (43.60, 1.44, 2154),    # Toulouse
        (47.22, -1.55, 2154),   # Nantes
        (48.57, 7.75, 2154),    # Strasbourg
        # CH1903+/LV95
        (47.37, 8.54, 2056),    # Zurich
        (46.95, 7.45, 2056),    # Bern
        (46.20, 6.14, 2056),    # Geneva
        (46.00, 8.95, 2056),    # Lugano
        (47.05, 8.31, 2056),    # Lucerne
    ])
    def test_roundtrip(self, lat, lon, epsg):
        e, n = latlon_to_epsg(lat, lon, epsg)
        lat2, lon2 = epsg_to_latlon(e, n, epsg)
        assert abs(lat2 - lat) < 1e-6, f"Lat roundtrip failed for EPSG:{epsg} ({lat}, {lon})"
        assert abs(lon2 - lon) < 1e-6, f"Lon roundtrip failed for EPSG:{epsg} ({lat}, {lon})"


class TestEPSGEdgeCases:
    """Test error handling and UTM interop."""

    def test_utm_via_epsg(self):
        """latlon_to_epsg should work for UTM North zones."""
        lat, lon = 32.28, -95.28
        zone, e1, n1 = latlon_to_utm(lat, lon)
        epsg = 32600 + zone
        e2, n2 = latlon_to_epsg(lat, lon, epsg)
        assert abs(e1 - e2) < 1.0
        assert abs(n1 - n2) < 1.0

    def test_unsupported_epsg_raises(self):
        from pyproj.exceptions import CRSError
        with pytest.raises(CRSError):
            latlon_to_epsg(46.0, 14.5, 9999)

    def test_inverse_unsupported_raises(self):
        from pyproj.exceptions import CRSError
        with pytest.raises(CRSError):
            epsg_to_latlon(500000.0, 100000.0, 9999)


# ── UTM backward compatibility ──────────────────────────────────────────

class TestUTMBackwardCompat:
    """Ensure the refactored UTM functions produce identical results."""

    def test_tyler_texas(self):
        """Tyler TX — same as original test."""
        zone, e, n = latlon_to_utm(32.28, -95.28)
        assert zone == 15
        assert 200_000 < e < 800_000
        assert 3_500_000 < n < 3_700_000

    def test_roundtrip(self):
        lat, lon = 32.2873, -95.2934
        zone, e, n = latlon_to_utm(lat, lon)
        lat2, lon2 = utm_to_latlon(zone, e, n, northern=True)
        assert abs(lat2 - lat) < 1e-8
        assert abs(lon2 - lon) < 1e-8


# ── Provider auto-detection ─────────────────────────────────────────────

class TestProviderDetection:
    """Test the provider auto-detection logic."""

    def test_us_detects_usgs(self):
        from custom_components.solar_shade.elevation_provider import detect_provider
        assert detect_provider("US") == "usgs"

    def test_france_detects_ign(self):
        from custom_components.solar_shade.elevation_provider import detect_provider
        assert detect_provider("FR") == "ign"

    def test_france_lowercase_detects_ign(self):
        from custom_components.solar_shade.elevation_provider import detect_provider
        assert detect_provider("fr") == "ign"

    def test_switzerland_detects_swisstopo(self):
        from custom_components.solar_shade.elevation_provider import detect_provider
        assert detect_provider("CH") == "swisstopo"

    def test_liechtenstein_detects_swisstopo(self):
        from custom_components.solar_shade.elevation_provider import detect_provider
        assert detect_provider("LI") == "swisstopo"

    def test_unknown_country_defaults_to_usgs(self):
        from custom_components.solar_shade.elevation_provider import detect_provider
        assert detect_provider("GB") == "usgs"

    def test_none_country_defaults_to_usgs(self):
        from custom_components.solar_shade.elevation_provider import detect_provider
        assert detect_provider(None) == "usgs"

    def test_us_territory_detects_usgs(self):
        from custom_components.solar_shade.elevation_provider import detect_provider
        assert detect_provider("PR") == "usgs"


# ── Provider registration ───────────────────────────────────────────────

class TestProviderRegistry:
    """Test the provider registry and instantiation."""

    def test_usgs_registered(self):
        from custom_components.solar_shade.elevation_provider import get_provider
        p = get_provider("usgs")
        assert p.PROVIDER_ID == "usgs"

    def test_ign_registered(self):
        from custom_components.solar_shade.elevation_provider import get_provider
        p = get_provider("ign")
        assert p.PROVIDER_ID == "ign"
        assert p.NATIVE_EPSG == 2154

    def test_swisstopo_registered(self):
        from custom_components.solar_shade.elevation_provider import get_provider
        p = get_provider("swisstopo")
        assert p.PROVIDER_ID == "swisstopo"
        assert p.NATIVE_EPSG == 2056

    def test_unknown_raises(self):
        from custom_components.solar_shade.elevation_provider import get_provider
        with pytest.raises(KeyError):
            get_provider("nonexistent")

    def test_list_providers_has_all(self):
        from custom_components.solar_shade.elevation_provider import list_providers
        providers = list_providers()
        ids = [p[0] for p in providers]
        assert "usgs" in ids
        assert "ign" in ids
        assert "swisstopo" in ids



# ── CRS handling in usgs_downloader ──────────────────────────────────────

class TestProjectToEpsg:
    """Test updated _project_to_epsg with international EPSG codes."""

    def test_sweref99tm_with_expected_epsg(self):
        """When expected_epsg matches target, returns (None, None)."""
        from custom_components.solar_shade.usgs_downloader import _project_to_epsg
        result = _project_to_epsg(670000.0, 6580000.0, 3006, expected_epsg=3006)
        assert result == (None, None)

    def test_d96tm_with_expected_epsg(self):
        from custom_components.solar_shade.usgs_downloader import _project_to_epsg
        result = _project_to_epsg(461000.0, 100000.0, 3794, expected_epsg=3794)
        assert result == (None, None)

    def test_sweref99tm_without_expected_returns_none(self):
        """SWEREF99 TM is a known CRS, should return (None, None)."""
        from custom_components.solar_shade.usgs_downloader import _project_to_epsg
        result = _project_to_epsg(670000.0, 6580000.0, 3006)
        assert result == (None, None)

    def test_d96tm_without_expected_returns_none(self):
        from custom_components.solar_shade.usgs_downloader import _project_to_epsg
        result = _project_to_epsg(461000.0, 100000.0, 3794)
        assert result == (None, None)

    def test_utm_north_still_works(self):
        from custom_components.solar_shade.usgs_downloader import _project_to_epsg
        result = _project_to_epsg(500000.0, 3575000.0, 32615)
        assert result == (None, None)

    def test_nad83_still_works(self):
        from custom_components.solar_shade.usgs_downloader import _project_to_epsg
        result = _project_to_epsg(500000.0, 3575000.0, 26915)
        assert result == (None, None)

    def test_unknown_epsg_returns_none(self):
        from custom_components.solar_shade.usgs_downloader import _project_to_epsg
        result = _project_to_epsg(500000.0, 3575000.0, 2277)
        assert result == (None, None)

    def test_lambert93_with_expected_epsg(self):
        """Lambert-93 (EPSG:2154) with matching expected_epsg."""
        from custom_components.solar_shade.usgs_downloader import _project_to_epsg
        result = _project_to_epsg(650000.0, 6860000.0, 2154, expected_epsg=2154)
        assert result == (None, None)

    def test_lv95_with_expected_epsg(self):
        """CH1903+/LV95 (EPSG:2056) with matching expected_epsg."""
        from custom_components.solar_shade.usgs_downloader import _project_to_epsg
        result = _project_to_epsg(2683000.0, 1248000.0, 2056, expected_epsg=2056)
        assert result == (None, None)


# ── IGN provider latlon_to_native ────────────────────────────────────────

class TestIGNLatLonToNative:
    """Test IGNProvider.latlon_to_native produces Lambert-93 coordinates."""

    def test_paris(self):
        from custom_components.solar_shade.ign_provider import IGNProvider
        p = IGNProvider()
        e, n = p.latlon_to_native(48.86, 2.35)
        assert 640_000 < e < 660_000
        assert 6_850_000 < n < 6_870_000

    def test_lyon(self):
        from custom_components.solar_shade.ign_provider import IGNProvider
        p = IGNProvider()
        e, n = p.latlon_to_native(45.76, 4.83)
        assert 830_000 < e < 850_000
        assert 6_510_000 < n < 6_530_000


# ── swisstopo provider latlon_to_native ──────────────────────────────────

class TestSwisstopoLatLonToNative:
    """Test SwisstopoProvider.latlon_to_native produces LV95 coordinates."""

    def test_zurich(self):
        from custom_components.solar_shade.swisstopo_provider import SwisstopoProvider
        p = SwisstopoProvider()
        e, n = p.latlon_to_native(47.37, 8.54)
        assert 2_680_000 < e < 2_690_000
        assert 1_240_000 < n < 1_260_000

    def test_bern(self):
        from custom_components.solar_shade.swisstopo_provider import SwisstopoProvider
        p = SwisstopoProvider()
        e, n = p.latlon_to_native(46.95, 7.45)
        assert 2_590_000 < e < 2_610_000
        assert 1_190_000 < n < 1_210_000
