"""Unit tests for NRW and PDOK elevation providers.

Tests run fast (no network) using mocked HTTP responses.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════
# NRW Provider tests
# ═══════════════════════════════════════════════════════════════════════

class TestNRWProvider:
    """Test NRW provider tile URL construction and HEAD check."""

    def test_provider_registration(self):
        from custom_components.solar_shade.elevation_provider import get_provider
        provider = get_provider("nrw")
        assert provider.PROVIDER_ID == "nrw"
        assert provider.NATIVE_EPSG == 25832
        assert "DE" in provider.COUNTRY_CODES

    def test_auto_detect_germany(self):
        from custom_components.solar_shade.elevation_provider import detect_provider
        assert detect_provider("DE") == "nrw"

    def test_latlon_to_native(self):
        from custom_components.solar_shade.elevation_provider import get_provider
        provider = get_provider("nrw")
        e, n = provider.latlon_to_native(50.94, 6.96)
        # Cologne in EPSG:25832: ~356676 E, ~5645134 N
        assert 350000 < e < 360000
        assert 5640000 < n < 5650000

    def test_tile_url_construction(self):
        """Verify the tile URL pattern for Cologne."""
        from custom_components.solar_shade.elevation_provider import get_provider
        provider = get_provider("nrw")
        e, n = provider.latlon_to_native(50.94, 6.96)
        e_km = int(e / 1000)
        n_km = int(n / 1000)
        expected_url = (
            f"https://www.opengeodata.nrw.de/produkte/geobasis/hm/"
            f"3dm_l_las/3dm_l_las/3dm_32_{e_km}_{n_km}_1_nw.laz"
        )
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_resp
        mock_session = MagicMock()
        mock_session.head.return_value = mock_cm

        tiles = asyncio.run(provider.find_tiles(50.94, 6.96, mock_session))
        assert len(tiles) == 1
        assert tiles[0]["url"] == expected_url

    def test_tile_not_found(self):
        """HEAD returning 404 should return empty list."""
        from custom_components.solar_shade.elevation_provider import get_provider
        provider = get_provider("nrw")

        mock_resp = AsyncMock()
        mock_resp.status = 404
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_resp
        mock_session = MagicMock()
        mock_session.head.return_value = mock_cm

        tiles = asyncio.run(provider.find_tiles(48.0, 11.0, mock_session))
        assert tiles == []

    def test_head_network_error_returns_empty(self):
        """Network errors during HEAD check should return empty, not raise."""
        from custom_components.solar_shade.elevation_provider import get_provider
        provider = get_provider("nrw")

        mock_cm = AsyncMock()
        mock_cm.__aenter__.side_effect = aiohttp.ClientError("connection refused")
        mock_session = MagicMock()
        mock_session.head.return_value = mock_cm

        tiles = asyncio.run(provider.find_tiles(50.94, 6.96, mock_session))
        assert tiles == []

    def test_download_elevation_success(self):
        """download_elevation() should call find_tiles → download_and_rasterize_laz."""
        from custom_components.solar_shade.elevation_provider import get_provider
        provider = get_provider("nrw")

        fake_result = (
            np.ones((100, 100), dtype=np.float32),  # dsm
            np.ones((100, 100), dtype=np.float32),  # dtm
            np.full((100, 100), 2, dtype=np.uint8),  # classification
            None,  # canopy_base
            356000.0,  # x_min
            5645000.0,  # y_min
            356100.0,  # x_max
            5645100.0,  # y_max
            1.0,  # resolution
        )

        with patch(
            "custom_components.solar_shade.elevation_provider.aiohttp.ClientSession"
        ) as MockSession, patch(
            "custom_components.solar_shade.usgs_downloader.download_and_rasterize_laz",
            new_callable=AsyncMock,
            return_value=fake_result,
        ) as mock_rasterize:
            instance = MockSession.return_value
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)

            # Mock HEAD returning 200 (tile exists)
            head_resp = AsyncMock()
            head_resp.status = 200
            head_cm = AsyncMock()
            head_cm.__aenter__.return_value = head_resp
            instance.head = MagicMock(return_value=head_cm)

            result = asyncio.run(
                provider.download_elevation(50.94, 6.96, radius_m=150)
            )

        assert result is not None
        dsm, dtm, cls_grid, canopy, x_min, y_min, x_max, y_max, res = result
        assert dsm.shape == (100, 100)
        assert dtm is not None
        assert cls_grid is not None
        assert res == 1.0
        mock_rasterize.assert_called_once()

    def test_download_elevation_no_tile(self):
        """download_elevation() returns None when tile is not found."""
        from custom_components.solar_shade.elevation_provider import get_provider
        provider = get_provider("nrw")

        with patch(
            "custom_components.solar_shade.elevation_provider.aiohttp.ClientSession"
        ) as MockSession:
            instance = MockSession.return_value
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)

            # Mock HEAD returning 404 (tile not found)
            head_resp = AsyncMock()
            head_resp.status = 404
            head_cm = AsyncMock()
            head_cm.__aenter__.return_value = head_resp
            instance.head = MagicMock(return_value=head_cm)

            result = asyncio.run(
                provider.download_elevation(48.0, 11.0, radius_m=150)
            )

        assert result is None

    def test_download_elevation_rasterize_fails(self):
        """download_elevation() returns None when rasterization fails."""
        from custom_components.solar_shade.elevation_provider import get_provider
        provider = get_provider("nrw")

        with patch(
            "custom_components.solar_shade.elevation_provider.aiohttp.ClientSession"
        ) as MockSession, patch(
            "custom_components.solar_shade.usgs_downloader.download_and_rasterize_laz",
            new_callable=AsyncMock,
            return_value=None,
        ):
            instance = MockSession.return_value
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)

            head_resp = AsyncMock()
            head_resp.status = 200
            head_cm = AsyncMock()
            head_cm.__aenter__.return_value = head_resp
            instance.head = MagicMock(return_value=head_cm)

            result = asyncio.run(
                provider.download_elevation(50.94, 6.96, radius_m=150)
            )

        assert result is None


# ═══════════════════════════════════════════════════════════════════════
# PDOK Provider tests
# ═══════════════════════════════════════════════════════════════════════

class TestPDOKProvider:
    """Test Netherlands PDOK provider."""

    def test_provider_registration(self):
        from custom_components.solar_shade.elevation_provider import get_provider
        provider = get_provider("pdok")
        assert provider.PROVIDER_ID == "pdok"
        assert provider.NATIVE_EPSG == 28992
        assert "NL" in provider.COUNTRY_CODES

    def test_auto_detect_netherlands(self):
        from custom_components.solar_shade.elevation_provider import detect_provider
        assert detect_provider("NL") == "pdok"

    def test_latlon_to_native(self):
        from custom_components.solar_shade.elevation_provider import get_provider
        provider = get_provider("pdok")
        e, n = provider.latlon_to_native(52.37, 4.89)
        # Amsterdam in EPSG:28992 (RD New): ~121000 E, ~487000 N
        assert 115000 < e < 125000
        assert 480000 < n < 490000

    def test_find_tiles_parses_response(self):
        """Verify PDOK API response parsing."""
        from custom_components.solar_shade.elevation_provider import get_provider
        provider = get_provider("pdok")

        mock_json = {
            "features": [
                {
                    "properties": {
                        "bladnr": "1210_4870",
                        "download_link": "https://download.pdok.nl/test/DSM_1210_4870.laz",
                        "download_size_bytes": 3389539,
                    }
                },
            ]
        }
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_json)
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_resp
        mock_session = MagicMock()
        mock_session.get.return_value = mock_cm

        tiles = asyncio.run(provider.find_tiles(52.37, 4.89, mock_session))
        assert len(tiles) == 1
        assert tiles[0]["url"] == "https://download.pdok.nl/test/DSM_1210_4870.laz"
        assert tiles[0]["title"] == "1210_4870"

    def test_find_tiles_empty(self):
        """No features should return empty list."""
        from custom_components.solar_shade.elevation_provider import get_provider
        provider = get_provider("pdok")

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"features": []})
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_resp
        mock_session = MagicMock()
        mock_session.get.return_value = mock_cm

        tiles = asyncio.run(provider.find_tiles(52.37, 4.89, mock_session))
        assert tiles == []

    def test_find_tiles_http_error(self):
        """PDOK API returning non-200 should return empty list."""
        from custom_components.solar_shade.elevation_provider import get_provider
        provider = get_provider("pdok")

        mock_resp = AsyncMock()
        mock_resp.status = 503
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_resp
        mock_session = MagicMock()
        mock_session.get.return_value = mock_cm

        tiles = asyncio.run(provider.find_tiles(52.37, 4.89, mock_session))
        assert tiles == []

    def test_find_tiles_missing_download_link(self):
        """Features without download_link should be skipped."""
        from custom_components.solar_shade.elevation_provider import get_provider
        provider = get_provider("pdok")

        mock_json = {
            "features": [
                {"properties": {"bladnr": "1210_4870"}},  # no download_link
                {
                    "properties": {
                        "bladnr": "1211_4870",
                        "download_link": "https://download.pdok.nl/test/DSM_1211_4870.laz",
                    }
                },
            ]
        }
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_json)
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_resp
        mock_session = MagicMock()
        mock_session.get.return_value = mock_cm

        tiles = asyncio.run(provider.find_tiles(52.37, 4.89, mock_session))
        assert len(tiles) == 1
        assert tiles[0]["title"] == "1211_4870"

    def test_download_elevation_success(self):
        """download_elevation() should call find_tiles → download_and_rasterize_laz."""
        from custom_components.solar_shade.elevation_provider import get_provider
        provider = get_provider("pdok")

        fake_result = (
            np.ones((200, 200), dtype=np.float32),  # dsm
            np.ones((200, 200), dtype=np.float32),  # dtm
            np.full((200, 200), 2, dtype=np.uint8),  # classification
            None,  # canopy_base
            121000.0,  # x_min
            487000.0,  # y_min
            121200.0,  # x_max
            487200.0,  # y_max
            1.0,  # resolution
        )

        mock_pdok_json = {
            "features": [
                {
                    "properties": {
                        "bladnr": "1210_4870",
                        "download_link": "https://download.pdok.nl/test/DSM_1210_4870.laz",
                        "download_size_bytes": 3389539,
                    }
                },
            ]
        }

        with patch(
            "custom_components.solar_shade.elevation_provider.aiohttp.ClientSession"
        ) as MockSession, patch(
            "custom_components.solar_shade.usgs_downloader.download_and_rasterize_laz",
            new_callable=AsyncMock,
            return_value=fake_result,
        ) as mock_rasterize:
            instance = MockSession.return_value
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)

            # Mock GET returning OGC API Features response
            get_resp = AsyncMock()
            get_resp.status = 200
            get_resp.json = AsyncMock(return_value=mock_pdok_json)
            get_cm = AsyncMock()
            get_cm.__aenter__.return_value = get_resp
            instance.get = MagicMock(return_value=get_cm)

            result = asyncio.run(
                provider.download_elevation(52.37, 4.89, radius_m=150)
            )

        assert result is not None
        dsm, dtm, cls_grid, canopy, x_min, y_min, x_max, y_max, res = result
        assert dsm.shape == (200, 200)
        assert dtm is not None
        assert res == 1.0
        mock_rasterize.assert_called_once()

    def test_download_elevation_no_tiles(self):
        """download_elevation() returns None when PDOK has no tiles."""
        from custom_components.solar_shade.elevation_provider import get_provider
        provider = get_provider("pdok")

        with patch(
            "custom_components.solar_shade.elevation_provider.aiohttp.ClientSession"
        ) as MockSession:
            instance = MockSession.return_value
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)

            get_resp = AsyncMock()
            get_resp.status = 200
            get_resp.json = AsyncMock(return_value={"features": []})
            get_cm = AsyncMock()
            get_cm.__aenter__.return_value = get_resp
            instance.get = MagicMock(return_value=get_cm)

            result = asyncio.run(
                provider.download_elevation(52.37, 4.89, radius_m=150)
            )

        assert result is None

    def test_download_elevation_rasterize_fails(self):
        """download_elevation() returns None when LAZ rasterization fails."""
        from custom_components.solar_shade.elevation_provider import get_provider
        provider = get_provider("pdok")

        mock_pdok_json = {
            "features": [
                {
                    "properties": {
                        "bladnr": "1210_4870",
                        "download_link": "https://download.pdok.nl/test/DSM_1210_4870.laz",
                    }
                },
            ]
        }

        with patch(
            "custom_components.solar_shade.elevation_provider.aiohttp.ClientSession"
        ) as MockSession, patch(
            "custom_components.solar_shade.usgs_downloader.download_and_rasterize_laz",
            new_callable=AsyncMock,
            return_value=None,
        ):
            instance = MockSession.return_value
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)

            get_resp = AsyncMock()
            get_resp.status = 200
            get_resp.json = AsyncMock(return_value=mock_pdok_json)
            get_cm = AsyncMock()
            get_cm.__aenter__.return_value = get_resp
            instance.get = MagicMock(return_value=get_cm)

            result = asyncio.run(
                provider.download_elevation(52.37, 4.89, radius_m=150)
            )

        assert result is None


# ═══════════════════════════════════════════════════════════════════════
# Provider listing and detection tests
# ═══════════════════════════════════════════════════════════════════════

class TestProviderRegistry:
    """Test the provider registry with all 5 point-cloud providers."""

    def test_all_providers_registered(self):
        from custom_components.solar_shade.elevation_provider import list_providers
        providers = dict(list_providers())
        assert "usgs" in providers
        assert "ign" in providers
        assert "swisstopo" in providers
        assert "nrw" in providers
        assert "pdok" in providers
        assert len(providers) == 5

    def test_detect_provider_fallback(self):
        from custom_components.solar_shade.elevation_provider import detect_provider
        # Unknown country should fall back to USGS
        assert detect_provider("JP") == "usgs"
        assert detect_provider(None) == "usgs"

    def test_detect_all_countries(self):
        from custom_components.solar_shade.elevation_provider import detect_provider
        assert detect_provider("US") == "usgs"
        assert detect_provider("FR") == "ign"
        assert detect_provider("CH") == "swisstopo"
        assert detect_provider("DE") == "nrw"
        assert detect_provider("NL") == "pdok"
