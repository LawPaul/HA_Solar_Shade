"""Integration tests for USGS TNM API LAZ tile discovery.

These tests hit real USGS servers and verify the full end-to-end path
from lat/long to a downloadable LAZ file URL via the TNM API.

Run with: pytest tests/test_usgs_integration.py -m integration -v -s
"""

import json
import urllib.error
import urllib.parse
import urllib.request

import pytest

pytestmark = pytest.mark.integration

TNM_API_URL = "https://tnmaccess.nationalmap.gov/api/v1/products"


def _query_tnm(lat, lng, max_results=5):
    """Query TNM API for LAZ tiles covering a lat/lng (uses stdlib urllib).

    Retries up to 3 times with exponential backoff. Handles non-JSON error
    responses from overloaded sciencebase.gov backend.
    """
    import time

    delta = 0.002
    bbox = f"{lng - delta},{lat - delta},{lng + delta},{lat + delta}"
    params = urllib.parse.urlencode({
        "datasets": "Lidar Point Cloud (LPC)",
        "bbox": bbox,
        "max": str(max_results),
        "outputFormat": "JSON",
    })
    url = f"{TNM_API_URL}?{params}"

    last_err = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SolarShade/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                text = resp.read().decode()
            data = json.loads(text)
            # TNM API returns {errorMessage=...} when sciencebase is overloaded
            if "errorMessage" in data and "items" not in data:
                raise RuntimeError(f"TNM API error: {str(data.get('errorMessage',''))[:120]}")
            return data
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(5 * (attempt + 1))

    pytest.skip(f"TNM API unavailable after 3 retries: {last_err}")


def _head_url(url):
    """HTTP HEAD request to verify URL is downloadable."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "SolarShade/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


LOCATIONS = [
    ("Amityville NY", 40.6663, -73.4150),
    ("Sopranos House NJ", 40.8768, -74.2288),
    ("Austin TX", 30.267, -97.743),
    ("Denver CO", 39.7392, -104.9903),
]


class TestTNMDiscovery:
    """Test that the TNM API finds LAZ tiles for US locations."""

    @pytest.mark.parametrize("name,lat,lng", LOCATIONS)
    def test_finds_laz_tiles(self, name, lat, lng):
        data = _query_tnm(lat, lng)
        items = [
            i for i in data.get("items", [])
            if (i.get("downloadURL") or "").lower().endswith((".laz", ".las"))
        ]
        assert items, f"No LAZ tiles found at {name} ({lat}, {lng})"
        print(f"\n  {name}: {len(items)} tile(s)")
        for item in items[:3]:
            fname = (item.get("downloadURL") or "").rsplit("/", 1)[-1]
            print(f"    {fname} ({item.get('publicationDate', 'unknown')})")

    @pytest.mark.parametrize("name,lat,lng", LOCATIONS)
    def test_laz_url_downloadable(self, name, lat, lng):
        """Verify the first LAZ URL returns HTTP 200 on HEAD."""
        data = _query_tnm(lat, lng)
        items = [
            i for i in data.get("items", [])
            if (i.get("downloadURL") or "").lower().endswith((".laz", ".las"))
        ]
        assert items, f"No LAZ tiles at {name}"
        url = items[0]["downloadURL"]
        status = _head_url(url)
        assert status == 200, f"{name}: {url} returned {status}"
        fname = url.rsplit("/", 1)[-1]
        print(f"\n  {name}: HTTP 200 for {fname}")


class TestUTMConversion:
    """Test that _latlon_to_utm produces correct zones."""

    @pytest.mark.parametrize("name,lat,lng,expected_zone", [
        ("Amityville NY", 40.6663, -73.4150, 18),
        ("Austin TX", 30.267, -97.743, 14),
        ("Denver CO", 39.7392, -104.9903, 13),
    ])
    def test_utm_zone(self, name, lat, lng, expected_zone):
        from custom_components.solar_shade.usgs_downloader import _latlon_to_utm
        zone, _, _, northern = _latlon_to_utm(lat, lng)
        assert zone == expected_zone
        assert northern is True
