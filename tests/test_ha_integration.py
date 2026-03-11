"""HACS and Home Assistant integration contract tests.

These tests validate the structural and contractual requirements
for a valid HA custom integration distributed via HACS:
- manifest.json correctness
- hacs.json correctness
- Required files/folders presence
- Config flow class structure
- strings.json / translations completeness
- Setup/unload entry signatures
- Service registration
"""

import json
import importlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

INTEGRATION_DIR = Path(__file__).parent.parent / "custom_components" / "solar_shade"
ROOT_DIR = Path(__file__).parent.parent

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ── manifest.json ────────────────────────────────────────────────────


class TestManifest:
    """Validate manifest.json meets HA requirements."""

    @pytest.fixture(autouse=True)
    def load_manifest(self):
        with open(INTEGRATION_DIR / "manifest.json", encoding="utf-8") as f:
            self.manifest = json.load(f)

    def test_is_valid_json(self):
        assert isinstance(self.manifest, dict)

    def test_has_domain(self):
        assert "domain" in self.manifest
        assert self.manifest["domain"] == "solar_shade"

    def test_has_name(self):
        assert "name" in self.manifest
        assert len(self.manifest["name"]) > 0

    def test_has_version(self):
        assert "version" in self.manifest
        # Version should be semver-like
        parts = self.manifest["version"].split(".")
        assert len(parts) >= 2

    def test_has_documentation(self):
        assert "documentation" in self.manifest
        assert self.manifest["documentation"].startswith("https://")

    def test_has_codeowners(self):
        assert "codeowners" in self.manifest
        assert isinstance(self.manifest["codeowners"], list)
        assert len(self.manifest["codeowners"]) > 0
        for owner in self.manifest["codeowners"]:
            assert owner.startswith("@")

    def test_has_requirements(self):
        assert "requirements" in self.manifest
        assert isinstance(self.manifest["requirements"], list)

    def test_has_config_flow(self):
        """config_flow must be true for integrations with UI setup."""
        assert self.manifest.get("config_flow") is True

    def test_has_iot_class(self):
        """iot_class is required for HA integrations."""
        assert "iot_class" in self.manifest
        valid_classes = [
            "assumed_state", "calculated", "cloud_polling", "cloud_push",
            "local_polling", "local_push",
        ]
        assert self.manifest["iot_class"] in valid_classes

    def test_domain_matches_folder_name(self):
        assert self.manifest["domain"] == INTEGRATION_DIR.name

    def test_no_empty_requirements(self):
        for req in self.manifest.get("requirements", []):
            assert len(req.strip()) > 0


# ── hacs.json ────────────────────────────────────────────────────────


class TestHacsJson:
    """Validate hacs.json meets HACS requirements."""

    @pytest.fixture(autouse=True)
    def load_hacs(self):
        with open(ROOT_DIR / "hacs.json", encoding="utf-8") as f:
            self.hacs = json.load(f)

    def test_is_valid_json(self):
        assert isinstance(self.hacs, dict)

    def test_has_name(self):
        assert "name" in self.hacs
        assert len(self.hacs["name"]) > 0

    def test_has_render_readme(self):
        """render_readme should be true for HACS to display README."""
        assert self.hacs.get("render_readme") is True


# ── Required Files ───────────────────────────────────────────────────


class TestRequiredFiles:
    """Validate all required files exist."""

    def test_init_py_exists(self):
        assert (INTEGRATION_DIR / "__init__.py").is_file()

    def test_manifest_exists(self):
        assert (INTEGRATION_DIR / "manifest.json").is_file()

    def test_config_flow_exists(self):
        assert (INTEGRATION_DIR / "config_flow.py").is_file()

    def test_const_exists(self):
        assert (INTEGRATION_DIR / "const.py").is_file()

    def test_strings_json_exists(self):
        assert (INTEGRATION_DIR / "strings.json").is_file()

    def test_translations_en_exists(self):
        assert (INTEGRATION_DIR / "translations" / "en.json").is_file()

    def test_hacs_json_exists(self):
        assert (ROOT_DIR / "hacs.json").is_file()

    def test_readme_exists(self):
        assert (ROOT_DIR / "README.md").is_file()

    def test_services_yaml_exists(self):
        assert (INTEGRATION_DIR / "services.yaml").is_file()

    def test_sensor_platform_exists(self):
        assert (INTEGRATION_DIR / "sensor.py").is_file()


# ── strings.json / translations ──────────────────────────────────────


class TestStrings:
    """Validate strings.json and translations consistency."""

    @pytest.fixture(autouse=True)
    def load_strings(self):
        with open(INTEGRATION_DIR / "strings.json", encoding="utf-8") as f:
            self.strings = json.load(f)
        with open(
            INTEGRATION_DIR / "translations" / "en.json", encoding="utf-8"
        ) as f:
            self.translations = json.load(f)

    def test_has_config_section(self):
        assert "config" in self.strings

    def test_has_options_section(self):
        assert "options" in self.strings

    def test_config_has_step_user(self):
        assert "step" in self.strings["config"]
        assert "user" in self.strings["config"]["step"]

    def test_user_step_has_title(self):
        assert "title" in self.strings["config"]["step"]["user"]

    def test_options_has_step(self):
        assert "step" in self.strings["options"]

    def test_translations_matches_strings(self):
        """translations/en.json should match strings.json."""
        assert self.strings == self.translations

    def test_abort_reasons_exist(self):
        """Config flow abort reasons should be defined."""
        assert "abort" in self.strings["config"]
        assert "single_instance_allowed" in self.strings["config"]["abort"]


# ── Module Structure ────────────────────────────────────────────────


class TestModuleStructure:
    """Validate integration module exports required symbols."""

    def test_init_exports_setup_entry(self):
        from custom_components.solar_shade import async_setup_entry
        assert callable(async_setup_entry)

    def test_init_exports_unload_entry(self):
        from custom_components.solar_shade import async_unload_entry
        assert callable(async_unload_entry)

    def test_init_exports_remove_entry(self):
        from custom_components.solar_shade import async_remove_entry
        assert callable(async_remove_entry)

    def test_config_flow_class_exists(self):
        from custom_components.solar_shade.config_flow import SolarShadeConfigFlow
        # In test env, it inherits from MagicMock, but the class should exist
        assert SolarShadeConfigFlow is not None

    def test_config_flow_has_version(self):
        source = (INTEGRATION_DIR / "config_flow.py").read_text(encoding="utf-8")
        assert "VERSION" in source
        # Verify VERSION is a positive integer
        import re
        match = re.search(r'VERSION\s*=\s*(\d+)', source)
        assert match, "VERSION must be defined as an integer"
        assert int(match.group(1)) >= 1

    def test_sensor_platform_has_setup_entry(self):
        # Can't import sensor.py due to mock metaclass conflicts,
        # so verify the function signature exists in source
        source = (INTEGRATION_DIR / "sensor.py").read_text(encoding="utf-8")
        assert "async def async_setup_entry" in source

    def test_domain_constant_matches_manifest(self):
        from custom_components.solar_shade.const import DOMAIN
        with open(INTEGRATION_DIR / "manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)
        assert DOMAIN == manifest["domain"]

    def test_platforms_list_defined(self):
        from custom_components.solar_shade import PLATFORMS
        assert isinstance(PLATFORMS, list)
        assert "sensor" in PLATFORMS


# ── Services ─────────────────────────────────────────────────────────


class TestServices:
    """Validate services registration."""

    @pytest.fixture(autouse=True)
    def load_services(self):
        if not HAS_YAML:
            pytest.skip("PyYAML not installed")
        with open(INTEGRATION_DIR / "services.yaml", encoding="utf-8") as f:
            self.services = yaml.safe_load(f)

    def test_services_yaml_is_valid(self):
        assert isinstance(self.services, dict)

    def test_services_yaml_has_reload_site(self):
        from custom_components.solar_shade.const import SERVICE_RELOAD_SITE
        assert SERVICE_RELOAD_SITE in self.services

    def test_service_has_description(self):
        for name, svc in self.services.items():
            assert "description" in svc, f"Service {name} missing description"


# ── WebSocket Commands ───────────────────────────────────────────────


class TestWebSocketRegistration:
    """Validate websocket API handler exports."""

    def test_register_function_exists(self):
        from custom_components.solar_shade.websocket_api import (
            async_register_websocket_api,
        )
        assert callable(async_register_websocket_api)

    def test_all_handlers_are_callable(self):
        from custom_components.solar_shade import websocket_api
        handlers = [
            "ws_get_config",
            "ws_save_zones",
            "ws_update_radius",
            "ws_clear_cache",
            "ws_get_dsm_image",
            "ws_get_shadow_preview",
            "ws_get_dsm_data",
            "ws_get_shade_timeline",
            "ws_get_satellite_image",
            "ws_get_surface_type_image",
        ]
        for name in handlers:
            assert hasattr(websocket_api, name), f"Missing handler: {name}"
            assert callable(getattr(websocket_api, name))
