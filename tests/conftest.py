"""Conftest for Solar Shade tests.

Mocks the homeassistant package so shadow_engine and usgs_downloader
can be imported without a full HA installation.
"""

import sys
from unittest.mock import MagicMock

# Mock homeassistant and its submodules so __init__.py can be imported
ha_mock = MagicMock()
sys.modules["homeassistant"] = ha_mock
sys.modules["homeassistant.config_entries"] = ha_mock
sys.modules["homeassistant.core"] = ha_mock
sys.modules["homeassistant.helpers"] = ha_mock
sys.modules["homeassistant.helpers.update_coordinator"] = ha_mock
sys.modules["homeassistant.helpers.selector"] = ha_mock
sys.modules["homeassistant.helpers.entity_platform"] = ha_mock
sys.modules["homeassistant.helpers.entity"] = ha_mock
sys.modules["homeassistant.helpers.aiohttp_client"] = ha_mock
sys.modules["homeassistant.helpers.sun"] = ha_mock
sys.modules["homeassistant.helpers.config_validation"] = ha_mock
sys.modules["homeassistant.components"] = ha_mock
sys.modules["homeassistant.components.sensor"] = ha_mock
sys.modules["homeassistant.components.websocket_api"] = ha_mock
sys.modules["homeassistant.components.frontend"] = ha_mock
sys.modules["homeassistant.util"] = ha_mock
sys.modules["homeassistant.util.dt"] = ha_mock

# Provide real base classes for CoordinatorEntity and SensorEntity so that
# sensor.py can define classes inheriting from both without metaclass conflict.
class _CoordinatorEntity:
    def __init__(self, coordinator, *args, **kwargs):
        self.coordinator = coordinator

class _SensorEntity:
    pass

ha_mock.CoordinatorEntity = _CoordinatorEntity
ha_mock.SensorEntity = _SensorEntity
ha_mock.SensorDeviceClass = MagicMock()
ha_mock.SensorStateClass = MagicMock()

# DeviceInfo stub
ha_mock.DeviceInfo = lambda **kwargs: kwargs
sys.modules["voluptuous"] = MagicMock()
sys.modules["aiohttp"] = MagicMock()
