"""Constants for Solar Shade integration."""

DOMAIN = "solar_shade"

CONF_RADIATION_ENTITY = "radiation_entity"
CONF_DIFFUSE_ENTITY = "diffuse_entity"
CONF_DIFFUSE_FRACTION = "diffuse_fraction"
CONF_ZONES = "zones"
CONF_LIDAR_FILE = "lidar_file"
CONF_DSM_SOURCE = "dsm_source"

DSM_SOURCE_AUTO = "auto"
DSM_SOURCE_LAZ = "laz_file"

CONF_LIDAR_PROJECT = "lidar_project"

DEFAULT_DIFFUSE_FRACTION = 0.15
DEFAULT_RESOLUTION = 0.5
DEFAULT_MIN_CELL_SIZE = 0.5
DEFAULT_UPDATE_INTERVAL = 5
DEFAULT_DOWNLOAD_RADIUS = 50
DEFAULT_MIN_SHADOW_HEIGHT = 0.1
CONF_MIN_CELL_SIZE = "min_cell_size"
CONF_UPDATE_INTERVAL = "update_interval"
CONF_DOWNLOAD_RADIUS = "download_radius"
CONF_MIN_SHADOW_HEIGHT = "min_shadow_height"
CONF_SHADE_METHOD = "shade_method"
DATA_DIR = "solar_shade"

SHADE_METHOD_AVERAGE = "average"
SHADE_METHOD_SUNNIEST = "sunniest"
SHADE_METHOD_SHADIEST = "shadiest"
DEFAULT_SHADE_METHOD = SHADE_METHOD_AVERAGE

ATTR_ZONE_NAME = "zone_name"
ATTR_SHADE_FRACTION = "shade_fraction"
ATTR_RAW_RADIATION = "raw_radiation"
ATTR_SUN_AZIMUTH = "sun_azimuth"
ATTR_SUN_ELEVATION = "sun_elevation"

SERVICE_RELOAD_SITE = "reload_site"

CONF_ENABLE_OPEN_METEO = "enable_open_meteo"
DEFAULT_ENABLE_OPEN_METEO = True
OPEN_METEO_UPDATE_INTERVAL = 15  # minutes

# Default Open-Meteo entity IDs (used when Open-Meteo is enabled and no custom entity set)
DEFAULT_RADIATION_ENTITY = "sensor.solar_shade_shortwave_radiation"
DEFAULT_DIFFUSE_ENTITY = "sensor.solar_shade_diffuse_radiation"

CONF_LATITUDE = "latitude_override"
CONF_LONGITUDE = "longitude_override"

CONF_CANOPY_MODEL = "canopy_model"
CANOPY_MODEL_SOLID = "solid"
CANOPY_MODEL_RAISED = "raised"
DEFAULT_CANOPY_MODEL = CANOPY_MODEL_RAISED

CONF_DSM_GAP_FILL = "dsm_gap_fill"
DEFAULT_DSM_GAP_FILL = False

CONF_MANUAL_EPSG = "manual_epsg"
DEFAULT_MANUAL_EPSG = 0  # 0 = auto-detect from file metadata

CONF_DSM_PROVIDER = "dsm_provider"
DSM_PROVIDER_AUTO = "auto"
DSM_PROVIDER_USGS = "usgs"
DSM_PROVIDER_IGN = "ign"
DSM_PROVIDER_SWISSTOPO = "swisstopo"
DSM_PROVIDER_NRW = "nrw"
DSM_PROVIDER_PDOK = "pdok"
DEFAULT_DSM_PROVIDER = DSM_PROVIDER_AUTO
