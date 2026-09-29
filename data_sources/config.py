import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Automatically load .env file from project root
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(base_dir, ".env"))

def get_env_var(name: str, default: Optional[str] = None) -> Optional[str]:
    """Retrieve environment variable without printing or exposing secrets."""
    return os.environ.get(name, default)

@dataclass(frozen=True)
class DataSourceConfig:
    # Copernicus CDS
    cds_api_url: Optional[str] = os.environ.get("CDS_API_URL", "https://cds.climate.copernicus.eu/api")
    cds_api_key: Optional[str] = os.environ.get("CDS_API_KEY")

    # Copernicus CAMS
    cams_api_url: Optional[str] = os.environ.get("CAMS_API_URL", "https://ads.atmosphere.copernicus.eu/api")
    cams_api_key: Optional[str] = os.environ.get("CAMS_API_KEY")

    # NASA FIRMS
    firms_map_key: Optional[str] = os.environ.get("FIRMS_MAP_KEY")

    # Copernicus Sentinel-5P
    sentinel_client_id: Optional[str] = os.environ.get("SENTINEL_API_CLIENT_ID")
    sentinel_client_secret: Optional[str] = os.environ.get("SENTINEL_API_CLIENT_SECRET")

    # CPCB / DPCC
    cpcb_api_key: Optional[str] = os.environ.get("CPCB_API_KEY")
    cpcb_api_base_url: str = os.environ.get("CPCB_API_BASE_URL", "https://app.cpcbccr.com/caaqms")

    # WRF Output Storage
    wrf_output_dir: str = os.environ.get("WRF_OUTPUT_DIR", "data/wrf_output")
    wrf_grid_resolution_km: float = float(os.environ.get("WRF_GRID_RESOLUTION_KM", "3.0"))

    # Cache TTLs (seconds)
    cache_ttl_observations: int = int(os.environ.get("CACHE_TTL_OBSERVATIONS_SEC", "3600"))
    cache_ttl_cams: int = int(os.environ.get("CACHE_TTL_CAMS_SEC", "21600"))
    cache_ttl_firms: int = int(os.environ.get("CACHE_TTL_FIRMS_SEC", "10800"))
    cache_ttl_wrf: int = int(os.environ.get("CACHE_TTL_WRF_SEC", "21600"))

# Default configuration instance
config = DataSourceConfig()
