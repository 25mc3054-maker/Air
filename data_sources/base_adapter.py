import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

@dataclass
class DataRecord:
    """
    Standardized observation and feature record contract across all ATMOSAIR data sources.
    """
    timestamp: str            # ISO 8601 UTC timestamp of observation/forecast validity
    latitude: float           # Geographic coordinate
    longitude: float          # Geographic coordinate
    variable: str             # Normalized variable identifier (e.g. 'pm25', 'temp_c', 'frp')
    value: Optional[float]    # Observed or modeled value
    source: str               # Source identifier ('CPCB', 'CAMS', 'NASA_FIRMS', 'SENTINEL_5P', 'WRF', 'ERA5')
    quality_flag: str         # 'VALID', 'SUSPECT', 'IMPUTED', 'MISSING', 'FALLBACK'
    retrieval_timestamp: str  # ISO 8601 UTC timestamp when data was fetched

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class BaseAdapter(ABC):
    """
    Base class for all data source adapters with rate-limiting, error handling, 
    and offline fallback mechanisms.
    """
    def __init__(self, name: str, min_interval_seconds: float = 60.0):
        self.name = name
        self.min_interval = min_interval_seconds
        self.last_fetch_time: float = 0.0
        self.last_status: str = "INITIALIZED"
        self.last_error: Optional[str] = None

    def _rate_limit(self):
        """Enforces provider courtesy rate-limits."""
        now = time.time()
        elapsed = now - self.last_fetch_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_fetch_time = time.time()

    @staticmethod
    def current_utc_iso() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    @abstractmethod
    def is_configured(self) -> bool:
        """Returns True if required credentials / paths are present."""
        pass

    @abstractmethod
    def fetch(self, start_time: datetime, end_time: datetime, **kwargs) -> List[DataRecord]:
        """Fetches data from source or executes fallback."""
        pass
