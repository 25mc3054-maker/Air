import os
import json
import time
from typing import Any, Optional, Dict
from data_sources.config import config

class DataCache:
    """
    Tiered In-Memory and On-Disk Cache for External API Responses and Ingestion Buffers.
    Enforces source-specific Time-To-Live (TTL) policies to prevent rate-limit exhaustion.
    """
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self._memory_store: Dict[str, Dict[str, Any]] = {}

    def _get_ttl_for_source(self, source: str) -> int:
        s = source.upper()
        if "CPCB" in s or "DPCC" in s:
            return config.cache_ttl_observations
        elif "CAMS" in s:
            return config.cache_ttl_cams
        elif "FIRMS" in s:
            return config.cache_ttl_firms
        elif "WRF" in s:
            return config.cache_ttl_wrf
        return 3600

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        # Check memory
        if key in self._memory_store:
            entry = self._memory_store[key]
            if now < entry["expires_at"]:
                return entry["data"]
            else:
                del self._memory_store[key]

        # Check disk
        disk_path = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(disk_path):
            try:
                with open(disk_path, "r") as f:
                    entry = json.load(f)
                if now < entry["expires_at"]:
                    # Populate memory
                    self._memory_store[key] = entry
                    return entry["data"]
                else:
                    os.remove(disk_path)
            except Exception:
                pass
        return None

    def set(self, key: str, data: Any, source: str = "DEFAULT", custom_ttl_sec: Optional[int] = None):
        ttl = custom_ttl_sec if custom_ttl_sec is not None else self._get_ttl_for_source(source)
        expires_at = time.time() + ttl
        entry = {
            "key": key,
            "source": source,
            "saved_at": time.time(),
            "expires_at": expires_at,
            "data": data
        }
        self._memory_store[key] = entry
        
        disk_path = os.path.join(self.cache_dir, f"{key}.json")
        try:
            with open(disk_path, "w") as f:
                json.dump(entry, f)
        except Exception:
            pass

    def clear(self):
        self._memory_store.clear()
        for f in os.listdir(self.cache_dir):
            if f.endswith(".json"):
                try:
                    os.remove(os.path.join(self.cache_dir, f))
                except Exception:
                    pass

# Global Cache Singleton
cache = DataCache()
