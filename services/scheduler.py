import time
import threading
from datetime import datetime, timezone
from typing import Callable, Dict, Any, Optional
from services.cache import cache

class IngestionScheduler:
    """
    Cadence-aware ingestion scheduler for external environmental data providers.
    Prevents unnecessary continuous polling of static external endpoints.
    """
    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def register_job(self, name: str, interval_seconds: int, func: Callable, *args, **kwargs):
        self._jobs[name] = {
            "interval": interval_seconds,
            "func": func,
            "args": args,
            "kwargs": kwargs,
            "last_run": 0.0,
            "status": "REGISTERED"
        }

    def run_pending(self):
        now = time.time()
        for name, job in self._jobs.items():
            if now - job["last_run"] >= job["interval"]:
                try:
                    job["status"] = "RUNNING"
                    job["func"](*job["args"], **job["kwargs"])
                    job["last_run"] = time.time()
                    job["status"] = "SUCCESS"
                except Exception as e:
                    job["status"] = f"ERROR: {e}"

    def start_background(self):
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            self.run_pending()
            time.sleep(10.0)

scheduler = IngestionScheduler()
