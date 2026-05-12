import json
import os
import logging
from datetime import datetime
from typing import List, Optional, Dict
from dataclasses import dataclass, asdict

from autotune.fc.pid import PIDProfile
from autotune.fc.rate import RateProfile

logger = logging.getLogger(__name__)

HISTORY_DIR = "tuning_history"
MAX_HISTORY_ENTRIES = 100


@dataclass
class TuningEntry:
    timestamp: str
    pid_before: dict
    pid_after: Optional[dict]
    rate_before: dict
    rate_after: Optional[dict]
    quality_score: Optional[float] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TuningEntry":
        return cls(**data)


class TuningHistory:
    def __init__(self):
        self._entries: List[TuningEntry] = []
        self._load_history()

    def _get_history_path(self) -> str:
        os.makedirs(HISTORY_DIR, exist_ok=True)
        return os.path.join(HISTORY_DIR, "history.json")

    def _load_history(self):
        path = self._get_history_path()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._entries = [TuningEntry.from_dict(item) for item in data]
            except Exception as e:
                logger.error(f"Failed to load tuning history: {e}")
                self._entries = []

    def _save_history(self):
        path = self._get_history_path()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump([e.to_dict() for e in self._entries], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save tuning history: {e}")

    def add_entry(
        self,
        pid_before: PIDProfile,
        pid_after: Optional[PIDProfile] = None,
        rate_before: Optional[RateProfile] = None,
        rate_after: Optional[RateProfile] = None,
        quality_score: Optional[float] = None,
        notes: str = "",
    ):
        entry = TuningEntry(
            timestamp=datetime.now().isoformat(),
            pid_before=pid_before.to_dict(),
            pid_after=pid_after.to_dict() if pid_after else None,
            rate_before=rate_before.to_dict() if rate_before else {},
            rate_after=rate_after.to_dict() if rate_after else None,
            quality_score=quality_score,
            notes=notes,
        )
        self._entries.insert(0, entry)
        if len(self._entries) > MAX_HISTORY_ENTRIES:
            self._entries = self._entries[:MAX_HISTORY_ENTRIES]
        self._save_history()
        logger.info(f"Tuning history entry added: {entry.timestamp}")

    def get_entries(self) -> List[TuningEntry]:
        return list(self._entries)

    def get_entry(self, index: int) -> Optional[TuningEntry]:
        if 0 <= index < len(self._entries):
            return self._entries[index]
        return None

    def clear_history(self):
        self._entries = []
        self._save_history()
        logger.info("Tuning history cleared")

    def get_stats(self) -> Dict[str, int]:
        return {
            "total_entries": len(self._entries),
            "has_pid_tuning": sum(1 for e in self._entries if e.pid_after is not None),
            "has_rate_tuning": sum(1 for e in self._entries if e.rate_after is not None),
        }
