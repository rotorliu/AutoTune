import os
import time
import json
import logging
import csv
import numpy as np
from datetime import datetime
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class FlightRecorder:
    def __init__(self, output_dir: str = "recordings"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self._recording = False
        self._data: list[dict] = []
        self._start_time = 0.0
        self._metadata: dict = {}
        self._filepath: str = ""

        self._on_start_callback: Optional[Callable] = None
        self._on_stop_callback: Optional[Callable] = None
        self._on_sample_callback: Optional[Callable] = None

    def set_callbacks(self, on_start=None, on_stop=None, on_sample=None):
        self._on_start_callback = on_start
        self._on_stop_callback = on_stop
        self._on_sample_callback = on_sample

    def set_metadata(self, metadata: dict):
        self._metadata.update(metadata)

    def start(self, label: str = "flight"):
        if self._recording:
            return

        self._recording = True
        self._start_time = time.time()
        self._data = []
        self._filepath = ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = label.replace(" ", "_").replace("/", "_")
        self._filepath = os.path.join(
            self.output_dir,
            f"recording_{safe_label}_{timestamp}",
        )

        self._metadata.update({
            "label": label,
            "timestamp": timestamp,
            "start_time_iso": datetime.now().isoformat(),
        })

        logger.info(f"Recording started: {label}")

        if self._on_start_callback:
            self._on_start_callback()

    def stop(self) -> dict:
        if not self._recording:
            return {}

        self._recording = False
        elapsed = time.time() - self._start_time
        self._metadata["elapsed_time"] = elapsed
        self._metadata["sample_count"] = len(self._data)
        self._metadata["stop_time_iso"] = datetime.now().isoformat()

        result = {
            "data": self._data,
            "metadata": self._metadata,
            "elapsed_time": elapsed,
            "sample_count": len(self._data),
        }

        logger.info(
            f"Recording stopped: {self._metadata.get('label')} - "
            f"{len(self._data)} samples in {elapsed:.2f}s"
        )

        if self._on_stop_callback:
            self._on_stop_callback(result)

        return result

    def record_sample(self, data: dict):
        if not self._recording:
            return

        sample = dict(data)
        sample["record_index"] = len(self._data)
        sample["record_time"] = time.time() - self._start_time
        self._data.append(sample)

        if self._on_sample_callback:
            self._on_sample_callback(sample)

    def is_recording(self) -> bool:
        return self._recording

    def save_csv(self, filepath: str = None) -> str:
        if filepath is None:
            filepath = self._filepath + ".csv"

        if not self._data:
            logger.warning("No data to save")
            return ""

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        columns = ["record_index", "record_time"]
        if self._data:
            columns.extend(k for k in self._data[0].keys() if k not in columns)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._data)

        logger.info(f"Recording saved as CSV: {filepath}")
        return filepath

    def save_metadata(self, filepath: str = None) -> str:
        if filepath is None:
            filepath = self._filepath + "_meta.json"

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        meta_copy = dict(self._metadata)
        for key, value in meta_copy.items():
            if isinstance(value, (np.ndarray,)):
                meta_copy[key] = f"<ndarray: {value.shape}>"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(meta_copy, f, indent=2, ensure_ascii=False)

        return filepath

    def save_all(self) -> dict[str, str]:
        result = {}
        if self._data:
            result["csv"] = self.save_csv()
        result["meta"] = self.save_metadata()
        return result

    def get_data_as_array(self) -> np.ndarray:
        if not self._data:
            return np.array([])

        fields = ["record_index", "record_time"]
        for key in self._data[0].keys():
            if key not in fields and isinstance(self._data[0][key], (int, float)):
                fields.append(key)
            elif key not in fields and isinstance(self._data[0][key], (np.integer, np.floating)):
                fields.append(key)

        dtype = np.dtype([
            (name, np.float64) for name in fields
            if name in self._data[0]
        ])

        valid_fields = [name for name in fields if name in self._data[0]]
        dtype = np.dtype([(name, np.float64) for name in valid_fields])

        arr = np.zeros(len(self._data), dtype=dtype)
        for i, item in enumerate(self._data):
            for name in valid_fields:
                try:
                    arr[name][i] = float(item.get(name, 0.0))
                except (ValueError, TypeError):
                    arr[name][i] = 0.0

        return arr