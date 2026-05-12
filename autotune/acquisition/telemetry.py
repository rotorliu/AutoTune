import time
import logging
import threading
from collections import deque
from typing import Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 1000
MAX_BUFFER_SIZE = 60000


class TelemetryCollector:
    def __init__(self, controller, sample_rate: int = DEFAULT_SAMPLE_RATE):
        self.controller = controller
        self.sample_rate = sample_rate
        self._interval = 1.0 / sample_rate

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._data_buffer: deque = deque(maxlen=MAX_BUFFER_SIZE)
        self._sample_count = 0
        self._start_time = 0.0

        self._on_data_callback: Optional[Callable] = None

    def set_callback(self, callback: Callable[[dict], None]):
        self._on_data_callback = callback

    def start(self):
        if self._running:
            return

        self._running = True
        self._start_time = time.time()
        self._sample_count = 0
        self._data_buffer.clear()
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()
        logger.info(f"Telemetry collection started at {self.sample_rate} Hz")

    def stop(self) -> list[dict]:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        with self._lock:
            data = list(self._data_buffer)

        logger.info(f"Telemetry collection stopped. Collected {len(data)} samples")
        return data

    def _collect_loop(self):
        while self._running:
            loop_start = time.perf_counter()
            try:
                if self.controller.is_connected:
                    data = self.controller.read_telemetry_snapshot()
                    data["sample_index"] = self._sample_count
                    data["timestamp"] = time.time() - self._start_time

                    with self._lock:
                        self._data_buffer.append(data)
                        self._sample_count += 1

                    if self._on_data_callback is not None:
                        try:
                            self._on_data_callback(data)
                        except Exception:
                            pass
            except Exception as e:
                logger.debug(f"Telemetry read error: {e}")

            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, self._interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_data_as_array(self) -> np.ndarray:
        with self._lock:
            data_list = list(self._data_buffer)

        if not data_list:
            return np.array([])

        fields = [
            "gyro_x", "gyro_y", "gyro_z",
            "acc_x", "acc_y", "acc_z",
            "motor_0", "motor_1", "motor_2", "motor_3",
            "roll", "pitch", "yaw",
            "cycle_time", "system_load",
            "sample_index", "timestamp",
        ]

        dtype = np.dtype([
            (name, np.float64) for name in fields
        ])

        arr = np.zeros(len(data_list), dtype=dtype)
        for i, item in enumerate(data_list):
            for name in fields:
                arr[name][i] = float(item.get(name, 0.0))

        return arr

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def elapsed_time(self) -> float:
        if self._running:
            return time.time() - self._start_time
        return 0.0

    @property
    def is_running(self) -> bool:
        return self._running