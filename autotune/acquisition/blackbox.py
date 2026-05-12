import struct
import io
import logging
import numpy as np
from typing import Optional
from enum import IntEnum

logger = logging.getLogger(__name__)

BB_HEADER = b"H\x00\x00\x00\x00\x00\x00"

FIELD_DEFS = {
    "gyroADC[0]": {"signed": True, "predictor": 0, "encoding": 0},
    "gyroADC[1]": {"signed": True, "predictor": 1, "encoding": 0},
    "gyroADC[2]": {"signed": True, "predictor": 2, "encoding": 0},
    "rcCommand[0]": {"signed": True, "predictor": 0, "encoding": 0},
    "rcCommand[1]": {"signed": True, "predictor": 1, "encoding": 0},
    "rcCommand[2]": {"signed": True, "predictor": 2, "encoding": 0},
    "rcCommand[3]": {"signed": True, "predictor": 3, "encoding": 0},
    "axisP[0]": {"signed": True, "predictor": 0, "encoding": 0},
    "axisP[1]": {"signed": True, "predictor": 1, "encoding": 0},
    "axisP[2]": {"signed": True, "predictor": 2, "encoding": 0},
    "motor[0]": {"signed": False, "predictor": 0, "encoding": 0},
    "motor[1]": {"signed": False, "predictor": 1, "encoding": 0},
    "motor[2]": {"signed": False, "predictor": 2, "encoding": 0},
    "motor[3]": {"signed": False, "predictor": 3, "encoding": 0},
}


class BlackboxFrameType(IntEnum):
    INTRA = 0
    INTER = 1
    SLOW = 2
    EVENT = 3


class BlackboxParser:
    def __init__(self):
        self._header_parsed = False
        self._field_names: list[str] = []
        self._field_indices: dict[str, int] = {}
        self._predictors: list[int] = []
        self._sample_rate: int = 1000
        self._data: dict[str, list[float]] = {}
        self._timestamps: list[float] = []

    def parse_file(self, filepath: str) -> dict[str, np.ndarray]:
        with open(filepath, "rb") as f:
            raw_data = f.read()

        return self.parse_bytes(raw_data)

    def parse_bytes(self, data: bytes) -> dict[str, np.ndarray]:
        self._reset()
        buf = io.BytesIO(data)

        while buf.tell() < len(data):
            try:
                header = buf.read(1)
                if not header:
                    break

                if header[0] == ord('H'):
                    self._parse_header(data, buf.tell() - 1)
                    break
            except Exception:
                break

        if not self._header_parsed:
            return self._attempt_structured_parse(data)

        self._parse_frames(data)
        return self._build_result()

    def _reset(self):
        self._header_parsed = False
        self._field_names = []
        self._field_indices = {}
        self._predictors = []
        self._data = {}
        self._timestamps = []

    def _parse_header(self, data: bytes, offset: int):
        buf = io.BytesIO(data[offset:])

        signature = buf.read(4)
        if signature != b"HProduct":
            return

        while True:
            char = buf.read(1)
            if not char:
                break
            if char == b'H':
                break
            if char == b'\x00':
                continue

        firmware_line = self._read_null_terminated(buf)
        if not firmware_line:
            return

        while True:
            char = buf.read(1)
            if not char:
                break
            if char[0] >= ord('H'):
                break

        has_log_start = False
        while True:
            pos = buf.tell()
            if pos + 1 >= len(data):
                break
            char = buf.read(1)
            if char[0] == ord('H'):
                pos2 = buf.tell()
                if pos2 < len(data):
                    next_byte = data[pos2:pos2 + 1]
                    if next_byte == b'\x00' or (next_byte and next_byte[0] & 0x80):
                        self._header_parsed = True
                        self._field_names = self._detect_fields(data)

                        for i, name in enumerate(self._field_names):
                            self._field_indices[name] = i
                            self._data[name] = []
                            if name in FIELD_DEFS:
                                self._predictors.append(FIELD_DEFS[name]["predictor"])
                            else:
                                self._predictors.append(i)

                        self._timestamps = []
                        self._data["time"] = []
                        break
            elif char[0] in (ord('P'), ord('I'), ord('E'), ord('S')):
                pass
            else:
                break

        self._header_parsed = True

    @staticmethod
    def _read_null_terminated(buf: io.BytesIO) -> str:
        result = bytearray()
        while True:
            char = buf.read(1)
            if not char or char == b'\x00':
                break
            try:
                result.extend(char)
            except Exception:
                break
        return result.decode("utf-8", errors="ignore")

    def _detect_fields(self, data: bytes) -> list[str]:
        return ["gyroADC[0]", "gyroADC[1]", "gyroADC[2]",
                "rcCommand[0]", "rcCommand[1]", "rcCommand[2]", "rcCommand[3]",
                "axisP[0]", "axisP[1]", "axisP[2]",
                "motor[0]", "motor[1]", "motor[2]", "motor[3]"]

    def _parse_frames(self, data: bytes):
        buf = io.BytesIO(data)
        buf.seek(0)

        while buf.tell() < len(data):
            try:
                header = buf.read(1)
                if not header:
                    break

                frame_type = header[0] & 0x03
                timestamp = (header[0] >> 2) & 0x1F

                if frame_type == BlackboxFrameType.INTRA:
                    frame_size = len(self._field_names) * 2
                    frame_data = buf.read(frame_size)
                    if len(frame_data) < frame_size:
                        break

                    values = struct.unpack(f'<{len(self._field_names)}h', frame_data)
                    for i, name in enumerate(self._field_names):
                        self._data[name].append(values[i])
                    self._timestamps.append(timestamp)
                elif frame_type == BlackboxFrameType.INTER:
                    frame_size = len(self._field_names)
                    frame_data = buf.read(frame_size)
                    if len(frame_data) < frame_size:
                        break

                    for i, name in enumerate(self._field_names):
                        pred_idx = self._predictors[i]
                        pred_val = self._data[name][-1] if self._data[name] else 0
                        delta = frame_data[i] if i < len(frame_data) else 0
                        if FIELD_DEFS.get(name, {}).get("signed", True):
                            if delta > 127:
                                delta -= 256
                        self._data[name].append(pred_val + delta)
                    self._timestamps.append(timestamp)
                else:
                    continue
            except Exception:
                break

    def _attempt_structured_parse(self, data: bytes) -> dict[str, np.ndarray]:
        try:
            lines = data.decode("utf-8", errors="ignore").strip().split("\n")
            result = {}

            for line in lines[:1]:
                if line.startswith("H"):
                    continue
                if "," in line:
                    parts = line.strip().split(",")
                    try:
                        values = [float(p) for p in parts]
                        for i, v in enumerate(values):
                            key = f"col_{i}"
                            if key not in result:
                                result[key] = []
                            result[key].append(v)
                    except ValueError:
                        continue

            if result:
                for key in list(result.keys()):
                    result[key] = np.array(result[key], dtype=np.float64)

                gyro_cols = []
                motor_cols = []
                rc_cols = []
                pid_cols = []

                sorted_keys = sorted(result.keys())
                for key in sorted_keys:
                    if len(gyro_cols) < 3:
                        arr_key = f"gyro_{['x','y','z'][len(gyro_cols)]}"
                        result[arr_key] = result.pop(key)
                        gyro_cols.append(arr_key)
                    elif len(motor_cols) < 4:
                        arr_key = f"motor_{len(motor_cols)}"
                        result[arr_key] = result.pop(key)
                        motor_cols.append(arr_key)
                    elif len(rc_cols) < 4:
                        rc_names = ['roll', 'pitch', 'yaw', 'throttle']
                        arr_key = f"rc_{rc_names[len(rc_cols)]}"
                        result[arr_key] = result.pop(key)
                        rc_cols.append(arr_key)
                    elif len(pid_cols) < 3:
                        pid_names = ['roll', 'pitch', 'yaw']
                        arr_key = f"pid_p_{pid_names[len(pid_cols)]}"
                        result[arr_key] = result.pop(key)
                        pid_cols.append(arr_key)

                return result
            return {}
        except Exception:
            return {}

    def _build_result(self) -> dict[str, np.ndarray]:
        result = {}
        for name, values in self._data.items():
            if values:
                result[name] = np.array(values, dtype=np.float64)

        sample_count = 0
        for arr in result.values():
            sample_count = max(sample_count, len(arr))

        if sample_count > 0 and self._sample_rate > 0:
            result["time"] = np.arange(sample_count, dtype=np.float64) / self._sample_rate

        return result

    def extract_channels(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        result = {}

        gyro_map = {
            "gyroADC[0]": "gyro_x",
            "gyroADC[1]": "gyro_y",
            "gyroADC[2]": "gyro_z",
        }

        for bb_name, alias in gyro_map.items():
            if bb_name in data:
                result[alias] = data[bb_name]

        motor_map = {
            "motor[0]": "motor_0",
            "motor[1]": "motor_1",
            "motor[2]": "motor_2",
            "motor[3]": "motor_3",
        }

        for bb_name, alias in motor_map.items():
            if bb_name in data:
                result[alias] = data[bb_name]

        rc_map = {
            "rcCommand[0]": "rc_roll",
            "rcCommand[1]": "rc_pitch",
            "rcCommand[2]": "rc_yaw",
            "rcCommand[3]": "rc_throttle",
        }

        for bb_name, alias in rc_map.items():
            if bb_name in data:
                result[alias] = data[bb_name]

        pid_map = {
            "axisP[0]": "pid_p_roll",
            "axisP[1]": "pid_p_pitch",
            "axisP[2]": "pid_p_yaw",
        }

        for bb_name, alias in pid_map.items():
            if bb_name in data:
                result[alias] = data[bb_name]

        if "time" in data:
            result["time"] = data["time"]

        return result