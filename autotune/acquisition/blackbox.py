import struct
import io
import logging
import numpy as np
from typing import Optional
from enum import IntEnum

logger = logging.getLogger(__name__)


class BlackboxFrameType(IntEnum):
    INTRA = 0
    INTER = 1
    SLOW = 2
    EVENT = 3


class BlackboxEncoding(IntEnum):
    """Betaflight 4.5 field encoding types."""
    SIGNED_VB = 0
    UNSIGNED_VB = 1
    NULL_TERMINATED_STRING = 2
    TAG2_3S32 = 3
    TAG8_4S16 = 4
    TAG2_3SVB = 5
    TAG8_8SVB = 6
    NEG_14BIT = 7
    FLOAT = 8
    FLOAT_VB = 9


class BlackboxPredictor(IntEnum):
    NONE = 0
    MIN = 1
    MAX = 2
    STRIDE_0 = 3
    STRIDE_1 = 4
    STRIDE_2 = 5
    STRIDE_3 = 6
    STRIDE_4 = 7
    STRIDE_5 = 8
    STRIDE_6 = 9
    STRIDE_7 = 10
    STRIDE_8 = 11
    STRIDE_9 = 12
    STRIDE_10 = 13
    STRIDE_11 = 14
    STRIDE_12 = 15
    LAST_2 = 16
    MIN_3 = 17
    MAX_3 = 18
    POLYNOMIAL_2 = 19
    POLYNOMIAL_3 = 20
    AUTO_0 = 21
    AUTO_1 = 22
    AUTO_2 = 23
    AUTO_3 = 24
    AUTO_4 = 25
    AUTO_5 = 26
    AUTO_6 = 27
    AUTO_7 = 28
    AUTO_8 = 29
    AUTO_9 = 30
    AUTO_10 = 31


class BlackboxParser:
    def __init__(self):
        self._reset()

    def _reset(self):
        self._header_parsed = False
        self._field_names: list[str] = []
        self._field_signed: list[bool] = []
        self._field_predictors: list[int] = []
        self._field_p_predictors: list[int] = []
        self._field_encodings: list[list[int]] = []
        self._sample_rate: int = 1000
        self._data: dict[str, list[float]] = {}
        self._timestamps: list[float] = []
        self._data_start_pos = 0
        self._data_end_pos = 0
        self._running_min: list[float] = []
        self._running_max: list[float] = []
        self._i_interval: int = 128
        self._p_interval: int = 4
        self._firmware_type: str = ""
        self._firmware_version: str = ""

    def parse_file(self, filepath: str) -> dict[str, np.ndarray]:
        with open(filepath, "rb") as f:
            raw_data = f.read()
        return self.parse_bytes(raw_data)

    def parse_bytes(self, data: bytes) -> dict[str, np.ndarray]:
        self._reset()
        self._parse_header(data)
        
        if not self._header_parsed:
            logger.warning("Failed to parse Blackbox header")
            return {}

        try:
            self._parse_frames(data)
        except Exception as e:
            logger.warning(f"Error parsing frames: {e}")

        return self._build_result()

    def _parse_header(self, data: bytes):
        headers = {}
        self._data_start_pos = 0
        self._data_end_pos = len(data)

        # Collect all header line positions
        header_lines = []
        i = 0
        while i < len(data):
            if data[i:i+1] == b'H':
                colon_pos = data.find(b':', i, i + 200)
                if colon_pos > i:
                    newline_pos = data.find(b'\n', i)
                    if newline_pos == -1:
                        newline_pos = len(data)
                    header_lines.append((i, newline_pos))
                    i = newline_pos + 1
                    continue
            i += 1

        if not header_lines:
            return

        file_len = len(data)

        # Group headers: contiguous headers near the beginning of the file
        # are the real headers. Headers later in the file are embedded data.
        # Find the initial contiguous block of headers at the start.
        begin_headers = {}
        end_headers = {}
        begin_last_end = 0

        # Find the first contiguous block: headers at the beginning, separated
        # from the rest of the file by a gap (non-header data)
        prev_end = -1
        block_end = 0
        for pos, end in header_lines:
            if prev_end < 0:
                # First header
                block_end = end
            elif pos <= prev_end + 100:
                # Contiguous (within 100 bytes of previous header end)
                block_end = end
            else:
                # Gap detected — this marks the end of the initial header block
                break
            prev_end = end

        # Parse headers in the initial contiguous block
        for pos, end in header_lines:
            if pos > block_end + 100:
                break
            line_bytes = data[pos:end]
            try:
                line = line_bytes.decode('ascii', errors='replace')
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    if key.startswith('H '):
                        key = key[2:].strip()
                    begin_headers[key] = value
            except Exception:
                pass

        # Parse headers at the end of file (last 5%)
        for pos, end in header_lines:
            if pos > file_len * 0.95:
                line_bytes = data[pos:end]
                try:
                    line = line_bytes.decode('ascii', errors='replace')
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        if key.startswith('H '):
                            key = key[2:].strip()
                        end_headers[key] = value
                except Exception:
                    pass

        # Data starts right after the initial header block
        self._data_start_pos = block_end + 1

        logger.info(f"Begin headers: {len(begin_headers)}, end headers: {len(end_headers)}")
        logger.info(f"Data range: [{self._data_start_pos}, {self._data_end_pos})")

        if not begin_headers:
            return

        self._firmware_type = begin_headers.get('Firmware type', '')
        self._firmware_version = begin_headers.get('Firmware revision', '')

        if 'I interval' in begin_headers:
            try:
                self._i_interval = int(begin_headers['I interval'])
                self._sample_rate = 1000000 // self._i_interval
                logger.info(f"Sampling rate: {self._sample_rate} Hz")
            except ValueError:
                pass

        if 'Field I name' in begin_headers:
            self._field_names = begin_headers['Field I name'].split(',')
            self._field_signed = [bool(int(x)) for x in begin_headers.get('Field I signed', '').split(',')]
            self._field_predictors = [int(x) for x in begin_headers.get('Field I predictor', '').split(',')]

            # P-frame predictor is separate from I-frame predictor
            self._field_p_predictors = [
                int(x) for x in begin_headers.get('Field P predictor', '').split(',')
            ] if 'Field P predictor' in begin_headers else self._field_predictors

            encodings = begin_headers.get('Field I encoding', '').split(',')
            self._field_encodings = [[int(x)] for x in encodings]

            if 'Field P encoding' in begin_headers:
                p_encodings = begin_headers['Field P encoding'].split(',')
                for i, enc in enumerate(p_encodings):
                    if i < len(self._field_encodings):
                        self._field_encodings[i].append(int(enc))

            self._data = {name: [] for name in self._field_names}
            self._running_min = [float('inf')] * len(self._field_names)
            self._running_max = [float('-inf')] * len(self._field_names)
            self._header_parsed = True

    def _parse_frames(self, data: bytes):
        """Parse I-frames only by scanning for 'I' (0x49) frame markers.
        
        P-frames are skipped because they use complex TAG encodings that
        require group-aware parsing. I-frames provide full-resolution data
        at I_interval/P_interval rate (e.g., 61 Hz at 7812 Hz with I=128/P=4),
        which is sufficient for PID and Rate tuning analysis.
        
        False I-frame markers (0x49 bytes in encoded data) are filtered by
        checking that loopIteration increments by exactly I_interval from the
        previous valid I-frame.
        """
        buf = io.BytesIO(data)
        file_len = len(data)

        buf.seek(self._data_start_pos)
        logger.info(
            f"I-frame-only parsing: start={self._data_start_pos}, "
            f"I_interval={self._i_interval}, P_interval={self._p_interval}"
        )

        predictors = [0] * len(self._field_names)
        frame_count = 0
        i_frame_count = 0
        pos = self._data_start_pos
        expected_loop_iteration = 0

        scan_window = 4096
        loop_iter_idx = self._field_names.index('loopIteration') if 'loopIteration' in self._field_names else 0

        while pos < file_len:
            marker_pos = data.find(b'I', pos, min(pos + scan_window, file_len))
            if marker_pos < 0:
                logger.info(f"No more I-frame markers found after position {pos}")
                break

            if marker_pos > file_len * 0.9:
                peek = data[marker_pos:marker_pos + 40]
                if peek.startswith(b'H ') or peek.startswith(b'H\t'):
                    logger.info(f"Reached end headers region at position {marker_pos}")
                    break

            buf.seek(marker_pos + 1)

            try:
                parsed_loop, frame_values = self._parse_single_intra_frame(buf)
                if parsed_loop is not None and parsed_loop == expected_loop_iteration:
                    for i, value in enumerate(frame_values):
                        self._data[self._field_names[i]].append(value)
                        predictors[i] = value
                        if value < self._running_min[i]:
                            self._running_min[i] = value
                        if value > self._running_max[i]:
                            self._running_max[i] = value
                    self._timestamps.append(len(self._timestamps))
                    frame_count += 1
                    i_frame_count += 1
                    expected_loop_iteration += self._i_interval
                    pos = buf.tell()
                    if i_frame_count % 1000 == 0:
                        logger.info(f"Parsed {i_frame_count} I-frames "
                                   f"(loopIteration={parsed_loop})...")
                else:
                    pos = marker_pos + 1
            except Exception:
                pos = marker_pos + 1

        logger.info(f"Parsed {i_frame_count} I-frames ({frame_count} total frames)")

    def _parse_single_intra_frame(self, buf: io.BytesIO):
        """Parse a single I-frame without committing to instance data.
        
        Returns (loopIteration, list_of_values) on success, (None, []) on failure.
        """
        values = []
        loop_iteration = None

        for i in range(len(self._field_names)):
            encoding = self._field_encodings[i][0]
            value = self._read_encoded_value(buf, encoding, self._field_signed[i])

            if encoding in (0, 1) and abs(value) > 1000000000:
                return None, []

            if i == 0:
                loop_iteration = value

            values.append(value)

        return loop_iteration, values

    def _try_parse_intra_frame(self, buf: io.BytesIO, predictors: list) -> bool:
        """Try to parse an I-frame. Returns True if successful, False if data looks invalid."""
        for i in range(len(self._field_names)):
            encoding = self._field_encodings[i][0]
            value = self._read_encoded_value(buf, encoding, self._field_signed[i])

            # Validate: reject clearly bogus values
            if encoding == 0 or encoding == 1:
                # SIGNED_VB / UNSIGNED_VB: values should be within reasonable range
                if abs(value) > 1000000000:
                    return False

            predictors[i] = value
            self._data[self._field_names[i]].append(value)
            if value < self._running_min[i]:
                self._running_min[i] = value
            if value > self._running_max[i]:
                self._running_max[i] = value
        self._timestamps.append(len(self._timestamps))
        return True

    def _parse_inter_frame(self, buf: io.BytesIO, predictors: list):
        buf.read(1)  # Skip 'P' frame marker byte
        for i in range(len(self._field_names)):
            # Use P-frame encoding if available, otherwise fall back to I-frame encoding
            if len(self._field_encodings[i]) > 1:
                encoding = self._field_encodings[i][1]
            else:
                encoding = self._field_encodings[i][0]

            predictor = self._field_p_predictors[i] if i < len(self._field_p_predictors) else 0

            delta = self._read_encoded_value(buf, encoding, self._field_signed[i])
            predicted = self._predict_value(i, predictor, predictors)
            value = predicted + delta

            predictors[i] = value
            self._data[self._field_names[i]].append(value)
            if value < self._running_min[i]:
                self._running_min[i] = value
            if value > self._running_max[i]:
                self._running_max[i] = value
        self._timestamps.append(len(self._timestamps))

    def _parse_slow_frame(self, buf: io.BytesIO, predictors: list):
        pass

    def _parse_event_frame(self, buf: io.BytesIO):
        pass

    def _read_encoded_value(self, buf: io.BytesIO, encoding: int, signed: bool) -> int:
        try:
            if encoding == BlackboxEncoding.SIGNED_VB:
                return self._read_signed_vb(buf)
            elif encoding == BlackboxEncoding.UNSIGNED_VB:
                return self._read_unsigned_vb(buf)
            elif encoding == BlackboxEncoding.NULL_TERMINATED_STRING:
                result = bytearray()
                while True:
                    b = buf.read(1)
                    if not b or b == b'\x00':
                        break
                    result.extend(b)
                return 0
            elif encoding == BlackboxEncoding.TAG2_3S32:
                return self._read_tag2_3s32(buf)
            elif encoding == BlackboxEncoding.TAG8_4S16:
                return self._read_tag8_4s16(buf)
            elif encoding == BlackboxEncoding.TAG2_3SVB:
                return self._read_tag2_3svb(buf)
            elif encoding == BlackboxEncoding.TAG8_8SVB:
                return self._read_tag8_8svb(buf)
            elif encoding == BlackboxEncoding.NEG_14BIT:
                b = buf.read(2)
                if len(b) < 2:
                    return 0
                val = struct.unpack('<h', b)[0]
                return val
            elif encoding == BlackboxEncoding.FLOAT:
                b = buf.read(4)
                if len(b) < 4:
                    return 0
                return int(struct.unpack('<f', b)[0])
            elif encoding == BlackboxEncoding.FLOAT_VB:
                raw = self._read_signed_vb(buf)
                return int(raw / 1000.0)
            else:
                return 0
        except Exception:
            return 0

    def _read_tag2_3s32(self, buf: io.BytesIO) -> int:
        """TAG2_3S32: 2-bit count header (0-3) followed by 0-3 int32 values."""
        tag = buf.read(1)
        if len(tag) < 1:
            return 0
        count = (tag[0] >> 6) & 0x03
        for j in range(count):
            b = buf.read(4)
            if len(b) < 4:
                return 0
        return 0

    def _read_tag8_4s16(self, buf: io.BytesIO) -> int:
        """TAG8_4S16: 8-bit count header followed by 0-255 int16 values."""
        tag = buf.read(1)
        if len(tag) < 1:
            return 0
        count = tag[0]
        for j in range(count):
            b = buf.read(2)
            if len(b) < 2:
                return 0
        return 0

    def _read_tag2_3svb(self, buf: io.BytesIO) -> int:
        """TAG2_3SVB: 2-bit count header (0-3) followed by 0-3 signed VB values."""
        tag = buf.read(1)
        if len(tag) < 1:
            return 0
        count = (tag[0] >> 6) & 0x03
        for j in range(count):
            self._read_signed_vb(buf)
        return 0

    def _read_tag8_8svb(self, buf: io.BytesIO) -> int:
        """TAG8_8SVB: 8-bit count header followed by 0-255 signed VB values."""
        tag = buf.read(1)
        if len(tag) < 1:
            return 0
        count = tag[0]
        for j in range(count):
            self._read_signed_vb(buf)
        return 0

    def _read_unsigned_vb(self, buf: io.BytesIO) -> int:
        result = 0
        shift = 0
        while True:
            b = buf.read(1)
            if not b:
                break
            byte = b[0]
            result |= (byte & 0x7F) << shift
            if not (byte & 0x80):
                break
            shift += 7
        return result

    def _read_signed_vb(self, buf: io.BytesIO) -> int:
        result = self._read_unsigned_vb(buf)
        if result & 1:
            return -(result >> 1)
        else:
            return result >> 1

    def _read_unsigned_16bits_vb(self, buf: io.BytesIO) -> int:
        result = 0
        b = buf.read(1)
        if not b:
            return 0
        result = b[0]
        if result & 0x80:
            result = (result & 0x7F) << 8
            b = buf.read(1)
            if b:
                result |= b[0]
        return result

    def _predict_value(self, field_idx: int, predictor: int, predictors: list) -> int:
        history = self._data[self._field_names[field_idx]]
        history_len = len(history)

        if predictor == BlackboxPredictor.NONE:
            return 0
        elif predictor == BlackboxPredictor.MIN:
            return int(self._running_min[field_idx]) if self._running_min[field_idx] != float('inf') else 0
        elif predictor == BlackboxPredictor.MAX:
            return int(self._running_max[field_idx]) if self._running_max[field_idx] != float('-inf') else 0
        elif predictor == BlackboxPredictor.LAST_2:
            return predictors[field_idx] if history_len > 0 else 0
        elif BlackboxPredictor.STRIDE_0 <= predictor <= BlackboxPredictor.STRIDE_12:
            stride = predictor - BlackboxPredictor.STRIDE_0
            idx = history_len - 1 - stride
            if idx >= 0:
                return history[idx]
            return predictors[field_idx] if history_len > 0 else 0
        elif predictor == BlackboxPredictor.MIN_THROTTLE:
            return int(self._running_min[field_idx]) if self._running_min[field_idx] != float('inf') else 0
        elif predictor == BlackboxPredictor.AVERAGE_3:
            if history_len >= 3:
                try:
                    return int(sum(history[-3:]) / 3)
                except (ValueError, ZeroDivisionError):
                    pass
            return predictors[field_idx] if history_len > 0 else 0
        elif predictor == BlackboxPredictor.INC:
            if history_len >= 1:
                prev = history[-1]
                if history_len >= 2:
                    delta = history[-1] - history[-2]
                    return prev + delta
                return prev
            return 0
        elif predictor == BlackboxPredictor.HOME:
            if history_len > 1:
                return history[0]
            return predictors[field_idx] if history_len > 0 else 0
        elif predictor == BlackboxPredictor.FIVE_SAMPLE:
            if history_len >= 5:
                return history[-5]
            return predictors[field_idx] if history_len > 0 else 0
        elif predictor == BlackboxPredictor.ABOVE_MEAN:
            if history_len >= 3:
                mean = sum(history) / history_len
                above = [v for v in history if v > mean]
                if above:
                    return sum(above) / len(above)
            return predictors[field_idx] if history_len > 0 else 0
        elif predictor == BlackboxPredictor.MIN_3:
            if history_len >= 3:
                return min(history[-3:])
            return min(history) if history else 0
        elif predictor == BlackboxPredictor.MAX_3:
            if history_len >= 3:
                return max(history[-3:])
            return max(history) if history else 0
        elif predictor == BlackboxPredictor.POLYNOMIAL_2:
            if history_len >= 3:
                try:
                    x = np.array([-2, -1, 0])
                    y = np.array(history[-3:])
                    coeffs = np.polyfit(x, y, 2)
                    predicted = np.polyval(coeffs, 1)
                    return int(predicted) if np.isfinite(predicted) else predictors[field_idx]
                except Exception:
                    pass
            return predictors[field_idx] if history_len > 0 else 0
        elif predictor == BlackboxPredictor.POLYNOMIAL_3:
            if history_len >= 4:
                try:
                    x = np.array([-3, -2, -1, 0])
                    y = np.array(history[-4:])
                    coeffs = np.polyfit(x, y, 3)
                    predicted = np.polyval(coeffs, 1)
                    return int(predicted) if np.isfinite(predicted) else predictors[field_idx]
                except Exception:
                    pass
            return predictors[field_idx] if history_len > 0 else 0
        else:
            return predictors[field_idx] if history_len > 0 else 0

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
        
        logger.info(f"Built result with {sample_count} samples")
        return result

    def extract_channels(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        result = {}

        # Gyro - convert gyroADC to deg/s
        # In Betaflight, typical scaling factor for gyro is usually around 16.4 LSB/(deg/s) for 2000 deg/s
        gyro_scale = 1.0 / 16.4
        gyro_map = {
            "gyroADC[0]": "gyro_x",
            "gyroADC[1]": "gyro_y",
            "gyroADC[2]": "gyro_z",
        }

        for bb_name, alias in gyro_map.items():
            if bb_name in data:
                result[alias] = data[bb_name] * gyro_scale

        # Motors
        motor_map = {
            "motor[0]": "motor_0",
            "motor[1]": "motor_1",
            "motor[2]": "motor_2",
            "motor[3]": "motor_3",
        }

        for bb_name, alias in motor_map.items():
            if bb_name in data:
                result[alias] = data[bb_name]

        # RC commands - rcCommand is typically in 1000-2000 range, convert to -500 to 0 to 500 or similar
        rc_map = {
            "rcCommand[0]": "rc_roll",
            "rcCommand[1]": "rc_pitch",
            "rcCommand[2]": "rc_yaw",
            "rcCommand[3]": "rc_throttle",
        }

        for bb_name, alias in rc_map.items():
            if bb_name in data:
                # Convert from ~1000-2000 to -500 to 500 centered around 1500
                result[alias] = data[bb_name]

        # Setpoint for PID tuning
        setpoint_map = {
            "setpoint[0]": "setpoint_x",
            "setpoint[1]": "setpoint_y",
            "setpoint[2]": "setpoint_z",
        }
        for bb_name, alias in setpoint_map.items():
            if bb_name in data:
                # Convert setpoint to deg/s as well
                result[alias] = data[bb_name] * gyro_scale

        # PID values
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
