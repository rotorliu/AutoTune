import time
import logging
from typing import Optional, List

import serial
import serial.tools.list_ports

from autotune.msp.commands import MSPCommand, MSPVersion
from autotune.msp.protocol import (
    MSPV1Protocol,
    MSPV2Protocol,
    create_protocol,
    MSPProtocol,
)

logger = logging.getLogger(__name__)

DEFAULT_BAUDRATE = 115200
DEFAULT_TIMEOUT = 2.0
READ_BUFFER_SIZE = 4096
MAX_RETRIES = 3
RETRY_DELAY = 0.1

BETAFLIGHT_VID_PID_PAIRS = [
    (0x0483, 0x5740),
    (0x1915, 0x521F),
    (0x1FC9, 0x0036),
    (0x1FC9, 0x0037),
    (0x239A, None),
    (0x2E8A, None),
    (0x303A, None),
    (0x1209, None),
]


class MSPTransport:
    def __init__(self, version: MSPVersion = MSPVersion.V1):
        self.protocol: MSPProtocol = create_protocol(version)
        self.version = version
        self.serial: Optional[serial.Serial] = None
        self._buffer = bytearray()
        self._connected = False
        self._port_name = ""

    @property
    def is_connected(self) -> bool:
        return self._connected and self.serial is not None and self.serial.is_open

    @property
    def port_name(self) -> str:
        return self._port_name

    @staticmethod
    def list_ports() -> List[dict]:
        ports = []
        available = serial.tools.list_ports.comports()
        for port in available:
            ports.append({
                "device": port.device,
                "description": port.description,
                "hardware_id": port.hwid,
                "vid": port.vid,
                "pid": port.pid,
                "serial_number": port.serial_number,
                "manufacturer": port.manufacturer,
                "product": port.product,
            })
        return ports

    @staticmethod
    def filter_betaflight_ports(ports: List[dict] = None) -> List[dict]:
        if ports is None:
            ports = MSPTransport.list_ports()

        bf_ports = []
        for port in ports:
            if port.get("description", "").upper() in ("STM32 VIRTUAL COM PORT", "CP210X", "CP2102"):
                bf_ports.append(port)
                continue

            vid = port.get("vid")
            pid = port.get("pid")
            if vid is not None and pid is not None:
                for bf_vid, bf_pid in BETAFLIGHT_VID_PID_PAIRS:
                    if vid == bf_vid and (bf_pid is None or pid == bf_pid):
                        bf_ports.append(port)
                        break

        for port in ports:
            if port not in bf_ports:
                bf_ports.append(port)

        return bf_ports

    def connect(
        self,
        port: str,
        baudrate: int = DEFAULT_BAUDRATE,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> bool:
        if self.is_connected:
            self.disconnect()

        try:
            self.serial = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=timeout,
                write_timeout=timeout,
            )
            self._port_name = port
            self._connected = True
            self._buffer = bytearray()
            time.sleep(0.5)
            logger.info(f"Connected to {port} at {baudrate} baud")
            return True
        except serial.SerialException as e:
            logger.error(f"Failed to connect to {port}: {e}")
            self._connected = False
            self.serial = None
            return False

    def disconnect(self):
        if self.serial and self.serial.is_open:
            self.serial.close()
        self._connected = False
        self.serial = None
        self._buffer = bytearray()
        logger.info("Disconnected from flight controller")

    def send_command(
        self,
        command: MSPCommand,
        payload: bytes = b"",
        retries: int = MAX_RETRIES,
    ) -> bytes:
        message = self.protocol.pack(int(command), payload)
        return self._send_raw(message, retries)

    def _send_raw(self, data: bytes, retries: int = MAX_RETRIES) -> bytes:
        if not self.is_connected:
            raise ConnectionError("Not connected to flight controller")

        last_error = None
        for attempt in range(retries + 1):
            try:
                self.serial.reset_input_buffer()
                self.serial.write(data)
                self.serial.flush()

                response = bytearray()
                start_time = time.time()

                while (time.time() - start_time) < DEFAULT_TIMEOUT:
                    waiting = self.serial.in_waiting
                    if waiting > 0:
                        chunk = self.serial.read(min(waiting, READ_BUFFER_SIZE))
                        response.extend(chunk)

                    result = self.protocol.unpack_header(bytes(response))
                    if result is not None:
                        remaining = response[result["total_length"]:]
                        self._buffer = bytearray(remaining)
                        return result["payload"]

                    time.sleep(0.001)

                raise TimeoutError("No response from flight controller")

            except (serial.SerialException, TimeoutError, OSError) as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1}/{retries + 1} failed: {e}")
                if attempt < retries:
                    time.sleep(RETRY_DELAY)
                    try:
                        self.serial.reset_input_buffer()
                        self.serial.reset_output_buffer()
                    except Exception:
                        pass

        raise TimeoutError(f"Command failed after {retries + 1} attempts: {last_error}")

    def read_status(self) -> dict:
        payload = self.send_command(MSPCommand.MSP_STATUS)
        return MSPTransport._parse_status(payload)

    def read_imu(self) -> dict:
        payload = self.send_command(MSPCommand.MSP_RAW_IMU)
        return MSPTransport._parse_imu(payload)

    def read_motors(self) -> List[int]:
        payload = self.send_command(MSPCommand.MSP_MOTOR)
        if len(payload) >= 16:
            return list(struct.unpack('<8H', payload[:16]))
        return []

    def read_attitude(self) -> dict:
        payload = self.send_command(MSPCommand.MSP_ATTITUDE)
        return MSPTransport._parse_attitude(payload)

    def read_fc_variant(self) -> str:
        payload = self.send_command(MSPCommand.MSP_FC_VARIANT)
        return payload.decode("utf-8", errors="ignore").strip("\x00")

    def read_fc_version(self) -> dict:
        payload = self.send_command(MSPCommand.MSP_FC_VERSION)
        parts = payload.split(b"\x00")
        return {
            "version": parts[0].decode("utf-8", errors="ignore") if len(parts) > 0 else "",
            "build_info": parts[1].decode("utf-8", errors="ignore") if len(parts) > 1 else "",
        }

    def read_board_info(self) -> dict:
        payload = self.send_command(MSPCommand.MSP_BOARD_INFO)
        if len(payload) < 3:
            return {}
        return {
            "board_identifier": payload[:4].decode("utf-8", errors="ignore"),
            "hardware_revision": int.from_bytes(payload[4:6], "little", signed=False) if len(payload) > 5 else 0,
        }

    def read_build_info(self) -> dict:
        payload = self.send_command(MSPCommand.MSP_BUILD_INFO)
        parts = payload.split(b"\x00")
        return {
            "build_date": parts[0].decode("utf-8", errors="ignore") if len(parts) > 0 else "",
            "build_time": parts[1].decode("utf-8", errors="ignore") if len(parts) > 1 else "",
            "target": parts[2].decode("utf-8", errors="ignore") if len(parts) > 2 else "",
        }

    @staticmethod
    def _parse_status(payload: bytes) -> dict:
        import struct
        items = struct.unpack("<HHHIBBH", payload[:12])
        return {
            "cycle_time": items[0],
            "i2c_errors": items[1],
            "sensor_status": items[2],
            "flight_mode_flags": items[3],
            "pid_profile_index": items[4],
            "system_load": items[5],
            "gyro_cycle_time": items[6] if len(items) > 6 else 0,
        }

    @staticmethod
    def _parse_imu(payload: bytes) -> dict:
        import struct
        items = struct.unpack("<9h", payload[:18])
        return {
            "acc_x": items[0], "acc_y": items[1], "acc_z": items[2],
            "gyro_x": items[3], "gyro_y": items[4], "gyro_z": items[5],
            "mag_x": items[6], "mag_y": items[7], "mag_z": items[8],
        }

    @staticmethod
    def _parse_attitude(payload: bytes) -> dict:
        import struct
        items = struct.unpack("<3h", payload[:6])
        return {
            "roll": items[0] / 10.0,
            "pitch": items[1] / 10.0,
            "yaw": items[2],
        }