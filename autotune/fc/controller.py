from __future__ import annotations
import json
import os
import logging
from datetime import datetime
from typing import Optional

from autotune.msp.transport import MSPTransport
from autotune.msp.commands import MSPCommand
from autotune.fc.pid import PIDProfile, PIDAxis, PIDAdvancedAxis
from autotune.fc.rate import RateProfile, RateAxis
from autotune.fc.config import FCConfig, FCInfo

logger = logging.getLogger(__name__)


class FCController:
    def __init__(self, transport: MSPTransport):
        self.transport = transport
        self._pid_profile: Optional[PIDProfile] = None
        self._rate_profile: Optional[RateProfile] = None
        self._config: Optional[FCConfig] = None
        self._backup_dir = "backups"

    @property
    def is_connected(self) -> bool:
        return self.transport.is_connected

    @property
    def pid_profile(self) -> PIDProfile:
        if self._pid_profile is None:
            self._pid_profile = PIDProfile()
        return self._pid_profile

    @property
    def rate_profile(self) -> RateProfile:
        if self._rate_profile is None:
            self._rate_profile = RateProfile()
        return self._rate_profile

    @property
    def config(self) -> FCConfig:
        if self._config is None:
            self._config = FCConfig()
        return self._config

    def identify(self) -> FCInfo:
        info = FCInfo()
        try:
            info.identifier = self.transport.read_fc_variant()
        except Exception:
            info.identifier = "Betaflight"

        try:
            version_data = self.transport.read_fc_version()
            info.version = version_data.get("version", "")
        except Exception:
            pass

        try:
            build_data = self.transport.read_build_info()
            info.build_date = build_data.get("build_date", "")
            info.build_time = build_data.get("build_time", "")
            info.target = build_data.get("target", "")
        except Exception:
            pass

        self._config = FCConfig()
        self._config.fc_info = info
        logger.info(f"Identified FC: {info.identifier}, version: {info.version}")
        return info

    def read_pid_profile(self) -> PIDProfile:
        try:
            payload = self.transport.send_command(MSPCommand.MSP_PID)
            profile = PIDProfile.from_legacy_payload(payload)

            status_payload = self.transport.send_command(MSPCommand.MSP_STATUS)
            if len(status_payload) >= 5:
                import struct
                profile.profile_index = struct.unpack("<B", status_payload[8:9])[0] if len(status_payload) > 8 else 0
        except Exception:
            profile = PIDProfile()

        self._pid_profile = profile
        return profile

    def write_pid_profile(self, profile: PIDProfile = None):
        if profile is None:
            profile = self.pid_profile

        pid_bytes = bytearray()
        for axis in (profile.roll_pid, profile.pitch_pid, profile.yaw_pid):
            pid_bytes.append(int(axis.p))
            pid_bytes.append(int(axis.i))
            pid_bytes.append(int(axis.d))

        self.transport.send_command(MSPCommand.MSP_SET_PID, bytes(pid_bytes))
        self._pid_profile = profile
        logger.info("PID profile written to flight controller")

    def read_rate_profile(self) -> RateProfile:
        try:
            payload = self.transport.send_command(MSPCommand.MSP_RC_TUNING)
            import struct
            items = struct.unpack("<BBB", payload[:3])
            rate = RateProfile()
            rc_rate = items[0] / 100.0
            rc_expo = items[1] / 100.0
            roll_pitch_rate = items[2] / 100.0

            rate.roll = RateAxis(rc_rate=rc_rate, super_rate=roll_pitch_rate, rc_expo=rc_expo)
            rate.pitch = RateAxis(rc_rate=rc_rate, super_rate=roll_pitch_rate, rc_expo=rc_expo)
            rate.yaw = RateAxis(rc_rate=rc_rate, super_rate=roll_pitch_rate, rc_expo=rc_expo)
        except Exception:
            rate = RateProfile()

        self._rate_profile = rate
        return rate

    def write_rate_profile(self, profile: RateProfile = None):
        if profile is None:
            profile = self.rate_profile

        rc_tuning_bytes = bytearray()
        rc_tuning_bytes.append(int(profile.roll.rc_rate * 100.0))
        rc_tuning_bytes.append(int(profile.roll.rc_expo * 100.0))
        rc_tuning_bytes.append(int(profile.roll.super_rate * 100.0))

        self.transport.send_command(MSPCommand.MSP_SET_RC_TUNING, bytes(rc_tuning_bytes))
        self._rate_profile = profile
        logger.info("Rate profile written to flight controller")

    def read_all(self) -> dict:
        self.identify()
        pid = self.read_pid_profile()
        rate = self.read_rate_profile()
        return {
            "pid": pid.to_dict(),
            "rate": rate.to_dict(),
            "config": self.config.to_dict(),
        }

    def backup_config(self, backup_dir: str = None) -> str:
        if backup_dir is None:
            backup_dir = self._backup_dir

        os.makedirs(backup_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backup_{self.config.fc_info.identifier}_{timestamp}.json"
        filepath = os.path.join(backup_dir, filename)

        config_data = {
            "timestamp": timestamp,
            "fc_info": self.config.fc_info.to_dict(),
            "pid": self.pid_profile.to_dict(),
            "rate": self.rate_profile.to_dict(),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Configuration backed up to {filepath}")
        return filepath

    def restore_config(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        pid = PIDProfile.from_dict(config_data.get("pid", {}))
        rate = RateProfile.from_dict(config_data.get("rate", {}))

        self.write_pid_profile(pid)
        self.write_rate_profile(rate)

        logger.info(f"Configuration restored from {filepath}")

    def read_telemetry_snapshot(self) -> dict:
        imu = self.transport.read_imu()
        attitude = self.transport.read_attitude()
        motors = self.transport.read_motors()
        status = self.transport.read_status()

        return {
            "gyro_x": imu.get("gyro_x", 0),
            "gyro_y": imu.get("gyro_y", 0),
            "gyro_z": imu.get("gyro_z", 0),
            "acc_x": imu.get("acc_x", 0),
            "acc_y": imu.get("acc_y", 0),
            "acc_z": imu.get("acc_z", 0),
            "roll": attitude.get("roll", 0.0),
            "pitch": attitude.get("pitch", 0.0),
            "yaw": attitude.get("yaw", 0.0),
            "motor_0": motors[0] if len(motors) > 0 else 0,
            "motor_1": motors[1] if len(motors) > 1 else 0,
            "motor_2": motors[2] if len(motors) > 2 else 0,
            "motor_3": motors[3] if len(motors) > 3 else 0,
            "cycle_time": status.get("cycle_time", 0),
            "system_load": status.get("system_load", 0),
        }