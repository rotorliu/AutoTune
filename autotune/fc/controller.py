from __future__ import annotations
import json
import os
import logging
import struct
from datetime import datetime
from typing import Optional

from autotune.msp.transport import MSPTransport
from autotune.msp.commands import MSPCommand
from autotune.fc.pid import PIDProfile, PIDAxis, PIDAdvancedAxis
from autotune.fc.rate import RateProfile, RateAxis
from autotune.fc.config import FCConfig, FCInfo, FilterConfig

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

            try:
                profile_payload = self.transport.send_command(MSPCommand.MSP_PID_PROFILE)
                if len(profile_payload) >= 1:
                    profile.profile_index = profile_payload[0]
            except Exception:
                pass
        except Exception:
            profile = PIDProfile()

        self._pid_profile = profile
        return profile

    def write_pid_profile(self, profile: PIDProfile = None):
        if profile is None:
            profile = self.pid_profile

        pid_bytes = bytearray()
        for axis in (profile.roll_pid, profile.pitch_pid, profile.yaw_pid):
            pid_bytes.append(max(0, min(255, round(axis.p))))
            pid_bytes.append(max(0, min(255, round(axis.i))))
            pid_bytes.append(max(0, min(255, round(axis.d))))

        self.transport.send_command(MSPCommand.MSP_SET_PID, bytes(pid_bytes))

        if profile.use_advanced:
            self.write_pid_profile_advanced(profile)

        self._pid_profile = profile
        logger.info("PID profile written to flight controller")

    def read_pid_profile_advanced(self) -> PIDProfile:
        try:
            payload = self.transport.send_command(MSPCommand.MSP_PID_ADVANCED)
            profile = self.pid_profile.clone()

            if len(payload) >= 24:
                items = struct.unpack("<BBBBBBBBBBBBBBBBBBBBBBBB", payload[:24])
                for ax_idx, offset in enumerate([0, 8, 16]):
                    axis_adv = profile.get_axis_advanced(ax_idx)
                    axis_adv.ff_gain = items[offset + 3]
                    axis_adv.d_min = items[offset + 4]
                    axis_adv.d_min_gain = items[offset + 5]
                    axis_adv.d_min_advance = items[offset + 6]
                    axis_adv.d_gain_boost = items[offset + 7]

                profile.use_advanced = True
                self._pid_profile = profile
                logger.info("Advanced PID profile read from flight controller")
            elif len(payload) >= 12:
                items = struct.unpack("<BBBBBBBBBBBB", payload[:12])
                for ax_idx, offset in enumerate([0, 4, 8]):
                    axis_adv = profile.get_axis_advanced(ax_idx)
                    axis_adv.ff_gain = items[offset + 3]
                profile.use_advanced = True
                self._pid_profile = profile
                logger.info("Advanced PID (basic) read from flight controller")

            return profile
        except Exception as e:
            logger.debug(f"Failed to read advanced PID: {e}")
            return self.pid_profile

    def write_pid_profile_advanced(self, profile: PIDProfile = None):
        if profile is None:
            profile = self.pid_profile

        adv_bytes = bytearray()
        for ax_idx in range(3):
            axis = profile.get_axis(ax_idx)
            adv_bytes.append(max(0, min(255, round(axis.p))))
            adv_bytes.append(max(0, min(255, round(axis.i))))
            adv_bytes.append(max(0, min(255, round(axis.d))))

            axis_adv = profile.get_axis_advanced(ax_idx)
            adv_bytes.append(max(0, min(255, round(axis_adv.ff_gain))))
            adv_bytes.append(max(0, min(255, round(axis_adv.d_min))))
            adv_bytes.append(max(0, min(255, round(axis_adv.d_min_gain))))
            adv_bytes.append(max(0, min(255, round(axis_adv.d_min_advance))))
            adv_bytes.append(max(0, min(255, round(axis_adv.d_gain_boost))))

        self.transport.send_command(MSPCommand.MSP_SET_PID_ADVANCED, bytes(adv_bytes))
        logger.info("Advanced PID profile written to flight controller")

    def write_rate_profile(self, profile: RateProfile = None):
        if profile is None:
            profile = self.rate_profile

        # Try modern per-axis MSP_SET_RATE_PROFILE (Betaflight 4.2+)
        # Format: 16 bytes — throttle(3) + tpa(4) + yaw_thr(1) + roll(3) + pitch(3) + yaw(3)
        try:
            existing = self._read_rate_profile_full()
        except Exception:
            existing = None

        if existing is not None:
            throttle_rc_rate = max(1, min(255, round(existing.throttle_rc_rate * 100.0)))
            throttle_rc_expo = max(0, min(255, round(existing.throttle_rc_expo * 100.0)))
            tpa_rate = max(0, min(1000, round(existing.tpa_rate * 100.0)))
            tpa_breakpoint = existing.tpa_breakpoint
            yaw_rc_rate_thr = max(1, min(255, round(existing.throttle_rc_rate * 100.0)))
        else:
            throttle_rc_rate = 100
            throttle_rc_expo = 0
            tpa_rate = 0
            tpa_breakpoint = 1500
            yaw_rc_rate_thr = 100

        buf = bytearray()
        buf.append(throttle_rc_rate)
        buf.append(throttle_rc_expo)
        buf.extend(struct.pack("<H", tpa_rate))
        buf.extend(struct.pack("<H", tpa_breakpoint))
        buf.append(yaw_rc_rate_thr)

        for axis in (profile.roll, profile.pitch, profile.yaw):
            buf.append(max(1, min(255, round(axis.rc_rate * 100.0))))
            buf.append(max(0, min(255, round(axis.super_rate * 100.0))))
            buf.append(max(0, min(255, round(axis.rc_expo * 100.0))))

        try:
            self.transport.send_command(MSPCommand.MSP_SET_RATE_PROFILE, bytes(buf))
            self._rate_profile = profile
            logger.info("Rate profile written to flight controller (per-axis)")
            return
        except Exception as e:
            logger.warning(f"MSP_SET_RATE_PROFILE failed, falling back to legacy: {e}")

        # Fallback: legacy MSP_SET_RC_TUNING (applies same values to Roll+Pitch, Yaw unchanged)
        rc_tuning_bytes = bytearray()
        rc_tuning_bytes.append(max(1, min(255, round(profile.roll.rc_rate * 100.0))))
        rc_tuning_bytes.append(max(0, min(255, round(profile.roll.rc_expo * 100.0))))
        rc_tuning_bytes.append(max(0, min(255, round(profile.roll.super_rate * 100.0))))

        self.transport.send_command(MSPCommand.MSP_SET_RC_TUNING, bytes(rc_tuning_bytes))
        self._rate_profile = profile
        logger.info("Rate profile written to flight controller (legacy)")

    def _read_rate_profile_full(self) -> Optional[RateProfile]:
        """Read full per-axis rate profile via MSP_RATE_PROFILE (Betaflight 4.2+)."""
        payload = self.transport.send_command(MSPCommand.MSP_RATE_PROFILE)
        if len(payload) < 16:
            return None

        # Format: B(thr_rc) B(thr_expo) H(dynPID) H(breakpoint) B(thr_yaw)
        #          B(roll_rc) B(roll_sr) B(roll_expo)
        #          B(pitch_rc) B(pitch_sr) B(pitch_expo)
        #          B(yaw_rc) B(yaw_sr) B(yaw_expo) = 14 items, 16 bytes
        items = struct.unpack("<BBHHBBBBBBBBBB", payload[:16])
        profile = RateProfile()
        profile.throttle_rc_rate = items[0] / 100.0
        profile.throttle_rc_expo = items[1] / 100.0
        profile.tpa_rate = items[2] / 100.0
        profile.tpa_breakpoint = items[3]

        profile.roll = RateAxis(
            rc_rate=items[5] / 100.0,
            super_rate=items[6] / 100.0,
            rc_expo=items[7] / 100.0,
        )
        profile.pitch = RateAxis(
            rc_rate=items[8] / 100.0,
            super_rate=items[9] / 100.0,
            rc_expo=items[10] / 100.0,
        )
        profile.yaw = RateAxis(
            rc_rate=items[11] / 100.0,
            super_rate=items[12] / 100.0,
            rc_expo=items[13] / 100.0,
        )

        return profile

    def read_rate_profile(self) -> RateProfile:
        try:
            full = self._read_rate_profile_full()
            if full is not None:
                self._rate_profile = full
                return full
        except Exception:
            pass

        # Legacy fallback: MSP_RC_TUNING (only 3 bytes, Roll+Pitch shared)
        try:
            payload = self.transport.send_command(MSPCommand.MSP_RC_TUNING)
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

    def read_filter_config(self) -> FilterConfig:
        try:
            payload = self.transport.send_command(MSPCommand.MSP_FILTER_CONFIG)
            if len(payload) >= 25:
                items = struct.unpack("<BHBBHBHBBHBHHBH", payload[:25])
                fc = FilterConfig()
                fc.gyro_lowpass_hz = items[0]
                fc.gyro_lowpass2_hz = items[4]
                fc.gyro_lowpass_type = items[3]
                fc.gyro_notch_hz = items[6]
                fc.gyro_notch_cutoff = items[7]
                fc.dterm_lowpass_hz = items[8]
                fc.dterm_lowpass2_hz = items[12]
                fc.dterm_lowpass_type = items[11]
                fc.dterm_notch_hz = items[14]
                fc.dterm_notch_cutoff = items[15]
                fc.yaw_lowpass_hz = items[16]
            elif len(payload) >= 17:
                items = struct.unpack("<BHBBHBHBBHBHB", payload[:17])
                fc = FilterConfig()
                fc.gyro_lowpass_hz = items[0]
                fc.gyro_lowpass2_hz = items[4]
                fc.gyro_lowpass_type = items[3]
                fc.gyro_notch_hz = items[6]
                fc.gyro_notch_cutoff = items[7]
                fc.dterm_lowpass_hz = items[8]
                fc.dterm_lowpass2_hz = items[12]
                fc.dterm_lowpass_type = items[11]
                fc.dterm_notch_hz = items[14]
                fc.dterm_notch_cutoff = items[15]
                fc.yaw_lowpass_hz = items[16] if len(items) > 16 else 0.0
            else:
                fc = FilterConfig()

            if self._config is None:
                self._config = FCConfig()
            self._config.filter_config = fc
            logger.info("Filter config read from flight controller")
            return fc
        except Exception:
            return FilterConfig()

    def write_filter_config(self, config: FilterConfig = None):
        if config is None:
            config = self.config.filter_config

        buf = bytearray()
        buf.append(max(0, min(255, round(config.gyro_lowpass_hz))))
        buf.extend(struct.pack("<H", 0))
        buf.extend(struct.pack("<H", 0))
        buf.append(max(0, min(255, config.gyro_lowpass_type)))
        buf.append(max(0, min(255, round(config.gyro_lowpass2_hz))))
        buf.extend(struct.pack("<H", 0))
        buf.append(max(0, min(255, round(config.gyro_notch_hz))))
        buf.extend(struct.pack("<H", round(config.gyro_notch_cutoff)))
        buf.append(max(0, min(255, round(config.dterm_lowpass_hz))))
        buf.extend(struct.pack("<H", 0))
        buf.extend(struct.pack("<H", 0))
        buf.append(max(0, min(255, config.dterm_lowpass_type)))
        buf.append(max(0, min(255, round(config.dterm_lowpass2_hz))))
        buf.extend(struct.pack("<H", 0))
        buf.append(max(0, min(255, round(config.dterm_notch_hz))))
        buf.extend(struct.pack("<H", round(config.dterm_notch_cutoff)))
        buf.append(max(0, min(255, round(config.yaw_lowpass_hz))))

        self.transport.send_command(MSPCommand.MSP_SET_FILTER_CONFIG, bytes(buf))
        if self._config is None:
            self._config = FCConfig()
        self._config.filter_config = config
        logger.info("Filter config written to flight controller")

    def read_all(self) -> dict:
        self.identify()
        pid = self.read_pid_profile()
        rate = self.read_rate_profile()
        filt = self.read_filter_config()
        return {
            "pid": pid.to_dict(),
            "rate": rate.to_dict(),
            "filter": filt.to_dict(),
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
            "filter": self.config.filter_config.to_dict(),
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

        filter_data = config_data.get("filter", {})
        if filter_data:
            fc = FilterConfig()
            fc.gyro_lowpass_hz = filter_data.get("gyro_lowpass_hz", fc.gyro_lowpass_hz)
            fc.gyro_lowpass2_hz = filter_data.get("gyro_lowpass2_hz", fc.gyro_lowpass2_hz)
            fc.dterm_lowpass_hz = filter_data.get("dterm_lowpass_hz", fc.dterm_lowpass_hz)
            fc.dterm_lowpass2_hz = filter_data.get("dterm_lowpass2_hz", fc.dterm_lowpass2_hz)
            fc.gyro_notch_hz = filter_data.get("gyro_notch_hz", fc.gyro_notch_hz)
            fc.gyro_notch_cutoff = filter_data.get("gyro_notch_cutoff", fc.gyro_notch_cutoff)
            fc.dterm_notch_hz = filter_data.get("dterm_notch_hz", fc.dterm_notch_hz)
            fc.dterm_notch_cutoff = filter_data.get("dterm_notch_cutoff", fc.dterm_notch_cutoff)
            fc.yaw_lowpass_hz = filter_data.get("yaw_lowpass_hz", fc.yaw_lowpass_hz)
            self.write_filter_config(fc)

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