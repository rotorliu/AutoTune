from __future__ import annotations
from typing import Optional


class FilterConfig:
    def __init__(self):
        self.gyro_lowpass_hz: float = 200.0
        self.gyro_lowpass2_hz: float = 250.0
        self.gyro_lowpass_type: int = 0
        self.gyro_lowpass2_type: int = 0
        self.gyro_notch_hz: float = 0.0
        self.gyro_notch_cutoff: float = 0.0
        self.dterm_lowpass_hz: float = 100.0
        self.dterm_lowpass2_hz: float = 150.0
        self.dterm_lowpass_type: int = 0
        self.dterm_lowpass2_type: int = 0
        self.dterm_notch_hz: float = 0.0
        self.dterm_notch_cutoff: float = 0.0
        self.yaw_lowpass_hz: float = 0.0

    def to_dict(self) -> dict:
        return {
            "gyro_lowpass_hz": self.gyro_lowpass_hz,
            "gyro_lowpass2_hz": self.gyro_lowpass2_hz,
            "gyro_lowpass_type": self.gyro_lowpass_type,
            "gyro_lowpass2_type": self.gyro_lowpass2_type,
            "gyro_notch_hz": self.gyro_notch_hz,
            "gyro_notch_cutoff": self.gyro_notch_cutoff,
            "dterm_lowpass_hz": self.dterm_lowpass_hz,
            "dterm_lowpass2_hz": self.dterm_lowpass2_hz,
            "dterm_lowpass_type": self.dterm_lowpass_type,
            "dterm_lowpass2_type": self.dterm_lowpass2_type,
            "dterm_notch_hz": self.dterm_notch_hz,
            "dterm_notch_cutoff": self.dterm_notch_cutoff,
            "yaw_lowpass_hz": self.yaw_lowpass_hz,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FilterConfig:
        config = cls()
        config.gyro_lowpass_hz = data.get("gyro_lowpass_hz", config.gyro_lowpass_hz)
        config.gyro_lowpass2_hz = data.get("gyro_lowpass2_hz", config.gyro_lowpass2_hz)
        config.gyro_lowpass_type = data.get("gyro_lowpass_type", config.gyro_lowpass_type)
        config.gyro_lowpass2_type = data.get("gyro_lowpass2_type", config.gyro_lowpass2_type)
        config.gyro_notch_hz = data.get("gyro_notch_hz", config.gyro_notch_hz)
        config.gyro_notch_cutoff = data.get("gyro_notch_cutoff", config.gyro_notch_cutoff)
        config.dterm_lowpass_hz = data.get("dterm_lowpass_hz", config.dterm_lowpass_hz)
        config.dterm_lowpass2_hz = data.get("dterm_lowpass2_hz", config.dterm_lowpass2_hz)
        config.dterm_lowpass_type = data.get("dterm_lowpass_type", config.dterm_lowpass_type)
        config.dterm_lowpass2_type = data.get("dterm_lowpass2_type", config.dterm_lowpass2_type)
        config.dterm_notch_hz = data.get("dterm_notch_hz", config.dterm_notch_hz)
        config.dterm_notch_cutoff = data.get("dterm_notch_cutoff", config.dterm_notch_cutoff)
        config.yaw_lowpass_hz = data.get("yaw_lowpass_hz", config.yaw_lowpass_hz)
        return config


class FCInfo:
    def __init__(self):
        self.identifier: str = ""
        self.version: str = ""
        self.target: str = ""
        self.build_date: str = ""
        self.build_time: str = ""

    def to_dict(self) -> dict:
        return {
            "identifier": self.identifier,
            "version": self.version,
            "target": self.target,
            "build_date": self.build_date,
            "build_time": self.build_time,
        }


class FCConfig:
    def __init__(self):
        self.fc_info = FCInfo()
        self.filter_config = FilterConfig()
        self.esc_protocol: int = 0
        self.mixer_type: int = 0
        self.voltage_meter_type: int = 0
        self.current_meter_type: int = 0

    def to_dict(self) -> dict:
        return {
            "fc_info": self.fc_info.to_dict(),
            "filter_config": self.filter_config.to_dict(),
            "esc_protocol": self.esc_protocol,
        }