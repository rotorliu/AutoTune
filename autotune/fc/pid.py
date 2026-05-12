from __future__ import annotations
from typing import Optional
import struct
import copy
import json


class PIDAxis:
    def __init__(self, p: float = 0.0, i: float = 0.0, d: float = 0.0):
        self.p = p
        self.i = i
        self.d = d

    def __repr__(self):
        return f"PIDAxis(P={self.p}, I={self.i}, D={self.d})"

    def to_dict(self) -> dict:
        return {"P": self.p, "I": self.i, "D": self.d}

    @classmethod
    def from_dict(cls, data: dict) -> PIDAxis:
        return cls(p=data.get("P", 0.0), i=data.get("I", 0.0), d=data.get("D", 0.0))

    def clone(self) -> PIDAxis:
        return PIDAxis(p=self.p, i=self.i, d=self.d)


class PIDAdvancedAxis:
    def __init__(
        self,
        p_gain: float = 0.0,
        i_gain: float = 0.0,
        d_gain: float = 0.0,
        ff_gain: float = 0.0,
        d_min: float = 0.0,
        d_gain_boost: float = 0.0,
        d_min_gain: float = 0.0,
        d_min_advance: float = 0.0,
    ):
        self.p_gain = p_gain
        self.i_gain = i_gain
        self.d_gain = d_gain
        self.ff_gain = ff_gain
        self.d_min = d_min
        self.d_gain_boost = d_gain_boost
        self.d_min_gain = d_min_gain
        self.d_min_advance = d_min_advance

    def __repr__(self):
        return (
            f"PIDAdvancedAxis(P={self.p_gain}, I={self.i_gain}, D={self.d_gain}, "
            f"FF={self.ff_gain}, D_Min={self.d_min})"
        )

    def to_dict(self) -> dict:
        return {
            "P": self.p_gain,
            "I": self.i_gain,
            "D": self.d_gain,
            "FF": self.ff_gain,
            "D_Min": self.d_min,
            "D_Gain_Boost": self.d_gain_boost,
            "D_Min_Gain": self.d_min_gain,
            "D_Min_Advance": self.d_min_advance,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PIDAdvancedAxis:
        return cls(
            p_gain=data.get("P", 0.0),
            i_gain=data.get("I", 0.0),
            d_gain=data.get("D", 0.0),
            ff_gain=data.get("FF", 0.0),
            d_min=data.get("D_Min", 0.0),
            d_gain_boost=data.get("D_Gain_Boost", 0.0),
            d_min_gain=data.get("D_Min_Gain", 0.0),
            d_min_advance=data.get("D_Min_Advance", 0.0),
        )

    def clone(self) -> PIDAdvancedAxis:
        return PIDAdvancedAxis(
            p_gain=self.p_gain,
            i_gain=self.i_gain,
            d_gain=self.d_gain,
            ff_gain=self.ff_gain,
            d_min=self.d_min,
            d_gain_boost=self.d_gain_boost,
            d_min_gain=self.d_min_gain,
            d_min_advance=self.d_min_advance,
        )


class PIDProfile:
    AXIS_NAMES = ("Roll", "Pitch", "Yaw")

    def __init__(self):
        self.profile_index: int = 0
        self.profile_name: str = ""
        self.roll_pid = PIDAxis()
        self.pitch_pid = PIDAxis()
        self.yaw_pid = PIDAxis()
        self.roll_pid_adv = PIDAdvancedAxis()
        self.pitch_pid_adv = PIDAdvancedAxis()
        self.yaw_pid_adv = PIDAdvancedAxis()
        self.use_advanced: bool = False

    def get_axis(self, index: int) -> PIDAxis:
        return [self.roll_pid, self.pitch_pid, self.yaw_pid][index]

    def get_axis_advanced(self, index: int) -> PIDAdvancedAxis:
        return [self.roll_pid_adv, self.pitch_pid_adv, self.yaw_pid_adv][index]

    @staticmethod
    def _parse_legacy_pid(payload: bytes) -> dict:
        items = struct.unpack("<9B", payload[:9])
        return {
            "roll": PIDAxis(p=items[0], i=items[1], d=items[2]),
            "pitch": PIDAxis(p=items[3], i=items[4], d=items[5]),
            "yaw": PIDAxis(p=items[6], i=items[7], d=items[8]),
        }

    @classmethod
    def from_legacy_payload(cls, payload: bytes) -> PIDProfile:
        profile = cls()
        data = cls._parse_legacy_pid(payload)
        profile.roll_pid = data["roll"]
        profile.pitch_pid = data["pitch"]
        profile.yaw_pid = data["yaw"]
        profile.use_advanced = False
        return profile

    def to_dict(self) -> dict:
        return {
            "profile_index": self.profile_index,
            "profile_name": self.profile_name,
            "use_advanced": self.use_advanced,
            "Roll": self.roll_pid.to_dict(),
            "Pitch": self.pitch_pid.to_dict(),
            "Yaw": self.yaw_pid.to_dict(),
            "Roll_Advanced": self.roll_pid_adv.to_dict(),
            "Pitch_Advanced": self.pitch_pid_adv.to_dict(),
            "Yaw_Advanced": self.yaw_pid_adv.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> PIDProfile:
        profile = cls()
        profile.profile_index = data.get("profile_index", 0)
        profile.profile_name = data.get("profile_name", "")
        profile.use_advanced = data.get("use_advanced", False)
        profile.roll_pid = PIDAxis.from_dict(data.get("Roll", {}))
        profile.pitch_pid = PIDAxis.from_dict(data.get("Pitch", {}))
        profile.yaw_pid = PIDAxis.from_dict(data.get("Yaw", {}))
        profile.roll_pid_adv = PIDAdvancedAxis.from_dict(data.get("Roll_Advanced", {}))
        profile.pitch_pid_adv = PIDAdvancedAxis.from_dict(data.get("Pitch_Advanced", {}))
        profile.yaw_pid_adv = PIDAdvancedAxis.from_dict(data.get("Yaw_Advanced", {}))
        return profile

    def clone(self) -> PIDProfile:
        return PIDProfile.from_dict(self.to_dict())

    def diff(self, other: PIDProfile) -> dict:
        changes = {}
        self_dict = self.to_dict()
        other_dict = other.to_dict()
        for axis in self.AXIS_NAMES:
            for key in ("P", "I", "D"):
                v1 = self_dict[axis].get(key, 0.0)
                v2 = other_dict[axis].get(key, 0.0)
                if abs(v1 - v2) > 0.001:
                    changes[f"{axis}_{key}"] = {"from": v1, "to": v2}
        return changes