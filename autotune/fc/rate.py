from __future__ import annotations
import struct
import copy


class RateAxis:
    def __init__(
        self,
        rc_rate: float = 1.0,
        super_rate: float = 0.7,
        rc_expo: float = 0.0,
    ):
        self.rc_rate = rc_rate
        self.super_rate = super_rate
        self.rc_expo = rc_expo

    def __repr__(self):
        return (
            f"RateAxis(RC_Rate={self.rc_rate}, "
            f"Super_Rate={self.super_rate}, RC_Expo={self.rc_expo})"
        )

    def compute_angular_rate(self, rc_input: float) -> float:
        abs_rc = abs(rc_input)
        if abs_rc > 1.0:
            abs_rc = 1.0

        expo_factor = 1.0 - self.rc_expo * (1.0 - abs_rc)
        rate = abs_rc * self.rc_rate * (1.0 + self.super_rate * abs_rc ** 2)
        rate *= expo_factor
        return rate * (1.0 if rc_input >= 0 else -1.0)

    def compute_max_rate(self) -> float:
        return self.compute_angular_rate(1.0)

    def to_dict(self) -> dict:
        return {
            "RC_Rate": self.rc_rate,
            "Super_Rate": self.super_rate,
            "RC_Expo": self.rc_expo,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RateAxis:
        return cls(
            rc_rate=data.get("RC_Rate", 1.0),
            super_rate=data.get("Super_Rate", 0.7),
            rc_expo=data.get("RC_Expo", 0.0),
        )

    def clone(self) -> RateAxis:
        return RateAxis(
            rc_rate=self.rc_rate,
            super_rate=self.super_rate,
            rc_expo=self.rc_expo,
        )


class RateProfile:
    AXIS_NAMES = ("Roll", "Pitch", "Yaw")

    def __init__(self):
        self.profile_index: int = 0
        self.profile_name: str = ""
        self.roll = RateAxis()
        self.pitch = RateAxis()
        self.yaw = RateAxis()
        self.throttle_rc_rate: float = 1.0
        self.throttle_rc_expo: float = 0.0
        self.tpa_rate: float = 0.0
        self.tpa_breakpoint: int = 1500

    def get_axis(self, index: int) -> RateAxis:
        return [self.roll, self.pitch, self.yaw][index]

    def to_dict(self) -> dict:
        return {
            "profile_index": self.profile_index,
            "profile_name": self.profile_name,
            "Roll": self.roll.to_dict(),
            "Pitch": self.pitch.to_dict(),
            "Yaw": self.yaw.to_dict(),
            "Throttle_RC_Rate": self.throttle_rc_rate,
            "Throttle_RC_Expo": self.throttle_rc_expo,
            "TPA_Rate": self.tpa_rate,
            "TPA_Breakpoint": self.tpa_breakpoint,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RateProfile:
        profile = cls()
        profile.profile_index = data.get("profile_index", 0)
        profile.profile_name = data.get("profile_name", "")
        profile.roll = RateAxis.from_dict(data.get("Roll", {}))
        profile.pitch = RateAxis.from_dict(data.get("Pitch", {}))
        profile.yaw = RateAxis.from_dict(data.get("Yaw", {}))
        profile.throttle_rc_rate = data.get("Throttle_RC_Rate", 1.0)
        profile.throttle_rc_expo = data.get("Throttle_RC_Expo", 0.0)
        profile.tpa_rate = data.get("TPA_Rate", 0.0)
        profile.tpa_breakpoint = data.get("TPA_Breakpoint", 1500)
        return profile

    def clone(self) -> RateProfile:
        return RateProfile.from_dict(self.to_dict())

    @staticmethod
    def _parse_rc_tuning(payload: bytes) -> dict:
        items = struct.unpack("<BBB", payload[:3])
        return {
            "rc_rate": items[0] / 100.0,
            "rc_expo": items[1] / 100.0,
            "roll_pitch_rate": items[2],
        }

    @classmethod
    def from_legacy_payload(cls, payload: bytes) -> RateProfile:
        data = cls._parse_rc_tuning(payload)
        roll = RateAxis(rc_rate=data["rc_rate"], super_rate=data["roll_pitch_rate"] / 100.0, rc_expo=data["rc_expo"])
        return cls(
            roll=roll,
            pitch=roll.clone(),
            yaw=RateAxis(rc_rate=data["rc_rate"], super_rate=data.get("yaw_rate", 0.0), rc_expo=data["rc_expo"]),
        )