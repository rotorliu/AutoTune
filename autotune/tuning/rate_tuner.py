import logging
from typing import Optional, Callable
import numpy as np

from autotune.fc.rate import RateProfile, RateAxis

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 1000.0
MAX_RATE_CHANGE_PCT = 25.0


class RateTuner:
    AXES = ["Roll", "Pitch", "Yaw"]
    RC_KEYS = ["rc_roll", "rc_pitch", "rc_yaw"]
    GYRO_KEYS = ["gyro_x", "gyro_y", "gyro_z"]

    def __init__(
        self,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        conservative: bool = True,
    ):
        self.sample_rate = sample_rate
        self.conservative = conservative
        self._max_change_pct = 15.0 if conservative else MAX_RATE_CHANGE_PCT
        self._progress_callback: Optional[Callable] = None

    def set_progress_callback(self, callback: Callable):
        self._progress_callback = callback

    def _notify_progress(self, message: str, percent: int):
        if self._progress_callback:
            self._progress_callback(message, percent)

    def tune(
        self,
        data: dict[str, np.ndarray],
        initial_profile: RateProfile,
        target_max_rate: float = 720.0,
    ) -> RateProfile:
        tuned = initial_profile.clone()

        for axis_idx, (axis_name, rc_key, gyro_key) in enumerate(
            zip(self.AXES, self.RC_KEYS, self.GYRO_KEYS)
        ):
            rc = data.get(rc_key)
            gyro = data.get(gyro_key)

            if rc is None or gyro is None or len(rc) < 10:
                logger.warning(f"No data for {axis_name} rate tuning, skipping")
                continue

            self._notify_progress(f"正在分析 {axis_name} Rate 数据...", 30 + axis_idx * 20)

            rate_axis = tuned.get_axis(axis_idx)
            new_rate = self._tune_single_axis(rate_axis, rc, gyro, target_max_rate, axis_name)

            tuned_axis = tuned.get_axis(axis_idx)
            tuned_axis.rc_rate = new_rate.rc_rate
            tuned_axis.super_rate = new_rate.super_rate
            tuned_axis.rc_expo = new_rate.rc_expo

        throttle_rc = data.get("rc_throttle")
        if throttle_rc is not None and len(throttle_rc) > 10:
            self._notify_progress("正在分析 TPA 参数...", 85)
            self._tune_tpa(tuned, throttle_rc, data)

        return tuned

    def _tune_single_axis(
        self,
        current_rate: RateAxis,
        rc_input: np.ndarray,
        gyro_output: np.ndarray,
        target_max_rate: float,
        axis_name: str,
    ) -> RateAxis:
        new_rate = current_rate.clone()

        max_rc = np.max(np.abs(rc_input)) if len(rc_input) > 0 else 1.0
        if max_rc < 0.1:
            max_rc = 1.0

        normalized_rc = rc_input / max(max_rc, 0.01)

        max_gyro = np.max(np.abs(gyro_output))

        logger.info(
            f"[{axis_name}] Max RC: {max_rc:.2f}, "
            f"Max Gyro: {max_gyro:.1f} deg/s, "
            f"Target: {target_max_rate:.1f} deg/s"
        )

        if max_gyro > 0 and max_rc > 0:
            if max_gyro < target_max_rate * 0.9:
                rate_diff_pct = (target_max_rate - max_gyro) / target_max_rate

                super_rate_increase = 1.0 + rate_diff_pct * 0.5
                new_rate.super_rate = current_rate.super_rate * super_rate_increase

                rc_rate_increase = 1.0 + rate_diff_pct * 0.3
                new_rate.rc_rate = current_rate.rc_rate * rc_rate_increase

                logger.info(
                    f"[{axis_name}] Increasing rates: "
                    f"Super Rate {current_rate.super_rate:.3f} -> {new_rate.super_rate:.3f}, "
                    f"RC Rate {current_rate.rc_rate:.3f} -> {new_rate.rc_rate:.3f}"
                )

            elif max_gyro > target_max_rate * 1.05:
                rate_diff_pct = (max_gyro - target_max_rate) / target_max_rate

                super_rate_decrease = 1.0 - rate_diff_pct * 0.5
                new_rate.super_rate = max(0.1, current_rate.super_rate * super_rate_decrease)

        center_rc = normalized_rc[np.abs(normalized_rc) < 0.3]
        center_gyro = gyro_output[np.abs(normalized_rc) < 0.3]

        valid_mask = np.abs(center_rc) > 0.02
        valid_rc = center_rc[valid_mask]
        valid_gyro = center_gyro[valid_mask]

        if len(valid_rc) > 10 and len(valid_gyro) > 10:
            try:
                rc_gradient = np.polyfit(valid_rc, valid_gyro, 1)
                center_sensitivity = abs(rc_gradient[0]) if len(rc_gradient) > 0 else 0.0
            except (np.linalg.LinAlgError, ValueError) as e:
                logger.warning(f"[{axis_name}] Failed to compute center sensitivity: {e}")
                center_sensitivity = 0.0

            reference_sensitivity = target_max_rate * 0.3

            if center_sensitivity > reference_sensitivity * 1.3 and center_sensitivity > 0:
                new_rate.rc_expo = min(0.9, current_rate.rc_expo + 0.1)
                logger.info(f"[{axis_name}] Center too sensitive, increasing Expo: "
                            f"{current_rate.rc_expo:.2f} -> {new_rate.rc_expo:.2f}")
            elif center_sensitivity < reference_sensitivity * 0.7 and center_sensitivity > 0:
                new_rate.rc_expo = max(0.0, current_rate.rc_expo - 0.05)
                logger.info(f"[{axis_name}] Center not sensitive enough, decreasing Expo: "
                            f"{current_rate.rc_expo:.2f} -> {new_rate.rc_expo:.2f}")

        new_rate.super_rate = max(0.0, min(0.99, new_rate.super_rate))
        new_rate.rc_rate = max(0.5, min(2.5, new_rate.rc_rate))
        new_rate.rc_expo = max(0.0, min(0.95, new_rate.rc_expo))

        new_rate.super_rate = self._clamp_change(current_rate.super_rate, new_rate.super_rate)
        new_rate.rc_rate = self._clamp_change(current_rate.rc_rate, new_rate.rc_rate)

        return new_rate

    def _tune_tpa(self, profile: RateProfile, throttle_rc: np.ndarray, data: dict):
        high_throttle_mask = throttle_rc > 1500
        high_count = np.sum(high_throttle_mask)

        if high_count < 50:
            logger.info("Insufficient high-throttle data for TPA tuning, skipping")
            return

        gyro_x = data.get("gyro_x", np.array([]))
        gyro_y = data.get("gyro_y", np.array([]))
        gyro_z = data.get("gyro_z", np.array([]))

        motor_data = []
        for i in range(4):
            m = data.get(f"motor_{i}")
            if m is not None and len(m) > 0:
                motor_data.append(m)

        needs_tpa = False
        if len(motor_data) >= 4:
            all_motors = np.column_stack(motor_data)
            avg_motor = np.mean(all_motors, axis=1)
            high_throttle_avg = np.mean(avg_motor[high_throttle_mask]) if high_count > 0 else 0
            low_throttle_mask = (throttle_rc > 1000) & (throttle_rc < 1300)
            low_throttle_avg = np.mean(avg_motor[low_throttle_mask]) if np.sum(low_throttle_mask) > 0 else 0

            if high_throttle_avg > 0 and low_throttle_avg > 0:
                motor_saturation = (high_throttle_avg - low_throttle_avg) / max(low_throttle_avg, 1.0)
            else:
                motor_saturation = 0

            high_throttle_oscillation = 0
            if len(gyro_x) > 0 and high_count > 50:
                high_gyro_x = gyro_x[high_throttle_mask[:len(gyro_x)]]
                high_gyro_y = gyro_y[high_throttle_mask[:len(gyro_y)]]
                high_gyro_z = gyro_z[high_throttle_mask[:len(gyro_z)]]
                high_osc = (np.std(high_gyro_x) + np.std(high_gyro_y) + np.std(high_gyro_z)) / 3.0
                low_throttle_mask_gyro = (throttle_rc > 1000) & (throttle_rc < 1300)
                low_osc = 0
                if np.sum(low_throttle_mask_gyro) > 50:
                    low_gyro_x = gyro_x[low_throttle_mask_gyro[:len(gyro_x)]]
                    low_gyro_y = gyro_y[low_throttle_mask_gyro[:len(gyro_y)]]
                    low_gyro_z = gyro_z[low_throttle_mask_gyro[:len(gyro_z)]]
                    low_osc = (np.std(low_gyro_x) + np.std(low_gyro_y) + np.std(low_gyro_z)) / 3.0

                if low_osc > 0:
                    high_throttle_oscillation = high_osc / low_osc

            if motor_saturation > 0.15 or high_throttle_oscillation > 1.3:
                needs_tpa = True

        if needs_tpa:
            new_tpa = min(0.5, profile.tpa_rate + 0.05)
            if new_tpa > 0.01:
                profile.tpa_rate = new_tpa
                profile.tpa_breakpoint = min(1800, profile.tpa_breakpoint + 50)
                logger.info(
                    f"TPA adjusted: rate={profile.tpa_rate:.2f}, "
                    f"breakpoint={profile.tpa_breakpoint}"
                )
            else:
                profile.tpa_rate = 0.10
                profile.tpa_breakpoint = 1500
                logger.info("TPA enabled with conservative defaults")

    def _clamp_change(self, original: float, new: float) -> float:
        if abs(original) < 0.001:
            return new
        max_change = abs(original) * (self._max_change_pct / 100.0)
        return max(original - max_change, min(original + max_change, new))

    def compute_rate_report(
        self,
        original: RateProfile,
        tuned: RateProfile,
    ) -> dict:
        report = {"axes": {}, "global": {}}

        for i, axis_name in enumerate(self.AXES):
            orig = original.get_axis(i)
            tuned_axis = tuned.get_axis(i)

            orig_max = orig.compute_max_rate()
            tuned_max = tuned_axis.compute_max_rate()

            report["axes"][axis_name] = {
                "RC_Rate": {"original": orig.rc_rate, "tuned": tuned_axis.rc_rate,
                            "change_pct": _pct_change(orig.rc_rate, tuned_axis.rc_rate)},
                "Super_Rate": {"original": orig.super_rate, "tuned": tuned_axis.super_rate,
                               "change_pct": _pct_change(orig.super_rate, tuned_axis.super_rate)},
                "RC_Expo": {"original": orig.rc_expo, "tuned": tuned_axis.rc_expo,
                            "change_pct": _pct_change(orig.rc_expo, tuned_axis.rc_expo)},
                "Max_Angular_Rate": {"original": orig_max, "tuned": tuned_max,
                                     "change_pct": _pct_change(orig_max, tuned_max)},
            }

        report["global"] = {
            "TPA_Rate": {"original": original.tpa_rate, "tuned": tuned.tpa_rate,
                         "change_pct": _pct_change(original.tpa_rate, tuned.tpa_rate)},
            "TPA_Breakpoint": {"original": original.tpa_breakpoint, "tuned": tuned.tpa_breakpoint,
                               "change_pct": _pct_change(original.tpa_breakpoint, tuned.tpa_breakpoint)},
            "Throttle_RC_Rate": {"original": original.throttle_rc_rate, "tuned": tuned.throttle_rc_rate,
                                 "change_pct": _pct_change(original.throttle_rc_rate, tuned.throttle_rc_rate)},
            "Throttle_RC_Expo": {"original": original.throttle_rc_expo, "tuned": tuned.throttle_rc_expo,
                                 "change_pct": _pct_change(original.throttle_rc_expo, tuned.throttle_rc_expo)},
        }

        return report


def _pct_change(original: float, new: float) -> float:
    if abs(original) < 0.001:
        return 0.0
    return (new - original) / abs(original) * 100.0