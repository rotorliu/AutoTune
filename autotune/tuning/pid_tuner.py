import logging
from typing import Optional, Callable
import numpy as np

from autotune.fc.pid import PIDProfile, PIDAxis, PIDAdvancedAxis
from autotune.analysis.step_response import analyze_step_response, extract_step_response
from autotune.analysis.signal_processing import analyze_gyro_data
from autotune.analysis.metrics import evaluate_flight_quality
from autotune.tuning.rules import RuleEngine
from autotune.tuning.flight_scenes import FlightScene, SceneTuningPreferences, get_scene_preferences

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 1000.0
MAX_PID_ITERATIONS = 5
PID_CHANGE_LIMIT_PCT = 30.0


class PIDTuner:
    AXES = ["Roll", "Pitch", "Yaw"]
    GYRO_KEYS = ["gyro_x", "gyro_y", "gyro_z"]
    SETPOINT_KEYS = ["setpoint_x", "setpoint_y", "setpoint_z"]

    PID_RANGES = {
        "p": (5.0, 200.0),
        "i": (5.0, 200.0),
        "d": (0.0, 100.0),
        "ff": (0.0, 255.0),
        "d_min": (0.0, 100.0),
        "d_min_gain": (0.0, 100.0),
        "d_min_advance": (0.0, 50.0),
        "d_gain_boost": (0.0, 100.0),
    }

    PARAM_NAMES = ["p", "i", "d", "ff", "d_min", "d_min_gain", "d_min_advance", "d_gain_boost"]

    def __init__(
        self,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        conservative: bool = True,
        scene: Optional[FlightScene] = None,
    ):
        self.sample_rate = sample_rate
        self.conservative = conservative
        self.scene = scene
        self.scene_prefs = get_scene_preferences(scene) if scene else None
        self.rule_engine = RuleEngine.create_pid_rules(self.scene_prefs)
        base_change_limit = 20.0 if conservative else PID_CHANGE_LIMIT_PCT
        if self.scene_prefs:
            self._max_change_pct = base_change_limit * self.scene_prefs.aggressiveness
        else:
            self._max_change_pct = base_change_limit
        self._progress_callback: Optional[Callable] = None

    def set_progress_callback(self, callback: Callable):
        self._progress_callback = callback

    def _notify_progress(self, message: str, percent: int):
        if self._progress_callback:
            self._progress_callback(message, percent)

    def tune(
        self,
        data: dict[str, np.ndarray],
        initial_profile: PIDProfile,
    ) -> PIDProfile:
        tuned = initial_profile.clone()

        gyro_x = data.get("gyro_x", np.array([]))
        gyro_y = data.get("gyro_y", np.array([]))
        gyro_z = data.get("gyro_z", np.array([]))
        setpoint_x = data.get("setpoint_x", np.array([]))
        setpoint_y = data.get("setpoint_y", np.array([]))
        setpoint_z = data.get("setpoint_z", np.array([]))
        motor_0 = data.get("motor_0", np.array([]))
        motor_1 = data.get("motor_1", np.array([]))
        motor_2 = data.get("motor_2", np.array([]))
        motor_3 = data.get("motor_3", np.array([]))

        for axis_idx, (axis_name, gyro_key, sp_key) in enumerate(
            zip(self.AXES, self.GYRO_KEYS, self.SETPOINT_KEYS)
        ):
            gyro = data.get(gyro_key)
            setpoint = data.get(sp_key)

            if gyro is None or len(gyro) < 10:
                logger.warning(f"No data for {axis_name}, skipping")
                continue

            if setpoint is None:
                setpoint = np.zeros_like(gyro)

            self._notify_progress(f"正在分析 {axis_name} 轴数据...", 20 + axis_idx * 20)

            axis_pid = tuned.get_axis(axis_idx)
            axis_adv = tuned.get_axis_advanced(axis_idx)
            gyro_analysis = analyze_gyro_data(gyro, self.sample_rate)
            step_metrics = analyze_step_response(setpoint, gyro, sample_rate=self.sample_rate)

            quality = evaluate_flight_quality(
                gyro_x, gyro_y, gyro_z,
                motor_0, motor_1, motor_2, motor_3,
                setpoint_x, setpoint_y, setpoint_z,
                self.sample_rate,
            )

            new_pid, new_adv = self._tune_single_axis(
                axis_pid, axis_adv, gyro_analysis, step_metrics, quality, axis_name,
            )

            tuned_axis = tuned.get_axis(axis_idx)
            tuned_axis.p = new_pid.p
            tuned_axis.i = new_pid.i
            tuned_axis.d = new_pid.d

            tuned_adv = tuned.get_axis_advanced(axis_idx)
            tuned_adv.ff_gain = new_adv.ff_gain
            tuned_adv.d_min = new_adv.d_min
            tuned_adv.d_min_gain = new_adv.d_min_gain
            tuned_adv.d_min_advance = new_adv.d_min_advance
            tuned_adv.d_gain_boost = new_adv.d_gain_boost
            tuned.use_advanced = True

            self._notify_progress(f"{axis_name} 轴调优完成", 40 + axis_idx * 20)

        return tuned

    def _tune_single_axis(
        self,
        current_pid: PIDAxis,
        current_adv: PIDAdvancedAxis,
        gyro_analysis: dict,
        step_metrics,
        quality,
        axis_name: str,
    ):
        new_pid = current_pid.clone()
        new_adv = current_adv.clone()

        for iteration in range(MAX_PID_ITERATIONS):
            context = {
                "current_p": new_pid.p,
                "current_i": new_pid.i,
                "current_d": new_pid.d,
                "current_ff": new_adv.ff_gain,
                "current_d_min": new_adv.d_min,
                "current_d_min_gain": new_adv.d_min_gain,
                "current_d_min_advance": new_adv.d_min_advance,
                "current_d_gain_boost": new_adv.d_gain_boost,
                "overshoot_pct": step_metrics.overshoot_pct if step_metrics else 0.0,
                "rise_time_ms": step_metrics.rise_time if step_metrics else 50.0,
                "settling_time_ms": step_metrics.settling_time if step_metrics else 0.0,
                "steady_state_error_pct": step_metrics.steady_state_error if step_metrics else 0.0,
                "oscillation_index": quality.oscillation_index if quality else 0.0,
                "motor_saturation_pct": quality.motor_saturation_pct if quality else 0.0,
                "energy_high_pct": gyro_analysis.get("energy_high_pct", 0),
                "energy_low_pct": gyro_analysis.get("energy_low_pct", 0),
                "energy_mid_pct": gyro_analysis.get("energy_mid_pct", 0),
                "new_p": new_pid.p,
                "new_i": new_pid.i,
                "new_d": new_pid.d,
                "new_ff": new_adv.ff_gain,
                "new_d_min": new_adv.d_min,
                "new_d_min_gain": new_adv.d_min_gain,
                "new_d_min_advance": new_adv.d_min_advance,
                "new_d_gain_boost": new_adv.d_gain_boost,
                "applied_rules": [],
            }

            self.rule_engine.evaluate(context)

            if not context.get("applied_rules"):
                break

            new_pid.p = self._clamp_param(current_pid.p, context.get("new_p", new_pid.p), "p")
            new_pid.i = self._clamp_param(current_pid.i, context.get("new_i", new_pid.i), "i")
            new_pid.d = self._clamp_param(current_pid.d, context.get("new_d", new_pid.d), "d")
            new_adv.ff_gain = self._clamp_param(current_adv.ff_gain, context.get("new_ff", new_adv.ff_gain), "ff")
            new_adv.d_min = self._clamp_param(current_adv.d_min, context.get("new_d_min", new_adv.d_min), "d_min")
            new_adv.d_min_gain = self._clamp_param(current_adv.d_min_gain, context.get("new_d_min_gain", new_adv.d_min_gain), "d_min_gain")
            new_adv.d_min_advance = self._clamp_param(current_adv.d_min_advance, context.get("new_d_min_advance", new_adv.d_min_advance), "d_min_advance")
            new_adv.d_gain_boost = self._clamp_param(current_adv.d_gain_boost, context.get("new_d_gain_boost", new_adv.d_gain_boost), "d_gain_boost")

            for rule_msg in context.get("applied_rules", []):
                logger.info(f"[{axis_name}] Iteration {iteration + 1}: {rule_msg}")

        return new_pid, new_adv

    def _clamp_param(self, original: float, new: float, param_name: str = "p") -> float:
        if abs(original) < 0.01:
            result = new
        else:
            max_change = abs(original) * (self._max_change_pct / 100.0)
            result = max(original - max_change, min(original + max_change, new))

        lo, hi = self.PID_RANGES.get(param_name, (0.0, 255.0))
        return max(lo, min(hi, result))

    def tune_from_telemetry(
        self,
        telemetry_data: list[dict],
        initial_profile: PIDProfile,
    ) -> PIDProfile:
        arrays: dict[str, list[float]] = {}

        for sample in telemetry_data:
            for key, value in sample.items():
                if key not in arrays:
                    arrays[key] = []
                arrays[key].append(float(value) if isinstance(value, (int, float)) else 0.0)

        np_data = {key: np.array(values, dtype=np.float64)
                   for key, values in arrays.items()}

        if len(np_data.get("gyro_x", [])) < 10:
            logger.warning("Insufficient telemetry data for tuning")
            return initial_profile

        return self.tune(np_data, initial_profile)

    def compute_pid_report(
        self,
        original: PIDProfile,
        tuned: PIDProfile,
    ) -> dict:
        report = {"axes": {}}

        param_keys = [
            ("P", "p"),
            ("I", "i"),
            ("D", "d"),
            ("FF", "ff"),
            ("D_Min", "d_min"),
            ("D_Min_Gain", "d_min_gain"),
            ("D_Min_Advance", "d_min_advance"),
            ("D_Gain_Boost", "d_gain_boost"),
        ]

        for axis_idx, axis_name in enumerate(self.AXES):
            orig_pid = original.get_axis(axis_idx)
            tuned_pid = tuned.get_axis(axis_idx)
            orig_adv = original.get_axis_advanced(axis_idx)
            tuned_adv = tuned.get_axis_advanced(axis_idx)

            orig_values = {
                "p": orig_pid.p, "i": orig_pid.i, "d": orig_pid.d,
                "ff": orig_adv.ff_gain, "d_min": orig_adv.d_min,
                "d_min_gain": orig_adv.d_min_gain, "d_min_advance": orig_adv.d_min_advance,
                "d_gain_boost": orig_adv.d_gain_boost,
            }
            tuned_values = {
                "p": tuned_pid.p, "i": tuned_pid.i, "d": tuned_pid.d,
                "ff": tuned_adv.ff_gain, "d_min": tuned_adv.d_min,
                "d_min_gain": tuned_adv.d_min_gain, "d_min_advance": tuned_adv.d_min_advance,
                "d_gain_boost": tuned_adv.d_gain_boost,
            }

            axis_report = {}
            for label, key in param_keys:
                axis_report[label] = {
                    "original": orig_values[key],
                    "tuned": tuned_values[key],
                    "change_pct": _pct_change(orig_values[key], tuned_values[key]),
                }
            report["axes"][axis_name] = axis_report

        return report


def _pct_change(original: float, new: float) -> float:
    if abs(original) < 0.01:
        return 0.0
    return (new - original) / abs(original) * 100.0