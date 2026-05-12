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
            gyro_analysis = analyze_gyro_data(gyro, self.sample_rate)
            step_metrics = analyze_step_response(setpoint, gyro, sample_rate=self.sample_rate)

            quality = evaluate_flight_quality(
                gyro_x, gyro_y, gyro_z,
                motor_0, motor_1, motor_2, motor_3,
                setpoint_x, setpoint_y, setpoint_z,
                self.sample_rate,
            )

            new_pid = self._tune_single_axis(
                axis_pid,
                gyro_analysis,
                step_metrics,
                quality,
                axis_name,
            )

            tuned_axis = tuned.get_axis(axis_idx)
            tuned_axis.p = new_pid.p
            tuned_axis.i = new_pid.i
            tuned_axis.d = new_pid.d

            self._notify_progress(f"{axis_name} 轴调优完成", 40 + axis_idx * 20)

        return tuned

    def _tune_single_axis(
        self,
        current_pid: PIDAxis,
        gyro_analysis: dict,
        step_metrics,
        quality,
        axis_name: str,
    ) -> PIDAxis:
        new_pid = current_pid.clone()

        for iteration in range(MAX_PID_ITERATIONS):
            context = {
                "current_p": new_pid.p,
                "current_i": new_pid.i,
                "current_d": new_pid.d,
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
                "applied_rules": [],
            }

            self.rule_engine.evaluate(context)

            if not context.get("applied_rules"):
                break

            new_p = context.get("new_p", new_pid.p)
            new_i = context.get("new_i", new_pid.i)
            new_d = context.get("new_d", new_pid.d)

            new_p = self._clamp_change(current_pid.p, new_p)
            new_i = self._clamp_change(current_pid.i, new_i)
            new_d = self._clamp_change(current_pid.d, new_d)

            new_pid.p = max(5.0, min(200.0, new_p))
            new_pid.i = max(5.0, min(200.0, new_i))
            new_pid.d = max(0.0, min(100.0, new_d))

            for rule_msg in context.get("applied_rules", []):
                logger.info(f"[{axis_name}] Iteration {iteration + 1}: {rule_msg}")

        return new_pid

    def _clamp_change(self, original: float, new: float) -> float:
        if abs(original) < 0.01:
            return new
        max_change = abs(original) * (self._max_change_pct / 100.0)
        clamped = max(original - max_change, min(original + max_change, new))
        return clamped

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

        for i, axis_name in enumerate(self.AXES):
            orig_axis = original.get_axis(i)
            tuned_axis = tuned.get_axis(i)

            report["axes"][axis_name] = {
                "P": {"original": orig_axis.p, "tuned": tuned_axis.p,
                      "change_pct": _pct_change(orig_axis.p, tuned_axis.p)},
                "I": {"original": orig_axis.i, "tuned": tuned_axis.i,
                      "change_pct": _pct_change(orig_axis.i, tuned_axis.i)},
                "D": {"original": orig_axis.d, "tuned": tuned_axis.d,
                      "change_pct": _pct_change(orig_axis.d, tuned_axis.d)},
            }

        return report


def _pct_change(original: float, new: float) -> float:
    if abs(original) < 0.01:
        return 0.0
    return (new - original) / abs(original) * 100.0