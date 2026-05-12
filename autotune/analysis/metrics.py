import numpy as np
from typing import Optional, Tuple


class FlightQualityMetrics:
    def __init__(self):
        self.oscillation_index: float = 0.0
        self.noise_level: float = 0.0
        self.motor_saturation_count: int = 0
        self.motor_saturation_pct: float = 0.0
        self.tracking_error: float = 0.0
        self.i_term_windup: float = 0.0
        self.overall_score: float = 0.0

    def __repr__(self):
        return f"FlightQuality(score={self.overall_score:.1f}/100, osc={self.oscillation_index:.2f})"

    def to_dict(self) -> dict:
        return {
            "oscillation_index": self.oscillation_index,
            "noise_level": self.noise_level,
            "motor_saturation_pct": self.motor_saturation_pct,
            "tracking_error": self.tracking_error,
            "i_term_windup": self.i_term_windup,
            "overall_score": self.overall_score,
        }


def compute_oscillation_index(
    gyro: np.ndarray,
    sample_rate: float = 1000.0,
) -> float:
    if len(gyro) < 4:
        return 0.0

    zero_crossings = 0
    for i in range(1, len(gyro)):
        if gyro[i - 1] * gyro[i] < 0:
            zero_crossings += 1

    duration = len(gyro) / sample_rate
    crossings_per_second = zero_crossings / duration if duration > 0 else 0

    energy = np.sum(np.square(gyro - np.mean(gyro)))
    n = len(gyro)
    energy_per_sample = energy / n if n > 0 else 0.0

    def _count_threshold_crossings(data: np.ndarray, threshold: float) -> int:
        crossings = 0
        for i in range(1, len(data)):
            if abs(data[i - 1]) < abs(threshold) and abs(data[i]) >= abs(threshold):
                crossings += 1
        return crossings

    threshold = np.std(gyro) * 2
    high_crossings = _count_threshold_crossings(gyro, threshold)

    oscillation_score = 0.0
    oscillation_score += min(1.0, crossings_per_second / 200.0) * 0.4
    oscillation_score += min(1.0, energy_per_sample / 10000) * 0.4
    oscillation_score += min(1.0, high_crossings / max(len(gyro), 1) * 1000) * 0.2

    return oscillation_score


def detect_motor_saturation(
    motor_data: np.ndarray,
    max_motor_value: float = 2000.0,
    saturation_threshold: float = 1950.0,
) -> Tuple[int, float]:
    saturated = np.sum(motor_data >= saturation_threshold)
    total = len(motor_data)
    percentage = (saturated / total * 100.0) if total > 0 else 0.0
    return int(saturated), percentage


def compute_tracking_error(
    setpoint: np.ndarray,
    gyro: np.ndarray,
) -> float:
    error = setpoint - gyro
    return np.mean(np.abs(error)) if len(error) > 0 else 0.0


def compute_i_term_windup(
    gyro: np.ndarray,
    setpoint: np.ndarray,
    sample_rate: float = 1000.0,
) -> float:
    error = setpoint - gyro

    accumulated = np.cumsum(error) / sample_rate

    max_accum = np.max(np.abs(accumulated))

    reference = np.max(np.abs(error)) * len(error) / sample_rate * 0.5
    reference = max(reference, 1.0)

    return max_accum / reference


def evaluate_flight_quality(
    gyro_x: np.ndarray,
    gyro_y: np.ndarray,
    gyro_z: np.ndarray,
    motor_0: np.ndarray,
    motor_1: np.ndarray,
    motor_2: np.ndarray,
    motor_3: np.ndarray,
    setpoint_x: np.ndarray = None,
    setpoint_y: np.ndarray = None,
    setpoint_z: np.ndarray = None,
    sample_rate: float = 1000.0,
) -> FlightQualityMetrics:
    result = FlightQualityMetrics()

    osc_roll = compute_oscillation_index(gyro_x, sample_rate)
    osc_pitch = compute_oscillation_index(gyro_y, sample_rate)
    osc_yaw = compute_oscillation_index(gyro_z, sample_rate)

    result.oscillation_index = (osc_roll * 0.4 + osc_pitch * 0.4 + osc_yaw * 0.2)

    noise_roll = np.std(gyro_x) if len(gyro_x) > 0 else 0.0
    noise_pitch = np.std(gyro_y) if len(gyro_y) > 0 else 0.0
    noise_yaw = np.std(gyro_z) if len(gyro_z) > 0 else 0.0
    result.noise_level = (noise_roll + noise_pitch + noise_yaw) / 3.0

    motors = np.vstack([motor_0, motor_1, motor_2, motor_3])
    max_motor = np.max(motors, axis=0)
    total_saturated = 0
    for i in range(4):
        sat_count, _ = detect_motor_saturation(
            motors[i],
            saturation_threshold=1950.0,
        )
        total_saturated += sat_count

    total_samples = len(motor_0) * 4
    result.motor_saturation_pct = (
        total_saturated / total_samples * 100.0
        if total_samples > 0 else 0.0
    )

    if setpoint_x is not None and len(setpoint_x) > 0:
        track_x = compute_tracking_error(setpoint_x, gyro_x)
        track_y = compute_tracking_error(setpoint_y, gyro_y) if setpoint_y is not None else 0.0
        track_z = compute_tracking_error(setpoint_z, gyro_z) if setpoint_z is not None else 0.0
        result.tracking_error = (track_x + track_y + track_z) / 3.0

    if setpoint_x is not None and len(setpoint_x) > 0:
        result.i_term_windup = compute_i_term_windup(gyro_x, setpoint_x, sample_rate)

    score = 100.0
    score -= result.oscillation_index * 40.0
    score -= min(0.2, result.noise_level / 200.0) * 100.0
    score -= result.motor_saturation_pct * 0.5
    score -= min(50.0, result.tracking_error) * 0.5

    result.overall_score = max(0.0, min(100.0, score))

    return result