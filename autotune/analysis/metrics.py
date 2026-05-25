import numpy as np
from typing import Optional, Tuple

from autotune.analysis.signal_processing import (
    welch_transfer_function,
    sensitivity_from_transfer,
    compute_mean_coherence,
)


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


# ---- Frequency-Domain Metric Extraction (inspired by Betaflight Autotune) ----


class FrequencyDomainMetrics:
    """Container for closed-loop frequency-domain quality metrics."""

    def __init__(self):
        self.bandwidth_hz: float = 0.0
        self.phase_margin_deg: float = 0.0
        self.resonant_peak_db: float = 0.0
        self.resonant_freq_hz: float = 0.0
        self.noise_floor_hz: float = 150.0
        self.mean_coherence: float = 0.0
        self.coherence_quality: str = "unknown"
        self.sensitivity_peak_db: float = 0.0
        self.low_freq_error_db: float = 0.0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def __repr__(self):
        return (
            f"FreqMetrics(BW={self.bandwidth_hz:.1f}Hz, "
            f"PM={self.phase_margin_deg:.1f}°, "
            f"Resonance={self.resonant_peak_db:.1f}dB @ {self.resonant_freq_hz:.1f}Hz, "
            f"Coh={self.mean_coherence:.2f}[{self.coherence_quality}])"
        )


def analyze_frequency_domain(
    setpoint: np.ndarray,
    gyro: np.ndarray,
    sample_rate: float,
    eval_range_hz: Tuple[float, float] = (5.0, 80.0),
) -> FrequencyDomainMetrics:
    """Extract closed-loop frequency-domain metrics from setpoint/gyro data.

    Uses Welch transfer function estimation to compute bandwidth, phase margin,
    resonance peak, noise floor, coherence quality, and sensitivity peak.
    """
    metrics = FrequencyDomainMetrics()

    tf = welch_transfer_function(setpoint, gyro, sample_rate)
    freq = tf["freq"]
    mag_db = tf["magnitude_db"]
    phase_deg = tf["phase_deg"]
    coherence = tf["coherence"]

    if len(freq) == 0:
        return metrics

    # Mean coherence for data quality
    metrics.mean_coherence = compute_mean_coherence(coherence, freq, eval_range_hz)
    if metrics.mean_coherence > 0.7:
        metrics.coherence_quality = "excellent"
    elif metrics.mean_coherence > 0.3:
        metrics.coherence_quality = "acceptable"
    else:
        metrics.coherence_quality = "poor"

    # Bandwidth: frequency where magnitude first crosses -3dB
    metrics.bandwidth_hz = _find_bandwidth(freq, mag_db)

    # Phase margin: phase at gain crossover plus 180
    metrics.phase_margin_deg = _find_phase_margin(freq, mag_db, phase_deg)

    # Resonance peak: maximum magnitude in passband
    peak_db, peak_freq = _find_resonant_peak(freq, mag_db, eval_range_hz[:1][0], 500.0)
    metrics.resonant_peak_db = peak_db
    metrics.resonant_freq_hz = peak_freq

    # Noise floor: frequency where coherence drops below 0.5
    metrics.noise_floor_hz = _find_noise_floor(freq, coherence)

    # Sensitivity peak
    mag_linear = 10 ** (mag_db / 20.0)
    sens = sensitivity_from_transfer(mag_linear, phase_deg)
    metrics.sensitivity_peak_db = float(20 * np.log10(np.max(sens)))

    # Low-frequency tracking error: average magnitude 2-15 Hz
    metrics.low_freq_error_db = _compute_low_freq_error(freq, mag_db)

    return metrics


def _find_bandwidth(freq: np.ndarray, mag_db: np.ndarray) -> float:
    """Find -3dB closed-loop bandwidth."""
    if len(freq) < 2:
        return 0.0
    cross_idx = np.where(mag_db <= -3.0)[0]
    if len(cross_idx) == 0:
        return float(freq[-1])
    return float(freq[cross_idx[0]])


def _find_phase_margin(
    freq: np.ndarray,
    mag_db: np.ndarray,
    phase_deg: np.ndarray,
) -> float:
    """Compute phase margin at the gain crossover frequency (0dB)."""
    if len(freq) < 2:
        return 0.0
    # Find where magnitude crosses 0dB (unity gain)
    above_zero = mag_db > 0
    cross_indices = np.where(np.diff(above_zero))[0]
    if len(cross_indices) == 0:
        return 90.0  # default if never crosses 0dB
    idx = cross_indices[-1]  # last crossing (gain crossover)
    if idx + 1 >= len(freq):
        return 0.0
    # Interpolate phase at the crossing frequency
    t = -mag_db[idx] / (mag_db[idx + 1] - mag_db[idx] + 1e-10)
    phi = phase_deg[idx] + t * (phase_deg[idx + 1] - phase_deg[idx])
    pm = phi + 180.0
    return float(max(0.0, min(180.0, pm)))


def _find_resonant_peak(
    freq: np.ndarray,
    mag_db: np.ndarray,
    min_freq: float,
    max_freq: float = 500.0,
) -> Tuple[float, float]:
    """Find the maximum magnitude peak in the passband."""
    mask = (freq >= min_freq) & (freq <= max_freq)
    if not np.any(mask):
        return 0.0, 0.0
    idx = np.argmax(mag_db[mask])
    return float(mag_db[mask][idx]), float(freq[mask][idx])


def _find_noise_floor(freq: np.ndarray, coherence: np.ndarray) -> float:
    """Find the frequency where coherence drops below 0.5."""
    if len(freq) < 2:
        return 150.0
    low_coh = np.where(coherence < 0.5)[0]
    if len(low_coh) == 0:
        return float(freq[-1])
    return float(freq[low_coh[0]])


def _compute_low_freq_error(freq: np.ndarray, mag_db: np.ndarray) -> float:
    """Average magnitude deviation from 0dB in the 2-15 Hz band."""
    mask = (freq >= 2.0) & (freq <= 15.0)
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(mag_db[mask])))