import numpy as np
from typing import Optional, Tuple


class StepResponseMetrics:
    def __init__(
        self,
        rise_time: float = 0.0,
        peak_time: float = 0.0,
        overshoot_pct: float = 0.0,
        settling_time: float = 0.0,
        steady_state_error: float = 0.0,
        itae: float = 0.0,
        peak_value: float = 0.0,
        final_value: float = 0.0,
    ):
        self.rise_time = rise_time
        self.peak_time = peak_time
        self.overshoot_pct = overshoot_pct
        self.settling_time = settling_time
        self.steady_state_error = steady_state_error
        self.itae = itae
        self.peak_value = peak_value
        self.final_value = final_value

    def __repr__(self):
        return (
            f"StepResponse(overshoot={self.overshoot_pct:.1f}%, "
            f"rise={self.rise_time:.1f}ms, settling={self.settling_time:.1f}ms)"
        )

    def to_dict(self) -> dict:
        return {
            "rise_time_ms": self.rise_time,
            "peak_time_ms": self.peak_time,
            "overshoot_pct": self.overshoot_pct,
            "settling_time_ms": self.settling_time,
            "steady_state_error_pct": self.steady_state_error,
            "itae": self.itae,
        }

    def quality_score(self) -> float:
        score = 100.0

        score -= self.overshoot_pct * 1.5

        if self.rise_time > 30:
            score -= (self.rise_time - 30) * 0.5

        if self.settling_time > 80:
            score -= (self.settling_time - 80) * 0.2

        score -= abs(self.steady_state_error) * 2.0

        return max(0.0, min(100.0, score))


def extract_step_response(
    setpoint: np.ndarray,
    gyro: np.ndarray,
    time: np.ndarray,
    step_min_amplitude: float = 50.0,
    min_hold_time: float = 0.05,
    sample_rate: float = 1000.0,
) -> list[dict]:
    segments = []
    if len(setpoint) < 3 or len(gyro) < 3:
        return segments

    min_hold_samples = int(min_hold_time * sample_rate)

    diff = np.abs(np.diff(setpoint))
    rising = np.where(diff > step_min_amplitude)[0]

    if len(rising) < 2:
        return segments

    for i in range(len(rising) - 1):
        edge_start = rising[i]
        edge_end = rising[i + 1]

        pre_samples = edge_start
        post_samples = min(edge_end - edge_start, int(0.2 * sample_rate))

        if pre_samples < 5 or post_samples < 5:
            continue

        start = max(0, edge_start - pre_samples)
        end = min(len(gyro), edge_start + post_samples + min_hold_samples)

        seg = slice(start, end)
        segments.append({
            "start_idx": start,
            "end_idx": end,
            "setpoint": setpoint[seg],
            "gyro": gyro[seg],
            "time": time[seg] if len(time) > 0 else np.arange(end - start),
        })

    return segments


def analyze_step_response(
    setpoint: np.ndarray,
    gyro: np.ndarray,
    time: np.ndarray = None,
    sample_rate: float = 1000.0,
    settling_band_pct: float = 5.0,
) -> StepResponseMetrics:
    metrics = StepResponseMetrics()

    if len(setpoint) < 3 or len(gyro) < 3:
        return metrics

    segments = extract_step_response(
        setpoint, gyro,
        time if time is not None else np.arange(len(setpoint)),
        sample_rate=sample_rate,
    )

    if not segments:
        return _analyze_single_response(setpoint, gyro, time, sample_rate, settling_band_pct)

    all_metrics = []
    for seg in segments:
        seg_gyro = seg["gyro"]
        seg_setpoint = seg["setpoint"]
        seg_time = seg["time"]

        if len(seg_gyro) > 5:
            m = _analyze_single_response(
                seg_setpoint, seg_gyro, seg_time, sample_rate, settling_band_pct
            )
            all_metrics.append(m)

    if all_metrics:
        avg = StepResponseMetrics()
        keys = [
            "rise_time", "peak_time", "overshoot_pct",
            "settling_time", "steady_state_error", "itae",
        ]
        for key in keys:
            values = [getattr(m, key) for m in all_metrics if getattr(m, key) > 0]
            if values:
                setattr(avg, key, np.mean(values))
        return avg

    return metrics


def _analyze_single_response(
    setpoint: np.ndarray,
    gyro: np.ndarray,
    time: np.ndarray = None,
    sample_rate: float = 1000.0,
    settling_band_pct: float = 5.0,
) -> StepResponseMetrics:
    dt = 1000.0 / sample_rate

    final_setpoint = setpoint[-1] if setpoint[-1] != 0 else setpoint[-int(0.05 * sample_rate):].max()
    final_setpoint = final_setpoint if abs(final_setpoint) > 1 else np.max(np.abs(setpoint))

    if abs(final_setpoint) < 1.0:
        final_setpoint = np.max(np.abs(setpoint))
        if final_setpoint < 1.0:
            return StepResponseMetrics()

    steady_gyro = gyro[-int(0.1 * sample_rate):].mean() if len(gyro) > int(0.1 * sample_rate) else gyro[-1]
    steady_setpoint = setpoint[-int(0.1 * sample_rate):].mean() if len(setpoint) > int(0.1 * sample_rate) else setpoint[-1]

    if abs(steady_setpoint) < 1.0:
        steady_setpoint = final_setpoint

    steady_error = abs(steady_gyro - steady_setpoint) / max(abs(steady_setpoint), 1.0) * 100.0

    threshold_10 = 0.1 * abs(steady_setpoint) if steady_setpoint != 0 else 0.1
    threshold_90 = 0.9 * abs(steady_setpoint) if steady_setpoint != 0 else 0.9

    abs_gyro = np.abs(gyro - gyro[0])
    rise_start_idx = np.argmax(abs_gyro >= threshold_10 / 2) if np.any(abs_gyro >= threshold_10 / 2) else 0
    rise_end_idx = np.argmax(abs_gyro >= threshold_90)

    if rise_end_idx == 0:
        rise_end_idx = min(len(gyro) - 1, rise_start_idx + int(0.1 * sample_rate))

    rise_time = (rise_end_idx - rise_start_idx) * dt

    peak_idx = np.argmax(np.abs(gyro - steady_setpoint))
    peak_val = gyro[peak_idx]
    overshoot = max(0.0, (abs(peak_val - steady_setpoint) - abs(steady_setpoint) * settling_band_pct / 100.0)
                    / max(abs(steady_setpoint), 1.0) * 100.0)

    settling_time = 0.0
    band = settling_band_pct / 100.0 * max(abs(steady_setpoint), 1.0)
    for i in range(len(gyro) - 1, 0, -1):
        if abs(gyro[i] - steady_setpoint) > band:
            settling_time = (i + 1) * dt
            break

    t = np.arange(len(setpoint)) * dt / 1000.0
    error = setpoint - gyro
    itae = np.sum(t * np.abs(error)) * dt

    return StepResponseMetrics(
        rise_time=rise_time,
        peak_time=(peak_idx - rise_start_idx) * dt if peak_idx > rise_start_idx else 0.0,
        overshoot_pct=overshoot,
        settling_time=settling_time,
        steady_state_error=steady_error,
        itae=itae,
        peak_value=peak_val,
        final_value=steady_setpoint,
    )


def segment_flight_data(
    gyro: np.ndarray,
    motor: np.ndarray,
    rc_command: np.ndarray,
    time: np.ndarray,
    min_segment_length: float = 0.2,
    sample_rate: float = 1000.0,
) -> list[dict]:
    activity = np.abs(np.diff(gyro))
    threshold = np.mean(activity) + 2 * np.std(activity)
    active = activity > threshold

    min_samples = int(min_segment_length * sample_rate)
    segments = []

    in_segment = False
    seg_start = 0
    for i in range(len(active)):
        if not in_segment and active[i]:
            in_segment = True
            seg_start = i
        elif in_segment and not active[i]:
            seg_len = i - seg_start
            if seg_len > min_samples:
                segments.append({
                    "type": "maneuver" if np.max(np.abs(gyro[seg_start:i])) > 100 else "hover",
                    "start_idx": seg_start,
                    "end_idx": i,
                    "duration": seg_len / sample_rate,
                })
            in_segment = False

    if in_segment:
        seg_len = len(active) - seg_start
        if seg_len > min_samples:
            segments.append({
                "type": "maneuver",
                "start_idx": seg_start,
                "end_idx": len(active),
                "duration": seg_len / sample_rate,
            })

    return segments