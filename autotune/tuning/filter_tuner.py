import logging
from typing import Optional, Callable
import numpy as np

from autotune.fc.config import FilterConfig
from autotune.analysis.signal_processing import analyze_gyro_data, compute_fft, find_peak_frequencies

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 1000.0


class FilterTuner:

    def __init__(
        self,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        conservative: bool = True,
    ):
        self.sample_rate = sample_rate
        self.conservative = conservative
        self._progress_callback: Optional[Callable] = None

    def set_progress_callback(self, callback: Callable):
        self._progress_callback = callback

    def _notify_progress(self, message: str, percent: int):
        if self._progress_callback:
            self._progress_callback(message, percent)

    def tune(
        self,
        data: dict[str, np.ndarray],
        initial_config: FilterConfig,
    ) -> FilterConfig:
        tuned = FilterConfig()
        tuned.gyro_lowpass_hz = initial_config.gyro_lowpass_hz
        tuned.gyro_lowpass2_hz = initial_config.gyro_lowpass2_hz
        tuned.gyro_lowpass_type = initial_config.gyro_lowpass_type
        tuned.gyro_lowpass2_type = initial_config.gyro_lowpass2_type
        tuned.gyro_notch_hz = initial_config.gyro_notch_hz
        tuned.gyro_notch_cutoff = initial_config.gyro_notch_cutoff
        tuned.dterm_lowpass_hz = initial_config.dterm_lowpass_hz
        tuned.dterm_lowpass2_hz = initial_config.dterm_lowpass2_hz
        tuned.dterm_lowpass_type = initial_config.dterm_lowpass_type
        tuned.dterm_lowpass2_type = initial_config.dterm_lowpass2_type
        tuned.dterm_notch_hz = initial_config.dterm_notch_hz
        tuned.dterm_notch_cutoff = initial_config.dterm_notch_cutoff
        tuned.yaw_lowpass_hz = initial_config.yaw_lowpass_hz

        self._notify_progress("正在分析陀螺仪噪声频谱...", 5)

        gyro_axes = {
            "gyro_x": data.get("gyro_x"),
            "gyro_y": data.get("gyro_y"),
            "gyro_z": data.get("gyro_z"),
        }

        gyro_combined = None
        for key, gyro in gyro_axes.items():
            if gyro is not None and len(gyro) > 10:
                if gyro_combined is None:
                    gyro_combined = gyro.copy()
                else:
                    gyro_combined = np.concatenate([gyro_combined, gyro])

        if gyro_combined is None or len(gyro_combined) < 100:
            logger.warning("Insufficient gyro data for filter tuning")
            return tuned

        analysis = analyze_gyro_data(gyro_combined, self.sample_rate)
        peak_freqs = analysis.get("peak_frequencies", [])

        self._notify_progress("正在调优陀螺仪低通滤波器...", 20)

        noise_100_500 = analysis.get("noise_100_500hz", 0)
        energy_high = analysis.get("energy_high_pct", 0)

        if energy_high > 25.0 or noise_100_500 > 0:
            recommended_gyro_lpf = self._compute_recommended_lpf(
                peak_freqs, energy_high, self.sample_rate
            )
            if recommended_gyro_lpf > 0:
                aged_lpf = min(tuned.gyro_lowpass_hz, recommended_gyro_lpf)
                tuned.gyro_lowpass_hz = max(80.0, min(500.0, aged_lpf))
                logger.info(
                    f"Gyro LPF1 adjusted: {initial_config.gyro_lowpass_hz:.0f}Hz -> "
                    f"{tuned.gyro_lowpass_hz:.0f}Hz"
                )

        if energy_high > 15.0:
            recommended_gyro_lpf2 = 2.0 * tuned.gyro_lowpass_hz if tuned.gyro_lowpass_hz > 0 else 400.0
            tuned.gyro_lowpass2_hz = max(120.0, min(500.0, recommended_gyro_lpf2))
            logger.info(
                f"Gyro LPF2 set to: {tuned.gyro_lowpass2_hz:.0f}Hz"
            )

        self._notify_progress("正在调优 D-Term 低通滤波器...", 40)

        dterm_lpf = tuned.gyro_lowpass_hz * 0.65 if tuned.gyro_lowpass_hz > 0 else 130.0
        tuned.dterm_lowpass_hz = max(50.0, min(250.0, dterm_lpf))
        tuned.dterm_lowpass2_hz = max(90.0, min(350.0, dterm_lpf * 1.6))

        logger.info(
            f"D-Term LPF1: {initial_config.dterm_lowpass_hz:.0f}Hz -> "
            f"{tuned.dterm_lowpass_hz:.0f}Hz"
        )

        self._notify_progress("正在分析陷波滤波器需求...", 55)

        if len(peak_freqs) > 0:
            dominant_freq, dominant_amp = peak_freqs[0]

            if dominant_amp > 0 and len(peak_freqs) > 1:
                avg_amp = np.mean([a for _, a in peak_freqs[1:]]) if len(peak_freqs) > 1 else 0
                if avg_amp > 0 and dominant_amp / avg_amp > 3.0 and dominant_freq > 60:
                    tuned.gyro_notch_hz = dominant_freq
                    tuned.gyro_notch_cutoff = max(40.0, dominant_freq * 0.7)
                    logger.info(
                        f"Gyro Notch set: center={tuned.gyro_notch_hz:.0f}Hz, "
                        f"cutoff={tuned.gyro_notch_cutoff:.0f}Hz"
                    )

        tuned.yaw_lowpass_hz = max(50.0, tuned.gyro_lowpass_hz * 0.8)

        self._notify_progress("滤波器调优完成", 100)
        return tuned

    def _compute_recommended_lpf(
        self,
        peak_freqs: list,
        energy_high_pct: float,
        sample_rate: float,
    ) -> float:
        if energy_high_pct < 15.0:
            return 0

        if len(peak_freqs) == 0:
            return 250.0

        dominant_freq = peak_freqs[0][0] if len(peak_freqs) > 0 else 200.0

        if energy_high_pct > 40.0:
            return max(80.0, dominant_freq * 0.8)
        elif energy_high_pct > 25.0:
            return max(120.0, dominant_freq * 1.2)
        else:
            return max(180.0, dominant_freq * 1.5)

    def compute_filter_report(
        self,
        original: FilterConfig,
        tuned: FilterConfig,
    ) -> dict:
        def _chg(o, n, prec=1):
            fmt = f"{{:.{prec}f}}"
            pct = (n - o) / abs(o) * 100.0 if abs(o) > 0.01 else 0.0
            return {"original": fmt.format(o), "tuned": fmt.format(n), "change_pct": pct}

        return {
            "gyro_lowpass_hz": _chg(original.gyro_lowpass_hz, tuned.gyro_lowpass_hz),
            "gyro_lowpass2_hz": _chg(original.gyro_lowpass2_hz, tuned.gyro_lowpass2_hz),
            "dterm_lowpass_hz": _chg(original.dterm_lowpass_hz, tuned.dterm_lowpass_hz),
            "dterm_lowpass2_hz": _chg(original.dterm_lowpass2_hz, tuned.dterm_lowpass2_hz),
            "gyro_notch_hz": _chg(original.gyro_notch_hz, tuned.gyro_notch_hz),
            "gyro_notch_cutoff": _chg(original.gyro_notch_cutoff, tuned.gyro_notch_cutoff),
            "yaw_lowpass_hz": _chg(original.yaw_lowpass_hz, tuned.yaw_lowpass_hz),
        }