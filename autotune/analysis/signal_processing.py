import numpy as np
from scipy import signal, fft
from typing import Optional, Tuple


def lowpass_filter(
    data: np.ndarray,
    cutoff_hz: float,
    sample_rate: float,
    order: int = 4,
) -> np.ndarray:
    nyquist = 0.5 * sample_rate
    if cutoff_hz >= nyquist:
        return data.copy()
    normalized_cutoff = cutoff_hz / nyquist
    b, a = signal.butter(order, normalized_cutoff, btype="low", analog=False)
    return signal.filtfilt(b, a, data)


def bandpass_filter(
    data: np.ndarray,
    low_cutoff: float,
    high_cutoff: float,
    sample_rate: float,
    order: int = 4,
) -> np.ndarray:
    nyquist = 0.5 * sample_rate
    low = max(1.0, low_cutoff) / nyquist
    high = min(high_cutoff, nyquist - 1.0) / nyquist
    if low >= high:
        return data.copy()
    b, a = signal.butter(order, [low, high], btype="band", analog=False)
    return signal.filtfilt(b, a, data)


def notch_filter(
    data: np.ndarray,
    notch_freq: float,
    q_factor: float,
    sample_rate: float,
) -> np.ndarray:
    nyquist = 0.5 * sample_rate
    if notch_freq >= nyquist:
        return data.copy()
    w0 = notch_freq / nyquist
    b, a = signal.iirnotch(w0, q_factor)
    return signal.filtfilt(b, a, data)


def compute_fft(
    data: np.ndarray,
    sample_rate: float,
    remove_dc: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    if remove_dc:
        data = data - np.mean(data)

    n = len(data)
    if n == 0:
        return np.array([]), np.array([])

    freq = fft.rfftfreq(n, d=1.0 / sample_rate)
    spectrum = np.abs(fft.rfft(data))
    return freq, spectrum


def find_peak_frequencies(
    data: np.ndarray,
    sample_rate: float,
    min_freq: float = 10.0,
    max_freq: float = 500.0,
    top_n: int = 5,
) -> list[Tuple[float, float]]:
    freq, spectrum = compute_fft(data, sample_rate)

    mask = (freq >= min_freq) & (freq <= max_freq)
    freq = freq[mask]
    spectrum = spectrum[mask]

    if len(spectrum) < 2:
        return []

    peaks, properties = signal.find_peaks(
        spectrum,
        distance=max(1, len(spectrum) // 20),
        prominence=np.max(spectrum) * 0.05 if np.max(spectrum) > 0 else 0,
    )

    if len(peaks) == 0:
        return [(freq[np.argmax(spectrum)], np.max(spectrum))]

    peak_tuples = [(freq[i], spectrum[i]) for i in peaks]
    peak_tuples.sort(key=lambda x: x[1], reverse=True)

    return peak_tuples[:top_n]


def compute_cross_correlation(
    x: np.ndarray,
    y: np.ndarray,
    max_lag_samples: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    correlation = signal.correlate(x - np.mean(x), y - np.mean(y), mode="full")
    lags = signal.correlation_lags(len(x), len(y), mode="full")

    if max_lag_samples is not None:
        center = len(correlation) // 2
        start = max(0, center - max_lag_samples)
        end = min(len(correlation), center + max_lag_samples + 1)
        correlation = correlation[start:end]
        lags = lags[start:end]

    return lags, correlation


def estimate_delay(
    setpoint: np.ndarray,
    response: np.ndarray,
    sample_rate: float,
    max_delay_ms: float = 50.0,
) -> float:
    max_lag = int(max_delay_ms / 1000.0 * sample_rate)
    lags, corr = compute_cross_correlation(setpoint, response, max_lag)

    if len(corr) == 0:
        return 0.0

    best_lag_idx = np.argmax(corr)
    delay_samples = lags[best_lag_idx]
    return delay_samples / sample_rate * 1000.0


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1:
        return data.copy()
    return np.convolve(data, np.ones(window_size) / window_size, mode="same")


def compute_rms(data: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square(data - np.mean(data)))) if len(data) > 0 else 0.0


def compute_noise_density(
    data: np.ndarray,
    sample_rate: float,
    freq_range: Tuple[float, float] = (100, 500),
) -> float:
    freq, spectrum = compute_fft(data, sample_rate)
    mask = (freq >= freq_range[0]) & (freq <= freq_range[1])
    if not np.any(mask):
        return 0.0
    return np.mean(spectrum[mask])


def analyze_gyro_data(
    gyro_data: np.ndarray,
    sample_rate: float,
    motor_freq_hz: float = 0.0,
) -> dict:
    result = {}

    result["rms"] = compute_rms(gyro_data)
    result["peak_to_peak"] = np.ptp(gyro_data) if len(gyro_data) > 0 else 0.0

    result["peak_frequencies"] = find_peak_frequencies(
        gyro_data, sample_rate, min_freq=10, max_freq=500, top_n=5
    )

    result["noise_100_500hz"] = compute_noise_density(
        gyro_data, sample_rate, (100, 500)
    )

    freq, spectrum = compute_fft(gyro_data, sample_rate)
    total_energy = np.sum(spectrum)
    if total_energy > 0:
        low_mask = freq <= 50
        mid_mask = (freq > 50) & (freq <= 150)
        high_mask = freq > 150
        result["energy_low_pct"] = np.sum(spectrum[low_mask]) / total_energy * 100
        result["energy_mid_pct"] = np.sum(spectrum[mid_mask]) / total_energy * 100
        result["energy_high_pct"] = np.sum(spectrum[high_mask]) / total_energy * 100
    else:
        result["energy_low_pct"] = 0
        result["energy_mid_pct"] = 0
        result["energy_high_pct"] = 0

    return result


# ---- Welch Transfer Function Estimation (inspired by Betaflight Autotune) ----


def welch_transfer_function(
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    sample_rate: float,
    segment_size: int = 1024,
    overlap: float = 0.5,
) -> dict:
    """Estimate closed-loop transfer function H(f) = Sxy(f) / Sxx(f) using Welch's
    averaged periodogram method with Hanning window.

    Returns a dict with keys: 'freq', 'magnitude_db', 'phase_deg', 'coherence',
    'sxx', 'sxy', 'syy'.
    """
    result = {
        "freq": np.array([]),
        "magnitude_db": np.array([]),
        "phase_deg": np.array([]),
        "coherence": np.array([]),
        "sxx": np.array([]),
        "sxy": np.array([]),
        "syy": np.array([]),
    }

    n = len(input_signal)
    if n < segment_size or len(output_signal) < segment_size:
        return result

    step = int(segment_size * (1 - overlap))
    if step < 1:
        step = 1

    window = np.hanning(segment_size)
    n_segments = max(1, (n - segment_size) // step + 1)

    freq = np.fft.rfftfreq(segment_size, d=1.0 / sample_rate)
    sxx_accum = np.zeros(len(freq), dtype=np.float64)
    syy_accum = np.zeros(len(freq), dtype=np.float64)
    sxy_accum = np.zeros(len(freq), dtype=np.complex128)

    for i in range(n_segments):
        start = i * step
        end = start + segment_size
        x_seg = input_signal[start:end] - np.mean(input_signal[start:end])
        y_seg = output_signal[start:end] - np.mean(output_signal[start:end])

        xw = x_seg * window
        yw = y_seg * window

        xf = np.fft.rfft(xw)
        yf = np.fft.rfft(yw)

        sxx_accum += np.abs(xf) ** 2
        syy_accum += np.abs(yf) ** 2
        sxy_accum += np.conj(xf) * yf

    win_power = np.sum(window ** 2) / segment_size
    sxx_accum /= (n_segments * win_power)
    syy_accum /= (n_segments * win_power)
    sxy_accum /= (n_segments * win_power)

    with np.errstate(divide="ignore", invalid="ignore"):
        tf_est = sxy_accum / sxx_accum
        coherence = np.abs(sxy_accum) ** 2 / (sxx_accum * syy_accum)

    magnitude = np.abs(tf_est)
    magnitude_db = 20 * np.log10(magnitude, where=magnitude > 0)
    magnitude_db = np.where(magnitude > 0, magnitude_db, -80.0)
    phase_deg = np.angle(tf_est, deg=True)

    result["freq"] = freq
    result["magnitude_db"] = magnitude_db
    result["phase_deg"] = phase_deg
    result["coherence"] = np.nan_to_num(coherence, nan=0.0)
    result["sxx"] = sxx_accum
    result["sxy"] = sxy_accum
    result["syy"] = syy_accum

    return result


def sensitivity_from_transfer(
    tf_magnitude: np.ndarray,
    tf_phase_deg: np.ndarray,
) -> np.ndarray:
    """Compute sensitivity function S(f) = 1 - T(f) from closed-loop transfer T(f).
    Higher |S| indicates worse disturbance rejection at that frequency.
    """
    tf_complex = tf_magnitude * np.exp(1j * np.deg2rad(tf_phase_deg))
    s_complex = 1.0 - tf_complex
    return np.abs(s_complex)


def compute_mean_coherence(
    coherence: np.ndarray,
    freq: np.ndarray,
    freq_range: Tuple[float, float] = (5.0, 100.0),
) -> float:
    """Average coherence within a frequency band — used as data quality metric.
    > 0.7 = excellent, 0.3-0.7 = acceptable, < 0.3 = poor.
    """
    mask = (freq >= freq_range[0]) & (freq <= freq_range[1])
    if not np.any(mask):
        return 0.0
    return float(np.mean(coherence[mask]))