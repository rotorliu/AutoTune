import unittest
import numpy as np
from autotune.analysis.signal_processing import (
    lowpass_filter,
    bandpass_filter,
    notch_filter,
    compute_fft,
    find_peak_frequencies,
    compute_cross_correlation,
    estimate_delay,
    moving_average,
    compute_rms,
    compute_noise_density,
    analyze_gyro_data,
)


class TestSignalProcessing(unittest.TestCase):

    def setUp(self):
        self.sample_rate = 1000.0
        self.t = np.arange(0, 1.0, 1.0 / self.sample_rate)

    def test_lowpass_filter_shape(self):
        data = np.sin(2 * np.pi * 10 * self.t)
        filtered = lowpass_filter(data, cutoff_hz=50, sample_rate=self.sample_rate)
        self.assertEqual(len(filtered), len(data))
        self.assertFalse(np.any(np.isnan(filtered)))

    def test_lowpass_filter_cutoff_too_high(self):
        data = np.sin(2 * np.pi * 10 * self.t)
        filtered = lowpass_filter(data, cutoff_hz=600, sample_rate=self.sample_rate)
        self.assertEqual(len(filtered), len(data))

    def test_bandpass_filter(self):
        data = np.sin(2 * np.pi * 50 * self.t) + 0.5 * np.sin(2 * np.pi * 200 * self.t)
        filtered = bandpass_filter(data, 30, 100, self.sample_rate)
        self.assertEqual(len(filtered), len(data))
        self.assertFalse(np.any(np.isnan(filtered)))

    def test_notch_filter(self):
        data = np.sin(2 * np.pi * 100 * self.t) + 0.3 * np.sin(2 * np.pi * 50 * self.t)
        filtered = notch_filter(data, notch_freq=100, q_factor=30, sample_rate=self.sample_rate)
        self.assertEqual(len(filtered), len(data))

        original_100hz = np.sum(np.abs(compute_fft(data, self.sample_rate, remove_dc=False)[1]))
        filtered_100hz = np.sum(np.abs(compute_fft(filtered, self.sample_rate, remove_dc=False)[1]))

    def test_compute_fft(self):
        data = np.sin(2 * np.pi * 50 * self.t)
        freq, spectrum = compute_fft(data, self.sample_rate)
        self.assertEqual(len(freq), len(spectrum))
        self.assertGreater(len(freq), 0)

        peak_idx = np.argmax(spectrum)
        self.assertAlmostEqual(freq[peak_idx], 50, delta=5)

    def test_compute_fft_no_dc(self):
        data = np.sin(2 * np.pi * 50 * self.t) + 10.0
        freq, spectrum = compute_fft(data, self.sample_rate, remove_dc=True)
        self.assertAlmostEqual(spectrum[0], 0, delta=1e-6)

    def test_find_peak_frequencies(self):
        data = (np.sin(2 * np.pi * 50 * self.t) * 100 +
                np.sin(2 * np.pi * 150 * self.t) * 50)

        peaks = find_peak_frequencies(data, self.sample_rate, top_n=3)
        self.assertGreater(len(peaks), 0)

    def test_find_peak_frequencies_empty_data(self):
        peaks = find_peak_frequencies(np.array([]), self.sample_rate)
        self.assertEqual(len(peaks), 0)

    def test_compute_cross_correlation(self):
        x = np.sin(2 * np.pi * 10 * self.t)
        y = np.roll(x, 10)
        lags, corr = compute_cross_correlation(x, y)
        self.assertEqual(len(lags), len(corr))
        self.assertGreater(len(lags), 0)

    def test_estimate_delay(self):
        x = np.sin(2 * np.pi * 10 * self.t)
        delay_samples = 25
        y = np.roll(x, delay_samples)
        delay_ms = estimate_delay(x, y, self.sample_rate)
        self.assertIsInstance(delay_ms, float)

    def test_moving_average(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        smoothed = moving_average(data, 3)
        self.assertEqual(len(smoothed), len(data))
        self.assertAlmostEqual(smoothed[1], 2.0, delta=0.01)

    def test_compute_rms(self):
        data = np.array([1.0, -1.0, 2.0, -2.0])
        rms = compute_rms(data)
        self.assertGreater(rms, 0)

    def test_compute_rms_empty(self):
        self.assertEqual(compute_rms(np.array([])), 0.0)

    def test_compute_noise_density(self):
        data = (np.sin(2 * np.pi * 50 * self.t) * 100 +
                np.random.randn(len(self.t)) * 10)
        noise = compute_noise_density(data, self.sample_rate)
        self.assertGreater(noise, 0)

    def test_analyze_gyro_data(self):
        data = np.sin(2 * np.pi * 50 * self.t) * 100
        result = analyze_gyro_data(data, self.sample_rate)

        self.assertIn("rms", result)
        self.assertIn("peak_frequencies", result)
        self.assertIn("noise_100_500hz", result)
        self.assertIn("energy_low_pct", result)
        self.assertIn("energy_mid_pct", result)
        self.assertIn("energy_high_pct", result)


if __name__ == "__main__":
    unittest.main()