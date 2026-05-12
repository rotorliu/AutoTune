import unittest
import numpy as np
from autotune.fc.pid import PIDProfile, PIDAxis, PIDAdvancedAxis
from autotune.fc.rate import RateProfile, RateAxis
from autotune.tuning.pid_tuner import PIDTuner
from autotune.tuning.rate_tuner import RateTuner
from autotune.tuning.rules import RuleEngine, TuningRule
from autotune.tuning.optimizer import PIDOptimizer
from autotune.analysis.step_response import (
    StepResponseMetrics,
    analyze_step_response,
    extract_step_response,
    segment_flight_data,
)
from autotune.analysis.metrics import (
    FlightQualityMetrics,
    compute_oscillation_index,
    detect_motor_saturation,
    evaluate_flight_quality,
)


class TestPIDProfile(unittest.TestCase):

    def test_create_default(self):
        profile = PIDProfile()
        self.assertEqual(profile.roll_pid.p, 0.0)
        self.assertEqual(profile.roll_pid.i, 0.0)
        self.assertEqual(profile.roll_pid.d, 0.0)

    def test_to_dict_and_from_dict(self):
        profile = PIDProfile()
        profile.roll_pid = PIDAxis(p=45, i=65, d=25)
        profile.pitch_pid = PIDAxis(p=50, i=70, d=28)
        profile.yaw_pid = PIDAxis(p=65, i=75, d=20)

        data = profile.to_dict()
        restored = PIDProfile.from_dict(data)

        self.assertEqual(restored.roll_pid.p, 45)
        self.assertEqual(restored.roll_pid.i, 65)
        self.assertEqual(restored.roll_pid.d, 25)
        self.assertEqual(restored.pitch_pid.p, 50)
        self.assertEqual(restored.yaw_pid.p, 65)

    def test_clone(self):
        profile = PIDProfile()
        profile.roll_pid = PIDAxis(p=45, i=65, d=25)
        cloned = profile.clone()

        self.assertEqual(cloned.roll_pid.p, 45)
        cloned.roll_pid.p = 100
        self.assertEqual(profile.roll_pid.p, 45)

    def test_diff(self):
        p1 = PIDProfile()
        p1.roll_pid = PIDAxis(p=45, i=65, d=25)

        p2 = PIDProfile()
        p2.roll_pid = PIDAxis(p=50, i=65, d=25)

        diff = p1.diff(p2)
        self.assertIn("Roll_P", diff)

        p2.roll_pid = PIDAxis(p=45, i=65, d=25)
        diff = p1.diff(p2)
        self.assertEqual(len(diff), 0)


class TestRateProfile(unittest.TestCase):

    def test_create_default(self):
        rate = RateProfile()
        self.assertEqual(rate.roll.rc_rate, 1.0)

    def test_rate_axis_compute(self):
        axis = RateAxis(rc_rate=1.0, super_rate=0.7, rc_expo=0.3)
        rate = axis.compute_max_rate()
        self.assertGreater(rate, 0)

    def test_to_dict_and_from_dict(self):
        rate = RateProfile()
        rate.roll = RateAxis(rc_rate=1.0, super_rate=0.7, rc_expo=0.3)
        rate.pitch = RateAxis(rc_rate=1.0, super_rate=0.7, rc_expo=0.3)
        rate.yaw = RateAxis(rc_rate=0.8, super_rate=0.5, rc_expo=0.2)

        data = rate.to_dict()
        restored = RateProfile.from_dict(data)

        self.assertEqual(restored.roll.rc_rate, 1.0)
        self.assertEqual(restored.yaw.rc_rate, 0.8)

    def test_clone(self):
        rate = RateProfile()
        rate.roll = RateAxis(rc_rate=1.0, super_rate=0.7)
        cloned = rate.clone()

        self.assertEqual(cloned.roll.rc_rate, 1.0)
        cloned.roll.rc_rate = 2.0
        self.assertEqual(rate.roll.rc_rate, 1.0)


class TestStepResponse(unittest.TestCase):

    def setUp(self):
        self.sample_rate = 1000.0
        self.t = np.arange(0, 1.0, 1.0 / self.sample_rate)

    def test_step_response_metrics(self):
        metrics = StepResponseMetrics(
            rise_time=25.0,
            overshoot_pct=10.0,
            settling_time=60.0,
            steady_state_error=2.0,
        )
        score = metrics.quality_score()
        self.assertGreater(100, score)
        self.assertGreater(score, 0)

    def test_analyze_step_response_empty(self):
        setpoint = np.array([])
        gyro = np.array([])
        result = analyze_step_response(setpoint, gyro, sample_rate=self.sample_rate)
        self.assertIsInstance(result, StepResponseMetrics)

    def test_extract_step_response(self):
        setpoint = np.zeros(200)
        setpoint[50:100] = 200
        setpoint[100:150] = -200
        gyro = setpoint * 0.9

        segments = extract_step_response(setpoint, gyro, self.t[:200],
                                         sample_rate=self.sample_rate)
        self.assertIsInstance(segments, list)

    def test_segment_flight_data(self):
        gyro = np.random.randn(1000) * 50
        motor = np.abs(np.random.randn(1000)) * 500 + 1000
        rc = np.random.randn(1000) * 100
        time = np.arange(1000) / 1000.0

        segments = segment_flight_data(gyro, motor, rc, time, sample_rate=self.sample_rate)
        self.assertIsInstance(segments, list)


class TestFlightQualityMetrics(unittest.TestCase):

    def setUp(self):
        self.sample_rate = 1000.0
        self.t = np.arange(0, 1.0, 1.0 / self.sample_rate)
        self.gyro = np.sin(2 * np.pi * 50 * self.t) * 200
        self.motor = np.abs(np.sin(2 * np.pi * 30 * self.t)) * 500 + 1000

    def test_compute_oscillation_index(self):
        idx = compute_oscillation_index(self.gyro, self.sample_rate)
        self.assertGreaterEqual(idx, 0.0)
        self.assertLessEqual(idx, 1.0)

    def test_detect_motor_saturation(self):
        motor = np.array([1000, 1500, 1950, 2000, 1500, 1000])
        count, pct = detect_motor_saturation(motor)
        self.assertGreater(count, 0)
        self.assertGreater(pct, 0)

    def test_evaluate_flight_quality(self):
        result = evaluate_flight_quality(
            self.gyro, self.gyro, self.gyro,
            self.motor, self.motor, self.motor, self.motor,
            self.gyro, self.gyro, self.gyro,
            self.sample_rate,
        )
        self.assertIsInstance(result, FlightQualityMetrics)
        self.assertGreaterEqual(result.overall_score, 0)
        self.assertLessEqual(result.overall_score, 100)


class TestRuleEngine(unittest.TestCase):

    def test_rule_engine_creation(self):
        engine = RuleEngine()
        self.assertEqual(len(engine.rules), 0)

    def test_add_rule(self):
        engine = RuleEngine()

        def condition(ctx):
            return True

        def action(ctx):
            ctx["changed"] = True

        rule = TuningRule("test", "test rule", condition, action, 5)
        engine.add_rule(rule)
        self.assertEqual(len(engine.rules), 1)

    def test_evaluate_no_rules(self):
        engine = RuleEngine()
        context = {}
        result = engine.evaluate(context)
        self.assertEqual(result, {})

    def test_pid_rules_creation(self):
        engine = RuleEngine.create_pid_rules()
        self.assertGreater(len(engine.rules), 0)

    def test_overshoot_rule(self):
        engine = RuleEngine.create_pid_rules()
        context = {
            "current_p": 50.0,
            "current_i": 60.0,
            "current_d": 25.0,
            "overshoot_pct": 20.0,
            "oscillation_index": 0.0,
            "motor_saturation_pct": 0.0,
        }
        engine.evaluate(context)
        if "new_p" in context:
            self.assertLess(context["new_p"], context["current_p"])


class TestPIDOptimizer(unittest.TestCase):

    def setUp(self):
        self.t = np.linspace(0, 0.5, 500)
        self.sample_rate = 1000.0

    def test_optimizer_creation(self):
        opt = PIDOptimizer()
        self.assertIsNotNone(opt.SAFETY_BOUNDS)

    def test_itae_objective(self):
        setpoint = np.where(self.t > 0.1, 200.0, 0.0)
        gyro = setpoint * 0.95
        time = np.arange(len(setpoint)) * 0.001
        motor = np.abs(np.random.randn(len(setpoint))) * 200 + 1200

        score = PIDOptimizer.itae_objective(setpoint, gyro, time, motor)
        self.assertGreater(score, 0)

    def test_optimize_simple(self):
        opt = PIDOptimizer(max_iterations=3)

        def objective(params):
            p = params.get("P", 50)
            i = params.get("I", 60)
            d = params.get("D", 25)
            target_p = 55.0
            target_i = 65.0
            return abs(p - target_p) + abs(i - target_i) * 0.5 + abs(d) * 0.1

        initial = {"P": 50.0, "I": 60.0, "D": 25.0}
        result = opt.optimize(initial, objective)

        self.assertIn("P", result)
        self.assertGreaterEqual(result["P"], opt.SAFETY_BOUNDS["P"][0])
        self.assertLessEqual(result["P"], opt.SAFETY_BOUNDS["P"][1])

    def test_history_recorded(self):
        opt = PIDOptimizer(max_iterations=2)

        def objective(params):
            return abs(params.get("P", 50) - 55)

        result = opt.optimize({"P": 50.0}, objective)
        history = opt.get_history()
        self.assertGreater(len(history), 0)


class TestPIDTuner(unittest.TestCase):

    def setUp(self):
        self.sample_rate = 1000.0
        self.t = np.arange(0, 2.0, 1.0 / self.sample_rate)

    def _generate_test_data(self):
        gyro = np.where(self.t > 0.3, 300.0, 0.0)
        gyro = gyro + np.random.randn(len(gyro)) * 20

        data = {
            "gyro_x": gyro,
            "gyro_y": gyro * 0.8,
            "gyro_z": gyro * 0.6,
            "setpoint_x": np.where(self.t > 0.3, 300.0, 0.0),
            "setpoint_y": np.where(self.t > 0.3, 240.0, 0.0),
            "setpoint_z": np.where(self.t > 0.3, 180.0, 0.0),
            "motor_0": np.abs(np.random.randn(len(self.t))) * 200 + 1200,
            "motor_1": np.abs(np.random.randn(len(self.t))) * 200 + 1200,
            "motor_2": np.abs(np.random.randn(len(self.t))) * 200 + 1200,
            "motor_3": np.abs(np.random.randn(len(self.t))) * 200 + 1200,
        }
        return data

    def test_tuner_creation(self):
        tuner = PIDTuner(conservative=True)
        self.assertTrue(tuner.conservative)

    def test_tune_returns_profile(self):
        tuner = PIDTuner(conservative=True)
        data = self._generate_test_data()
        profile = PIDProfile()
        profile.roll_pid = PIDAxis(p=45, i=65, d=25)
        profile.pitch_pid = PIDAxis(p=50, i=70, d=28)
        profile.yaw_pid = PIDAxis(p=65, i=75, d=20)

        result = tuner.tune(data, profile)
        self.assertIsInstance(result, PIDProfile)

    def test_tune_no_data_returns_original(self):
        tuner = PIDTuner()
        data = {"gyro_x": np.array([1.0]), "setpoint_x": np.array([1.0])}
        profile = PIDProfile()
        profile.roll_pid = PIDAxis(p=45, i=65, d=25)

        result = tuner.tune(data, profile)
        self.assertIsInstance(result, PIDProfile)


class TestRateTuner(unittest.TestCase):

    def setUp(self):
        self.sample_rate = 1000.0

    def _generate_test_data(self):
        n = 2000
        self.n = n
        self.t = np.arange(n) / self.sample_rate

        rc_roll = np.sin(2 * np.pi * 2 * self.t) * 0.8
        gyro_x = rc_roll * 500

        return {
            "rc_roll": rc_roll,
            "gyro_x": gyro_x,
            "rc_pitch": rc_roll * 0.7,
            "gyro_y": gyro_x * 0.7,
            "rc_yaw": rc_roll * 0.5,
            "gyro_z": gyro_x * 0.5,
        }

    def test_tuner_creation(self):
        tuner = RateTuner(conservative=True)
        self.assertTrue(tuner.conservative)

    def test_tune_returns_profile(self):
        tuner = RateTuner(conservative=True)
        data = self._generate_test_data()
        profile = RateProfile()

        result = tuner.tune(data, profile, target_max_rate=720)
        self.assertIsInstance(result, RateProfile)


if __name__ == "__main__":
    unittest.main()