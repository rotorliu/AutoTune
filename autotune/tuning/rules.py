from typing import Optional

from autotune.fc.pid import PIDProfile, PIDAxis, PIDAdvancedAxis
from autotune.analysis.signal_processing import analyze_gyro_data
from autotune.analysis.step_response import StepResponseMetrics, analyze_step_response
from autotune.analysis.metrics import evaluate_flight_quality
from autotune.tuning.flight_scenes import SceneTuningPreferences


class TuningRule:
    def __init__(self, name: str, description: str, condition_fn, action_fn, priority: int = 0):
        self.name = name
        self.description = description
        self.condition_fn = condition_fn
        self.action_fn = action_fn
        self.priority = priority

    def evaluate(self, context: dict) -> bool:
        return self.condition_fn(context)

    def apply(self, context: dict):
        self.action_fn(context)


class RuleEngine:
    def __init__(self):
        self.rules: list[TuningRule] = []

    def add_rule(self, rule: TuningRule):
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def evaluate(self, context: dict) -> dict:
        applied = []
        for rule in self.rules:
            if rule.evaluate(context):
                rule.apply(context)
                applied.append(rule.name)
        return context

    @staticmethod
    def create_pid_rules(scene_prefs: Optional[SceneTuningPreferences] = None) -> "RuleEngine":
        engine = RuleEngine()

        overshoot_threshold = scene_prefs.max_overshoot_tolerance if scene_prefs else 15.0
        rise_time_threshold = scene_prefs.min_rise_time_ms if scene_prefs else 50.0
        p_factor = scene_prefs.pid_p_multiplier if scene_prefs else 1.0
        i_factor = scene_prefs.pid_i_multiplier if scene_prefs else 1.0
        d_factor = scene_prefs.pid_d_multiplier if scene_prefs else 1.0

        engine.add_rule(TuningRule(
            name="high_overshoot",
            description=f"Reduce P or increase D when overshoot > {overshoot_threshold}%",
            condition_fn=lambda ctx, thresh=overshoot_threshold: ctx.get("overshoot_pct", 0) > thresh,
            action_fn=lambda ctx, df=d_factor: _reduce_p_or_increase_d(ctx, d_factor=df),
            priority=10,
        ))

        engine.add_rule(TuningRule(
            name="slow_response",
            description=f"Increase P when rise time > {rise_time_threshold}ms",
            condition_fn=lambda ctx, rt=rise_time_threshold, ot=overshoot_threshold: ctx.get("rise_time_ms", 0) > rt and ctx.get("overshoot_pct", 0) < ot * 0.6,
            action_fn=lambda ctx, pf=p_factor: _increase_p(ctx, p_factor=pf),
            priority=9,
        ))

        engine.add_rule(TuningRule(
            name="steady_oscillation",
            description="Reduce P or adjust D when oscillation detected",
            condition_fn=lambda ctx: ctx.get("oscillation_index", 0) > 0.5,
            action_fn=lambda ctx, df=d_factor: _reduce_p_or_increase_d(ctx, d_factor=df),
            priority=9,
        ))

        engine.add_rule(TuningRule(
            name="high_freq_noise",
            description="Increase D when high frequency oscillation detected",
            condition_fn=lambda ctx: ctx.get("energy_high_pct", 0) > 30.0,
            action_fn=lambda ctx, df=d_factor: _increase_d(ctx, d_factor=df),
            priority=8,
        ))

        engine.add_rule(TuningRule(
            name="low_freq_wander",
            description="Increase I for low frequency error",
            condition_fn=lambda ctx: ctx.get("energy_low_pct", 0) > 40.0
                                     and ctx.get("steady_state_error_pct", 0) > 3.0,
            action_fn=lambda ctx, ifac=i_factor: _increase_i(ctx, i_factor=ifac),
            priority=7,
        ))

        engine.add_rule(TuningRule(
            name="steady_state_error",
            description="Increase I when steady state error > 5%",
            condition_fn=lambda ctx: ctx.get("steady_state_error_pct", 0) > 5.0,
            action_fn=lambda ctx, ifac=i_factor: _increase_i(ctx, i_factor=ifac),
            priority=7,
        ))

        engine.add_rule(TuningRule(
            name="motor_saturation",
            description="Reduce P when motor saturation detected",
            condition_fn=lambda ctx: ctx.get("motor_saturation_pct", 0) > 10.0,
            action_fn=lambda ctx, pf=p_factor: _reduce_p(ctx, p_factor=pf),
            priority=10,
        ))

        engine.add_rule(TuningRule(
            name="d_term_excessive",
            description="Reduce D if oscillation index too high despite D being high",
            condition_fn=lambda ctx: (ctx.get("oscillation_index", 0) > 0.6
                                      and ctx.get("current_d", 0) > ctx.get("current_p", 0) * 0.5),
            action_fn=lambda ctx, df=d_factor: _reduce_d(ctx, d_factor=df),
            priority=8,
        ))

        return engine


def _reduce_p(context: dict, p_factor: float = 1.0):
    factor = 0.85 * p_factor
    current = context.get("current_p", 40.0)
    if current < 0.5:
        current = 40.0
    context["new_p"] = current * factor
    if "applied_rules" not in context:
        context["applied_rules"] = []
    context["applied_rules"].append(f"Reduce P: {context.get('current_p', 40.0):.1f} -> {context['new_p']:.1f}")


def _increase_p(context: dict, p_factor: float = 1.0):
    factor = 1.15 * p_factor
    current = context.get("current_p", 40.0)
    if current < 0.5:
        current = 40.0
    context["new_p"] = current * factor
    if "applied_rules" not in context:
        context["applied_rules"] = []
    context["applied_rules"].append(f"Increase P: {context.get('current_p', 40.0):.1f} -> {context['new_p']:.1f}")


def _reduce_p_or_increase_d(context: dict, d_factor: float = 1.0):
    p = context.get("current_p", 40.0)
    if p < 0.5:
        p = 40.0
    d = context.get("current_d", 25.0)
    if d < 0.5:
        d = 25.0
    overshoot = context.get("overshoot_pct", 0)

    if overshoot > 25.0 or d > p * 0.6:
        _reduce_p(context)
    else:
        _increase_d(context, d_factor=d_factor)


def _increase_d(context: dict, d_factor: float = 1.0):
    factor = 1.2 * d_factor
    current = context.get("current_d", 25.0)
    if current < 0.5:
        current = 25.0
    context["new_d"] = current * factor
    if "applied_rules" not in context:
        context["applied_rules"] = []
    context["applied_rules"].append(f"Increase D: {context.get('current_d', 25.0):.1f} -> {context['new_d']:.1f}")


def _reduce_d(context: dict, d_factor: float = 1.0):
    factor = 0.8 / d_factor if d_factor > 0 else 0.8
    current = context.get("current_d", 25.0)
    if current < 0.5:
        current = 25.0
    context["new_d"] = current * factor
    if "applied_rules" not in context:
        context["applied_rules"] = []
    context["applied_rules"].append(f"Reduce D: {context.get('current_d', 25.0):.1f} -> {context['new_d']:.1f}")


def _increase_i(context: dict, i_factor: float = 1.0):
    factor = 1.2 * i_factor
    current = context.get("current_i", 60.0)
    if current < 0.5:
        current = 60.0
    context["new_i"] = current * factor
    if "applied_rules" not in context:
        context["applied_rules"] = []
    context["applied_rules"].append(f"Increase I: {context.get('current_i', 60.0):.1f} -> {context['new_i']:.1f}")


def _reduce_i(context: dict, i_factor: float = 1.0):
    factor = 0.85 / i_factor if i_factor > 0 else 0.85
    current = context.get("current_i", 60.0)
    if current < 0.5:
        current = 60.0
    context["new_i"] = current * factor
    if "applied_rules" not in context:
        context["applied_rules"] = []
    context["applied_rules"].append(f"Reduce I: {context.get('current_i', 60.0):.1f} -> {context['new_i']:.1f}")
