import numpy as np
import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class PIDOptimizer:
    MAX_ITERATIONS = 30
    CONVERGENCE_TOLERANCE = 0.001
    SAFETY_BOUNDS = {
        "P": (5.0, 200.0),
        "I": (5.0, 200.0),
        "D": (0.0, 100.0),
        "FF": (0.0, 200.0),
    }

    def __init__(self, bounds: dict = None, max_iterations: int = None):
        self.bounds = bounds or self.SAFETY_BOUNDS.copy()
        if max_iterations is not None:
            self.max_iterations = max_iterations
        self.max_iterations = max_iterations or self.MAX_ITERATIONS
        self._history: list[dict] = []

    def optimize(
        self,
        initial_params: dict,
        objective_fn: Callable[[dict], float],
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        self._history = []
        best_params = dict(initial_params)
        best_score = objective_fn(best_params)

        self._history.append({
            "iteration": 0,
            "params": dict(best_params),
            "score": best_score,
        })

        logger.info(f"Initial score: {best_score:.4f} with params {best_params}")

        step_sizes = {key: abs(val * 0.1) if abs(val) > 0 else 1.0
                      for key, val in best_params.items() if key in self.SAFETY_BOUNDS}

        for iteration in range(1, self.max_iterations + 1):
            improved = False

            for param_name in list(best_params.keys()):
                if param_name not in self.SAFETY_BOUNDS:
                    continue

                param_min, param_max = self.SAFETY_BOUNDS[param_name]
                current_val = best_params[param_name]
                step = step_sizes.get(param_name, max(abs(current_val * 0.1), 1.0))

                for direction in [1.0, -1.0]:
                    trial_val = current_val + direction * step

                    trial_val = max(param_min, min(param_max, trial_val))

                    if abs(trial_val - current_val) < 0.01:
                        continue

                    trial_params = dict(best_params)
                    trial_params[param_name] = trial_val

                    try:
                        trial_score = objective_fn(trial_params)
                    except Exception as e:
                        logger.debug(f"Objective error for {param_name}={trial_val}: {e}")
                        continue

                    if trial_score < best_score:
                        best_score = trial_score
                        best_params = trial_params
                        improved = True
                        step_sizes[param_name] = step * 1.1

            if not improved:
                for key in step_sizes:
                    step_sizes[key] *= 0.5

            self._history.append({
                "iteration": iteration,
                "params": dict(best_params),
                "score": best_score,
                "improved": improved,
            })

            if progress_callback:
                progress_callback(iteration, self.max_iterations, best_params, best_score)

            if not improved and all(s < 0.02 for s in step_sizes.values()):
                logger.info(f"Converged at iteration {iteration}")
                break

            max_step = max(step_sizes.values()) if step_sizes else 0
            if max_step < 0.01:
                logger.info(f"Step sizes converged at iteration {iteration}")
                break

        logger.info(f"Optimization complete. Best score: {best_score:.4f}")
        return best_params

    def get_history(self) -> list[dict]:
        return list(self._history)

    @staticmethod
    def itae_objective(
        setpoint: np.ndarray,
        gyro: np.ndarray,
        time: np.ndarray,
        motor: np.ndarray = None,
        motor_penalty_weight: float = 10.0,
    ) -> float:
        error = setpoint - gyro
        dt = time[1] - time[0] if len(time) > 1 else 0.001

        t_weight = np.arange(1, len(error) + 1) * dt
        itae = np.sum(np.abs(error) * t_weight * dt)

        if motor is not None and len(motor) > 0:
            saturated = np.sum(motor > 1950)
            itae += saturated * motor_penalty_weight

        oscillation_penalty = 0.0
        if len(gyro) > 3:
            gyro_diff = np.abs(np.diff(gyro))
            oscillation_penalty = np.sum(gyro_diff > np.std(gyro) * 5) * 0.5

        return itae + oscillation_penalty