from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, Optional


class FlightScene(Enum):
    CINEMATOGRAPHY = "cinematography"
    FREESTYLE = "freestyle"
    RACING = "racing"
    BEGINNER = "beginner"
    FPV = "fpv"


@dataclass
class SceneTuningPreferences:
    name: str
    description: str
    pid_p_multiplier: float = 1.0
    pid_i_multiplier: float = 1.0
    pid_d_multiplier: float = 1.0
    max_overshoot_tolerance: float = 15.0
    min_rise_time_ms: float = 30.0
    aggressiveness: float = 1.0
    smoothness_weight: float = 0.5
    responsiveness_weight: float = 0.5
    noise_reduction_enabled: bool = False
    filter_strength: float = 1.0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "SceneTuningPreferences":
        return cls(**data)


SCENE_CONFIGS: Dict[FlightScene, SceneTuningPreferences] = {
    FlightScene.CINEMATOGRAPHY: SceneTuningPreferences(
        name="航拍",
        description="适合航拍和电影拍摄，追求平稳流畅的飞行体验",
        pid_p_multiplier=0.8,
        pid_i_multiplier=0.7,
        pid_d_multiplier=1.2,
        max_overshoot_tolerance=8.0,
        min_rise_time_ms=40.0,
        aggressiveness=0.5,
        smoothness_weight=0.8,
        responsiveness_weight=0.2,
        noise_reduction_enabled=True,
        filter_strength=1.5,
    ),
    FlightScene.FREESTYLE: SceneTuningPreferences(
        name="花飞",
        description="适合特技飞行和花式动作，平衡响应速度和稳定性",
        pid_p_multiplier=1.0,
        pid_i_multiplier=1.0,
        pid_d_multiplier=1.0,
        max_overshoot_tolerance=15.0,
        min_rise_time_ms=25.0,
        aggressiveness=0.75,
        smoothness_weight=0.4,
        responsiveness_weight=0.6,
        noise_reduction_enabled=False,
        filter_strength=1.0,
    ),
    FlightScene.RACING: SceneTuningPreferences(
        name="竞速",
        description="适合竞速比赛，追求极致的响应速度和操控性",
        pid_p_multiplier=1.3,
        pid_i_multiplier=1.2,
        pid_d_multiplier=0.7,
        max_overshoot_tolerance=20.0,
        min_rise_time_ms=15.0,
        aggressiveness=1.0,
        smoothness_weight=0.2,
        responsiveness_weight=0.8,
        noise_reduction_enabled=False,
        filter_strength=0.5,
    ),
    FlightScene.BEGINNER: SceneTuningPreferences(
        name="新手",
        description="适合初学者，更保守的参数确保飞行安全稳定",
        pid_p_multiplier=0.6,
        pid_i_multiplier=0.5,
        pid_d_multiplier=1.4,
        max_overshoot_tolerance=5.0,
        min_rise_time_ms=50.0,
        aggressiveness=0.3,
        smoothness_weight=0.9,
        responsiveness_weight=0.1,
        noise_reduction_enabled=True,
        filter_strength=2.0,
    ),
    FlightScene.FPV: SceneTuningPreferences(
        name="穿越",
        description="适合第一视角穿越飞行，兼顾速度和稳定性",
        pid_p_multiplier=1.1,
        pid_i_multiplier=0.9,
        pid_d_multiplier=0.9,
        max_overshoot_tolerance=12.0,
        min_rise_time_ms=20.0,
        aggressiveness=0.85,
        smoothness_weight=0.35,
        responsiveness_weight=0.65,
        noise_reduction_enabled=False,
        filter_strength=0.8,
    ),
}


def get_scene_preferences(scene: FlightScene) -> SceneTuningPreferences:
    return SCENE_CONFIGS.get(scene, SCENE_CONFIGS[FlightScene.FREESTYLE])


def get_all_scenes() -> list[FlightScene]:
    return list(FlightScene)


def get_scene_by_name(name: str) -> Optional[FlightScene]:
    for scene in FlightScene:
        if scene.value == name or scene.name.lower() == name.lower():
            return scene
    return None
