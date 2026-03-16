from dataclasses import dataclass, field
from typing import Union
from dataclasses import asdict


@dataclass
class ObservationMasks:
    """Controls what each agent can observe."""
    mask_pitch_lhs: Union[bool, list[bool]] = False
    mask_pitch_rhs: Union[bool, list[bool]] = False
    mask_pitch_bhs: Union[bool, list[bool]] = False
    mask_pitch_ths: Union[bool, list[bool]] = False
    mask_ball: Union[bool, list[bool]] = False
    mask_opponent: Union[bool, list[bool]] = False
    mask_ball_by_distance: Union[bool, list[bool]] = False
    mask_opponent_by_distance: Union[bool, list[bool]] = False
    mask_if_far: bool = False  # modifier for the distance masks


@dataclass  
class OpponentDifficulty:
    """Controls AI opponent strength attributes."""
    ai_strength: float = 1.0
    ai_decision_strength: float = 1.0
    ai_precision_strength: float = 1.0

    def __post_init__(self):
        for attr in ["ai_strength", "ai_decision_strength", "ai_precision_strength"]:
            val = getattr(self, attr)
            assert val >= 0.0, f"{attr} must be non-negative"


@dataclass
class AsymmetryConfig:
    masks: ObservationMasks = field(default_factory=ObservationMasks)
    opponent: OpponentDifficulty = field(default_factory=OpponentDifficulty)

    def to_env_kwargs(self) -> dict:
        """Converts to the flat dict your VmasEnv still expects."""
        return {**asdict(self.masks), **asdict(self.opponent)}

    def label(self) -> str:
        """Generates a human-readable experiment label."""
        active_masks = [k for k, v in asdict(self.masks).items() 
            if (isinstance(v, bool) and v) or (isinstance(v, list) and any(v))]
        return "_".join(active_masks) if active_masks else "baseline"