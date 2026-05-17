from dataclasses import dataclass
from enum import StrEnum


class CompressionPreset(StrEnum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


@dataclass(frozen=True, slots=True)
class PresetConfig:
    max_items: int
    relevance_weight: float
    temporal_sigma_seconds: float


PRESETS: dict[CompressionPreset, PresetConfig] = {
    CompressionPreset.CONSERVATIVE: PresetConfig(
        max_items=24,
        relevance_weight=0.78,
        temporal_sigma_seconds=18.0,
    ),
    CompressionPreset.BALANCED: PresetConfig(
        max_items=12,
        relevance_weight=0.72,
        temporal_sigma_seconds=14.0,
    ),
    CompressionPreset.AGGRESSIVE: PresetConfig(
        max_items=6,
        relevance_weight=0.66,
        temporal_sigma_seconds=10.0,
    ),
}

