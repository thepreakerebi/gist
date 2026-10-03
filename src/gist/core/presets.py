"""The three evidence budgets.

**Hand-chosen for this project; not taken from any published configuration.**

The shape is principled: a smaller budget pairs with a lower ``relevance_weight``
(MMR's lambda), because with only a handful of slots you cannot afford two of them
to be near-duplicates, so diversity must count for more. A smaller budget likewise
pairs with a smaller ``temporal_sigma_seconds``, so "too close together" is judged
on a tighter scale.

The specific values were not swept, and no ablation varies lambda or sigma. The
ablations vary the input signal and the post-processing instead. Stated here so
the limitation is visible at the point of definition rather than only in the
write-up.
"""

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

