from hashlib import sha256
import json
from pathlib import Path

from pydantic import BaseModel, Field


class SpatialMask(BaseModel):
    evidence_id: str
    query: str
    grid_size: int = Field(gt=0)
    retention_ratio: float = Field(gt=0, le=1)
    retained_patch_indexes: list[int]

    @property
    def total_patches(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def retained_patches(self) -> int:
        return len(self.retained_patch_indexes)


def build_query_spatial_mask(
    evidence_id: str,
    query: str,
    grid_size: int = 14,
    retention_ratio: float = 0.35,
) -> SpatialMask:
    if grid_size <= 0:
        raise ValueError("grid_size must be greater than zero")
    if not 0 < retention_ratio <= 1:
        raise ValueError("retention_ratio must be between 0 and 1")

    total_patches = grid_size * grid_size
    retained_count = max(1, round(total_patches * retention_ratio))
    ranked = sorted(
        range(total_patches),
        key=lambda patch_index: _patch_score(evidence_id, query, patch_index),
        reverse=True,
    )
    return SpatialMask(
        evidence_id=evidence_id,
        query=query,
        grid_size=grid_size,
        retention_ratio=retention_ratio,
        retained_patch_indexes=sorted(ranked[:retained_count]),
    )


def write_spatial_mask(mask: SpatialMask, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mask.model_dump(), indent=2))
    return path


def estimate_spatial_tokens(
    selected_visual_count: int,
    grid_size: int = 14,
    retention_ratio: float = 0.35,
) -> tuple[int, int, float]:
    if selected_visual_count < 0:
        raise ValueError("selected_visual_count must be non-negative")
    if grid_size <= 0:
        raise ValueError("grid_size must be greater than zero")
    if not 0 < retention_ratio <= 1:
        raise ValueError("retention_ratio must be between 0 and 1")

    baseline_tokens = selected_visual_count * grid_size * grid_size
    retained_tokens = round(baseline_tokens * retention_ratio)
    reduction_percent = (
        0.0 if baseline_tokens == 0 else (1 - (retained_tokens / baseline_tokens)) * 100
    )
    return baseline_tokens, retained_tokens, reduction_percent


def _patch_score(evidence_id: str, query: str, patch_index: int) -> int:
    payload = f"{evidence_id}|{query}|{patch_index}".encode()
    return int.from_bytes(sha256(payload).digest()[:8], "big")
