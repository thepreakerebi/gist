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
    saliency_strategy: str = "hash"

    @property
    def total_patches(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def retained_patches(self) -> int:
        return len(self.retained_patch_indexes)


def build_query_spatial_mask(
    evidence_id: str,
    query: str,
    evidence_text: str = "",
    grid_size: int = 14,
    retention_ratio: float = 0.35,
) -> SpatialMask:
    if grid_size <= 0:
        raise ValueError("grid_size must be greater than zero")
    if not 0 < retention_ratio <= 1:
        raise ValueError("retention_ratio must be between 0 and 1")

    total_patches = grid_size * grid_size
    retained_count = max(1, round(total_patches * retention_ratio))
    strategy = _saliency_strategy(query=query, evidence_text=evidence_text)
    ranked = sorted(
        range(total_patches),
        key=lambda patch_index: _patch_score(
            evidence_id=evidence_id,
            query=query,
            patch_index=patch_index,
            grid_size=grid_size,
            strategy=strategy,
        ),
        reverse=True,
    )
    return SpatialMask(
        evidence_id=evidence_id,
        query=query,
        grid_size=grid_size,
        retention_ratio=retention_ratio,
        retained_patch_indexes=sorted(ranked[:retained_count]),
        saliency_strategy=strategy,
    )


def write_spatial_mask(mask: SpatialMask, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mask.model_dump(), indent=2))
    return path


def write_spatial_mask_preview(mask: SpatialMask, path: Path, cell_size: int = 22) -> Path:
    if cell_size <= 0:
        raise ValueError("cell_size must be greater than zero")

    path.parent.mkdir(parents=True, exist_ok=True)
    retained = set(mask.retained_patch_indexes)
    size = mask.grid_size * cell_size
    cells = []
    for patch_index in range(mask.total_patches):
        row, column = divmod(patch_index, mask.grid_size)
        retained_patch = patch_index in retained
        fill = "#145c43" if retained_patch else "#e8eee9"
        stroke = "#d2ddd6"
        opacity = "0.92" if retained_patch else "0.55"
        cells.append(
            f'<rect x="{column * cell_size}" y="{row * cell_size}" '
            f'width="{cell_size}" height="{cell_size}" fill="{fill}" '
            f'fill-opacity="{opacity}" stroke="{stroke}" stroke-width="1" />'
        )

    title = _svg_escape(f"{mask.evidence_id} - {mask.saliency_strategy}")
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" role="img" aria-label="{title}">'
        f"<title>{title}</title>"
        f"<rect width='{size}' height='{size}' fill='#fbfcf9' />"
        + "".join(cells)
        + "</svg>\n"
    )
    path.write_text(svg)
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


def _saliency_strategy(query: str, evidence_text: str) -> str:
    normalized = f"{query} {evidence_text}".lower()
    if "on-screen text near" in normalized or any(
        term in normalized for term in {"text", "title", "caption", "slide", "screen says"}
    ):
        return "text_band"
    if any(
        term in normalized
        for term in {
            "show",
            "shown",
            "visible",
            "look",
            "see",
            "person",
            "hand",
            "object",
            "action",
            "scene",
        }
    ):
        return "center_object"
    return "hash"


def _patch_score(
    evidence_id: str,
    query: str,
    patch_index: int,
    grid_size: int,
    strategy: str,
) -> float:
    payload = f"{evidence_id}|{query}|{patch_index}".encode()
    noise = int.from_bytes(sha256(payload).digest()[:8], "big") / float(2**64 - 1)
    row, column = divmod(patch_index, grid_size)
    center_prior = _center_prior(row=row, column=column, grid_size=grid_size)
    text_band_prior = _text_band_prior(row=row, column=column, grid_size=grid_size)

    if strategy == "text_band":
        return (0.6 * text_band_prior) + (0.25 * center_prior) + (0.15 * noise)
    if strategy == "center_object":
        return (0.7 * center_prior) + (0.3 * noise)
    return noise


def _center_prior(row: int, column: int, grid_size: int) -> float:
    center = (grid_size - 1) / 2
    max_distance = (2 * (center**2)) ** 0.5 or 1.0
    distance = ((row - center) ** 2 + (column - center) ** 2) ** 0.5
    return max(0.0, 1.0 - (distance / max_distance))


def _text_band_prior(row: int, column: int, grid_size: int) -> float:
    center = (grid_size - 1) / 2
    row_distance = abs(row - center)
    column_prior = _center_prior(row=center, column=column, grid_size=grid_size)
    band = max(0.0, 1.0 - (row_distance / max(center, 1.0)))
    return (0.75 * band) + (0.25 * column_prior)


def _svg_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
