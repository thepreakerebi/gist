import shutil
import subprocess
import tempfile
from pathlib import Path

from gist.media.models import ExtractedFrame


class TesseractFrameOcr:
    """Extract on-screen text from sampled frames when Tesseract is available."""

    def __init__(self, executable: str = "tesseract", timeout_seconds: float = 8.0) -> None:
        self.executable = executable
        self.timeout_seconds = timeout_seconds

    def extract_text(self, frames: list[ExtractedFrame]) -> dict[Path, str]:
        if not frames or shutil.which(self.executable) is None:
            return {}

        results: dict[Path, str] = {}
        for frame in frames:
            if not frame.path.exists() or not frame.path.is_file():
                continue
            text = self._extract_frame_text(frame.path)
            if text:
                results[frame.path] = text
        return results

    def _extract_frame_text(self, frame_path: Path) -> str:
        text = self._run_tesseract(frame_path, page_segmentation_mode="6")
        if _ocr_information_score(text) >= 2:
            return text

        fallback = self._extract_projected_content(frame_path)
        return max((text, fallback), key=_ocr_information_score)

    def _extract_projected_content(self, frame_path: Path) -> str:
        try:
            from PIL import Image, ImageEnhance, ImageFilter

            with Image.open(frame_path) as image:
                width, height = image.size
                crop = image.crop(
                    (
                        int(width * 0.15),
                        0,
                        int(width * 0.85),
                        int(height * 0.82),
                    )
                )
                crop = crop.resize(
                    (crop.width * 3, crop.height * 3),
                    Image.Resampling.LANCZOS,
                )
                crop = ImageEnhance.Contrast(crop).enhance(1.5)
                crop = crop.filter(ImageFilter.SHARPEN)
                with tempfile.NamedTemporaryFile(suffix=".png") as output:
                    crop.save(output.name)
                    return self._run_tesseract(
                        Path(output.name),
                        page_segmentation_mode="11",
                    )
        except (ImportError, OSError):
            return ""

    def _run_tesseract(
        self,
        frame_path: Path,
        page_segmentation_mode: str,
    ) -> str:
        try:
            completed = subprocess.run(
                [
                    self.executable,
                    str(frame_path),
                    "stdout",
                    "--psm",
                    page_segmentation_mode,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""

        if completed.returncode != 0:
            return ""
        return _normalize_ocr_text(completed.stdout)


def _normalize_ocr_text(value: str) -> str:
    return " ".join(value.replace("\x0c", " ").split())


def _ocr_information_score(value: str) -> int:
    return sum(
        1
        for token in value.split()
        if len("".join(character for character in token if character.isalpha())) >= 3
    )
