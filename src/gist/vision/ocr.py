from pathlib import Path
import shutil
import subprocess

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
        try:
            completed = subprocess.run(
                [
                    self.executable,
                    str(frame_path),
                    "stdout",
                    "--psm",
                    "6",
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
