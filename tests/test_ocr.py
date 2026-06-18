from pathlib import Path

from gist.vision.ocr import TesseractFrameOcr, _normalize_ocr_text


def test_normalize_ocr_text_removes_form_feed_and_extra_space() -> None:
    assert _normalize_ocr_text(" hello\n\nworld \x0c ") == "hello world"


def test_sparse_ocr_uses_projected_content_fallback(monkeypatch) -> None:
    ocr = TesseractFrameOcr()
    monkeypatch.setattr(ocr, "_run_tesseract", lambda *_args, **_kwargs: "J")
    monkeypatch.setattr(
        ocr,
        "_extract_projected_content",
        lambda _path: "KINECT for Windows",
    )

    assert ocr._extract_frame_text(Path("frame.jpg")) == "KINECT for Windows"
