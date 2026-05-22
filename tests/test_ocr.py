from gist.vision.ocr import _normalize_ocr_text


def test_normalize_ocr_text_removes_form_feed_and_extra_space() -> None:
    assert _normalize_ocr_text(" hello\n\nworld \x0c ") == "hello world"
