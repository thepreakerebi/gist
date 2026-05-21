from pathlib import Path

from gist.cli import _clear_previous_clips


def test_clear_previous_clips_removes_stale_mp4_files_only(tmp_path: Path) -> None:
    clips = tmp_path / "clips"
    clips.mkdir()
    stale = clips / "old.mp4"
    keep = clips / "notes.txt"
    stale.write_bytes(b"old")
    keep.write_text("keep")

    _clear_previous_clips(clips)

    assert stale.exists() is False
    assert keep.exists() is True
