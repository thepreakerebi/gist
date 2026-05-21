import math

from gist.media.longform import ProcessingMode, plan_ingestion


def test_auto_plan_uses_long_mode_for_one_hour_plus_video() -> None:
    plan = plan_ingestion(duration_seconds=90 * 60, mode=ProcessingMode.AUTO)

    assert plan.mode == ProcessingMode.LONG
    assert plan.sample_count == 512
    assert plan.audio_window_seconds >= 30.0
    assert plan.max_audio_windows <= 240
    assert plan.audio_context_window_count == 0


def test_long_plan_increases_audio_window_to_bound_file_count() -> None:
    duration_seconds = 4 * 60 * 60

    plan = plan_ingestion(duration_seconds=duration_seconds, mode=ProcessingMode.LONG)

    assert plan.audio_window_seconds == math.ceil(duration_seconds / 240)
    assert plan.max_audio_windows == 240


def test_manual_overrides_are_respected_when_safe() -> None:
    plan = plan_ingestion(
        duration_seconds=20 * 60,
        mode=ProcessingMode.MEDIUM,
        sample_count=42,
        audio_window_seconds=12.0,
    )

    assert plan.mode == ProcessingMode.MEDIUM
    assert plan.sample_count == 42
    assert plan.audio_window_seconds == 12.0
