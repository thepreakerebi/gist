import json

import pytest

from gist.eval.profiling import (
    DeviceInfo,
    MeasurementLog,
    StageMeasurement,
    is_out_of_memory,
    load_measurements,
    measure,
)


def test_measure_times_the_block_and_yields_a_mutable_record():
    with measure("answer", condition="gist", qid="q1") as record:
        record.extra["frames"] = 3
    assert record.stage == "answer"
    assert record.wall_seconds >= 0
    assert record.ok
    assert record.extra == {"condition": "gist", "qid": "q1", "frames": 3}


def test_memory_fields_are_none_without_cuda_rather_than_zero():
    """A missing measurement must not masquerade as a measured zero."""
    with measure("answer") as record:
        pass
    if record.peak_allocated_gb is None:
        assert record.activation_gb is None
        assert record.resident_gb is None


def test_oom_is_recorded_as_data_and_suppressed():
    with measure("answer", condition="dense") as record:
        raise RuntimeError("CUDA out of memory. Tried to allocate 16.80 GiB")
    assert record.oom is True
    assert not record.ok
    assert "out of memory" in record.error.lower()


def test_non_oom_errors_still_propagate_after_being_recorded():
    with pytest.raises(ValueError), measure("answer") as record:
        raise ValueError("bad frame path")
    assert record.oom is False
    assert "ValueError" in record.error


def test_oom_can_be_made_fatal_when_the_caller_wants_it():
    with pytest.raises(RuntimeError), measure("answer", suppress_oom=False):
        raise RuntimeError("CUDA out of memory")


def test_is_out_of_memory_matches_the_bare_runtime_error_form():
    assert is_out_of_memory(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert not is_out_of_memory(ValueError("missing option D"))


def test_log_round_trips_through_jsonl_with_the_device_header(tmp_path):
    path = tmp_path / "m.jsonl"
    log = MeasurementLog(path, device=DeviceInfo(available=True, name="RTX 4090",
                                                 total_memory_gb=23.99, torch_version="2.6.0"))
    with log.measure("answer", condition="gist", qid="q1"):
        pass
    with log.measure("answer", condition="dense", qid="q1"):
        raise RuntimeError("CUDA out of memory")

    device, stages = load_measurements(path)
    assert device.name == "RTX 4090"
    assert [s.extra["condition"] for s in stages] == ["gist", "dense"]
    assert stages[1].oom is True
    # the header is a discriminated record, not a stage
    first = json.loads(path.read_text().splitlines()[0])
    assert first["record"] == "device"


def test_device_describe_is_explicit_about_having_no_gpu():
    assert "no CUDA device" in DeviceInfo(available=False).describe()


def test_stage_measurement_json_round_trip():
    row = StageMeasurement(stage="answer", wall_seconds=1.5, extra={"condition": "gist"})
    assert StageMeasurement.from_json(row.to_json()) == row


def test_log_persists_the_row_even_when_the_error_propagates(tmp_path):
    """Regression: an outer context manager persisted the row too early.

    Wrapping ``measure`` in a second context manager put the append in a
    ``finally`` that fires while the exception is still travelling through the
    inner generator, so the row reached disk before the timings and the error
    had been written to it. The sink callback exists to fix that.
    """
    path = tmp_path / "m.jsonl"
    log = MeasurementLog(path, device=DeviceInfo(available=False))
    with pytest.raises(ValueError), log.measure("answer", condition="gist"):
        sum(range(200_000))
        raise ValueError("bad frame path")

    _, stages = load_measurements(path)
    assert len(stages) == 1
    assert "ValueError" in stages[0].error
    assert stages[0].wall_seconds > 0  # timed, not a bare pre-finalisation row


def test_sink_fires_once_on_the_success_path_too():
    seen = []
    with measure("answer", sink=seen.append, condition="gist"):
        pass
    assert len(seen) == 1
    assert seen[0].extra["condition"] == "gist"
