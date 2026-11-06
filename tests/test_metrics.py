import pytest

from mlbench.cli import rows_from_results, validate
from mlbench.inference import agreement, make_inputs
from mlbench.metrics import latency_summary


def test_throughput_uses_total_samples_over_total_time():
    result = latency_summary([0.01, 0.03], 8)
    assert result["throughput_samples_s"] == pytest.approx(400)
    assert result["latency_mean_ms"] == pytest.approx(20)
    assert result["latency_p95_ms"] == pytest.approx(29)


@pytest.mark.parametrize("timings", [[], [0], [-1], [float("nan")], [float("inf")]])
def test_invalid_timings_rejected(timings):
    with pytest.raises(ValueError):
        latency_summary(timings, 1)


def test_seeded_inputs_identical_across_precision():
    assert make_inputs(128, 2, 16, 42)["input_ids"].equal(make_inputs(128, 2, 16, 42)["input_ids"])
    assert not make_inputs(128, 2, 16, 42)["input_ids"].equal(
        make_inputs(128, 2, 16, 43)["input_ids"]
    )


def test_prediction_agreement_is_not_accuracy():
    assert agreement([1, 2, 3], [1, 4, 3]) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        agreement([1], [1, 2])


def test_bad_config_fails_before_download():
    with pytest.raises(ValueError, match="threads"):
        validate({"threads": 0})


def test_pairing_excludes_failed_cells_and_matches_workload():
    common = dict(
        model={"id": "test", "revision": "sha"},
        trial=0,
        batch_size=1,
        sequence_length=32,
        status="ok",
    )
    fp = dict(common, precision="fp32", latency_mean_ms=20, serialized_state_mib=100)
    quant = dict(common, precision="int8", latency_mean_ms=10, serialized_state_mib=40)
    failed = dict(common, precision="int8", status="failed", sequence_length=128)
    rows = rows_from_results([quant, failed, fp])
    assert rows[0]["latency_speedup_vs_fp32"] == 2
    assert rows[0]["state_size_reduction_pct"] == 60
    assert "latency_speedup_vs_fp32" not in rows[1]
