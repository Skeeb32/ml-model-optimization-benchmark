import pytest
import torch

from mlbench.inference import make_inputs
from mlbench.models import load_model
from mlbench.quantization import quantize


def test_real_quantized_model_runs_and_changes_linear_layers():
    torch.manual_seed(42)
    model, _ = load_model({}, smoke=True)
    inputs = make_inputs(128, 1, 16, 42)
    with torch.inference_mode():
        baseline = model(**inputs).logits
    model, engine, count = quantize(model)
    with torch.inference_mode():
        optimized = model(**inputs).logits
    assert engine != "none"
    assert count > 0
    assert optimized.shape == baseline.shape
    assert torch.isfinite(optimized).all()
    assert (baseline - optimized).abs().max() < 0.1


def test_invalid_backend_fails_explicitly():
    model, _ = load_model({}, smoke=True)
    with pytest.raises(RuntimeError, match="No supported"):
        quantize(model, "unsupported-device")
