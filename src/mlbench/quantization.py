"""CPU dynamic INT8 Linear layers. Embeddings and layer norms remain FP32."""

import torch


def choose_engine(requested="auto"):
    supported = torch.backends.quantized.supported_engines
    choices = [requested] if requested != "auto" else ["x86", "fbgemm", "qnnpack"]
    for engine in choices:
        if engine in supported and engine != "none":
            torch.backends.quantized.engine = engine
            return engine
    raise RuntimeError(f"No supported INT8 engine for {requested!r}; available={supported}")


def quantize(model, engine="auto"):
    selected = choose_engine(engine)
    # Kept as an explicit legacy baseline; migration to torchao is documented.
    from torch.ao.quantization import quantize_dynamic

    quantized = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    count = sum(isinstance(m, torch.ao.nn.quantized.dynamic.Linear) for m in quantized.modules())
    if count == 0:
        raise RuntimeError("No Linear layers were quantized; refusing a mislabeled INT8 result")
    return quantized, selected, count
