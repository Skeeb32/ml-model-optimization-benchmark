"""One process per experiment cell; no cross-model allocator reuse."""

import gc
import json
import resource
import sys
import tempfile
import threading
import time
from pathlib import Path

import psutil
import torch

from .inference import make_inputs, quality_probe, time_forward
from .metrics import latency_summary
from .models import load_model
from .quantization import quantize


def run(request):
    torch.set_num_threads(request["threads"])
    torch.set_num_interop_threads(1)
    torch.manual_seed(request["seed"])
    process = psutil.Process()
    baseline = process.memory_info().rss
    model, tokenizer = load_model(request["model"], request["smoke"])
    parameters = sum(p.numel() for p in model.parameters())
    engine, quantized_layers = "none", 0
    if request["precision"] == "int8":
        model, engine, quantized_layers = quantize(model, request["engine"])
    with tempfile.TemporaryDirectory() as directory:
        artifact = Path(directory) / "state.pt"
        torch.save(model.state_dict(), artifact)
        artifact_bytes = artifact.stat().st_size
    inputs = make_inputs(
        model.config.vocab_size, request["batch_size"], request["sequence_length"], request["seed"]
    )
    gc.collect()
    timings = time_forward(model, inputs, request["warmup"], request["iterations"])
    # Separate memory pass avoids polling-thread interference with timed latency.
    samples = [process.memory_info().rss]
    stop = threading.Event()

    def poll():
        while not stop.wait(0.005):
            samples.append(process.memory_info().rss)

    monitor = threading.Thread(target=poll, daemon=True)
    monitor.start()
    with torch.inference_mode():
        for _ in range(max(5, request["iterations"])):
            output = model(**inputs)
            samples.append(process.memory_info().rss)
            del output
    stop.set()
    monitor.join()
    highwater = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    highwater_bytes = highwater if sys.platform == "darwin" else highwater * 1024
    result = {
        **request,
        **latency_summary(timings, request["batch_size"]),
        "status": "ok",
        "engine_used": engine,
        "quantized_linear_layers": quantized_layers,
        "original_parameters": parameters,
        "serialized_state_mib": artifact_bytes / 2**20,
        "baseline_rss_mib": baseline / 2**20,
        "sampled_inference_peak_rss_mib": max(samples) / 2**20,
        "process_highwater_before_quality_mib": highwater_bytes / 2**20,
        "raw_latency_seconds": timings,
    }
    if not request["smoke"]:
        result.update(quality_probe(model, tokenizer, request["texts"], request["seed"]))
    result.pop("texts", None)
    return result


if __name__ == "__main__":
    request_path, result_path = map(Path, sys.argv[1:])
    request = json.loads(request_path.read_text())
    started = time.time()
    result = run(request)
    result["worker_elapsed_s"] = time.time() - started
    result_path.write_text(json.dumps(result, indent=2) + "\n")
