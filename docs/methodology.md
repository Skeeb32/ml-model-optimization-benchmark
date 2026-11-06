# Measurement contract

## Scope

The suite benchmarks **masked language modeling on CPU** using pretrained DistilBERT, BERT Base, and BERT Large. All use their pretrained masked-LM heads. No random classification head is attached to a pretrained encoder. These models do not generate chat responses: time-to-first-token and decoding tokens/second are not applicable.

Each model/precision/batch/length/trial cell runs sequentially in a new process. Cell order is seeded and shuffled. Models resolve to immutable Hugging Face revisions before workers start. Worker results, individual latency samples, package versions, thread counts, corpus hash, resolved configuration, and failures are saved. Existing output directories are rejected.

## Timing

- CPU only, FP32 vs dynamic INT8 Linear layers; identical eager attention path.
- Evaluation mode and inference mode; configured intra-op threads, one inter-op thread.
- Pre-generated integer token inputs with full attention masks, fixed batch and sequence dimensions. These are synthetic workloads, not production traffic.
- Warmups excluded; `perf_counter_ns` measures completed synchronous CPU forward passes, including masked-LM logits for every position.
- Tokenization, downloads, loading, quantization, serialization, quality evaluation, and memory polling excluded.
- p50/p95/p99 are NumPy linearly interpolated sample quantiles per trial. Small sample counts make tail estimates unstable.
- Throughput = batch size × timed iterations / sum of timed forward-pass durations. This is single-worker offline throughput, not concurrent service capacity.
- Charts display the mean of each trial's metric and sample standard deviation across trials, not confidence intervals. One trial has no estimated variation.

Use at least three trials and 50 iterations for exploratory runs. For tail-latency claims use hundreds or thousands of iterations, stable thermal conditions, and repeated sessions. Record power mode and background workloads in a run note. No CPU affinity, thermal control, or energy measurement is implemented.

## Memory

Three different quantities are intentionally separate:

1. **Serialized state MiB:** actual `torch.save(state_dict)` file bytes. Dynamic INT8 does not quantize embeddings or layer norms; packed weights and metadata affect size. Shared/tied FP32 weights may become independently packed. This is not a deployable Core ML artifact.
2. **Sampled inference peak RSS MiB:** total process RSS during a separate, post-warmup memory pass, sampled every 5 ms and after each forward. Native tensor allocations are included. Short peaks may be missed. Runtime, inputs, outputs, retained allocator pages and model state all contribute.
3. **Process high-water RSS before quality:** OS high-water mark through imports, loading, conversion, serialization, timing and memory measurement. On macOS `ru_maxrss` is bytes; on Linux it is KiB. This is not an inference-only peak.

A fresh worker avoids previous model cells contaminating the allocator. Conversion still loads FP32 first and can leave retained pages; do not call RSS differences pure model memory savings. Memory polling runs separately to avoid perturbing timed latency.

## Quantization

Dynamic INT8 replaces supported `torch.nn.Linear` layers with quantized dynamic Linear modules. The tool records the selected CPU engine and converted layer count and fails if no layers convert. Activations are dynamically quantized inside those operations; other operators remain floating point. FP32 and INT8 are not necessarily faster/slower in a fixed order.

This deliberately implements the established `torch.ao.quantization.quantize_dynamic` baseline. PyTorch is migrating quantization development to torchao; the legacy baseline is version locked in the tested environment. A torchao backend should be a separately named experiment, not silently substituted. An unavailable engine produces an explicit failure, never FP32 labeled as INT8.

## Prediction quality

`data/quality_corpus.txt` contains 24 original project-authored sentences. Each document has a seeded 15% subset (at least one) of non-special tokens replaced by `[MASK]`. The suite reports masked-token top-1 accuracy, mean negative log likelihood, and INT8 agreement with the paired FP32 model. Inputs and targets are hashed; agreement is only calculated for matching hashes.

This is a **small diagnostic probe**, not GLUE, a held-out natural-language benchmark, or a measure of application accuracy. Plausible alternative words can be scored wrong. Each sentence is capped at 128 tokens; quality evaluation is independent of the synthetic timing shape. Quality repetition across cells does not increase the number of independent evaluation examples. Cross-model accuracy is only descriptive and does not isolate model size from architecture/training differences.

## Hardware interpretation

Running this suite on an Apple silicon Mac measures **CPU** execution. It does not measure MPS GPU, Metal, Apple Neural Engine, MLX, or Core ML. CPU dynamic INT8 kernels are not portable proxies for those runtimes. A future Core ML or MLX comparison needs its own conversion validation, synchronization, memory instrumentation, and precision support matrix.
