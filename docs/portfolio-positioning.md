# Portfolio positioning

This repository demonstrates CPU inference benchmarking, model optimization, reproducibility, and performance analysis. It does **not** establish experience implementing Core ML applications, MLX inference, hybrid cloud routing, or battery optimization.

A truthful résumé entry after running and understanding the project:

> Built a reproducible PyTorch benchmark comparing DistilBERT, BERT Base, and BERT Large under FP32 and dynamic INT8 CPU inference, with isolated trials, latency distributions, memory measurements, diagnostic prediction-quality evaluation, and interactive reports.

Only add measured numerical improvements with their model, workload, hardware, and quality qualifications. A slowdown is a useful engineering finding, not a failed portfolio project.

## Complementary Apple projects

| Repository | Evidence to implement | Acceptance criteria |
| --- | --- | --- |
| `on-device-ml-ios` | Swift app with Core ML model integration, input preprocessing, inference and result presentation | Runs on a named device; demo; latency/memory measurements; offline behavior verified |
| `apple-silicon-mlx-benchmarks` | MLX experiments comparing model sizes/precision with synchronization and repeatable measurement | Pinned models; real local runs; latency, throughput and memory data; quality checks |
| `hybrid-inference-system` | Local/cloud routing prototype with explicit privacy policy and fallback behavior | Sensitive requests stay local; cloud failure handled; routing tests; architecture diagram |

Separate repositories make independently runnable projects easier to review. They are not a requirement for résumé entries. A design document supports “designed”; a tested implementation supports “built”; measurements support performance claims. Do not list all three as completed based on repository scaffolding.
