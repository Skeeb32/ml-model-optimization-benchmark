# When quantization makes sense

This file describes the decision framework. Numerical findings are generated from measured runs in `results/<run>/findings.md`; no expected speedup is presented as a result.

Adopt an optimization when it meets the workload's quality budget and improves a resource that matters. For an interactive feature, compare p95 latency at the intended batch size. For offline processing, compare samples/second under a defined memory budget. For artifact distribution, compare serialized bytes separately from runtime memory.

INT8 can reduce Linear-layer weight storage while leaving substantial FP32 state. Kernel support, packing overhead, matrix shapes, batch size, and thread count can erase or reverse the latency benefit. A smaller model may outperform a quantized larger model, but architecture and pretraining differ, so this is a deployment choice rather than a controlled parameter-count experiment.

For any published conclusion, include:

- Exact model revision, runtime versions, CPU architecture, thread count and input shape.
- Multiple independent trials with raw timings and variability.
- Separate artifact size, total sampled RSS, and lifecycle high-water RSS.
- Quality change on a representative held-out dataset, with an agreed acceptance budget.
- Explicit failures and unsupported combinations.

The included 24-sentence quality probe is useful for catching regressions, but insufficient for a product deployment decision. A next milestone is a pinned, licensed external evaluation dataset and bootstrap confidence intervals over independent examples.
