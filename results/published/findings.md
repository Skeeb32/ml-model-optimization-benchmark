# CPU inference benchmark: measured findings

Error bars: sample standard deviation across independent process trials; zero for one trial. RSS is total process memory, not model-only memory. Quality uses a small diagnostic corpus. Latency excludes tokenization, loading, and network time. These are encoder models, not text-generation LLMs.

Successful cells: 18. Failed cells: 0.

## Paired FP32 / INT8 comparisons

Values below are means of matched trial ratios. Above 1× means INT8 is faster.

- **bert-base-uncased, B=1 L=32:** 0.89× latency speedup; 53.2% smaller serialized state.
  Diagnostic accuracy change: -2.17 percentage points; prediction agreement: 82.6%.
- **bert-large-uncased, B=1 L=32:** 1.24× latency speedup; 65.5% smaller serialized state.
  Diagnostic accuracy change: -4.35 percentage points; prediction agreement: 84.8%.
- **distilbert-base-uncased, B=1 L=32:** 0.89× latency speedup; 39.4% smaller serialized state.
  Diagnostic accuracy change: -2.17 percentage points; prediction agreement: 71.7%.

## Interpretation

Choose INT8 only if measured latency, memory, and task-quality budgets improve for the target workload. Reduced artifact size does not guarantee lower RSS or faster inference. Cross-model comparisons also change architecture and pretraining; they do not isolate parameter count. Do not generalize these results to Apple GPU, Neural Engine, Core ML, or MLX.

See manifest.json, resolved_config.yaml, quality_corpus.txt and raw/*.json for provenance.
