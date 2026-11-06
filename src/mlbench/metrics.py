"""Units and aggregation shared by workers and tests."""

import numpy as np


def latency_summary(seconds, batch_size):
    values = np.asarray(seconds, dtype=float)
    if values.size == 0 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Timings must be finite, positive, and nonempty")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return {
        "latency_mean_ms": float(values.mean() * 1000),
        "latency_p50_ms": float(np.percentile(values, 50) * 1000),
        "latency_p95_ms": float(np.percentile(values, 95) * 1000),
        "latency_p99_ms": float(np.percentile(values, 99) * 1000),
        "throughput_samples_s": float(batch_size * len(values) / values.sum()),
    }
