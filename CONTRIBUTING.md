# Contributing

Run `ruff check src tests`, `pytest -q`, and an offline smoke run before proposing changes. Changes to timing, memory sampling, quality evaluation, or model conversion must update the measurement contract and tests. Keep results attached to raw evidence and explicit environment information. Do not commit model weights, caches, credentials, or fabricated measurements.

Use a new output directory for each experiment. Report failures with the relevant worker log and runtime versions. If a new backend is unsupported, fail explicitly rather than silently falling back to a different precision or device.
