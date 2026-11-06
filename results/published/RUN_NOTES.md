# Run notes

This exploratory run completed 18/18 cells. The initial attempt had download and local-installation failures and was retained locally in `results/initial-failed-run`; none of those measurements are mixed into this run. The successful rerun used `HF_HUB_DISABLE_XET=1`.

The machine was an Apple M1 Mac with 16 GiB RAM. Power mode and thermal conditions were not controlled. Treat variation as observed workstation behavior, not a hardware specification. No battery, GPU, Neural Engine, Core ML, or MLX measurements were taken.

The interactive published dashboard uses a CDN script to keep the Git repository small. The original self-contained HTML remains in the local run directory, and `mlbench report` regenerates it.
