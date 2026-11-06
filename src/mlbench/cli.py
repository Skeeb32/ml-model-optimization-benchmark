"""Experiment orchestration with immutable manifests and raw evidence."""

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import psutil
import yaml
from huggingface_hub import HfApi

from .inference import agreement


def validate(config):
    for key in ("threads", "warmup", "iterations", "trials"):
        value = config.get(key)
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("batch_sizes", "sequence_lengths"):
        values = config.get(key)
        if not values or any(type(x) is not int or x < 1 for x in values):
            raise ValueError(f"{key} must contain positive integers")
    if max(config["sequence_lengths"]) > 512:
        raise ValueError("BERT sequence length must not exceed 512")
    if not config.get("precisions") or not set(config["precisions"]) <= {"fp32", "int8"}:
        raise ValueError("precisions must contain fp32 and/or int8")
    if not config.get("models"):
        raise ValueError("At least one model is required")
    if type(config.get("seed")) is not int:
        raise ValueError("seed must be an integer")


def rows_from_results(results):
    references = {}
    for result in results:
        key = (
            result["model"]["id"],
            result["batch_size"],
            result["sequence_length"],
            result["trial"],
        )
        if result["status"] == "ok" and result["precision"] == "fp32":
            references[key] = result
    rows = []
    for result in results:
        key = (
            result["model"]["id"],
            result["batch_size"],
            result["sequence_length"],
            result["trial"],
        )
        row = {k: v for k, v in result.items() if not isinstance(v, (dict, list))}
        row.update(model_id=result["model"]["id"], revision=result["model"]["revision"])
        ref = references.get(key)
        if result["status"] == "ok" and ref and result["precision"] == "int8":
            row["latency_speedup_vs_fp32"] = ref["latency_mean_ms"] / result["latency_mean_ms"]
            row["state_size_reduction_pct"] = 100 * (
                1 - result["serialized_state_mib"] / ref["serialized_state_mib"]
            )
            if (
                result.get("quality_input_sha256") == ref.get("quality_input_sha256")
                and "quality_predictions" in result
            ):
                row["prediction_agreement_vs_fp32"] = agreement(
                    ref["quality_predictions"], result["quality_predictions"]
                )
                row["accuracy_delta_pp"] = 100 * (
                    result["masked_token_accuracy"] - ref["masked_token_accuracy"]
                )
        rows.append(row)
    return rows


def run(args):
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    if args.smoke:
        config.update(
            models=[{"id": "tiny-random-bert", "revision": "seeded-local"}],
            batch_sizes=[1],
            sequence_lengths=[32],
            trials=1,
            warmup=2,
            iterations=5,
        )
    validate(config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "raw").mkdir()
    if not args.smoke:
        api = HfApi()
        for spec in config["models"]:
            spec["revision"] = api.model_info(spec["id"], revision=spec["revision"]).sha
    corpus_path = (config_path.parent.parent / config["quality_corpus"]).resolve()
    corpus_bytes = corpus_path.read_bytes()
    texts = [line.strip() for line in corpus_bytes.decode().splitlines() if line.strip()]
    (out / "quality_corpus.txt").write_bytes(corpus_bytes)
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    packages = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "numpy", "pandas", "psutil", "huggingface-hub")
    }
    git_result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    manifest = {
        "schema_version": 1,
        "started_utc": datetime.now(UTC).isoformat(),
        "run_type": "smoke_random_weights" if args.smoke else "measured_pretrained",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(),
        "ram_gib": psutil.virtual_memory().total / 2**30,
        "device": "cpu",
        "packages": packages,
        "source_commit": git_result.stdout.strip() if git_result.returncode == 0 else None,
        "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "config": config,
        "quality_scope": "24 original sentences; diagnostic only",
        "timing_scope": "warm forward-only masked-LM logits; synthetic token IDs; no tokenization",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    jobs = list(
        itertools.product(
            config["models"],
            config["precisions"],
            config["batch_sizes"],
            config["sequence_lengths"],
            range(config["trials"]),
        )
    )
    random.Random(config["seed"]).shuffle(jobs)
    results = []
    for index, (model, precision, batch, length, trial) in enumerate(jobs):
        cell = {
            "model": model,
            "precision": precision,
            "batch_size": batch,
            "sequence_length": length,
            "trial": trial,
            "smoke": args.smoke,
            "threads": config["threads"],
            "seed": config["seed"],
            "warmup": config["warmup"],
            "iterations": config["iterations"],
            "engine": config.get("engine", "auto"),
            "texts": texts,
        }
        stem = out / "raw" / f"cell-{index:03d}"
        request_path, result_path = stem.with_suffix(".request.json"), stem.with_suffix(".json")
        request_path.write_text(json.dumps(cell))
        print(
            f"[{index + 1}/{len(jobs)}] {model['id']} {precision} B={batch} L={length} trial={trial}",
            flush=True,
        )
        env = {
            **os.environ,
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": str(config["threads"]),
        }
        with stem.with_suffix(".log").open("w") as log:
            try:
                completed = subprocess.run(
                    [sys.executable, "-m", "mlbench.worker", str(request_path), str(result_path)],
                    check=False,
                    stdout=log,
                    stderr=log,
                    env=env,
                    timeout=args.timeout,
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"Worker exit {completed.returncode}; see raw/{stem.name}.log"
                    )
                result = json.loads(result_path.read_text())
            except (subprocess.TimeoutExpired, RuntimeError) as exc:
                result = {k: v for k, v in cell.items() if k != "texts"}
                result.update(status="failed", error=str(exc))
                result_path.write_text(json.dumps(result, indent=2))
        request_path.unlink()
        results.append(result)
        pd.DataFrame(rows_from_results(results)).to_csv(out / "benchmark_results.csv", index=False)
    failures = sum(r["status"] != "ok" for r in results)
    manifest.update(
        completed_utc=datetime.now(UTC).isoformat(),
        failed_cells=failures,
        completed_cells=len(results) - failures,
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    from .report import report

    report(out)
    if failures:
        raise SystemExit(
            f"{failures} cells failed. Inspect raw logs; failures are not performance results."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    benchmark = commands.add_parser("run")
    benchmark.add_argument("--config", default="config/benchmark_config.yaml")
    benchmark.add_argument(
        "--output", required=True, help="New directory; existing runs are never overwritten"
    )
    benchmark.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny random BERT; verifies plumbing, not model quality",
    )
    benchmark.add_argument("--timeout", type=int, default=1800, help="Seconds per isolated cell")
    analysis = commands.add_parser("report")
    analysis.add_argument("run_directory")
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        from .report import report

        report(Path(args.run_directory))
