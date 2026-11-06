"""Static export and standalone interactive HTML, generated only from measured rows."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

METRICS = {
    "latency_mean_ms": "Mean batch latency (ms) ↓",
    "latency_p95_ms": "p95 batch latency (ms) ↓",
    "throughput_samples_s": "Throughput (samples/s) ↑",
    "sampled_inference_peak_rss_mib": "Sampled process RSS peak (MiB) ↓",
    "serialized_state_mib": "Serialized state (MiB) ↓",
    "masked_token_accuracy": "Diagnostic masked-token accuracy ↑",
}


def report(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    frame = pd.read_csv(directory / "benchmark_results.csv")
    good = frame[frame.status == "ok"].copy()
    if good.empty:
        (directory / "findings.md").write_text(
            "# No successful measurements\n\nInspect raw worker logs.\n"
        )
        return
    good["workload"] = "B=" + good.batch_size.astype(str) + " L=" + good.sequence_length.astype(str)
    good["model"] = good.model_id.str.split("/").str[-1]
    plots = directory / "plots"
    plots.mkdir(exist_ok=True)
    title = (
        "SMOKE TEST — RANDOM WEIGHTS"
        if manifest["run_type"].startswith("smoke")
        else "CPU inference benchmark"
    )
    sections = []
    for metric, label in METRICS.items():
        if metric not in good:
            continue
        grouped = (
            good.groupby(["model", "precision", "workload"], sort=False)[metric]
            .agg(["mean", "std"])
            .reset_index()
        )
        grouped["std"] = grouped["std"].fillna(0)
        fig = px.bar(
            grouped,
            x="model",
            y="mean",
            color="precision",
            barmode="group",
            facet_col="workload",
            error_y="std",
            title=label,
            labels={"mean": label},
            color_discrete_map={"fp32": "#64748b", "int8": "#0d9488"},
        )
        fig.update_layout(template="plotly_white", height=420)
        sections.append(
            fig.to_html(full_html=False, include_plotlyjs=True if not sections else False)
        )
        workloads = list(grouped.workload.unique())
        figure, axes = plt.subplots(
            1, len(workloads), figsize=(6 * len(workloads), 4), squeeze=False
        )
        for ax, workload in zip(axes[0], workloads):
            subset = grouped[grouped.workload == workload]
            pivot = subset.pivot(index="model", columns="precision", values="mean")
            errors = subset.pivot(index="model", columns="precision", values="std")
            pivot.plot.bar(ax=ax, yerr=errors, color=["#64748b", "#0d9488"], rot=20, capsize=3)
            ax.set_title(workload)
            ax.set_ylabel(label)
            ax.set_xlabel("")
        figure.suptitle(f"{title} · {label}")
        figure.tight_layout()
        figure.savefig(plots / f"{metric}.svg", bbox_inches="tight")
        plt.close(figure)
    note = (
        "Error bars: sample standard deviation across independent process trials; zero for one trial. "
        "RSS is total process memory, not model-only memory. Quality uses a small diagnostic corpus. "
        "Latency excludes tokenization, loading, and network time. These are encoder models, not text-generation LLMs."
    )
    (directory / "dashboard.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{title}</title><style>body{{font:16px system-ui;margin:40px auto;max-width:1500px;padding:0 24px;color:#172033}}"
        "p{max-width:1000px;line-height:1.6}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}</style>"
        f"<h1>{title}</h1><p>{note}</p>"
        + "".join(sections)
        + "<h2>Raw trial results</h2>"
        + good.to_html(index=False, escape=True)
        + "</html>"
    )
    lines = [
        f"# {title}: measured findings",
        "",
        note,
        "",
        f"Successful cells: {len(good)}. Failed cells: {len(frame) - len(good)}.",
        "",
        "## Paired FP32 / INT8 comparisons",
        "",
        "Values below are means of matched trial ratios. Above 1× means INT8 is faster.",
        "",
    ]
    paired = good[good.precision == "int8"]
    for (model, workload), rows in paired.groupby(["model", "workload"]):
        if "latency_speedup_vs_fp32" not in rows or rows.latency_speedup_vs_fp32.isna().all():
            continue
        lines.append(
            f"- **{model}, {workload}:** {rows.latency_speedup_vs_fp32.mean():.2f}× latency speedup; "
            f"{rows.state_size_reduction_pct.mean():.1f}% smaller serialized state."
        )
        if "accuracy_delta_pp" in rows:
            lines.append(
                f"  Diagnostic accuracy change: {rows.accuracy_delta_pp.mean():+.2f} percentage points; "
                f"prediction agreement: {rows.prediction_agreement_vs_fp32.mean():.1%}."
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Choose INT8 only if measured latency, memory, and task-quality budgets improve for the target workload. "
            "Reduced artifact size does not guarantee lower RSS or faster inference. "
            "Cross-model comparisons also change architecture and pretraining; they do not isolate parameter count. "
            "Do not generalize these results to Apple GPU, Neural Engine, Core ML, or MLX.",
            "",
            "See manifest.json, resolved_config.yaml, quality_corpus.txt and raw/*.json for provenance.",
        ]
    )
    (directory / "findings.md").write_text("\n".join(lines) + "\n")
