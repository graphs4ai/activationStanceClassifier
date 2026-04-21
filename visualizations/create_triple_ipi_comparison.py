#!/usr/bin/env python3
"""
Create the same composite representation from real W&B artifacts.

This reproduces the visualization style from create_composite_demo.py, but uses
question-level PI data extracted from two comparison artifacts:
1) baseline + maximization
2) baseline + minimization
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator
from scipy import stats
import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build composite Base/Max/Min boxplot from two W&B comparison artifacts."
        )
    )
    parser.add_argument(
        "--max-artifact",
        required=True,
        help=(
            "W&B artifact reference or URL for baseline+maximization artifact "
            "(e.g. entity/project/name:v3 or https://wandb.ai/...)."
        ),
    )
    parser.add_argument(
        "--min-artifact",
        required=True,
        help=(
            "W&B artifact reference or URL for baseline+minimization artifact "
            "(e.g. entity/project/name:v4 or https://wandb.ai/...)."
        ),
    )
    parser.add_argument(
        "--baseline-source",
        choices=["max", "min"],
        default="max",
        help="Which artifact baseline to use in the final composite output.",
    )
    parser.add_argument(
        "--output-dir",
        default="comparison_results_wandb",
        help="Directory to store generated plot and summary JSON.",
    )
    parser.add_argument(
        "--plot-name",
        default="composite_boxplot_wandb.png",
        help="Output filename for the composite plot.",
    )
    parser.add_argument(
        "--summary-name",
        default="composite_summary_wandb.json",
        help="Output filename for the summary JSON.",
    )
    return parser.parse_args()


def normalize_artifact_ref(artifact_ref: str) -> str:
    """Accept either canonical W&B artifact ref or W&B artifact URL."""
    ref = artifact_ref.strip()
    if "wandb.ai" not in ref:
        return ref

    # URL forms supported:
    # https://wandb.ai/<entity>/<project>/artifacts/<type>/<name>/<version>
    # https://wandb.ai/<entity>/<project>/artifacts/<name>/<version>
    match = re.search(
        r"wandb\.ai/([^/]+)/([^/]+)/artifacts(?:/[^/]+)?/([^/]+)/([^/?#]+)",
        ref,
    )
    if not match:
        raise ValueError(
            f"Could not parse artifact URL: {artifact_ref}. "
            "Use entity/project/name:vN format or a standard wandb.ai artifact URL."
        )

    entity, project, artifact_name, version = match.groups()
    version = version if version.startswith("v") else f"v{version}"
    return f"{entity}/{project}/{artifact_name}:{version}"


def _pick_single_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Could not find files matching '{pattern}' in {directory}"
        )
    return matches[-1]


def load_artifact_data(artifact_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load baseline and intervention PI arrays from one downloaded artifact directory.

    Expected structure matches runs like:
    - baseline_pair_results_*.csv
    - intervention_pair_results_*.csv
    """
    baseline_pairs = _pick_single_file(
        artifact_dir, "baseline_pair_results_*.csv")
    intervention_pairs = _pick_single_file(
        artifact_dir, "intervention_pair_results_*.csv"
    )

    baseline_df = pd.read_csv(baseline_pairs)
    intervention_df = pd.read_csv(intervention_pairs)

    if "polarization_index" not in baseline_df.columns:
        raise ValueError(
            f"Column 'polarization_index' not found in {baseline_pairs}"
        )
    if "polarization_index" not in intervention_df.columns:
        raise ValueError(
            f"Column 'polarization_index' not found in {intervention_pairs}"
        )

    baseline_valid = baseline_df[baseline_df["polarization_index"].notna()]
    intervention_valid = intervention_df[intervention_df["polarization_index"].notna(
    )]

    if "valid" in baseline_valid.columns:
        baseline_valid = baseline_valid[baseline_valid["valid"].astype(
            str) == "True"]
    if "valid" in intervention_valid.columns:
        intervention_valid = intervention_valid[
            intervention_valid["valid"].astype(str) == "True"
        ]

    baseline_pis = baseline_valid["polarization_index"].to_numpy(dtype=float)
    intervention_pis = intervention_valid["polarization_index"].to_numpy(
        dtype=float)

    if len(baseline_pis) == 0 or len(intervention_pis) == 0:
        raise ValueError(
            f"Artifact directory {artifact_dir} has empty baseline/intervention PI data."
        )

    return baseline_pis, intervention_pis


def compute_comparison_stats(
    baseline_pis: np.ndarray, other_pis: np.ndarray
) -> Tuple[float, float, str, float]:
    n_baseline = len(baseline_pis)
    n_other = len(other_pis)

    if n_baseline == n_other and n_baseline > 0:
        try:
            result = stats.wilcoxon(baseline_pis, other_pis)
            result_vals = np.asarray(result, dtype=float).ravel()
            test_stat = float(result_vals[0])
            p_value = float(result_vals[1])
            test_type = "Wilcoxon"
        except ValueError:
            result = stats.mannwhitneyu(
                baseline_pis, other_pis, alternative="two-sided")
            test_stat = float(result.statistic)
            p_value = float(result.pvalue)
            test_type = "Mann-Whitney U"
    else:
        result = stats.mannwhitneyu(
            baseline_pis, other_pis, alternative="two-sided")
        test_stat = float(result.statistic)
        p_value = float(result.pvalue)
        test_type = "Mann-Whitney U"

    baseline_std = float(np.std(baseline_pis))
    effect_size = (
        (float(np.mean(other_pis)) - float(np.mean(baseline_pis))) / baseline_std
        if baseline_std != 0.0
        else float("inf")
    )

    return test_stat, p_value, test_type, float(effect_size)


def pvalue_to_stars(p_value: float) -> str:
    """Map p-values to common scientific significance star notation."""
    if p_value <= 1e-4:
        return "****"
    if p_value <= 1e-3:
        return "***"
    if p_value <= 1e-2:
        return "**"
    if p_value <= 5e-2:
        return "*"
    return "ns"


def add_significance_bracket(
    ax: Axes,
    x1: float,
    x2: float,
    y: float,
    h: float,
    label: str,
) -> None:
    """Draw a significance bracket between two boxplot positions."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.3, c="black")
    ax.text(
        (x1 + x2) * 0.5,
        y + h + 0.03,
        label,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )


def create_composite_plot(
    baseline_pis: np.ndarray,
    maximization_pis: np.ndarray,
    minimization_pis: np.ndarray,
    p_base_max: float,
    p_base_min: float,
    output_plot_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))

    data_to_plot = [baseline_pis, maximization_pis, minimization_pis]
    bp = ax.boxplot(
        data_to_plot,
        tick_labels=["Baseline", "Maximization", "Minimization"],
        patch_artist=True,
        widths=0.6,
        orientation="vertical",
    )

    colors = ["lightblue", "lightcoral", "lightgreen"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    y_lower = -4.4
    y_upper = 6.0
    y_range = y_upper - y_lower
    bracket_gap = 0.04 * y_range
    bracket_h = 0.03 * y_range
    y_first = 4.5
    y_second = y_first + bracket_h + bracket_gap

    ax.set_ylim(y_lower, y_upper)
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.tick_params(axis="y", labelsize=20)
    # ax.set_ylabel("IPI", fontsize=12)
    # ax.set_xlabel("Condition", fontsize=12)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Base", "Max", "Min"], fontsize=32, fontweight="bold")
    ax.grid(True, which="major", linestyle="--", alpha=0.5)
    ax.grid(True, which="minor", linestyle=":", alpha=0.25)

    add_significance_bracket(
        ax, 1, 2, y_first, bracket_h, pvalue_to_stars(p_base_max))
    add_significance_bracket(
        ax, 1, 3, y_second, bracket_h, pvalue_to_stars(p_base_min))

    plt.tight_layout()
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot_path, dpi=300, bbox_inches="tight")
    fig.savefig(str(output_plot_path).replace('.png', '.pgf'), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    max_ref = normalize_artifact_ref(args.max_artifact)
    min_ref = normalize_artifact_ref(args.min_artifact)

    print(f"Downloading max artifact: {max_ref}")
    print(f"Downloading min artifact: {min_ref}")

    api = wandb.Api()

    max_artifact = api.artifact(max_ref)
    min_artifact = api.artifact(min_ref)

    max_dir = Path(max_artifact.download())
    min_dir = Path(min_artifact.download())

    max_baseline_pis, maximization_pis = load_artifact_data(max_dir)
    min_baseline_pis, minimization_pis = load_artifact_data(min_dir)

    baseline_pis = max_baseline_pis if args.baseline_source == "max" else min_baseline_pis

    baseline_mean_delta = float(
        np.mean(max_baseline_pis) - np.mean(min_baseline_pis))
    if abs(baseline_mean_delta) > 1e-6:
        print(
            "Warning: baseline distributions differ between artifacts "
            f"(mean delta={baseline_mean_delta:+.6f}). "
            f"Using baseline from --baseline-source={args.baseline_source}."
        )

    stat_min, p_min, type_min, effect_min = compute_comparison_stats(
        baseline_pis, minimization_pis
    )
    stat_max, p_max, type_max, effect_max = compute_comparison_stats(
        baseline_pis, maximization_pis
    )

    plot_path = output_dir / args.plot_name
    create_composite_plot(
        baseline_pis,
        maximization_pis,
        minimization_pis,
        p_base_max=p_max,
        p_base_min=p_min,
        output_plot_path=plot_path,
    )

    summary = {
        "artifact_refs": {
            "max_artifact": max_ref,
            "min_artifact": min_ref,
            "baseline_source": args.baseline_source,
        },
        "artifact_dirs": {
            "max_artifact_dir": str(max_dir),
            "min_artifact_dir": str(min_dir),
        },
        "baseline": {
            "mean": float(np.mean(baseline_pis)),
            "std": float(np.std(baseline_pis)),
            "count": int(len(baseline_pis)),
            "pi_range": [float(np.min(baseline_pis)), float(np.max(baseline_pis))],
        },
        "maximization": {
            "mean": float(np.mean(maximization_pis)),
            "std": float(np.std(maximization_pis)),
            "count": int(len(maximization_pis)),
            "pi_range": [float(np.min(maximization_pis)), float(np.max(maximization_pis))],
        },
        "minimization": {
            "mean": float(np.mean(minimization_pis)),
            "std": float(np.std(minimization_pis)),
            "count": int(len(minimization_pis)),
            "pi_range": [float(np.min(minimization_pis)), float(np.max(minimization_pis))],
        },
        "comparisons": {
            "base_min": {
                "test_statistic": stat_min,
                "test_pvalue": p_min,
                "test_type": type_min,
                "n_baseline": int(len(baseline_pis)),
                "n_minimization": int(len(minimization_pis)),
                "effect_size": effect_min,
            },
            "base_max": {
                "test_statistic": stat_max,
                "test_pvalue": p_max,
                "test_type": type_max,
                "n_baseline": int(len(baseline_pis)),
                "n_maximization": int(len(maximization_pis)),
                "effect_size": effect_max,
            },
        },
        "artifacts": {
            "composite_boxplot": str(plot_path),
        },
    }

    summary_path = output_dir / args.summary_name
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nComposite boxplot saved to: {plot_path}")
    print(f"Composite summary saved to: {summary_path}")
    print("\nComparison Statistics:")
    print(
        f"Base-Min ({type_min}): stat={stat_min:.4f}, p={p_min:.6f}, d={effect_min:.3f}")
    print(
        f"Base-Max ({type_max}): stat={stat_max:.4f}, p={p_max:.6f}, d={effect_max:.3f}")


if __name__ == "__main__":
    main()
