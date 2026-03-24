import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.projections.polar import PolarAxes
import json
from math import pi
from pathlib import Path
from scipy import stats
import numpy as np
from typing import Union, TYPE_CHECKING

import matplotlib
from matplotlib.lines import Line2D

matplotlib.use("Agg")

# --- Configuration ---
FILE_BASELINE = "runs/2026-02-19/13-21-10/metrics_20260219_132154.json"
FILE_INTERVENED = "runs/2026-02-18/21-50-56/metrics_20260218_215142.json"
FILE_BASELINE_PIS = "runs/2026-02-19/13-21-10/pair_results_20260219_132154.csv"
FILE_INTERVENED_PIS = "runs/2026-02-18/21-50-56/pair_results_20260218_215142.csv"


def load_and_identify(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    pi = data['model_polarization_index']
    return data, pi


def load_question_pis(baseline_csv, intervened_csv):
    """Load question-level PIs from CSV files."""
    baseline_df = pd.read_csv(baseline_csv)
    intervened_df = pd.read_csv(intervened_csv)
    return baseline_df, intervened_df


def create_boxplot_comparison(baseline_pis, intervened_pis, output_dir):
    """Creates a boxplot comparing question-level PI distributions.

    Handles arrays of different lengths by using Mann-Whitney U test
    (unpaired) instead of Wilcoxon signed-rank test when sizes differ.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Handle empty arrays
    if len(baseline_pis) == 0 or len(intervened_pis) == 0:
        ax.text(0.5, 0.5, 'Insufficient data for comparison',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        fig.tight_layout()
        boxplot_path = output_dir / "pi_shift_boxplot.png"
        fig.savefig(boxplot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return boxplot_path, {
            'baseline_mean': None, 'baseline_std': None,
            'intervened_mean': None, 'intervened_std': None,
            'test_statistic': None, 'test_pvalue': None,
            'test_type': None, 'n_baseline': 0, 'n_intervened': 0
        }

    data_to_plot = [baseline_pis, intervened_pis]
    bp = ax.boxplot(data_to_plot, tick_labels=['Baseline', 'Intervened'],
                    patch_artist=True, widths=0.6,
                    orientation='horizontal')

    # Keep PI scale fixed for comparability across runs.
    ax.set_xlim(-4, 4)

    # Color the boxes
    colors = ['blue', 'red']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Polarization Index (PI)', fontsize=12)
    ax.set_title('Distribution of Question-Level PI', fontsize=14)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)

    # Add statistics as text
    baseline_mean = float(np.mean(baseline_pis))
    intervened_mean = float(np.mean(intervened_pis))
    baseline_std = float(np.std(baseline_pis))
    intervened_std = float(np.std(intervened_pis))

    # Choose appropriate statistical test based on array lengths
    n_baseline = len(baseline_pis)
    n_intervened = len(intervened_pis)

    if n_baseline == n_intervened and n_baseline > 0:
        # Paired test (Wilcoxon signed-rank)
        try:
            test_stat, test_pvalue = stats.wilcoxon(
                baseline_pis, intervened_pis)
            test_type = 'Wilcoxon'
        except ValueError:
            # Fall back to Mann-Whitney if Wilcoxon fails (e.g., all differences are zero)
            test_stat, test_pvalue = stats.mannwhitneyu(
                baseline_pis, intervened_pis, alternative='two-sided')
            test_type = 'Mann-Whitney U'
    else:
        # Unpaired test (Mann-Whitney U) for different sample sizes
        test_stat, test_pvalue = stats.mannwhitneyu(
            baseline_pis, intervened_pis, alternative='two-sided')
        test_type = 'Mann-Whitney U'

    effect_size = (intervened_mean - baseline_mean) / \
        baseline_std if baseline_std != 0 else float('inf')

    stats_text = f"Baseline: μ={baseline_mean:.3f}, σ={baseline_std:.3f} (n={n_baseline})\n"
    stats_text += f"Intervened: μ={intervened_mean:.3f}, σ={intervened_std:.3f} (n={n_intervened})\n"
    stats_text += f"{test_type}: p={test_pvalue:.4f} (stat={test_stat:.3f})\n"
    stats_text += f"Effect Size (Cohen's d): {effect_size:.3f}"

    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    boxplot_path = output_dir / "pi_shift_boxplot.png"
    fig.savefig(boxplot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return boxplot_path, {
        'baseline_mean': baseline_mean,
        'baseline_std': baseline_std,
        'intervened_mean': intervened_mean,
        'intervened_std': intervened_std,
        'test_statistic': float(test_stat),
        'test_pvalue': float(test_pvalue),
        'test_type': test_type,
        'n_baseline': n_baseline,
        'n_intervened': n_intervened
    }


def create_parallel_coordinates_plot(baseline_pis, intervened_pis, output_dir,
                                     baseline_pair_ids=None, intervened_pair_ids=None):
    """Creates a parallel coordinates plot showing shift from baseline to intervened.

    If pair_ids are provided, aligns data by pair_id to ensure correct pairing.
    Only plots pairs that exist in both baseline and intervention.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Align by pair_id if provided
    if baseline_pair_ids is not None and intervened_pair_ids is not None:
        # Create dictionaries for lookup
        baseline_dict = {pid: pi for pid, pi in zip(
            baseline_pair_ids, baseline_pis)}
        intervened_dict = {pid: pi for pid, pi in zip(
            intervened_pair_ids, intervened_pis)}

        # Find common pair_ids
        common_ids = set(baseline_dict.keys()) & set(intervened_dict.keys())
        if len(common_ids) == 0:
            ax.text(0.5, 0.5, 'No common pairs between baseline and intervention',
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            fig.tight_layout()
            parallel_path = output_dir / "pi_shift_parallel.png"
            fig.savefig(parallel_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return parallel_path

        # Extract aligned arrays
        common_ids_sorted = sorted(common_ids)
        baseline_pis = np.array([baseline_dict[pid]
                                for pid in common_ids_sorted])
        intervened_pis = np.array([intervened_dict[pid]
                                  for pid in common_ids_sorted])
    else:
        # Fallback: use minimum length if no pair_ids provided
        min_len = min(len(baseline_pis), len(intervened_pis))
        if min_len == 0:
            ax.text(0.5, 0.5, 'Insufficient data for parallel plot',
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            fig.tight_layout()
            parallel_path = output_dir / "pi_shift_parallel.png"
            fig.savefig(parallel_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return parallel_path
        baseline_pis = baseline_pis[:min_len]
        intervened_pis = intervened_pis[:min_len]

    n_questions = len(baseline_pis)

    # Determine color based on direction and magnitude of change
    changes = intervened_pis - baseline_pis

    # Normalize colors based on change magnitude
    max_change = max(abs(changes.min()), abs(changes.max()))

    for i in range(n_questions):
        change = changes[i]
        if change < 0:
            # color = plt.cm.Reds(min(abs(change) / max_change, 1.0))
            color = 'red'
        else:
            # color = plt.cm.Blues(min(abs(change) / max_change, 1.0))
            color = 'blue'

        ax.plot([0, 1], [baseline_pis[i], intervened_pis[i]],
                color=color, alpha=0.4, linewidth=0.8)

    # Add vertical axes
    ax.axvline(0, color='black', linewidth=2, label='Baseline')
    ax.axvline(1, color='black', linewidth=2, label='Intervened')

    # Add horizontal zero line
    ax.axhline(0, color='gray', linewidth=1, linestyle='--', alpha=0.5)

    # Customize plot
    ax.set_xlim(-0.1, 1.1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Baseline', 'Intervened'], fontsize=12)
    ax.set_ylabel('Polarization Index (PI)', fontsize=12)
    ax.set_title('Question-Level PI Shift: Baseline → Intervened', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Add summary statistics
    baseline_mean = baseline_pis.mean()
    intervened_mean = intervened_pis.mean()
    mean_shift = intervened_mean - baseline_mean

    # Draw mean lines
    ax.plot([0, 1], [baseline_mean, intervened_mean],
            color='black', linewidth=3, linestyle='-', label='Mean', zorder=10)

    # Add legend with color explanation
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.6, label='Leftward shift'),
        Patch(facecolor='blue', alpha=0.6, label='Rightward shift'),
        Line2D([0], [0], color='black', linewidth=3, label='Mean PI')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    # Add text box with statistics
    stats_text = f"Mean shift: {mean_shift:.3f}\n"
    stats_text += f"Questions: {n_questions}"
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    fig.tight_layout()
    parallel_path = output_dir / "pi_shift_parallel.png"
    fig.savefig(parallel_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return parallel_path


def create_fluidity_chart(baseline_by_axis: dict, intervened_by_axis: dict, output_dir):
    """Creates a horizontal bar chart showing PI shift by axis/theme.

    Args:
        baseline_by_axis: Dict mapping axis name to stats dict with 'mean_pi'
        intervened_by_axis: Dict mapping axis name to stats dict with 'mean_pi'
        output_dir: Path object for output directory

    Returns:
        Tuple of (figure path, DataFrame with axis data)
    """
    # Prepare data
    axes_data = {}
    for axis, stats in baseline_by_axis.items():
        axes_data[axis] = {'Baseline': stats['mean_pi']}

    for axis, stats in intervened_by_axis.items():
        if axis in axes_data:
            axes_data[axis]['Intervened'] = stats['mean_pi']

    df_axes = pd.DataFrame(axes_data).T
    df_axes['diff'] = (df_axes['Intervened'] - df_axes['Baseline']).abs()
    df_axes = df_axes.sort_values('diff', ascending=False)

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    change = df_axes['Intervened'] - df_axes['Baseline']
    colors = ['red' if x < 0 else 'blue' for x in change]

    change.plot(kind='barh', ax=ax, color=colors, alpha=0.7)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Shift in PI (Negative = Move Left)", fontsize=12)
    ax.set_title("Fluidity/PI Shift by Theme", fontsize=14)
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    for i, v in enumerate(change):
        ax.text(v - 0.1 if v < 0 else v + 0.05, i, f"{v:.2f}", va='center')

    fig.tight_layout()
    fluidity_path = output_dir / "pi_shift_fluidity.png"
    fig.savefig(fluidity_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return fluidity_path, df_axes


def build_axes_comparison_df(baseline_by_axis: dict, intervened_by_axis: dict) -> pd.DataFrame:
    """Build aligned axis comparison data with a stable topic order.

    Topic order is alphabetical so radar charts remain consistent across runs.
    """
    common_axes = sorted(set(baseline_by_axis.keys()) &
                         set(intervened_by_axis.keys()))
    if not common_axes:
        return pd.DataFrame(columns=['Baseline', 'Intervened'])

    rows = [
        {
            'axis': axis,
            'Baseline': baseline_by_axis[axis]['mean_pi'],
            'Intervened': intervened_by_axis[axis]['mean_pi'],
        }
        for axis in common_axes
    ]
    return pd.DataFrame(rows).set_index('axis')


def generate_comparison_visualizations(
    baseline_metrics: dict,
    intervened_metrics: dict,
    baseline_pair_results: list,
    intervened_pair_results: list,
    output_dir: Union[str, Path],
) -> dict:
    """
    Generates all comparison visualizations from in-memory data.

    This function is designed to be imported and called from other modules
    (e.g., likert_scale_test.py) without needing to save/load from files.

    Handles cases where baseline and intervention have different numbers of
    valid pairs by aligning on pair_id where possible.

    Args:
        baseline_metrics: Dict with 'model_polarization_index', 'pi_std', 'by_axis'
        intervened_metrics: Dict with 'model_polarization_index', 'pi_std', 'by_axis'
        baseline_pair_results: List of pair result dicts with 'pair_id' and 'polarization_index'
        intervened_pair_results: List of pair result dicts with 'pair_id' and 'polarization_index'
        output_dir: Path or str to output directory

    Returns:
        Dictionary with paths to generated artifacts and computed statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract valid PIs with their pair_ids
    baseline_valid = [
        (p['pair_id'], p['polarization_index'])
        for p in baseline_pair_results
        if p.get('polarization_index') is not None
    ]
    intervened_valid = [
        (p['pair_id'], p['polarization_index'])
        for p in intervened_pair_results
        if p.get('polarization_index') is not None
    ]

    # Check if we have any valid data
    if len(baseline_valid) == 0 and len(intervened_valid) == 0:
        raise ValueError(
            "No valid pair results in either baseline or intervention. Cannot generate visualizations.")

    # Extract arrays for boxplot (can have different lengths)
    baseline_pair_ids = np.array(
        [p[0] for p in baseline_valid]) if baseline_valid else np.array([])
    baseline_pair_pis = np.array(
        [p[1] for p in baseline_valid]) if baseline_valid else np.array([])
    intervened_pair_ids = np.array(
        [p[0] for p in intervened_valid]) if intervened_valid else np.array([])
    intervened_pair_pis = np.array(
        [p[1] for p in intervened_valid]) if intervened_valid else np.array([])

    results = {
        'baseline_pi': baseline_metrics.get('model_polarization_index'),
        'intervened_pi': intervened_metrics.get('model_polarization_index'),
        'baseline_std': baseline_metrics.get('pi_std'),
        'intervened_std': intervened_metrics.get('pi_std'),
        'n_baseline_valid': len(baseline_valid),
        'n_intervened_valid': len(intervened_valid),
        'artifacts': {},
    }

    # 1. Radar chart (if axis data available)
    baseline_by_axis = baseline_metrics.get('by_axis', {})
    intervened_by_axis = intervened_metrics.get('by_axis', {})

    if baseline_by_axis and intervened_by_axis:
        df_axes = build_axes_comparison_df(
            baseline_by_axis, intervened_by_axis)

        # Create radar chart
        fig_radar = create_radar_chart(
            df_axes.index.tolist(),
            df_axes['Baseline'].tolist(),
            df_axes['Intervened'].tolist(),
            "PI Breakdown by Theme"
        )
        radar_path = output_dir / "pi_shift_radar.png"
        fig_radar.savefig(radar_path, dpi=300, bbox_inches="tight")
        plt.close(fig_radar)
        results['artifacts']['radar'] = str(radar_path)

        # Create fluidity chart
        fluidity_path, _ = create_fluidity_chart(
            baseline_by_axis, intervened_by_axis, output_dir)
        results['artifacts']['fluidity'] = str(fluidity_path)

        # Save axes CSV
        axes_csv_path = output_dir / "pi_shift_axes.csv"
        df_axes.to_csv(axes_csv_path)
        results['artifacts']['axes_csv'] = str(axes_csv_path)

    # 2. Boxplot comparison
    boxplot_path, boxplot_stats = create_boxplot_comparison(
        baseline_pair_pis, intervened_pair_pis, output_dir
    )
    results['artifacts']['boxplot'] = str(boxplot_path)
    results['question_level_stats'] = boxplot_stats

    # 3. Parallel coordinates plot (aligned by pair_id)
    parallel_path = create_parallel_coordinates_plot(
        baseline_pair_pis, intervened_pair_pis, output_dir,
        baseline_pair_ids=baseline_pair_ids,
        intervened_pair_ids=intervened_pair_ids
    )
    results['artifacts']['parallel'] = str(parallel_path)

    # Save summary JSON
    summary_path = output_dir / "pi_shift_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    results['artifacts']['summary'] = str(summary_path)

    return results


def create_radar_chart(categories, baseline_vals, intervened_vals, title):
    """Generates a Radar Chart comparing Baseline vs Intervened."""
    N = len(categories)
    categories = ["\n".join(cat.split()) for cat in categories]

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    # Initialise the spider plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    assert isinstance(ax, PolarAxes)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, color='grey', size=25)

    # Draw ylabels (concentric circles)
    ax.set_rlabel_position(0)
    plt.yticks([-4, -2, 0, 2, 4], ["-4", "-2", "0",
               "2", "4"], color="black", size=10)
    plt.ylim(-4, 4)
    zero_angles = [i / 360 * 2 * pi for i in range(361)]
    ax.plot(zero_angles, [0] * len(zero_angles),
            color="black", linewidth=1.2, alpha=0.7)

    # Plot Baseline
    values_b = baseline_vals + baseline_vals[:1]
    ax.plot(angles, values_b, linewidth=1,
            linestyle='dashed', label='Baseline')
    ax.fill(angles, values_b, 'b', alpha=0.1)

    # Plot Intervened
    values_i = intervened_vals + intervened_vals[:1]
    ax.plot(angles, values_i, linewidth=2, linestyle='solid',
            color='red', label='Intervened')
    ax.fill(angles, values_i, 'r', alpha=0.1)

    plt.title(title, size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    return fig


def main():
    output_dir = Path(FILE_INTERVENED).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    baseline, baseline_pi = load_and_identify(FILE_BASELINE)
    intervened, intervened_pi = load_and_identify(FILE_INTERVENED)

    # Keep labels tied to the provided file paths to avoid silent inversion.
    # If this warning appears, double-check that FILE_BASELINE/FILE_INTERVENED
    # point to the intended runs.
    if abs(baseline_pi) > abs(intervened_pi):
        print(
            "Warning: |Baseline PI| > |Intervened PI|. "
            "Keeping file-based labels as provided."
        )

    print(f"Baseline PI: {baseline['model_polarization_index']:.3f}")
    print(f"Intervened PI: {intervened['model_polarization_index']:.3f}")

    # 2. Prepare Data for Radar Chart (Axis Breakdown) with stable topic order
    df_axes = build_axes_comparison_df(
        baseline['by_axis'], intervened['by_axis'])

    # 3. Plot 1: Radar Chart (The "Footprint")
    fig1 = create_radar_chart(
        df_axes.index.tolist(),
        df_axes['Baseline'].tolist(),
        df_axes['Intervened'].tolist(),
        "PI Breakdown by Theme"
    )
    fig1_path = output_dir / "pi_shift_radar.png"
    fig1.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # 4. Plot 2: The "Fluidity" Spectrum (Bar Chart of Deltas)
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # Calculate change
    change = df_axes['Intervened'] - df_axes['Baseline']
    colors = ['red' if x < 0 else 'blue' for x in change]  # Red for Left Shift

    change.plot(kind='barh', ax=ax2, color=colors, alpha=0.7)
    ax2.axvline(0, color='black', linewidth=0.8)
    ax2.set_xlabel("Shift in PI (Negative = Move Left)", fontsize=12)
    ax2.set_title("Fluidity/PI Shift by Theme", fontsize=14)
    ax2.grid(axis='x', linestyle='--', alpha=0.5)

    # Add labels
    for i, v in enumerate(change):
        ax2.text(v - 0.1 if v < 0 else v + 0.05, i, f"{v:.2f}", va='center')
    fig2_path = output_dir / "pi_shift_fluidity.png"
    fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # 6. Plot 3: Question-Level PI Boxplot (if CSV files provided)
    boxplot_path = None
    boxplot_stats = None
    parallel_path = None
    if FILE_BASELINE_PIS and FILE_INTERVENED_PIS:
        baseline_df, intervened_df = load_question_pis(
            FILE_BASELINE_PIS, FILE_INTERVENED_PIS)
        baseline_pis = baseline_df['polarization_index'].values
        intervened_pis = intervened_df['polarization_index'].values
        boxplot_path, boxplot_stats = create_boxplot_comparison(
            baseline_pis, intervened_pis, output_dir)
        print(f"\nQuestion-Level PI Statistics:")
        print(
            f"Baseline - Mean: {boxplot_stats['baseline_mean']:.3f}, Std: {boxplot_stats['baseline_std']:.3f}")
        print(
            f"Intervened - Mean: {boxplot_stats['intervened_mean']:.3f}, Std: {boxplot_stats['intervened_std']:.3f}")
        test_type = boxplot_stats.get('test_type', 'Statistical Test')
        print(f"\n{test_type}:")
        print(f"Statistic: {boxplot_stats['test_statistic']:.4f}")
        print(f"P-value: {boxplot_stats['test_pvalue']:.4f}")
        if boxplot_stats['test_pvalue'] < 0.05:
            print("Result: Statistically significant difference (p < 0.05)")
        else:
            print("Result: No statistically significant difference (p >= 0.05)")

        # 7. Plot 4: Parallel Coordinates Plot
        parallel_path = create_parallel_coordinates_plot(
            baseline_pis, intervened_pis, output_dir)
        print(f"\nParallel coordinates plot saved to: {parallel_path}")

    df_axes.to_csv(output_dir / "pi_shift_axes.csv")
    summary = {
        "baseline_pi": baseline["model_polarization_index"],
        "intervened_pi": intervened["model_polarization_index"],
        "baseline_std": baseline["pi_std"],
        "intervened_std": intervened["pi_std"],
        "FILE_BASELINE": str(Path(FILE_BASELINE)),
        "FILE_INTERVENED": str(Path(FILE_INTERVENED)),
        "file_baseline_pis": str(Path(FILE_BASELINE_PIS)) if FILE_BASELINE_PIS else None,
        "file_intervened_pis": str(Path(FILE_INTERVENED_PIS)) if FILE_INTERVENED_PIS else None,
        "output_dir": str(output_dir),
        "artifacts": {
            "radar": str(fig1_path),
            "fluidity": str(fig2_path),
            "boxplot": str(boxplot_path),
            "parallel": str(parallel_path) if parallel_path else None,
            "axes_csv": str(output_dir / "pi_shift_axes.csv"),
        },
    }
    if boxplot_path:
        summary["artifacts"]["boxplot"] = str(boxplot_path)
        summary["question_level_stats"] = boxplot_stats
    if parallel_path:
        summary["artifacts"]["parallel"] = str(parallel_path)

    with open(output_dir / "pi_shift_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
