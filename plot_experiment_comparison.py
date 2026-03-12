#!/usr/bin/env python3
"""
Plot comparison of AVP vs no-AVP experiments across tasks and val splits.

Reads metrics_summary.json from eval_ep*/metrics/ directories and generates
grouped bar charts comparing xyz_path_distance_avg, xyz_paired_mse_avg,
and joints_paired_mse_avg (labeled as paired_mse) for AVP vs no-AVP.

Usage:
    python plot_experiment_comparison.py --input_dir trained_models_highlevel/test
    python plot_experiment_comparison.py --input_dir trained_models_highlevel/test --output_dir plots/
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_experiment_dir(dirname):
    """Parse experiment directory name into (task, avp, val_ratio).

    Examples:
        PutKiwiInCenterOfTable_avp_val0.3_DT_2026-03-05-18-55-44
        TurnMugRightsideUp_val0.5_DT_2026-03-03-23-15-32
    """
    # Match: TaskName[_avp]_val{ratio}_DT_{timestamp}
    m = re.match(r"^(.+?)(_avp)?_val([\d.]+)_DT_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$", dirname)
    if not m:
        return None
    task = m.group(1)
    avp = m.group(2) is not None
    val_ratio = m.group(3)
    return task, avp, val_ratio


def find_metrics(exp_dir):
    """Find the first eval_ep*/metrics/metrics_summary.json in an experiment dir."""
    for entry in sorted(os.listdir(exp_dir)):
        if entry.startswith("eval_ep"):
            metrics_path = os.path.join(exp_dir, entry, "metrics", "metrics_summary.json")
            if os.path.exists(metrics_path):
                return metrics_path
    return None


def load_all_experiments(input_dir):
    """Load metrics from all experiment directories.

    Returns: dict keyed by (task, val_ratio) -> {"avp": metrics_dict, "no_avp": metrics_dict}
    """
    experiments = defaultdict(dict)

    for dirname in os.listdir(input_dir):
        full_path = os.path.join(input_dir, dirname)
        if not os.path.isdir(full_path):
            continue

        parsed = parse_experiment_dir(dirname)
        if parsed is None:
            print(f"  Skipping unrecognized directory: {dirname}")
            continue

        task, avp, val_ratio = parsed
        metrics_path = find_metrics(full_path)
        if metrics_path is None:
            print(f"  No metrics found in: {dirname}")
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)

        key = (task, val_ratio)
        label = "avp" if avp else "no_avp"
        experiments[key][label] = metrics
        print(f"  Loaded: {dirname} -> {task}, val={val_ratio}, {label}")

    return experiments


# Short display names for tasks
TASK_SHORT_NAMES = {
    "PutKiwiInCenterOfTable": "Kiwi",
    "TurnMugRightsideUp": "Turn Mug",
    "TurnCupUpsideDown": "Turn Cup",
    "BimanualPlaceAppleFromBowlOnCuttingBoard": "Bimanual Apple",
}

METRICS_TO_PLOT = [
    ("xyz_path_distance_avg", "XYZ Path Distance (avg) \u2193", "xyz_path_distance_stderr"),
    ("xyz_paired_mse_avg", "XYZ Paired MSE (avg) \u2193", "xyz_paired_mse_stderr"),
]


def plot_per_val_split(experiments, output_dir):
    """Create one figure per val_ratio with grouped bars per task."""
    # Group by val_ratio
    by_val = defaultdict(dict)
    for (task, val_ratio), data in experiments.items():
        by_val[val_ratio][task] = data

    for val_ratio, tasks_data in sorted(by_val.items()):
        fig, axes = plt.subplots(1, len(METRICS_TO_PLOT), figsize=(5 * len(METRICS_TO_PLOT), 5))
        if len(METRICS_TO_PLOT) == 1:
            axes = [axes]

        task_names = sorted(tasks_data.keys())
        x = np.arange(len(task_names))
        width = 0.35

        for ax, (metric_key, metric_label, stderr_key) in zip(axes, METRICS_TO_PLOT):
            avp_vals, no_avp_vals, avp_errs, no_avp_errs = [], [], [], []
            for task in task_names:
                data = tasks_data[task]
                avp_vals.append(data.get("avp", {}).get(metric_key, 0))
                no_avp_vals.append(data.get("no_avp", {}).get(metric_key, 0))
                avp_errs.append(data.get("avp", {}).get(stderr_key, 0))
                no_avp_errs.append(data.get("no_avp", {}).get(stderr_key, 0))

            bars1 = ax.bar(x - width / 2, no_avp_vals, width, label="No AVP", color="#4C72B0",
                           yerr=no_avp_errs, capsize=4, error_kw={"elinewidth": 1.5})
            bars2 = ax.bar(x + width / 2, avp_vals, width, label="AVP", color="#DD8452",
                           yerr=avp_errs, capsize=4, error_kw={"elinewidth": 1.5})

            ax.set_ylabel(metric_label)
            ax.set_xticks(x)
            short_names = [TASK_SHORT_NAMES.get(t, t) for t in task_names]
            ax.set_xticklabels(short_names, rotation=30, ha="right")
            ax.legend()

            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    h = bar.get_height()
                    if h > 0:
                        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                                    xytext=(0, 3), textcoords="offset points",
                                    ha="center", va="bottom", fontsize=7)

        fig.suptitle(f"AVP vs No-AVP Comparison (val_ratio={val_ratio})", fontsize=14, fontweight="bold")
        fig.tight_layout()

        out_path = os.path.join(output_dir, f"comparison_val{val_ratio}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


def plot_per_task(experiments, output_dir):
    """Create one figure per task with grouped bars per val_ratio."""
    # Group by task
    by_task = defaultdict(dict)
    for (task, val_ratio), data in experiments.items():
        by_task[task][val_ratio] = data

    for task, val_data in sorted(by_task.items()):
        fig, axes = plt.subplots(1, len(METRICS_TO_PLOT), figsize=(5 * len(METRICS_TO_PLOT), 5))
        if len(METRICS_TO_PLOT) == 1:
            axes = [axes]

        val_ratios = sorted(val_data.keys())
        x = np.arange(len(val_ratios))
        width = 0.35

        for ax, (metric_key, metric_label, stderr_key) in zip(axes, METRICS_TO_PLOT):
            avp_vals, no_avp_vals, avp_errs, no_avp_errs = [], [], [], []
            for vr in val_ratios:
                data = val_data[vr]
                avp_vals.append(data.get("avp", {}).get(metric_key, 0))
                no_avp_vals.append(data.get("no_avp", {}).get(metric_key, 0))
                avp_errs.append(data.get("avp", {}).get(stderr_key, 0))
                no_avp_errs.append(data.get("no_avp", {}).get(stderr_key, 0))

            bars1 = ax.bar(x - width / 2, no_avp_vals, width, label="No AVP", color="#4C72B0",
                           yerr=no_avp_errs, capsize=4, error_kw={"elinewidth": 1.5})
            bars2 = ax.bar(x + width / 2, avp_vals, width, label="AVP", color="#DD8452",
                           yerr=avp_errs, capsize=4, error_kw={"elinewidth": 1.5})

            ax.set_ylabel(metric_label)
            ax.set_xticks(x)
            ax.set_xticklabels([f"val={vr}" for vr in val_ratios])
            ax.legend()

            for bars in [bars1, bars2]:
                for bar in bars:
                    h = bar.get_height()
                    if h > 0:
                        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                                    xytext=(0, 3), textcoords="offset points",
                                    ha="center", va="bottom", fontsize=7)

        short = TASK_SHORT_NAMES.get(task, task)
        fig.suptitle(f"{short}: AVP vs No-AVP by Val Split", fontsize=14, fontweight="bold")
        fig.tight_layout()

        safe_name = task.replace(" ", "_")
        out_path = os.path.join(output_dir, f"comparison_{safe_name}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot AVP vs no-AVP experiment comparisons")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing experiment folders")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots (default: <input_dir>/plots)")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: {args.input_dir} is not a directory")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(args.input_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading experiments...")
    experiments = load_all_experiments(args.input_dir)

    if not experiments:
        print("No experiments found!")
        sys.exit(1)

    print(f"\nFound {len(experiments)} experiment pairs.")

    print("\nGenerating per-val-split plots...")
    plot_per_val_split(experiments, output_dir)

    print("\nGenerating per-task plots...")
    plot_per_task(experiments, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
