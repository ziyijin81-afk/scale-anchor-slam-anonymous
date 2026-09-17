#!/usr/bin/env python3
"""Evaluate offline-replayed Anchor weights on TUM trajectory and scale drift."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.compare_tum_anchor_runs import associate
from evals.eval_tum_submap_scale_drift import drift_metrics, extract_values, run_scales
from evals.eval_tum_trajectory_local import read_tum_trajectory, umeyama


def evaluate_trajectory(trajectory, groundtruth):
    source, target, _ = associate(groundtruth, trajectory)
    se3, _, _, _ = umeyama(source, target, False)
    sim3, _, _, scale = umeyama(source, target, True)
    se3_errors = np.linalg.norm(se3 - target, axis=1)
    sim3_errors = np.linalg.norm(sim3 - target, axis=1)
    return {
        "se3_ate_rmse_m": float(np.sqrt(np.mean(se3_errors**2))),
        "sim3_ate_rmse_m": float(np.sqrt(np.mean(sim3_errors**2))),
        "sim3_endpoint_error_m": float(sim3_errors[-1]),
        "sim3_scale": float(scale),
    }


def evaluate_run(path, groundtruth, frames_per_submap, minimum_gt_baseline):
    trajectory = read_tum_trajectory(path)
    submaps = run_scales(
        trajectory, groundtruth, frames_per_submap, minimum_gt_baseline, 0.02
    )
    return {
        "trajectory": evaluate_trajectory(trajectory, groundtruth),
        "pairwise_scale_drift": drift_metrics(
            extract_values(submaps, "pairwise_relative_scale_r")
        ),
        "umeyama_scale_drift": drift_metrics(
            extract_values(submaps, "umeyama_relative_scale_r")
        ),
        "pairwise_r": extract_values(submaps, "pairwise_relative_scale_r").tolist(),
        "umeyama_r": extract_values(submaps, "umeyama_relative_scale_r").tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groundtruth", required=True, type=Path)
    parser.add_argument("--anchor-off", required=True, type=Path)
    parser.add_argument("--sweep-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frames-per-submap", type=int, default=17)
    parser.add_argument("--minimum-gt-baseline", type=float, default=0.1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    groundtruth = read_tum_trajectory(args.groundtruth)

    results = {0.0: evaluate_run(
        args.anchor_off, groundtruth, args.frames_per_submap, args.minimum_gt_baseline
    )}
    for summary_path in sorted(args.sweep_dir.glob("w_*/scale_replay_summary.json")):
        summary = json.loads(summary_path.read_text())
        weight = float(summary["anchor_weight"])
        results[weight] = evaluate_run(
            summary_path.parent / "poses.txt", groundtruth,
            args.frames_per_submap, args.minimum_gt_baseline,
        )
    results = dict(sorted(results.items()))

    metric_paths = {
        "pairwise_rmse_log_r": ("pairwise_scale_drift", "rmse_log_r"),
        "umeyama_rmse_log_r": ("umeyama_scale_drift", "rmse_log_r"),
        "pairwise_final_abs_log_r": ("pairwise_scale_drift", "final_abs_log_r"),
        "umeyama_final_abs_log_r": ("umeyama_scale_drift", "final_abs_log_r"),
        "sim3_ate_rmse_m": ("trajectory", "sim3_ate_rmse_m"),
        "se3_ate_rmse_m": ("trajectory", "se3_ate_rmse_m"),
    }
    best_weights = {}
    for name, (section, key) in metric_paths.items():
        best_weights[name] = min(results, key=lambda weight: results[weight][section][key])

    output = {
        "configuration": {
            "vggt_rerun": False,
            "overlap_weight": 1.0,
            "frames_per_submap": args.frames_per_submap,
            "minimum_gt_baseline_m": args.minimum_gt_baseline,
        },
        "best_weights": best_weights,
        "results": {str(weight): result for weight, result in results.items()},
    }
    (args.output_dir / "anchor_weight_sweep.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )

    weights = list(results)
    labels = ["off" if weight == 0 else str(weight) for weight in weights]
    x = np.arange(len(weights))
    panels = (
        ("pairwise_scale_drift", "rmse_log_r", "Pairwise RMSE(log r)", "RMSE(log r)"),
        ("umeyama_scale_drift", "rmse_log_r", "Umeyama RMSE(log r)", "RMSE(log r)"),
        ("trajectory", "sim3_ate_rmse_m", "Sim(3) trajectory ATE", "RMSE [m]"),
        ("pairwise_scale_drift", "final_abs_log_r", "Final submap scale drift", "abs(log r_final)"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    for axis, (section, key, title, ylabel) in zip(axes.flat, panels):
        values = [results[weight][section][key] for weight in weights]
        axis.plot(x, values, marker="o")
        best_index = int(np.argmin(values))
        axis.scatter([best_index], [values[best_index]], color="red", zorder=3)
        axis.set_xticks(x, labels)
        axis.set(title=title, xlabel="Anchor weight (overlap weight=1)", ylabel=ylabel)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.output_dir / "anchor_weight_sweep.png", dpi=180)
    plt.close(figure)
    print(json.dumps({"best_weights": best_weights, "results": output["results"]}, indent=2))


if __name__ == "__main__":
    main()
