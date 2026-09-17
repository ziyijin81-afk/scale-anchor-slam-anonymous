#!/usr/bin/env python3
"""Compare per-submap residual scale drift for TUM Anchor-on/off runs."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.eval_tum_trajectory_local import read_tum_trajectory, umeyama


def unique_rows_by_timestamp(rows):
    seen = set()
    unique = []
    for row in rows:
        timestamp = float(row[0])
        if timestamp not in seen:
            seen.add(timestamp)
            unique.append(row)
    return np.asarray(unique)


def associate_groundtruth(rows, groundtruth, max_time_difference):
    estimated, truth, timestamps = [], [], []
    for row in unique_rows_by_timestamp(rows):
        index = int(np.argmin(np.abs(groundtruth[:, 0] - row[0])))
        difference = float(abs(groundtruth[index, 0] - row[0]))
        if difference <= max_time_difference:
            timestamps.append(float(row[0]))
            estimated.append(row[1:4])
            truth.append(groundtruth[index, 1:4])
    return np.asarray(estimated), np.asarray(truth), timestamps


def pairwise_scale(estimated, truth, minimum_gt_baseline):
    row, column = np.triu_indices(len(estimated), k=1)
    estimated_distances = np.linalg.norm(estimated[row] - estimated[column], axis=1)
    truth_distances = np.linalg.norm(truth[row] - truth[column], axis=1)
    valid = (
        np.isfinite(estimated_distances)
        & np.isfinite(truth_distances)
        & (estimated_distances > 1e-12)
        & (truth_distances >= minimum_gt_baseline)
    )
    ratios = estimated_distances[valid] / truth_distances[valid]
    if len(ratios) == 0:
        raise ValueError("No valid frame pairs for scale estimation")
    median = float(np.median(ratios))
    mad = float(np.median(np.abs(ratios - median)))
    return {
        "estimated_units_per_meter": median,
        "mad": mad,
        "relative_mad": mad / median,
        "valid_pair_count": int(len(ratios)),
        "minimum_gt_baseline_m": minimum_gt_baseline,
    }


def run_scales(trajectory, groundtruth, frames_per_submap, minimum_gt_baseline, max_time_difference):
    if len(trajectory) % frames_per_submap:
        raise ValueError(
            f"Trajectory has {len(trajectory)} rows, not divisible by "
            f"{frames_per_submap} frames per submap"
        )
    submaps = []
    for submap_index, start in enumerate(range(0, len(trajectory), frames_per_submap)):
        rows = trajectory[start : start + frames_per_submap]
        estimated, truth, timestamps = associate_groundtruth(
            rows, groundtruth, max_time_difference
        )
        if len(estimated) < 3:
            raise ValueError(f"Submap {submap_index} has fewer than three associated poses")
        pairwise = pairwise_scale(estimated, truth, minimum_gt_baseline)
        _, _, _, metric_per_estimated_unit = umeyama(estimated, truth, True)
        submaps.append({
            "submap_index": submap_index,
            "unique_pose_count": len(estimated),
            "first_timestamp": timestamps[0],
            "last_timestamp": timestamps[-1],
            "pairwise": pairwise,
            "umeyama_metric_per_estimated_unit": float(metric_per_estimated_unit),
        })

    c0 = submaps[0]["pairwise"]["estimated_units_per_meter"]
    m0 = submaps[0]["umeyama_metric_per_estimated_unit"]
    for submap in submaps:
        ck = submap["pairwise"]["estimated_units_per_meter"]
        mk = submap["umeyama_metric_per_estimated_unit"]
        submap["pairwise_relative_scale_r"] = float(c0 / ck)
        submap["umeyama_relative_scale_r"] = float(mk / m0)
    return submaps


def drift_metrics(values):
    values = np.asarray(values, dtype=np.float64)
    log_values = np.log(values)
    second_half_start = len(values) // 2
    second_half_indices = np.arange(second_half_start, len(values), dtype=np.float64)
    slope = float(np.polyfit(second_half_indices, log_values[second_half_start:], 1)[0])
    return {
        "rmse_log_r": float(np.sqrt(np.mean(log_values * log_values))),
        "std_log_r": float(np.std(log_values)),
        "max_abs_log_r": float(np.max(np.abs(log_values))),
        "final_abs_log_r": float(abs(log_values[-1])),
        "second_half_log_slope_per_submap": slope,
        "mean_abs_percent_from_one": float(np.mean(np.abs(values - 1.0)) * 100.0),
        "max_abs_percent_from_one": float(np.max(np.abs(values - 1.0)) * 100.0),
    }


def extract_values(submaps, key):
    return np.asarray([submap[key] for submap in submaps], dtype=np.float64)


def percent_change(on_value, off_value):
    return float((on_value / off_value - 1.0) * 100.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groundtruth", required=True, type=Path)
    parser.add_argument("--anchor-on", required=True, type=Path)
    parser.add_argument("--anchor-off", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frames-per-submap", type=int, default=17)
    parser.add_argument("--minimum-gt-baseline", type=float, default=0.1)
    parser.add_argument("--max-time-difference", type=float, default=0.02)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    groundtruth = read_tum_trajectory(args.groundtruth)
    runs = {
        "anchor_on": run_scales(
            read_tum_trajectory(args.anchor_on), groundtruth,
            args.frames_per_submap, args.minimum_gt_baseline,
            args.max_time_difference,
        ),
        "anchor_off": run_scales(
            read_tum_trajectory(args.anchor_off), groundtruth,
            args.frames_per_submap, args.minimum_gt_baseline,
            args.max_time_difference,
        ),
    }
    for on_submap, off_submap in zip(runs["anchor_on"], runs["anchor_off"]):
        if (on_submap["first_timestamp"], on_submap["last_timestamp"]) != (
            off_submap["first_timestamp"], off_submap["last_timestamp"]
        ):
            raise ValueError("Anchor-on/off submap timestamps differ")

    metrics = {}
    for run_name, submaps in runs.items():
        metrics[run_name] = {
            "pairwise": drift_metrics(extract_values(submaps, "pairwise_relative_scale_r")),
            "umeyama": drift_metrics(extract_values(submaps, "umeyama_relative_scale_r")),
        }
    metrics["anchor_on_change_percent_vs_off"] = {
        method: {
            key: percent_change(metrics["anchor_on"][method][key], metrics["anchor_off"][method][key])
            for key in (
                "rmse_log_r", "std_log_r", "max_abs_log_r",
                "final_abs_log_r", "mean_abs_percent_from_one",
            )
        }
        for method in ("pairwise", "umeyama")
    }

    output = {
        "configuration": {
            "frames_per_submap": args.frames_per_submap,
            "minimum_gt_baseline_m": args.minimum_gt_baseline,
            "max_time_difference_s": args.max_time_difference,
            "submap_count": len(runs["anchor_on"]),
            "pairwise_definition": "c_k=median(||p_est_i-p_est_j||/||p_gt_i-p_gt_j||), r_k=c_0/c_k",
            "umeyama_definition": "r_k=m_k/m_0 where p_gt ~= m_k R p_est + t",
        },
        "metrics": metrics,
        "runs": runs,
    }
    (args.output_dir / "submap_scale_drift.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )

    indices = np.arange(len(runs["anchor_on"]))
    figure, axes = plt.subplots(2, 2, figsize=(16, 11))
    for run_name, label in (("anchor_off", "Anchor off"), ("anchor_on", "Anchor on")):
        pairwise = extract_values(runs[run_name], "pairwise_relative_scale_r")
        umeyama_values = extract_values(runs[run_name], "umeyama_relative_scale_r")
        axes[0, 0].plot(indices, pairwise, marker="o", label=label)
        axes[0, 1].plot(indices, umeyama_values, marker="o", label=label)
        axes[1, 0].plot(indices, 100.0 * (pairwise - 1.0), marker="o", label=label)
        axes[1, 1].plot(indices, 100.0 * (umeyama_values - 1.0), marker="o", label=label)
    for axis_index, axis in enumerate(axes.flat):
        axis.axhline(1.0 if axis_index < 2 else 0.0, color="black", linestyle="--", linewidth=1)
        axis.grid(alpha=0.25)
        axis.set_xlabel("submap index")
        axis.legend()
    axes[0, 0].set(title="Pairwise-distance relative scale r_k", ylabel="r_k")
    axes[0, 1].set(title="Per-submap Umeyama relative scale", ylabel="r_k")
    axes[1, 0].set(title="Pairwise scale deviation from submap0", ylabel="100(r_k-1) [%]")
    axes[1, 1].set(title="Umeyama scale deviation from submap0", ylabel="100(r_k-1) [%]")
    figure.tight_layout()
    figure.savefig(args.output_dir / "submap_scale_drift.png", dpi=180)
    plt.close(figure)
    print(json.dumps({"metrics": metrics, "submaps": runs}, indent=2))


if __name__ == "__main__":
    main()
