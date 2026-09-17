#!/usr/bin/env python3
"""Dependency-light TUM trajectory evaluation with SE(3)/Sim(3) alignment."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_tum_trajectory(path):
    rows = []
    for line in path.read_text().splitlines():
        if line and not line.startswith("#"):
            values = [float(value) for value in line.split()[:8]]
            if len(values) == 8:
                rows.append(values)
    return np.asarray(rows, dtype=np.float64)


def umeyama(source, target, with_scale):
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = source_centered.T @ target_centered / len(source)
    left, singular_values, right_t = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(right_t.T @ left.T) < 0:
        sign[-1] = -1
    rotation = right_t.T @ np.diag(sign) @ left.T
    if with_scale:
        variance = float(np.sum(source_centered * source_centered) / len(source))
        scale = float(np.sum(singular_values * sign) / variance)
    else:
        scale = 1.0
    translation = target_mean - scale * (rotation @ source_mean)
    aligned = scale * (source @ rotation.T) + translation
    return aligned, rotation, translation, scale


def error_stats(errors):
    return {
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "std": float(np.std(errors)),
        "min": float(np.min(errors)),
        "max": float(np.max(errors)),
    }


def path_length(points):
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def deduplicate_estimates(estimated):
    groups = defaultdict(list)
    order = []
    for row in estimated:
        timestamp = float(row[0])
        if timestamp not in groups:
            order.append(timestamp)
        groups[timestamp].append(row)
    unique = np.asarray([groups[timestamp][0] for timestamp in order])
    discontinuities = []
    for timestamp in order:
        rows = np.asarray(groups[timestamp])
        if len(rows) > 1:
            translations = rows[:, 1:4]
            pair_distances = np.linalg.norm(
                translations[:, None, :] - translations[None, :, :], axis=-1
            )
            discontinuities.append({
                "timestamp": timestamp,
                "copies": len(rows),
                "max_translation_difference": float(np.max(pair_distances)),
            })
    return unique, discontinuities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groundtruth", required=True, type=Path)
    parser.add_argument("--estimate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-time-difference", type=float, default=0.02)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    groundtruth = read_tum_trajectory(args.groundtruth)
    estimated_all = read_tum_trajectory(args.estimate)
    estimated, discontinuities = deduplicate_estimates(estimated_all)

    matched_estimated = []
    matched_groundtruth = []
    matched_times = []
    time_differences = []
    for row in estimated:
        index = int(np.argmin(np.abs(groundtruth[:, 0] - row[0])))
        difference = float(abs(groundtruth[index, 0] - row[0]))
        if difference <= args.max_time_difference:
            matched_estimated.append(row[1:4])
            matched_groundtruth.append(groundtruth[index, 1:4])
            matched_times.append(row[0])
            time_differences.append(difference)
    matched_estimated = np.asarray(matched_estimated)
    matched_groundtruth = np.asarray(matched_groundtruth)
    matched_times = np.asarray(matched_times)
    if len(matched_estimated) < 3:
        raise ValueError("Too few timestamp associations")

    se3, se3_rotation, se3_translation, _ = umeyama(
        matched_estimated, matched_groundtruth, False
    )
    sim3, sim3_rotation, sim3_translation, sim3_scale = umeyama(
        matched_estimated, matched_groundtruth, True
    )
    se3_errors = np.linalg.norm(se3 - matched_groundtruth, axis=1)
    sim3_errors = np.linalg.norm(sim3 - matched_groundtruth, axis=1)

    duplicate_values = [
        record["max_translation_difference"] for record in discontinuities
    ]
    output = {
        "input": {
            "groundtruth": str(args.groundtruth),
            "estimate": str(args.estimate),
            "estimate_rows": len(estimated_all),
            "unique_estimate_timestamps": len(estimated),
            "associated_poses": len(matched_estimated),
            "max_time_difference_s": max(time_differences),
        },
        "se3_alignment": {
            "translation_ate_m": error_stats(se3_errors),
            "endpoint_error_m": float(se3_errors[-1]),
            "path_length_m": path_length(se3),
            "rotation": se3_rotation.tolist(),
            "translation": se3_translation.tolist(),
        },
        "sim3_alignment": {
            "translation_ate_m": error_stats(sim3_errors),
            "endpoint_error_m": float(sim3_errors[-1]),
            "path_length_m": path_length(sim3),
            "scale": sim3_scale,
            "rotation": sim3_rotation.tolist(),
            "translation": sim3_translation.tolist(),
        },
        "groundtruth_matched_path_length_m": path_length(matched_groundtruth),
        "overlap_boundary_duplicates": {
            "duplicate_timestamp_count": len(discontinuities),
            "max_translation_difference": max(duplicate_values, default=0.0),
            "median_translation_difference": float(
                np.median(duplicate_values) if duplicate_values else 0.0
            ),
            "records": discontinuities,
        },
    }
    (args.output_dir / "trajectory_evaluation.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )

    time_axis = matched_times - matched_times[0]
    figure, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes[0, 0].plot(matched_groundtruth[:, 0], matched_groundtruth[:, 1], label="TUM GT")
    axes[0, 0].plot(se3[:, 0], se3[:, 1], label="VGGT-SLAM SE(3)")
    axes[0, 0].set(title="XY trajectory — SE(3) alignment", xlabel="x [m]", ylabel="y [m]")
    axes[0, 0].axis("equal")
    axes[0, 0].legend()

    axes[0, 1].plot(matched_groundtruth[:, 0], matched_groundtruth[:, 1], label="TUM GT")
    axes[0, 1].plot(sim3[:, 0], sim3[:, 1], label="VGGT-SLAM Sim(3)")
    axes[0, 1].set(title=f"XY trajectory — Sim(3), scale={sim3_scale:.4f}", xlabel="x [m]", ylabel="y [m]")
    axes[0, 1].axis("equal")
    axes[0, 1].legend()

    axes[1, 0].plot(time_axis, se3_errors, label="SE(3)")
    axes[1, 0].plot(time_axis, sim3_errors, label="Sim(3)")
    axes[1, 0].set(title="Absolute translation error", xlabel="time [s]", ylabel="error [m]")
    axes[1, 0].legend()

    axes[1, 1].plot(matched_groundtruth[:, 0], matched_groundtruth[:, 2], label="TUM GT")
    axes[1, 1].plot(sim3[:, 0], sim3[:, 2], label="VGGT-SLAM Sim(3)")
    axes[1, 1].set(title="XZ trajectory — Sim(3) alignment", xlabel="x [m]", ylabel="z [m]")
    axes[1, 1].axis("equal")
    axes[1, 1].legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "trajectory_evaluation.png", dpi=180)
    plt.close(figure)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
