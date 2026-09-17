#!/usr/bin/env python3
"""Evaluate KITTI trajectory shape and local scale after one global Sim(3) fit."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_estimate(path: Path):
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] < 4:
        raise ValueError("Estimate must contain: frame_id x y z ...")
    frame_ids = data[:, 0].astype(int)
    # VGGT-SLAM logs the one-frame submap overlap twice and pads the final
    # submap by repeating its last frame. Keep one pose per KITTI frame.
    _, first = np.unique(frame_ids, return_index=True)
    first.sort()
    return frame_ids[first], data[first, 1:4]


def load_kitti_poses(path: Path):
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] != 12:
        raise ValueError("KITTI pose file must have 12 values per row")
    return data.reshape(-1, 3, 4)[:, :, 3]


def umeyama(source, target):
    """Return scale, rotation, translation mapping source onto target."""
    src_mean = source.mean(axis=0)
    dst_mean = target.mean(axis=0)
    src = source - src_mean
    dst = target - dst_mean
    covariance = dst.T @ src / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    variance = np.mean(np.sum(src * src, axis=1))
    scale = np.trace(np.diag(singular) @ correction) / variance
    translation = dst_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimate", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--window", type=int, default=10,
                        help="Number of estimated keyframe intervals per local scale sample")
    args = parser.parse_args()

    frame_ids, estimate = load_estimate(args.estimate)
    ground_truth_all = load_kitti_poses(args.ground_truth)
    valid = (frame_ids >= 0) & (frame_ids < len(ground_truth_all))
    frame_ids, estimate = frame_ids[valid], estimate[valid]
    ground_truth = ground_truth_all[frame_ids]
    if len(estimate) <= args.window:
        raise ValueError("Not enough matched poses for the requested window")

    global_scale, rotation, translation = umeyama(estimate, ground_truth)
    aligned = global_scale * (estimate @ rotation.T) + translation
    errors = np.linalg.norm(aligned - ground_truth, axis=1)
    ate_rmse = float(np.sqrt(np.mean(errors ** 2)))

    gt_steps = np.linalg.norm(np.diff(ground_truth, axis=0), axis=1)
    est_steps = np.linalg.norm(np.diff(aligned, axis=0), axis=1)
    gt_distance = np.r_[0.0, np.cumsum(gt_steps)]
    centers, multipliers = [], []
    for start in range(0, len(gt_steps) - args.window + 1, args.window):
        stop = start + args.window
        est_length = est_steps[start:stop].sum()
        if est_length > 1e-9:
            centers.append((gt_distance[start] + gt_distance[stop]) / 2)
            multipliers.append(gt_steps[start:stop].sum() / est_length)
    centers = np.asarray(centers)
    multipliers = np.asarray(multipliers)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.output_dir / "local_scale.csv",
        np.c_[centers, multipliers],
        delimiter=",",
        header="gt_distance_m,local_scale_multiplier",
        comments="",
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(ground_truth[:, 0], ground_truth[:, 2], label="KITTI GT", linewidth=2)
    axes[0].plot(aligned[:, 0], aligned[:, 2], label="VGGT-SLAM 2.0", linewidth=1)
    axes[0].set(title="Trajectory after one global Sim(3)", xlabel="x [m]", ylabel="z [m]")
    axes[0].axis("equal")
    axes[0].legend()

    axes[1].plot(gt_distance, errors)
    axes[1].set(title=f"Position error (ATE RMSE {ate_rmse:.1f} m)",
                xlabel="GT traveled distance [m]", ylabel="error [m]")

    axes[2].plot(centers, multipliers, marker=".", linewidth=1)
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[2].set(title="Local scale after global scale alignment",
                xlabel="GT traveled distance [m]", ylabel="local scale multiplier")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "kitti_scale_diagnostics.png", dpi=180)
    plt.close(fig)

    summary = {
        "matched_keyframes": int(len(estimate)),
        "global_sim3_scale": float(global_scale),
        "ate_rmse_m": ate_rmse,
        "gt_sampled_path_length_m": float(gt_distance[-1]),
        "local_scale_window_keyframes": args.window,
        "local_scale_median": float(np.median(multipliers)),
        "local_scale_p05": float(np.percentile(multipliers, 5)),
        "local_scale_p95": float(np.percentile(multipliers, 95)),
        "local_scale_min": float(multipliers.min()),
        "local_scale_max": float(multipliers.max()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
