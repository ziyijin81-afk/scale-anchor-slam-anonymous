#!/usr/bin/env python3
"""Compare VGGT-SLAM trajectories with a ROS 2 /Odometry reference."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rosbags.highlevel import AnyReader


def read_odometry(bag_dir):
    timestamps, positions = [], []
    with AnyReader([Path(bag_dir)]) as reader:
        connections = [c for c in reader.connections if c.topic == "/Odometry"]
        if not connections:
            raise ValueError(f"No /Odometry topic in {bag_dir}")
        for connection, _, rawdata in reader.messages(connections=connections):
            message = reader.deserialize(rawdata, connection.msgtype)
            stamp = message.header.stamp
            point = message.pose.pose.position
            timestamps.append(int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec))
            positions.append((point.x, point.y, point.z))
    return np.asarray(timestamps, dtype=np.int64), np.asarray(positions, dtype=float)


def read_estimate(path):
    values = np.loadtxt(path, ndmin=2)
    timestamps = values[:, 0].astype(np.int64)
    # Shared submap frames occur twice in poses.txt. Keep their final occurrence.
    _, reverse_indices = np.unique(timestamps[::-1], return_index=True)
    keep = np.sort(len(timestamps) - 1 - reverse_indices)
    return timestamps[keep], values[keep, 1:4], values


def interpolate_reference(query_timestamps, reference_timestamps, reference_positions):
    positions = np.column_stack([
        np.interp(query_timestamps, reference_timestamps, reference_positions[:, axis])
        for axis in range(3)
    ])
    return positions


def umeyama(source, target):
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1, -1] = -1
    rotation = u @ sign @ vt
    variance = np.sum(source_centered * source_centered) / len(source)
    scale = np.trace(np.diag(singular_values) @ sign) / variance
    translation = target_mean - scale * (rotation @ source_mean)
    return float(scale), rotation, translation


def apply_sim3(points, scale, rotation, translation):
    return (scale * (rotation @ points.T)).T + translation


def fixed_distance_rpe(aligned, reference, distance=5.0):
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(reference, axis=0), axis=1))))
    errors = []
    for start in range(len(reference) - 1):
        end = int(np.searchsorted(arc, arc[start] + distance))
        if end >= len(reference):
            continue
        reference_delta = reference[end] - reference[start]
        estimate_delta = aligned[end] - aligned[start]
        errors.append(np.linalg.norm(estimate_delta - reference_delta) / distance)
    return np.asarray(errors)


def submap_scales(raw_rows, reference_timestamps, reference_positions, sim3_scale, submap_frames=17):
    records = []
    for submap_index, start in enumerate(range(0, len(raw_rows), submap_frames)):
        rows = raw_rows[start:start + submap_frames]
        timestamps = rows[:, 0].astype(np.int64)
        _, unique_indices = np.unique(timestamps, return_index=True)
        rows = rows[np.sort(unique_indices)]
        timestamps = rows[:, 0].astype(np.int64)
        in_range = (timestamps >= reference_timestamps[0]) & (timestamps <= reference_timestamps[-1])
        timestamps, estimate = timestamps[in_range], rows[in_range, 1:4]
        if len(estimate) < 2:
            continue
        reference = interpolate_reference(timestamps, reference_timestamps, reference_positions)
        estimate_distances, reference_distances = [], []
        for left in range(len(estimate)):
            for right in range(left + 1, len(estimate)):
                reference_distance = np.linalg.norm(reference[right] - reference[left])
                estimate_distance = np.linalg.norm(estimate[right] - estimate[left])
                if reference_distance >= 0.10 and estimate_distance > np.finfo(float).eps:
                    reference_distances.append(reference_distance)
                    estimate_distances.append(estimate_distance)
        reference_distances = np.asarray(reference_distances)
        estimate_distances = np.asarray(estimate_distances)
        conversion = float(np.median(reference_distances / estimate_distances)) if len(estimate_distances) else np.nan
        # After one global Sim(3), one is ideal. Variation around one is residual scale drift.
        local_scale_ratio = float(sim3_scale / conversion) if np.isfinite(conversion) else np.nan
        records.append({
            "submap": submap_index,
            "frames": len(estimate),
            "reference_path_m": float(np.linalg.norm(np.diff(reference, axis=0), axis=1).sum()),
            "reference_span_m": float(np.max(reference_distances)) if len(reference_distances) else 0.0,
            "metric_conversion_gt_per_est": conversion,
            "aligned_local_scale_ratio": local_scale_ratio,
        })
    return records


def evaluate(path, reference_timestamps, reference_positions, submap_frames, align_start=False):
    timestamps, estimate, raw_rows = read_estimate(path)
    in_range = (timestamps >= reference_timestamps[0]) & (timestamps <= reference_timestamps[-1])
    timestamps, estimate = timestamps[in_range], estimate[in_range]
    reference = interpolate_reference(timestamps, reference_timestamps, reference_positions)
    scale, rotation, translation = umeyama(estimate, reference)
    if align_start:
        # Keep the globally estimated scale and rotation, but anchor translation
        # at the first timestamp instead of minimizing translation error over
        # the complete trajectory.
        translation = reference[0] - scale * (rotation @ estimate[0])
    aligned = apply_sim3(estimate, scale, rotation, translation)
    ate = np.linalg.norm(aligned - reference, axis=1)
    rpe = fixed_distance_rpe(aligned, reference)
    scales = submap_scales(
        raw_rows, reference_timestamps, reference_positions, scale, submap_frames=submap_frames
    )
    observable = np.asarray([
        row["aligned_local_scale_ratio"]
        for row in scales
        if row["reference_span_m"] >= 0.5 and row["aligned_local_scale_ratio"] > 0
    ])
    summary = {
        "estimate": str(path),
        "translation_alignment": "first_matched_pose" if align_start else "global_centroid",
        "matched_poses": int(len(estimate)),
        "sim3_scale": scale,
        "ate_rmse_m": float(np.sqrt(np.mean(ate * ate))),
        "ate_median_m": float(np.median(ate)),
        "ate_p95_m": float(np.percentile(ate, 95)),
        "rpe_5m_mean_percent": float(100.0 * np.mean(rpe)),
        "rpe_5m_rmse_percent": float(100.0 * np.sqrt(np.mean(rpe * rpe))),
        "aligned_path_length_m": float(np.linalg.norm(np.diff(aligned, axis=0), axis=1).sum()),
        "reference_path_length_m": float(np.linalg.norm(np.diff(reference, axis=0), axis=1).sum()),
        "observable_scale_submaps": int(len(observable)),
        "local_scale_log_rmse": float(np.sqrt(np.mean(np.log(observable) ** 2))),
        "local_scale_ratio_std": float(np.std(observable)),
        "local_scale_ratio_median": float(np.median(observable)),
    }
    return summary, scales, timestamps, reference, aligned, ate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--on", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--submap-frames", type=int, default=17)
    parser.add_argument(
        "--align-start",
        action="store_true",
        help=(
            "Keep the global Sim(3) scale/rotation, but choose translation so "
            "the first matched estimate and reference positions coincide"
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference_timestamps, reference_positions = read_odometry(args.bag)
    results = {}
    plot_data = {}
    for label, path in (("anchor_off", args.off), ("anchor_on", args.on)):
        result = evaluate(
            path,
            reference_timestamps,
            reference_positions,
            args.submap_frames,
            align_start=args.align_start,
        )
        summary, scales, timestamps, reference, aligned, ate = result
        results[label] = summary
        plot_data[label] = (timestamps, reference, aligned, ate, scales)
        with (args.output_dir / f"{label}_submap_scale.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=scales[0].keys())
            writer.writeheader()
            writer.writerows(scales)

    results["comparison"] = {
        "ate_rmse_change_percent": 100.0 * (
            results["anchor_on"]["ate_rmse_m"] / results["anchor_off"]["ate_rmse_m"] - 1.0
        ),
        "local_scale_log_rmse_change_percent": 100.0 * (
            results["anchor_on"]["local_scale_log_rmse"]
            / results["anchor_off"]["local_scale_log_rmse"] - 1.0
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")

    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    reference = plot_data["anchor_off"][1]
    axes[0].plot(reference[:, 0], reference[:, 1], "k-", label="Odometry reference", linewidth=2)
    for label, color in (("anchor_off", "tab:orange"), ("anchor_on", "tab:blue")):
        _, _, aligned, _, scales = plot_data[label]
        axes[0].plot(aligned[:, 0], aligned[:, 1], color=color, label=label, alpha=0.9)
        valid_scales = [(r["submap"], r["aligned_local_scale_ratio"]) for r in scales if r["reference_span_m"] >= 0.5]
        axes[2].plot(*zip(*valid_scales), marker="o", markersize=3, color=color, label=label)
    axes[0].set_title(
        "Trajectory after global Sim(3), start aligned"
        if args.align_start else "Trajectory after global Sim(3)"
    )
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].legend()

    start_time = plot_data["anchor_off"][0][0]
    for label, color in (("anchor_off", "tab:orange"), ("anchor_on", "tab:blue")):
        timestamps, _, _, ate, _ = plot_data[label]
        axes[1].plot((timestamps - start_time) / 1e9, ate, color=color, label=label)
    axes[1].set_title("Absolute translation error")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("error [m]")
    axes[1].legend()

    axes[2].axhline(1.0, color="k", linestyle="--", linewidth=1)
    axes[2].set_title("Residual local scale (1 is ideal)")
    axes[2].set_xlabel("submap")
    axes[2].set_ylabel("aligned estimated / reference scale")
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "anchor_on_off_odometry_comparison.png", dpi=180)
    plt.close(figure)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
