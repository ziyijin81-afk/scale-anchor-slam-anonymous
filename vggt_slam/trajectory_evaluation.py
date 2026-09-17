"""Optional timestamp-associated trajectory exports against a reference."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _timestamps_to_ns(timestamps):
    timestamps = np.asarray(timestamps, dtype=float)
    magnitude = float(np.nanmedian(np.abs(timestamps)))
    if 1e10 <= magnitude < 1e12:
        # Some image datasets (for example LaMAR iPhone sequences) use an
        # integer image identifier in the timestamp column. Preserve those
        # identifiers exactly instead of overflowing int64 by treating them
        # as Unix seconds.
        return np.rint(timestamps).astype(np.int64)
    if magnitude < 1e12:       # seconds, including TUM timestamps
        timestamps = timestamps * 1e9
    elif magnitude < 1e15:     # milliseconds
        timestamps = timestamps * 1e6
    elif magnitude < 1e17:     # microseconds
        timestamps = timestamps * 1e3
    return np.rint(timestamps).astype(np.int64)


def read_estimated_positions(path):
    values = np.loadtxt(path, ndmin=2)
    timestamps = _timestamps_to_ns(values[:, 0])
    # Overlap frames are written in both neighboring submaps. Keep the final
    # occurrence because it reflects the latest globally replayed placement.
    _, reverse_indices = np.unique(timestamps[::-1], return_index=True)
    keep = np.sort(len(timestamps) - 1 - reverse_indices)
    return timestamps[keep], np.asarray(values[keep, 1:4], dtype=float)


def read_reference_text(path):
    values = np.loadtxt(path, comments="#", ndmin=2)
    if values.shape[1] < 4:
        raise ValueError("Reference trajectory must contain timestamp x y z")
    timestamps = _timestamps_to_ns(values[:, 0])
    order = np.argsort(timestamps)
    timestamps, positions = timestamps[order], np.asarray(values[order, 1:4], dtype=float)
    timestamps, unique_indices = np.unique(timestamps, return_index=True)
    return timestamps, positions[unique_indices]


def read_reference_rosbag(bag_path, topic="/Odometry"):
    from rosbags.highlevel import AnyReader

    timestamps, positions = [], []
    with AnyReader([Path(bag_path)]) as reader:
        connections = [connection for connection in reader.connections if connection.topic == topic]
        if not connections:
            raise ValueError(f"No {topic} topic in {bag_path}")
        for connection, _, rawdata in reader.messages(connections=connections):
            message = reader.deserialize(rawdata, connection.msgtype)
            stamp = message.header.stamp
            point = message.pose.pose.position
            timestamps.append(int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec))
            positions.append((point.x, point.y, point.z))
    timestamps = np.asarray(timestamps, dtype=np.int64)
    positions = np.asarray(positions, dtype=float)
    order = np.argsort(timestamps)
    return timestamps[order], positions[order]


def interpolate_reference(query_timestamps, reference_timestamps, reference_positions):
    return np.column_stack([
        np.interp(query_timestamps, reference_timestamps, reference_positions[:, axis])
        for axis in range(3)
    ])


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
    if variance <= np.finfo(float).eps:
        raise ValueError("Estimated trajectory has no observable spatial extent")
    scale = np.trace(np.diag(singular_values) @ sign) / variance
    translation = target_mean - scale * (rotation @ source_mean)
    return float(scale), rotation, translation


def apply_sim3(points, scale, rotation, translation):
    return (scale * (rotation @ points.T)).T + translation


def _summary(aligned, reference):
    errors = np.linalg.norm(aligned - reference, axis=1)
    return {
        "ate_rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        "ate_median_m": float(np.median(errors)),
        "ate_p95_m": float(np.percentile(errors, 95)),
        "maximum_error_m": float(np.max(errors)),
        "aligned_path_length_m": float(np.linalg.norm(np.diff(aligned, axis=0), axis=1).sum()),
        "reference_path_length_m": float(np.linalg.norm(np.diff(reference, axis=0), axis=1).sum()),
        "first_pose_error_m": float(errors[0]),
    }, errors


def _write_aligned_csv(path, timestamps, reference, aligned, errors):
    with open(path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("timestamp_ns", "reference_x", "reference_y", "reference_z",
                         "estimate_x", "estimate_y", "estimate_z", "translation_error_m"))
        for timestamp, ref, est, error in zip(timestamps, reference, aligned, errors):
            writer.writerow((int(timestamp), *ref.tolist(), *est.tolist(), float(error)))


def _write_plot(path, timestamps, reference, aligned, errors, title):
    elapsed = (timestamps - timestamps[0]) / 1e9
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(reference[:, 0], reference[:, 1], "k-", linewidth=2, label="reference")
    axes[0].plot(aligned[:, 0], aligned[:, 1], color="tab:blue", label="estimate")
    axes[0].scatter(reference[0, 0], reference[0, 1], color="limegreen", s=70,
                    edgecolor="black", zorder=5, label="start")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_title(title)
    axes[0].legend()
    axes[1].plot(elapsed, errors, color="tab:red")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("translation error [m]")
    axes[1].set_title("Absolute translation error")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_trajectory_comparisons(
    estimate_path,
    output_dir,
    reference_rosbag=None,
    reference_trajectory=None,
    reference_topic="/Odometry",
):
    """Write global-Umeyama and first-position-aligned trajectory products."""
    if bool(reference_rosbag) == bool(reference_trajectory):
        raise ValueError("Provide exactly one reference_rosbag or reference_trajectory")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if reference_rosbag:
        reference_timestamps, reference_positions = read_reference_rosbag(
            reference_rosbag, reference_topic
        )
        reference_source = str(reference_rosbag)
    else:
        reference_timestamps, reference_positions = read_reference_text(reference_trajectory)
        reference_source = str(reference_trajectory)

    timestamps, estimate = read_estimated_positions(estimate_path)
    in_range = (
        (timestamps >= reference_timestamps[0])
        & (timestamps <= reference_timestamps[-1])
    )
    timestamps, estimate = timestamps[in_range], estimate[in_range]
    if len(estimate) < 3:
        raise ValueError("Fewer than three timestamp-associated poses overlap the reference")
    reference = interpolate_reference(timestamps, reference_timestamps, reference_positions)
    scale, rotation, centroid_translation = umeyama(estimate, reference)
    start_translation = reference[0] - scale * (rotation @ estimate[0])
    configurations = {
        "umeyama_sim3": centroid_translation,
        "start_aligned": start_translation,
    }
    result = {
        "estimate": str(estimate_path),
        "reference": reference_source,
        "reference_topic": reference_topic if reference_rosbag else None,
        "matched_poses": int(len(estimate)),
        "sim3_scale": scale,
        "alignments": {},
    }
    for name, translation in configurations.items():
        aligned = apply_sim3(estimate, scale, rotation, translation)
        summary, errors = _summary(aligned, reference)
        result["alignments"][name] = summary
        _write_aligned_csv(
            output_dir / f"trajectory_{name}.csv", timestamps, reference, aligned, errors
        )
        _write_plot(
            output_dir / f"trajectory_{name}.png",
            timestamps,
            reference,
            aligned,
            errors,
            "Trajectory after global Umeyama Sim(3)"
            if name == "umeyama_sim3"
            else "Trajectory with first position aligned",
        )
    with open(output_dir / "trajectory_evaluation.json", "w") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        f"[trajectory-eval] directory={output_dir} matched={len(estimate)} "
        f"sim3_rmse={result['alignments']['umeyama_sim3']['ate_rmse_m']:.8g} "
        f"start_rmse={result['alignments']['start_aligned']['ate_rmse_m']:.8g}"
    )
    return result
