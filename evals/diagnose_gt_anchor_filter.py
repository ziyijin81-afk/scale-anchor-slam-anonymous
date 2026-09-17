#!/usr/bin/env python3
"""GT-only diagnostic: remove confidently wrong anchor scale factors.

This script is deliberately not a deployment gate.  It replays one recorded
scale graph twice: first unchanged, then with only anchors whose independently
estimated GT scale error exceeds one predeclared threshold removed.  Overlap
measurements, weights, loop corrections and all VGGT outputs remain fixed.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import io
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from vggt_slam.scale_solver import ScaleFactorGraph
from vggt_slam.trajectory_evaluation import (
    apply_sim3,
    interpolate_reference,
    read_reference_text,
    read_reference_rosbag,
    umeyama,
)


FACTOR_RE = re.compile(
    r"^\[scale-graph\] factor=(overlap|anchor) "
    r"edge=\((\d+),(\d+)\) measurement=([0-9.eE+-]+) "
    r"sigma=([0-9.eE+-]+)",
)
BACKEND_RE = re.compile(r"^\[scale-graph\] backend=")


def parse_final_scales(log_text: str) -> dict[int, float]:
    matches = re.findall(
        r"^\[scale-graph\] backend=.*? after=(\{.*?\}) diagnostics=",
        log_text,
        re.MULTILINE,
    )
    if not matches:
        raise ValueError("No scale-graph snapshot found in run.log")
    return {
        int(node): float(value)
        for node, value in ast.literal_eval(matches[-1]).items()
    }


def replay(
    log_text: str,
    removed: set[tuple[int, int]],
    replacements: dict[tuple[int, int], float] | None = None,
) -> dict[int, float]:
    replacements = replacements or {}
    graph = ScaleFactorGraph(use_isam2=True)
    pending = False
    with contextlib.redirect_stdout(io.StringIO()):
        for line in log_text.splitlines():
            match = FACTOR_RE.match(line)
            if match:
                factor_type, node_i, node_j, measurement, sigma = match.groups()
                edge = int(node_i), int(node_j)
                if factor_type == "anchor" and edge in removed:
                    continue
                factor_measurement = float(measurement)
                if factor_type == "anchor" and edge in replacements:
                    factor_measurement = float(replacements[edge])
                graph.update_factor(
                    factor_type,
                    edge[0],
                    edge[1],
                    factor_measurement,
                    float(sigma),
                )
                pending = True
            elif pending and BACKEND_RE.match(line):
                graph.optimize()
                pending = False
        if pending:
            graph.optimize()
    return dict(graph.nodes)


def rebuild_trajectory(
    poses: np.ndarray,
    nodes: list[int],
    old_scales: dict[int, float],
    new_scales: dict[int, float],
    batch_size: int,
) -> np.ndarray:
    expected_rows = len(nodes) * batch_size
    if poses.shape != (expected_rows, 8):
        raise ValueError(
            f"Expected poses shape ({expected_rows}, 8), got {poses.shape}"
        )
    source = poses.reshape(len(nodes), batch_size, 8)
    rebuilt = source.copy()
    for order, node in enumerate(nodes):
        old_positions = source[order, :, 1:4]
        old_anchor = old_positions[0]
        if order == 0:
            new_anchor = old_anchor.copy()
        else:
            old_parent = source[order - 1, -1, 1:4]
            new_parent = rebuilt[order - 1, -1, 1:4]
            new_anchor = old_anchor + (new_parent - old_parent)
        rebuilt[order, :, 1:4] = new_anchor + (
            new_scales[node] / old_scales[node]
        ) * (old_positions - old_anchor)
    return rebuilt.reshape(expected_rows, 8)


def load_gt_submap_scales(
    csv_path: Path,
    nodes: list[int],
) -> tuple[dict[int, float], dict[int, float]]:
    rows = list(csv.DictReader(csv_path.open()))
    if len(rows) != len(nodes) - 1:
        raise ValueError(
            f"Expected {len(nodes) - 1} GT overlap rows, found {len(rows)}"
        )
    relative_by_order = {0: 1.0}
    rmse_by_order: dict[int, float] = {}
    for row in rows:
        previous = int(row["previous_submap"])
        current = int(row["current_submap"])
        if current != previous + 1 or previous not in relative_by_order:
            raise ValueError(f"Non-contiguous GT overlap row: {row}")
        relative_by_order[current] = (
            relative_by_order[previous] * float(row["gt_scale"])
        )
        rmse_by_order[previous] = max(
            rmse_by_order.get(previous, 0.0),
            float(row["previous_fit_rmse_m"]),
        )
        rmse_by_order[current] = max(
            rmse_by_order.get(current, 0.0),
            float(row["current_fit_rmse_m"]),
        )
    return (
        {node: relative_by_order[order] for order, node in enumerate(nodes)},
        {node: rmse_by_order[order] for order, node in enumerate(nodes)},
    )


def classify_anchors(
    log_text: str,
    gt_scales: dict[int, float],
    gt_rmse: dict[int, float],
    max_gt_fit_rmse_m: float,
    max_multiplicative_error: float,
) -> tuple[list[dict], set[tuple[int, int]]]:
    records = []
    removed = set()
    for line in log_text.splitlines():
        match = FACTOR_RE.match(line)
        if not match or match.group(1) != "anchor":
            continue
        node_i, node_j = int(match.group(2)), int(match.group(3))
        measurement = float(match.group(4))
        gt_measurement = gt_scales[node_j] / gt_scales[node_i]
        multiplicative_error = math.exp(abs(math.log(measurement / gt_measurement)))
        endpoint_rmse = max(gt_rmse[node_i], gt_rmse[node_j])
        reliable = endpoint_rmse <= max_gt_fit_rmse_m
        reject = reliable and multiplicative_error > max_multiplicative_error
        if reject:
            removed.add((node_i, node_j))
        records.append({
            "node_i": node_i,
            "node_j": node_j,
            "measurement": measurement,
            "gt_measurement": gt_measurement,
            "multiplicative_error": multiplicative_error,
            "maximum_endpoint_gt_fit_rmse_m": endpoint_rmse,
            "reliable_gt_alignment": reliable,
            "removed": reject,
        })
    return records, removed


def _deduplicate_last(rows: np.ndarray) -> np.ndarray:
    _, reverse_indices = np.unique(rows[::-1, 0], return_index=True)
    keep = np.sort(len(rows) - 1 - reverse_indices)
    return rows[keep]


def _associate_reference(
    poses: np.ndarray,
    reference_type: str,
    reference_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    poses = _deduplicate_last(poses)
    if reference_type == "indoor":
        transforms = np.loadtxt(reference_path).reshape(-1, 4, 4)
        frame_ids = np.rint(poses[:, 0]).astype(int) + 8
        valid = (frame_ids >= 0) & (frame_ids < len(transforms))
        return (
            poses[valid, 0],
            poses[valid, 1:4],
            transforms[frame_ids[valid], :3, 3],
        )
    if reference_type == "rosbag":
        reference_timestamps, reference_positions = read_reference_rosbag(
            reference_path
        )
        query = np.rint(poses[:, 0]).astype(np.int64)
        valid = (
            (query >= reference_timestamps[0])
            & (query <= reference_timestamps[-1])
        )
        query = query[valid]
        return (
            query,
            poses[valid, 1:4],
            interpolate_reference(
                query, reference_timestamps, reference_positions
            ),
        )
    if reference_type == "trajectory":
        reference_timestamps, reference_positions = read_reference_text(
            reference_path
        )
        timestamps = np.asarray(poses[:, 0], dtype=float)
        if float(np.nanmedian(np.abs(timestamps))) < 1e12:
            timestamps = timestamps * 1e9
        query = np.rint(timestamps).astype(np.int64)
        valid = (
            (query >= reference_timestamps[0])
            & (query <= reference_timestamps[-1])
        )
        query = query[valid]
        return (
            query,
            poses[valid, 1:4],
            interpolate_reference(
                query, reference_timestamps, reference_positions
            ),
        )
    raise ValueError(f"Unsupported reference type: {reference_type}")


def evaluate(
    poses: np.ndarray,
    reference_type: str,
    reference_path: Path,
) -> tuple[dict, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    timestamps, estimate, reference = _associate_reference(
        poses, reference_type, reference_path
    )
    estimate_mean = estimate.mean(axis=0)
    reference_mean = reference.mean(axis=0)
    estimate_centered = estimate - estimate_mean
    reference_centered = reference - reference_mean
    covariance = reference_centered.T @ estimate_centered / len(estimate)
    left, _, right_t = np.linalg.svd(covariance)
    reflection = np.eye(3)
    if np.linalg.det(left @ right_t) < 0:
        reflection[-1, -1] = -1
    se3_rotation = left @ reflection @ right_t
    se3_translation = reference_mean - se3_rotation @ estimate_mean
    se3 = apply_sim3(estimate, 1.0, se3_rotation, se3_translation)
    se3_errors = np.linalg.norm(se3 - reference, axis=1)
    scale, rotation, translation = umeyama(estimate, reference)
    sim3 = apply_sim3(estimate, scale, rotation, translation)
    start_translation = reference[0] - scale * (rotation @ estimate[0])
    start_aligned = apply_sim3(estimate, scale, rotation, start_translation)
    sim3_errors = np.linalg.norm(sim3 - reference, axis=1)
    start_errors = np.linalg.norm(start_aligned - reference, axis=1)
    result = {
        "matched_poses": int(len(estimate)),
        "se3_ate_rmse_m": float(np.sqrt(np.mean(se3_errors ** 2))),
        "sim3_scale": float(scale),
        "sim3_ate_rmse_m": float(np.sqrt(np.mean(sim3_errors ** 2))),
        "start_aligned_ate_rmse_m": float(np.sqrt(np.mean(start_errors ** 2))),
        "sim3_path_length_m": float(
            np.linalg.norm(np.diff(sim3, axis=0), axis=1).sum()
        ),
        "reference_path_length_m": float(
            np.linalg.norm(np.diff(reference, axis=0), axis=1).sum()
        ),
    }
    return result, (timestamps, reference, sim3, sim3_errors)


def save_comparison_plot(
    path: Path,
    baseline: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    filtered: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    replaced: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    base_t, reference, base_trajectory, base_errors = baseline
    filtered_t, _, filtered_trajectory, filtered_errors = filtered
    replaced_t, _, replaced_trajectory, replaced_errors = replaced
    axes[0].plot(reference[:, 0], reference[:, 1], "k-", linewidth=2, label="GT")
    axes[0].plot(base_trajectory[:, 0], base_trajectory[:, 1], label="all anchors")
    axes[0].plot(
        filtered_trajectory[:, 0], filtered_trajectory[:, 1],
        label="bad anchor removed",
    )
    axes[0].plot(
        replaced_trajectory[:, 0], replaced_trajectory[:, 1],
        label="anchor measurement replaced by GT",
    )
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].legend()
    axes[0].set_title("Trajectory after global Sim(3) alignment")
    base_elapsed = np.arange(len(base_t)) if len(base_t) < 2 else base_t - base_t[0]
    filtered_elapsed = (
        np.arange(len(filtered_t)) if len(filtered_t) < 2
        else filtered_t - filtered_t[0]
    )
    axes[1].plot(base_elapsed, base_errors, label="all anchors")
    axes[1].plot(filtered_elapsed, filtered_errors, label="filtered")
    replaced_elapsed = (
        np.arange(len(replaced_t)) if len(replaced_t) < 2
        else replaced_t - replaced_t[0]
    )
    axes[1].plot(replaced_elapsed, replaced_errors, label="GT replaced")
    axes[1].set_xlabel("timestamp offset")
    axes[1].set_ylabel("translation error [m]")
    axes[1].set_title("Absolute translation error")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--overlap-gt-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=17)
    parser.add_argument("--max-gt-fit-rmse-m", type=float, default=0.2)
    parser.add_argument("--max-multiplicative-error", type=float, default=1.2)
    parser.add_argument(
        "--reference-type",
        choices=("indoor", "rosbag", "trajectory"),
        required=True,
    )
    parser.add_argument("--reference-path", required=True, type=Path)
    args = parser.parse_args()

    log_text = (args.run_dir / "run.log").read_text(errors="replace")
    source_scales = parse_final_scales(log_text)
    nodes = sorted(source_scales)
    gt_scales, gt_rmse = load_gt_submap_scales(args.overlap_gt_csv, nodes)
    anchors, removed = classify_anchors(
        log_text,
        gt_scales,
        gt_rmse,
        args.max_gt_fit_rmse_m,
        args.max_multiplicative_error,
    )
    unchanged_scales = replay(log_text, set())
    filtered_scales = replay(log_text, removed)
    replacements = {
        (record["node_i"], record["node_j"]): record["gt_measurement"]
        for record in anchors
        if record["removed"]
    }
    replaced_scales = replay(log_text, set(), replacements=replacements)
    if any(
        sorted(scales) != nodes
        for scales in (unchanged_scales, filtered_scales, replaced_scales)
    ):
        raise RuntimeError("Replayed scale graph nodes differ from the source run")

    source_poses = np.loadtxt(args.run_dir / "poses.txt", ndmin=2)
    unchanged_poses = rebuild_trajectory(
        source_poses, nodes, source_scales, unchanged_scales, args.batch_size
    )
    filtered_poses = rebuild_trajectory(
        source_poses, nodes, source_scales, filtered_scales, args.batch_size
    )
    replaced_poses = rebuild_trajectory(
        source_poses, nodes, source_scales, replaced_scales, args.batch_size
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output_dir / "poses_all_anchors.txt", unchanged_poses, fmt="%.8f")
    np.savetxt(args.output_dir / "poses_gt_filtered_anchors.txt", filtered_poses, fmt="%.8f")
    np.savetxt(args.output_dir / "poses_gt_replaced_anchors.txt", replaced_poses, fmt="%.8f")
    unchanged_metrics, unchanged_plot = evaluate(
        unchanged_poses, args.reference_type, args.reference_path
    )
    filtered_metrics, filtered_plot = evaluate(
        filtered_poses, args.reference_type, args.reference_path
    )
    replaced_metrics, replaced_plot = evaluate(
        replaced_poses, args.reference_type, args.reference_path
    )
    save_comparison_plot(
        args.output_dir / "trajectory_comparison.png",
        unchanged_plot,
        filtered_plot,
        replaced_plot,
    )
    with (args.output_dir / "anchor_gt_diagnostics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(anchors[0]))
        writer.writeheader()
        writer.writerows(anchors)
    summary = {
        "diagnostic_only": True,
        "source_run": str(args.run_dir),
        "controlled_variables": (
            "same keyframes, VGGT outputs, overlap factors, factor sigmas and loops; "
            "only confidently wrong anchor factors are removed"
        ),
        "predeclared_rule": {
            "maximum_endpoint_gt_fit_rmse_m": args.max_gt_fit_rmse_m,
            "maximum_allowed_anchor_multiplicative_error": args.max_multiplicative_error,
        },
        "anchor_count": len(anchors),
        "reliable_anchor_count": sum(a["reliable_gt_alignment"] for a in anchors),
        "removed_anchor_count": len(removed),
        "removed_anchor_edges": [list(edge) for edge in sorted(removed)],
        "gt_replacement_measurements": {
            f"{edge[0]}->{edge[1]}": value
            for edge, value in sorted(replacements.items())
        },
        "all_anchors": unchanged_metrics,
        "gt_filtered_anchors": filtered_metrics,
        "gt_replaced_anchors": replaced_metrics,
        "sim3_ate_change_percent": 100.0 * (
            filtered_metrics["sim3_ate_rmse_m"]
            / unchanged_metrics["sim3_ate_rmse_m"]
            - 1.0
        ),
        "start_aligned_ate_change_percent": 100.0 * (
            filtered_metrics["start_aligned_ate_rmse_m"]
            / unchanged_metrics["start_aligned_ate_rmse_m"]
            - 1.0
        ),
        "gt_replaced_vs_all_percent": {
            metric: 100.0 * (
                replaced_metrics[metric] / unchanged_metrics[metric] - 1.0
            )
            for metric in (
                "se3_ate_rmse_m",
                "sim3_ate_rmse_m",
                "start_aligned_ate_rmse_m",
            )
        },
        "source_scales": source_scales,
        "unchanged_replay_scales": unchanged_scales,
        "filtered_replay_scales": filtered_scales,
        "gt_replaced_replay_scales": replaced_scales,
    }
    (args.output_dir / "anchor_gt_filter_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
