#!/usr/bin/env python3
"""Validate scale-anchor measurements against trajectory ground truth.

This is an offline evaluator. Ground truth is never fed back into SLAM or the
factor graph.
"""

import argparse
import ast
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FACTOR_RE = re.compile(
    r"^\[scale-graph\] factor=(overlap|anchor) "
    r"edge=\((\d+),(\d+)\) measurement=([0-9.eE+-]+) "
    r"sigma=([0-9.eE+-]+) weight=([0-9.eE+-]+)",
    re.MULTILINE,
)
ANCHOR_RESIDUAL_RE = re.compile(
    r"^\[scale-graph\] residual .*?type=anchor edge=\(0,(\d+)\) "
    r"residual=([0-9.eE+-]+) normalized=([0-9.eE+-]+)",
    re.MULTILINE,
)


def load_ground_truth(path):
    values = np.loadtxt(path, ndmin=2)
    if values.shape[1] != 12:
        raise ValueError("KITTI ground-truth poses must contain 12 values per row")
    return values.reshape(-1, 3, 4)[:, :, 3]


def parse_final_scales(log_text):
    """Read the last optimized ordinary-scale snapshot from a run log."""
    matches = re.findall(
        r"^\[scale-graph\] backend=.*? after=(\{.*?\}) diagnostics=",
        log_text,
        re.MULTILINE,
    )
    if not matches:
        matches = re.findall(
            r"^\[scale-graph\] optimized .*? after=(\{.*?\})$",
            log_text,
            re.MULTILINE,
        )
    if not matches:
        raise ValueError("No final scale-graph snapshot was found in run.log")
    parsed = ast.literal_eval(matches[-1])
    scales = {int(node): float(value) for node, value in parsed.items()}
    if scales.get(0) != 1.0:
        raise ValueError(f"Expected fixed S_0=1, got {scales.get(0)!r}")
    if any(not np.isfinite(value) or value <= 0 for value in scales.values()):
        raise ValueError(f"Final scale graph contains a non-positive scale: {scales}")
    return scales


def parse_factors(log_text):
    """Return the latest measurement for every typed edge."""
    factors = {}
    for factor_type, node_i, node_j, measurement, sigma, weight in FACTOR_RE.findall(log_text):
        key = factor_type, int(node_i), int(node_j)
        factors[key] = {
            "measurement": float(measurement),
            "sigma": float(sigma),
            "weight": float(weight),
        }
    anchor_residuals = {}
    for node_j, residual, normalized in ANCHOR_RESIDUAL_RE.findall(log_text):
        anchor_residuals[int(node_j)] = {
            "residual": float(residual),
            "normalized_residual": float(normalized),
        }
    return factors, anchor_residuals


def build_overlap_chain(nodes, factors):
    chain = {0: 1.0}
    for prior, current in zip(nodes[:-1], nodes[1:]):
        key = "overlap", prior, current
        if key not in factors:
            raise ValueError(f"Missing overlap factor {key}")
        chain[current] = chain[prior] * factors[key]["measurement"]
    return chain


def derive_gt_relative_scales(poses, gt_positions, nodes, optimized, batch_size):
    """Recover immutable local scale from final intra-submap distances.

    Final distances within submap i have already been multiplied by S_i. After
    division by S_i, the median GT/local-distance ratio q_i gives metres per
    original local unit. The requested conversion into submap0 units is q_i/q_0.
    """
    expected_rows = len(nodes) * batch_size
    if len(poses) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} pose rows ({len(nodes)} nodes x {batch_size}), "
            f"but found {len(poses)}. Use the batch size from the evaluated run."
        )

    statistics = {}
    for order, node in enumerate(nodes):
        block = poses[order * batch_size:(order + 1) * batch_size]
        frame_ids = np.rint(block[:, 0]).astype(int)
        translations = block[:, 1:4]
        # Remove carried overlap/padding duplicates inside this submap while
        # preserving temporal order.
        keep = np.r_[True, frame_ids[1:] != frame_ids[:-1]]
        frame_ids = frame_ids[keep]
        translations = translations[keep]
        valid_ids = (frame_ids >= 0) & (frame_ids < len(gt_positions))
        frame_ids = frame_ids[valid_ids]
        translations = translations[valid_ids]
        if len(frame_ids) < 2:
            raise ValueError(f"Submap {node} has fewer than two distinct GT-matched frames")

        estimated_distance = np.linalg.norm(np.diff(translations, axis=0), axis=1)
        estimated_distance /= optimized[node]
        gt_distance = np.linalg.norm(np.diff(gt_positions[frame_ids], axis=0), axis=1)
        valid = (
            np.isfinite(estimated_distance)
            & np.isfinite(gt_distance)
            & (estimated_distance > 1e-8)
            & (gt_distance > 1e-8)
        )
        ratios = gt_distance[valid] / estimated_distance[valid]
        if ratios.size == 0:
            raise ValueError(f"Submap {node} has no valid translation-scale samples")
        median = float(np.median(ratios))
        mad = float(np.median(np.abs(ratios - median)))
        statistics[node] = {
            "metres_per_local_unit": median,
            "mad": mad,
            "support": int(ratios.size),
            "first_frame": int(frame_ids[0]),
            "last_frame": int(frame_ids[-1]),
        }

    root_scale = statistics[0]["metres_per_local_unit"]
    relative = {
        node: statistics[node]["metres_per_local_unit"] / root_scale
        for node in nodes
    }
    return relative, statistics


def error_summary(prediction, truth, nodes):
    valid_nodes = [node for node in nodes if node != 0 and node in prediction]
    if not valid_nodes:
        return {
            "node_count": 0,
            "median_absolute_relative_error": None,
            "p90_absolute_relative_error": None,
            "median_multiplicative_error": None,
            "p90_multiplicative_error": None,
        }
    ratios = np.asarray([prediction[node] / truth[node] for node in valid_nodes])
    relative = np.abs(ratios - 1.0)
    multiplicative = np.exp(np.abs(np.log(ratios)))
    return {
        "node_count": len(valid_nodes),
        "median_absolute_relative_error": float(np.median(relative)),
        "p90_absolute_relative_error": float(np.percentile(relative, 90)),
        "median_multiplicative_error": float(np.median(multiplicative)),
        "p90_multiplicative_error": float(np.percentile(multiplicative, 90)),
    }


def save_csv(path, rows):
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path, rows, title):
    order = np.asarray([row["submap_order"] for row in rows])
    truth = np.asarray([row["gt_relative_scale"] for row in rows])
    anchor = np.asarray([row["anchor_measurement"] for row in rows])
    overlap = np.asarray([row["overlap_chain_scale"] for row in rows])
    optimized = np.asarray([row["optimized_scale"] for row in rows])

    fig, axes = plt.subplots(2, 2, figsize=(17, 10))
    axes[0, 0].plot(order, truth, color="black", linewidth=2.2, label="GT true scale")
    axes[0, 0].plot(order, anchor, marker=".", label="Anchor measurement")
    axes[0, 0].plot(order, overlap, marker=".", label="Overlap chain")
    axes[0, 0].plot(order, optimized, marker=".", label="Optimized S_i")
    axes[0, 0].set(title="Scale conversion into submap0 units", xlabel="submap order", ylabel="scale")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend()

    axes[0, 1].plot(order, anchor / truth, marker=".", label="Anchor / GT")
    axes[0, 1].plot(order, overlap / truth, marker=".", label="Overlap / GT")
    axes[0, 1].plot(order, optimized / truth, marker=".", label="Optimized / GT")
    axes[0, 1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set(title="Ratio to true scale (ideal = 1)", xlabel="submap order", ylabel="prediction / GT")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend()

    axes[1, 0].plot(order, 100.0 * np.abs(anchor / truth - 1.0), marker=".", label="Anchor")
    axes[1, 0].plot(order, 100.0 * np.abs(overlap / truth - 1.0), marker=".", label="Overlap")
    axes[1, 0].plot(order, 100.0 * np.abs(optimized / truth - 1.0), marker=".", label="Optimized")
    axes[1, 0].set(title="Absolute relative scale error", xlabel="submap order", ylabel="error [%]")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    valid = np.isfinite(anchor)
    sigma = np.asarray([row["anchor_sigma"] for row in rows])
    normalized = np.abs(np.asarray([row["anchor_normalized_residual"] for row in rows]))
    axes[1, 1].plot(order[valid], sigma[valid], color="tab:blue", marker=".", label="Anchor sigma")
    axes[1, 1].set(title="Anchor uncertainty and final residual", xlabel="submap order")
    axes[1, 1].set_ylabel("sigma", color="tab:blue")
    axes[1, 1].tick_params(axis="y", labelcolor="tab:blue")
    axes[1, 1].grid(alpha=0.25)
    residual_axis = axes[1, 1].twinx()
    residual_axis.plot(order[valid], normalized[valid], color="tab:red", marker=".", label="|normalized residual|")
    residual_axis.axhline(3.0, color="tab:red", linestyle="--", linewidth=1, alpha=0.7)
    residual_axis.set_ylabel("|residual / sigma|", color="tab:red")
    residual_axis.tick_params(axis="y", labelcolor="tab:red")
    handles_a, labels_a = axes[1, 1].get_legend_handles_labels()
    handles_b, labels_b = residual_axis.get_legend_handles_labels()
    axes[1, 1].legend(handles_a + handles_b, labels_a + labels_b, loc="upper left")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare scale anchors, overlap-chain scales and optimized S_i against GT"
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=17)
    parser.add_argument("--title", default="Scale-anchor validation against ground truth")
    args = parser.parse_args()

    run_log = args.run_dir / "run.log"
    pose_path = args.run_dir / "poses.txt"
    if not run_log.is_file() or not pose_path.is_file():
        raise FileNotFoundError(f"Expected run.log and poses.txt below {args.run_dir}")
    if args.batch_size <= 1:
        raise ValueError("batch-size must be greater than one")

    output_dir = args.output_dir or args.run_dir / "anchor_scale_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_text = run_log.read_text(errors="replace")
    optimized = parse_final_scales(log_text)
    nodes = sorted(optimized)
    factors, anchor_residuals = parse_factors(log_text)
    anchors = {
        node_j: record["measurement"]
        for (factor_type, node_i, node_j), record in factors.items()
        if factor_type == "anchor" and node_i == 0
    }
    missing = [node for node in nodes if node != 0 and node not in anchors]

    overlap_chain = build_overlap_chain(nodes, factors)
    poses = np.loadtxt(pose_path, ndmin=2)
    if poses.shape[1] < 4:
        raise ValueError("poses.txt must contain frame_id x y z ...")
    gt_positions = load_ground_truth(args.ground_truth)
    gt_relative, gt_statistics = derive_gt_relative_scales(
        poses, gt_positions, nodes, optimized, args.batch_size
    )

    rows = []
    for order, node in enumerate(nodes):
        anchor_record = factors.get(("anchor", 0, node), {})
        residual_record = anchor_residuals.get(node, {})
        anchor_value = anchors.get(node, np.nan)
        rows.append({
            "submap_order": order,
            "submap_id": node,
            "first_frame": gt_statistics[node]["first_frame"],
            "last_frame": gt_statistics[node]["last_frame"],
            "gt_sample_count": gt_statistics[node]["support"],
            "gt_metres_per_local_unit": gt_statistics[node]["metres_per_local_unit"],
            "gt_scale_sample_mad": gt_statistics[node]["mad"],
            "gt_relative_scale": gt_relative[node],
            "anchor_measurement": anchor_value,
            "overlap_chain_scale": overlap_chain[node],
            "optimized_scale": optimized[node],
            "anchor_sigma": anchor_record.get("sigma", np.nan),
            "anchor_weight": anchor_record.get("weight", np.nan),
            "anchor_residual": residual_record.get("residual", np.nan),
            "anchor_normalized_residual": residual_record.get("normalized_residual", np.nan),
            "anchor_ratio_to_gt": anchor_value / gt_relative[node] if node else np.nan,
            "overlap_ratio_to_gt": overlap_chain[node] / gt_relative[node],
            "optimized_ratio_to_gt": optimized[node] / gt_relative[node],
        })

    anchor_summary = error_summary(anchors, gt_relative, nodes)
    overlap_summary = error_summary(overlap_chain, gt_relative, nodes)
    optimized_summary = error_summary(optimized, gt_relative, nodes)
    non_root = nodes[1:]
    anchor_closer = sum(
        abs(np.log(anchors[node] / gt_relative[node]))
        < abs(np.log(overlap_chain[node] / gt_relative[node]))
        for node in non_root if node in anchors
    )
    optimized_closer = sum(
        abs(np.log(optimized[node] / gt_relative[node]))
        < abs(np.log(overlap_chain[node] / gt_relative[node]))
        for node in non_root
    )
    summary = {
        "run_dir": str(args.run_dir),
        "ground_truth": str(args.ground_truth),
        "batch_size": args.batch_size,
        "submap_count": len(nodes),
        "anchor": anchor_summary,
        "overlap_chain": overlap_summary,
        "optimized": optimized_summary,
        "anchor_closer_than_overlap_count": int(anchor_closer),
        "accepted_anchor_count": int(len(anchors)),
        "missing_anchor_nodes": missing,
        "optimized_closer_than_overlap_count": int(optimized_closer),
        "evaluated_non_root_nodes": len(non_root),
        "note": "Ground truth is used only by this offline evaluator.",
    }

    save_csv(output_dir / "anchor_scale_validation.csv", rows)
    (output_dir / "anchor_scale_validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    save_plot(output_dir / "anchor_scale_validation.png", rows, args.title)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
