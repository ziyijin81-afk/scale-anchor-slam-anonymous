#!/usr/bin/env python3
"""Compare the same frames before and after one joint two-image VGGT call."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vggt_slam.project_paths import prefer_bundled_third_party


prefer_bundled_third_party()

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from evals.visualize_joint_vggt_pair import equalize_3d_axes
from evals.visualize_prejoint_reference_clouds import read_submap_batches
from vggt_slam.scale_solver import camera_points_from_depth, robust_sample_statistics
from vggt_slam.solver import _short_batch_camera_head_fp32


def infer(model, images, log_prefix):
    with torch.no_grad():
        with _short_batch_camera_head_fp32(model, images.shape[0], log_prefix):
            prediction = model(images)
    _, intrinsic = pose_encoding_to_extri_intri(
        prediction["pose_enc"], images.shape[-2:]
    )
    return {
        "depth": prediction["depth"][0].detach().float().cpu().numpy(),
        "confidence": prediction["depth_conf"][0].detach().float().cpu().numpy(),
        "intrinsic": intrinsic[0].detach().float().cpu().numpy(),
    }


def frame_arrays(prediction, frame_index):
    depth = np.asarray(prediction["depth"][frame_index], dtype=float).squeeze()
    confidence = np.asarray(prediction["confidence"][frame_index], dtype=float).squeeze()
    points = camera_points_from_depth(depth, prediction["intrinsic"][frame_index])
    return depth, confidence, points


def compare_frame(original, original_index, joint, joint_index, percentile):
    original_depth, original_conf, original_points = frame_arrays(original, original_index)
    joint_depth, joint_conf, joint_points = frame_arrays(joint, joint_index)
    original_threshold = float(np.percentile(original["confidence"], percentile) + 1e-6)
    joint_threshold = float(np.percentile(joint["confidence"], percentile) + 1e-6)

    original_radius = np.linalg.norm(original_points, axis=-1)
    joint_radius = np.linalg.norm(joint_points, axis=-1)
    valid = original_conf > original_threshold
    valid &= joint_conf > joint_threshold
    valid &= np.isfinite(original_depth) & np.isfinite(joint_depth)
    valid &= np.all(np.isfinite(original_points), axis=-1)
    valid &= np.all(np.isfinite(joint_points), axis=-1)
    valid &= (original_depth > 0) & (joint_depth > 0)
    valid &= (original_radius > 1e-12) & (joint_radius > 1e-12)

    radius_ratio = original_radius[valid] / joint_radius[valid]
    depth_ratio = original_depth[valid] / joint_depth[valid]
    radius_stats = robust_sample_statistics(radius_ratio)
    depth_stats = robust_sample_statistics(depth_ratio)
    scale = float(radius_stats["median"])
    aligned_joint = scale * joint_points[valid]
    original_selected = original_points[valid]
    relative_vector_error = np.linalg.norm(
        original_selected - aligned_joint, axis=1
    ) / np.maximum(np.linalg.norm(original_selected, axis=1), 1e-12)
    original_unit = original_selected / np.maximum(
        np.linalg.norm(original_selected, axis=1, keepdims=True), 1e-12
    )
    joint_unit = aligned_joint / np.maximum(
        np.linalg.norm(aligned_joint, axis=1, keepdims=True), 1e-12
    )
    angular_error = np.degrees(np.arccos(np.clip(
        np.sum(original_unit * joint_unit, axis=1), -1.0, 1.0
    )))

    ratio_map = np.full(original_depth.shape, np.nan, dtype=float)
    ratio_map[valid] = np.log(radius_ratio / scale)
    return {
        "original_depth": original_depth,
        "joint_depth": joint_depth,
        "original_points": original_selected,
        "joint_points": joint_points[valid],
        "aligned_joint_points": aligned_joint,
        "valid": valid,
        "radius_ratio": radius_ratio,
        "ratio_map": ratio_map,
        "stats": {
            "valid_pixels": int(np.count_nonzero(valid)),
            "original_confidence_threshold": original_threshold,
            "joint_confidence_threshold": joint_threshold,
            "radius_scale_original_over_joint": scale,
            "radius_ratio_mad": float(radius_stats["mad"]),
            "radius_ratio_relative_mad": float(radius_stats["mad"] / scale),
            "radius_ratio_outlier_ratio": float(radius_stats["outlier_ratio"]),
            "depth_scale_original_over_joint": float(depth_stats["median"]),
            "depth_ratio_relative_mad": float(depth_stats["mad"] / depth_stats["median"]),
            "within_5_percent": float(np.mean(np.abs(radius_ratio / scale - 1.0) <= 0.05)),
            "within_10_percent": float(np.mean(np.abs(radius_ratio / scale - 1.0) <= 0.10)),
            "aligned_vector_error_median": float(np.median(relative_vector_error)),
            "aligned_vector_error_p90": float(np.percentile(relative_vector_error, 90)),
            "ray_angular_error_median_deg": float(np.median(angular_error)),
            "ray_angular_error_p90_deg": float(np.percentile(angular_error, 90)),
        },
    }


def plot_diagnostics(path, images, comparisons, labels):
    figure, axes = plt.subplots(2, 5, figsize=(22, 9))
    limit = np.log(1.25)
    for row, (image, result, label) in enumerate(zip(images, comparisons, labels)):
        axes[row, 0].imshow(np.clip(image, 0.0, 1.0))
        axes[row, 0].set_title(label)
        axes[row, 0].axis("off")

        depths = np.concatenate((
            result["original_depth"][result["valid"]],
            result["joint_depth"][result["valid"]],
        ))
        lower, upper = np.percentile(depths, (2, 98))
        axes[row, 1].imshow(result["original_depth"], cmap="turbo", vmin=lower, vmax=upper)
        axes[row, 1].set_title("Original 17-frame depth")
        axes[row, 1].axis("off")
        axes[row, 2].imshow(result["joint_depth"], cmap="turbo", vmin=lower, vmax=upper)
        axes[row, 2].set_title("Joint two-frame depth")
        axes[row, 2].axis("off")

        heatmap = axes[row, 3].imshow(
            result["ratio_map"], cmap="coolwarm", vmin=-limit, vmax=limit
        )
        axes[row, 3].set_title("log(pixel scale / median)")
        axes[row, 3].axis("off")
        figure.colorbar(heatmap, ax=axes[row, 3], fraction=0.046, pad=0.04)

        normalized = result["radius_ratio"] / result["stats"]["radius_scale_original_over_joint"]
        axes[row, 4].hist(normalized, bins=120, range=(0.5, 1.5), density=True)
        axes[row, 4].axvline(1.0, color="black", linewidth=1)
        axes[row, 4].axvspan(0.95, 1.05, color="green", alpha=0.15)
        axes[row, 4].set_xlim(0.5, 1.5)
        axes[row, 4].set_title(
            f"q={result['stats']['radius_scale_original_over_joint']:.4f}, "
            f"relMAD={result['stats']['radius_ratio_relative_mad']:.3%}"
        )
        axes[row, 4].set_xlabel("pixel scale / median scale")
    figure.suptitle("Same-frame geometry: original 17-frame inference vs one joint VGGT call")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_cloud_overlays(path, comparisons, labels, max_points):
    rng = np.random.default_rng(0)
    figure = plt.figure(figsize=(16, 14))
    for row, (result, label) in enumerate(zip(comparisons, labels)):
        count = min(len(result["original_points"]), max_points)
        indices = rng.choice(len(result["original_points"]), count, replace=False)
        original = result["original_points"][indices]
        joint = result["joint_points"][indices]
        aligned = result["aligned_joint_points"][indices]
        for column, (other, subtitle) in enumerate((
            (joint, "Before scale alignment"),
            (aligned, "After median scale alignment"),
        )):
            axis = figure.add_subplot(2, 2, row * 2 + column + 1, projection="3d")
            axis.scatter(*original.T, c="red", s=0.2, alpha=0.30, label="original")
            axis.scatter(*other.T, c="blue", s=0.2, alpha=0.30, label="joint")
            combined = np.concatenate((original, other))
            equalize_3d_axes(axis, combined)
            axis.view_init(elev=18, azim=-70)
            axis.set_title(f"{label}\n{subtitle}")
            axis.set(xlabel="X", ylabel="Y", zlabel="Z")
            axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_overlay(path, result):
    original = result["original_points"]
    aligned = result["aligned_joint_points"]
    colors = np.concatenate((
        np.tile((1.0, 0.1, 0.05), (len(original), 1)),
        np.tile((0.05, 0.25, 1.0), (len(aligned), 1)),
    ))
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.concatenate((original, aligned)))
    cloud.colors = o3d.utility.Vector3dVector(colors)
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False):
        raise RuntimeError(f"Failed to write {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-log", required=True, type=Path)
    parser.add_argument("--submap-orders", nargs=2, type=int, default=(0, 1))
    parser.add_argument("--frame-indices", nargs=2, type=int, default=(0, 5))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--confidence-percentile", type=float, default=25.0)
    parser.add_argument("--plot-points-per-frame", type=int, default=30000)
    args = parser.parse_args()

    batches = read_submap_batches(args.run_log)
    if len(batches) <= max(args.submap_orders):
        raise ValueError(f"Only found {len(batches)} batches in {args.run_log}")
    selected_batches = [batches[index] for index in args.submap_orders]
    selected_paths = [
        batch[index] for batch, index in zip(selected_batches, args.frame_indices)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGT()
    weights_url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(weights_url))
    model.eval().to(torch.bfloat16).to(device)

    original_predictions = []
    for order, batch in zip(args.submap_orders, selected_batches):
        images = load_and_preprocess_images(batch).to(device)
        original_predictions.append(infer(model, images, f"[original-submap-{order}]") )
    joint_images = load_and_preprocess_images(selected_paths).to(device)
    selected_images = joint_images.detach().float().cpu().permute(0, 2, 3, 1).numpy()
    joint_prediction = infer(model, joint_images, "[joint-pair]")

    comparisons = [
        compare_frame(original, frame_index, joint_prediction, joint_index, args.confidence_percentile)
        for joint_index, (original, frame_index) in enumerate(
            zip(original_predictions, args.frame_indices)
        )
    ]
    labels = [
        f"submap order {order}, frame index {index}: {Path(path).name}"
        for order, index, path in zip(args.submap_orders, args.frame_indices, selected_paths)
    ]
    plot_diagnostics(
        args.output_dir / "same_frame_scale_diagnostics.png",
        selected_images,
        comparisons,
        labels,
    )
    plot_cloud_overlays(
        args.output_dir / "same_frame_cloud_overlays.png",
        comparisons,
        labels,
        args.plot_points_per_frame,
    )
    for index, result in enumerate(comparisons):
        write_overlay(args.output_dir / f"frame_{index}_aligned_overlay.ply", result)

    q0, qk = [
        result["stats"]["radius_scale_original_over_joint"] for result in comparisons
    ]
    output = {
        "device": device,
        "original_vggt_calls": 2,
        "joint_vggt_calls": 1,
        "joint_call_frame_count": 2,
        "submap_orders": list(args.submap_orders),
        "frame_indices": list(args.frame_indices),
        "images": selected_paths,
        "frames": [result["stats"] for result in comparisons],
        "pointwise_anchor_q0_over_qk": float(q0 / qk),
    }
    (args.output_dir / "same_frame_scale_stats.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
