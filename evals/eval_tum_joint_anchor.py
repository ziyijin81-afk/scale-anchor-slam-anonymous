#!/usr/bin/env python3
"""Validate a two-frame VGGT scale anchor against TUM metric depth."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vggt_slam.project_paths import prefer_bundled_third_party


prefer_bundled_third_party()

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from evals.visualize_tum_loop_pair import select_keyframes_and_loop
from vggt_slam.scale_solver import (
    anchor_scale,
    camera_points_from_depth,
    robust_sample_statistics,
    same_frame_scale_consistency,
)
from vggt_slam.solver import _short_batch_camera_head_fp32


def read_tum_file_list(path):
    records = []
    for line in path.read_text().splitlines():
        if line and not line.startswith("#"):
            timestamp, relative_path = line.split()[:2]
            records.append((float(timestamp), relative_path))
    return records


def nearest_depth(dataset_root, rgb_path):
    records = read_tum_file_list(dataset_root / "depth.txt")
    timestamps = np.asarray([record[0] for record in records])
    rgb_timestamp = float(rgb_path.stem)
    index = int(np.argmin(np.abs(timestamps - rgb_timestamp)))
    return dataset_root / records[index][1], float(timestamps[index] - rgb_timestamp)


def preprocess_tum_depth(depth_path, output_shape):
    raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Cannot decode TUM depth {depth_path}")
    height, width = output_shape
    resized = cv2.resize(raw, (width, height), interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.float64) / 5000.0


def infer(model, image_paths, device, label):
    images = load_and_preprocess_images([str(path) for path in image_paths]).to(device)
    with torch.no_grad():
        with _short_batch_camera_head_fp32(model, len(image_paths), label):
            prediction = model(images)
    _, intrinsic = pose_encoding_to_extri_intri(prediction["pose_enc"], images.shape[-2:])
    result = {
        "depth": prediction["depth"][0].detach().float().cpu().numpy(),
        "confidence": prediction["depth_conf"][0].detach().float().cpu().numpy(),
        "intrinsic": intrinsic[0].detach().float().cpu().numpy(),
        "images": images.detach().float().cpu().permute(0, 2, 3, 1).numpy(),
    }
    del prediction, images, intrinsic
    torch.cuda.empty_cache()
    return result


def frame_data(prediction, frame_index):
    depth = np.asarray(prediction["depth"][frame_index], dtype=np.float64).squeeze()
    confidence = np.asarray(
        prediction["confidence"][frame_index], dtype=np.float64
    ).squeeze()
    intrinsic = np.asarray(prediction["intrinsic"][frame_index], dtype=np.float64)
    points = camera_points_from_depth(depth, intrinsic)
    return depth, confidence, points


def relative_mad(values):
    stats = robust_sample_statistics(values)
    median = float(stats["median"])
    return {
        **stats,
        "relative_mad": float(stats["mad"] / median) if median > 0 else float("inf"),
    }


def evaluate_frame(original, original_index, joint, joint_index, gt_depth, percentile):
    original_depth, original_confidence, original_points = frame_data(
        original, original_index
    )
    joint_depth, joint_confidence, joint_points = frame_data(joint, joint_index)
    original_threshold = float(np.percentile(original["confidence"], percentile) + 1e-6)
    joint_threshold = float(np.percentile(joint["confidence"], percentile) + 1e-6)

    consistency = same_frame_scale_consistency(
        original_points,
        original_confidence,
        original_threshold,
        joint_points,
        joint_confidence,
        joint_threshold,
    )

    original_radius = np.linalg.norm(original_points, axis=-1)
    joint_radius = np.linalg.norm(joint_points, axis=-1)
    valid = original_confidence > original_threshold
    valid &= joint_confidence > joint_threshold
    valid &= np.isfinite(gt_depth) & (gt_depth > 0)
    valid &= np.isfinite(original_depth) & (original_depth > 0)
    valid &= np.isfinite(joint_depth) & (joint_depth > 0)
    valid &= np.all(np.isfinite(original_points), axis=-1)
    valid &= np.all(np.isfinite(joint_points), axis=-1)
    valid &= (original_radius > 1e-12) & (joint_radius > 1e-12)

    original_to_joint = relative_mad(original_radius[valid] / joint_radius[valid])
    metric_from_original = relative_mad(gt_depth[valid] / original_depth[valid])
    metric_from_joint = relative_mad(gt_depth[valid] / joint_depth[valid])
    A = relative_mad(original_radius[valid])
    a = relative_mad(joint_radius[valid])

    ratio_map = np.full(gt_depth.shape, np.nan, dtype=np.float64)
    ratio_map[valid] = np.log(
        (original_radius[valid] / joint_radius[valid]) / original_to_joint["median"]
    )
    original_metric_map = np.full(gt_depth.shape, np.nan, dtype=np.float64)
    original_metric_map[valid] = np.log(
        (gt_depth[valid] / original_depth[valid]) / metric_from_original["median"]
    )
    joint_metric_map = np.full(gt_depth.shape, np.nan, dtype=np.float64)
    joint_metric_map[valid] = np.log(
        (gt_depth[valid] / joint_depth[valid]) / metric_from_joint["median"]
    )
    return {
        "valid": valid,
        "gt_depth": gt_depth,
        "original_depth": original_depth,
        "joint_depth": joint_depth,
        "ratio_map": ratio_map,
        "original_metric_map": original_metric_map,
        "joint_metric_map": joint_metric_map,
        "original_to_joint_values": original_radius[valid] / joint_radius[valid],
        "project_consistency": consistency,
        "gt_eval": {
            "valid_pixels": int(np.count_nonzero(valid)),
            "original_confidence_threshold": original_threshold,
            "joint_confidence_threshold": joint_threshold,
            "A_original_radius": A,
            "a_joint_radius": a,
            "original_over_joint_radius": original_to_joint,
            "metric_from_original_depth": metric_from_original,
            "metric_from_joint_depth": metric_from_joint,
        },
    }


def json_safe_consistency(value):
    output = dict(value)
    output["original_scale_stats"] = dict(output["original_scale_stats"])
    output["joint_scale_stats"] = dict(output["joint_scale_stats"])
    return output


def plot_diagnostics(path, images, evaluations, labels):
    figure, axes = plt.subplots(2, 6, figsize=(26, 9))
    limit = np.log(1.25)
    for row, (image, result, label) in enumerate(zip(images, evaluations, labels)):
        valid = result["valid"]
        gt_eval = result["gt_eval"]
        axes[row, 0].imshow(np.clip(image, 0, 1))
        axes[row, 0].set_title(label)
        axes[row, 0].axis("off")

        depth_values = result["gt_depth"][valid]
        lower, upper = np.percentile(depth_values, (2, 98))
        axes[row, 1].imshow(result["gt_depth"], cmap="turbo", vmin=lower, vmax=upper)
        axes[row, 1].set_title("TUM metric depth")
        axes[row, 1].axis("off")

        for column, key, title in (
            (2, "original_metric_map", "GT/original scale variation"),
            (3, "joint_metric_map", "GT/joint scale variation"),
            (4, "ratio_map", "original/joint scale variation"),
        ):
            shown = axes[row, column].imshow(
                result[key], cmap="coolwarm", vmin=-limit, vmax=limit
            )
            axes[row, column].set_title(title)
            axes[row, column].axis("off")
            figure.colorbar(shown, ax=axes[row, column], fraction=0.046, pad=0.04)

        scale = gt_eval["original_over_joint_radius"]["median"]
        normalized = result["original_to_joint_values"] / scale
        axes[row, 5].hist(normalized, bins=120, range=(0.5, 1.5), density=True)
        axes[row, 5].axvline(1.0, color="black", linewidth=1)
        axes[row, 5].axvspan(0.95, 1.05, color="green", alpha=0.15)
        axes[row, 5].set_xlim(0.5, 1.5)
        axes[row, 5].set_title(
            f"N={gt_eval['valid_pixels']}, relMAD="
            f"{gt_eval['original_over_joint_radius']['relative_mad']:.2%}"
        )
        axes[row, 5].set_xlabel("(original/joint) / median")
    figure.suptitle("TUM depth validation of the two-frame VGGT scale bridge")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_scale_summary(path, summary):
    names = [
        "GT relative\nscale",
        "Anchor formula\n(project mask)",
        "Anchor formula\n(GT eval mask)",
    ]
    values = [
        summary["groundtruth_relative_scale_k_to_0"],
        summary["anchor_formula_project_mask"],
        summary["anchor_formula_gt_eval_mask"],
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(names, values, color=("black", "tab:orange", "tab:blue"))
    axes[0].axhline(values[0], color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("scale multiplier")
    axes[0].set_title("Anchor measurement versus TUM depth truth")
    for index, value in enumerate(values):
        axes[0].text(index, value, f"{value:.4f}", ha="center", va="bottom")

    joint_values = [
        summary["metric_scale_joint_F0"], summary["metric_scale_joint_Fk"]
    ]
    axes[1].bar(("joint F0", "joint Fk"), joint_values, color=("tab:red", "tab:blue"))
    axes[1].set_ylabel("metric multiplier: TUM depth / VGGT depth")
    axes[1].set_title("One joint call: shared-scale check")
    for index, value in enumerate(joint_values):
        axes[1].text(index, value, f"{value:.4f}", ha="center", va="bottom")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-disparity", type=float, default=50.0)
    parser.add_argument("--submap-size", type=int, default=16)
    parser.add_argument("--overlap", type=int, default=1)
    parser.add_argument("--confidence-percentile", type=float, default=25.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selection = select_keyframes_and_loop(
        args.dataset_root, args.min_disparity, args.submap_size, args.overlap
    )
    loop_index = selection["loop_selected_index"]
    loop_submap = selection["loop_submap_order"]
    loop_frame = selection["loop_frame_index"]
    selected_paths = [selection["selected"][0], selection["selected"][loop_index]]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url(
        "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    ))
    model.eval().to(torch.bfloat16).to(device)

    originals = [
        infer(model, selection["batches"][0], device, "[tum-depth-original-0]"),
        infer(
            model, selection["batches"][loop_submap], device,
            "[tum-depth-original-k]",
        ),
    ]
    joint = infer(model, selected_paths, device, "[tum-depth-joint]")
    original_indices = [0, loop_frame]

    depth_matches = [nearest_depth(args.dataset_root, path) for path in selected_paths]
    output_shape = np.asarray(joint["depth"][0]).squeeze().shape
    gt_depths = [
        preprocess_tum_depth(depth_path, output_shape)
        for depth_path, _ in depth_matches
    ]
    evaluations = [
        evaluate_frame(
            original, original_index, joint, joint_index, gt_depth,
            args.confidence_percentile,
        )
        for joint_index, (original, original_index, gt_depth) in enumerate(
            zip(originals, original_indices, gt_depths)
        )
    ]

    project_A0 = evaluations[0]["project_consistency"]["original_scale_stats"]["median"]
    project_Ak = evaluations[1]["project_consistency"]["original_scale_stats"]["median"]
    project_a0 = evaluations[0]["project_consistency"]["joint_scale_stats"]["median"]
    project_ak = evaluations[1]["project_consistency"]["joint_scale_stats"]["median"]
    eval_A0 = evaluations[0]["gt_eval"]["A_original_radius"]["median"]
    eval_Ak = evaluations[1]["gt_eval"]["A_original_radius"]["median"]
    eval_a0 = evaluations[0]["gt_eval"]["a_joint_radius"]["median"]
    eval_ak = evaluations[1]["gt_eval"]["a_joint_radius"]["median"]
    metric_original = [
        result["gt_eval"]["metric_from_original_depth"]["median"]
        for result in evaluations
    ]
    metric_joint = [
        result["gt_eval"]["metric_from_joint_depth"]["median"]
        for result in evaluations
    ]

    truth = float(metric_original[1] / metric_original[0])
    project_anchor = anchor_scale(project_A0, project_Ak, project_a0, project_ak)
    eval_anchor = anchor_scale(eval_A0, eval_Ak, eval_a0, eval_ak)
    pointwise_anchor = float(
        evaluations[0]["project_consistency"]["scale_ratio"]
        / evaluations[1]["project_consistency"]["scale_ratio"]
    )
    summary = {
        "groundtruth_relative_scale_k_to_0": truth,
        "anchor_formula_project_mask": project_anchor,
        "anchor_formula_gt_eval_mask": eval_anchor,
        "pointwise_anchor_project_mask": pointwise_anchor,
        "formula_pointwise_log_disagreement": float(
            abs(np.log(project_anchor / pointwise_anchor))
        ),
        "anchor_project_relative_error": float(abs(project_anchor / truth - 1.0)),
        "anchor_gt_eval_relative_error": float(abs(eval_anchor / truth - 1.0)),
        "metric_scale_original_F0": float(metric_original[0]),
        "metric_scale_original_Fk": float(metric_original[1]),
        "metric_scale_joint_F0": float(metric_joint[0]),
        "metric_scale_joint_Fk": float(metric_joint[1]),
        "joint_scale_log_disagreement": float(abs(np.log(metric_joint[1] / metric_joint[0]))),
        "joint_scale_relative_difference": float(abs(metric_joint[1] / metric_joint[0] - 1.0)),
    }

    labels = [f"F0: {selected_paths[0].name}", f"Fk: {selected_paths[1].name}"]
    images = [joint["images"][0], joint["images"][1]]
    plot_diagnostics(
        args.output_dir / "tum_depth_scale_diagnostics.png",
        images, evaluations, labels,
    )
    plot_scale_summary(args.output_dir / "anchor_vs_tum_truth.png", summary)

    output = {
        "configuration": {
            "device": device,
            "min_disparity": args.min_disparity,
            "submap_size": args.submap_size,
            "overlap": args.overlap,
            "confidence_percentile": args.confidence_percentile,
            "original_vggt_calls": 2,
            "frames_per_original_call": args.submap_size + args.overlap,
            "joint_vggt_calls": 1,
            "frames_in_joint_call": 2,
        },
        "frames": [
            {
                "label": label,
                "rgb": str(rgb_path),
                "depth": str(depth_match[0]),
                "rgb_depth_timestamp_difference_s": depth_match[1],
                "submap": submap,
                "frame_index": frame_index,
                "project_consistency": json_safe_consistency(result["project_consistency"]),
                "gt_eval": result["gt_eval"],
            }
            for label, rgb_path, depth_match, submap, frame_index, result in zip(
                labels,
                selected_paths,
                depth_matches,
                [0, loop_submap],
                original_indices,
                evaluations,
            )
        ],
        "anchor": {
            "project_mask_values": {
                "A0": project_A0, "Ak": project_Ak,
                "a0": project_a0, "ak": project_ak,
            },
            "gt_eval_mask_values": {
                "A0": eval_A0, "Ak": eval_Ak,
                "a0": eval_a0, "ak": eval_ak,
            },
            **summary,
        },
        "interpretation_thresholds": {
            "same_frame_max_relative_mad": 0.10,
            "same_frame_min_points": 500,
            "formula_log_disagreement_gate": 0.10,
        },
    }
    (args.output_dir / "tum_anchor_depth_validation.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
