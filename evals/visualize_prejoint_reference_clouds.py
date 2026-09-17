#!/usr/bin/env python3
"""Compare two reference clouds from their original independent submap runs."""

import argparse
import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch

from vggt_slam.project_paths import prefer_bundled_third_party


prefer_bundled_third_party()

from vggt.models.vggt import VGGT
from vggt.utils.geometry import depth_to_world_coords_points
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from evals.visualize_joint_vggt_pair import equalize_3d_axes


def read_submap_batches(run_log):
    batches = []
    for line in run_log.read_text(errors="replace").splitlines():
        stripped = line.strip()
        # tqdm writes progress updates with carriage returns, so tee may place
        # the following Python image list later on the same physical line.
        embedded_start = stripped.find("['/")
        if embedded_start >= 0:
            stripped = stripped[embedded_start:]
        if not stripped.startswith("[") or not stripped.endswith("]"):
            continue
        try:
            value = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            continue
        if (
            isinstance(value, list)
            and value
            and all(isinstance(item, str) for item in value)
            and all(Path(item).suffix.lower() in {".png", ".jpg", ".jpeg"} for item in value)
        ):
            batches.append(value)
    return batches


def infer_submap(model, image_paths, device, confidence_percentile):
    images = load_and_preprocess_images(image_paths).to(device)
    with torch.no_grad():
        prediction = model(images)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        prediction["pose_enc"], images.shape[-2:]
    )
    return {
        "images": images.detach().float().cpu().permute(0, 2, 3, 1).numpy(),
        "depth": prediction["depth"][0].detach().float().cpu().numpy(),
        "confidence": prediction["depth_conf"][0].detach().float().cpu().numpy(),
        "extrinsic": extrinsic[0].detach().float().cpu().numpy(),
        "intrinsic": intrinsic[0].detach().float().cpu().numpy(),
        "confidence_percentile": confidence_percentile,
    }


def extract_reference(prediction, frame_index):
    confidence = prediction["confidence"]
    threshold = float(
        np.percentile(confidence, prediction["confidence_percentile"]) + 1e-6
    )
    world_points, camera_points, positive_depth = depth_to_world_coords_points(
        prediction["depth"][frame_index].squeeze(-1),
        prediction["extrinsic"][frame_index],
        prediction["intrinsic"][frame_index],
    )
    valid = positive_depth & (confidence[frame_index] > threshold)
    valid &= np.all(np.isfinite(world_points), axis=-1)
    valid &= np.all(np.isfinite(camera_points), axis=-1)
    return {
        "world_points": world_points[valid],
        "camera_points": camera_points[valid],
        "colors": prediction["images"][frame_index][valid],
        "confidence_threshold": threshold,
    }


def tinted_colors(colors, tint):
    luminance = np.mean(np.clip(colors, 0.0, 1.0), axis=1, keepdims=True)
    return np.clip(0.25 * colors + 0.75 * luminance * np.asarray(tint), 0.0, 1.0)


def write_combined_ply(path, point_sets, color_sets):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(
        np.concatenate(point_sets).astype(np.float64)
    )
    cloud.colors = o3d.utility.Vector3dVector(np.concatenate(color_sets))
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False):
        raise RuntimeError(f"Failed to write {path}")


def plot_overlay(path, inputs, point_sets, color_sets, title, plot_points_per_frame):
    rng = np.random.default_rng(0)
    sampled_points = []
    sampled_colors = []
    for points, colors in zip(point_sets, color_sets):
        count = min(len(points), plot_points_per_frame)
        indices = rng.choice(len(points), size=count, replace=False)
        sampled_points.append(points[indices])
        sampled_colors.append(colors[indices])
    combined_points = np.concatenate(sampled_points)
    combined_colors = np.concatenate(sampled_colors)

    figure = plt.figure(figsize=(18, 11))
    for index, (image, label) in enumerate(inputs):
        axis = figure.add_subplot(2, 3, index + 1)
        axis.imshow(image)
        axis.set_title(label)
        axis.axis("off")

    views = ((22, -65, "3D overlay"), (90, -90, "Top view"), (5, 0, "Side view"))
    for offset, (elevation, azimuth, view_title) in enumerate(views):
        axis = figure.add_subplot(2, 3, offset + 4, projection="3d")
        axis.scatter(
            combined_points[:, 0], combined_points[:, 1], combined_points[:, 2],
            c=combined_colors, s=0.3, linewidths=0,
        )
        equalize_3d_axes(axis, combined_points)
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set(title=view_title, xlabel="X", ylabel="Y", zlabel="Z")
    figure.suptitle(title, fontsize=15)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-log", required=True, type=Path)
    parser.add_argument("--submap-orders", nargs=2, type=int, default=(0, 1))
    parser.add_argument("--frame-indices", nargs=2, type=int, default=(1, 6))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--confidence-percentile", type=float, default=25.0)
    parser.add_argument("--plot-points-per-frame", type=int, default=30000)
    args = parser.parse_args()

    batches = read_submap_batches(args.run_log)
    if len(batches) <= max(args.submap_orders):
        raise ValueError(
            f"Only found {len(batches)} image batches in {args.run_log}; "
            f"requested orders {args.submap_orders}"
        )
    selected_batches = [batches[index] for index in args.submap_orders]
    for batch in selected_batches:
        if len(batch) != 17:
            raise ValueError(f"Expected a 17-frame original submap batch, got {len(batch)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGT()
    weights_url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(weights_url))
    model.eval().to(torch.bfloat16).to(device)

    predictions = [
        infer_submap(model, batch, device, args.confidence_percentile)
        for batch in selected_batches
    ]
    references = [
        extract_reference(prediction, frame_index)
        for prediction, frame_index in zip(predictions, args.frame_indices)
    ]
    selected_paths = [
        batch[frame_index]
        for batch, frame_index in zip(selected_batches, args.frame_indices)
    ]

    display_colors = [
        tinted_colors(references[0]["colors"], (1.0, 0.15, 0.05)),
        tinted_colors(references[1]["colors"], (0.05, 0.35, 1.0)),
    ]
    camera_points = [reference["camera_points"] for reference in references]
    world_points = [reference["world_points"] for reference in references]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_combined_ply(
        args.output_dir / "prejoint_camera_origin_overlay.ply",
        camera_points,
        display_colors,
    )
    write_combined_ply(
        args.output_dir / "prejoint_raw_submap_coordinates.ply",
        world_points,
        display_colors,
    )

    inputs = [
        (
            prediction["images"][frame_index],
            f"submap order {submap_order}, index {frame_index}: {Path(path).name}",
        )
        for prediction, frame_index, submap_order, path in zip(
            predictions, args.frame_indices, args.submap_orders, selected_paths
        )
    ]
    plot_overlay(
        args.output_dir / "prejoint_camera_origin_overlay.png",
        inputs,
        camera_points,
        display_colors,
        "Before joint VGGT: original 17-frame clouds overlaid at their own camera origins",
        args.plot_points_per_frame,
    )
    plot_overlay(
        args.output_dir / "prejoint_raw_submap_coordinates.png",
        inputs,
        world_points,
        display_colors,
        "Before joint VGGT: raw independent submap-local coordinates (not spatially aligned)",
        args.plot_points_per_frame,
    )

    stats = {
        "device": device,
        "vggt_calls": 2,
        "frames_per_call": 17,
        "coordinate_warning": (
            "The two original submap frames have independent gauges. The camera-origin "
            "overlay is for scale/shape comparison, not a global map alignment."
        ),
        "references": [
            {
                "submap_order": int(submap_order),
                "frame_index": int(frame_index),
                "image": path,
                "valid_points": int(len(reference["camera_points"])),
                "median_camera_radius": float(
                    np.median(np.linalg.norm(reference["camera_points"], axis=1))
                ),
                "confidence_threshold": reference["confidence_threshold"],
            }
            for submap_order, frame_index, path, reference in zip(
                args.submap_orders, args.frame_indices, selected_paths, references
            )
        ],
    }
    (args.output_dir / "prejoint_reference_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n"
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
