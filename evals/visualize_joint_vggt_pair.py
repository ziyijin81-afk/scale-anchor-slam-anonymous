#!/usr/bin/env python3
"""Run one two-image VGGT inference and export its shared-frame point cloud."""

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
from vggt.utils.geometry import depth_to_world_coords_points
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from vggt_slam.solver import _short_batch_camera_head_fp32


def equalize_3d_axes(axis, points):
    lower = np.nanpercentile(points, 1, axis=0)
    upper = np.nanpercentile(points, 99, axis=0)
    center = (lower + upper) / 2.0
    radius = max(float(np.max(upper - lower)) / 2.0, 1e-6)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def tinted_colors(colors, tint):
    luminance = np.mean(np.clip(colors, 0.0, 1.0), axis=1, keepdims=True)
    return np.clip(0.25 * colors + 0.75 * luminance * np.asarray(tint), 0.0, 1.0)


def robust_unit_interval(values, valid=None):
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if valid is not None:
        finite &= valid
    if not np.any(finite):
        return np.zeros_like(values)
    lower, upper = np.percentile(values[finite], (1.0, 99.0))
    if upper <= lower:
        return np.zeros_like(values)
    return np.clip((values - lower) / (upper - lower), 0.0, 1.0)


def save_prediction_visualizations(output_dir, colors, depth, confidence, threshold):
    rows = len(colors)
    figure, axes = plt.subplots(rows, 5, figsize=(20, 4 * rows), squeeze=False)
    for frame_index in range(rows):
        rgb = np.clip(colors[frame_index], 0.0, 1.0)
        frame_depth = np.squeeze(depth[frame_index])
        frame_confidence = np.squeeze(confidence[frame_index])
        valid_depth = np.isfinite(frame_depth) & (frame_depth > 0)
        confidence_unit = robust_unit_interval(frame_confidence, valid_depth)
        depth_unit = robust_unit_interval(frame_depth, valid_depth)
        confidence_rgb = plt.get_cmap("turbo")(confidence_unit)[..., :3]
        depth_rgb = plt.get_cmap("magma")(depth_unit)[..., :3]
        mask = valid_depth & (frame_confidence > threshold)
        overlay = np.clip(0.55 * rgb + 0.45 * confidence_rgb, 0.0, 1.0)

        np.save(output_dir / f"frame_{frame_index}_depth_conf.npy", frame_confidence)
        np.save(output_dir / f"frame_{frame_index}_depth.npy", frame_depth)
        plt.imsave(
            output_dir / f"frame_{frame_index}_depth_conf_heatmap.png",
            confidence_rgb,
        )
        plt.imsave(
            output_dir / f"frame_{frame_index}_confidence_mask.png",
            mask,
            cmap="gray",
        )
        plt.imsave(
            output_dir / f"frame_{frame_index}_confidence_overlay.png",
            overlay,
        )
        plt.imsave(
            output_dir / f"frame_{frame_index}_depth_heatmap.png",
            depth_rgb,
        )

        panels = (
            (rgb, "RGB"),
            (confidence_rgb, "depth_conf (high = warm)"),
            (mask, f"kept mask: conf > {threshold:.4f}"),
            (overlay, "RGB + depth_conf"),
            (depth_rgb, "predicted depth"),
        )
        for column, (panel, title) in enumerate(panels):
            axes[frame_index, column].imshow(panel, cmap="gray" if column == 2 else None)
            axes[frame_index, column].set_title(f"F{frame_index} — {title}")
            axes[frame_index, column].axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "depth_conf_visualization.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs=2, required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--confidence-percentile", type=float, default=25.0)
    parser.add_argument("--plot-points-per-frame", type=int, default=30000)
    args = parser.parse_args()

    for image_path in args.images:
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = load_and_preprocess_images([str(path) for path in args.images])
    images = images.to(device)
    model = VGGT()
    weights_url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(weights_url))
    model.eval().to(torch.bfloat16).to(device)

    # The two frames are one batch and VGGT is called exactly once.
    with torch.no_grad():
        with _short_batch_camera_head_fp32(model, 2, "[joint-pair-export]"):
            prediction = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        prediction["pose_enc"], images.shape[-2:]
    )
    depth = prediction["depth"][0].detach().float().cpu().numpy()
    confidence = prediction["depth_conf"][0].detach().float().cpu().numpy()
    extrinsic = extrinsic[0].detach().float().cpu().numpy()
    intrinsic = intrinsic[0].detach().float().cpu().numpy()
    colors = images.detach().float().cpu().permute(0, 2, 3, 1).numpy()

    threshold = float(np.percentile(confidence, args.confidence_percentile) + 1e-6)
    save_prediction_visualizations(
        args.output_dir, colors, depth, confidence, threshold
    )
    frame_points = []
    frame_colors = []
    all_valid_frame_points = []
    all_valid_frame_colors = []
    camera_centers = []
    statistics = []
    for frame_index in range(2):
        world_points, _, positive_depth = depth_to_world_coords_points(
            depth[frame_index].squeeze(-1),
            extrinsic[frame_index],
            intrinsic[frame_index],
        )
        valid = positive_depth & (confidence[frame_index] > threshold)
        valid &= np.all(np.isfinite(world_points), axis=-1)
        all_valid = positive_depth & np.all(np.isfinite(world_points), axis=-1)
        selected_points = world_points[valid]
        selected_colors = colors[frame_index][valid]
        frame_points.append(selected_points)
        frame_colors.append(selected_colors)
        all_valid_frame_points.append(world_points[all_valid])
        all_valid_frame_colors.append(colors[frame_index][all_valid])

        world_from_camera = np.eye(4, dtype=float)
        world_from_camera[:3, :4] = extrinsic[frame_index]
        world_from_camera = np.linalg.inv(world_from_camera)
        camera_centers.append(world_from_camera[:3, 3])
        radii = np.linalg.norm(selected_points - world_from_camera[:3, 3], axis=1)
        statistics.append({
            "frame_index": frame_index,
            "image": str(args.images[frame_index]),
            "valid_points": int(len(selected_points)),
            "all_positive_finite_points": int(np.count_nonzero(all_valid)),
            "median_camera_radius": float(np.median(radii)),
            "camera_center_joint_world": world_from_camera[:3, 3].tolist(),
        })

    all_points = np.concatenate(frame_points)
    all_colors = np.concatenate(frame_colors)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(all_points.astype(np.float64))
    point_cloud.colors = o3d.utility.Vector3dVector(np.clip(all_colors, 0.0, 1.0))
    if not o3d.io.write_point_cloud(
        str(args.output_dir / "joint_vggt_pair.ply"), point_cloud, write_ascii=False
    ):
        raise RuntimeError("Open3D failed to write the joint point cloud")

    display_frame_colors = [
        tinted_colors(frame_colors[0], (1.0, 0.15, 0.05)),
        tinted_colors(frame_colors[1], (0.05, 0.35, 1.0)),
    ]
    tinted_cloud = o3d.geometry.PointCloud()
    tinted_cloud.points = o3d.utility.Vector3dVector(all_points.astype(np.float64))
    tinted_cloud.colors = o3d.utility.Vector3dVector(
        np.concatenate(display_frame_colors)
    )
    if not o3d.io.write_point_cloud(
        str(args.output_dir / "joint_vggt_pair_tinted.ply"),
        tinted_cloud,
        write_ascii=False,
    ):
        raise RuntimeError("Open3D failed to write the tinted joint point cloud")

    # Dense diagnostic exports deliberately ignore depth_conf. They are useful
    # for viewing the full prediction, but scale estimation must continue to
    # use the filtered points above.
    all_valid_points = np.concatenate(all_valid_frame_points)
    all_valid_colors = np.concatenate(all_valid_frame_colors)
    all_valid_cloud = o3d.geometry.PointCloud()
    all_valid_cloud.points = o3d.utility.Vector3dVector(
        all_valid_points.astype(np.float64)
    )
    all_valid_cloud.colors = o3d.utility.Vector3dVector(
        np.clip(all_valid_colors, 0.0, 1.0)
    )
    if not o3d.io.write_point_cloud(
        str(args.output_dir / "joint_vggt_pair_all_valid_rgb.pcd"),
        all_valid_cloud,
        write_ascii=False,
    ):
        raise RuntimeError("Open3D failed to write the dense RGB joint point cloud")

    all_valid_tinted_cloud = o3d.geometry.PointCloud()
    all_valid_tinted_cloud.points = o3d.utility.Vector3dVector(
        all_valid_points.astype(np.float64)
    )
    all_valid_tinted_cloud.colors = o3d.utility.Vector3dVector(np.concatenate((
        tinted_colors(all_valid_frame_colors[0], (1.0, 0.15, 0.05)),
        tinted_colors(all_valid_frame_colors[1], (0.05, 0.35, 1.0)),
    )))
    if not o3d.io.write_point_cloud(
        str(args.output_dir / "joint_vggt_pair_all_valid_tinted.pcd"),
        all_valid_tinted_cloud,
        write_ascii=False,
    ):
        raise RuntimeError("Open3D failed to write the dense tinted joint point cloud")

    rng = np.random.default_rng(0)
    plot_points = []
    plot_colors = []
    for points, rgb in zip(frame_points, display_frame_colors):
        count = min(len(points), args.plot_points_per_frame)
        indices = rng.choice(len(points), size=count, replace=False)
        plot_points.append(points[indices])
        plot_colors.append(rgb[indices])
    sampled_points = np.concatenate(plot_points)
    sampled_colors = np.concatenate(plot_colors)
    camera_centers = np.asarray(camera_centers)

    figure = plt.figure(figsize=(18, 11))
    for index in range(2):
        axis = figure.add_subplot(2, 3, index + 1)
        axis.imshow(colors[index])
        axis.set_title(f"Input F{index}: {args.images[index].name}")
        axis.axis("off")

    views = ((25, -65, "Joint point cloud"), (90, -90, "Top view"), (10, 0, "Side view"))
    for offset, (elevation, azimuth, title) in enumerate(views):
        axis = figure.add_subplot(2, 3, offset + 4, projection="3d")
        axis.scatter(
            sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2],
            c=np.clip(sampled_colors, 0.0, 1.0), s=0.25, linewidths=0,
        )
        axis.scatter(
            camera_centers[:, 0], camera_centers[:, 1], camera_centers[:, 2],
            c=["red", "blue"], marker="^", s=80, depthshade=False,
        )
        for frame_index, center in enumerate(camera_centers):
            axis.text(*center, f" F{frame_index}", color=("red", "blue")[frame_index])
        equalize_3d_axes(axis, sampled_points)
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set(title=title, xlabel="X", ylabel="Y", zlabel="Z")

    figure.suptitle(
        "One joint two-frame VGGT inference — both clouds in the predicted shared frame",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(args.output_dir / "joint_vggt_pair_views.png", dpi=180)
    plt.close(figure)

    output_statistics = {
        "device": device,
        "single_vggt_call_frame_count": 2,
        "confidence_percentile": args.confidence_percentile,
        "confidence_threshold": threshold,
        "total_exported_points": int(len(all_points)),
        "total_all_positive_finite_points": int(len(all_valid_points)),
        "frames": statistics,
    }
    (args.output_dir / "joint_vggt_pair_stats.json").write_text(
        json.dumps(output_statistics, indent=2) + "\n"
    )
    print(json.dumps(output_statistics, indent=2))


if __name__ == "__main__":
    main()
