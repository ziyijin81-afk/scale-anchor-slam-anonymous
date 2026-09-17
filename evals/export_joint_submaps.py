#!/usr/bin/env python3
"""Run one VGGT call for two logged submaps and export their joint point cloud."""

import argparse
import ast
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

from vggt_slam.scale_solver import (
    confidence_selection_mask,
    frame_confidence_threshold,
)
from vggt_slam.solver import _short_batch_camera_head_fp32


def read_logged_batches(path):
    batches = []
    for line in path.read_text(errors="replace").splitlines():
        start = line.find("['/")
        if start < 0:
            continue
        try:
            value = ast.literal_eval(line[start:])
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
            batches.append([Path(item) for item in value])
    if not batches:
        raise ValueError(f"No image batches found in {path}")
    return batches


def unique_paths(first, second):
    result = []
    seen = set()
    for path in first + second:
        resolved = str(path.resolve())
        if resolved not in seen:
            result.append(path)
            seen.add(resolved)
    return result


def equalize_axes(axis, points):
    lower, upper = np.percentile(points, (1, 99), axis=0)
    center = (lower + upper) / 2
    radius = max(float(np.max(upper - lower)) / 2, 1e-6)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-log", required=True, type=Path)
    parser.add_argument("--submaps", nargs=2, required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--confidence-percentile", type=float, default=25.0)
    parser.add_argument("--preview-points", type=int, default=160000)
    args = parser.parse_args()

    batches = read_logged_batches(args.run_log)
    first_index, second_index = args.submaps
    if min(args.submaps) < 0 or max(args.submaps) >= len(batches):
        raise IndexError(f"submaps {args.submaps} outside 0..{len(batches) - 1}")
    first_batch, second_batch = batches[first_index], batches[second_index]
    paths = unique_paths(first_batch, second_batch)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Assign the shared boundary image to the first submap.  This affects only
    # diagnostic coloring; all images are passed through one joint VGGT call.
    first_set = {str(path.resolve()) for path in first_batch}
    groups = np.asarray([
        first_index if str(path.resolve()) in first_set else second_index
        for path in paths
    ])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = load_and_preprocess_images([str(path) for path in paths]).to(device)
    model = VGGT()
    weights_url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(weights_url))
    model.eval().to(torch.bfloat16).to(device)

    # Exactly one model call: all unique frames from both submaps share the
    # same VGGT prediction coordinate system.
    with torch.no_grad():
        with _short_batch_camera_head_fp32(model, len(paths), "[joint-submaps-export]"):
            prediction = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        prediction["pose_enc"], images.shape[-2:]
    )
    depth = prediction["depth"][0].detach().float().cpu().numpy()
    confidence = prediction["depth_conf"][0].detach().float().cpu().numpy()
    extrinsic = extrinsic[0].detach().float().cpu().numpy()
    intrinsic = intrinsic[0].detach().float().cpu().numpy()
    colors = images.detach().float().cpu().permute(0, 2, 3, 1).numpy()

    all_points, all_colors, all_groups = [], [], []
    camera_centers = []
    frame_stats = []
    for index, path in enumerate(paths):
        world, _, positive = depth_to_world_coords_points(
            depth[index].squeeze(-1), extrinsic[index], intrinsic[index]
        )
        threshold = frame_confidence_threshold(
            confidence[index], args.confidence_percentile
        )
        confidence_mask, degenerate = confidence_selection_mask(
            confidence[index], threshold
        )
        valid = positive & confidence_mask & np.all(np.isfinite(world), axis=-1)
        points = world[valid]
        rgb = np.clip(colors[index][valid], 0.0, 1.0)
        all_points.append(points)
        all_colors.append(rgb)
        all_groups.append(np.full(len(points), groups[index], dtype=np.int32))

        camera_from_world = np.eye(4)
        camera_from_world[:3, :4] = extrinsic[index]
        center = np.linalg.inv(camera_from_world)[:3, 3]
        camera_centers.append(center)
        frame_stats.append({
            "joint_frame_index": index,
            "submap": int(groups[index]),
            "image": str(path),
            "confidence_threshold": threshold,
            "confidence_degenerate": bool(degenerate),
            "selected_points": int(len(points)),
            "camera_center": center.tolist(),
        })

    points = np.concatenate(all_points)
    rgb = np.concatenate(all_colors)
    point_groups = np.concatenate(all_groups)
    camera_centers = np.asarray(camera_centers)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    cloud_path = args.output_dir / f"submap{first_index}_{second_index}_joint_vggt.pcd"
    if not o3d.io.write_point_cloud(str(cloud_path), cloud, write_ascii=False):
        raise RuntimeError(f"Failed to write {cloud_path}")

    rng = np.random.default_rng(0)
    count = min(len(points), args.preview_points)
    selected = rng.choice(len(points), count, replace=False)
    preview_points = points[selected]
    preview_groups = point_groups[selected]
    preview_colors = np.where(
        (preview_groups == first_index)[:, None],
        np.asarray([0.95, 0.2, 0.1]),
        np.asarray([0.1, 0.4, 1.0]),
    )
    figure = plt.figure(figsize=(17, 6))
    views = ((25, -65, "3D view"), (90, -90, "top view"), (5, 0, "side view"))
    for plot_index, (elev, azim, title) in enumerate(views, 1):
        axis = figure.add_subplot(1, 3, plot_index, projection="3d")
        axis.scatter(
            preview_points[:, 0], preview_points[:, 1], preview_points[:, 2],
            c=preview_colors, s=0.18, linewidths=0,
        )
        axis.scatter(
            camera_centers[:, 0], camera_centers[:, 1], camera_centers[:, 2],
            c=np.where((groups == first_index)[:, None],
                       np.asarray([0.95, 0.2, 0.1]),
                       np.asarray([0.1, 0.4, 1.0])),
            s=14, marker="^", depthshade=False,
        )
        equalize_axes(axis, preview_points)
        axis.view_init(elev=elev, azim=azim)
        axis.set(title=title, xlabel="X", ylabel="Y", zlabel="Z")
    figure.suptitle(
        f"One joint VGGT call: submap {first_index} (red) + {second_index} (blue)"
    )
    figure.tight_layout()
    preview_path = args.output_dir / f"submap{first_index}_{second_index}_joint_preview.png"
    figure.savefig(preview_path, dpi=180)
    plt.close(figure)

    stats = {
        "run_log": str(args.run_log),
        "submaps": [first_index, second_index],
        "source_batch_lengths": [len(first_batch), len(second_batch)],
        "shared_boundary_images": len(first_batch) + len(second_batch) - len(paths),
        "single_vggt_call_frame_count": len(paths),
        "confidence_percentile": args.confidence_percentile,
        "total_exported_points": int(len(points)),
        "point_cloud": str(cloud_path),
        "preview": str(preview_path),
        "frames": frame_stats,
    }
    stats_path = args.output_dir / "joint_submaps_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({key: value for key, value in stats.items() if key != "frames"}, indent=2))


if __name__ == "__main__":
    main()
