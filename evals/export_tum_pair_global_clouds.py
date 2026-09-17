#!/usr/bin/env python3
"""Place two original VGGT frame clouds in the TUM metric world frame."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.visualize_tum_loop_pair import (
    load_groundtruth,
    pose_matrix,
    tint,
    transform_points,
    write_cloud,
)


def read_tum_list(path):
    rows = []
    for line in path.read_text().splitlines():
        if line and not line.startswith("#"):
            timestamp, relative_path = line.split()[:2]
            rows.append((float(timestamp), relative_path))
    return rows


def nearest_row(rows, timestamp):
    return min(rows, key=lambda row: abs(row[0] - timestamp))


def load_cloud(path):
    cloud = o3d.io.read_point_cloud(str(path))
    return np.asarray(cloud.points), np.asarray(cloud.colors)


def metric_scale_from_tum_depth(points, depth_path, height=392, width=518):
    if len(points) != height * width:
        raise ValueError(
            f"Expected dense {width}x{height} cloud, got {len(points)} points"
        )
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise ValueError(f"Cannot decode {depth_path}")
    metric_depth = cv2.resize(
        depth_raw, (width, height), interpolation=cv2.INTER_NEAREST
    ).astype(np.float64) / 5000.0
    predicted_z = points[:, 2].reshape(height, width)
    valid = (
        np.isfinite(metric_depth) & (metric_depth > 0)
        & np.isfinite(predicted_z) & (predicted_z > 1e-8)
    )
    ratios = metric_depth[valid] / predicted_z[valid]
    return {
        "scale": float(np.median(ratios)),
        "valid_pixels": int(len(ratios)),
        "mad": float(np.median(np.abs(ratios - np.median(ratios)))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--pair-dir", required=True, type=Path)
    args = parser.parse_args()

    pair_stats = json.loads((args.pair_dir / "comparison_stats.json").read_text())
    image_paths = [Path(pair_stats[key]["image"]) for key in ("F0", "Fk")]
    timestamps = [float(path.stem) for path in image_paths]
    points0, colors0 = load_cloud(args.pair_dir / "01_original_F0.pcd")
    pointsk, colorsk = load_cloud(args.pair_dir / "02_original_Fk.pcd")

    depth_rows = read_tum_list(args.dataset_root / "depth.txt")
    depth_records = [nearest_row(depth_rows, timestamp) for timestamp in timestamps]
    depth_paths = [args.dataset_root / record[1] for record in depth_records]
    depth_scale0 = metric_scale_from_tum_depth(points0, depth_paths[0])
    depth_scalek = metric_scale_from_tum_depth(pointsk, depth_paths[1])

    groundtruth = load_groundtruth(args.dataset_root / "groundtruth.txt")
    gt_rows = [groundtruth[np.argmin(np.abs(groundtruth[:, 0] - timestamp))] for timestamp in timestamps]
    world_from_camera = [pose_matrix(row) for row in gt_rows]

    # Use one common conversion obtained from F0 for both clouds.  This places
    # VGGT units and GT translation in the same metric unit without erasing the
    # relative scale drift of Fk by calibrating it independently.
    common_scale = depth_scale0["scale"]
    world_points = [
        transform_points(points0 * common_scale, world_from_camera[0]),
        transform_points(pointsk * common_scale, world_from_camera[1]),
    ]
    shown_colors = [tint(colors0, (1.0, 0.15, 0.05)), tint(colorsk, (0.05, 0.35, 1.0))]
    write_cloud(
        args.pair_dir / "03b_original_TUM_global_common_F0_scale.pcd",
        world_points,
        shown_colors,
    )

    centers = np.asarray([transform[:3, 3] for transform in world_from_camera])
    rng = np.random.default_rng(0)
    figure = plt.figure(figsize=(15, 7))
    axis = figure.add_subplot(1, 1, 1, projection="3d")
    for points, colors, label in zip(world_points, shown_colors, ("F0", "Fk")):
        count = min(60000, len(points))
        indices = rng.choice(len(points), count, replace=False)
        axis.scatter(
            points[indices, 0], points[indices, 1], points[indices, 2],
            c=colors[indices], s=0.35, linewidths=0, label=label,
        )
    axis.scatter(
        centers[:, 0], centers[:, 1], centers[:, 2],
        c=["darkred", "darkblue"], s=140, marker="*", depthshade=False,
        label="camera centers",
    )
    axis.plot(centers[:, 0], centers[:, 1], centers[:, 2], "k--", linewidth=1)
    for label, center in zip(("F0 camera", "Fk camera"), centers):
        axis.text(*center, label, fontsize=11)
    all_points = np.concatenate(world_points)
    lower, upper = np.percentile(all_points, [1, 99], axis=0)
    center = (lower + upper) / 2
    radius = max(np.max(upper - lower) / 2, 1e-6)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_xlabel("TUM world X [m]")
    axis.set_ylabel("TUM world Y [m]")
    axis.set_zlabel("TUM world Z [m]")
    axis.set_title("Original VGGT frame clouds at their TUM global poses\n(common F0 metric scale; no cloud-to-cloud alignment)")
    axis.legend()
    axis.view_init(elev=24, azim=-60)
    figure.tight_layout()
    figure.savefig(args.pair_dir / "original_TUM_global_positions.png", dpi=200)
    plt.close(figure)

    output = {
        "images": [str(path) for path in image_paths],
        "depth_files": [str(path) for path in depth_paths],
        "F0_metric_scale": depth_scale0,
        "Fk_metric_scale_diagnostic_only": depth_scalek,
        "common_scale_applied_to_both_clouds": common_scale,
        "camera_centers_TUM_world_m": centers.tolist(),
        "camera_center_distance_m": float(np.linalg.norm(centers[1] - centers[0])),
        "note": "Only F0 scale calibrates both clouds; Fk is not independently rescaled.",
    }
    (args.pair_dir / "original_TUM_global_stats.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
