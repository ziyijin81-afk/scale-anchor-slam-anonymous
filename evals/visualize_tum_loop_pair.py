#!/usr/bin/env python3
"""Export original-submap and joint-VGGT clouds for a TUM loop pair."""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch

from vggt_slam.frame_overlap import FrameTracker
from vggt_slam.project_paths import prefer_bundled_third_party


prefer_bundled_third_party()

from vggt.models.vggt import VGGT
from vggt.utils.geometry import depth_to_world_coords_points
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from vggt_slam.solver import _short_batch_camera_head_fp32


def load_groundtruth(path):
    rows = []
    for line in path.read_text().splitlines():
        if line and not line.startswith("#"):
            rows.append([float(value) for value in line.split()[:8]])
    return np.asarray(rows, dtype=np.float64)


def pose_matrix(row):
    qx, qy, qz, qw = row[4:8]
    rotation = o3d.geometry.get_rotation_matrix_from_quaternion([qw, qx, qy, qz])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = row[1:4]
    return transform


def quaternion_angle_degrees(q0, q1):
    cosine = abs(float(np.dot(q0, q1))) / (
        float(np.linalg.norm(q0) * np.linalg.norm(q1)) + 1e-12
    )
    return float(np.degrees(2.0 * np.arccos(np.clip(cosine, -1.0, 1.0))))


def select_keyframes_and_loop(dataset_root, min_disparity, submap_size, overlap):
    image_paths = sorted(
        (dataset_root / "rgb").glob("*.png"), key=lambda path: float(path.stem)
    )
    tracker = FrameTracker()
    selected = []
    raw_indices = []
    for raw_index, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot decode {image_path}")
        if tracker.compute_disparity(image, min_disparity, False):
            selected.append(image_path)
            raw_indices.append(raw_index)

    target_size = submap_size + overlap
    batches = []
    start = 0
    while start < len(selected):
        batch = selected[start : start + target_size]
        carried_only = len(batch) <= overlap and bool(batches)
        if carried_only:
            break
        if len(batch) < target_size:
            batch += [batch[-1]] * (target_size - len(batch))
        batches.append(batch)
        start += submap_size

    groundtruth = load_groundtruth(dataset_root / "groundtruth.txt")
    selected_gt_indices = np.asarray([
        int(np.argmin(np.abs(groundtruth[:, 0] - float(path.stem))))
        for path in selected
    ])
    selected_poses = groundtruth[selected_gt_indices]
    translations = np.linalg.norm(
        selected_poses[:, 1:4] - selected_poses[0, 1:4], axis=1
    )
    angles = np.asarray([
        quaternion_angle_degrees(selected_poses[0, 4:8], row[4:8])
        for row in selected_poses
    ])
    late_start = int(0.7 * len(selected))
    # Prefer a late frame that is both spatially close and similarly oriented.
    scores = translations + 0.01 * angles
    loop_selected_index = late_start + int(np.argmin(scores[late_start:]))
    loop_submap_order = loop_selected_index // submap_size
    loop_frame_index = loop_selected_index % submap_size
    return {
        "all_images": image_paths,
        "selected": selected,
        "raw_indices": raw_indices,
        "batches": batches,
        "groundtruth": groundtruth,
        "selected_gt_indices": selected_gt_indices,
        "loop_selected_index": loop_selected_index,
        "loop_submap_order": loop_submap_order,
        "loop_frame_index": loop_frame_index,
        "loop_translation_m": float(translations[loop_selected_index]),
        "loop_rotation_deg": float(angles[loop_selected_index]),
    }


def infer_original_reference(model, batch, frame_index, device, percentile, label):
    images = load_and_preprocess_images([str(path) for path in batch]).to(device)
    with torch.no_grad():
        with _short_batch_camera_head_fp32(model, len(batch), label):
            prediction = model(images)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        prediction["pose_enc"], images.shape[-2:]
    )
    all_confidence = prediction["depth_conf"][0].detach().float().cpu().numpy()
    result = extract_frame(
        prediction["depth"][0, frame_index].detach().float().cpu().numpy(),
        all_confidence[frame_index],
        extrinsic[0, frame_index].detach().float().cpu().numpy(),
        intrinsic[0, frame_index].detach().float().cpu().numpy(),
        images[frame_index].detach().float().cpu().permute(1, 2, 0).numpy(),
        float(np.percentile(all_confidence, percentile) + 1e-6),
    )
    del prediction, images, extrinsic, intrinsic
    torch.cuda.empty_cache()
    return result


def extract_frame(depth, confidence, extrinsic, intrinsic, colors, threshold):
    world_points, camera_points, positive_depth = depth_to_world_coords_points(
        depth.squeeze(-1), extrinsic, intrinsic
    )
    finite = (
        np.all(np.isfinite(world_points), axis=-1)
        & np.all(np.isfinite(camera_points), axis=-1)
    )
    full_mask = positive_depth & finite
    filtered_mask = full_mask & (confidence > threshold)
    return {
        "camera_full": camera_points[full_mask],
        "camera_filtered": camera_points[filtered_mask],
        "world_full": world_points[full_mask],
        "world_filtered": world_points[filtered_mask],
        "colors_full": colors[full_mask],
        "colors_filtered": colors[filtered_mask],
        "image": colors,
        "threshold": threshold,
    }


def infer_joint(model, paths, device, percentile):
    images = load_and_preprocess_images([str(path) for path in paths]).to(device)
    with torch.no_grad():
        # This is the only joint call: both frames are passed in one tensor.
        with _short_batch_camera_head_fp32(model, 2, "[tum-loop-joint]"):
            prediction = model(images)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        prediction["pose_enc"], images.shape[-2:]
    )
    confidence = prediction["depth_conf"][0].detach().float().cpu().numpy()
    threshold = float(np.percentile(confidence, percentile) + 1e-6)
    results = []
    for index in range(2):
        results.append(extract_frame(
            prediction["depth"][0, index].detach().float().cpu().numpy(),
            confidence[index],
            extrinsic[0, index].detach().float().cpu().numpy(),
            intrinsic[0, index].detach().float().cpu().numpy(),
            images[index].detach().float().cpu().permute(1, 2, 0).numpy(),
            threshold,
        ))
    del prediction, images, extrinsic, intrinsic
    torch.cuda.empty_cache()
    return results


def tint(colors, rgb):
    luminance = np.mean(np.clip(colors, 0.0, 1.0), axis=1, keepdims=True)
    return np.clip(0.25 * colors + 0.75 * luminance * np.asarray(rgb), 0.0, 1.0)


def transform_points(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def write_cloud(path, point_sets, color_sets):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.concatenate(point_sets).astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(np.clip(np.concatenate(color_sets), 0, 1))
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False):
        raise RuntimeError(f"Failed to write {path}")


def equal_axes(axis, points, radius=None):
    lower, upper = np.percentile(points, [1, 99], axis=0)
    center = (lower + upper) / 2.0
    if radius is None:
        radius = max(float(np.max(upper - lower)) / 2.0, 1e-6)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def add_cloud_axis(axis, point_sets, color_sets, title, rng, max_points, radius=None):
    shown_points, shown_colors = [], []
    for points, colors in zip(point_sets, color_sets):
        count = min(len(points), max_points)
        indices = rng.choice(len(points), count, replace=False)
        shown_points.append(points[indices])
        shown_colors.append(colors[indices])
    points = np.concatenate(shown_points)
    colors = np.concatenate(shown_colors)
    axis.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=0.25, linewidths=0)
    equal_axes(axis, points, radius=radius)
    axis.view_init(elev=22, azim=-65)
    axis.set_title(title)
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-disparity", type=float, default=50.0)
    parser.add_argument("--submap-size", type=int, default=16)
    parser.add_argument("--overlap", type=int, default=1)
    parser.add_argument("--confidence-percentile", type=float, default=25.0)
    parser.add_argument("--plot-points-per-frame", type=int, default=25000)
    parser.add_argument(
        "--first-selected-index", type=int, default=0,
        help="Selected-keyframe index for the first reference (default: 0).",
    )
    parser.add_argument(
        "--second-selected-index", type=int, default=None,
        help="Selected-keyframe index for the second reference; default uses the detected loop frame.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selection = select_keyframes_and_loop(
        args.dataset_root, args.min_disparity, args.submap_size, args.overlap
    )
    first_index = args.first_selected_index
    second_index = (
        selection["loop_selected_index"]
        if args.second_selected_index is None
        else args.second_selected_index
    )
    if not 0 <= first_index < len(selection["selected"]):
        raise ValueError(f"first selected index {first_index} is out of range")
    if not 0 <= second_index < len(selection["selected"]):
        raise ValueError(f"second selected index {second_index} is out of range")

    # A keyframe at an overlap boundary belongs to both adjacent batches.  For
    # an explicitly requested first-submap frame, keep it in submap 0 instead
    # of silently moving selected index == submap_size into submap 1.
    first_submap = min(first_index // args.submap_size, len(selection["batches"]) - 1)
    first_frame = first_index - first_submap * args.submap_size
    if first_index <= args.submap_size:
        first_submap, first_frame = 0, first_index
    second_submap = min(second_index // args.submap_size, len(selection["batches"]) - 1)
    second_frame = second_index - second_submap * args.submap_size
    paths = [selection["selected"][first_index], selection["selected"][second_index]]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGT()
    weights_url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(weights_url))
    model.eval().to(torch.bfloat16).to(device)

    original = [
        infer_original_reference(
            model, selection["batches"][first_submap], first_frame, device,
            args.confidence_percentile, "[tum-original-submap0]",
        ),
        infer_original_reference(
            model, selection["batches"][second_submap], second_frame, device,
            args.confidence_percentile, "[tum-original-loop-submap]",
        ),
    ]
    joint = infer_joint(model, paths, device, args.confidence_percentile)

    gt = selection["groundtruth"]
    gt_indices = selection["selected_gt_indices"]
    world_from_camera0 = pose_matrix(gt[gt_indices[first_index]])
    world_from_camerak = pose_matrix(gt[gt_indices[second_index]])
    camera0_from_camerak = np.linalg.inv(world_from_camera0) @ world_from_camerak
    relative_translation = float(
        np.linalg.norm(gt[gt_indices[first_index], 1:4] - gt[gt_indices[second_index], 1:4])
    )
    relative_rotation = quaternion_angle_degrees(
        gt[gt_indices[first_index], 4:8], gt[gt_indices[second_index], 4:8]
    )

    red_full = tint(original[0]["colors_full"], (1.0, 0.15, 0.05))
    blue_full = tint(original[1]["colors_full"], (0.05, 0.35, 1.0))
    red_filtered = tint(original[0]["colors_filtered"], (1.0, 0.15, 0.05))
    blue_filtered = tint(original[1]["colors_filtered"], (0.05, 0.35, 1.0))
    joint_red_full = tint(joint[0]["colors_full"], (1.0, 0.15, 0.05))
    joint_blue_full = tint(joint[1]["colors_full"], (0.05, 0.35, 1.0))
    joint_red_filtered = tint(joint[0]["colors_filtered"], (1.0, 0.15, 0.05))
    joint_blue_filtered = tint(joint[1]["colors_filtered"], (0.05, 0.35, 1.0))

    original_gt_full = [
        original[0]["camera_full"],
        transform_points(original[1]["camera_full"], camera0_from_camerak),
    ]
    original_gt_filtered = [
        original[0]["camera_filtered"],
        transform_points(original[1]["camera_filtered"], camera0_from_camerak),
    ]

    exports = {
        "01_original_F0": ([original[0]["camera_full"]], [original[0]["colors_full"]]),
        "02_original_Fk": ([original[1]["camera_full"]], [original[1]["colors_full"]]),
        "03_original_GT_pose_overlay": (original_gt_full, [red_full, blue_full]),
        "04_joint_VGGT_overlay": (
            [joint[0]["world_full"], joint[1]["world_full"]],
            [joint_red_full, joint_blue_full],
        ),
    }
    filtered_exports = {
        "01_original_F0_filtered": (
            [original[0]["camera_filtered"]], [original[0]["colors_filtered"]]
        ),
        "02_original_Fk_filtered": (
            [original[1]["camera_filtered"]], [original[1]["colors_filtered"]]
        ),
        "03_original_GT_pose_overlay_filtered": (
            original_gt_filtered, [red_filtered, blue_filtered]
        ),
        "04_joint_VGGT_overlay_filtered": (
            [joint[0]["world_filtered"], joint[1]["world_filtered"]],
            [joint_red_filtered, joint_blue_filtered],
        ),
    }
    for name, (point_sets, color_sets) in {**exports, **filtered_exports}.items():
        write_cloud(args.output_dir / f"{name}.pcd", point_sets, color_sets)

    rng = np.random.default_rng(0)
    figure = plt.figure(figsize=(18, 14))
    panels = [
        ([original[0]["camera_full"]], [original[0]["colors_full"]], "1. Original F0 in camera coordinates"),
        ([original[1]["camera_full"]], [original[1]["colors_full"]], "2. Original Fk in camera coordinates"),
        (original_gt_full, [red_full, blue_full], "3. Original clouds + GT relative SE(3), no scaling"),
        ([joint[0]["world_full"], joint[1]["world_full"]], [joint_red_full, joint_blue_full], "4. One joint two-frame VGGT call"),
    ]
    # Keep an identical metric span in every panel so independent autoscaling
    # cannot visually hide a scale difference between the four results.
    shared_radius = max(
        float(np.max(np.diff(np.percentile(np.concatenate(point_sets), [1, 99], axis=0), axis=0))) / 2.0
        for point_sets, _, _ in panels
    )
    for panel_index, (point_sets, color_sets, title) in enumerate(panels, 1):
        axis = figure.add_subplot(2, 2, panel_index, projection="3d")
        add_cloud_axis(
            axis, point_sets, color_sets, title, rng,
            args.plot_points_per_frame, radius=shared_radius,
        )
    figure.suptitle(f"TUM loop pair: {paths[0].name} vs {paths[1].name}", fontsize=15)
    figure.tight_layout()
    figure.savefig(args.output_dir / "four_cloud_comparison.png", dpi=180)
    plt.close(figure)

    input_figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, image, label in zip(axes, [original[0]["image"], original[1]["image"]], ["F0", "Fk"]):
        axis.imshow(image)
        axis.set_title(label)
        axis.axis("off")
    input_figure.tight_layout()
    input_figure.savefig(args.output_dir / "input_pair.png", dpi=180)
    plt.close(input_figure)

    stats = {
        "device": device,
        "min_disparity": args.min_disparity,
        "submap_size": args.submap_size,
        "overlap": args.overlap,
        "raw_frame_count": len(selection["all_images"]),
        "selected_keyframe_count": len(selection["selected"]),
        "submap_count": len(selection["batches"]),
        "original_vggt_calls": 2,
        "frames_per_original_call": args.submap_size + args.overlap,
        "joint_vggt_calls": 1,
        "frames_in_joint_call": 2,
        "F0": {
            "image": str(paths[0]), "selected_keyframe_index": first_index,
            "submap": first_submap, "frame": first_frame,
            "full_points": len(original[0]["camera_full"]),
            "filtered_points": len(original[0]["camera_filtered"]),
            "original_median_camera_radius": float(np.median(np.linalg.norm(original[0]["camera_filtered"], axis=1))),
            "joint_median_camera_radius": float(np.median(np.linalg.norm(joint[0]["camera_filtered"], axis=1))),
        },
        "Fk": {
            "image": str(paths[1]), "selected_keyframe_index": second_index,
            "submap": second_submap, "frame": second_frame,
            "full_points": len(original[1]["camera_full"]),
            "filtered_points": len(original[1]["camera_filtered"]),
            "original_median_camera_radius": float(np.median(np.linalg.norm(original[1]["camera_filtered"], axis=1))),
            "joint_median_camera_radius": float(np.median(np.linalg.norm(joint[1]["camera_filtered"], axis=1))),
        },
        "groundtruth_relative_pose": {
            "translation_m": relative_translation,
            "rotation_deg": relative_rotation,
            "camera0_from_camerak": camera0_from_camerak.tolist(),
            "scale_applied": 1.0,
        },
        "coordinate_notes": {
            "individual_originals": "Each cloud is in its own reference camera coordinates.",
            "original_overlay": "Fk is transformed to F0 with TUM GT SE(3); no scaling is applied.",
            "joint_overlay": "Both clouds use the shared frame predicted by one two-image VGGT call.",
        },
    }
    (args.output_dir / "comparison_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
