#!/usr/bin/env python3
"""Uniformly sample a video, run one joint VGGT inference, and export PCDs."""

import argparse
import json
import tempfile
from pathlib import Path

import cv2
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


def sample_video(video_path: Path, output_dir: Path, frame_count: int):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if total_frames < 2:
        raise ValueError(f"Video contains too few frames: {total_frames}")
    count = min(frame_count, total_frames)
    indices = np.linspace(0, total_frames - 1, count, dtype=np.int64)
    indices = np.unique(indices)
    image_paths = []
    raw_rgb = []
    for order, frame_index in enumerate(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, bgr = capture.read()
        if not success:
            raise RuntimeError(f"Unable to read video frame {frame_index}")
        path = output_dir / f"sample_{order:02d}_frame_{frame_index:06d}.jpg"
        if not cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Unable to write temporary frame: {path}")
        image_paths.append(path)
        raw_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    return image_paths, raw_rgb, indices, total_frames, fps


def write_cloud(path: Path, points: np.ndarray, colors: np.ndarray):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False):
        raise RuntimeError(f"Open3D failed to write {path}")


def save_contact_sheet(images, indices, output_path: Path):
    columns = 5
    rows = int(np.ceil(len(images) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(15, 3 * rows))
    axes = np.asarray(axes).reshape(-1)
    for axis, image, frame_index in zip(axes, images, indices):
        axis.imshow(image)
        axis.set_title(f"frame {int(frame_index)}")
        axis.axis("off")
    for axis in axes[len(images):]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frame-count", type=int, default=17)
    parser.add_argument("--confidence-percentile", type=float, default=25.0)
    args = parser.parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    if args.frame_count < 2:
        raise ValueError("--frame-count must be at least 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with tempfile.TemporaryDirectory(prefix="vggt_video_frames_") as temp:
        image_paths, raw_rgb, frame_indices, total_frames, fps = sample_video(
            args.video, Path(temp), args.frame_count
        )
        images = load_and_preprocess_images([str(path) for path in image_paths]).to(device)
        model = VGGT()
        weights_url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(weights_url))
        model.eval().to(torch.bfloat16).to(device)
        with torch.no_grad():
            prediction = model(images)

        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            prediction["pose_enc"], images.shape[-2:]
        )
        depth = prediction["depth"][0].detach().float().cpu().numpy()
        confidence = prediction["depth_conf"][0].detach().float().cpu().numpy()
        extrinsic = extrinsic[0].detach().float().cpu().numpy()
        intrinsic = intrinsic[0].detach().float().cpu().numpy()
        colors = images.detach().float().cpu().permute(0, 2, 3, 1).numpy()

        filtered_points, filtered_colors = [], []
        all_points, all_colors = [], []
        statistics = []
        for order in range(len(image_paths)):
            world_points, _, positive_depth = depth_to_world_coords_points(
                depth[order].squeeze(-1), extrinsic[order], intrinsic[order]
            )
            finite = np.all(np.isfinite(world_points), axis=-1)
            all_valid = positive_depth & finite
            threshold = float(
                np.percentile(confidence[order][all_valid], args.confidence_percentile)
                + 1e-6
            )
            selected = all_valid & (confidence[order] > threshold)
            filtered_points.append(world_points[selected])
            filtered_colors.append(colors[order][selected])
            all_points.append(world_points[all_valid])
            all_colors.append(colors[order][all_valid])

            world_from_camera = np.eye(4, dtype=np.float64)
            world_from_camera[:3, :4] = extrinsic[order]
            camera_center = np.linalg.inv(world_from_camera)[:3, 3]
            radii = np.linalg.norm(world_points[selected] - camera_center, axis=1)
            statistics.append({
                "sample_order": order,
                "source_frame_index": int(frame_indices[order]),
                "source_time_s": float(frame_indices[order] / fps) if fps > 0 else None,
                "confidence_threshold": threshold,
                "filtered_points": int(np.count_nonzero(selected)),
                "all_positive_finite_points": int(np.count_nonzero(all_valid)),
                "median_camera_radius": float(np.median(radii)),
                "camera_center_joint_world": camera_center.tolist(),
            })

        filtered_points = np.concatenate(filtered_points)
        filtered_colors = np.concatenate(filtered_colors)
        all_points = np.concatenate(all_points)
        all_colors = np.concatenate(all_colors)
        write_cloud(
            args.output_dir / "vggt_video_conf_filtered_rgb.pcd",
            filtered_points,
            filtered_colors,
        )
        write_cloud(
            args.output_dir / "vggt_video_all_valid_rgb.pcd",
            all_points,
            all_colors,
        )
        save_contact_sheet(
            raw_rgb, frame_indices, args.output_dir / "sampled_frames_contact_sheet.png"
        )

    result = {
        "video": str(args.video),
        "device": device,
        "video_total_frames": total_frames,
        "video_fps": fps,
        "video_duration_s": float(total_frames / fps) if fps > 0 else None,
        "single_vggt_call_frame_count": len(frame_indices),
        "sampled_frame_indices": frame_indices.tolist(),
        "confidence_percentile_per_frame": args.confidence_percentile,
        "filtered_point_count": int(len(filtered_points)),
        "all_positive_finite_point_count": int(len(all_points)),
        "frames": statistics,
    }
    (args.output_dir / "vggt_video_stats.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
