#!/usr/bin/env python3
"""Export exactly four dense PCDs for first/last submap frames and their joint call."""

import argparse
import ast
from pathlib import Path
import sys

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


def read_logged_batches(path):
    batches = []
    marker = "['/"
    for line in path.read_text(errors="replace").splitlines():
        start = line.find(marker)
        if start < 0:
            continue
        try:
            value = ast.literal_eval(line[start:])
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
            batches.append([Path(item) for item in value])
    if not batches:
        raise ValueError(f"No submap image batches found in {path}")
    return batches


def run_vggt(model, paths, device, label):
    images = load_and_preprocess_images([str(path) for path in paths]).to(device)
    with torch.no_grad():
        with _short_batch_camera_head_fp32(model, len(paths), label):
            prediction = model(images)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        prediction["pose_enc"], images.shape[-2:]
    )
    return {
        "depth": prediction["depth"][0].detach().float().cpu().numpy(),
        "extrinsic": extrinsic[0].detach().float().cpu().numpy(),
        "intrinsic": intrinsic[0].detach().float().cpu().numpy(),
        "colors": images.detach().float().cpu().permute(0, 2, 3, 1).numpy(),
    }


def dense_frame(prediction, index, use_joint_world):
    world, camera, positive = depth_to_world_coords_points(
        prediction["depth"][index].squeeze(-1),
        prediction["extrinsic"][index],
        prediction["intrinsic"][index],
    )
    points = world if use_joint_world else camera
    valid = positive & np.all(np.isfinite(points), axis=-1)
    return points[valid], prediction["colors"][index][valid]


def write_pcd(path, points, colors):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False):
        raise RuntimeError(f"Failed to write {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-log", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    batches = read_logged_batches(args.run_log)
    first_batch, last_batch = batches[0], batches[-1]
    first_path = first_batch[0]
    last_path = last_batch[-1]
    # The final submap may repeat its last real keyframe for padding.  Export
    # the prediction at the first occurrence, i.e. its genuine batch slot.
    last_real_index = last_batch.index(last_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGT()
    weights_url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(weights_url))
    model.eval().to(torch.bfloat16).to(device)

    original_first = run_vggt(
        model, first_batch, device, "[export-original-first-submap]"
    )
    original_last = run_vggt(
        model, last_batch, device, "[export-original-last-submap]"
    )
    joint = run_vggt(
        model, [first_path, last_path], device, "[export-joint-first-last]"
    )

    exports = (
        ("01_original_submap0_first_frame.pcd", *dense_frame(original_first, 0, False)),
        ("02_original_last_submap_last_frame.pcd", *dense_frame(original_last, last_real_index, False)),
        ("03_joint_first_frame.pcd", *dense_frame(joint, 0, True)),
        ("04_joint_last_frame.pcd", *dense_frame(joint, 1, True)),
    )
    for name, points, colors in exports:
        write_pcd(args.output_dir / name, points, colors)
        print(f"{name}: {len(points)} points")
    print(f"first image: {first_path}")
    print(f"last image: {last_path} (real frame index {last_real_index})")


if __name__ == "__main__":
    main()
