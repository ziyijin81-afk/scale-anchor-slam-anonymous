#!/usr/bin/env python3
"""Re-run logged submaps and quantify every shared-frame scale alignment."""

import argparse
import ast
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vggt_slam.project_paths import prefer_bundled_third_party


prefer_bundled_third_party()

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from vggt_slam.scale_solver import estimate_scale_pairwise
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


def infer(model, paths, device, index, percentile):
    images = load_and_preprocess_images([str(path) for path in paths]).to(device)
    with torch.no_grad():
        with _short_batch_camera_head_fp32(model, len(paths), f"[diagnose-submap-{index}]"):
            prediction = model(images)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        prediction["pose_enc"], images.shape[-2:]
    )
    depth = prediction["depth"][0].detach().float().cpu().numpy()
    confidence = prediction["depth_conf"][0].detach().float().cpu().numpy()
    extrinsic = extrinsic[0].detach().float().cpu().numpy()
    intrinsic = intrinsic[0].detach().float().cpu().numpy()
    camera_points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)
    finite_conf = confidence[np.isfinite(confidence)]
    if np.ptp(finite_conf) <= np.finfo(finite_conf.dtype).eps:
        threshold = np.nextafter(
            finite_conf[0], np.asarray(-np.inf, dtype=finite_conf.dtype)
        ).item()
    else:
        threshold = float(np.percentile(finite_conf, percentile) + 1e-6)
    return {
        "first_points": camera_points[0],
        "last_points": camera_points[-1],
        "first_conf": confidence[0],
        "last_conf": confidence[-1],
        "first_intrinsic": intrinsic[0],
        "last_intrinsic": intrinsic[-1],
        "threshold": threshold,
    }


def homogeneous_intrinsic(intrinsic):
    result = np.eye(4)
    result[:3, :3] = intrinsic
    return result


def boundary_metrics(previous, current, previous_index, current_index):
    previous_conf = previous["last_conf"]
    current_conf = current["first_conf"]
    threshold = previous["threshold"]
    mask = (previous_conf > threshold) & (current_conf > threshold)
    mask_mode = "intersection"
    if np.count_nonzero(mask) < 100:
        mask = previous_conf > threshold
        mask_mode = "prior_only"
        if np.count_nonzero(mask) < 100:
            mask = previous_conf > 0
            mask_mode = "prior_positive"
    mask = mask.reshape(-1)

    transform = (
        np.linalg.inv(homogeneous_intrinsic(previous["last_intrinsic"]))
        @ homogeneous_intrinsic(current["first_intrinsic"])
    )
    current_points = current["first_points"].reshape(-1, 3)[mask]
    previous_points = previous["last_points"].reshape(-1, 3)[mask]
    transformed_current = (transform[:3, :3] @ current_points.T).T
    finite = (
        np.all(np.isfinite(transformed_current), axis=1)
        & np.all(np.isfinite(previous_points), axis=1)
    )
    transformed_current = transformed_current[finite]
    previous_points = previous_points[finite]
    scale = float(estimate_scale_pairwise(transformed_current, previous_points)[0])

    current_radius = np.linalg.norm(transformed_current, axis=1)
    previous_radius = np.linalg.norm(previous_points, axis=1)
    ratios = previous_radius / np.maximum(current_radius, 1e-12)
    ratio_mad = float(np.median(np.abs(ratios - np.median(ratios))))
    aligned = transformed_current * scale
    residual = np.linalg.norm(aligned - previous_points, axis=1)
    relative_residual = residual / np.maximum(previous_radius, 1e-12)
    dots = np.sum(transformed_current * previous_points, axis=1)
    cosine = dots / np.maximum(current_radius * previous_radius, 1e-12)
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return {
        "previous_submap": previous_index,
        "current_submap": current_index,
        "shared_image_matches": bool(
            previous_index + 1 == current_index
        ),
        "mask_mode": mask_mode,
        "valid_pixels": int(len(aligned)),
        "scale_measurement": scale,
        "ratio_mad": ratio_mad,
        "relative_ratio_mad": ratio_mad / max(scale, 1e-12),
        "median_relative_xyz_residual": float(np.median(relative_residual)),
        "p90_relative_xyz_residual": float(np.percentile(relative_residual, 90)),
        "p95_relative_xyz_residual": float(np.percentile(relative_residual, 95)),
        "median_direction_error_deg": float(np.median(angles)),
        "p90_direction_error_deg": float(np.percentile(angles, 90)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-log", required=True, type=Path)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--confidence-percentile", type=float, default=25.0)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    batches = read_logged_batches(args.run_log)[: args.count]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url(
        "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    ))
    model.eval().to(torch.bfloat16).to(device)

    results = []
    previous = None
    previous_paths = None
    for index, paths in enumerate(batches):
        current = infer(
            model, paths, device, index, args.confidence_percentile
        )
        if previous is not None:
            result = boundary_metrics(previous, current, index - 1, index)
            result["shared_image_matches"] = bool(previous_paths[-1] == paths[0])
            result["shared_image"] = str(paths[0])
            results.append(result)
            print(json.dumps(result))
        previous = current
        previous_paths = paths

    csv_path = args.output_dir / "overlap_boundary_diagnostics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (args.output_dir / "overlap_boundary_diagnostics.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )

    edges = np.arange(1, len(results) + 1)
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(edges, [r["scale_measurement"] for r in results], "o-")
    axes[0].axhline(1.0, color="gray", linestyle="--")
    axes[0].set_ylabel("overlap scale")
    axes[1].plot(edges, [r["median_relative_xyz_residual"] for r in results], "o-")
    axes[1].set_ylabel("median XYZ residual / radius")
    axes[2].semilogy(edges, [r["valid_pixels"] for r in results], "o-")
    axes[2].set_ylabel("valid pixels")
    axes[2].set_xlabel("boundary: submap (k-1) -> k")
    axes[2].set_xticks(edges)
    figure.tight_layout()
    figure.savefig(args.output_dir / "overlap_boundary_diagnostics.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
