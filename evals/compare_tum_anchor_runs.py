#!/usr/bin/env python3
"""Compare matched TUM Anchor-on/off trajectories against ground truth."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.eval_tum_trajectory_local import (
    deduplicate_estimates,
    read_tum_trajectory,
    umeyama,
)


def associate(groundtruth, estimate, max_time_difference=0.02):
    estimate, _ = deduplicate_estimates(estimate)
    estimated_points, groundtruth_points, times = [], [], []
    for row in estimate:
        index = int(np.argmin(np.abs(groundtruth[:, 0] - row[0])))
        if abs(groundtruth[index, 0] - row[0]) <= max_time_difference:
            estimated_points.append(row[1:4])
            groundtruth_points.append(groundtruth[index, 1:4])
            times.append(row[0])
    return np.asarray(estimated_points), np.asarray(groundtruth_points), np.asarray(times)


def percent_change(on_value, off_value):
    return float((on_value / off_value - 1.0) * 100.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groundtruth", required=True, type=Path)
    parser.add_argument("--anchor-on", required=True, type=Path)
    parser.add_argument("--anchor-off", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-time-difference", type=float, default=0.02)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    groundtruth = read_tum_trajectory(args.groundtruth)
    trajectories = {}
    for label, path in (("Anchor on", args.anchor_on), ("Anchor off", args.anchor_off)):
        estimate = read_tum_trajectory(path)
        source, target, times = associate(
            groundtruth, estimate, args.max_time_difference
        )
        se3, _, _, _ = umeyama(source, target, False)
        sim3, _, _, scale = umeyama(source, target, True)
        trajectories[label] = {
            "target": target,
            "times": times,
            "se3": se3,
            "sim3": sim3,
            "scale": scale,
            "se3_errors": np.linalg.norm(se3 - target, axis=1),
            "sim3_errors": np.linalg.norm(sim3 - target, axis=1),
        }

    on = trajectories["Anchor on"]
    off = trajectories["Anchor off"]
    metrics = {
        "anchor_on": {
            "se3_ate_rmse_m": float(np.sqrt(np.mean(on["se3_errors"] ** 2))),
            "sim3_ate_rmse_m": float(np.sqrt(np.mean(on["sim3_errors"] ** 2))),
            "sim3_endpoint_error_m": float(on["sim3_errors"][-1]),
            "sim3_scale": float(on["scale"]),
        },
        "anchor_off": {
            "se3_ate_rmse_m": float(np.sqrt(np.mean(off["se3_errors"] ** 2))),
            "sim3_ate_rmse_m": float(np.sqrt(np.mean(off["sim3_errors"] ** 2))),
            "sim3_endpoint_error_m": float(off["sim3_errors"][-1]),
            "sim3_scale": float(off["scale"]),
        },
    }
    metrics["anchor_on_change_percent_vs_off"] = {
        key: percent_change(metrics["anchor_on"][key], metrics["anchor_off"][key])
        for key in ("se3_ate_rmse_m", "sim3_ate_rmse_m", "sim3_endpoint_error_m")
    }
    (args.output_dir / "anchor_on_vs_off.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )

    figure, axes = plt.subplots(2, 2, figsize=(15, 11))
    target = on["target"]
    axes[0, 0].plot(target[:, 0], target[:, 1], label="TUM GT", linewidth=2)
    axes[0, 0].plot(off["sim3"][:, 0], off["sim3"][:, 1], label="Anchor off")
    axes[0, 0].plot(on["sim3"][:, 0], on["sim3"][:, 1], label="Anchor on")
    axes[0, 0].set(title="Sim(3)-aligned XY trajectories", xlabel="x [m]", ylabel="y [m]")
    axes[0, 0].axis("equal")
    axes[0, 0].legend()

    time_axis = on["times"] - on["times"][0]
    axes[0, 1].plot(time_axis, off["sim3_errors"], label="Anchor off")
    axes[0, 1].plot(time_axis, on["sim3_errors"], label="Anchor on")
    axes[0, 1].set(title="Sim(3) absolute translation error", xlabel="time [s]", ylabel="error [m]")
    axes[0, 1].legend()

    names = ("SE(3) ATE RMSE", "Sim(3) ATE RMSE", "Sim(3) endpoint")
    on_values = (
        metrics["anchor_on"]["se3_ate_rmse_m"],
        metrics["anchor_on"]["sim3_ate_rmse_m"],
        metrics["anchor_on"]["sim3_endpoint_error_m"],
    )
    off_values = (
        metrics["anchor_off"]["se3_ate_rmse_m"],
        metrics["anchor_off"]["sim3_ate_rmse_m"],
        metrics["anchor_off"]["sim3_endpoint_error_m"],
    )
    x = np.arange(len(names)); width = 0.36
    axes[1, 0].bar(x - width / 2, off_values, width, label="Anchor off")
    axes[1, 0].bar(x + width / 2, on_values, width, label="Anchor on")
    axes[1, 0].set_xticks(x, names)
    axes[1, 0].set_ylabel("error [m]")
    axes[1, 0].set_title("Lower is better")
    axes[1, 0].legend()

    axes[1, 1].plot(target[:, 0], target[:, 2], label="TUM GT", linewidth=2)
    axes[1, 1].plot(off["sim3"][:, 0], off["sim3"][:, 2], label="Anchor off")
    axes[1, 1].plot(on["sim3"][:, 0], on["sim3"][:, 2], label="Anchor on")
    axes[1, 1].set(title="Sim(3)-aligned XZ trajectories", xlabel="x [m]", ylabel="z [m]")
    axes[1, 1].axis("equal")
    axes[1, 1].legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "anchor_on_vs_off.png", dpi=180)
    plt.close(figure)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
