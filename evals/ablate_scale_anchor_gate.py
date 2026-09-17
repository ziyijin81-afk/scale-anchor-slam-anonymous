#!/usr/bin/env python3
"""Offline scale-anchor gate ablation using previously logged VGGT measurements.

This script never calls VGGT.  It filters the candidate root-anchor factors in
an Anchor-OFF/ON run pair, replays the one-dimensional GTSAM scale graph, and
rebuilds the continuous Anchor-OFF trajectory with the selected scales.  Each
configuration is evaluated against the dataset's timestamped ground truth with
a global Umeyama Sim(3), matching the project's trajectory metric.
"""

from __future__ import annotations

import argparse
import ast
import csv
import contextlib
import io
import itertools
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from vggt_slam.scale_solver import ScaleFactorGraph
from vggt_slam.trajectory_evaluation import apply_sim3, umeyama


FACTOR_RE = re.compile(
    r"^\[scale-graph\] factor=(overlap|anchor) "
    r"edge=\((\d+),(\d+)\) measurement=([0-9.eE+-]+)"
)
BACKEND_RE = re.compile(r"^\[scale-graph\] backend=")
KEY_VALUE_RE = re.compile(r"([A-Za-z_]+)=([^\s]+)")


@dataclass(frozen=True)
class GateConfig:
    min_points: int
    max_relative_mad: float
    max_outlier_ratio: float
    max_log_disagreement: float
    max_overlap_log_disagreement: float | None
    allow_confidence_fallback: bool

    @property
    def name(self):
        overlap = (
            "none" if self.max_overlap_log_disagreement is None
            else f"{self.max_overlap_log_disagreement:g}"
        )
        return (
            f"p{self.min_points}_mad{self.max_relative_mad:g}_"
            f"out{self.max_outlier_ratio:g}_log{self.max_log_disagreement:g}_"
            f"ov{overlap}_fb{int(self.allow_confidence_fallback)}"
        )


@dataclass
class DatasetSpec:
    name: str
    off_dir: Path
    on_dir: Path
    reference_type: str
    reference_path: Path
    auxiliary_path: Path | None = None


def parse_final_scales(log_text):
    matches = re.findall(
        r"^\[scale-graph\] backend=.*? after=(\{.*?\}) diagnostics=",
        log_text,
        re.MULTILINE,
    )
    if not matches:
        raise ValueError("No final scale-graph state in log")
    return {
        int(key): float(value)
        for key, value in ast.literal_eval(matches[-1]).items()
    }


def _float(fields, key, default=math.inf):
    try:
        return float(fields[key])
    except (KeyError, ValueError):
        return float(default)


def parse_candidates(log_text):
    candidates = {}
    fallback = {}
    measurements = {}
    for line in log_text.splitlines():
        if line.startswith("[scale-anchor] submap=") and "hat_s_k_to_0=" in line:
            fields = dict(KEY_VALUE_RE.findall(line))
            node = int(fields["submap"])
            fallback[node] = fields.get("confidence_fallback", "False") == "True"
        elif line.startswith("[scale-anchor-gate] submap=") and "root_valid=" in line:
            fields = dict(KEY_VALUE_RE.findall(line))
            node = int(fields["submap"])
            candidates[node] = {
                "root_valid": int(fields["root_valid"]),
                "current_valid": int(fields["current_valid"]),
                "root_relative_mad": _float(fields, "root_relative_MAD"),
                "current_relative_mad": _float(fields, "current_relative_MAD"),
                "root_outlier_ratio": _float(fields, "root_outlier_ratio"),
                "current_outlier_ratio": _float(fields, "current_outlier_ratio"),
                "log_disagreement": _float(fields, "log_disagreement"),
            }
        factor = FACTOR_RE.match(line)
        if factor and factor.group(1) == "anchor":
            measurements[int(factor.group(3))] = float(factor.group(4))
    for node, candidate in candidates.items():
        candidate["confidence_fallback"] = bool(fallback.get(node, False))
        candidate["measurement"] = measurements.get(node)
    return candidates


def overlap_chain_scales(log_text):
    scales = {0: 1.0}
    for line in log_text.splitlines():
        factor = FACTOR_RE.match(line)
        if not factor or factor.group(1) != "overlap":
            continue
        node_i, node_j = int(factor.group(2)), int(factor.group(3))
        scales[node_j] = scales[node_i] * float(factor.group(4))
    return scales


def candidate_passes(node, candidate, config, overlap_scales):
    measurement = candidate.get("measurement")
    if measurement is None or not np.isfinite(measurement) or measurement <= 0:
        return False
    if candidate["confidence_fallback"] and not config.allow_confidence_fallback:
        return False
    if min(candidate["root_valid"], candidate["current_valid"]) < config.min_points:
        return False
    if max(candidate["root_relative_mad"], candidate["current_relative_mad"]) > config.max_relative_mad:
        return False
    if max(candidate["root_outlier_ratio"], candidate["current_outlier_ratio"]) > config.max_outlier_ratio:
        return False
    if candidate["log_disagreement"] > config.max_log_disagreement:
        return False
    if config.max_overlap_log_disagreement is not None:
        overlap = overlap_scales.get(node)
        if overlap is None or overlap <= 0:
            return False
        if abs(math.log(measurement / overlap)) > config.max_overlap_log_disagreement:
            return False
    return True


def replay_selected_factors(
    log_text,
    accepted_nodes,
    overlap_weight=1.0,
    anchor_weight=0.1,
    incremental=False,
):
    # Screen the final robust objective in one batch.  The winning gate is
    # subsequently validated through the real incremental iSAM2 pipeline.
    graph = ScaleFactorGraph(use_isam2=incremental)
    sigmas = {
        "overlap": 1.0 / math.sqrt(overlap_weight),
        "anchor": 1.0 / math.sqrt(anchor_weight),
    }
    with contextlib.redirect_stdout(io.StringIO()):
        pending = False
        for line in log_text.splitlines():
            factor = FACTOR_RE.match(line)
            if not factor:
                if incremental and pending and BACKEND_RE.match(line):
                    graph.optimize()
                    pending = False
                continue
            factor_type, node_i, node_j, measurement = factor.groups()
            node_i, node_j = int(node_i), int(node_j)
            if factor_type == "anchor" and node_j not in accepted_nodes:
                continue
            graph.update_factor(
                factor_type, node_i, node_j, float(measurement), sigmas[factor_type]
            )
            pending = True
        if pending or not incremental:
            graph.optimize()
    return dict(graph.nodes)


def rebuild_trajectory(off_poses, nodes, old_scales, new_scales, batch_size=17):
    if len(off_poses) != len(nodes) * batch_size:
        raise ValueError(
            f"Expected {len(nodes) * batch_size} poses, got {len(off_poses)}"
        )
    source = off_poses.reshape(len(nodes), batch_size, 8)
    rebuilt = source.copy()
    for order, node in enumerate(nodes):
        old_scale = float(old_scales[node])
        new_scale = float(new_scales[node])
        old_positions = source[order, :, 1:4]
        old_anchor = old_positions[0]
        if order == 0:
            new_anchor = old_anchor.copy()
        else:
            old_parent = source[order - 1, -1, 1:4]
            new_parent = rebuilt[order - 1, -1, 1:4]
            new_anchor = old_anchor + (new_parent - old_parent)
        rebuilt[order, :, 1:4] = (
            new_anchor + (new_scale / old_scale) * (old_positions - old_anchor)
        )
    return rebuilt.reshape(-1, 8)


def deduplicate(rows):
    timestamps = rows[:, 0]
    _, reverse_indices = np.unique(timestamps[::-1], return_index=True)
    keep = np.sort(len(timestamps) - 1 - reverse_indices)
    return rows[keep]


def interpolate(query, reference_timestamps, reference_positions):
    order = np.argsort(reference_timestamps)
    reference_timestamps = np.asarray(reference_timestamps)[order]
    reference_positions = np.asarray(reference_positions)[order]
    return np.column_stack([
        np.interp(query, reference_timestamps, reference_positions[:, axis])
        for axis in range(3)
    ])


def load_reference(spec, estimate_rows):
    rows = deduplicate(estimate_rows)
    estimate = rows[:, 1:4]
    if spec.reference_type == "indoor":
        transforms = np.loadtxt(spec.reference_path).reshape(-1, 4, 4)
        indices = np.rint(rows[:, 0]).astype(int) + 8
        valid = (indices >= 0) & (indices < len(transforms))
        return rows[valid, 0], estimate[valid], transforms[indices[valid], :3, 3]
    if spec.reference_type == "rosbag":
        from eval_rosbag_odometry_reference import read_odometry
        timestamps, positions = read_odometry(spec.reference_path)
        query = np.rint(rows[:, 0]).astype(np.int64)
        valid = (query >= timestamps.min()) & (query <= timestamps.max())
        return query[valid], estimate[valid], interpolate(query[valid], timestamps, positions)
    if spec.reference_type == "advio":
        arkit = np.loadtxt(spec.auxiliary_path, delimiter=",")
        image_indices = np.rint(rows[:, 0]).astype(int) - 1
        valid = (image_indices >= 0) & (image_indices < len(arkit))
        query = arkit[image_indices[valid], 0]
        gt = np.loadtxt(spec.reference_path, comments="#")
        return query, estimate[valid], interpolate(query, gt[:, 0], gt[:, 1:4])
    if spec.reference_type == "lamar":
        reference = np.loadtxt(spec.reference_path, delimiter=",", comments="#", usecols=(0, 6, 7, 8))
        query = rows[:, 0]
        valid = (query >= reference[:, 0].min()) & (query <= reference[:, 0].max())
        return query[valid], estimate[valid], interpolate(query[valid], reference[:, 0], reference[:, 1:4])
    raise ValueError(f"Unknown reference type {spec.reference_type}")


def evaluate(spec, poses):
    timestamps, estimate, reference = load_reference(spec, poses)
    scale, rotation, translation = umeyama(estimate, reference)
    aligned = apply_sim3(estimate, scale, rotation, translation)
    errors = np.linalg.norm(aligned - reference, axis=1)
    return {
        "matched_poses": int(len(estimate)),
        "ate_rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        "ate_median_m": float(np.median(errors)),
        "ate_p95_m": float(np.percentile(errors, 95)),
        "sim3_scale": float(scale),
    }, (timestamps, reference, aligned, errors)


def dataset_specs():
    workspace = PROJECT_ROOT.parent
    return [
        DatasetSpec(
            "Indoor_ntu",
            workspace / "Indoor_ntu/result/long_corridor01_md25_anchor_off_no_loop_current_replay_20260722",
            workspace / "Indoor_ntu/result/long_corridor01_md25_anchor_on_w01_no_gate_no_loop_full_pcd_20260722",
            "indoor",
            workspace / "Indoor_ntu/dataset/groundtruth/cam_trajs.txt",
        ),
        DatasetSpec(
            "rosbag2",
            workspace / "rosbag/result/rosbag2_anchor_off_md25_no_loop_20260722",
            workspace / "rosbag/result/rosbag2_anchor_on_md25_no_gate_no_loop_20260722",
            "rosbag",
            workspace / "rosbag/dataset/rosbag2_2026_07_21-12_02_14",
        ),
        DatasetSpec(
            "ADVIO",
            workspace / "ADVIO/result/advio02_iphone_anchor_off_md25_no_loop_full_pcd",
            workspace / "ADVIO/result/advio02_iphone_anchor_on_no_gate_md25_no_loop_full_pcd",
            "advio",
            workspace / "ADVIO/result/advio02_anchor_scale_drift/groundtruth_tum.txt",
            workspace / "ADVIO/dataset/advio-02/iphone/arkit.csv",
        ),
        DatasetSpec(
            "LaMAR",
            workspace / "LaMAR/result/HGE/ios_2022-06-21_11.43.41_md25_anchor_off_no_loop_20260721",
            workspace / "LaMAR/result/HGE/ios_2022-06-21_11.43.41_md25_anchor_on_w01_no_gate_no_loop_20260721",
            "lamar",
            workspace / "LaMAR/dataset/HGE/raw_iphone/ios_2022-06-21_11.43.41/trajectories.txt",
        ),
    ]


def gate_grid():
    for values in itertools.product(
        (100, 500),
        (0.10, 0.15, 0.20, 0.25, 0.30),
        (0.25, 0.35, 0.50),
        (0.10, 0.15, 0.25),
        (0.35, 0.50, 0.75, None),
        (False, True),
    ):
        yield GateConfig(*values)


def write_comparison_plot(path, plot_data, title):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for label, color in (("anchor_off", "tab:orange"), ("best_gate", "tab:blue")):
        timestamps, reference, aligned, errors = plot_data[label]
        if label == "anchor_off":
            axes[0].plot(reference[:, 0], reference[:, 1], "k-", linewidth=2, label="ground truth")
        axes[0].plot(aligned[:, 0], aligned[:, 1], color=color, label=label)
        elapsed = np.arange(len(errors)) if len(timestamps) < 2 else np.arange(len(errors))
        axes[1].plot(elapsed, errors, color=color, label=label)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_title(f"{title}: global Umeyama Sim(3)")
    axes[0].legend()
    axes[1].set_xlabel("matched pose index")
    axes[1].set_ylabel("translation error [m]")
    axes[1].set_title("Absolute translation error")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_summary_plot(path, baseline_plots, best_plots, baseline, validation):
    figure, axes = plt.subplots(2, 2, figsize=(13, 11))
    for axis, name in zip(axes.flat, baseline_plots):
        _, reference, off_aligned, _ = baseline_plots[name]
        _, _, best_aligned, _ = best_plots[name]
        axis.plot(reference[:, 0], reference[:, 1], "k-", linewidth=2, label="GT")
        axis.plot(off_aligned[:, 0], off_aligned[:, 1], color="tab:orange", label="Anchor OFF")
        axis.plot(best_aligned[:, 0], best_aligned[:, 1], color="tab:blue", label="Best gate")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_title(
            f"{name}: {baseline[name]['ate_rmse_m']:.3f} → "
            f"{validation[name]['ate_rmse_m']:.3f} m"
        )
        axis.legend(fontsize=8)
    figure.suptitle("Scale Anchor gate: global Umeyama Sim(3) trajectory comparison")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchor-weight", type=float, default=0.1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cached = {}
    baseline = {}
    for spec in dataset_specs():
        off_log = (spec.off_dir / "run.log").read_text(errors="replace")
        on_log = (spec.on_dir / "run.log").read_text(errors="replace")
        off_scales = parse_final_scales(off_log)
        off_poses = np.loadtxt(spec.off_dir / "poses.txt", ndmin=2)
        nodes = sorted(off_scales)
        candidates = parse_candidates(on_log)
        overlap_scales = overlap_chain_scales(on_log)
        off_metrics, off_plot = evaluate(spec, off_poses)
        baseline[spec.name] = off_metrics
        cached[spec.name] = {
            "spec": spec,
            "on_log": on_log,
            "off_scales": off_scales,
            "off_poses": off_poses,
            "nodes": nodes,
            "candidates": candidates,
            "overlap_scales": overlap_scales,
            "off_plot": off_plot,
        }

    records = []
    # Many numerical threshold combinations retain exactly the same anchors.
    # Cache those equivalent graph solutions instead of solving them again.
    solution_cache = {name: {} for name in cached}
    for config in gate_grid():
        per_dataset = {}
        ratios = []
        accepted_total = 0
        for name, data in cached.items():
            accepted = frozenset(
                node for node, candidate in data["candidates"].items()
                if candidate_passes(node, candidate, config, data["overlap_scales"])
            )
            metrics = solution_cache[name].get(accepted)
            if metrics is None:
                new_scales = replay_selected_factors(
                    data["on_log"], accepted, anchor_weight=args.anchor_weight
                )
                if sorted(new_scales) != data["nodes"]:
                    raise RuntimeError(f"Scale nodes differ for {name}/{config.name}")
                poses = rebuild_trajectory(
                    data["off_poses"], data["nodes"], data["off_scales"], new_scales
                )
                metrics, _ = evaluate(data["spec"], poses)
                metrics["accepted_anchor_count"] = len(accepted)
                metrics["accepted_anchor_ids"] = sorted(accepted)
                solution_cache[name][accepted] = metrics
            per_dataset[name] = metrics
            ratios.append(metrics["ate_rmse_m"] / baseline[name]["ate_rmse_m"])
            accepted_total += len(accepted)
        record = {
            "name": config.name,
            "gate": asdict(config),
            "geomean_rmse_ratio_vs_off": float(np.exp(np.mean(np.log(ratios)))),
            "mean_rmse_ratio_vs_off": float(np.mean(ratios)),
            "datasets_improved": int(sum(ratio < 1.0 for ratio in ratios)),
            "accepted_anchor_total": int(accepted_total),
            "datasets": per_dataset,
        }
        records.append(record)

    records.sort(key=lambda row: (
        -row["datasets_improved"],
        row["geomean_rmse_ratio_vs_off"],
        row["mean_rmse_ratio_vs_off"],
        -row["gate"]["min_points"],
        row["gate"]["max_relative_mad"],
        row["gate"]["max_outlier_ratio"],
        row["gate"]["max_log_disagreement"],
    ))
    best = records[0]
    with (args.output_dir / "gate_ablation.csv").open("w", newline="") as stream:
        fieldnames = [
            "rank", "name", "datasets_improved", "geomean_rmse_ratio_vs_off",
            "mean_rmse_ratio_vs_off", "accepted_anchor_total",
        ] + [f"{name}_rmse" for name in cached] + [f"{name}_anchors" for name in cached]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for rank, record in enumerate(records, 1):
            row = {key: record[key] for key in fieldnames if key in record}
            row["rank"] = rank
            for name in cached:
                row[f"{name}_rmse"] = record["datasets"][name]["ate_rmse_m"]
                row[f"{name}_anchors"] = record["datasets"][name]["accepted_anchor_count"]
            writer.writerow(row)

    best_dir = args.output_dir / "best"
    best_dir.mkdir(exist_ok=True)
    best_config = GateConfig(**best["gate"])
    incremental_validation = {}
    best_plots = {}
    for name, data in cached.items():
        accepted = {
            node for node, candidate in data["candidates"].items()
            if candidate_passes(node, candidate, best_config, data["overlap_scales"])
        }
        new_scales = replay_selected_factors(
            data["on_log"], accepted, anchor_weight=args.anchor_weight,
            incremental=True,
        )
        poses = rebuild_trajectory(
            data["off_poses"], data["nodes"], data["off_scales"], new_scales
        )
        incremental_metrics, best_plot = evaluate(data["spec"], poses)
        incremental_metrics["accepted_anchor_count"] = len(accepted)
        incremental_metrics["accepted_anchor_ids"] = sorted(accepted)
        incremental_validation[name] = incremental_metrics
        best_plots[name] = best_plot
        dataset_dir = best_dir / name
        dataset_dir.mkdir(exist_ok=True)
        np.savetxt(dataset_dir / "poses.txt", poses, fmt="%.8f")
        write_comparison_plot(
            dataset_dir / "trajectory_comparison.png",
            {"anchor_off": data["off_plot"], "best_gate": best_plot},
            name,
        )
    write_summary_plot(
        args.output_dir / "best_gate_trajectory_comparison.png",
        {name: data["off_plot"] for name, data in cached.items()},
        best_plots,
        baseline,
        incremental_validation,
    )
    with (args.output_dir / "best_gate_metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "dataset", "anchor_off_ate_rmse_m", "best_gate_ate_rmse_m",
            "improvement_percent", "accepted_anchor_count",
        ])
        for name in cached:
            off_rmse = baseline[name]["ate_rmse_m"]
            best_rmse = incremental_validation[name]["ate_rmse_m"]
            writer.writerow([
                name, off_rmse, best_rmse,
                100.0 * (off_rmse - best_rmse) / off_rmse,
                incremental_validation[name]["accepted_anchor_count"],
            ])
    (args.output_dir / "gate_ablation.json").write_text(json.dumps({
        "metric": "global Umeyama Sim(3) ATE RMSE",
        "anchor_weight": args.anchor_weight,
        "baseline": baseline,
        "best": best,
        "best_incremental_isam2_validation": incremental_validation,
        "top20": records[:20],
        "configurations_evaluated": len(records),
        "unique_graph_solutions": {
            name: len(solutions) for name, solutions in solution_cache.items()
        },
        "offline_replay": True,
    }, indent=2) + "\n")
    print(json.dumps({"baseline": baseline, "best": best}, indent=2))


if __name__ == "__main__":
    main()
