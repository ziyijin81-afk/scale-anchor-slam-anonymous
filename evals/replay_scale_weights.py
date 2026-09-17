#!/usr/bin/env python3
"""Replay logged scale factors with new weights, without running VGGT again."""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np

# Do not let a globally configured PYTHONPATH select the separate legacy
# VGGT-SLAM checkout when this script is launched from the project wrapper.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from vggt_slam.scale_solver import ScaleFactorGraph


FACTOR_RE = re.compile(
    r"^\[scale-graph\] factor=(overlap|anchor) "
    r"edge=\((\d+),(\d+)\) measurement=([0-9.eE+-]+)",
)
BACKEND_RE = re.compile(r"^\[scale-graph\] backend=")


def parse_final_scales(log_text):
    matches = re.findall(
        r"^\[scale-graph\] backend=.*? after=(\{.*?\}) diagnostics=",
        log_text,
        re.MULTILINE,
    )
    if not matches:
        raise ValueError("No scale-graph snapshot found in run.log")
    return {int(node): float(value) for node, value in ast.literal_eval(matches[-1]).items()}


def replay_factors(log_text, overlap_weight, anchor_weight):
    if overlap_weight <= 0 or anchor_weight < 0:
        raise ValueError("Overlap weight must be positive and Anchor weight non-negative")
    sigmas = {
        "overlap": 1.0 / np.sqrt(float(overlap_weight)),
        "anchor": (
            None
            if anchor_weight == 0
            else 1.0 / np.sqrt(float(anchor_weight))
        ),
    }
    graph = ScaleFactorGraph(use_isam2=True)
    factor_count = {"overlap": 0, "anchor": 0}
    pending = False
    for line in log_text.splitlines():
        factor_match = FACTOR_RE.match(line)
        if factor_match:
            factor_type, node_i, node_j, measurement = factor_match.groups()
            if factor_type == "anchor" and anchor_weight == 0:
                continue
            node_i, node_j = int(node_i), int(node_j)
            measurement = float(measurement)
            factor_sigma = sigmas[factor_type]
            graph.update_factor(
                factor_type, node_i, node_j, measurement, factor_sigma
            )
            factor_count[factor_type] += 1
            pending = True
        elif pending and BACKEND_RE.match(line):
            graph.optimize()
            pending = False
    if pending:
        graph.optimize()
    if graph.nodes.get(0) != 1.0:
        raise RuntimeError(f"Scale replay changed fixed S_0: {graph.nodes.get(0)!r}")
    if any(not np.isfinite(value) or value <= 0 for value in graph.nodes.values()):
        raise RuntimeError(f"Scale replay produced an invalid scale: {graph.nodes}")
    return graph, factor_count, sigmas



def rebuild_trajectory(poses, nodes, old_scales, new_scales, batch_size):
    """Apply new final scales about each submap anchor and propagate shifts."""
    expected_rows = len(nodes) * batch_size
    if poses.shape != (expected_rows, 8):
        raise ValueError(
            f"Expected poses shape ({expected_rows}, 8), got {poses.shape}; "
            "check --batch-size and the source run"
        )

    source_blocks = poses.reshape(len(nodes), batch_size, 8)
    rebuilt = source_blocks.copy()
    for order, node in enumerate(nodes):
        old_scale = float(old_scales[node])
        new_scale = float(new_scales[node])
        old_positions = source_blocks[order, :, 1:4]
        old_anchor = old_positions[0]
        if order == 0:
            new_anchor = old_anchor.copy()
        else:
            old_parent = source_blocks[order - 1, -1, 1:4]
            new_parent = rebuilt[order - 1, -1, 1:4]
            # Preserve the raw boundary offset while propagating every earlier
            # submap's newly scaled endpoint.
            new_anchor = old_anchor + (new_parent - old_parent)
        rebuilt[order, :, 1:4] = (
            new_anchor + (new_scale / old_scale) * (old_positions - old_anchor)
        )
    return rebuilt.reshape(expected_rows, 8)


def main():
    parser = argparse.ArgumentParser(
        description="Replay scale factors with fixed weights without VGGT inference"
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overlap-weight", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=9)
    args = parser.parse_args()

    log_path = args.run_dir / "run.log"
    poses_path = args.run_dir / "poses.txt"
    if not log_path.is_file() or not poses_path.is_file():
        raise FileNotFoundError("The source run must contain run.log and poses.txt")

    log_text = log_path.read_text(errors="replace")
    old_scales = parse_final_scales(log_text)
    graph, factor_count, sigmas = replay_factors(
        log_text, args.overlap_weight, args.anchor_weight
    )
    new_scales = dict(graph.nodes)
    nodes = sorted(old_scales)
    if nodes != sorted(new_scales):
        raise RuntimeError(
            f"Replayed nodes differ from source nodes: source={nodes}, replay={sorted(new_scales)}"
        )

    poses = np.loadtxt(poses_path, ndmin=2)
    rebuilt = rebuild_trajectory(
        poses, nodes, old_scales, new_scales, args.batch_size
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output_dir / "poses.txt", rebuilt, fmt="%.8f")

    summary = {
        "source_run": str(args.run_dir),
        "vggt_rerun": False,
        "batch_size": args.batch_size,
        "factor_count": factor_count,
        "overlap_weight": float(args.overlap_weight),
        "anchor_weight": float(args.anchor_weight),
        "overlap_sigma": float(sigmas["overlap"]),
        "anchor_sigma": (
            None if sigmas["anchor"] is None else float(sigmas["anchor"])
        ),
        "s0": float(new_scales[0]),
        "minimum_scale": float(min(new_scales.values())),
        "old_scales": old_scales,
        "new_scales": new_scales,
        "new_over_old_corrections": {
            node: new_scales[node] / old_scales[node] for node in nodes
        },
    }
    (args.output_dir / "scale_replay_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
