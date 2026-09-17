#!/usr/bin/env python3
"""Sweep fixed Anchor weights using cached measurements and the selected gate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "evals"))

from ablate_scale_anchor_gate import (
    GateConfig,
    candidate_passes,
    dataset_specs,
    evaluate,
    overlap_chain_scales,
    parse_candidates,
    parse_final_scales,
    rebuild_trajectory,
    replay_selected_factors,
)


BEST_GATE = GateConfig(
    min_points=500,
    max_relative_mad=0.20,
    max_outlier_ratio=0.35,
    max_log_disagreement=0.15,
    max_overlap_log_disagreement=None,
    allow_confidence_fallback=True,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--weights", type=float, nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cached = {}
    baseline = {}
    for spec in dataset_specs():
        off_log = (spec.off_dir / "run.log").read_text(errors="replace")
        on_log = (spec.on_dir / "run.log").read_text(errors="replace")
        off_poses = np.loadtxt(spec.off_dir / "poses.txt", ndmin=2)
        off_metrics, _ = evaluate(spec, off_poses)
        candidates = parse_candidates(on_log)
        overlap_scales = overlap_chain_scales(on_log)
        accepted = {
            node for node, candidate in candidates.items()
            if candidate_passes(node, candidate, BEST_GATE, overlap_scales)
        }
        cached[spec.name] = {
            "spec": spec,
            "on_log": on_log,
            "off_log": off_log,
            "off_poses": off_poses,
            "off_scales": parse_final_scales(off_log),
            "accepted": accepted,
        }
        baseline[spec.name] = off_metrics["ate_rmse_m"]

    records = []
    for weight in args.weights:
        dataset_metrics = {}
        ratios = []
        for name, data in cached.items():
            new_scales = replay_selected_factors(
                data["on_log"], data["accepted"], anchor_weight=weight,
                incremental=True,
            )
            nodes = sorted(data["off_scales"])
            poses = rebuild_trajectory(
                data["off_poses"], nodes, data["off_scales"], new_scales
            )
            metrics, _ = evaluate(data["spec"], poses)
            dataset_metrics[name] = metrics["ate_rmse_m"]
            ratios.append(metrics["ate_rmse_m"] / baseline[name])
        records.append({
            "anchor_weight": float(weight),
            "geomean_rmse_ratio_vs_off": float(np.exp(np.mean(np.log(ratios)))),
            "mean_rmse_ratio_vs_off": float(np.mean(ratios)),
            "datasets": dataset_metrics,
        })

    records.sort(key=lambda row: (
        row["geomean_rmse_ratio_vs_off"], row["mean_rmse_ratio_vs_off"]
    ))
    best = records[0]
    payload = {
        "metric": "global Umeyama Sim(3) ATE RMSE",
        "gate": BEST_GATE.__dict__,
        "baseline": baseline,
        "best": best,
        "results_sorted": records,
    }
    (args.output_dir / "anchor_weight_sweep.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    with (args.output_dir / "anchor_weight_sweep.csv").open("w", newline="") as stream:
        names = list(cached)
        writer = csv.writer(stream)
        writer.writerow([
            "anchor_weight", "geomean_rmse_ratio_vs_off",
            "mean_rmse_ratio_vs_off", *[f"{name}_rmse_m" for name in names],
        ])
        for row in sorted(records, key=lambda item: item["anchor_weight"]):
            writer.writerow([
                row["anchor_weight"], row["geomean_rmse_ratio_vs_off"],
                row["mean_rmse_ratio_vs_off"],
                *[row["datasets"][name] for name in names],
            ])

    figure, axis = plt.subplots(figsize=(10, 6))
    ordered = sorted(records, key=lambda row: row["anchor_weight"])
    for name in cached:
        axis.plot(
            [row["anchor_weight"] for row in ordered],
            [row["datasets"][name] for row in ordered],
            "o-", label=name,
        )
    axis.axvline(best["anchor_weight"], color="black", linestyle="--", label="best aggregate")
    axis.set_xlabel("Anchor factor weight")
    axis.set_ylabel("ATE RMSE after global Sim(3) [m]")
    axis.set_title("Best Gate: Anchor weight sweep")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "anchor_weight_sweep.png", dpi=180)
    plt.close(figure)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
