# Scale Anchor SLAM

Anonymous code release for the double-blind submission on fixed-root scale
anchoring and sequence-consistent loop closure for VGGT-based monocular SLAM.

This repository contains the modified VGGT-SLAM 2.0 pipeline and the minimal
TUM evaluation utilities. Dataset files, model checkpoints, experiment logs,
and author-identifying material are intentionally excluded.

## Installation

The code requires Python 3.11, PyTorch, GTSAM with SL(4) support, VGGT, and
SALAD. The setup script installs the third-party components into this checkout:

```bash
conda create -n scale-anchor-slam python=3.11
conda activate scale-anchor-slam
bash setup.sh
```

Set `VGGT_SLAM_PYTHON` if a different Python interpreter is required. Model
checkpoints are downloaded separately and must not be committed to this
repository.

## Quick start

Run the system on a folder of RGB images:

```bash
./project_python.sh main.py \
  --image_folder /path/to/images \
  --submap_size 16 \
  --overlapping_window_size 1 \
  --min_disparity 50 \
  --max_loops 1 \
  --conf_threshold 25 \
  --log_results
```

The default configuration enables the fixed-root scale-anchor path. To run the
overlap-only baseline, add `--no_scale_anchor`. Dense-map export can be disabled
with `--skip_dense_log`.

## Reproducibility

The main entry point exposes the submap, keyframe, anchor, retrieval, and loop
closure options used by the experiments. The minimal TUM evaluation scripts
are under `evals/`.

No dataset or checkpoint is bundled. Download each dataset from its official
source and pass local paths through the command-line arguments or evaluation
scripts. Generated trajectories, point clouds, and logs should be written to a
separate output directory.

## Anonymity notice

This mirror is prepared for double-blind review. It contains no author names,
affiliations, email addresses, or local machine paths. The original repository
can be maintained separately after the review period.

## License

See `LICENSE` for the license of the released code and its upstream components.
