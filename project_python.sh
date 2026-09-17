#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${VGGT_SLAM_PYTHON:-}" ]]; then
    PYTHON_BIN="${VGGT_SLAM_PYTHON}"
elif [[ -x /opt/venvs/metric-vggt-slam2/bin/python ]]; then
    PYTHON_BIN=/opt/venvs/metric-vggt-slam2/bin/python
else
    PYTHON_BIN=/opt/conda/envs/vggt-slam/bin/python
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "VGGT-SLAM Python is not executable: ${PYTHON_BIN}" >&2
    echo "Set VGGT_SLAM_PYTHON to an interpreter containing torch and gtsam." >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" "$@"
