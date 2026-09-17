"""VGGT-SLAM package bootstrap for this checkout."""

# Importing any vggt_slam submodule first executes this file. Prepending the
# bundled sources here prevents notebooks/tests from silently importing a
# stale editable VGGT installation.
from vggt_slam.project_paths import prefer_bundled_third_party


prefer_bundled_third_party()
