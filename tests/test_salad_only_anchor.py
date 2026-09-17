"""SALAD-only Anchor regression tests; no model weights, viewer or GPU needed."""
from contextlib import nullcontext
import inspect
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import vggt_slam.solver as solver_module
from vggt_slam.solver import Solver
from vggt_slam.scale_solver import ScaleFactorGraph, camera_points_from_depth


class FakeSubmap:
    def __init__(self, submap_id, vectors):
        self.submap_id = submap_id
        self.vectors = torch.tensor(vectors, dtype=torch.float32)

    def get_id(self):
        return self.submap_id

    def get_all_retrieval_vectors(self):
        return self.vectors

    def get_frame_at_index(self, index):
        return torch.zeros((3, 24, 24))

    def get_img_names_at_index(self, index):
        return f"frame_{self.submap_id}_{index}.png"


def make_solver():
    solver = Solver.__new__(Solver)
    solver.init_conf_threshold = 25
    solver.scale_anchor_sigma = 1.0
    solver.scale_anchor_min_points = 500
    solver.scale_anchor_fallback_min_points = 100
    solver.scale_anchor_salad_max_distance = 0.9
    solver.scale_anchor_salad_weight_sigma = 0.5
    solver.scale_anchor_min_ordinal_span = 12
    solver.scale_anchor_target_ordinal_span = 12
    solver.scale_anchor_submap_interval = 1
    solver.scale_anchor_reference_mode = "salad-far"
    solver.enable_scale_anchor_conf_filter = True
    solver.enable_scale_anchor_conf_fallback = True
    solver.vggt_timer = nullcontext()
    return solver


def set_map(solver, submaps):
    lookup = {s.get_id(): s for s in submaps}
    solver.map = SimpleNamespace(non_lc_submap_ids=list(lookup), get_submap=lookup.__getitem__)


class SaladOnlyAnchorTests(unittest.TestCase):
    def test_salad_far_selects_oldest_passing_history(self):
        solver = make_solver()
        history = [FakeSubmap(i * 8, [[0.4 if i == 0 else 0.1, 0.0]])
                   for i in range(12)]
        set_map(solver, history)
        selected = solver._select_scale_anchor_reference(
            FakeSubmap(96, [[0.0, 0.0]]))
        self.assertEqual(selected["reference_submap_id"], 0)

    def test_salad_far_requires_minimum_ordinal_span(self):
        solver = make_solver()
        set_map(solver, [FakeSubmap(i * 8, [[0.0, 0.0]]) for i in range(11)])
        self.assertIsNone(solver._select_scale_anchor_reference(
            FakeSubmap(88, [[0.0, 0.0]])))

    def test_salad_far_uses_closest_frame_pair(self):
        solver = make_solver()
        history = [FakeSubmap(i * 8, [[0.8, 0.0], [0.2, 0.0]])
                   for i in range(12)]
        set_map(solver, history)
        selected = solver._select_scale_anchor_reference(
            FakeSubmap(96, [[0.7, 0.0], [0.21, 0.0]]))
        self.assertEqual(selected["reference_submap_id"], 0)
        self.assertEqual(selected["reference_frame"], 1)
        self.assertEqual(selected["current_frame"], 1)

    def test_salad_far_rejects_distance_above_cutoff(self):
        solver = make_solver()
        set_map(solver, [FakeSubmap(i * 8, [[-1.0, 0.0]]) for i in range(12)])
        self.assertIsNone(solver._select_scale_anchor_reference(
            FakeSubmap(96, [[1.0, 0.0]])))

    def test_fixed_root_uses_root_without_distance_cutoff(self):
        solver = make_solver()
        solver.scale_anchor_reference_mode = "fixed-root"
        history = [FakeSubmap(i * 8, [[-2.0 if i == 0 else 0.0, 0.0]])
                   for i in range(12)]
        set_map(solver, history)
        selected = solver._select_scale_anchor_reference(
            FakeSubmap(96, [[2.0, 0.0]]))
        self.assertEqual(selected["reference_submap_id"], 0)
        self.assertEqual(selected["weight_scale"], 1.0)
        self.assertGreater(selected["salad_distance"], 0.9)

    def test_anchor_submap_interval_is_relative_to_first_eligible_submap(self):
        solver = make_solver()
        solver.scale_anchor_reference_mode = "fixed-root"
        solver.scale_anchor_min_ordinal_span = 6
        solver.scale_anchor_submap_interval = 5
        set_map(solver, [FakeSubmap(i * 8, [[0.0, 0.0]]) for i in range(6)])
        self.assertIsNotNone(solver._select_scale_anchor_reference(
            FakeSubmap(48, [[0.0, 0.0]])))
        set_map(solver, [FakeSubmap(i * 8, [[0.0, 0.0]]) for i in range(7)])
        self.assertIsNone(solver._select_scale_anchor_reference(
            FakeSubmap(56, [[0.0, 0.0]])))

    def test_nonfinite_descriptor_is_not_an_anchor(self):
        solver = make_solver()
        set_map(solver, [FakeSubmap(i * 8, [[float("nan"), 0.0]])
                         for i in range(12)])
        self.assertIsNone(solver._select_scale_anchor_reference(
            FakeSubmap(96, [[1.0, 0.0]])))

    def compute_anchor(self, points=576, sigma=1.0):
        solver = make_solver()
        solver.scale_anchor_sigma = sigma
        root = FakeSubmap(0, [[0.0, 0.0], [1.0, 0.0]])
        current = FakeSubmap(17, [[0.01, 0.0], [1.0, 0.0]])
        set_map(solver, [root])
        # A broad ratio distribution intentionally fails the former MAD gate.
        root_depth = np.repeat([0.2, 1.0, 8.0], 192).reshape(24, 24)
        if points < root_depth.size:
            root_depth.flat[points:] = np.nan
        intrinsic = np.array([[100.0, 0, 12.0], [0, 100.0, 12.0], [0, 0, 1.0]])
        root.local_pointclouds = [camera_points_from_depth(root_depth, intrinsic)] * 2
        root.conf_masks = np.ones((2, 24, 24))
        predictions = {"depth": np.full((2, 24, 24), 2.0),
                       "depth_conf": np.ones((2, 24, 24)),
                       "intrinsic": np.stack([intrinsic, intrinsic])}
        joint = {"pose_enc": None, "depth": np.ones((2, 24, 24)),
                 "depth_conf": np.ones((2, 24, 24))}
        model = unittest.mock.Mock(return_value=joint)
        # Fix the pair to frame 0 so the bad geometry above is exercised.
        reference = {"reference_submap_id": 0, "reference_frame": 0,
                     "current_frame": 0, "salad_distance": 0.01, "weight_scale": 0.1}
        with patch.object(solver, "_select_scale_anchor_reference", return_value=reference), \
             patch.object(solver_module, "_short_batch_camera_head_fp32", return_value=nullcontext()), \
             patch.object(solver_module, "pose_encoding_to_extri_intri",
                          return_value=(None, np.stack([intrinsic, intrinsic]))):
            result = solver._compute_joint_anchor(model, predictions, current)
        self.assertEqual(model.call_count, 1)
        return result

    def test_large_mad_is_diagnostic_only_and_factor_enters_scale_graph(self):
        result = self.compute_anchor()
        self.assertGreater(result["root_consistency"]["relative_mad"], 0.20)
        self.assertTrue(result["accepted"])
        self.assertFalse(result["gate_enabled"])
        self.assertEqual(result["gate_reasons"], [])
        self.assertAlmostEqual(result["sigma"], np.sqrt(10.0))
        expected = result["A_0"] * result["a_k"] / (result["a_0"] * result["A_k"])
        self.assertEqual(result["measurement"], expected)
        graph = ScaleFactorGraph(use_isam2=True)
        graph.ensure_node(0)
        graph.ensure_node(17)
        graph.update_anchor(17, result["measurement"], result["sigma"])
        graph.optimize()
        self.assertAlmostEqual(graph.pending_corrections()[17], expected, places=5)

    def test_quality_diagnostics_cannot_reject_or_reweight(self):
        original = solver_module.same_frame_scale_consistency
        def bad_diagnostics(*args, **kwargs):
            result = original(*args, **kwargs)
            # Force every old quality signal to fail, preserving the four medians.
            result.update(accepted=False, reasons=["forced_bad_quality"],
                          relative_mad=99.0, outlier_ratio=0.99, scale_ratio=1.0)
            return result
        baseline = self.compute_anchor()
        with patch.object(solver_module, "same_frame_scale_consistency", side_effect=bad_diagnostics):
            changed = self.compute_anchor()
        self.assertTrue(changed["accepted"])
        self.assertGreater(changed["log_disagreement"], 0.15)
        self.assertEqual(changed["measurement"], baseline["measurement"])
        self.assertEqual(changed["sigma"], baseline["sigma"])

    def test_insufficient_pixels_still_cannot_supply_scale(self):
        result = self.compute_anchor(points=10)
        self.assertFalse(result["accepted"])
        self.assertFalse(result["hard_min_points_passed"])

    def test_nonfinite_sigma_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            self.compute_anchor(sigma=float("nan"))

    def test_nonfinite_or_nonpositive_final_scale_is_rejected(self):
        for scale in (float("inf"), float("nan"), 0.0, -1.0):
            with self.subTest(scale=scale), \
                 patch.object(solver_module, "anchor_scale", return_value=scale), \
                 self.assertRaisesRegex(ValueError, "finite and positive"):
                self.compute_anchor()

    def test_quality_gate_options_removed_from_solver(self):
        parameters = inspect.signature(Solver).parameters
        self.assertIn("scale_anchor_salad_max_distance", parameters)
        self.assertIn("scale_anchor_min_ordinal_span", parameters)
        for name in ("enable_scale_anchor_gate", "scale_anchor_max_relative_mad",
                     "scale_anchor_max_outlier_ratio", "scale_anchor_max_log_disagreement",
                     "scale_anchor_target_span"):
            self.assertNotIn(name, parameters)

    def test_geometric_reference_selector_is_removed(self):
        import vggt_slam.scale_solver as scale_module
        self.assertFalse(hasattr(scale_module, "select_reference_frame"))


if __name__ == "__main__":
    unittest.main()
