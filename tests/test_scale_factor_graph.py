import pathlib
import importlib
import sys
import types
import unittest

import numpy as np
import torch

try:
    from vggt_slam.graph import PoseGraph
except ImportError:
    PoseGraph = None

from vggt_slam.scale_solver import (
    ScaleFactorGraph,
    anchor_has_minimum_points,
    anchor_scale,
    confidence_selection_mask,
    estimate_scale_pairwise,
    frame_confidence_threshold,
    gtsam,
    same_frame_scale_consistency,
)
from vggt_slam.submap_batching import pad_submap_frames, should_process_submap
from vggt_slam.solver import Solver, _short_batch_camera_head_fp32


ROOT = pathlib.Path(__file__).resolve().parents[1]


class ScaleMeasurementTests(unittest.TestCase):
    def test_fixed_root_anchor_always_selects_submap_zero(self):
        class FakeSubmap:
            def __init__(self, submap_id, vectors):
                self.submap_id = submap_id
                self.vectors = torch.tensor(vectors, dtype=torch.float32)

            def get_id(self):
                return self.submap_id

            def get_all_retrieval_vectors(self):
                return self.vectors

        class FakeMap:
            def __init__(self, submaps):
                self.submaps = {submap.get_id(): submap for submap in submaps}
                self.non_lc_submap_ids = [submap.get_id() for submap in submaps]

            def get_submap(self, submap_id):
                return self.submaps[submap_id]

        solver = Solver.__new__(Solver)
        solver.map = FakeMap([
            FakeSubmap(0, [[0.10, 0.00]]),
            FakeSubmap(17, [[0.30, 0.00]]),
            FakeSubmap(34, [[0.00, 0.00]]),  # More similar history is ignored.
        ])
        current = FakeSubmap(51, [[0.11, 0.00]])

        selected = solver._select_scale_anchor_reference(current)

        self.assertEqual(selected["reference_submap_id"], 0)
        self.assertEqual(selected["reference_frame"], 0)
        self.assertEqual(selected["current_frame"], 0)

    def test_original_overlap_formula_is_radial_ratio_median(self):
        X = np.array([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 4.0]])
        Y = X * np.array([[2.0], [3.0], [100.0]])
        measured, _ = estimate_scale_pairwise(X, Y)
        self.assertEqual(measured, 3.0)

    def test_anchor_formula_direction(self):
        self.assertAlmostEqual(anchor_scale(2.0, 4.0, 5.0, 15.0), 1.5)

    def test_same_frame_global_scale_is_accepted_from_shared_trusted_pixels(self):
        yy, xx = np.indices((25, 25), dtype=float)
        joint = np.stack((0.01 * xx, 0.01 * yy, np.ones_like(xx) * 2.0), axis=-1)
        original = joint * 3.0
        confidence = np.ones((25, 25), dtype=float)
        result = same_frame_scale_consistency(
            original, confidence, 0.5,
            joint, confidence, 0.5,
            min_points=500,
        )
        self.assertTrue(result["accepted"])
        self.assertAlmostEqual(result["scale_ratio"], 3.0)
        self.assertAlmostEqual(result["relative_mad"], 0.0)

    def test_nonuniform_same_frame_geometry_is_rejected(self):
        yy, xx = np.indices((25, 25), dtype=float)
        joint = np.stack((0.01 * xx, 0.01 * yy, np.ones_like(xx) * 2.0), axis=-1)
        spatially_varying_scale = np.linspace(0.5, 1.5, 625).reshape(25, 25)
        original = joint * spatially_varying_scale[..., None]
        confidence = np.ones((25, 25), dtype=float)
        result = same_frame_scale_consistency(
            original, confidence, 0.5,
            joint, confidence, 0.5,
            min_points=500,
            max_relative_mad=0.10,
        )
        self.assertFalse(result["accepted"])
        self.assertTrue(result["reasons"])

    def test_same_frame_gate_uses_only_confident_pixels_from_both_predictions(self):
        points = np.ones((25, 25, 3), dtype=float)
        original_confidence = np.ones((25, 25), dtype=float)
        joint_confidence = np.zeros((25, 25), dtype=float)
        joint_confidence[:10] = 1.0
        result = same_frame_scale_consistency(
            points * 2.0, original_confidence, 0.5,
            points, joint_confidence, 0.5,
            min_points=500,
        )
        self.assertEqual(result["count"], 250)
        self.assertFalse(result["accepted"])
        self.assertIn("valid_points<500", result["reasons"])

    def test_anchor_scale_statistics_use_the_same_pixel_intersection(self):
        joint = np.zeros((2, 2, 3), dtype=float)
        joint[..., 2] = np.array([[1.0, 2.0], [100.0, 200.0]])
        original = joint * 2.0
        original_confidence = np.array([[1.0, 1.0], [1.0, 0.0]])
        joint_confidence = np.array([[1.0, 1.0], [0.0, 1.0]])
        result = same_frame_scale_consistency(
            original, original_confidence, 0.5,
            joint, joint_confidence, 0.5,
            min_points=1,
        )
        # Only the first row is trusted by both predictions.  Independent
        # masks would include a different far-depth pixel on each side.
        self.assertEqual(result["count"], 2)
        self.assertAlmostEqual(
            result["original_scale_stats"]["median"], 3.0
        )
        self.assertAlmostEqual(
            result["joint_scale_stats"]["median"], 1.5
        )
        self.assertAlmostEqual(result["scale_ratio"], 2.0)

    def test_same_frame_can_use_all_geometrically_valid_pixels(self):
        points = np.ones((25, 25, 3), dtype=float)
        no_confidence = np.zeros((25, 25), dtype=float)
        result = same_frame_scale_consistency(
            points * 2.0, no_confidence, 0.5,
            points, no_confidence, 0.5,
            min_points=500,
            use_confidence=False,
        )
        self.assertEqual(result["count"], 625)
        self.assertTrue(result["accepted"])
        self.assertAlmostEqual(result["scale_ratio"], 2.0)

    def test_anchor_confidence_thresholds_are_calibrated_per_frame(self):
        # A pooled threshold gives unequal retained fractions when the two
        # inference frames use different confidence ranges.  Per-frame
        # percentiles retain 75% in each despite the numeric offset.
        low = np.arange(100, dtype=float).reshape(10, 10)
        high = low + 1000.0
        pooled = np.percentile(np.stack((low, high)), 25) + 1e-6
        self.assertEqual(np.count_nonzero(low > pooled), 50)
        self.assertEqual(np.count_nonzero(high > pooled), 100)

        low_threshold = frame_confidence_threshold(low, 25)
        high_threshold = frame_confidence_threshold(high, 25)
        self.assertEqual(np.count_nonzero(low > low_threshold), 75)
        self.assertEqual(np.count_nonzero(high > high_threshold), 75)

        tied = np.ones((10, 10), dtype=float)
        tied_threshold = frame_confidence_threshold(tied, 25)
        self.assertEqual(np.count_nonzero(tied > tied_threshold), 0)
        tied_mask, degenerate = confidence_selection_mask(tied, tied_threshold)
        self.assertTrue(degenerate)
        self.assertEqual(np.count_nonzero(tied_mask), tied.size)

    def test_same_frame_intersection_uses_finite_pixels_for_constant_confidence(self):
        points = np.ones((25, 25, 3), dtype=float)
        constant_confidence = np.ones((25, 25), dtype=float)
        result = same_frame_scale_consistency(
            points * 2.0, constant_confidence, 1.0,
            points, constant_confidence, 1.0,
            min_points=500,
        )
        self.assertEqual(result["count"], 625)
        self.assertTrue(result["accepted"])
        self.assertTrue(result["original_confidence_degenerate"])
        self.assertTrue(result["joint_confidence_degenerate"])

    def test_anchor_minimum_point_count_is_a_hard_condition(self):
        self.assertTrue(anchor_has_minimum_points(500, 500, 500))
        self.assertFalse(anchor_has_minimum_points(499, 10000, 500))
        self.assertFalse(anchor_has_minimum_points(10000, 499, 500))

    def test_final_submap_is_padded_to_fixed_size(self):
        self.assertTrue(should_process_submap(33, 33, 1, 2, False))
        self.assertTrue(should_process_submap(7, 33, 1, 2, True))
        self.assertFalse(should_process_submap(1, 33, 1, 2, True))
        self.assertTrue(should_process_submap(1, 33, 1, 0, True))
        padded, count = pad_submap_frames(["overlap", "new"], 5)
        self.assertEqual(padded, ["overlap", "new", "new", "new", "new"])
        self.assertEqual(count, 3)
        full, count = pad_submap_frames(["a", "b"], 2)
        self.assertEqual(full, ["a", "b"])
        self.assertEqual(count, 0)

    def test_short_batch_camera_head_uses_fp32_and_restores_bfloat16(self):
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.camera_head = torch.nn.Linear(2, 2).to(torch.bfloat16)

        model = FakeModel()
        for frame_count in (2, 7):
            with _short_batch_camera_head_fp32(model, frame_count, "[test]"):
                self.assertEqual(next(model.camera_head.parameters()).dtype, torch.float32)
                self.assertEqual(len(model.camera_head._forward_pre_hooks), 1)
            self.assertEqual(next(model.camera_head.parameters()).dtype, torch.bfloat16)
            self.assertEqual(len(model.camera_head._forward_pre_hooks), 0)

        with _short_batch_camera_head_fp32(model, 17, "[test]"):
            self.assertEqual(next(model.camera_head.parameters()).dtype, torch.bfloat16)


class ScaleFactorGraphTests(unittest.TestCase):
    def make_mesh(self):
        graph = ScaleFactorGraph()
        graph.update_overlap(0, 1, 2.0, 0.02)
        graph.update_anchor(1, 2.1, 0.10)
        graph.update_overlap(1, 2, 0.5, 0.02)
        graph.update_anchor(2, 1.05, 0.10)
        graph.update_overlap(2, 3, 1.2, 0.02)
        graph.update_anchor(3, 1.25, 0.10)
        return graph

    def test_root_fixed_and_all_scales_positive(self):
        graph = self.make_mesh()
        graph.optimize()
        self.assertEqual(graph.nodes[0], 1.0)
        self.assertTrue(all(value > 0 for value in graph.nodes.values()))

    def test_overlap_and_anchor_residual_definitions(self):
        graph = ScaleFactorGraph()
        graph.update_overlap(0, 1, 2.0, 1e-3)
        graph.update_anchor(1, 2.0, 1e-3)
        graph.optimize()
        self.assertAlmostEqual(graph.nodes[1] - graph.nodes[0] * 2.0, 0.0, places=7)
        overlap = graph.factors[("overlap", 0, 1)]
        self.assertAlmostEqual(
            graph._raw_residual(overlap, {0: 1.0, 1: 4.0}),
            np.log(2.0),
            places=7,
        )

    def test_semantic_anchor_can_connect_two_non_root_submaps(self):
        graph = ScaleFactorGraph(use_isam2=False)
        graph.update_overlap(0, 1, 2.0, 1e-3)
        graph.update_overlap(1, 2, 3.0, 1e-3)
        graph.update_anchor(
            2,
            3.0,
            1e-3,
            reference_id=1,
        )
        graph.optimize()
        self.assertAlmostEqual(
            graph.nodes[2] - graph.nodes[1] * 3.0,
            0.0,
            places=6,
        )
        self.assertIn(("anchor", 1, 2), set(graph.topology()))

    def test_three_nodes_form_mesh_and_historical_paths(self):
        graph = self.make_mesh()
        topology = set(graph.topology())
        self.assertTrue({
            ("overlap", 0, 1), ("overlap", 1, 2), ("overlap", 2, 3),
            ("anchor", 0, 1), ("anchor", 0, 2), ("anchor", 0, 3),
        }.issubset(topology))
        # Node 3 is constrained directly by g03 and indirectly by every older
        # anchor followed by the remaining overlap chain.
        self.assertIn(("anchor", 0, 1), topology)
        self.assertIn(("overlap", 1, 2), topology)
        self.assertIn(("overlap", 2, 3), topology)

    def test_live_snapshot_revision_values_and_residuals(self):
        graph = self.make_mesh()
        graph.optimize()
        snapshot = graph.snapshot()
        self.assertEqual(snapshot["revision"], 1)
        self.assertEqual(snapshot["nodes"][0], 1.0)
        self.assertEqual(len(snapshot["factors"]), 6)
        self.assertTrue(all("residual" in factor for factor in snapshot["factors"]))
        self.assertIn(snapshot["backend"], ("gtsam-isam2", "scipy-batch-fallback"))

    def test_isam2_backend_indices_are_saved_and_replaced(self):
        class FakeISAM2Backend:
            def __init__(self):
                self.next_index = 40
                self.removals = []

            def update(self, records, remove_indices, nodes):
                self.removals.append(list(remove_indices))
                indices = list(range(self.next_index, self.next_index + len(records)))
                self.next_index += len(records)
                optimized = {
                    record.submap_j: nodes[record.submap_i] * record.measurement
                    for record in records
                }
                return optimized, indices, {
                    "variables_relinearized": len(records),
                    "variables_reeliminated": len(records),
                    "cliques": len(optimized),
                }

        graph = ScaleFactorGraph(use_isam2=False)
        backend = FakeISAM2Backend()
        graph.isam2_backend = backend
        graph.backend_name = "gtsam-isam2"
        graph.update_overlap(0, 1, 1.2, 0.1)
        graph.update_anchor(1, 1.1, 0.2)
        graph.optimize()
        anchor = graph.factors[("anchor", 0, 1)]
        old_backend_index = anchor.backend_factor_index
        self.assertIsNotNone(old_backend_index)
        self.assertEqual(backend.removals[-1], [])

        graph.update_anchor(1, 1.3, 0.2)
        graph.optimize()
        new_anchor = graph.factors[("anchor", 0, 1)]
        self.assertEqual(backend.removals[-1], [old_backend_index])
        self.assertNotEqual(new_anchor.backend_factor_index, old_backend_index)
        self.assertEqual(new_anchor.measurement_version, 2)
        self.assertEqual(graph.snapshot()["backend"], "gtsam-isam2")

    def test_isam2_source_fixes_s0_and_uses_custom_scalar_factors(self):
        source = (ROOT / "vggt_slam" / "scale_solver.py").read_text()
        backend = source.split("class GtsamISAM2ScaleBackend", 1)[1].split(
            "_FALLBACK_WARNING_EMITTED", 1
        )[0]
        self.assertIn('gtsam.CustomFactor', backend)
        self.assertIn('values.atVector', backend)
        self.assertIn('if record.submap_i == 0', backend)
        self.assertNotIn('new_values.insert(self.key(0)', backend)
        self.assertIn('remove_indices', backend)

    @unittest.skipIf(gtsam is None, "GTSAM is not installed in the base test environment")
    def test_real_gtsam_isam2_increment_and_factor_replacement(self):
        graph = ScaleFactorGraph()
        self.assertEqual(graph.backend_name, "gtsam-isam2")
        graph.update_overlap(0, 1, 1.10, 0.02)
        graph.update_anchor(1, 1.05, 0.10)
        graph.optimize()
        old_s1 = graph.nodes[1]
        graph.update_overlap(1, 2, 1.20, 0.02)
        graph.update_anchor(2, 1.25, 0.10)
        graph.optimize()
        self.assertNotEqual(graph.nodes[1], old_s1)
        old_index = graph.factors[("anchor", 0, 2)].backend_factor_index
        graph.update_anchor(2, 1.30, 0.08)
        graph.optimize()
        new_factor = graph.factors[("anchor", 0, 2)]
        self.assertNotEqual(new_factor.backend_factor_index, old_index)
        self.assertEqual(new_factor.measurement_version, 2)
        self.assertEqual(graph.nodes[0], 1.0)
        self.assertTrue(all(value > 0 for value in graph.nodes.values()))

    def test_factor_update_replaces_old_measurement(self):
        graph = ScaleFactorGraph()
        old, _ = graph.update_anchor(2, 1.1, 0.1, timestamp=1.0)
        new, replaced = graph.update_anchor(2, 1.2, 0.2, timestamp=2.0)
        self.assertEqual(replaced, old.factor_index)
        self.assertEqual(new.measurement_version, 2)
        self.assertEqual(len(graph.factors), 1)
        self.assertEqual(next(iter(graph.factors.values())).measurement, 1.2)

        old_overlap, _ = graph.update_overlap(1, 2, 0.9, 0.1)
        new_overlap, replaced_overlap = graph.update_overlap(1, 2, 1.0, 0.1)
        self.assertEqual(replaced_overlap, old_overlap.factor_index)
        self.assertEqual(new_overlap.measurement_version, 2)
        self.assertEqual(len(graph.factors), 2)

    def test_incremental_application_cannot_repeat_scale(self):
        graph = ScaleFactorGraph()
        graph.update_anchor(1, 2.0, 1e-3)
        graph.optimize()
        self.assertAlmostEqual(graph.pending_corrections()[1], 2.0, places=6)
        graph.mark_applied(1)
        self.assertNotIn(1, graph.pending_corrections())
        graph.update_anchor(1, 3.0, 1e-3)
        graph.optimize()
        self.assertAlmostEqual(graph.pending_corrections()[1], 1.5, places=6)

    def test_new_measurement_can_correct_historical_nodes(self):
        graph = self.make_mesh()
        graph.optimize()
        for node_id in graph.nodes:
            graph.mark_applied(node_id)
        graph.update_anchor(3, 1.8, 0.02)
        graph.optimize()
        changed = graph.pending_corrections(tolerance=1e-7)
        self.assertIn(3, changed)
        self.assertTrue(any(node_id in changed for node_id in (1, 2)))

    def test_joint_vggt_source_has_one_two_frame_call(self):
        source = (ROOT / "vggt_slam" / "solver.py").read_text()
        body = source.split("def _compute_joint_anchor", 1)[1].split("def ", 1)[0]
        self.assertEqual(body.count("joint_prediction = model(joint_images)"), 1)
        self.assertIn("torch.stack", body)
        self.assertNotIn("model(root_submap.get_frame", body)
        self.assertNotIn("model(current_submap.get_frame", body)

    def test_scale_anchor_switch_gates_joint_inference_and_factor(self):
        source = (ROOT / "vggt_slam" / "solver.py").read_text()
        anchor_body = source.split("def _compute_joint_anchor", 1)[1].split("def ", 1)[0]
        self.assertIn("enable_scale_anchor: bool = True", source)
        self.assertIn(
            "self.scale_graph = ScaleFactorGraph(use_isam2=True)",
            source,
        )
        self.assertIn("if self.enable_scale_anchor and self.map.non_lc_submap_ids", source)
        self.assertIn('anchor.get("accepted", True)', source)
        self.assertIn("skipped_factor", source)
        self.assertIn("anchor_measurement_invalid", source)
        self.assertIn(
            "root_submap.local_pointclouds[root_index]",
            anchor_body,
        )
        self.assertNotIn("root_submap.pointclouds[root_index]", anchor_body)
        self.assertNotIn("world_to_camera=root_submap.poses", anchor_body)
        self.assertNotIn("root_homogeneous", anchor_body)
        for entrypoint in ("main.py", "main_realtime.py"):
            entrypoint_source = (ROOT / entrypoint).read_text()
            self.assertIn('"--no_scale_anchor", "--no-scale-anchor"', entrypoint_source)
            self.assertIn("enable_scale_anchor=args.scale_anchor", entrypoint_source)

    def test_scale_application_does_not_touch_rotation_or_intrinsics(self):
        source = (ROOT / "vggt_slam" / "submap.py").read_text()
        body = source.split("def rebuild_from_local_scale", 1)[1].split("def ", 1)[0]
        self.assertIn("self.local_pointclouds * new_scale", body)
        self.assertIn("self.local_depths * new_scale", body)
        self.assertIn("self.local_poses[:, :3, 3] * new_scale", body)
        self.assertNotIn("proj_mats", body)
        self.assertNotIn("poses[:, :3, :3]", body)

    def test_absolute_geometry_rebuild_is_repeatable_and_preserves_rotation(self):
        submap_module = importlib.import_module("vggt_slam.submap")
        submap = submap_module.Submap(0)
        poses = np.repeat(np.eye(4)[None], 2, axis=0)
        poses[1, :3, :3] = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        poses[1, :3, 3] = (1.0, 2.0, 3.0)
        points = np.ones((2, 1, 1, 3), dtype=float)
        depths = np.ones((2, 1, 1, 1), dtype=float)
        submap.add_all_poses(poses)
        submap.local_pointclouds = points.copy()
        submap.pointclouds = points.copy()
        submap.local_depths = depths.copy()
        submap.depths = depths.copy()

        submap.rebuild_from_local_scale(2.0)
        first_points = submap.pointclouds.copy()
        first_poses = submap.poses.copy()
        submap.rebuild_from_local_scale(2.0)

        np.testing.assert_allclose(submap.pointclouds, first_points)
        np.testing.assert_allclose(submap.poses, first_poses)
        np.testing.assert_allclose(submap.local_pointclouds, points)
        np.testing.assert_allclose(submap.local_poses, poses)
        np.testing.assert_allclose(submap.poses[:, :3, :3], poses[:, :3, :3])
        np.testing.assert_allclose(submap.poses[1, :3, 3], (2.0, 4.0, 6.0))

    def test_degenerate_confidence_keeps_points_with_strict_mask(self):
        submap_module = importlib.import_module("vggt_slam.submap")
        submap = submap_module.Submap(0)
        points = np.ones((1, 2, 3, 3), dtype=np.float32)
        colors = np.ones_like(points)
        confidence = np.ones((1, 2, 3), dtype=np.float32)
        submap.add_all_points(
            points,
            colors,
            confidence,
            conf_threshold_percentile=25.0,
            intrinsics_inv=np.eye(4, dtype=np.float32)[None],
        )

        self.assertLess(submap.conf_threshold, 1.0)
        self.assertEqual(submap.filter_data_by_confidence(points).shape[0], 6)

    @unittest.skipIf(PoseGraph is None, "GTSAM pose graph is not installed")
    def test_historical_pose_scale_keeps_anchor_and_propagates_to_descendants(self):
        class FakeSubmap:
            def __init__(self, submap_id, pose_count):
                self.submap_id = submap_id
                self.poses = np.repeat(np.eye(4)[None], pose_count, axis=0)

            def get_id(self):
                return self.submap_id

            @staticmethod
            def get_lc_status():
                return False

        def translated(x, y=0.0, z=0.0):
            matrix = np.eye(4)
            matrix[:3, 3] = (x, y, z)
            return matrix

        graph = PoseGraph()
        root = FakeSubmap(0, 2)
        current = FakeSubmap(2, 2)
        descendant = FakeSubmap(4, 2)

        # Raw chain: root [0,10], current [10,12], descendant [12,15].
        for node_id, x in enumerate((0.0, 10.0, 10.0, 12.0, 12.0, 15.0)):
            matrix = translated(x)
            if node_id == 5:
                matrix[:3, :3] = np.array([
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ])
            graph.add_homography(node_id, matrix)
        graph.register_scale_submap(root)
        graph.register_scale_submap(current, parent_overlap_node_id=1)
        graph.register_scale_submap(descendant, parent_overlap_node_id=3)

        graph.rebuild_scale_adjusted_homographies(
            [root, current, descendant], {0: 1.0, 2: 2.0, 4: 1.0}
        )
        # The current submap's first/overlap node stays at x=10. Its internal
        # motion doubles from 2 to 4, moving the last node to x=14.
        self.assertAlmostEqual(graph.get_homography(2)[0, 3], 10.0)
        self.assertAlmostEqual(graph.get_homography(3)[0, 3], 14.0)
        # The descendant inherits the +2 anchor displacement without changing
        # its own local length: [12,15] becomes [14,17].
        self.assertAlmostEqual(graph.get_homography(4)[0, 3], 14.0)
        self.assertAlmostEqual(graph.get_homography(5)[0, 3], 17.0)

        # A descendant scale update is also around its propagated first node.
        graph.rebuild_scale_adjusted_homographies(
            [root, current, descendant], {0: 1.0, 2: 2.0, 4: 0.5}
        )
        self.assertAlmostEqual(graph.get_homography(4)[0, 3], 14.0)
        self.assertAlmostEqual(graph.get_homography(5)[0, 3], 15.5)
        np.testing.assert_allclose(
            graph.get_homography(5)[:3, :3],
            graph.get_base_homography(5)[:3, :3],
        )

    @unittest.skipIf(PoseGraph is None, "GTSAM pose graph is not installed")
    def test_full_replay_uses_absolute_scales_without_accumulation(self):
        class FakeSubmap:
            def __init__(self, submap_id):
                self.submap_id = submap_id
                self.poses = np.repeat(np.eye(4)[None], 2, axis=0)

            def get_id(self):
                return self.submap_id

            @staticmethod
            def get_lc_status():
                return False

        def translated(x):
            matrix = np.eye(4)
            matrix[0, 3] = x
            return matrix

        graph = PoseGraph()
        root, child = FakeSubmap(0), FakeSubmap(2)
        for node_id, x in enumerate((0.0, 2.0, 2.0, 5.0)):
            graph.add_homography(node_id, translated(x))
        graph.register_scale_submap(root)
        graph.register_scale_submap(child, parent_overlap_node_id=1)

        graph.rebuild_scale_adjusted_homographies(
            [root, child], {0: 1.0, 2: 2.0}
        )
        first_result = graph.get_homography(3).copy()
        graph.rebuild_scale_adjusted_homographies(
            [root, child], {0: 1.0, 2: 2.0}
        )
        np.testing.assert_allclose(graph.get_homography(3), first_result)
        self.assertAlmostEqual(first_result[0, 3], 8.0)

    def test_overlap_measurement_reads_immutable_local_points(self):
        source = (ROOT / "vggt_slam" / "submap.py").read_text()
        body = source.split("def get_frame_pointcloud", 1)[1].split("def ", 1)[0]
        self.assertIn("self.local_pointclouds[pose_index]", body)

    def test_backend_publish_redraws_all_scale_changed_submaps(self):
        source = (ROOT / "vggt_slam" / "solver.py").read_text()
        body = source.split("def finalize_backend_update", 1)[1].split("def ", 1)[0]
        self.assertIn("self.scale_dirty_submap_ids", body)
        self.assertIn("self.viewer.visualize_scale_graph", body)
        self.assertIn("self.set_submap_point_cloud(submap)", body)
        self.assertIn("self.set_submap_poses(submap)", body)

    def test_overlap_and_anchor_use_requested_fixed_weights(self):
        source = (ROOT / "vggt_slam" / "solver.py").read_text()
        self.assertIn("OVERLAP_FACTOR_WEIGHT = 1.0", source)
        self.assertIn("ANCHOR_FACTOR_WEIGHT = 1.0", source)
        self.assertIn("OVERLAP_FACTOR_SIGMA = 1.0 / np.sqrt(OVERLAP_FACTOR_WEIGHT)", source)
        self.assertIn("ANCHOR_FACTOR_SIGMA = 1.0 / np.sqrt(ANCHOR_FACTOR_WEIGHT)", source)
        self.assertIn("overlap_sigma = float(OVERLAP_FACTOR_SIGMA)", source)
        self.assertIn("scale_anchor_weight: float = ANCHOR_FACTOR_WEIGHT", source)
        self.assertIn("sigma = float(self.scale_anchor_sigma)", source)
        self.assertIn('anchor["measurement"]', source)
        self.assertIn('anchor["sigma"]', source)
        self.assertIn("reference_id=reference_id", source)

    def test_anchor_quality_gate_removed_but_estimator_support_retained(self):
        solver_source = (ROOT / "vggt_slam" / "solver.py").read_text()
        entrypoint_source = (ROOT / "main.py").read_text()
        self.assertIn("ANCHOR_CONSISTENCY_MIN_POINTS = 500", solver_source)
        self.assertIn("ANCHOR_FALLBACK_MIN_POINTS = 100", solver_source)
        for constant in (
            "ANCHOR_CONSISTENCY_MAX_RELATIVE_MAD",
            "ANCHOR_CONSISTENCY_MAX_OUTLIER_RATIO",
            "ANCHOR_CONSISTENCY_MAX_LOG_DISAGREEMENT",
        ):
            self.assertNotIn(constant, solver_source)
        for option in (
            "--scale-anchor-min-points",
            "--scale-anchor-fallback-min-points",
        ):
            self.assertIn(option, entrypoint_source)
        for option in (
            '"--scale-anchor-gate"',
            "--scale-anchor-max-relative-mad",
            "--scale-anchor-max-outlier-ratio",
            "--scale-anchor-max-log-disagreement",
        ):
            self.assertNotIn(option, entrypoint_source)
            self.assertNotIn(option, (ROOT / "main_realtime.py").read_text())

    def test_scale_graph_viewer_updates_3d_graph_and_numeric_panel(self):
        fake_viser = types.ModuleType("viser")
        fake_viser.FrameHandle = object
        fake_viser.CameraFrustumHandle = object
        fake_transforms = types.ModuleType("viser.transforms")
        fake_viser.transforms = fake_transforms
        old_viser = sys.modules.get("viser")
        old_transforms = sys.modules.get("viser.transforms")
        sys.modules["viser"] = fake_viser
        sys.modules["viser.transforms"] = fake_transforms
        try:
            sys.modules.pop("vggt_slam.viewer", None)
            viewer_module = importlib.import_module("vggt_slam.viewer")

            class Handle:
                visible = True

                def remove(self):
                    pass

            class Scene:
                def __init__(self):
                    self.calls = []

                def add_icosphere(self, name, **kwargs):
                    self.calls.append(("node", name))
                    return Handle()

                def add_label(self, name, **kwargs):
                    self.calls.append(("label", name))
                    return Handle()

                def add_line_segments(self, name, **kwargs):
                    self.calls.append(("edge", name))
                    return Handle()

            viewer = viewer_module.Viewer.__new__(viewer_module.Viewer)
            viewer.server = types.SimpleNamespace(scene=Scene())
            viewer.gui_show_scale_graph = types.SimpleNamespace(value=True)
            viewer.gui_scale_graph_status = types.SimpleNamespace(content="")
            viewer.scale_graph_handles = []
            graph = self.make_mesh()
            graph.optimize()
            viewer.visualize_scale_graph(graph.snapshot(), {1: 1.01})
            call_types = [kind for kind, _ in viewer.server.scene.calls]
            self.assertEqual(call_types.count("node"), 4)
            self.assertEqual(call_types.count("edge"), 6)
            self.assertIn("Revision:", viewer.gui_scale_graph_status.content)
            self.assertIn("S1", viewer.gui_scale_graph_status.content)
            self.assertIn("overlap", viewer.gui_scale_graph_status.content)
            self.assertIn("anchor", viewer.gui_scale_graph_status.content)
        finally:
            sys.modules.pop("vggt_slam.viewer", None)
            if old_viser is None:
                sys.modules.pop("viser", None)
            else:
                sys.modules["viser"] = old_viser
            if old_transforms is None:
                sys.modules.pop("viser.transforms", None)
            else:
                sys.modules["viser.transforms"] = old_transforms


if __name__ == "__main__":
    unittest.main()
