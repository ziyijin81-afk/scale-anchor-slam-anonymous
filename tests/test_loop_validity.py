import unittest
import numpy as np

from vggt_slam.loop_closure import verify_common_cross_submap_transform
from vggt_slam.graph import PoseGraph


def poses(centers, rotations=None):
    result = np.tile(np.eye(4), (len(centers), 1, 1))
    if rotations is not None:
        result[:, :3, :3] = rotations
    result[:, :3, 3] = -np.einsum('nij,nj->ni', result[:, :3, :3], centers)
    return result


def rotation(rng):
    u, _, vt = np.linalg.svd(rng.normal(size=(3, 3)))
    return u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt


def data(C, D, scale, translation, gauges, R=None, Q=None, H=None):
    C, D = np.array(C, float), np.array(D, float)
    R = np.eye(3) if R is None else R
    Q = np.tile(np.eye(3), (3, 1, 1)) if Q is None else Q
    H = np.tile(np.eye(3), (3, 1, 1)) if H is None else H
    pairs = np.tile(np.eye(4), (3, 2, 1, 1))
    pairs[:, 1, :3, :3] = H @ R @ Q.transpose(0, 2, 1)
    pairs[:, 1, :3, 3] = np.einsum(
        'nij,nj->ni', H, scale * (C @ R.T) + translation - D
    ) / np.asarray(gauges)[:, None]
    return poses(C, Q), poses(D, H), pairs


class LoopValidityTests(unittest.TestCase):
    def test_exact_positive_similarities_and_reverse_order(self):
        rng = np.random.default_rng(937)
        for trial in range(100):
            C, D = rng.normal(size=(2, 3, 3))
            if trial % 2:
                D = D[::-1].copy()
            R = rotation(rng)
            Q, H = [np.stack([rotation(rng) for _ in range(3)]) for _ in range(2)]
            s, t, gauges = rng.uniform(.3, 3), rng.normal(size=3), rng.uniform(.2, 4, 3)
            result = verify_common_cross_submap_transform(*data(C, D, s, t, gauges, R, Q, H))
            self.assertTrue(result['accepted'], (trial, result))
            self.assertEqual(result['linear_rank'], 7)
            np.testing.assert_allclose(result['cross_scale'], s, atol=1e-9)
            np.testing.assert_allclose(result['cross_translation'], t, atol=1e-9)
            np.testing.assert_allclose(result['cross_rotation'], R, atol=1e-9)
            np.testing.assert_allclose(result['pair_scales'], gauges, atol=1e-9)

    def test_negative_cross_scale_is_rejected_even_with_positive_gauges(self):
        args = data([[0, 0, 0], [1, 0, 0], [.2, .8, .1]],
                    [[-.5, .1, .7], [.8, -.4, .1], [.3, .6, 1.2]],
                    -2., np.array([4., 1., .5]), [.7, 1.3, 2.1])
        result = verify_common_cross_submap_transform(*args)
        self.assertTrue(result['positive_scales'])
        self.assertEqual(result['linear_rank'], 7)
        self.assertFalse(result['accepted'])
        self.assertIn('nonpositive_cross_scale', result['rejection_reasons'])

    def test_rank_deficient_positive_low_residual_fit_is_rejected(self):
        args = data([[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                    [[.5818304208696166, 0, 0], [1.260256244273397, 0, 0], [1.439850733216793, 0, 0]],
                    1.575730405916055, np.array([-2.5417698432712665, 0, 0]),
                    [2.7238681835979746, 1.243596659204803, 1.4257078088731308])
        result = verify_common_cross_submap_transform(*args)
        self.assertTrue(result['positive_scales'])
        self.assertTrue(result['positive_cross_scale'])
        self.assertLess(result['relative_position_rmse'], 1e-10)
        self.assertEqual(result['linear_rank'], 5)
        self.assertFalse(result['accepted'])
        self.assertIn('rank_deficient_geometry', result['rejection_reasons'])

    def test_nonfinite_input_is_rejected(self):
        args = list(data([[0, 0, 0], [1, 0, 0], [.2, .8, .1]],
                         [[-.5, .1, .7], [.8, -.4, .1], [.3, .6, 1.2]],
                         2., np.array([4., 1., .5]), [.7, 1.3, 2.1]))
        args[2][0, 1, 0, 3] = np.nan
        with self.assertRaises(ValueError):
            verify_common_cross_submap_transform(*args)


class LoopScaleUnitTests(unittest.TestCase):
    def make_graph(self):
        graph = PoseGraph()
        graph.add_homography(17, np.eye(4))
        h = np.eye(4)
        h[0, 3] = 6.
        graph.add_homography(34, h)
        graph.node_submap_anchors.update({17: 17, 34: 34})
        graph.submap_output_scales.update({17: 2., 34: 1.})
        return graph

    def add_loop(self, graph, scale=2.):
        measured = np.eye(4)
        measured[:3, 3] = [6., -2., 4.]
        graph.add_common_sim3_loop_factor(17, 34, np.eye(4), np.eye(4), measured,
                                         reference_scale_at_estimation=scale)
        return measured

    def test_same_unit_reproduces_measurement_and_refresh_is_not_cumulative(self):
        graph = self.make_graph()
        measured = self.add_loop(graph)
        record = graph.loop_common_sim3_measurements[0]
        stored = record['measurement_reference_local'].copy()
        np.testing.assert_allclose(graph._common_sim3_measurement_in_current_scale(record).matrix(), measured)
        for scale in [3., 3., .5, 2.]:
            graph.submap_output_scales[17] = scale
            updated = graph._common_sim3_measurement_in_current_scale(record).matrix()
            np.testing.assert_allclose(updated[:3, 3], measured[:3, 3] * scale / 2.)
            np.testing.assert_array_equal(record['measurement_reference_local'], stored)
        self.assertEqual(graph._common_sim3_measurement_in_current_scale(record).scale(), 1.)

    def test_scale_can_change_between_estimation_and_insertion(self):
        graph = self.make_graph()
        graph.submap_output_scales[17] = 3.
        measured = self.add_loop(graph, scale=2.)
        actual = graph._common_sim3_measurement_in_current_scale(graph.loop_common_sim3_measurements[0]).matrix()
        np.testing.assert_allclose(actual[:3, 3], measured[:3, 3] * 1.5)

    def test_invalid_estimation_and_replay_scales_fail(self):
        for scale in [0., -1., np.nan, np.inf]:
            with self.assertRaises(ValueError):
                self.add_loop(self.make_graph(), scale)
        graph = self.make_graph()
        self.add_loop(graph)
        for scale in [0., -1., np.nan, np.inf]:
            graph.submap_output_scales[17] = scale
            with self.assertRaises(ValueError):
                graph._common_sim3_measurement_in_current_scale(graph.loop_common_sim3_measurements[0])


if __name__ == '__main__':
    unittest.main()
