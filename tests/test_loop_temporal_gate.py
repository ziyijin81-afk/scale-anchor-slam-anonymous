import numpy as np
import torch

from vggt_slam.loop_closure import (
    ImageRetrieval,
    centered_window_indices,
    select_sequence_pair_indices,
    verify_common_cross_submap_transform,
)
from vggt_slam.map import GraphMap
from vggt_slam.solver import _pose_sequence_numpy


class _Submap:
    def __init__(self, submap_id, descriptor):
        self._submap_id = submap_id
        self._descriptor = torch.tensor([descriptor], dtype=torch.float32)

    def get_id(self):
        return self._submap_id

    def get_lc_status(self):
        return False

    def get_all_retrieval_vectors(self):
        return self._descriptor


def test_gap_four_excludes_three_most_recent_ordinary_submaps():
    graph_map = GraphMap()
    graph_map.add_submap(_Submap(0, 0.30))
    graph_map.add_submap(_Submap(10, 0.00))
    graph_map.add_submap(_Submap(20, 0.10))
    graph_map.add_submap(_Submap(30, 0.20))

    score, submap_id, frame_id = graph_map.retrieve_best_score_frame(
        torch.tensor([0.0]),
        current_submap_id=40,
        min_submap_gap=4,
    )

    assert submap_id == 0
    assert frame_id == 0
    assert abs(score - 0.30) < 1e-6


def test_legacy_call_still_excludes_only_latest_submap():
    graph_map = GraphMap()
    graph_map.add_submap(_Submap(0, 0.30))
    graph_map.add_submap(_Submap(10, 0.00))
    graph_map.add_submap(_Submap(20, 0.10))

    _, submap_id, _ = graph_map.retrieve_best_score_frame(
        torch.tensor([0.0]),
        current_submap_id=30,
        ignore_last_submap=True,
    )

    assert submap_id == 10


def test_loop_retrieval_uses_gap_four_by_default():
    class _QuerySubmap:
        def get_id(self):
            return 30

        def get_all_retrieval_vectors(self):
            return [torch.tensor([0.0])]

    class _RecordingMap:
        requested_gap = None

        def retrieve_best_score_frame(
            self,
            query_vector,
            current_submap_id,
            ignore_last_submap,
            min_submap_gap,
        ):
            self.requested_gap = min_submap_gap
            return 0.2, 0, 0

    graph_map = _RecordingMap()
    retrieval = object.__new__(ImageRetrieval)
    loops = retrieval.find_loop_closures(
        graph_map,
        _QuerySubmap(),
        max_similarity_thres=0.8,
        max_loop_closures=1,
    )

    assert graph_map.requested_gap == 4
    assert len(loops) == 1


def _world_to_camera(center, camera_to_world_rotation):
    world_to_camera = np.eye(4)
    world_to_camera[:3, :3] = camera_to_world_rotation.T
    world_to_camera[:3, 3] = -world_to_camera[:3, :3] @ center
    return world_to_camera


def _transformed_trajectory(scale, rotation, translation):
    centers = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.2, 0.0],
        [2.0, 0.5, 0.1],
    ])
    source = np.stack([
        _world_to_camera(center, np.eye(3)) for center in centers
    ])
    target = np.stack([
        _world_to_camera(
            scale * rotation @ center + translation,
            rotation,
        )
        for center in centers
    ])
    return source, target


def test_centered_window_shifts_at_boundaries():
    assert centered_window_indices(0, 8) == [0, 1, 2]
    assert centered_window_indices(4, 8) == [3, 4, 5]
    assert centered_window_indices(7, 8) == [5, 6, 7]


def test_pose_sequence_accepts_batched_and_unbatched_vggt_shapes():
    unbatched = torch.zeros((6, 3, 4))
    assert _pose_sequence_numpy(unbatched).shape == (6, 3, 4)
    assert _pose_sequence_numpy(unbatched.unsqueeze(0)).shape == (6, 3, 4)


def test_sequence_pair_selection_keeps_center_and_chooses_direction():
    descriptors = np.arange(5, dtype=float)[:, None]
    result = select_sequence_pair_indices(
        2, 5, 2, 5, descriptors, descriptors
    )
    assert result["query_indices"] == [1, 2, 3]
    assert result["reference_indices"] == [1, 2, 3]
    assert result["direction"] == "forward"


def test_sequence_pair_selection_handles_boundary_with_reverse_motion():
    query_descriptors = np.array([[9.0], [9.0], [3.0], [2.0], [1.0]])
    reference_descriptors = np.array([[9.0], [1.0], [2.0], [3.0], [9.0]])
    result = select_sequence_pair_indices(
        4, 5, 1, 5, query_descriptors, reference_descriptors
    )
    assert result["query_indices"] == [2, 3, 4]
    assert result["reference_indices"] == [3, 2, 1]
    assert result["direction"] == "reverse"


def _common_transform_example(cross_scale=1.3):
    query_centers = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.2, 0.0],
        [2.0, 0.5, 0.1],
    ])
    reference_centers = np.array([
        [0.2, -0.1, 0.0],
        [1.1, 0.4, 0.1],
        [2.3, 0.4, 0.2],
    ])
    query = np.stack([
        _world_to_camera(center, np.eye(3)) for center in query_centers
    ])
    reference = np.stack([
        _world_to_camera(center, np.eye(3)) for center in reference_centers
    ])
    angle = np.deg2rad(15.0)
    cross_rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    cross_translation = np.array([4.0, -2.0, 0.5])
    pair_gauges = np.array([0.7, 1.4, 2.1])
    pairs = []
    for query_pose, reference_pose, query_center, reference_center, gauge in zip(
        query, reference, query_centers, reference_centers, pair_gauges
    ):
        measured_rotation = (
            reference_pose[:3, :3]
            @ cross_rotation
            @ query_pose[:3, :3].T
        )
        metric_translation = reference_pose[:3, :3] @ (
            cross_scale * cross_rotation @ query_center
            + cross_translation
            - reference_center
        )
        pair = np.stack((np.eye(4), np.eye(4)))
        pair[1, :3, :3] = measured_rotation
        pair[1, :3, 3] = metric_translation / gauge
        pairs.append(pair)
    return query, reference, np.stack(pairs)


def test_short_sequence_accepts_one_common_cross_submap_transform():
    query, reference, pairs = _common_transform_example()
    result = verify_common_cross_submap_transform(
        query[:, :3, :4], reference, pairs[:, :, :3, :4]
    )
    assert result["accepted"]
    assert result["relative_position_rmse"] < 1e-10
    assert result["rotation_rmse_deg"] < 1e-5
    assert result["positive_scales"]


def test_short_sequence_rejects_negative_cross_scale_with_positive_pair_gauges():
    query, reference, pairs = _common_transform_example(cross_scale=-0.2)
    result = verify_common_cross_submap_transform(query, reference, pairs)

    assert not result["accepted"]
    assert result["positive_scales"]
    assert not result["positive_cross_scale"]
    assert "nonpositive_cross_scale" in result["rejection_reasons"]


def test_short_sequence_rejects_inconsistent_cross_rotation():
    query, reference, pairs = _common_transform_example()
    bad_angle = np.deg2rad(35.0)
    bad_rotation = np.array([
        [np.cos(bad_angle), -np.sin(bad_angle), 0.0],
        [np.sin(bad_angle), np.cos(bad_angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    pairs[1, 1, :3, :3] = bad_rotation @ pairs[1, 1, :3, :3]
    result = verify_common_cross_submap_transform(query, reference, pairs)
    assert not result["accepted"]
    assert result["rotation_rmse_deg"] > 10.0
