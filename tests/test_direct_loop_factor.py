import numpy as np

from vggt_slam.graph import PoseGraph, _camera_pose_from_graph_homography


def _pose(rotation=np.eye(3), translation=np.zeros(3)):
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def test_direct_common_sim3_factor_is_stored_outside_sl4_graph():
    graph = PoseGraph()
    graph.add_homography(0, _pose())
    graph.add_homography(1, _pose(translation=np.array([0.25, -0.1, 0.0])))
    graph.node_submap_anchors = {0: 0, 1: 0}
    graph.submap_output_scales = {0: 1.0}
    initial_error = graph.add_common_sim3_loop_factor(
        0,
        1,
        np.eye(4),
        np.eye(4),
        _pose(translation=np.array([0.25, -0.1, 0.0])),
        reference_scale_at_estimation=1.0,
    )
    assert initial_error < 1e-12
    assert graph.graph.size() == 0
    assert len(graph.loop_common_sim3_measurements) == 1


def test_similarity_correction_uses_full_translation_length():
    graph = PoseGraph()
    replayed = {
        0: _pose(),
        1: _pose(translation=np.array([1.0, 0.0, 0.0])),
        2: _pose(translation=np.array([2.0, 0.0, 0.0])),
    }
    graph.node_intrinsics = {node_id: np.eye(4) for node_id in replayed}
    graph.submap_output_scales = {0: 1.0}
    graph.loop_common_sim3_measurements.append({
        "reference_node": 0,
        "query_node": 2,
        "reference_submap": 0,
        "reference_scale_at_estimation": 1.0,
        "measurement_reference_local": _pose(translation=np.array([1.0, 0.0, 0.0])),
    })
    corrected = graph._apply_common_sim3_corrections(replayed, [0, 1, 2])
    _, endpoint = _camera_pose_from_graph_homography(
        corrected[2], np.eye(4)
    )
    assert endpoint[0] < 1.95
