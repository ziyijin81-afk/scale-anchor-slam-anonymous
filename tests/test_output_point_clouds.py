import numpy as np
import open3d as o3d
import cv2
from types import SimpleNamespace

from vggt_slam.map import GraphMap
from vggt_slam.solver import Solver
from vggt_slam.trajectory_evaluation import write_trajectory_comparisons


class _PointSubmap:
    def __init__(self, points, colors):
        self._points = np.asarray(points, dtype=float)
        self._colors = np.asarray(colors, dtype=float)

    def get_points_in_world_frame(self, graph):
        return self._points

    def get_points_colors(self):
        return self._colors


def test_saved_map_is_voxel_downsampled(tmp_path):
    graph_map = GraphMap()
    points = np.array([
        [0.001, 0.001, 0.001],
        [0.002, 0.002, 0.002],
        [1.000, 0.000, 0.000],
    ])
    graph_map.submaps[0] = _PointSubmap(points, np.ones((3, 3)))
    output = tmp_path / "map.pcd"

    graph_map.write_points_to_file(None, str(output), voxel_size=0.05)

    saved = o3d.io.read_point_cloud(str(output))
    assert len(saved.points) == 2


def test_trajectory_export_contains_one_sampled_sphere_per_pose(tmp_path):
    graph_map = GraphMap()
    camera_matrices = []
    for center in ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]):
        world_to_camera = np.eye(4)
        world_to_camera[:3, 3] = -np.asarray(center)
        camera_matrices.append(world_to_camera)
    graph_map.get_all_cam_matricies = lambda graph, give_camera_mat: np.stack(camera_matrices)
    output = tmp_path / "trajectory_spheres.pcd"

    graph_map.write_trajectory_spheres_to_file(
        None,
        str(output),
        sphere_radius=0.1,
        points_per_sphere=32,
    )

    saved = o3d.io.read_point_cloud(str(output))
    assert len(saved.points) == 3 * 32
    points = np.asarray(saved.points).reshape(3, 32, 3)
    recovered_centers = points.mean(axis=1)
    np.testing.assert_allclose(
        recovered_centers,
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        atol=1e-6,
    )


def test_dashed_anchor_line_has_visible_gaps_and_endpoints():
    points = Solver._dashed_line_points([0, 0, 0], [1, 0, 0], dash_length=0.2)
    np.testing.assert_allclose(points[0], [0, 0, 0])
    np.testing.assert_allclose(points[-1], [1, 0, 0])
    steps = np.diff(points[:, 0])
    assert np.max(steps) > 2.0 * np.median(steps)


class _AnchorSubmap:
    def __init__(self, submap_id, image, center):
        self._id = submap_id
        self._image = str(image)
        self._center = np.asarray(center, dtype=float)
        self.joint_anchor_measurement = None

    def get_img_names_at_index(self, index):
        return self._image

    def get_all_poses_world(self, graph, give_camera_mat=True):
        pose = np.eye(4)
        pose[:3, 3] = -self._center
        return np.stack([pose])


def test_anchor_artifacts_export_images_csv_and_dashed_cloud(tmp_path):
    root_image = tmp_path / "root.png"
    current_image = tmp_path / "current.png"
    cv2.imwrite(str(root_image), np.full((20, 30, 3), 80, dtype=np.uint8))
    cv2.imwrite(str(current_image), np.full((20, 30, 3), 160, dtype=np.uint8))
    root = _AnchorSubmap(0, root_image, [0, 0, 0])
    current = _AnchorSubmap(17, current_image, [2, 0, 0])
    current.joint_anchor_measurement = {
        "root_reference_frame": 0,
        "reference_frame": 0,
        "root_image_path": str(root_image),
        "current_image_path": str(current_image),
    }
    factor = SimpleNamespace(
        factor_type="anchor", submap_i=0, submap_j=17,
        measurement=1.2, sigma=1.0, factor_index=1,
        measurement_version=1, measurement_timestamp=123.0,
    )
    solver = Solver.__new__(Solver)
    solver.graph = None
    solver.map = SimpleNamespace(get_submap=lambda key: {0: root, 17: current}[key])
    solver.scale_graph = SimpleNamespace(
        factors={("anchor", 0, 17): factor}, nodes={0: 1.0, 17: 1.1}
    )
    trajectory_path = tmp_path / "trajectory.pcd"
    trajectory_cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    )
    trajectory_cloud.colors = o3d.utility.Vector3dVector(
        np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    )
    o3d.io.write_point_cloud(str(trajectory_path), trajectory_cloud)
    combined_path = tmp_path / "trajectory_with_anchors.pcd"
    result = solver.write_scale_anchor_artifacts(
        str(tmp_path / "anchors"),
        trajectory_cloud_path=str(trajectory_path),
        combined_output_path=str(combined_path),
    )
    assert result["count"] == 1
    assert (tmp_path / "anchors" / "anchors.csv").is_file()
    assert (tmp_path / "anchors" / "submap_000017" / "anchor_pair.jpg").is_file()
    edge_cloud = o3d.io.read_point_cloud(result["edges"])
    assert len(edge_cloud.points) > 2
    combined_cloud = o3d.io.read_point_cloud(result["combined_trajectory"])
    assert len(combined_cloud.points) == 2 + len(edge_cloud.points)


def test_reference_export_writes_both_alignment_modes(tmp_path):
    timestamps = np.arange(6, dtype=float) + 1_700_000_000.0
    reference_xyz = np.column_stack((np.arange(6), np.zeros(6), np.zeros(6)))
    estimate_xyz = 0.5 * reference_xyz + np.array([3.0, -2.0, 0.0])
    estimate = np.column_stack((timestamps, estimate_xyz, np.zeros((6, 4))))
    reference = np.column_stack((timestamps, reference_xyz, np.zeros((6, 4))))
    estimate_path = tmp_path / "estimate.txt"
    reference_path = tmp_path / "reference.txt"
    np.savetxt(estimate_path, estimate)
    np.savetxt(reference_path, reference)
    output = tmp_path / "trajectory_reference"
    result = write_trajectory_comparisons(
        estimate_path, output, reference_trajectory=reference_path
    )
    assert result["matched_poses"] == 6
    assert result["alignments"]["umeyama_sim3"]["ate_rmse_m"] < 1e-8
    assert result["alignments"]["start_aligned"]["first_pose_error_m"] < 1e-8
    for name in ("umeyama_sim3", "start_aligned"):
        assert (output / f"trajectory_{name}.png").is_file()
        assert (output / f"trajectory_{name}.csv").is_file()
