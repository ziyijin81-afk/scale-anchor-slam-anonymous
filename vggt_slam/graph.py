import gtsam
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from gtsam import NonlinearFactorGraph, Values, noiseModel
from gtsam import SL4, PriorFactorSL4, BetweenFactorSL4
from gtsam.symbol_shorthand import X

from vggt_slam.slam_utils import decompose_camera, normalize_to_sl4


def _project_rotation(matrix):
    """Return the closest proper rotation to one finite 3x3 matrix."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("Rotation must be one finite 3x3 matrix")
    u, _, vt = np.linalg.svd(matrix)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    return u @ correction @ vt


def _camera_pose_from_graph_homography(homography, intrinsics):
    """Recover camera-to-world rotation and center from one SL(4) node."""
    homography = np.asarray(homography, dtype=float)
    intrinsics = np.asarray(intrinsics, dtype=float)
    if homography.shape != (4, 4) or intrinsics.shape != (4, 4):
        raise ValueError("Homography and intrinsics must both be 4x4")
    projection = intrinsics @ np.linalg.inv(homography)
    _, rotation, center, _ = decompose_camera(projection[:3, :])
    return _project_rotation(rotation), np.asarray(center, dtype=float)


class PoseGraph:
    def __init__(self):
        """Initialize a factor graph for Pose3 nodes with BetweenFactors."""
        self.graph = NonlinearFactorGraph()
        self.values = Values()
        inner_noise = 0.05*np.ones(15, dtype=float)
        intra_noise = 0.05*np.ones(15, dtype=float)
        self.inner_submap_noise = noiseModel.Diagonal.Sigmas(inner_noise)
        self.intra_submap_noise = noiseModel.Diagonal.Sigmas(intra_noise)
        self.anchor_noise = noiseModel.Diagonal.Sigmas([1e-6] * 15)
        loop_base_noise = noiseModel.Diagonal.Sigmas([0.05] * 7)
        self.loop_common_sim3_noise = noiseModel.Robust.Create(
            noiseModel.mEstimator.Huber.Create(1.345), loop_base_noise
        )
        self.loop_common_sim3_measurements = []
        self.node_intrinsics = {}
        self.initialized_nodes = set()
        self.num_loop_closures = 0 # Just used for debugging and analysis

        self.auto_cal_H_mats = dict()  # Store homographies estimated by auto-calibration
        # Derived output poses are rebuilt from submap0 after every scale solve.
        # Pose-graph values remain the unmodified source of rotations and
        # adjacency relations; this cache is safe to replace wholesale.
        self.scale_replayed_homographies = dict()
        self.submap_output_scales = dict()
        self.node_submap_anchors = dict()
        self.submap_parent_overlap_nodes = dict()

    def add_homography(self, key, global_h):
        """Add a new homography node to the graph."""
        # print("det(global_h)", np.linalg.det(global_h))
        # global_h = normalize_to_sl4(global_h)
        key = X(key)
        if key in self.initialized_nodes:
            print(f"SL4 {key} already exists.")
            return
        self.values.insert(key, SL4(global_h))
        self.initialized_nodes.add(key)

    def add_between_factor(self, key1, key2, relative_h, noise):
        """Add a relative SL4 constraint between two nodes."""
        # relative_h = normalize_to_sl4(relative_h)
        key1 = X(key1)
        key2 = X(key2)
        if key1 not in self.initialized_nodes or key2 not in self.initialized_nodes:
            raise ValueError(f"Both poses {key1} and {key2} must exist before adding a factor.")
        self.graph.add(BetweenFactorSL4(key1, key2, SL4(relative_h), noise))

    def add_common_sim3_loop_factor(
        self,
        reference_key,
        query_key,
        reference_intrinsics,
        query_intrinsics,
        measured_relative,
        *,
        reference_scale_at_estimation,
    ):
        """Store one direct loop for the scale-replayed Similarity3 graph."""
        reference_node = int(reference_key)
        query_node = int(query_key)
        reference_key = X(reference_node)
        query_key = X(query_node)
        if (
            reference_key not in self.initialized_nodes
            or query_key not in self.initialized_nodes
        ):
            raise ValueError("Both direct-loop endpoints must already exist")

        measured_relative = np.asarray(measured_relative, dtype=float).copy()
        if measured_relative.shape != (4, 4) or not np.all(np.isfinite(measured_relative)):
            raise ValueError("Direct common-Sim(3) measurement must be finite 4x4")
        reference_scale_at_estimation = float(reference_scale_at_estimation)
        if not np.isfinite(reference_scale_at_estimation) or reference_scale_at_estimation <= 0:
            raise ValueError("Loop reference scale at estimation must be finite and positive")
        if reference_node not in self.node_submap_anchors:
            raise ValueError("Loop reference node must belong to a registered ordinary submap")
        # The verifier consumed already-scaled reference poses. Store its
        # measurement in immutable local reference units, then materialize it
        # using the current scalar-graph scale at every replay (not cumulatively).
        local_relative = measured_relative.copy()
        local_relative[:3, 3] /= reference_scale_at_estimation
        loop_record = {
            "reference_node": reference_node,
            "query_node": query_node,
            "reference_submap": self.node_submap_anchors[reference_node],
            "reference_scale_at_estimation": reference_scale_at_estimation,
            "measurement_reference_local": local_relative,
        }
        measurement = self._common_sim3_measurement_in_current_scale(loop_record)
        initial_values = gtsam.Values()
        reference_state = self._similarity_state(
            reference_node, self.get_homography(reference_node), reference_intrinsics
        )
        query_state = self._similarity_state(
            query_node, self.get_homography(query_node), query_intrinsics
        )
        initial_values.insert(reference_key, reference_state)
        initial_values.insert(query_key, query_state)
        diagnostic_factor = gtsam.BetweenFactorSimilarity3(
            reference_key,
            query_key,
            measurement,
            self.loop_common_sim3_noise,
        )
        initial_error = float(diagnostic_factor.error(initial_values))
        self.loop_common_sim3_measurements.append(loop_record)
        return initial_error

    def _common_sim3_measurement_in_current_scale(self, loop):
        """Express an immutable local loop measurement in the latest replay unit."""
        reference_submap = int(loop["reference_submap"])
        scale = float(self.submap_output_scales[reference_submap])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid loop reference replay scale {reference_submap}: {scale}")
        local = np.asarray(loop["measurement_reference_local"], dtype=float)
        return gtsam.Similarity3(
            _project_rotation(local[:3, :3]), local[:3, 3] * scale, 1.0
        )

    @staticmethod
    def _similarity_state(node_id, homography, intrinsics):
        rotation, center = _camera_pose_from_graph_homography(
            homography, intrinsics
        )
        return gtsam.Similarity3(rotation, center, 1.0)

    def _apply_common_sim3_corrections(self, replayed, ordered_node_ids):
        """Optimize a correction graph without changing the ordinary SL(4) graph."""
        if not self.loop_common_sim3_measurements:
            return replayed

        correction_graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        initial_states = {}
        for node_id in ordered_node_ids:
            state = self._similarity_state(
                node_id, replayed[node_id], self.node_intrinsics[node_id]
            )
            initial_states[node_id] = state
            initial.insert(X(node_id), state)

        correction_graph.add(gtsam.PriorFactorSimilarity3(
            X(ordered_node_ids[0]),
            initial_states[ordered_node_ids[0]],
            noiseModel.Isotropic.Sigma(7, 1e-6),
        ))
        odometry_noise = noiseModel.Diagonal.Sigmas([0.05] * 7)
        for previous_id, node_id in zip(ordered_node_ids, ordered_node_ids[1:]):
            correction_graph.add(gtsam.BetweenFactorSimilarity3(
                X(previous_id),
                X(node_id),
                initial_states[previous_id].between(initial_states[node_id]),
                odometry_noise,
            ))

        active_loops = 0
        for loop in self.loop_common_sim3_measurements:
            reference_node = loop["reference_node"]
            query_node = loop["query_node"]
            if reference_node not in initial_states or query_node not in initial_states:
                continue
            correction_graph.add(gtsam.BetweenFactorSimilarity3(
                X(reference_node),
                X(query_node),
                self._common_sim3_measurement_in_current_scale(loop),
                self.loop_common_sim3_noise,
            ))
            active_loops += 1
        if active_loops == 0:
            return replayed

        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(100)
        result = gtsam.LevenbergMarquardtOptimizer(
            correction_graph, initial, params
        ).optimize()
        corrected = {}
        scales = []
        for node_id in ordered_node_ids:
            old_state = initial_states[node_id]
            new_state = result.atSimilarity3(X(node_id))
            correction = new_state.compose(old_state.inverse()).matrix()
            correction = correction / correction[3, 3]
            corrected[node_id] = correction @ replayed[node_id]
            scales.append(new_state.scale())
        print(
            "[loop-common-sim3-graph] "
            f"nodes={len(ordered_node_ids)} loops={active_loops} "
            f"initial_error={correction_graph.error(initial):.8g} "
            f"final_error={correction_graph.error(result):.8g} "
            f"node_scale_range=({min(scales):.8g},{max(scales):.8g})"
        )
        return corrected

    def add_prior_factor(self, key, global_h):
        # global_h = normalize_to_sl4(global_h)
        key = X(key)
        if key not in self.initialized_nodes:
            raise ValueError(f"Trying to add prior factor for key {key} but it is not in the graph.")
        self.graph.add(PriorFactorSL4(key, SL4(global_h), self.anchor_noise))

    def get_base_homography(self, node_id):
        """Return the pose-graph estimate before scale output corrections."""
        node_id = int(node_id)
        auto_cal_H = self.auto_cal_H_mats.get(node_id, np.eye(4))
        return auto_cal_H @ self.values.atSL4(X(node_id)).matrix()

    def get_homography(self, node_id):
        """
        Get the optimized SL4 homography at a specific node.
        :param node_id: The ID of the node.
        :return: gtsam.SL4 homography of the node.
        """

        raw_node_id = int(node_id)
        replayed = self.scale_replayed_homographies.get(raw_node_id)
        if replayed is not None:
            return replayed.copy()
        return self.get_base_homography(raw_node_id).copy()

    def get_projection_matrix(self, node_id):
        """
        Get the optimized SL4 homography at a specific node.
        :param node_id: The ID of the node.
        :return: gtsam.SL4 homography of the node.
        """
        homography = self.get_homography(node_id)
        projection_matrix = np.linalg.inv(homography)
        return projection_matrix

    def register_scale_submap(self, submap, parent_overlap_node_id=None):
        """Register an ordinary submap and its fixed first-node anchor."""
        anchor_id = int(submap.get_id())
        for pose_num in range(len(submap.poses)):
            node_id = anchor_id + pose_num
            if X(node_id) not in self.initialized_nodes:
                raise ValueError(f"Cannot register missing pose node {node_id}")
            self.node_submap_anchors[node_id] = anchor_id
            projection_matrices = getattr(submap, "proj_mats", None)
            self.node_intrinsics[node_id] = (
                np.eye(4)
                if projection_matrices is None
                else np.asarray(projection_matrices[pose_num], dtype=float).copy()
            )
        self.submap_parent_overlap_nodes[anchor_id] = (
            None if parent_overlap_node_id is None else int(parent_overlap_node_id)
        )
        self.submap_output_scales.setdefault(anchor_id, 1.0)

    @staticmethod
    def _scaled_translation(relative_h, scale):
        """Scale only a relative transform's translation column."""
        result = np.asarray(relative_h, dtype=float).copy()
        result[:3, 3] *= float(scale)
        return result

    def rebuild_scale_adjusted_homographies(self, ordered_submaps, scales):
        """Replay every ordinary submap from submap0 using the latest S_i.

        The first node of each later submap follows its original pose-graph
        edge from the updated parent overlap node. Internal relative motion is
        then replayed in order with its submap's absolute scale. No first/last
        pose equality is imposed, and rotations/intrinsics are not changed.
        """
        replayed = {}
        ordered_node_ids = []
        for position, submap in enumerate(ordered_submaps):
            if submap.get_lc_status():
                continue
            anchor_id = int(submap.get_id())
            ordered_node_ids.extend(
                anchor_id + pose_num for pose_num in range(len(submap.poses))
            )
            scale = 1.0 if position == 0 else float(scales.get(anchor_id, 1.0))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"Invalid absolute scale S_{anchor_id}={scale}")
            self.submap_output_scales[anchor_id] = scale

            parent_id = self.submap_parent_overlap_nodes.get(anchor_id)
            anchor_base = self.get_base_homography(anchor_id)
            if parent_id is None:
                replayed[anchor_id] = anchor_base.copy()
            else:
                if parent_id not in replayed:
                    raise RuntimeError(
                        f"Parent overlap node {parent_id} for submap {anchor_id} "
                        "was not replayed from submap0"
                    )
                parent_base = self.get_base_homography(parent_id)
                original_boundary = np.linalg.inv(parent_base) @ anchor_base
                replayed[anchor_id] = replayed[parent_id] @ original_boundary

            previous_id = anchor_id
            for pose_num in range(1, len(submap.poses)):
                node_id = anchor_id + pose_num
                previous_base = self.get_base_homography(previous_id)
                node_base = self.get_base_homography(node_id)
                original_inner = np.linalg.inv(previous_base) @ node_base
                scaled_inner = self._scaled_translation(original_inner, scale)
                replayed[node_id] = replayed[previous_id] @ scaled_inner
                previous_id = node_id

        replayed = self._apply_common_sim3_corrections(
            replayed, ordered_node_ids
        )
        self.scale_replayed_homographies = replayed
        return set(replayed)

    
    def optimize(self, verbose=False):
        """Optimize the graph with Levenberg–Marquardt and print per-factor errors."""
        # Optional verbosity settings
        params = gtsam.LevenbergMarquardtParams()
        if verbose:
            params.setVerbosityLM("SUMMARY")
            params.setVerbosity("ERROR")

        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.values, params)

        # --- Initial total error ---
        initial_error = self.graph.error(self.values)
        print(f"Initial total error: {initial_error:.6f}")

        # --- Per-factor initial error ---
        if verbose:
            print("\nInitial per-factor errors:")
            for i in range(self.graph.size()):
                factor = self.graph.at(i)
                try:
                    e = factor.error(self.values)
                    print(f"  Factor {i:3d}: error = {e:.6f}")
                except RuntimeError as ex:
                    print(f"  Factor {i:3d}: error could not be computed ({ex})")

            keys = [gtsam.DefaultKeyFormatter(k) for k in factor.keys()]
            print(f"Factor {i} connects to {keys} with error {e:.6f}")

        # --- Optimize ---
        result = optimizer.optimize()

        # --- Final total error ---
        final_error = self.graph.error(result)
        # print(f"\nFinal total error: {final_error:.6f}")

        # --- Per-factor final error ---
        if verbose:
            print("\nFinal per-factor errors:")
            for i in range(self.graph.size()):
                factor = self.graph.at(i)
                try:
                    e = factor.error(result)
                    print(f"  Factor {i:3d}: error = {e:.6f}")
                except RuntimeError as ex:
                    print(f"  Factor {i:3d}: error could not be computed ({ex})")

        # --- Store optimized values ---
        self.values = result


    def print_estimates(self):
        """Print the optimized poses."""
        for key in sorted(self.initialized_nodes):
            print(f"Homography{key}:\n{self.values.atSL4(key)}\n")
    
    def increment_loop_closure(self):
        """Increment the loop closure count."""
        self.num_loop_closures += 1
    
    def get_num_loops(self):
        """Get the number of loop closures."""
        return self.num_loop_closures

    def update_all_homographies(self, map, auto_cal_H_mats):
        count = 0
        for submap in map.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue
            for pose_num in range(len(submap.poses)):
                id = int(submap.get_id() + pose_num)
                self.auto_cal_H_mats[id] = np.linalg.inv(auto_cal_H_mats[count])
                count += 1
        assert count == len(auto_cal_H_mats), "Number of auto-calibration homographies does not match number of poses in the map."
