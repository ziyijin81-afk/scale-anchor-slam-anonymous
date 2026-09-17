import numpy as np
import cv2
import gtsam
import matplotlib.pyplot as plt
import torch
import time
import open3d as o3d
import csv
import json
import os
import shutil
from contextlib import contextmanager
from termcolor import colored
from scipy.linalg import rq

from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.load_fn import load_and_preprocess_images

from vggt_slam.slam_utils import compute_image_embeddings, Accumulator, decompose_camera
from vggt_slam.loop_closure import (
    ImageRetrieval,
    select_sequence_pair_indices,
    verify_common_cross_submap_transform,
)
from vggt_slam.frame_overlap import FrameTracker
from vggt_slam.map import GraphMap
from vggt_slam.submap import Submap
from vggt_slam.graph import PoseGraph
from vggt_slam.scale_solver import (
    ScaleFactorGraph,
    anchor_scale,
    anchor_has_minimum_points,
    camera_points_from_depth,
    estimate_scale_pairwise,
    frame_confidence_threshold,
    measurement_sigma,
    same_frame_scale_consistency,
)
from vggt_slam.viewer import Viewer

DEBUG = False

# Fixed factor weights for the scale graph. GTSAM accepts a standard
# deviation, so sigma = 1 / sqrt(weight).
OVERLAP_FACTOR_WEIGHT = 1.0
ANCHOR_FACTOR_WEIGHT = 0.1
OVERLAP_FACTOR_SIGMA = 1.0 / np.sqrt(OVERLAP_FACTOR_WEIGHT)
ANCHOR_FACTOR_SIGMA = 1.0 / np.sqrt(ANCHOR_FACTOR_WEIGHT)
ANCHOR_CONSISTENCY_MIN_POINTS = 500
ANCHOR_FALLBACK_MIN_POINTS = 100


def _cast_last_camera_tokens_to_fp32(_, args):
    """Forward pre-hook used by the two-frame BF16 CameraHead workaround."""
    aggregated_tokens = list(args[0])
    aggregated_tokens[-1] = aggregated_tokens[-1].float()
    return (aggregated_tokens, *args[1:])


def _pose_sequence_numpy(poses):
    """Normalize VGGT pose output to one ``[S, rows, 4]`` NumPy sequence."""
    if isinstance(poses, torch.Tensor):
        poses = poses.detach().float().cpu().numpy()
    poses = np.asarray(poses, dtype=float)
    if poses.ndim == 4 and poses.shape[0] == 1:
        poses = poses[0]
    if poses.ndim != 3 or poses.shape[-1] != 4:
        raise ValueError(f"Unexpected VGGT pose shape {poses.shape}")
    return poses


@contextmanager
def _short_batch_camera_head_fp32(model, frame_count, log_prefix="[vggt]"):
    """Avoid the torch 2.3/CUDA BF16 CameraHead SIGFPE on short batches."""
    camera_head = getattr(model, "camera_head", None)
    camera_head_dtype = None
    camera_head_hook = None
    # The frozen batch is eight new frames plus one overlap frame. Observed
    # non-17-frame batches require the FP32 CameraHead workaround on this host.
    if int(frame_count) != 17 and camera_head is not None:
        camera_parameter = next(camera_head.parameters(), None)
        if camera_parameter is not None and camera_parameter.dtype == torch.bfloat16:
            camera_head_dtype = camera_parameter.dtype
            camera_head.float()
            camera_head_hook = camera_head.register_forward_pre_hook(
                _cast_last_camera_tokens_to_fp32
            )
            print(f"{log_prefix} short-batch CameraHead precision=float32 "
                  f"(BF16 S={int(frame_count)} SIGFPE workaround)")
    try:
        yield
    finally:
        if camera_head_hook is not None:
            camera_head_hook.remove()
        if camera_head_dtype is not None:
            camera_head.to(dtype=camera_head_dtype)


def debug_visualize(pcd1_points, pcd2_points):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pcd1_points)
    pcd1.paint_uniform_color([1, 0, 0])  # red

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pcd2_points)
    pcd2.paint_uniform_color([0, 0, 1])  # blue

    o3d.visualization.draw_geometries([pcd1, pcd2], window_name="Pairwise Point Clouds")

class Solver:
    def __init__(self,
        init_conf_threshold: float,  # represents percentage (e.g., 50 means filter lowest 50%)
        lc_thres: float = 0.80,
        vis_voxel_size: float = None,
        vis_imgs: bool = False,
        viewer_port: int = 8080,
        enable_scale_anchor: bool = True,
        enable_scale_anchor_conf_filter: bool = True,
        enable_scale_anchor_conf_fallback: bool = True,
        scale_anchor_weight: float = ANCHOR_FACTOR_WEIGHT,
        scale_anchor_min_points: int = ANCHOR_CONSISTENCY_MIN_POINTS,
        scale_anchor_fallback_min_points: int = ANCHOR_FALLBACK_MIN_POINTS,
        scale_anchor_salad_max_distance: float = 0.9,
        scale_anchor_salad_weight_sigma: float = 0.5,
        scale_anchor_min_ordinal_span: int = 6,
        scale_anchor_target_ordinal_span: int = 6,
        scale_anchor_submap_interval: int = 1,
        scale_anchor_reference_mode: str = "fixed-root"):

        self.init_conf_threshold = init_conf_threshold
        self.vis_voxel_size = vis_voxel_size
        self.vis_imgs = vis_imgs
        self.enable_scale_anchor = bool(enable_scale_anchor)
        self.enable_scale_anchor_conf_filter = bool(enable_scale_anchor_conf_filter)
        self.enable_scale_anchor_conf_fallback = bool(
            enable_scale_anchor_conf_fallback
        )
        if not np.isfinite(scale_anchor_weight) or scale_anchor_weight <= 0:
            raise ValueError("scale_anchor_weight must be finite and positive")
        self.scale_anchor_sigma = 1.0 / np.sqrt(float(scale_anchor_weight))
        self.scale_anchor_min_points = int(scale_anchor_min_points)
        self.scale_anchor_fallback_min_points = int(scale_anchor_fallback_min_points)
        if self.scale_anchor_min_points <= 0:
            raise ValueError("scale_anchor_min_points must be positive")
        if self.scale_anchor_fallback_min_points <= 0:
            raise ValueError("scale_anchor_fallback_min_points must be positive")
        if not np.isfinite(scale_anchor_salad_max_distance) or scale_anchor_salad_max_distance <= 0:
            raise ValueError("scale_anchor_salad_max_distance must be finite and positive")
        if not np.isfinite(scale_anchor_salad_weight_sigma) or scale_anchor_salad_weight_sigma <= 0:
            raise ValueError("scale_anchor_salad_weight_sigma must be finite and positive")
        if int(scale_anchor_min_ordinal_span) < 2:
            raise ValueError("scale_anchor_min_ordinal_span must be at least 2")
        if int(scale_anchor_target_ordinal_span) <= 0:
            raise ValueError("scale_anchor_target_ordinal_span must be positive")
        if int(scale_anchor_submap_interval) <= 0:
            raise ValueError("scale_anchor_submap_interval must be positive")
        if scale_anchor_reference_mode not in {"salad-far", "fixed-root"}:
            raise ValueError(
                "scale_anchor_reference_mode must be 'salad-far' or 'fixed-root'"
            )
        self.scale_anchor_salad_max_distance = float(scale_anchor_salad_max_distance)
        self.scale_anchor_salad_weight_sigma = float(scale_anchor_salad_weight_sigma)
        self.scale_anchor_min_ordinal_span = int(scale_anchor_min_ordinal_span)
        self.scale_anchor_target_ordinal_span = int(scale_anchor_target_ordinal_span)
        self.scale_anchor_submap_interval = int(scale_anchor_submap_interval)
        self.scale_anchor_reference_mode = scale_anchor_reference_mode

        self.viewer = Viewer(port=viewer_port)

        self.flow_tracker = FrameTracker()
        self.map = GraphMap()
        self.graph = PoseGraph()
        # Match the original scale graph: accumulate overlap and Anchor
        # factors with incremental GTSAM iSAM2 updates.
        self.scale_graph = ScaleFactorGraph(use_isam2=True)
        self.scale_dirty_submap_ids = set()
        self.last_scale_corrections = {}

        self.image_retrieval = ImageRetrieval()
        self.current_working_submap = None

        self.lc_thres = lc_thres

        self.temp_count = 0
        self.vggt_timer = Accumulator()
        self.loop_closure_timer = Accumulator()
        self.clip_timer = Accumulator()

    @staticmethod
    def _dashed_line_points(start, end, dash_length, samples_per_dash=6):
        """Return point samples for a 3-D dashed segment, including both ends."""
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        delta = end - start
        length = float(np.linalg.norm(delta))
        if not np.isfinite(length) or length <= np.finfo(float).eps:
            return start.reshape(1, 3)
        dash_length = max(float(dash_length), length / 1000.0)
        sample_step = dash_length / max(int(samples_per_dash), 2)
        distances = np.arange(0.0, length + sample_step, sample_step)
        distances = np.minimum(distances, length)
        visible = (np.floor(distances / dash_length).astype(int) % 2) == 0
        visible[0] = True
        visible[-1] = True
        return start + distances[visible, None] * delta[None, :] / length

    @staticmethod
    def merge_point_cloud_files(input_paths, output_path):
        """Merge colored PCD layers without changing coordinates or colors."""
        merged = o3d.geometry.PointCloud()
        valid_inputs = []
        for input_path in input_paths:
            if not input_path or not os.path.isfile(input_path):
                continue
            cloud = o3d.io.read_point_cloud(str(input_path))
            if len(cloud.points) == 0:
                continue
            merged += cloud
            valid_inputs.append(str(input_path))
        if not valid_inputs:
            raise ValueError("No non-empty point-cloud layers were provided")
        if not o3d.io.write_point_cloud(str(output_path), merged, write_ascii=False):
            raise IOError(f"Failed to write merged point cloud to {output_path}")
        print(
            f"[output] merged_point_cloud={output_path} "
            f"layers={valid_inputs} points={len(merged.points)}"
        )
        return str(output_path)

    def write_scale_anchor_artifacts(
        self,
        output_dir,
        trajectory_cloud_path=None,
        combined_output_path=None,
    ):
        """Export active root-anchor pairs, images, metadata, and dashed PCD edges."""
        os.makedirs(output_dir, exist_ok=True)
        anchor_factors = sorted(
            (
                factor for factor in self.scale_graph.factors.values()
                if factor.factor_type == "anchor"
            ),
            key=lambda factor: factor.submap_j,
        )
        csv_path = os.path.join(output_dir, "anchors.csv")
        fieldnames = [
            "submap_id", "root_submap_id", "root_reference_frame",
            "current_reference_frame", "root_image", "current_image",
            "measurement", "sigma", "weight", "residual", "factor_index",
            "measurement_version", "measurement_timestamp",
        ]
        rows = []
        edge_specs = []
        for factor in anchor_factors:
            root_submap = self.map.get_submap(factor.submap_i)
            current_submap = self.map.get_submap(factor.submap_j)
            metadata = getattr(current_submap, "joint_anchor_measurement", None) or {}
            root_index = int(metadata.get("root_reference_frame", 0))
            current_index = int(metadata.get("reference_frame", 0))
            root_image = str(metadata.get(
                "root_image_path", root_submap.get_img_names_at_index(root_index)
            ))
            current_image = str(metadata.get(
                "current_image_path", current_submap.get_img_names_at_index(current_index)
            ))

            pair_dir = os.path.join(output_dir, f"submap_{factor.submap_j:06d}")
            os.makedirs(pair_dir, exist_ok=True)
            copied = {}
            for label, source in (("root", root_image), ("current", current_image)):
                extension = os.path.splitext(source)[1] or ".png"
                destination = os.path.join(
                    pair_dir,
                    f"{label}_frame_{root_index if label == 'root' else current_index:02d}{extension}",
                )
                if os.path.isfile(source):
                    shutil.copy2(source, destination)
                    copied[label] = destination
                else:
                    copied[label] = source
                    print(f"[anchor-output] source image missing: {source}")

            root_poses = root_submap.get_all_poses_world(self.graph, give_camera_mat=True)
            current_poses = current_submap.get_all_poses_world(self.graph, give_camera_mat=True)
            _, _, root_center, _ = decompose_camera(root_poses[root_index])
            _, _, current_center, _ = decompose_camera(current_poses[current_index])
            edge_specs.append((factor.submap_j, np.asarray(root_center), np.asarray(current_center)))
            residual = (
                self.scale_graph.nodes[factor.submap_j]
                - self.scale_graph.nodes[factor.submap_i] * factor.measurement
            )
            row = {
                "submap_id": factor.submap_j,
                "root_submap_id": factor.submap_i,
                "root_reference_frame": root_index,
                "current_reference_frame": current_index,
                "root_image": copied["root"],
                "current_image": copied["current"],
                "measurement": factor.measurement,
                "sigma": factor.sigma,
                "weight": 1.0 / factor.sigma ** 2,
                "residual": residual,
                "factor_index": factor.factor_index,
                "measurement_version": factor.measurement_version,
                "measurement_timestamp": factor.measurement_timestamp,
            }
            rows.append(row)
            with open(os.path.join(pair_dir, "metadata.json"), "w") as stream:
                json.dump(row, stream, indent=2)
                stream.write("\n")

            root_bgr = cv2.imread(root_image)
            current_bgr = cv2.imread(current_image)
            if root_bgr is not None and current_bgr is not None:
                target_height = min(root_bgr.shape[0], current_bgr.shape[0], 720)
                def resize_height(image):
                    width = int(round(image.shape[1] * target_height / image.shape[0]))
                    return cv2.resize(image, (width, target_height))
                pair = np.hstack((resize_height(root_bgr), resize_height(current_bgr)))
                cv2.putText(pair, f"submap{factor.submap_i} reference", (12, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(pair, f"submap{factor.submap_j} reference",
                            (resize_height(root_bgr).shape[1] + 12, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imwrite(os.path.join(pair_dir, "anchor_pair.jpg"), pair)

        with open(csv_path, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        edge_path = os.path.join(output_dir, "scale_anchor_edges.pcd")
        if edge_specs:
            centers = np.vstack([center for _, left, right in edge_specs for center in (left, right)])
            diagonal = float(np.linalg.norm(np.ptp(centers, axis=0)))
            dash_length = max(diagonal / 180.0, 0.01)
            points, colors = [], []
            for edge_index, (_, left, right) in enumerate(edge_specs):
                samples = self._dashed_line_points(left, right, dash_length)
                points.append(samples)
                # Alternate red and magenta slightly so overlapping edges remain visible.
                color = (1.0, 0.1, 0.1) if edge_index % 2 == 0 else (1.0, 0.1, 0.8)
                colors.append(np.tile(color, (len(samples), 1)))
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(points)))
            cloud.colors = o3d.utility.Vector3dVector(np.vstack(colors))
            if not o3d.io.write_point_cloud(edge_path, cloud, write_ascii=False):
                raise IOError(f"Failed to write Anchor edge cloud to {edge_path}")
            print(
                f"[anchor-output] directory={output_dir} anchors={len(rows)} "
                f"dashed_edges={edge_path}"
            )
        else:
            edge_path = None
            print(f"[anchor-output] directory={output_dir} anchors=0")
        combined_path = None
        if trajectory_cloud_path and combined_output_path:
            layers = [trajectory_cloud_path]
            if edge_path:
                layers.append(edge_path)
            combined_path = self.merge_point_cloud_files(layers, combined_output_path)
        return {
            "directory": output_dir,
            "csv": csv_path,
            "edges": edge_path,
            "combined_trajectory": combined_path,
            "count": len(rows),
        }

    def _apply_scale_graph_corrections(self):
        """Rebuild all ordinary geometry and placement from submap0."""
        pending = self.scale_graph.pending_corrections()
        for submap_id, correction in pending.items():
            if submap_id == 0:
                self.scale_graph.mark_applied(0)
                continue
            try:
                submap = self.map.get_submap(submap_id)
            except KeyError:
                continue
            old_scale = submap.applied_scale
            new_scale = self.scale_graph.nodes[submap_id]
            self.last_scale_corrections[submap_id] = correction
            print(f"[scale-graph] submap={submap_id} S_applied_old={old_scale:.8g} "
                  f"S_new={new_scale:.8g} correction_c={correction:.8g} "
                  "full_replay_from_submap0=True")

        # Absolute rebuild is intentional even for unchanged historical nodes:
        # an upstream scale update changes every descendant's global placement.
        ordinary_submaps = [
            self.map.get_submap(submap_id)
            for submap_id in self.map.non_lc_submap_ids
        ]
        for submap in ordinary_submaps:
            submap_id = int(submap.get_id())
            new_scale = 1.0 if submap_id == 0 else float(
                self.scale_graph.nodes.get(submap_id, submap.applied_scale)
            )
            submap.rebuild_from_local_scale(new_scale)
            if submap_id in self.scale_graph.nodes:
                self.scale_graph.mark_applied(submap_id)

        registered = [
            submap for submap in ordinary_submaps
            if int(submap.get_id()) in self.graph.submap_parent_overlap_nodes
        ]
        self.graph.rebuild_scale_adjusted_homographies(
            registered, self.scale_graph.nodes
        )
        self.scale_dirty_submap_ids.update(
            int(submap.get_id()) for submap in ordinary_submaps
        )

    def finalize_backend_update(self, vis_map=False, force_all=False):
        """Publish one coherent pose/scale revision after pose optimization."""
        ordinary_submaps = [
            self.map.get_submap(submap_id)
            for submap_id in self.map.non_lc_submap_ids
        ]
        registered = [
            submap for submap in ordinary_submaps
            if int(submap.get_id()) in self.graph.submap_parent_overlap_nodes
        ]
        self.graph.rebuild_scale_adjusted_homographies(
            registered, self.scale_graph.nodes
        )
        snapshot = self.scale_graph.snapshot()
        self.viewer.visualize_scale_graph(snapshot, self.last_scale_corrections)
        if vis_map:
            if force_all:
                self.update_all_submap_vis()
            else:
                redraw_ids = set(self.scale_dirty_submap_ids)
                latest = self.map.get_largest_key(ignore_loop_closure_submaps=True)
                if latest is not None:
                    redraw_ids.add(latest)
                for submap_id in sorted(redraw_ids):
                    submap = self.map.get_submap(submap_id)
                    self.set_submap_point_cloud(submap)
                    self.set_submap_poses(submap)
        print(f"[scale-graph] published revision={snapshot['revision']} "
              f"redrawn_submaps={sorted(self.scale_dirty_submap_ids)}")
        self.scale_dirty_submap_ids.clear()
        self.last_scale_corrections.clear()

    @staticmethod
    def _numpy_batch(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
        value = np.asarray(value)
        return value[0] if value.ndim > 0 and value.shape[0] == 1 else value

    def _select_scale_anchor_reference(self, current_submap):
        """Select a fixed-root or oldest sufficiently distant SALAD reference.

        This restores the successful ``salad-far`` policy: the immediate
        overlap neighbourhood is excluded, every eligible historical submap
        must pass the same SALAD distance threshold, and the oldest passing
        reference is used.  Semantic similarity only scales the information
        of the single selected Anchor; it never creates duplicate factors.
        """
        current_id = int(current_submap.get_id())
        ordered = [
            int(sid) for sid in sorted(self.map.non_lc_submap_ids)
            if int(sid) != current_id
        ]
        if len(ordered) < 2:
            return None
        eligible = [
            sid for index, sid in enumerate(ordered[:-1])
            if len(ordered) - index >= self.scale_anchor_min_ordinal_span
        ]
        if not eligible:
            return None
        current_ordinal = len(ordered)
        if (
            current_ordinal - self.scale_anchor_min_ordinal_span
        ) % self.scale_anchor_submap_interval != 0:
            print(
                f"[scale-anchor-cadence] submap={current_id} accepted=False "
                f"current_ordinal={current_ordinal} "
                f"first_ordinal={self.scale_anchor_min_ordinal_span} "
                f"interval={self.scale_anchor_submap_interval}"
            )
            return None
        current_vectors = current_submap.get_all_retrieval_vectors()
        if current_vectors is None or len(current_vectors) == 0:
            return None

        if self.scale_anchor_reference_mode == "fixed-root":
            reference_id = eligible[0]
            reference_vectors = self.map.get_submap(
                reference_id
            ).get_all_retrieval_vectors()
            if reference_vectors is None or len(reference_vectors) == 0:
                return None
            distances = torch.linalg.norm(
                current_vectors.float()[:, None, :]
                - reference_vectors.float()[None, :, :],
                dim=-1,
            )
            distances = torch.where(
                torch.isfinite(distances), distances, torch.inf
            )
            flat_index = int(torch.argmin(distances).item())
            salad_distance = float(distances.flatten()[flat_index].item())
            if not np.isfinite(salad_distance):
                print(
                    f"[scale-anchor-fixed-root] submap={current_id} "
                    "accepted=False reason=no_finite_frame_pair"
                )
                return None
            current_frame, reference_frame = divmod(
                flat_index, distances.shape[1]
            )
            selected = {
                "reference_submap_id": reference_id,
                "reference_frame": int(reference_frame),
                "current_frame": int(current_frame),
                "salad_distance": salad_distance,
                "ordinal_span": len(ordered),
                "weight_scale": 1.0,
            }
            print(
                f"[scale-anchor-fixed-root] submap={current_id} accepted=True "
                f"reference_submap={reference_id} "
                f"reference_frame={selected['reference_frame']} "
                f"current_frame={selected['current_frame']} "
                f"salad_distance={salad_distance:.6g} "
                f"ordinal_span={selected['ordinal_span']} weight_scale=1"
            )
            return selected

        candidates = []
        for reference_id in eligible:
            reference_vectors = self.map.get_submap(
                reference_id
            ).get_all_retrieval_vectors()
            if reference_vectors is None or len(reference_vectors) == 0:
                continue
            distances = torch.linalg.norm(
                current_vectors.float()[:, None, :]
                - reference_vectors.float()[None, :, :],
                dim=-1,
            )
            distances = torch.where(
                torch.isfinite(distances), distances, torch.inf
            )
            flat_index = int(torch.argmin(distances).item())
            current_frame, reference_frame = divmod(
                flat_index, distances.shape[1]
            )
            salad_distance = float(
                distances[current_frame, reference_frame].item()
            )
            if (
                np.isfinite(salad_distance)
                and salad_distance <= self.scale_anchor_salad_max_distance
            ):
                ordinal_span = len(ordered) - ordered.index(reference_id)
                semantic_weight = float(np.exp(-(
                    salad_distance / self.scale_anchor_salad_weight_sigma
                ) ** 2))
                span_weight = float(np.sqrt(min(
                    1.0,
                    ordinal_span / self.scale_anchor_target_ordinal_span,
                )))
                candidates.append({
                    "reference_submap_id": reference_id,
                    "reference_frame": int(reference_frame),
                    "current_frame": int(current_frame),
                    "salad_distance": salad_distance,
                    "ordinal_span": ordinal_span,
                    "weight_scale": semantic_weight * span_weight,
                })

        if not candidates:
            print(
                f"[scale-anchor-salad-far] submap={current_id} "
                f"accepted=False eligible_submaps={len(eligible)} "
                f"max_distance={self.scale_anchor_salad_max_distance:.6g} "
                "reason=no_semantically_similar_submap"
            )
            return None

        selected = min(
            candidates, key=lambda item: item["reference_submap_id"]
        )
        print(
            f"[scale-anchor-salad-far] submap={current_id} "
            f"accepted=True "
            f"reference_submap={selected['reference_submap_id']} "
            f"reference_frame={selected['reference_frame']} "
            f"current_frame={selected['current_frame']} "
            f"salad_distance={selected['salad_distance']:.6g} "
            f"ordinal_span={selected['ordinal_span']} "
            f"weight_scale={selected['weight_scale']:.6g} "
            f"semantic_candidates={len(candidates)} "
            f"eligible_submaps={len(eligible)}"
        )
        return selected

    def _compute_joint_anchor(self, model, predictions, current_submap):
        """Measure one semantic two-frame VGGT scale Anchor."""
        reference = self._select_scale_anchor_reference(current_submap)
        if reference is None:
            return None
        reference_submap_id = reference["reference_submap_id"]
        root_submap = self.map.get_submap(reference_submap_id)

        depth = self._numpy_batch(predictions["depth"])
        conf = self._numpy_batch(predictions["depth_conf"])
        intrinsics = self._numpy_batch(predictions["intrinsic"])
        current_points = np.stack([
            camera_points_from_depth(depth[i], intrinsics[i]) for i in range(len(depth))
        ])
        root_index = int(reference["reference_frame"])
        current_index = int(reference["current_frame"])
        print(
            f"[scale-anchor-reference] submap={current_submap.get_id()} "
            f"mode={self.scale_anchor_reference_mode} selector=salad "
            f"reference_submap={reference_submap_id} "
            f"root_frame={root_index} current_frame={current_index} "
            f"salad_distance={reference['salad_distance']:.6g}"
        )
        root_image_path = root_submap.get_img_names_at_index(root_index)
        current_image_path = current_submap.get_img_names_at_index(current_index)
        joint_images = torch.stack((
            root_submap.get_frame_at_index(root_index),
            current_submap.get_frame_at_index(current_index),
        ), dim=0)
        # Exactly one joint call is permitted for this Anchor pair.
        with _short_batch_camera_head_fp32(model, joint_images.shape[0], "[scale-anchor]"):
            with torch.no_grad():
                with self.vggt_timer:
                    joint_prediction = model(joint_images)
        joint_extrinsic, joint_intrinsic = pose_encoding_to_extri_intri(
            joint_prediction["pose_enc"], joint_images.shape[-2:]
        )
        del joint_extrinsic  # Scale is deliberately measured in camera coordinates.
        joint_depth = self._numpy_batch(joint_prediction["depth"])
        joint_conf = self._numpy_batch(joint_prediction["depth_conf"])
        joint_intrinsic = self._numpy_batch(joint_intrinsic)
        joint_points = np.stack([
            camera_points_from_depth(joint_depth[i], joint_intrinsic[i]) for i in range(2)
        ])
        # Confidence values from the 17-frame original inference and the
        # two-frame joint inference are not calibrated to a common numeric
        # distribution.  Compute four independent per-frame percentile
        # thresholds.  M_0 and M_k below then intersect the two trusted masks
        # at identical pixels within each reference image.
        root_original_threshold = frame_confidence_threshold(
            root_submap.conf_masks[root_index], self.init_conf_threshold
        )
        current_original_threshold = frame_confidence_threshold(
            conf[current_index], self.init_conf_threshold
        )
        root_joint_threshold = frame_confidence_threshold(
            joint_conf[0], self.init_conf_threshold
        )
        current_joint_threshold = frame_confidence_threshold(
            joint_conf[1], self.init_conf_threshold
        )

        # Keep the original identical-pixel scale estimator. Scatter statistics
        # are diagnostic only: no MAD/outlier/disagreement quality gate remains.
        # Submap pointclouds are the camera-coordinate points returned by
        # unproject_depth_map_to_point_map().  Do not apply world_to_camera a
        # second time: that would mix pose translation into the scale statistic.
        # Compare the two raw VGGT predictions. ``pointclouds`` already
        # contains the scale-graph correction and would feed an earlier graph
        # result back into this new Anchor measurement.
        root_camera = np.asarray(
            root_submap.local_pointclouds[root_index], dtype=float
        )
        root_consistency = same_frame_scale_consistency(
            root_camera,
            root_submap.conf_masks[root_index],
            root_original_threshold,
            joint_points[0],
            joint_conf[0],
            root_joint_threshold,
            min_points=self.scale_anchor_min_points,
            max_relative_mad=np.inf,
            max_outlier_ratio=np.inf,
            use_confidence=self.enable_scale_anchor_conf_filter,
        )
        current_consistency = same_frame_scale_consistency(
            current_points[current_index],
            conf[current_index],
            current_original_threshold,
            joint_points[1],
            joint_conf[1],
            current_joint_threshold,
            min_points=self.scale_anchor_min_points,
            max_relative_mad=np.inf,
            max_outlier_ratio=np.inf,
            use_confidence=self.enable_scale_anchor_conf_filter,
        )

        # Match VGGT-SLAM2's overlap fallback: first try the restrictive
        # confidence support, then relax the mask when too few pixels survive.
        # Geometry validity and identical-pixel correspondence remain
        # mandatory.  The relaxed support still needs at least 100 samples.
        confidence_fallback = False
        if (
            self.enable_scale_anchor_conf_filter
            and self.enable_scale_anchor_conf_fallback
            and (
                root_consistency["count"] < self.scale_anchor_min_points
                or current_consistency["count"] < self.scale_anchor_min_points
            )
        ):
            print(
                f"[scale-anchor] submap={current_submap.get_id()} "
                "not enough high-confidence shared pixels; "
                "using the less restrictive valid-geometry mask "
                f"root_valid={root_consistency['count']} "
                f"current_valid={current_consistency['count']}"
            )
            root_consistency = same_frame_scale_consistency(
                root_camera,
                root_submap.conf_masks[root_index],
                root_original_threshold,
                joint_points[0],
                joint_conf[0],
                root_joint_threshold,
                min_points=self.scale_anchor_fallback_min_points,
                max_relative_mad=np.inf,
                max_outlier_ratio=np.inf,
                use_confidence=False,
            )
            current_consistency = same_frame_scale_consistency(
                current_points[current_index],
                conf[current_index],
                current_original_threshold,
                joint_points[1],
                joint_conf[1],
                current_joint_threshold,
                min_points=self.scale_anchor_fallback_min_points,
                max_relative_mad=np.inf,
                max_outlier_ratio=np.inf,
                use_confidence=False,
            )
            confidence_fallback = True

        required_min_points = (
            self.scale_anchor_fallback_min_points
            if confidence_fallback else self.scale_anchor_min_points
        )
        hard_min_points_passed = anchor_has_minimum_points(
            root_consistency["count"],
            current_consistency["count"],
            required_min_points,
        )
        if not hard_min_points_passed:
            hard_reasons = []
            if root_consistency["count"] < required_min_points:
                hard_reasons.append(
                    f"root:valid_points<{required_min_points}"
                )
            if current_consistency["count"] < required_min_points:
                hard_reasons.append(
                    f"current:valid_points<{required_min_points}"
                )
            print(
                f"[scale-anchor-min-points] submap={current_submap.get_id()} "
                f"accepted=False hard_condition=True "
                f"root_valid={root_consistency['count']} "
                f"current_valid={current_consistency['count']} "
                f"min_points={required_min_points} "
                f"confidence_fallback={confidence_fallback} "
                f"root_original_conf_threshold={root_original_threshold:.8g} "
                f"root_joint_conf_threshold={root_joint_threshold:.8g} "
                f"current_original_conf_threshold={current_original_threshold:.8g} "
                f"current_joint_conf_threshold={current_joint_threshold:.8g} "
                f"reasons={hard_reasons}"
            )
            return {
                "accepted": False,
                "gate_enabled": False,
                "gate_passed": False,
                "hard_min_points_passed": False,
                "confidence_fallback": confidence_fallback,
                "reference_submap_id": reference_submap_id,
                "salad_distance": reference.get("salad_distance"),
                "root_reference_frame": root_index,
                "reference_frame": current_index,
                "root_image_path": root_image_path,
                "current_image_path": current_image_path,
                "root_valid_points": root_consistency["count"],
                "current_valid_points": current_consistency["count"],
                "valid_points": min(
                    root_consistency["count"], current_consistency["count"]
                ),
                "gate_reasons": hard_reasons,
            }

        # The four medians must describe the same scene samples within each
        # frame.  M_0 is the original/joint trusted-pixel intersection for F_0;
        # M_k is the equivalent intersection for F_k.  The anchor equation and
        # direction remain unchanged.
        A_0 = root_consistency["original_scale_stats"]["median"]
        a_0 = root_consistency["joint_scale_stats"]["median"]
        A_k = current_consistency["original_scale_stats"]["median"]
        a_k = current_consistency["joint_scale_stats"]["median"]
        measured_scale = anchor_scale(A_0, A_k, a_0, a_k)
        weight_scale = reference.get("weight_scale")
        if (
            weight_scale is not None
            and np.isfinite(weight_scale)
            and weight_scale > 0
        ):
            sigma = float(self.scale_anchor_sigma / np.sqrt(weight_scale))
        else:
            sigma = float(self.scale_anchor_sigma)
        if (not np.isfinite(measured_scale) or measured_scale <= 0
                or not np.isfinite(sigma) or sigma <= 0):
            raise ValueError("Anchor scale and sigma must be finite and positive")
        pointwise_anchor = (
            root_consistency["scale_ratio"]
            / current_consistency["scale_ratio"]
        )
        log_disagreement = (
            np.inf
            if not np.isfinite(pointwise_anchor) or pointwise_anchor <= 0
            else abs(np.log(measured_scale / pointwise_anchor))
        )
        paired_scale_stats = (
            root_consistency["original_scale_stats"],
            current_consistency["original_scale_stats"],
            root_consistency["joint_scale_stats"],
            current_consistency["joint_scale_stats"],
        )
        combined = {
            "median": measured_scale,
            "mad": measured_scale * np.mean([
                stats["mad"] / max(stats["median"], 1e-12)
                for stats in paired_scale_stats
            ]),
            "count": min(root_consistency["count"], current_consistency["count"]),
            "outlier_ratio": max(
                root_consistency["outlier_ratio"],
                current_consistency["outlier_ratio"],
            ),
        }
        estimated_sigma = measurement_sigma(combined, joint_conf, base_sigma=0.08)
        print(f"[scale-anchor] submap={current_submap.get_id()} root_reference_frame={root_index} "
              f"reference_submap={reference_submap_id} "
              f"reference_frame={current_index} "
              f"A_ref={A_0:.8g} A_k={A_k:.8g} "
              f"a_ref={a_0:.8g} a_k={a_k:.8g} "
              f"hat_s_k_to_ref={measured_scale:.8g} "
              f"shared_pixel_statistics=True "
              f"confidence_filter={self.enable_scale_anchor_conf_filter} "
              f"confidence_fallback={confidence_fallback} "
              f"root_original_conf_degenerate={root_consistency['original_confidence_degenerate']} "
              f"root_joint_conf_degenerate={root_consistency['joint_confidence_degenerate']} "
              f"current_original_conf_degenerate={current_consistency['original_confidence_degenerate']} "
              f"current_joint_conf_degenerate={current_consistency['joint_confidence_degenerate']} "
              f"root_original_conf_threshold={root_original_threshold:.8g} "
              f"root_joint_conf_threshold={root_joint_threshold:.8g} "
              f"current_original_conf_threshold={current_original_threshold:.8g} "
              f"current_joint_conf_threshold={current_joint_threshold:.8g} "
              f"valid_points={combined['count']} sample_median={combined['median']:.8g} "
              f"sample_MAD={combined['mad']:.8g} estimated_sigma={estimated_sigma:.8g} "
              f"factor_sigma={sigma:.8g} factor_weight={1.0 / sigma**2:.8g}")
        print(
            f"[scale-anchor-diagnostics] submap={current_submap.get_id()} "
            f"accepted=True quality_gate=False "
            f"min_points={required_min_points} "
            f"root_valid={root_consistency['count']} "
            f"root_scale_ratio={root_consistency['scale_ratio']:.8g} "
            f"root_relative_MAD={root_consistency['relative_mad']:.8g} "
            f"root_outlier_ratio={root_consistency['outlier_ratio']:.8g} "
            f"current_valid={current_consistency['count']} "
            f"current_scale_ratio={current_consistency['scale_ratio']:.8g} "
            f"current_relative_MAD={current_consistency['relative_mad']:.8g} "
            f"current_outlier_ratio={current_consistency['outlier_ratio']:.8g} "
            f"pointwise_anchor={pointwise_anchor:.8g} "
            f"formula_anchor={measured_scale:.8g} "
            f"log_disagreement={log_disagreement:.8g}"
        )
        return {
            "measurement": measured_scale, "sigma": sigma, "accepted": True,
            "gate_enabled": False,
            "gate_passed": True,
            "hard_min_points_passed": True,
            "confidence_fallback": confidence_fallback,
            "reference_submap_id": reference_submap_id,
            "salad_distance": reference.get("salad_distance"),
            "salad_weight_scale": weight_scale,
            "root_reference_frame": root_index, "reference_frame": current_index,
            "root_image_path": root_image_path,
            "current_image_path": current_image_path,
            "A_0": A_0, "A_k": A_k, "a_0": a_0, "a_k": a_k,
            "valid_points": combined["count"], "mad": combined["mad"],
            "root_consistency": root_consistency,
            "current_consistency": current_consistency,
            "pointwise_anchor": pointwise_anchor,
            "log_disagreement": log_disagreement,
            "gate_reasons": [],
        }

    def set_point_cloud(self, points_in_world_frame, points_colors, name, point_size):
        if self.vis_voxel_size is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_in_world_frame.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(points_colors.astype(np.float64) / 255.0)
            pcd = pcd.voxel_down_sample(self.vis_voxel_size)
            points_in_world_frame = np.asarray(pcd.points, dtype=np.float32)
            points_colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        self.viewer.server.scene.add_point_cloud(
            name="pcd_"+name,
            points=points_in_world_frame,
            colors=points_colors,
            point_size=point_size,
            point_shape="circle",
        )

    def set_submap_point_cloud(self, submap):
        # Add the point cloud to the visualization.
        points_in_world_frame = submap.get_points_in_world_frame(self.graph)
        points_colors = submap.get_points_colors()
        name = str(submap.get_id())
        self.set_point_cloud(points_in_world_frame, points_colors, name, 0.001)

    def set_submap_poses(self, submap):
        # Add the camera poses to the visualization.
        extrinsics = submap.get_all_poses_world(self.graph)
        images = submap.get_all_frames() if self.vis_imgs else None
        self.viewer.visualize_frames(extrinsics, images, submap.get_id())

    def update_all_submap_vis(self):
        for submap in self.map.get_submaps():
            self.set_submap_point_cloud(submap)
            self.set_submap_poses(submap)

    def update_latest_submap_vis(self):
        submap = self.map.get_latest_submap()
        self.set_submap_point_cloud(submap)
        self.set_submap_poses(submap)

    def tranform_submap_to_canonical(self, proj_mat_world_to_cam, world_points):
        P_first_cam = proj_mat_world_to_cam[0].copy()

        # Apply transformation to camera matrices such that the first camera matrix of the submap is [I | 0]
        proj_mat_world_to_cam = proj_mat_world_to_cam @ np.linalg.inv(P_first_cam)

        # Apply transformation to points such that the first camera matrix of the submap is [I | 0]
        h, w = world_points.shape[1:3]
        for i in range(len(proj_mat_world_to_cam)):
            points_in_cam = world_points[i,...]
            points_in_cam_h = np.hstack([points_in_cam.reshape(-1, 3), np.ones((points_in_cam.shape[0] * points_in_cam.shape[1], 1))])
            points_in_cam_h = (P_first_cam @ points_in_cam_h.T).T
            points_in_cam = points_in_cam_h[:, :3] / points_in_cam_h[:, 3:]
            world_points[i] = points_in_cam.reshape(h, w, 3)
        
        return proj_mat_world_to_cam, world_points

    def add_edge(self, submap_id_curr, frame_id_curr, submap_id_prev=None, frame_id_prev=None, is_loop_closure=False):
        assert not (is_loop_closure and submap_id_prev is None), "Loop closure must have a previous submap"
        scale_factor = 1.0
        current_submap = self.map.get_submap(submap_id_curr)
        H_w_submap = np.eye(4)
        if submap_id_prev is not None:
            overlapping_node_id_prev = submap_id_prev + frame_id_prev

            # Estimate scale factor between submaps.
            prior_submap = self.map.get_submap(submap_id_prev)

            current_conf = current_submap.get_conf_masks_frame(frame_id_curr)
            prior_conf = prior_submap.get_conf_masks_frame(frame_id_prev)
            good_mask = (prior_conf > prior_submap.get_conf_threshold()) * (current_conf > prior_submap.get_conf_threshold())

            if np.sum(good_mask) < 100:
                print(colored("Not enough overlapping points to estimate scale factor, using a less restrictive mask", 'red'))
                good_mask = prior_conf > prior_submap.get_conf_threshold()
                if np.sum(good_mask) < 100: # Handle the case where loop closure frames do not have enough points. 
                    good_mask = prior_conf > 0

            P_temp = np.linalg.inv(prior_submap.proj_mats[-1]) @ current_submap.proj_mats[0]
            current_points = current_submap.get_frame_pointcloud(frame_id_curr)
            prior_points = prior_submap.get_frame_pointcloud(frame_id_prev)
            ordinary_scale_edge = (
                not current_submap.get_lc_status() and not prior_submap.get_lc_status()
            )
            point_shape = current_points.shape[:2]
            good_mask = np.asarray(good_mask, dtype=bool).reshape(point_shape)
            t1_map = (
                P_temp[0:3, 0:3] @ current_points.reshape(-1, 3).T
            ).T.reshape(current_points.shape)
            t2_map = prior_points.reshape(current_points.shape)
            t1 = t1_map[good_mask]
            t2 = t2_map[good_mask]
            scale_factor_est_output = estimate_scale_pairwise(t1, t2)
            scale_factor = scale_factor_est_output[0]
            print(colored("scale factor", 'green'), scale_factor_est_output)
            # Special loop-closure helper submaps are outside the scale graph
            # and retain their pre-existing alignment behavior.
            H_scale = np.diag((scale_factor, scale_factor, scale_factor, 1.0))

            # The original overlap estimate above is now a measurement, not an
            # immediate/final rescaling operation. Loop-closure helper submaps
            # are intentionally excluded from the ordinary submap scale graph.
            if ordinary_scale_edge:
                overlap_sigma = float(OVERLAP_FACTOR_SIGMA)
                self.scale_graph.update_overlap(
                    submap_id_prev, submap_id_curr, scale_factor, overlap_sigma
                )
                print(
                    f"[scale-overlap] submap={submap_id_curr} "
                    f"prior={submap_id_prev} mode=global "
                    f"source=depth-overlap "
                    f"hat_s_k_to_kminus1={scale_factor:.8g} "
                    f"valid_pixels={int(np.sum(good_mask))} "
                    f"factor_sigma={overlap_sigma:.8g} "
                    f"factor_weight={1.0 / overlap_sigma**2:.8g}"
                )
                anchor = getattr(current_submap, "joint_anchor_measurement", None)
                if (self.enable_scale_anchor and anchor is not None
                        and anchor.get("accepted", True)):
                    reference_id = int(anchor.get("reference_submap_id", 0))
                    # One selected Anchor per current submap; repeated
                    # measurements replace the same graph edge.
                    self.scale_graph.update_anchor(
                        submap_id_curr,
                        anchor["measurement"],
                        anchor["sigma"],
                        reference_id=reference_id,
                    )
                elif self.enable_scale_anchor and anchor is not None:
                    reference_id = int(anchor.get("reference_submap_id", 0))
                    print(
                        f"[scale-anchor-gate] skipped_factor "
                        f"edge=({reference_id},{submap_id_curr}) "
                        f"reasons={anchor.get('gate_reasons', [])}"
                    )
                self.scale_graph.optimize()
                self._apply_scale_graph_corrections()
                H_scale = np.eye(4)

            if DEBUG:
                print("Estimated scale factor between submaps:", scale_factor)
                debug_visualize(scale_factor*t1, t2)

            # Compute the first camera matrix of the new submap in world frame.
            # For ordinary submaps H_scale is identity because scale was already
            # applied once through c_i=S_new/S_applied. The non-identity branch
            # only preserves legacy handling for helper loop-closure submaps.
            H_overlap_prior_overlap_current = (
                np.linalg.inv(prior_submap.proj_mats[-1])
                @ current_submap.proj_mats[0]
                @ H_scale
            )
            # Ordinary nodes are registered into an anchor-preserving output
            # chain below. Build their raw SL(4) value from the raw parent so
            # the parent's output correction is propagated exactly once.
            if ordinary_scale_edge:
                prior_homography = self.graph.get_base_homography(
                    overlapping_node_id_prev
                )
            else:
                prior_homography = self.graph.get_homography(
                    overlapping_node_id_prev
                )
            H_w_submap = prior_homography @ H_overlap_prior_overlap_current

            # Add first node of the new submap to the graph.
            if not is_loop_closure:
                self.graph.add_homography(submap_id_curr + frame_id_curr, H_w_submap)

            # Add between factor for intra submaps constraint.
            self.graph.add_between_factor(overlapping_node_id_prev, submap_id_curr + frame_id_curr, H_overlap_prior_overlap_current, self.graph.intra_submap_noise)

            if DEBUG:
                print("Adding first homography of submap: \n", submap_id_curr + frame_id_curr, H_w_submap / H_w_submap[-1,-1])
                print("Adding between factor: \n", overlapping_node_id_prev, submap_id_curr + frame_id_curr, H_scale)

        else:
            assert (submap_id_curr == 0 and frame_id_curr == 0), "First added node must be submap 0 frame 0"
            self.scale_graph.ensure_node(0)
            self.graph.add_homography(submap_id_curr + frame_id_curr, H_w_submap)
            self.graph.add_prior_factor(submap_id_curr + frame_id_curr, H_w_submap)
            if DEBUG:
                print("Adding first homography of graph: \n", submap_id_curr + frame_id_curr, H_w_submap / H_w_submap[-1,-1])

        # Loop closure only gets intra submap constraints.
        if is_loop_closure:
            return

        # Add nodes and edges for the inner submap constraints.
        # Pose-graph source nodes are always constructed from immutable local
        # VGGT poses. Scale is applied only by the output replay above.
        world_to_cam = current_submap.local_poses
        for index, pose in enumerate(world_to_cam):
            if index == 0:
                continue

            H_inner = world_to_cam[index-1] @ np.linalg.inv(pose)
            current_node = self.graph.get_base_homography(submap_id_curr + index - 1) @ H_inner

            # Add node to graph.
            self.graph.add_homography(submap_id_curr + index, current_node)

            # Add between factor for inner submap constraint.
            self.graph.add_between_factor(submap_id_curr + index - 1, submap_id_curr + index, H_inner, self.graph.inner_submap_noise)

            if DEBUG:
                print("Adding homography: \n", submap_id_curr + index, current_node / current_node[-1,-1])
                print("Adding between factor: \n", submap_id_curr + index - 1, submap_id_curr + index, H_inner)

        # Helper loop-closure submaps remain outside the scale graph.
        if not current_submap.get_lc_status():
            parent_overlap_node_id = (
                None if submap_id_prev is None else overlapping_node_id_prev
            )
            self.graph.register_scale_submap(
                current_submap, parent_overlap_node_id=parent_overlap_node_id
            )
            ordinary_submaps = [
                self.map.get_submap(submap_id)
                for submap_id in self.map.non_lc_submap_ids
            ]
            self.graph.rebuild_scale_adjusted_homographies(
                ordinary_submaps, self.scale_graph.nodes
            )

    def add_points(self, pred_dict):
        """
        Args:
            pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        """
        # Unpack prediction dict
        t1 = time.time()
        images = pred_dict["images"]  # (S, 3, H, W)
        extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
        intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)

        detected_loops = pred_dict["detected_loops"]

        depth_map = pred_dict["depth"]  # (S, H, W, 1)
        conf = pred_dict["depth_conf"]  # (S, H, W)

        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)

        colors = (images.transpose(0, 2, 3, 1) * 255).astype(np.uint8)  # now (S, H, W, 3)
        cam_to_world = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4)
        h, w = world_points.shape[1:3]
        
        # Create projection matrices
        N = cam_to_world.shape[0]
        K_4x4 = np.tile(np.eye(4), (N, 1, 1))
        K_4x4[:, :3, :3] = intrinsics_cam
        world_to_cam = np.linalg.inv(cam_to_world)


        submap_id_prev = self.map.get_largest_key(ignore_loop_closure_submaps=True)
        submap_id_curr = self.current_working_submap.get_id()
        frame_id_curr = 0
        frame_id_prev = None

        first_edge = submap_id_prev is None

        if not first_edge:
            frame_id_prev = self.map.get_latest_submap(ignore_loop_closure_submaps=True).get_last_non_loop_frame_index()

        # Add attributes to submap and add submap to map.
        self.current_working_submap.add_all_poses(world_to_cam)
        self.current_working_submap.add_all_points(
            world_points, colors, conf, self.init_conf_threshold, K_4x4, depths=depth_map
        )
        self.current_working_submap.set_conf_masks(conf)
        self.current_working_submap.joint_anchor_measurement = pred_dict.get("joint_scale_anchor")
        self.map.add_submap(self.current_working_submap)

        # Add all constraints for the new submap.
        self.add_edge(submap_id_curr, frame_id_curr, submap_id_prev, frame_id_prev, is_loop_closure=False)

        # Add in loop closures if any were detected.
        for index, loop in enumerate(detected_loops):
            assert loop.query_submap_id == self.current_working_submap.get_id()
            verification = pred_dict.get("loop_sequence_verification") or {}
            query_indices = verification.get("query_indices", [])
            direct_relatives = verification.get("direct_relative_transforms", [])
            if loop.query_submap_frame not in query_indices:
                raise ValueError("Accepted loop is missing its direct Sim(3) measurement")
            slot = query_indices.index(loop.query_submap_frame)
            if slot >= len(direct_relatives):
                raise ValueError("Accepted loop direct Sim(3) slot is out of range")

            reference_submap = self.map.get_submap(loop.detected_submap_id)
            reference_node = loop.detected_submap_id + loop.detected_submap_frame
            query_node = loop.query_submap_id + loop.query_submap_frame
            initial_error = self.graph.add_common_sim3_loop_factor(
                reference_node,
                query_node,
                reference_submap.proj_mats[loop.detected_submap_frame],
                self.current_working_submap.proj_mats[loop.query_submap_frame],
                np.asarray(direct_relatives[slot], dtype=float),
                reference_scale_at_estimation=verification["reference_scale_at_estimation"],
            )
            self.graph.increment_loop_closure()
            print(
                "[loop-common-sim3-factor] "
                f"reference_node={reference_node} query_node={query_node} "
                f"initial_robust_error={initial_error:.8g} "
                f"cross_scale={verification.get('cross_scale')} "
                f"pair_scale={verification.get('pair_scales', [])[slot]} "
                f"reference_scale_at_estimation={verification['reference_scale_at_estimation']} "
                "helper_nodes=0 residual=rotation+full_translation+relative_scale"
            )

    def sample_pixel_coordinates(self, H, W, n):
        # Sample n random row indices (y-coordinates)
        y_coords = torch.randint(0, H, (n,), dtype=torch.float32)
        # Sample n random column indices (x-coordinates)
        x_coords = torch.randint(0, W, (n,), dtype=torch.float32)
        # Stack to create an (n,2) tensor
        pixel_coords = torch.stack((y_coords, x_coords), dim=1)
        return pixel_coords

    def run_predictions(self, image_names, model, max_loops, clip_model, clip_preprocess):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        t1 = time.time()
        with self.vggt_timer:
            images = load_and_preprocess_images(image_names).to(device)
        print(f"Loaded and preprocessed {len(image_names)} images in {time.time() - t1:.2f} seconds")
        print(f"Preprocessed images shape: {images.shape}")

        # print("Running inference...")
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        # First submap so set new pcd num to 0
        if self.map.get_largest_key() is None:
            new_pcd_num = 0
        else:
            new_pcd_num = self.map.get_largest_key() + self.map.get_latest_submap().get_last_non_loop_frame_index() + 1

        print(f"Creating new submap with id {new_pcd_num}")
        t1 = time.time()
        new_submap = Submap(new_pcd_num)
        new_submap.add_all_frames(images)
        new_submap.set_frame_ids(image_names)
        new_submap.set_last_non_loop_frame_index(images.shape[0] - 1)
        new_submap.set_all_retrieval_vectors(self.image_retrieval.get_all_submap_embeddings(new_submap))
        new_submap.set_img_names(image_names)

        with self.clip_timer:
            if clip_model is not None and clip_preprocess is not None:
                image_embs = compute_image_embeddings(clip_model, clip_preprocess, image_names)
                new_submap.set_all_semantic_vectors(image_embs)

        self.current_working_submap = new_submap
        print(f"Created new submap in {time.time() - t1:.2f} seconds")

        with torch.no_grad():
            t1 = time.time()
            with _short_batch_camera_head_fp32(model, images.shape[0], "[vggt]"):
                with self.vggt_timer:
                    predictions = model(images)
            print(f"VGGT model inference took {time.time() - t1:.2f} seconds")

        print("Converting pose encoding to extrinsic and intrinsic matrices...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        # Check for loop closures and add retrieval vectors from new submap to the database
        predictions_lc = None
        loop_sequence_verification = None
        with self.loop_closure_timer:
            detected_loops = self.image_retrieval.find_loop_closures(self.map, new_submap, max_loop_closures=max_loops, max_similarity_thres=self.lc_thres)
        loop_closure_frame_names = []
        if len(detected_loops) > 0:
            print(colored("detected_loops", "yellow"), detected_loops)
            retrieved_frames = self.map.get_frames_from_loops(detected_loops)
            with torch.no_grad():
                lc_frames = torch.stack((new_submap.get_frame_at_index(detected_loops[0].query_submap_frame), retrieved_frames[0]), axis=0)
                with _short_batch_camera_head_fp32(model, lc_frames.shape[0], "[loop-closure]"):
                    predictions_lc = model(lc_frames, compute_similarity=True)
                loop_closure_frame_names = [new_submap.get_img_names_at_index(detected_loops[0].query_submap_frame), 
                self.map.get_submap(detected_loops[0].detected_submap_id).get_img_names_at_index(detected_loops[0].detected_submap_frame)]

                image_match_ratio = float(
                    torch.as_tensor(predictions_lc["image_match_ratio"]).item()
                )
                if image_match_ratio < 0.95:
                    loop_sequence_verification = {
                        "accepted": False,
                        "reason": "center_image_match_ratio",
                        "image_match_ratio": image_match_ratio,
                    }
                else:
                    loop = detected_loops[0]
                    reference_submap = self.map.get_submap(
                        loop.detected_submap_id
                    )
                    try:
                        query_descriptors = (
                            new_submap.get_all_retrieval_vectors()
                            .detach().float().cpu().numpy()
                        )
                        reference_descriptors = (
                            reference_submap.get_all_retrieval_vectors()
                            .detach().float().cpu().numpy()
                        )
                        sequence_pairs = select_sequence_pair_indices(
                            loop.query_submap_frame,
                            len(new_submap.get_all_frames()),
                            loop.detected_submap_frame,
                            len(reference_submap.get_all_frames()),
                            query_descriptors,
                            reference_descriptors,
                        )
                        query_indices = sequence_pairs["query_indices"]
                        reference_indices = sequence_pairs["reference_indices"]
                        pair_extrinsics = []
                        pair_match_ratios = []
                        for query_index, reference_index in zip(
                            query_indices, reference_indices
                        ):
                            if (
                                query_index == loop.query_submap_frame
                                and reference_index == loop.detected_submap_frame
                            ):
                                pair_prediction = predictions_lc
                                pair_images = lc_frames
                                pair_match_ratio = image_match_ratio
                            else:
                                pair_images = torch.stack((
                                    new_submap.get_frame_at_index(query_index),
                                    reference_submap.get_frame_at_index(reference_index),
                                ), axis=0)
                                with _short_batch_camera_head_fp32(
                                    model,
                                    pair_images.shape[0],
                                    "[loop-sequence-pair]",
                                ):
                                    pair_prediction = model(
                                        pair_images, compute_similarity=True
                                    )
                                pair_match_ratio = float(torch.as_tensor(
                                    pair_prediction["image_match_ratio"]
                                ).item())
                            pair_extrinsic, _ = pose_encoding_to_extri_intri(
                                pair_prediction["pose_enc"],
                                pair_images.shape[-2:],
                            )
                            pair_extrinsics.append(
                                _pose_sequence_numpy(pair_extrinsic)
                            )
                            pair_match_ratios.append(pair_match_ratio)

                        if min(pair_match_ratios) < 0.95:
                            loop_sequence_verification = {
                                "accepted": False,
                                "reason": "sequence_image_match_ratio",
                            }
                        else:
                            loop_sequence_verification = (
                                verify_common_cross_submap_transform(
                                    _pose_sequence_numpy(
                                        predictions["extrinsic"]
                                    )[query_indices],
                                    reference_submap.get_all_poses()[
                                        reference_indices
                                    ],
                                    np.stack(pair_extrinsics),
                                )
                            )
                            loop_sequence_verification["reason"] = (
                                "accepted"
                                if loop_sequence_verification["accepted"]
                                else ",".join(loop_sequence_verification["rejection_reasons"])
                            )
                            # Capture the unit at inference time: add_edge() can
                            # update historical scales before this loop is inserted.
                            loop_sequence_verification["reference_scale_at_estimation"] = (
                                float(reference_submap.applied_scale)
                            )
                        loop_sequence_verification.update({
                            "image_match_ratio": image_match_ratio,
                            "pair_match_ratios": pair_match_ratios,
                            "query_indices": query_indices,
                            "reference_indices": reference_indices,
                            "sequence_direction": sequence_pairs["direction"],
                            "descriptor_cost": sequence_pairs["descriptor_cost"],
                        })
                    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                        loop_sequence_verification = {
                            "accepted": False,
                            "reason": f"short_sequence_error:{error}",
                            "image_match_ratio": image_match_ratio,
                        }

                print(
                    "[loop-sequence-gate] "
                    f"query=({detected_loops[0].query_submap_id},"
                    f"{detected_loops[0].query_submap_frame}) "
                    f"reference=({detected_loops[0].detected_submap_id},"
                    f"{detected_loops[0].detected_submap_frame}) "
                    f"result={loop_sequence_verification}"
                )
                if not loop_sequence_verification["accepted"]:
                    predictions_lc = None
                    detected_loops = []

            # Visualize loop closure frames
            if DEBUG:
                imgs = lc_frames.permute(0, 2, 3, 1).cpu().numpy()  # shape -> (2, H, W, C)
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                for i in range(2):
                    axes[i].imshow(imgs[i])
                    axes[i].axis('off')
                plt.tight_layout()
                plt.title("Loop Closure Frames. Left: Query Frame, Right: Retrieved Frame")
                plt.show()

        # For every non-root ordinary submap, choose one historical reference
        # and run exactly one joint two-frame VGGT scale measurement.
        if self.enable_scale_anchor and self.map.non_lc_submap_ids:
            try:
                predictions["joint_scale_anchor"] = self._compute_joint_anchor(
                    model, predictions, new_submap
                )
            except ValueError as error:
                # Invalid/insufficient trusted geometry is a rejected
                # measurement, not a reason to terminate the SLAM run.
                reason = f"anchor_measurement_invalid:{error}"
                predictions["joint_scale_anchor"] = {
                    "accepted": False,
                    "gate_reasons": [reason],
                }
                print(
                    f"[scale-anchor-gate] submap={new_submap.get_id()} "
                    f"accepted=False reasons={[reason]}"
                )

        predictions["detected_loops"] = detected_loops
        predictions["loop_sequence_verification"] = loop_sequence_verification
        
        if predictions_lc is not None:
            extrinsic_lc, intrinsic_lc = pose_encoding_to_extri_intri(predictions_lc["pose_enc"], retrieved_frames[0].shape[-2:])
            predictions["extrinsic_lc"] = extrinsic_lc
            predictions["intrinsic_lc"] = intrinsic_lc
            predictions["depth_lc"] = predictions_lc["depth"]
            predictions["depth_conf_lc"] = predictions_lc["depth_conf"]

            
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor) and key != "target_tokens":
                predictions[key] = predictions[key].float().cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy
    
        if predictions_lc is not None:
            predictions["frames_lc"] = lc_frames[0:2,...]
            print(loop_closure_frame_names)
            predictions["frames_lc_names"] = loop_closure_frame_names

        return predictions
