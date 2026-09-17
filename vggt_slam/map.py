import os
import numpy as np
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from vggt_slam.slam_utils import decompose_camera, cosine_similarity

class GraphMap:
    def __init__(self):
        self.submaps = dict()
        self.rectifying_H_mats = []
        self.non_lc_submap_ids = []
    
    def get_num_submaps(self):
        return len(self.submaps)

    def add_submap(self, submap):
        submap_id = submap.get_id()
        self.submaps[submap_id] = submap
        if not submap.get_lc_status():
            self.non_lc_submap_ids.append(submap_id)
    
    def get_largest_key(self, ignore_loop_closure_submaps=False):
        """
        Get the largest key of the first node of any submap.
        Return: The largest key, or None if the dictionary is empty.
        """
        if len(self.submaps) == 0:
            return None
        if ignore_loop_closure_submaps:
            non_lc_keys = [key for key, submap in self.submaps.items() if not submap.get_lc_status()]
            return max(non_lc_keys)
        return max(self.submaps.keys())
    
    def get_submap(self, id):
        return self.submaps[id]

    def get_latest_submap(self, ignore_loop_closure_submaps=False):
        return self.get_submap(self.get_largest_key(ignore_loop_closure_submaps))

    def retrieve_best_semantic_frame(self, query_text_vector):
        overall_best_score = 0.0
        overall_best_submap_id = 0
        overall_best_frame_index = 0
        # search for best image to target image
        sorted_keys = sorted(self.submaps.keys())
        for index, submap_key in enumerate(sorted_keys):
            submap = self.submaps[submap_key]
            if submap.get_lc_status():
                continue
            submap_embeddings = submap.get_all_semantic_vectors()
            scores = []
            for index, embedding in enumerate(submap_embeddings):
                score = cosine_similarity(embedding, query_text_vector)
                scores.append(score)
            
            best_score_id = np.argmax(scores)
            best_score = scores[best_score_id]

            if best_score > overall_best_score:
                overall_best_score = best_score
                overall_best_submap_id = submap_key
                overall_best_frame_index = best_score_id

        return overall_best_score, overall_best_submap_id, overall_best_frame_index
    
    def retrieve_best_score_frame(
        self,
        query_vector,
        current_submap_id,
        ignore_last_submap=True,
        min_submap_gap=None,
    ):
        """Return the closest historical frame outside the temporal exclusion.

        ``min_submap_gap`` is measured in ordinary-submap ordinal positions,
        because submap IDs are frame offsets rather than consecutive numbers.
        A new submap is treated as the next ordinary submap, so a gap of four
        excludes the three most recent ordinary submaps.  ``None`` preserves the
        legacy ``ignore_last_submap`` behaviour for callers that do not opt in.
        """
        overall_best_score = 1000
        overall_best_submap_id = 0
        overall_best_frame_index = 0
        # search for best image to target image
        sorted_keys = sorted(self.submaps.keys())
        ordinary_keys = sorted(self.non_lc_submap_ids)
        current_ordinal = (
            ordinary_keys.index(current_submap_id)
            if current_submap_id in ordinary_keys
            else len(ordinary_keys)
        )
        for index, submap_key in enumerate(sorted_keys):
            if submap_key == current_submap_id:
                continue

            if min_submap_gap is not None and submap_key in ordinary_keys:
                historical_ordinal = ordinary_keys.index(submap_key)
                if abs(current_ordinal - historical_ordinal) < min_submap_gap:
                    continue
            elif self.non_lc_submap_ids and ignore_last_submap and submap_key == self.non_lc_submap_ids[-1]:
                continue

            submap = self.submaps[submap_key]
            if submap.get_lc_status():
                continue
            submap_embeddings = submap.get_all_retrieval_vectors()
            scores = []
            for index, embedding in enumerate(submap_embeddings):
                score = torch.linalg.norm(embedding-query_vector)
                # score = embedding @ query_vector.t()
                scores.append(score.item())

            # for now assume we can only have at most one loop closure per submap

            best_score_id = np.argmin(scores)
            best_score = scores[best_score_id]

            if best_score < overall_best_score:
                overall_best_score = best_score
                overall_best_submap_id = submap_key
                overall_best_frame_index = best_score_id

        return overall_best_score, overall_best_submap_id, overall_best_frame_index

    def get_frames_from_loops(self, loops):
        frames = []
        for detected_loop in loops:
            frames.append(self.submaps[detected_loop.detected_submap_id].get_frame_at_index(detected_loop.detected_submap_frame))
        return frames
    
    def get_submaps(self):
        return self.submaps.values()

    def ordered_submaps_by_key(self):
        for k in sorted(self.submaps):
            yield self.submaps[k]
    
    def get_all_homographies(self, graph):
        homographies = []
        for submap in self.ordered_submaps_by_key():
            for pose_num in range(len(submap.poses)):
                id = int(submap.get_id() + pose_num)
                homographies.append(graph.get_homography(id))
        return np.stack(homographies)

    def get_all_cam_matricies(self, graph, give_camera_mat):
        cam_mats = []
        for submap in self.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue
            poses = submap.get_all_poses_world(graph, give_camera_mat=give_camera_mat)
            cam_mats.append(poses)
        return np.vstack(cam_mats)

    def write_poses_to_file(self, file_name, graph, give_camera_mat=False, kitti_format=False):
        all_poses = self.get_all_cam_matricies(give_camera_mat=True, graph=graph)
        with open(file_name, "w") as f:

            if self.rectifying_H_mats:
                assert len(self.rectifying_H_mats) == len(all_poses), "Number of rectifying mats and number of poses do not match"
                print("Using rectifying homographies when writing poses to file.")
            count = 0
            for submap_index, submap in enumerate(self.ordered_submaps_by_key()):
                if submap.get_lc_status():
                    continue
                frame_ids = submap.get_frame_ids()
                print(frame_ids)
                for frame_index, frame_id in enumerate(frame_ids):
                    pose = all_poses[count]
                    K, rotation_matrix, t, scale = decompose_camera(pose)
                    # print("Decomposed K:\n", K)
                    count += 1
                    x, y, z = t
                    if kitti_format:
                        pose_matrix = np.eye(4)
                        pose_matrix[:3, :3] = rotation_matrix
                        pose_matrix[:3, 3] = t
                        output = pose_matrix.flatten()[:-4]
                        output = np.array([float(frame_id), *output])
                    else:
                        quaternion = R.from_matrix(rotation_matrix).as_quat() # x, y, z, w
                        output = np.array([float(frame_id), x, y, z, *quaternion])
                    f.write(" ".join(f"{v:.8f}" for v in output) + "\n")    

    def write_points_to_file(self, graph, file_name, voxel_size=0.05):
        """Write the fused RGB cloud, downsampled by default for portability."""
        pcd_all = []
        colors_all = []
        for submap in self.ordered_submaps_by_key():
            pcd = submap.get_points_in_world_frame(graph)
            pcd = pcd.reshape(-1, 3)
            colors = np.asarray(submap.get_points_colors()).reshape(-1, 3)
            valid = np.all(np.isfinite(pcd), axis=1) & np.all(np.isfinite(colors), axis=1)
            pcd, colors = pcd[valid], colors[valid]
            if colors.size and colors.max() > 1.0:
                colors = colors / 255.0

            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
            cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
            if voxel_size is not None and voxel_size > 0:
                cloud = cloud.voxel_down_sample(float(voxel_size))
            pcd_all.append(np.asarray(cloud.points))
            colors_all.append(np.asarray(cloud.colors))
        pcd_all = np.concatenate(pcd_all, axis=0)
        colors_all = np.concatenate(colors_all, axis=0)
        pcd_all = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_all))
        pcd_all.colors = o3d.utility.Vector3dVector(colors_all)
        if voxel_size is not None and voxel_size > 0:
            # A second pass removes duplicate voxels along submap boundaries.
            pcd_all = pcd_all.voxel_down_sample(float(voxel_size))
        if not o3d.io.write_point_cloud(file_name, pcd_all, write_ascii=False):
            raise IOError(f"Failed to write point cloud to {file_name}")
        print(
            f"[output] point_cloud={file_name} points={len(pcd_all.points)} "
            f"voxel_size={float(voxel_size) if voxel_size else 0.0:.8g}"
        )

    @staticmethod
    def _trajectory_sphere_radius(centers, requested_radius=None):
        if requested_radius is not None and requested_radius > 0:
            return float(requested_radius)
        centers = np.asarray(centers, dtype=float)
        if len(centers) < 2:
            return 0.01
        steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        steps = steps[np.isfinite(steps) & (steps > np.finfo(float).eps)]
        extent = np.ptp(centers, axis=0)
        diagonal = float(np.linalg.norm(extent))
        if steps.size:
            radius = 0.20 * float(np.median(steps))
        else:
            radius = diagonal / 500.0 if diagonal > 0 else 0.01
        if diagonal > 0:
            radius = float(np.clip(radius, diagonal / 5000.0, diagonal / 100.0))
        return max(radius, np.finfo(float).eps)

    def write_trajectory_spheres_to_file(
        self,
        graph,
        file_name,
        sphere_radius=None,
        points_per_sphere=96,
    ):
        """Export camera centers as colored point-sampled spheres for CloudCompare."""
        poses = self.get_all_cam_matricies(give_camera_mat=True, graph=graph)
        centers = []
        for pose in poses:
            _, _, translation, _ = decompose_camera(pose)
            centers.append(translation)
        centers = np.asarray(centers, dtype=float)
        valid = np.all(np.isfinite(centers), axis=1)
        centers = centers[valid]
        if not len(centers):
            raise ValueError("Cannot export an empty trajectory")

        points_per_sphere = max(int(points_per_sphere), 8)
        radius = self._trajectory_sphere_radius(centers, sphere_radius)
        # Fibonacci sphere: a deterministic, nearly uniform surface sampling.
        sample_index = np.arange(points_per_sphere, dtype=float)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        z = 1.0 - 2.0 * (sample_index + 0.5) / points_per_sphere
        radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        unit_sphere = np.column_stack((
            radial * np.cos(golden_angle * sample_index),
            radial * np.sin(golden_angle * sample_index),
            z,
        ))
        # Keep the sampled cloud's centroid exactly on the camera center.
        unit_sphere -= unit_sphere.mean(axis=0, keepdims=True)
        sphere_points = (centers[:, None, :] + radius * unit_sphere[None, :, :]).reshape(-1, 3)

        # Blue -> cyan -> yellow -> red makes trajectory order visible.
        progress = np.linspace(0.0, 1.0, len(centers))
        colors = np.column_stack((
            np.clip(2.0 * progress, 0.0, 1.0),
            1.0 - np.abs(2.0 * progress - 1.0),
            np.clip(2.0 * (1.0 - progress), 0.0, 1.0),
        ))
        sphere_colors = np.repeat(colors, points_per_sphere, axis=0)
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sphere_points))
        cloud.colors = o3d.utility.Vector3dVector(sphere_colors)
        if not o3d.io.write_point_cloud(file_name, cloud, write_ascii=False):
            raise IOError(f"Failed to write trajectory spheres to {file_name}")
        print(
            f"[output] trajectory_spheres={file_name} poses={len(centers)} "
            f"points={len(sphere_points)} radius={radius:.8g} "
            f"points_per_sphere={points_per_sphere}"
        )

    def write_submap_previews(self, graph, output_dir, max_points=150000):
        """Write one RGB orthographic diagnostic PNG per ordinary submap."""
        os.makedirs(output_dir, exist_ok=True)
        projections = ((0, 1, "X", "Y"), (0, 2, "X", "Z"), (1, 2, "Y", "Z"))
        rng = np.random.default_rng(20260722)

        for submap_index, submap in enumerate(self.ordered_submaps_by_key()):
            if submap.get_lc_status():
                continue
            points = np.asarray(submap.get_points_in_world_frame(graph), dtype=float).reshape(-1, 3)
            colors = np.asarray(submap.get_points_colors(), dtype=float).reshape(-1, 3)
            valid = np.all(np.isfinite(points), axis=1) & np.all(np.isfinite(colors), axis=1)
            points, colors = points[valid], colors[valid]
            if not len(points):
                print(f"[submap-preview] skipping empty submap {submap.get_id()}")
                continue
            if colors.max() > 1.0:
                colors = colors / 255.0

            full_count = len(points)
            if full_count > max_points:
                chosen = rng.choice(full_count, size=max_points, replace=False)
                draw_points, draw_colors = points[chosen], colors[chosen]
            else:
                draw_points, draw_colors = points, colors

            # Center only the displayed coordinates. This preserves metric
            # dimensions and global orientation while making every submap easy
            # to inspect independently.
            center = np.median(draw_points, axis=0)
            draw_points = draw_points - center
            low, high = np.percentile(points, (1.0, 99.0), axis=0)
            robust_extent = high - low

            fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")
            for axis, (horizontal, vertical, xlabel, ylabel) in zip(axes, projections):
                x = draw_points[:, horizontal]
                y = draw_points[:, vertical]
                axis.scatter(x, y, s=0.15, c=np.clip(draw_colors, 0.0, 1.0), linewidths=0,
                             alpha=0.75, rasterized=True)
                limit = np.percentile(np.maximum(np.abs(x), np.abs(y)), 99.5)
                limit = max(float(limit), 1e-6)
                axis.set_xlim(-limit, limit)
                axis.set_ylim(-limit, limit)
                axis.set_aspect("equal", adjustable="box")
                axis.set_xlabel(f"global {xlabel}, centered")
                axis.set_ylabel(f"global {ylabel}, centered")
                axis.grid(alpha=0.15)

            scale = float(getattr(submap, "applied_scale", 1.0))
            fig.suptitle(
                f"submap {submap_index:02d} (node {submap.get_id()}) | "
                f"S={scale:.6g} | points={full_count:,} | "
                f"robust XYZ extent={robust_extent[0]:.3g}, "
                f"{robust_extent[1]:.3g}, {robust_extent[2]:.3g}"
            )
            fig.tight_layout()
            path = os.path.join(output_dir, f"submap_{submap_index:02d}_node_{submap.get_id()}.png")
            fig.savefig(path, dpi=140)
            plt.close(fig)
            print(f"[submap-preview] wrote {path}")
