import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import heapq
from typing import NamedTuple
import torchvision.transforms as T
import os

from salad.eval import load_model # load salad

device = 'cuda'

LOOP_SEQUENCE_RADIUS = 1
LOOP_SEQUENCE_MIN_FRAMES = 3
LOOP_SEQUENCE_MAX_RELATIVE_POSITION_RMSE = 0.20
LOOP_SEQUENCE_MAX_ROTATION_RMSE_DEG = 10.0
LOOP_SEQUENCE_MATCH_RADIUS = 2

tensor_transform = T.ToPILImage()
denormalize = T.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])


def centered_window_indices(center, count, radius=LOOP_SEQUENCE_RADIUS):
    """Return a fixed-width window shifted inward at sequence boundaries."""
    if count <= 0 or radius < 0 or not 0 <= center < count:
        raise ValueError("Invalid centered-window request")
    width = min(2 * radius + 1, count)
    start = min(max(center - radius, 0), count - width)
    return list(range(start, start + width))


def _camera_centers(world_to_camera):
    world_to_camera = np.asarray(world_to_camera, dtype=float)
    return -np.einsum(
        "nij,nj->ni",
        world_to_camera[:, :3, :3].transpose(0, 2, 1),
        world_to_camera[:, :3, 3],
    )


def _canonical_world_to_camera(poses):
    """Normalize a pose sequence to VGGT's ``[N, 3, 4]`` layout."""
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError(f"Unexpected world-to-camera pose shape {poses.shape}")
    return poses[:, :3, :4]


def select_sequence_pair_indices(
    query_center,
    query_count,
    reference_center,
    reference_count,
    query_descriptors,
    reference_descriptors,
    radius=LOOP_SEQUENCE_MATCH_RADIUS,
):
    """Choose forward/reverse three-frame matches while preserving the center."""
    query_descriptors = np.asarray(query_descriptors, dtype=float)
    reference_descriptors = np.asarray(reference_descriptors, dtype=float)
    candidates = []
    for direction in (1, -1):
        offsets = [
            offset for offset in range(-radius, radius + 1)
            if offset != 0
            and 0 <= query_center + offset < query_count
            and 0 <= reference_center + direction * offset < reference_count
        ]
        offsets = sorted(offsets, key=lambda value: (abs(value), value))[:2]
        if len(offsets) != 2:
            continue
        offsets.append(0)
        offsets.sort()
        query_indices = [query_center + offset for offset in offsets]
        reference_indices = [
            reference_center + direction * offset for offset in offsets
        ]
        descriptor_cost = float(np.mean([
            np.linalg.norm(
                query_descriptors[query_index]
                - reference_descriptors[reference_index]
            )
            for query_index, reference_index
            in zip(query_indices, reference_indices)
        ]))
        candidates.append((descriptor_cost, direction, query_indices, reference_indices))
    if not candidates:
        raise ValueError("Loop candidate has insufficient neighboring frames")
    descriptor_cost, direction, query_indices, reference_indices = min(
        candidates, key=lambda item: item[0]
    )
    return {
        "query_indices": query_indices,
        "reference_indices": reference_indices,
        "direction": "forward" if direction == 1 else "reverse",
        "descriptor_cost": descriptor_cost,
    }


def _homogeneous_pose(pose):
    result = np.eye(4)
    result[:3, :4] = pose[:3, :4]
    return result


def _project_rotation(matrix):
    u, _, vt = np.linalg.svd(matrix)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    return u @ correction @ vt


def _rotation_error_degrees(left, right):
    cosine = np.clip((np.trace(left.T @ right) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def verify_common_cross_submap_transform(
    query_local_world_to_camera,
    reference_local_world_to_camera,
    independently_predicted_pairs,
    max_relative_position_rmse=LOOP_SEQUENCE_MAX_RELATIVE_POSITION_RMSE,
    max_rotation_rmse_deg=LOOP_SEQUENCE_MAX_ROTATION_RMSE_DEG,
):
    """Require independent frame pairs to imply one cross-submap Sim(3).

    Every two-frame VGGT call has an arbitrary translation scale.  Rotation is
    compared directly.  Translation consistency is tested by fitting one
    cross-submap rotation, scale and translation plus one positive gauge per
    independent VGGT pair.
    """
    query = _canonical_world_to_camera(query_local_world_to_camera)
    reference = _canonical_world_to_camera(reference_local_world_to_camera)
    pairs = np.asarray(independently_predicted_pairs, dtype=float)
    if pairs.ndim == 5 and pairs.shape[1] == 1:
        pairs = pairs[:, 0]
    if (
        query.shape != reference.shape
        or query.shape[0] < LOOP_SEQUENCE_MIN_FRAMES
        or pairs.shape[0] != query.shape[0]
        or pairs.shape[1] != 2
    ):
        raise ValueError("Short-sequence verification requires three independent pairs")
    pairs = np.stack([
        _canonical_world_to_camera(pair) for pair in pairs
    ])
    if not (
        np.all(np.isfinite(query))
        and np.all(np.isfinite(reference))
        and np.all(np.isfinite(pairs))
    ):
        raise ValueError("Short-sequence poses must be finite")

    cross_rotations = []
    measured_directions = []
    pair_relatives = []
    for query_pose, reference_pose, pair in zip(query, reference, pairs):
        pair_relative = (
            _homogeneous_pose(pair[1])
            @ np.linalg.inv(_homogeneous_pose(pair[0]))
        )
        pair_relatives.append(pair_relative)
        cross_rotations.append(_project_rotation(
            reference_pose[:3, :3].T
            @ pair_relative[:3, :3]
            @ query_pose[:3, :3]
        ))
        measured_directions.append(
            reference_pose[:3, :3].T @ pair_relative[:3, 3]
        )

    mean_rotation = _project_rotation(np.sum(cross_rotations, axis=0))
    rotation_errors = np.array([
        _rotation_error_degrees(mean_rotation, rotation)
        for rotation in cross_rotations
    ])
    rotation_rmse_deg = float(np.sqrt(np.mean(rotation_errors ** 2)))

    query_centers = _camera_centers(query)
    reference_centers = _camera_centers(reference)
    pair_count = query.shape[0]
    system = np.zeros((3 * pair_count, 4 + pair_count), dtype=float)
    target = reference_centers.reshape(-1)
    for index, (query_center, direction) in enumerate(zip(
        query_centers, measured_directions
    )):
        row = slice(3 * index, 3 * index + 3)
        system[row, :3] = np.eye(3)
        system[row, 3] = mean_rotation @ query_center
        system[row, 4 + index] = -direction
    solution, _, rank, singular_values = np.linalg.lstsq(system, target, rcond=None)
    residual = (system @ solution - target).reshape(-1, 3)
    position_rmse = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))
    query_spread = float(np.sqrt(np.mean(np.sum(
        (query_centers - query_centers.mean(axis=0)) ** 2, axis=1
    ))))
    reference_spread = float(np.sqrt(np.mean(np.sum(
        (reference_centers - reference_centers.mean(axis=0)) ** 2, axis=1
    ))))
    cross_scale = float(solution[3])
    pair_scales = np.asarray(solution[4:], dtype=float)
    direct_relative_transforms = []
    for query_pose, reference_pose, pair_relative, pair_scale in zip(
        query, reference, pair_relatives, pair_scales
    ):
        # This is T_reference_camera_from_query_camera.  The shared Sim(3)
        # supplies a single cross-submap rotation and estimates each two-frame
        # VGGT translation gauge in the reference-submap coordinate unit.
        relative = np.eye(4)
        relative[:3, :3] = _project_rotation(
            reference_pose[:3, :3]
            @ mean_rotation
            @ query_pose[:3, :3].T
        )
        relative[:3, 3] = pair_scale * pair_relative[:3, 3]
        direct_relative_transforms.append(relative.tolist())
    normalizer = max(reference_spread, abs(cross_scale) * query_spread, 1e-12)
    relative_position_rmse = position_rmse / normalizer
    # A proper cross-submap Sim(3) requires positive scale even for reverse
    # traversal. A least-norm solution of a rank-deficient system does not
    # identify the translation gauges; a small residual is not sufficient.
    positive_cross_scale = bool(cross_scale > 0.0)
    positive_pair_scales = bool(np.all(pair_scales > 0.0))
    required_rank = system.shape[1]
    full_rank = int(rank) == required_rank
    finite_solution = bool(
        np.all(np.isfinite(solution))
        and np.all(np.isfinite(direct_relative_transforms))
        and np.isfinite(relative_position_rmse)
        and np.isfinite(rotation_rmse_deg)
    )
    rejection_reasons = []
    if not finite_solution:
        rejection_reasons.append("nonfinite_solution")
    if not full_rank:
        rejection_reasons.append("rank_deficient_geometry")
    if not positive_cross_scale:
        rejection_reasons.append("nonpositive_cross_scale")
    if not positive_pair_scales:
        rejection_reasons.append("nonpositive_pair_gauge")
    if relative_position_rmse > max_relative_position_rmse:
        rejection_reasons.append("position_inconsistent")
    if rotation_rmse_deg > max_rotation_rmse_deg:
        rejection_reasons.append("rotation_inconsistent")
    return {
        "accepted": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
        "relative_position_rmse": relative_position_rmse,
        "rotation_rmse_deg": rotation_rmse_deg,
        "max_relative_position_rmse": float(max_relative_position_rmse),
        "max_rotation_rmse_deg": float(max_rotation_rmse_deg),
        "cross_scale": cross_scale,
        "cross_rotation": mean_rotation.tolist(),
        "cross_translation": solution[:3].tolist(),
        "pair_scales": pair_scales.tolist(),
        "direct_relative_transforms": direct_relative_transforms,
        "positive_scales": positive_pair_scales,
        "positive_cross_scale": positive_cross_scale,
        "finite_solution": finite_solution,
        "linear_rank": int(rank),
        "linear_required_rank": int(required_rank),
        "linear_singular_values": singular_values.tolist(),
    }

def input_transform(image_size=None):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    transform_list = [T.ToTensor(), T.Normalize(mean=MEAN, std=STD)]
    if image_size:
        transform_list.insert(0, T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR))
    return T.Compose(transform_list)

class LoopMatch(NamedTuple):
    similarity_score: float
    query_submap_id: int
    query_submap_frame: int
    detected_submap_id: int
    detected_submap_frame: int

class LoopMatchQueue:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.heap = []  # Simulated max-heap by negating scores

    def add(self, match: LoopMatch):
        # Negate similarity_score to turn min-heap into max-heap
        item = (-match.similarity_score, match)
        # item = (-match.detected_submap_id, match)
        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, item)
        else:
            # Push new element and remove the largest (i.e., smallest negated)
            heapq.heappushpop(self.heap, item)

    def get_matches(self):
        """Return sorted list of matches (lowest value first)"""
        return [match for _, match in sorted(self.heap, reverse=True)]
        

class ImageRetrieval:
    def __init__(self, input_size=224):

        ckpt_pth = os.path.join(torch.hub.get_dir(), "checkpoints/dino_salad.ckpt")
        self.model = load_model(ckpt_pth)
        self.model.eval()
        self.transform = input_transform((input_size, input_size))

    def get_single_embeding(self, cv_img):
        with torch.no_grad():
            pil_img = self.transform(tensor_transform(cv_img))
            return self.model(pil_img.to(device))

    def get_batch_descriptors(self, imgs):
        # Expecting imgs to be a batch of images (B, C, H, W)
        with torch.no_grad():
            pil_imgs = [tensor_transform(img) for img in imgs]  # Convert each tensor to PIL Image
            imgs = torch.stack([self.transform(img) for img in pil_imgs])  # Apply transform and stack
            return self.model(imgs.to(device))
    
    def get_all_submap_embeddings(self, submap):
        # Frames is np array of shape (S, 3, H, W)
        frames = submap.get_all_frames()
        return self.get_batch_descriptors(frames)

    def find_loop_closures(
        self,
        map,
        submap,
        max_similarity_thres=0.80,
        max_loop_closures=0,
        min_submap_gap=4,
    ):
        """Find loop candidates outside the three-submap temporal neighborhood."""
        matches_queue = LoopMatchQueue(max_size=max_loop_closures)
        query_id = 0
        for query_vector in submap.get_all_retrieval_vectors():
            best_score, best_submap_id, best_frame_id = map.retrieve_best_score_frame(
                query_vector,
                submap.get_id(),
                ignore_last_submap=True,
                min_submap_gap=min_submap_gap,
            )
            if best_score < max_similarity_thres:
                new_match_data = LoopMatch(best_score, submap.get_id(), query_id, best_submap_id, best_frame_id)
                matches_queue.add(new_match_data)
            query_id += 1
        
        return matches_queue.get_matches()
