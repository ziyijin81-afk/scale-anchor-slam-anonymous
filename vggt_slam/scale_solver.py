"""Scale measurements and the independent one-dimensional scale factor graph."""

from dataclasses import dataclass
from datetime import datetime, timezone
import time
import warnings

import numpy as np
from scipy.optimize import least_squares

try:
    import gtsam
except ImportError:  # Unit-test/minimal environments may not carry the SLAM backend.
    gtsam = None


def debug_visualize(pcd1_points, pcd2_points):
    # Keep the optional dependency out of normal/test imports.
    import open3d as o3d

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pcd1_points)
    pcd1.paint_uniform_color([1, 0, 0])
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pcd2_points)
    pcd2.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([pcd1, pcd2], window_name="Pairwise Point Clouds")


def estimate_scale_pairwise(X, Y, DEBUG=False):
    """Original VGGT-SLAM2 overlap measurement; do not change this formula."""
    assert X.shape == Y.shape
    x_dists = np.linalg.norm(X, axis=1)
    y_dists = np.linalg.norm(Y, axis=1)
    scales = y_dists / x_dists
    scale = np.median(scales)

    if DEBUG:
        debug_visualize(X * scale, Y)

    return scale, None


def robust_sample_statistics(samples):
    samples = np.asarray(samples, dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples) & (samples > 0)]
    if samples.size == 0:
        return {"median": np.nan, "mad": np.inf, "count": 0, "outlier_ratio": 1.0}
    median = float(np.median(samples))
    deviations = np.abs(samples - median)
    mad = float(np.median(deviations))
    cutoff = 3.0 * 1.4826 * max(mad, np.finfo(float).eps)
    return {
        "median": median,
        "mad": mad,
        "count": int(samples.size),
        "outlier_ratio": float(np.mean(deviations > cutoff)),
    }


def camera_points_from_depth(depth, intrinsic):
    """Back-project depth into camera coordinates (never world-point norms)."""
    depth = np.asarray(depth, dtype=float).squeeze()
    K = np.asarray(intrinsic, dtype=float)
    height, width = depth.shape
    yy, xx = np.indices((height, width), dtype=float)
    z = depth
    x = (xx - K[0, 2]) * z / K[0, 0]
    y = (yy - K[1, 2]) * z / K[1, 1]
    return np.stack((x, y, z), axis=-1)


def frame_confidence_threshold(confidence, percentile):
    """Return a confidence cutoff calibrated only on one prediction frame."""
    confidence = np.asarray(confidence, dtype=float)
    finite = confidence[np.isfinite(confidence)]
    if finite.size == 0:
        raise ValueError("Cannot compute a confidence threshold from no finite values")
    # same_frame_scale_consistency deliberately uses a strict `>` comparison.
    # Values tied exactly at the 25th percentile are excluded; if that leaves
    # too few shared pixels, the caller rejects the Anchor as unobservable.
    return float(np.percentile(finite, float(percentile)))


def confidence_selection_mask(confidence, confidence_threshold):
    """Select trusted pixels, falling back when confidence has no ranking.

    VGGT confidence can saturate to one value (for example all ``1.0`` in
    bfloat16).  A strict percentile comparison would then reject every pixel.
    In that specific degenerate case confidence carries no ordering
    information, so retain every finite-confidence pixel and let the geometry,
    shared-pixel minimum, and consistency checks decide whether the Anchor is
    usable.
    """
    confidence = np.asarray(confidence, dtype=float)
    finite = np.isfinite(confidence)
    values = confidence[finite]
    if values.size == 0:
        return np.zeros(confidence.shape, dtype=bool), True
    degenerate = bool(np.ptp(values) == 0.0)
    if degenerate:
        return finite, True
    return finite & (confidence > float(confidence_threshold)), False


def anchor_has_minimum_points(root_count, current_count, min_points):
    """Hard observability condition that is independent of the optional gate."""
    return (
        int(root_count) >= int(min_points)
        and int(current_count) >= int(min_points)
    )


def frame_scale_statistics(camera_points, confidence, confidence_threshold):
    points = np.asarray(camera_points, dtype=float).reshape(-1, 3)
    conf = np.asarray(confidence, dtype=float).reshape(-1)
    mask, _ = confidence_selection_mask(conf, confidence_threshold)
    mask &= np.all(np.isfinite(points), axis=1)
    mask &= points[:, 2] > 0
    radii = np.linalg.norm(points[mask], axis=1)
    return robust_sample_statistics(radii)


def same_frame_scale_consistency(
    original_points,
    original_confidence,
    original_threshold,
    joint_points,
    joint_confidence,
    joint_threshold,
    min_points=500,
    max_relative_mad=0.10,
    max_outlier_ratio=0.25,
    use_confidence=True,
):
    """Test whether one frame changed by approximately one global scale.

    Only pixels trusted by both inferences are used.  This function gates an
    anchor measurement; it does not replace or alter the anchor formula.
    """
    original_points = np.asarray(original_points, dtype=float)
    joint_points = np.asarray(joint_points, dtype=float)
    original_confidence = np.asarray(original_confidence, dtype=float)
    joint_confidence = np.asarray(joint_confidence, dtype=float)
    if original_points.shape != joint_points.shape:
        raise ValueError(
            f"Original/joint point shapes differ: "
            f"{original_points.shape} vs {joint_points.shape}"
        )
    if original_confidence.shape != joint_confidence.shape:
        raise ValueError(
            f"Original/joint confidence shapes differ: "
            f"{original_confidence.shape} vs {joint_confidence.shape}"
        )

    original_radii = np.linalg.norm(original_points, axis=-1)
    joint_radii = np.linalg.norm(joint_points, axis=-1)
    if use_confidence:
        original_mask, original_confidence_degenerate = confidence_selection_mask(
            original_confidence, original_threshold
        )
        joint_mask, joint_confidence_degenerate = confidence_selection_mask(
            joint_confidence, joint_threshold
        )
        valid = original_mask & joint_mask
    else:
        # Diagnostic/ablation mode: retain every pixel whose geometry is
        # valid in both predictions.  Never admit NaN, non-positive depth, or
        # zero-radius samples even when confidence filtering is disabled.
        valid = np.ones(original_confidence.shape, dtype=bool)
        original_confidence_degenerate = False
        joint_confidence_degenerate = False
    valid &= np.all(np.isfinite(original_points), axis=-1)
    valid &= np.all(np.isfinite(joint_points), axis=-1)
    valid &= original_points[..., 2] > 0
    valid &= joint_points[..., 2] > 0
    valid &= original_radii > np.finfo(float).eps
    valid &= joint_radii > np.finfo(float).eps

    # Keep both scale statistics on exactly the same pixel support.  These
    # paired statistics are used as A_j/a_j by the joint anchor formula; using
    # independent confidence masks would compare different scene content.
    original_scale_stats = robust_sample_statistics(original_radii[valid])
    joint_scale_stats = robust_sample_statistics(joint_radii[valid])
    stats = robust_sample_statistics(original_radii[valid] / joint_radii[valid])
    median = float(stats["median"])
    relative_mad = (
        np.inf if not np.isfinite(median) or median <= 0
        else float(stats["mad"]) / median
    )
    reasons = []
    if stats["count"] < int(min_points):
        reasons.append(f"valid_points<{int(min_points)}")
    if not np.isfinite(relative_mad) or relative_mad > float(max_relative_mad):
        reasons.append(f"relative_MAD>{float(max_relative_mad):.6g}")
    if stats["outlier_ratio"] > float(max_outlier_ratio):
        reasons.append(f"outlier_ratio>{float(max_outlier_ratio):.6g}")
    return {
        **stats,
        "scale_ratio": median,
        "relative_mad": relative_mad,
        "original_scale_stats": original_scale_stats,
        "joint_scale_stats": joint_scale_stats,
        "original_confidence_degenerate": original_confidence_degenerate,
        "joint_confidence_degenerate": joint_confidence_degenerate,
        "accepted": not reasons,
        "reasons": reasons,
    }


def measurement_sigma(stats, confidence_values=None, base_sigma=0.03):
    """Turn sample support, confidence, MAD and outliers into a scale sigma."""
    median = max(float(stats.get("median", 1.0)), np.finfo(float).eps)
    relative_mad = float(stats.get("mad", np.inf)) / median
    count = max(int(stats.get("count", 0)), 1)
    support_penalty = np.sqrt(100.0 / min(count, 10000))
    outlier_penalty = 1.0 + 3.0 * float(stats.get("outlier_ratio", 0.0))
    conf_penalty = 1.0
    if confidence_values is not None:
        finite = np.asarray(confidence_values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            conf_penalty = 1.0 + 1.0 / max(float(np.median(finite)), 1e-3)
    sigma = base_sigma * (1.0 + 4.0 * relative_mad) * support_penalty * outlier_penalty * conf_penalty
    return float(np.clip(sigma, 1e-4, 2.0))


def anchor_scale(A_0, A_k, a_0, a_k):
    values = np.asarray([A_0, A_k, a_0, a_k], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"Scale statistics must be finite and positive, got {values}")
    return float(A_0 * a_k / (a_0 * A_k))


@dataclass
class ScaleFactorRecord:
    factor_type: str
    submap_i: int
    submap_j: int
    measurement: float
    sigma: float
    factor_index: int
    measurement_version: int
    measurement_timestamp: float
    backend_factor_index: int = None


class GtsamISAM2ScaleBackend:
    """Incremental scalar-scale backend using real iSAM2 factor replacement."""

    def __init__(self, huber_scale=1.345):
        if gtsam is None:
            raise RuntimeError("GTSAM is required for the iSAM2 scale backend")
        params = gtsam.ISAM2Params()
        # Ordinary-scale chains can span a large numeric range on long
        # sequences.  QR is more tolerant of the resulting ill-conditioning
        # than the default Cholesky elimination while preserving the same
        # variables, factors, and incremental iSAM2 update semantics.
        params.setFactorization("QR")
        params.setRelinearizeThreshold(1e-6)
        params.relinearizeSkip = 1
        params.findUnusedFactorSlots = True
        params.enableDetailedResults = True
        params.evaluateNonlinearError = True
        self.isam = gtsam.ISAM2(params)
        self.huber_scale = float(huber_scale)
        self.initialized_nodes = set()
        self.last_diagnostics = {
            "variables_relinearized": 0,
            "variables_reeliminated": 0,
            "cliques": 0,
        }

    @staticmethod
    def key(submap_id):
        return gtsam.symbol("s", int(submap_id))

    def _noise(self, sigma):
        gaussian = gtsam.noiseModel.Isotropic.Sigma(1, float(sigma))
        huber = gtsam.noiseModel.mEstimator.Huber.Create(self.huber_scale)
        return gtsam.noiseModel.Robust.Create(huber, gaussian)

    def make_factor(self, record):
        measurement = float(record.measurement)
        if record.submap_i == 0:
            keys = [self.key(record.submap_j)]

            def error_func(this, values, jacobians=None):
                scale_j = float(values.atVector(this.keys()[0])[0])
                if jacobians is not None:
                    jacobians[0] = np.array([[1.0]], dtype=float, order="F")
                return np.array([scale_j - measurement], dtype=float)
        else:
            keys = [self.key(record.submap_i), self.key(record.submap_j)]

            def error_func(this, values, jacobians=None):
                scale_i = float(values.atVector(this.keys()[0])[0])
                scale_j = float(values.atVector(this.keys()[1])[0])
                if jacobians is not None:
                    jacobians[0] = np.array([[-measurement]], dtype=float, order="F")
                    jacobians[1] = np.array([[1.0]], dtype=float, order="F")
                return np.array([scale_j - scale_i * measurement], dtype=float)
        return gtsam.CustomFactor(self._noise(record.sigma), keys, error_func)

    def update(self, records, remove_indices, nodes):
        new_factors = gtsam.NonlinearFactorGraph()
        new_values = gtsam.Values()
        newly_initialized_nodes = set()
        for record in records:
            new_factors.add(self.make_factor(record))
            for node_id in (record.submap_i, record.submap_j):
                if (node_id == 0 or node_id in self.initialized_nodes
                        or node_id in newly_initialized_nodes):
                    continue
                new_values.insert(self.key(node_id), np.array([nodes[node_id]], dtype=float))
                newly_initialized_nodes.add(node_id)

        result = self.isam.update(new_factors, new_values, list(remove_indices))
        self.initialized_nodes.update(newly_initialized_nodes)
        backend_indices = [int(index) for index in result.getNewFactorsIndices()]
        estimate = self.isam.calculateEstimate()
        optimized = {
            node_id: float(estimate.atVector(self.key(node_id))[0])
            for node_id in self.initialized_nodes
        }
        self.last_diagnostics = {
            "variables_relinearized": int(result.getVariablesRelinearized()),
            "variables_reeliminated": int(result.getVariablesReeliminated()),
            "cliques": int(result.getCliques()),
        }
        return optimized, backend_indices, dict(self.last_diagnostics)


_FALLBACK_WARNING_EMITTED = False


class ScaleFactorGraph:
    """Robust ordinary-scale graph with replaceable overlap/anchor factors."""

    def __init__(self, huber_scale=1.345, use_isam2=True):
        global _FALLBACK_WARNING_EMITTED
        self.nodes = {0: 1.0}
        self.applied = {0: 1.0}
        self.factors = {}
        self._next_factor_index = 0
        self.huber_scale = float(huber_scale)
        self.anchor_history = {}
        self.revision = 0
        self.last_optimized_at = None
        self.last_before = dict(self.nodes)
        self.last_success = True
        self.last_backend_diagnostics = {}
        self._pending_backend_updates = {}
        if use_isam2 and gtsam is not None:
            self.isam2_backend = GtsamISAM2ScaleBackend(huber_scale)
            self.backend_name = "gtsam-isam2"
        else:
            self.isam2_backend = None
            self.backend_name = "scipy-batch-fallback"
            if use_isam2 and not _FALLBACK_WARNING_EMITTED:
                warnings.warn(
                    "GTSAM is unavailable; scale graph is using the non-incremental SciPy fallback. "
                    "Install the project's gtsam-develop dependency for iSAM2.",
                    RuntimeWarning,
                )
                _FALLBACK_WARNING_EMITTED = True

    @staticmethod
    def _factor_key(factor_type, submap_i, submap_j):
        return str(factor_type), int(submap_i), int(submap_j)

    def ensure_node(self, submap_id, initial=None):
        submap_id = int(submap_id)
        if submap_id == 0:
            self.nodes[0] = 1.0
            self.applied[0] = 1.0
            return
        if submap_id not in self.nodes:
            self.nodes[submap_id] = max(float(1.0 if initial is None else initial), 1e-8)
            self.applied[submap_id] = 1.0

    def update_factor(self, factor_type, submap_i, submap_j, measurement, sigma, timestamp=None):
        measurement = float(measurement)
        sigma = float(sigma)
        if not np.isfinite(measurement) or measurement <= 0:
            raise ValueError("Scale factor measurements must be finite and positive")
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("Scale factor sigma must be finite and positive")
        self.ensure_node(submap_i)
        initial = self.nodes[submap_i] * measurement
        self.ensure_node(submap_j, initial=initial)
        key = self._factor_key(factor_type, submap_i, submap_j)
        old = self.factors.get(key)
        pending_old = self._pending_backend_updates.get(key)
        version = 1 if old is None else old.measurement_version + 1
        replaced_index = None if old is None else old.factor_index
        if pending_old is not None:
            replaced_backend_index = pending_old[1]
        else:
            replaced_backend_index = None if old is None else old.backend_factor_index
        record = ScaleFactorRecord(
            factor_type=str(factor_type), submap_i=int(submap_i), submap_j=int(submap_j),
            measurement=measurement, sigma=sigma, factor_index=self._next_factor_index,
            measurement_version=version,
            measurement_timestamp=float(time.time() if timestamp is None else timestamp),
        )
        self._next_factor_index += 1
        self.factors[key] = record
        self._pending_backend_updates[key] = (record, replaced_backend_index)
        stamp = datetime.fromtimestamp(record.measurement_timestamp, tz=timezone.utc).isoformat()
        print(f"[scale-graph] factor={factor_type} edge=({submap_i},{submap_j}) "
              f"measurement={measurement:.8g} sigma={sigma:.6g} weight={1.0/sigma**2:.6g} "
              f"factor_index={record.factor_index} replaced_factor_index={replaced_index} "
              f"replaced_backend_factor_index={replaced_backend_index} "
              f"version={version} timestamp={stamp}")
        return record, replaced_index

    def update_overlap(self, prior_id, current_id, measurement, sigma, timestamp=None):
        return self.update_factor("overlap", prior_id, current_id, measurement, sigma, timestamp)

    def update_anchor(
        self,
        current_id,
        measurement,
        sigma,
        timestamp=None,
        reference_id=0,
    ):
        history = self.anchor_history.setdefault(int(current_id), [])
        history.append(float(measurement))
        return self.update_factor(
            "anchor",
            int(reference_id),
            current_id,
            measurement,
            sigma,
            timestamp,
        )

    def _raw_residual(self, factor, values):
        # Scale factors are multiplicative: S_j = S_i * measured_scale.
        # Evaluate them in log space so the same relative scale error receives
        # the same penalty everywhere in the trajectory, independent of the
        # absolute magnitude accumulated by earlier submaps.
        return float(
            np.log(max(values[factor.submap_j], 1e-12))
            - np.log(
                max(
                    values[factor.submap_i] * factor.measurement,
                    1e-12,
                )
            )
        )

    def optimize(self):
        ids = sorted(node_id for node_id in self.nodes if node_id != 0)
        before = dict(self.nodes)
        if not ids or not self.factors:
            self.nodes[0] = 1.0
            return before, dict(self.nodes)
        if self.isam2_backend is not None:
            pending = list(self._pending_backend_updates.values())
            if not pending:
                return before, dict(self.nodes)
            records = [item[0] for item in pending]
            remove_indices = sorted({item[1] for item in pending if item[1] is not None})
            optimized, backend_indices, diagnostics = self.isam2_backend.update(
                records, remove_indices, self.nodes
            )
            if len(backend_indices) != len(records):
                raise RuntimeError(
                    f"iSAM2 returned {len(backend_indices)} factor indices for {len(records)} factors"
                )
            for record, backend_index in zip(records, backend_indices):
                record.backend_factor_index = backend_index
                print(f"[scale-graph] iSAM2 factor_index={record.factor_index} "
                      f"backend_factor_index={backend_index} type={record.factor_type} "
                      f"edge=({record.submap_i},{record.submap_j})")
            if any(not np.isfinite(value) or value <= 0 for value in optimized.values()):
                raise RuntimeError(f"iSAM2 produced a non-positive scale estimate: {optimized}")
            self.nodes.update(optimized)
            self.nodes[0] = 1.0
            self._pending_backend_updates.clear()
            self.last_backend_diagnostics = diagnostics
            self.revision += 1
            self.last_optimized_at = time.time()
            self.last_before = before
            self.last_success = True
            print(f"[scale-graph] backend={self.backend_name} revision={self.revision} "
                  f"before={before} after={self.nodes} diagnostics={diagnostics}")
            for factor in self.factors.values():
                residual = self._raw_residual(factor, self.nodes)
                print(f"[scale-graph] residual backend_factor_index={factor.backend_factor_index} "
                      f"type={factor.factor_type} edge=({factor.submap_i},{factor.submap_j}) "
                      f"residual={residual:.8g} normalized={residual/factor.sigma:.8g}")
            return before, dict(self.nodes)

        index = {node_id: position for position, node_id in enumerate(ids)}

        def unpack(x):
            values = {0: 1.0}
            values.update({node_id: float(x[index[node_id]]) for node_id in ids})
            return values

        factors = list(self.factors.values())

        def residuals(x):
            values = unpack(x)
            return np.asarray([self._raw_residual(f, values) / f.sigma for f in factors])

        x0 = np.asarray([self.nodes[node_id] for node_id in ids], dtype=float)
        result = least_squares(residuals, x0, bounds=(1e-8, np.inf), loss="huber", f_scale=self.huber_scale)
        self.nodes = unpack(result.x)
        self.nodes[0] = 1.0
        self.revision += 1
        self.last_optimized_at = time.time()
        self.last_before = before
        self.last_success = bool(result.success)
        self.last_backend_diagnostics = {"full_batch": True}
        self._pending_backend_updates.clear()
        print(f"[scale-graph] optimized success={result.success} before={before} after={self.nodes}")
        for factor in factors:
            residual = self._raw_residual(factor, self.nodes)
            print(f"[scale-graph] residual factor_index={factor.factor_index} type={factor.factor_type} "
                  f"edge=({factor.submap_i},{factor.submap_j}) residual={residual:.8g} normalized={residual/factor.sigma:.8g}")
        return before, dict(self.nodes)

    def pending_corrections(self, tolerance=1e-10):
        corrections = {}
        for node_id, value in self.nodes.items():
            applied = self.applied.get(node_id, 1.0)
            correction = value / applied
            if abs(correction - 1.0) > tolerance:
                corrections[node_id] = correction
        return corrections

    def mark_applied(self, node_id):
        self.applied[int(node_id)] = self.nodes[int(node_id)]

    def topology(self):
        return [(f.factor_type, f.submap_i, f.submap_j) for f in self.factors.values()]

    def snapshot(self):
        """Serializable live state used by logs, tests and the Viser panel."""
        factors = []
        for factor in sorted(self.factors.values(), key=lambda item: item.factor_index):
            residual = self._raw_residual(factor, self.nodes)
            factors.append({
                "factor_type": factor.factor_type,
                "submap_i": factor.submap_i,
                "submap_j": factor.submap_j,
                "measurement": factor.measurement,
                "sigma": factor.sigma,
                "weight": 1.0 / factor.sigma ** 2,
                "factor_index": factor.factor_index,
                "measurement_version": factor.measurement_version,
                "measurement_timestamp": factor.measurement_timestamp,
                "backend_factor_index": factor.backend_factor_index,
                "residual": residual,
                "normalized_residual": residual / factor.sigma,
            })
        return {
            "revision": self.revision,
            "optimized_at": self.last_optimized_at,
            "success": self.last_success,
            "backend": self.backend_name,
            "backend_diagnostics": dict(self.last_backend_diagnostics),
            "nodes": dict(self.nodes),
            "applied": dict(self.applied),
            "before": dict(self.last_before),
            "factors": factors,
        }
