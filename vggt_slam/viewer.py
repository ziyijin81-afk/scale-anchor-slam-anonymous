import time
import socket
from typing import Dict, List
import numpy as np
import torch
import viser
import viser.transforms as viser_tf


class Viewer:
    def __init__(self, port: int = 8080):
        if not 1 <= port <= 65535:
            raise ValueError(f"Viewer port must be in [1, 65535], got {port}")

        # Fail early with an actionable error instead of letting Viser emit an
        # opaque background-thread bind error.  This commonly happens on DSW
        # when an older SLAM process is still serving a different checkout.
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("0.0.0.0", port))
        except OSError as exc:
            raise RuntimeError(
                f"Viewer port {port} is already in use. Stop the old process "
                "or start this run with --viewer_port <free-port>. "
                f"Inspect the owner with: ss -ltnp | grep ':{port}'"
            ) from exc
        finally:
            probe.close()

        self.port = port
        self.local_url = f"http://127.0.0.1:{port}"
        print(f"Starting viser server on 0.0.0.0:{port}")

        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        print(f"Viser server ready at {self.local_url}")
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

        # --- GUI Elements ---
        self.gui_show_frames = self.server.gui.add_checkbox("Show Cameras", initial_value=True)
        self.gui_show_frames.on_update(self._on_update_show_frames)

        self.gui_show_scale_graph = self.server.gui.add_checkbox(
            "Show Scale Factor Graph", initial_value=True
        )
        self.gui_show_scale_graph.on_update(self._on_update_show_scale_graph)
        with self.server.gui.add_folder("Live Scale Factor Graph"):
            self.gui_scale_graph_status = self.server.gui.add_markdown(
                "*Waiting for the first scale-graph update...*"
            )

        # Add a button to trigger the walkthrough
        self.btn_walkthrough = self.server.gui.add_button("Play Walkthrough")
        self.btn_walkthrough.on_click(lambda _: self.run_walkthrough())

        self.submap_frames: Dict[int, List[viser.FrameHandle]] = {}
        self.submap_frustums: Dict[int, List[viser.CameraFrustumHandle]] = {}
        self.scale_graph_handles = []

        num_rand_colors = 250
        np.random.seed(100)
        self.random_colors = np.random.randint(0, 256, size=(num_rand_colors, 3), dtype=np.uint8)
        self.submap_id_to_color = dict()
        self.obj_id = 0

    def visualize_frames(self, extrinsics: np.ndarray, images_: np.ndarray, submap_id: int) -> None:
        """
        Add camera frames and frustums to the scene for a specific submap.
        extrinsics: (S, 3, 4)
        images_:    (S, 3, H, W) or None to show frustums without the camera images
        """

        if images_ is not None and isinstance(images_, torch.Tensor):
            images_ = images_.cpu().numpy()

        if submap_id not in self.submap_frames:
            next_id = len(self.submap_id_to_color) + 1
            self.submap_id_to_color[submap_id] = next_id
        self.submap_frames[submap_id] = []
        self.submap_frustums[submap_id] = []

        S = extrinsics.shape[0]
        for img_id in range(S):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            frame_name = f"submap_{submap_id}/frame_{img_id}"
            frustum_name = f"{frame_name}/frustum"

            # Add the coordinate frame
            frame_axis = self.server.scene.add_frame(
                frame_name,
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frame_axis.visible = self.gui_show_frames.value
            self.submap_frames[submap_id].append(frame_axis)

            # Convert image and add frustum. When images are not provided, show
            # just the frustum (no image) with a default FOV/aspect to keep the
            # visualization fast.
            fov = np.radians(90)  # default FOV if no images provided
            img = None
            aspect = 1.0
            if images_ is not None:
                img = images_[img_id]
                img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
                h, w = img.shape[:2]
                fy = 1.1 * h
                fov = 2 * np.arctan2(h / 2, fy)
                aspect = w / h

            frustum = self.server.scene.add_camera_frustum(
                frustum_name,
                fov=fov,
                aspect=aspect,
                scale=0.05,
                image=img,
                line_width=3.0,
                color=self.random_colors[self.submap_id_to_color[submap_id]]
            )
            frustum.visible = self.gui_show_frames.value
            self.submap_frustums[submap_id].append(frustum)

    def _on_update_show_frames(self, _) -> None:
        """Toggle visibility of all camera frames and frustums across all submaps."""
        visible = self.gui_show_frames.value
        for frames in self.submap_frames.values():
            for f in frames:
                f.visible = visible
        for frustums in self.submap_frustums.values():
            for fr in frustums:
                fr.visible = visible

    def _on_update_show_scale_graph(self, _) -> None:
        visible = self.gui_show_scale_graph.value
        for handle in self.scale_graph_handles:
            try:
                handle.visible = visible
            except Exception:
                pass

    def visualize_scale_graph(self, snapshot, corrections=None) -> None:
        """Refresh the live scale graph and its numeric GUI table."""
        corrections = corrections or {}
        visible = self.gui_show_scale_graph.value
        for handle in self.scale_graph_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.scale_graph_handles = []

        nodes = snapshot["nodes"]
        node_ids = sorted(nodes)
        # A compact graph strip below the SLAM map. Layout uses submap order,
        # while labels retain actual (possibly non-contiguous) submap IDs.
        positions = {
            node_id: np.array([0.45 * order, -0.8, 0.5], dtype=np.float32)
            for order, node_id in enumerate(node_ids)
        }
        for node_id in node_ids:
            position = positions[node_id]
            node = self.server.scene.add_icosphere(
                f"/scale_factor_graph/nodes/{node_id}",
                radius=0.055,
                color=(50, 205, 90) if node_id == 0 else (255, 170, 30),
                subdivisions=2,
                position=position,
                visible=visible,
            )
            label = self.server.scene.add_label(
                f"/scale_factor_graph/labels/node_{node_id}",
                text=f"S{node_id} = {nodes[node_id]:.6f}",
                position=position + np.array([0.0, 0.0, 0.09], dtype=np.float32),
                visible=visible,
            )
            self.scale_graph_handles.extend((node, label))

        factor_rows = []
        for factor in snapshot["factors"]:
            start = positions[factor["submap_i"]]
            end = positions[factor["submap_j"]]
            factor_type = factor["factor_type"]
            if factor_type == "anchor":
                height = 0.18 + 0.035 * node_ids.index(factor["submap_j"])
                midpoint = (start + end) / 2.0
                midpoint[1] += height
                points = np.asarray([[start, midpoint], [midpoint, end]], dtype=np.float32)
                color = (70, 130, 255)
                prefix = "g"
                label_position = midpoint
            else:
                points = np.asarray([[start, end]], dtype=np.float32)
                color = (255, 80, 80)
                prefix = "f"
                label_position = (start + end) / 2.0
            edge = self.server.scene.add_line_segments(
                f"/scale_factor_graph/edges/{factor_type}_{factor['submap_i']}_{factor['submap_j']}",
                points=points,
                colors=color,
                line_width=4.0,
                visible=visible,
            )
            edge_label = self.server.scene.add_label(
                f"/scale_factor_graph/labels/edge_{factor_type}_{factor['submap_i']}_{factor['submap_j']}",
                text=(f"{prefix}{factor['submap_i']},{factor['submap_j']} "
                      f"m={factor['measurement']:.4f} r={factor['residual']:+.2e}"),
                position=label_position + np.array([0.0, 0.0, 0.045], dtype=np.float32),
                visible=visible,
            )
            self.scale_graph_handles.extend((edge, edge_label))
            factor_rows.append(
                f"| {factor_type} | {factor['submap_i']}→{factor['submap_j']} | "
                f"{factor['measurement']:.6f} | {factor['sigma']:.3g} | "
                f"{factor['residual']:+.3e} | v{factor['measurement_version']} | "
                f"{factor['backend_factor_index'] if factor['backend_factor_index'] is not None else 'pending'} |"
            )

        before = snapshot.get("before", {})
        node_rows = []
        for node_id in node_ids:
            node_rows.append(
                f"| S{node_id} | {before.get(node_id, nodes[node_id]):.6f} | "
                f"{nodes[node_id]:.6f} | {corrections.get(node_id, 1.0):.6f} |"
            )
        factor_table = "\n".join(factor_rows) if factor_rows else "| — | — | — | — | — | — | — |"
        self.gui_scale_graph_status.content = (
            f"**Revision:** {snapshot['revision']}  "
            f"**Backend:** `{snapshot['backend']}`  "
            f"**Status:** {'OK' if snapshot['success'] else 'FAILED'}\n\n"
            "| Node | Before | Optimized | Applied c |\n"
            "|---|---:|---:|---:|\n" + "\n".join(node_rows) + "\n\n"
            "| Factor | Edge | Measurement | Sigma | Residual | Version | Backend index |\n"
            "|---|---|---:|---:|---:|---:|---:|\n" + factor_table +
            "\n\n<span style='color:#ff5050'>Red: overlap f</span> · "
            "<span style='color:#4682ff'>Blue: anchor g</span>"
        )

    def visualize_obb(
        self,
        center: np.ndarray,
        extent: np.ndarray,
        rotation: np.ndarray,
        color = (255, 0, 0),
        line_width: float = 2.0,
    ):
        """
        Visualize an oriented bounding box (OBB) in Viser.

        Parameters
        ----------
        name : str
            Identifier for the OBB in the scene.
        center : (3,) array
            World-space center of the OBB.
        extent : (3,) array
            Full side lengths of the OBB (dx, dy, dz).
        rotation : (3,3) array
            Rotation matrix of the OBB in world coordinates.
        color : tuple[int,int,int]
            RGB color of the wireframe box.
        line_width : float
            Thickness of the box edges.

        Notes
        -----
        The box is drawn as a wireframe with 12 edges.
        """

        # Compute local corners (8)
        dx, dy, dz = extent / 2.0
        corners_local = np.array([
            [-dx, -dy, -dz],
            [ dx, -dy, -dz],
            [ dx,  dy, -dz],
            [-dx,  dy, -dz],
            [-dx, -dy,  dz],
            [ dx, -dy,  dz],
            [ dx,  dy,  dz],
            [-dx,  dy,  dz],
        ], dtype=np.float32)  # shape (8,3)

        # Transform to world
        corners_world = (rotation @ corners_local.T).T + center  # shape (8,3)

        # Build edges (12 line segments) as start/end pairs
        edges_idx = [
            (0,1),(1,2),(2,3),(3,0),  # bottom face
            (4,5),(5,6),(6,7),(7,4),  # top face
            (0,4),(1,5),(2,6),(3,7)   # vertical edges
        ]

        segments = []
        for (i,j) in edges_idx:
            segments.append(corners_world[i])
            segments.append(corners_world[j])
        # segments is list of length 24, reshape into (N,2,3)
        segments = np.array(segments, dtype=np.float32).reshape(-1, 2, 3)

        name = f"obb_{self.obj_id}"
        self.obj_id += 1
        self.server.scene.add_line_segments(
            name=name,
            points=segments,
            colors=color,       # single color for all segments
            line_width=line_width,
            visible=True
        )

    def add_object_query_gui(self, solver, clip_model, clip_tokenizer, processor, data_lock) -> None:
        """Add an object query panel to the viser sidebar.

        Mirrors the terminal open-set query flow: retrieve the best-matching
        keyframe via CLIP, run SAM3 to segment the queried object, then draw an
        oriented bounding box per detected instance in the 3-D scene. Requires
        --run_os (clip_model / processor loaded) and a non-empty map.
        """
        import torch
        from torchvision.transforms.functional import to_pil_image
        import vggt_slam.slam_utils as utils

        with self.server.gui.add_folder("Object Query"):
            gui_query = self.server.gui.add_text("Query", initial_value="")
            gui_status = self.server.gui.add_markdown("*Enter a query and press Search.*")
            btn_search = self.server.gui.add_button("Search", color="green")

        @btn_search.on_click
        def _on_search(_) -> None:
            query = gui_query.value.strip()
            if not query:
                gui_status.content = "*Enter a query first.*"
                return
            if solver.map.get_num_submaps() == 0:
                gui_status.content = "*No map data yet. Capture some frames first.*"
                return
            gui_status.content = f"*Searching for '{query}'…*"
            try:
                with data_lock:
                    text_emb = utils.compute_text_embeddings(clip_model, clip_tokenizer, query)
                    best_score, best_submap_id, best_frame_index = \
                        solver.map.retrieve_best_semantic_frame(text_emb)
                    found_submap = solver.map.get_submap(best_submap_id)
                    best_img = found_submap.get_frame_at_index(best_frame_index)

                    with torch.no_grad():
                        pil_img = to_pil_image(best_img)
                        inference_state = processor.set_image(pil_img)
                        output = processor.set_text_prompt(state=inference_state, prompt=query)
                        masks = output["masks"]

                    n = masks.shape[0]
                    for i in range(n):
                        mask = masks[i].cpu().numpy()
                        obb_center, obb_extent, obb_rotation = utils.compute_obb_from_points(
                            found_submap.get_points_in_mask(best_frame_index, mask, solver.graph)
                        )
                        self.visualize_obb(
                            center=obb_center,
                            extent=obb_extent,
                            rotation=obb_rotation,
                            color=(255, 0, 0),
                            line_width=8.0,
                        )

                if n == 0:
                    gui_status.content = f"*No instances found for '{query}'.*"
                else:
                    gui_status.content = f"**Found {n} instance(s)** for *'{query}'*"
            except Exception as e:
                import traceback
                traceback.print_exc()
                gui_status.content = f"*Error: {e}*"

    def run_walkthrough(self, fps: float = 20.0):
            """
            Walks through the map using the current live positions of all frames.
            This accounts for loop closures because it pulls data from the scene handles.
            """
            # 1. Gather all submap IDs and sort them to ensure a logical sequence
            sorted_submap_ids = sorted(self.submap_frames.keys())
            
            if not sorted_submap_ids:
                print("No frames found to walk through.")
                return

            clients = self.server.get_clients()
            if not clients:
                print("No clients connected to perform walkthrough.")
                return

            print("Starting walkthrough of updated poses...")

            for sub_id in sorted_submap_ids:
                frames = self.submap_frames[sub_id]
                # Assumes frames were added in chronological order to the list
                for frame_handle in frames:
                    # Get the current world-space pose from the visualizer
                    # If a loop closure moved the submap, these values will be updated
                    current_pos = frame_handle.position
                    current_wxyz = frame_handle.wxyz

                    # Update all connected clients
                    for client in clients.values():
                        client.camera.position = current_pos
                        client.camera.wxyz = current_wxyz
                    
                    # Control speed (1/fps)
                    time.sleep(1.0 / fps)
