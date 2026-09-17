import os
import glob
import time
import argparse
import sys

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm
import cv2
import matplotlib.pyplot as plt

from vggt_slam.project_paths import prefer_bundled_third_party

prefer_bundled_third_party()

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver
from vggt_slam.submap import Submap
from vggt_slam.submap_batching import pad_submap_frames, should_process_submap

from vggt.models.vggt import VGGT

parser = argparse.ArgumentParser(description="VGGT-SLAM demo")
parser.add_argument("--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images")
parser.add_argument("--vis_map", action="store_true", help="Visualize point cloud in viser as it is being build, otherwise only show the final map")
parser.add_argument("--vis_imgs", action="store_true", help="Show camera images in the viser frustums. By default only the frustums are shown (faster visualization)")
parser.add_argument("--vis_voxel_size", type=float, default=None, help="Voxel size for downsampling the point cloud in the viewer (e.g. 0.05 for 5 cm). Default: no downsampling")
parser.add_argument("--viewer_port", type=int, default=8080, help="TCP port used by the Viser web viewer (default: 8080)")
parser.add_argument("--run_os", action="store_true", help="Enable open-set semantic search with Perception Encoder CLIP and SAM3")
parser.add_argument("--vis_flow", action="store_true", help="Visualize optical flow from RAFT for keyframe selection")
parser.add_argument(
    "--use_all_frames",
    action="store_true",
    help="Treat every input image as an already-selected keyframe",
)
parser.add_argument("--log_results", action="store_true", help="save txt file with results")
parser.add_argument("--skip_dense_log", action="store_true", help="by default, logging poses and logs dense point clouds. If this flag is set, dense logging is skipped")
parser.add_argument("--log_path", type=str, default="poses.txt", help="Path to save the log file")
parser.add_argument("--output_voxel_size", type=float, default=0.05, help="Voxel size used for the saved PCD (default: 0.05; <=0 disables downsampling)")
parser.add_argument("--trajectory_sphere_radius", type=float, default=None, help="Radius of trajectory spheres in the saved PCD (default: auto)")
parser.add_argument("--trajectory_sphere_points", type=int, default=96, help="Number of surface points per trajectory sphere (default: 96)")
reference_group = parser.add_mutually_exclusive_group()
reference_group.add_argument(
    "--reference_rosbag", type=str, default=None,
    help="Optional ROS 2 bag containing the timestamped reference trajectory",
)
reference_group.add_argument(
    "--reference_trajectory", type=str, default=None,
    help="Optional timestamped text reference: timestamp x y z [qx qy qz qw]",
)
parser.add_argument(
    "--reference_topic", type=str, default="/Odometry",
    help="Pose topic used with --reference_rosbag (default: /Odometry)",
)
parser.add_argument(
    "--submap_png_dir",
    type=str,
    default=None,
    help="Optional directory for one diagnostic PNG (XY/XZ/YZ) per submap",
)
parser.add_argument("--submap_size", type=int, default=16, help="Number of new frames per submap, does not include overlapping frames or loop closure frames")
parser.add_argument("--overlapping_window_size", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW. Number of overlapping frames, which are used in SL(4) estimation")
parser.add_argument("--max_loops", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW or 0 to disable loop closures.")
parser.add_argument("--min_disparity", type=float, default=50, help="Minimum disparity to generate a new keyframe")
parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")
parser.add_argument("--lc_thres", type=float, default=0.95, help="Threshold for image retrieval. Range: [0, 1.0]. Higher = more loop closures")
scale_anchor_group = parser.add_mutually_exclusive_group()
scale_anchor_group.add_argument(
    "--scale_anchor", "--scale-anchor", dest="scale_anchor", action="store_true",
    help="Enable semantic joint-VGGT scale-anchor factors (default)",
)
scale_anchor_group.add_argument(
    "--no_scale_anchor", "--no-scale-anchor", dest="scale_anchor", action="store_false",
    help="Disable joint VGGT scale anchors for an overlap-only baseline",
)
parser.set_defaults(scale_anchor=True)
# Compatibility alias: quality rejection has been removed, not made optional.
parser.add_argument(
    "--no_scale_anchor_gate", "--no-scale-anchor-gate",
    action="store_true", help=argparse.SUPPRESS,
)
parser.add_argument(
    "--scale-anchor-no-conf-filter",
    dest="scale_anchor_conf_filter",
    action="store_false",
    help=(
        "Use all geometrically valid original/joint pixels for Anchor scale "
        "statistics instead of intersecting depth-confidence masks"
    ),
)
parser.set_defaults(scale_anchor_conf_filter=True)
parser.add_argument(
    "--scale-anchor-strict-min-points",
    action="store_true",
    help=(
        "Require the configured number of high-confidence shared pixels; "
        "do not fall back to the less restrictive valid-geometry mask"
    ),
)
parser.add_argument(
    "--scale-anchor-min-points", type=int, default=500,
    help="Minimum same-pixel support for Anchor scale estimation (default: 500)",
)
parser.add_argument(
    "--scale-anchor-fallback-min-points", type=int, default=100,
    help="Minimum valid-geometry pixels after confidence fallback (default: 100)",
)
parser.add_argument(
    "--scale-anchor-weight",
    type=float,
    default=0.1,
    help="Base scale Anchor factor weight before SALAD decay (default: 0.1)",
)
parser.add_argument(
    "--scale-anchor-salad-max-distance", type=float, default=0.9,
    help="Maximum SALAD descriptor distance for an Anchor (default: 0.9)",
)
parser.add_argument(
    "--scale-anchor-salad-weight-sigma", type=float, default=0.5,
    help="Gaussian width for SALAD distance-to-weight decay (default: 0.5)",
)
parser.add_argument(
    "--scale-anchor-min-ordinal-span", type=int, default=6,
    help="Minimum historical submap span for an Anchor (default: 6)",
)
parser.add_argument(
    "--scale-anchor-target-ordinal-span", type=int, default=6,
    help="Submap span at which SALAD Anchor span weighting saturates (default: 6)",
)
parser.add_argument(
    "--scale-anchor-submap-interval", type=int, default=1,
    help="Add an Anchor every N submaps after the minimum span (default: 1)",
)
parser.add_argument(
    "--scale-anchor-reference-mode",
    choices=("salad-far", "fixed-root"),
    default="fixed-root",
    help="Anchor reference policy (default: fixed-root)",
)




def main():
    """
    Main function that wraps the entire pipeline of VGGT-SLAM.
    """
    args = parser.parse_args()

    use_optical_flow_downsample = not args.use_all_frames
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Python executable: {sys.executable}")
    print(f"Using device: {device}")
    print(f"Using VGGT source: {os.path.abspath(__import__(VGGT.__module__, fromlist=['']).__file__)}")

    solver = Solver(
        init_conf_threshold=args.conf_threshold,
        lc_thres=args.lc_thres,
        vis_voxel_size=args.vis_voxel_size,
        vis_imgs=args.vis_imgs,
        viewer_port=args.viewer_port,
        enable_scale_anchor=args.scale_anchor,
        enable_scale_anchor_conf_filter=args.scale_anchor_conf_filter,
        enable_scale_anchor_conf_fallback=(
            not args.scale_anchor_strict_min_points
        ),
        scale_anchor_weight=args.scale_anchor_weight,
        scale_anchor_min_points=args.scale_anchor_min_points,
        scale_anchor_fallback_min_points=args.scale_anchor_fallback_min_points,
        scale_anchor_salad_max_distance=args.scale_anchor_salad_max_distance,
        scale_anchor_salad_weight_sigma=args.scale_anchor_salad_weight_sigma,
        scale_anchor_min_ordinal_span=args.scale_anchor_min_ordinal_span,
        scale_anchor_target_ordinal_span=args.scale_anchor_target_ordinal_span,
        scale_anchor_submap_interval=args.scale_anchor_submap_interval,
        scale_anchor_reference_mode=args.scale_anchor_reference_mode,
    )
    print("Ordinary overlap scale: global (original)")
    print(f"Scale anchor factors: {'enabled' if args.scale_anchor else 'disabled'}")
    print("Scale anchor quality gate: removed (SALAD selection only)")
    print(
        "Scale anchor confidence filter: "
        f"{'enabled' if args.scale_anchor_conf_filter else 'disabled'}"
    )
    print(
        "Scale anchor confidence fallback: "
        f"{'disabled' if args.scale_anchor_strict_min_points else 'enabled'}"
    )
    print(
        "Scale anchor estimator support: "
        f"min_points={args.scale_anchor_min_points} "
        f"fallback_min_points={args.scale_anchor_fallback_min_points}"
    )
    print(
        f"Scale anchor reference: {args.scale_anchor_reference_mode} "
        f"max_distance={args.scale_anchor_salad_max_distance} "
        f"min_ordinal_span={args.scale_anchor_min_ordinal_span} "
        f"target_ordinal_span={args.scale_anchor_target_ordinal_span} "
        f"submap_interval={args.scale_anchor_submap_interval} "
        f"base_weight={args.scale_anchor_weight} "
        f"weight_sigma={args.scale_anchor_salad_weight_sigma}"
    )


    print("Initializing and loading VGGT model...")


    if args.run_os:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms

        sam3_model = build_sam3_image_model()
        processor = Sam3Processor(sam3_model, confidence_threshold=0.50)

        clip_model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)  # Downloads from HF
        clip_model = clip_model.cuda()
        clip_tokenizer = transforms.get_text_tokenizer(clip_model.context_length)
        clip_preprocess = transforms.get_image_transform(clip_model.image_size)
    else:
        clip_model, clip_preprocess = None, None
        clip_tokenizer = None

    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

    model.eval()
    model = model.to(torch.bfloat16)  # use half precision
    model = model.to(device)

    # Use the provided image folder path
    print(f"Loading images from {args.image_folder}...")
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_names = [
        path for path in glob.glob(os.path.join(args.image_folder, "*"))
        if os.path.isfile(path)
        and os.path.splitext(path)[1].lower() in image_extensions
        and cv2.haveImageReader(path)
    ]

    image_names = utils.sort_images_by_number(image_names)
    downsample_factor = 1
    image_names = utils.downsample_images(image_names, downsample_factor)
    print(f"Found {len(image_names)} images")
    if not image_names:
        raise FileNotFoundError(
            f"No readable image files found directly under {args.image_folder!r}. "
            "Pass the directory that contains the JPG/PNG frames, not its parent directory."
        )

    image_names_subset = []
    count = 0
    image_count = 0
    total_time_start = time.time()
    keyframe_time = utils.Accumulator()
    backend_time = utils.Accumulator()
    target_submap_size = args.submap_size + args.overlapping_window_size
    for image_name in tqdm(image_names):
        if use_optical_flow_downsample:
            with keyframe_time:
                img = cv2.imread(image_name)
                if img is None:
                    raise ValueError(f"OpenCV could not decode input image: {image_name}")
                enough_disparity = solver.flow_tracker.compute_disparity(img, args.min_disparity, args.vis_flow)
                if enough_disparity:
                    image_names_subset.append(image_name)
                    image_count += 1
        else:
            image_names_subset.append(image_name)
            image_count += 1

        # Always present VGGT with the configured fixed-size submap. A final
        # tail is padded by repeating its final selected keyframe.
        is_last_image = image_name == image_names[-1]
        process_submap = should_process_submap(
            len(image_names_subset), target_submap_size,
            args.overlapping_window_size, count, is_last_image,
        )
        if process_submap:
            batch_image_names = image_names_subset
            if is_last_image and len(image_names_subset) < target_submap_size:
                batch_image_names, padding_count = pad_submap_frames(
                    image_names_subset, target_submap_size
                )
                print(
                    f"Padding final submap from {len(image_names_subset)} to "
                    f"{target_submap_size} frames by repeating the final "
                    f"selected frame {padding_count} times"
                )
            count += 1
            print(batch_image_names)
            t1 = time.time()
            predictions = solver.run_predictions(batch_image_names, model, args.max_loops, clip_model, clip_preprocess)
            print("Solver total time", time.time()-t1)
            print(count, "submaps processed")

            solver.add_points(predictions)

            with backend_time:
                solver.graph.optimize()

            loop_closure_detected = len(predictions["detected_loops"]) > 0
            solver.finalize_backend_update(
                vis_map=args.vis_map,
                force_all=loop_closure_detected,
            )
            
            # Reset for next submap.
            image_names_subset = image_names_subset[-args.overlapping_window_size:]
        elif is_last_image and image_names_subset:
            print("Skipping final overlap-only buffer; it contains no new keyframe")

    total_time = time.time() - total_time_start
    average_fps = total_time / image_count
    print(image_count, "frames processed")
    print("Total time:", total_time)
    print(f"Total time for VGGT calls: {solver.vggt_timer.total_time:.4f}s")
    print("Average VGGT time per frame:", solver.vggt_timer.total_time / image_count)
    print("Average loop closure time per frame:", solver.loop_closure_timer.total_time / image_count)
    print("Average keyframe selection time per frame:", keyframe_time.total_time / image_count)
    print("Average backend time per frame:", backend_time.total_time / image_count)
    print("Average semantic time per frame:", solver.clip_timer.total_time / image_count)
    print("Average total time per frame:", total_time / image_count)
    print("Average FPS:", 1 / average_fps)
        
    print("Total number of submaps in map", solver.map.get_num_submaps())
    print("Total number of loop closures in map", solver.graph.get_num_loops())


    if args.run_os:
        # Register the viser object-query panel so the user can search for
        # objects in the viewer in addition to the terminal prompt below.
        import threading
        data_lock = threading.Lock()
        solver.viewer.add_object_query_gui(solver, clip_model, clip_tokenizer, processor, data_lock)

        while True:
            # Prompt user for text input
            query = input("\nEnter text query or q to quit: ").strip()
            if len(query) == 0:
                print("Empty query. Exiting.")
                return
            
            if query == "q":
                print("Exiting.")
                return
            
            text_emb = utils.compute_text_embeddings(clip_model, clip_tokenizer, query)
            overall_best_score, overall_best_submap_id, overall_best_frame_index = solver.map.retrieve_best_semantic_frame(text_emb)

            found_submap = solver.map.get_submap(overall_best_submap_id)

            # Display image
            best_img = found_submap.get_frame_at_index(overall_best_frame_index)
            print("Score:", overall_best_score)
            with torch.no_grad():
                # convert torch image to PIL
                best_img = to_pil_image(best_img)
                inference_state = processor.set_image(best_img)
                output = processor.set_text_prompt(state=inference_state, prompt=query)
                masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
                print(f"Found {masks.shape[0]} masks from SAM3 for the prompt '{query}'")
                print("Scores:", scores.cpu().numpy())


            masked_img = utils.overlay_masks(best_img, masks)
            masked_img.show()

            for i in range(masks.shape[0]):
                mask = masks[i].cpu().numpy()
                obb_center, obb_extent, obb_rotation = utils.compute_obb_from_points(found_submap.get_points_in_mask(overall_best_frame_index, mask, solver.graph))
                solver.viewer.visualize_obb(
                    center=obb_center,
                    extent=obb_extent,
                    rotation=obb_rotation,
                    color=(255, 0, 0),
                    line_width=8.0,
                )

    if not args.vis_map:
        # just show the map after all submaps have been processed
        solver.update_all_submap_vis()

    if args.log_results:
        solver.map.write_poses_to_file(args.log_path, solver.graph, kitti_format=False)
        output_stem = os.path.splitext(args.log_path)[0]
        output_dir = os.path.dirname(os.path.abspath(args.log_path))
        trajectory_cloud_path = output_stem + "_trajectory_spheres.pcd"
        solver.map.write_trajectory_spheres_to_file(
            solver.graph,
            trajectory_cloud_path,
            sphere_radius=args.trajectory_sphere_radius,
            points_per_sphere=args.trajectory_sphere_points,
        )

        if not args.skip_dense_log:
            # Keep the original full cloud and a portable downsampled copy.
            solver.map.write_points_to_file(
                solver.graph,
                output_stem + "_points.pcd",
                voxel_size=0,
            )
            if args.output_voxel_size is not None and args.output_voxel_size > 0:
                solver.map.write_points_to_file(
                    solver.graph,
                    output_stem + "_points_downsampled.pcd",
                    voxel_size=args.output_voxel_size,
                )
        solver.write_scale_anchor_artifacts(
            os.path.join(output_dir, "anchor_frames"),
            trajectory_cloud_path=trajectory_cloud_path,
            combined_output_path=output_stem + "_trajectory_with_anchors.pcd",
        )
        if args.reference_rosbag or args.reference_trajectory:
            from vggt_slam.trajectory_evaluation import write_trajectory_comparisons
            write_trajectory_comparisons(
                args.log_path,
                os.path.join(output_dir, "trajectory_reference"),
                reference_rosbag=args.reference_rosbag,
                reference_trajectory=args.reference_trajectory,
                reference_topic=args.reference_topic,
            )

    if args.submap_png_dir:
        solver.map.write_submap_previews(solver.graph, args.submap_png_dir)

    if args.vis_map:
        print(f"Visualization ready at http://127.0.0.1:{args.viewer_port}")
        print("Press Ctrl+C to stop the Viser server and exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping Viser server.")


if __name__ == "__main__":
    main()
