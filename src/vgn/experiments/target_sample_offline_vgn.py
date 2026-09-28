import collections
import argparse
import json
import math
import numpy as np
import os
import pandas as pd
import pyrender
import time
import tqdm
import trimesh
import uuid
import matplotlib.pyplot as plt
import torch
import random
from datetime import datetime
import sys
from utils_giga import filter_and_pad_point_clouds
# import trimesh
from vgn import io  # For CSV creation and I/O
from vgn.grasp import *
from vgn.ConvONets.eval import MeshEvaluator
from vgn.simulation import ClutterRemovalSim
from vgn.utils.transform import Rotation, Transform
from vgn.utils.implicit import (
    get_mesh_pose_list_from_world,
    get_scene_from_mesh_pose_list
)
from vgn.perception import camera_on_sphere
# Debug / logging utilities
from utils_targo import (
    record_occ_level_count,
    record_occ_level_success,
    cal_occ_level_sr,
    save_point_cloud_as_ply,
    generate_and_transform_grasp_meshes,
    compute_chamfer_and_iou,
    save_open3d_mesh
)
from vgn.utils_visualization import visualize_failure_case

skip_list = {
    "013_apple",
    "021_bleach_cleanser",
    "052_extra_large_clamp",
    "025_mug",
    "006_mustard_bottle",
    "016_pear",
    "010_potted_meat_can",
    "050_medium_clamp",
    "005_tomato_soup_can",
    "004_sugar_box"
}
MAX_CONSECUTIVE_FAILURES = 2
State = collections.namedtuple("State", ["tsdf", "pc"])


def _tsdf_grid_to_mesh(tsdf_grid):
    """Convert a TSDF grid to a non-empty trimesh, handling both local APIs."""
    from utils_targo import tsdf_to_mesh

    mesh_or_tuple = tsdf_to_mesh(tsdf_grid)
    if isinstance(mesh_or_tuple, trimesh.Trimesh):
        return None if mesh_or_tuple.is_empty else mesh_or_tuple

    if isinstance(mesh_or_tuple, tuple) and len(mesh_or_tuple) >= 2:
        vertices, faces = mesh_or_tuple[:2]
        if vertices is None or faces is None:
            return None
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return None if mesh.is_empty else mesh

    return None


def init_occ_level_count_dict():
    """
    Create a dictionary to count the number of scenes in each occlusion bin.
    """
    return {
        '0-0.1': 0,
        '0.1-0.2': 0,
        '0.2-0.3': 0,
        '0.3-0.4': 0,
        '0.4-0.5': 0,
        '0.5-0.6': 0,
        '0.6-0.7': 0,
        '0.7-0.8': 0,
        '0.8-0.9': 0,
    }


def init_occ_level_success_dict():
    """
    Create a dictionary to record successful grasps in each occlusion bin.
    """
    return {
        '0-0.1': 0,
        '0.1-0.2': 0,
        '0.2-0.3': 0,
        '0.3-0.4': 0,
        '0.4-0.5': 0,
        '0.5-0.6': 0,
        '0.6-0.7': 0,
        '0.7-0.8': 0,
        '0.8-0.9': 0,
    }


def read_target_names_from_file(target_file_path):
    """
    Read target names list from a specified txt file
    
    Args:
        target_file_path (str): Path to the txt file
        
    Returns:
        list: List of target names, returns empty list if file doesn't exist
    """
    target_names = []
    if target_file_path and os.path.exists(target_file_path):
        with open(target_file_path, 'r') as f:
            for line in f:
                name = line.strip()
                if name:  # Ensure empty lines are not added
                    target_names.append(name)
    return target_names




def run(
    grasp_plan_fn,
    logdir,
    description,
    scene,
    object_set,
    sim_gui=False,
    result_path=None,
    add_noise=False,
    sideview=False,
    resolution=40,
    visualize=False,
    type='vgn',
    test_root=None,
    occ_level_dict_path=None,
    model_type=None,
    video_recording=True,
    target_file_path=None,  
    data_type='ycb',
    max_scenes=0,  # Add parameter to limit number of scenes (0 means no limit)
    sc_net=None,  # Shape completion network
    vis_failure_only=False,  # Visualize only failure cases
    vis_rgb=False,  # Visualize RGB images
    vis_gt_target=True,  # Visualize GT target mesh
    vis_grasps=True,  # Visualize grasps on affordance mesh
    vis_target_mesh_pred=True,  # Save predicted target mesh
    # SAM-specific parameters
    use_sam=False,  # Whether to use SAM for segmentation
    sam_model_type='vit_h',  # SAM model type
    sam_checkpoint=None,  # Path to SAM checkpoint
):
    # Initialize SAM if needed
    sam_predictor = None
    sam_results_file = None
    if use_sam:
        try:
            from segment_anything import sam_model_registry, SamPredictor
            print(f"Initializing SAM model: {sam_model_type}")
            sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
            sam_predictor = SamPredictor(sam)
            print(f"SAM model loaded successfully from {sam_checkpoint}")
            
            # Create SAM segmentation results file
            sam_results_file = os.path.join(result_path, 'sam_segmentation_results.txt')
            with open(sam_results_file, 'w') as f:
                f.write("Scene_ID, IoU, Dice, Success\n")
            print(f"SAM results file created: {sam_results_file}")
        except Exception as e:
            print(f"Error initializing SAM: {e}")
            print("Falling back to ground truth segmentation")
            use_sam = False
            sam_results_file = None

    # Initialize the simulation
    sim = ClutterRemovalSim(
        scene, object_set,
        gui=sim_gui,
        add_noise=add_noise,
        sideview=sideview,
        test_root=test_root,
        egl_mode=False,
        sc_net=sc_net
    )
    logger = Logger(logdir, description, tgt_sample=True)

    # Track planning times
    planning_times, total_times = [], []

    # Paths for test data
    test_mesh_pose_list = f'{test_root}/mesh_pose_dict/'
    test_scenes = f'{test_root}/scenes/'
    
    # TARGO data preprocessing paths
    processed_scenes_targo_path = f'{test_root}/processed_scene_targo'
    if not os.path.exists(processed_scenes_targo_path):
        os.makedirs(processed_scenes_targo_path)

    # Dictionaries to track occlusion stats
    occ_level_count_dict = init_occ_level_count_dict()
    occ_level_success_dict = init_occ_level_success_dict()
    offline_occ_level_dict = {}
    
    plan_failure_count = 0
    visual_failure_count = 0

    count_label_dict = {}
    tgt_bbx_label_dict = {}
    skip_dict = {}
    targ_name_label = {}  # Dictionary to store target name and label
    completion_metrics_dict = {}  # Dictionary to store completion metrics (IoU, CD) for each scene

    occ_level_dict = json.load(open(occ_level_dict_path))
    
    # Get all scene files in a stable order; max_scenes=0 means evaluate everything.
    all_scene_files = sorted(os.listdir(test_mesh_pose_list))
    if max_scenes > 0 and len(all_scene_files) > max_scenes:
        import random
        random.seed(42)  # Set seed for reproducibility
        selected_scene_files = sorted(random.sample(all_scene_files, max_scenes))
        print(f"=== SCENE SELECTION INFO ===")
        print(f"Total available scenes: {len(all_scene_files)}")
        print(f"Randomly selected scenes: {max_scenes}")
        print(f"Random seed: 42 (for reproducibility)")
        print(f"First 5 selected scenes: {selected_scene_files[:5]}")
        print("=" * 30)
    else:
        selected_scene_files = all_scene_files
        print(f"=== SCENE SELECTION INFO ===")
        print(f"Processing all {len(selected_scene_files)} scenes")
        print("=" * 30)
    
    # Initialize dictionary to store target meshes for length_width calculation
    scene_target_mesh_dict = {}
    
    # Loop over the selected test set
    for num_id, curr_mesh_pose_list in enumerate(selected_scene_files):
        path_to_npz = os.path.join(test_scenes, curr_mesh_pose_list)
        scene_name = curr_mesh_pose_list[:-4]
        occ_level = occ_level_dict[scene_name]
        root_path = scene_name.split('_')[0]

        if scene_name not in occ_level_dict:
            continue

        # Prepare simulator
        sim.world.reset()
        sim.world.set_gravity([0.0, 0.0, -9.81])
        sim.draw_workspace()
        sim.save_state()

        # Manually adjust boundaries
        sim.lower = np.array([0.02, 0.02, 0.055])
        sim.upper = np.array([0.28, 0.28, 0.30000000000000004])
        
        # Create visual_dict for visualization if needed
        if visualize:
            visual_dict = {'mesh_name': scene_name, 'mesh_dir': logger.mesh_dir}
        else:
            visual_dict = {}

        tgt_id = int(scene_name[-1])  
        start_time = time.time()

        occluder_heights = []

        mp_data = np.load(
            os.path.join(test_mesh_pose_list, curr_mesh_pose_list),
            allow_pickle=True
        )["pc"]

        occluder_count = len(mp_data.item().keys()) - 1  # Total objects minus plane

        # Place objects
        for obj_id, mesh_info in enumerate(mp_data.item().values()):
            pose = Transform.from_matrix(mesh_info[2])
            if mesh_info[0].split('/')[-1] == 'plane.obj':
                urdf_path = mesh_info[0].replace(".obj", ".urdf")
            else:
                urdf_path = mesh_info[0].replace("_visual.obj", ".urdf")

            body = sim.world.load_urdf(
                urdf_path=urdf_path,
                pose=pose,
                scale=mesh_info[1]
            )

            # Get bounding heights
            if obj_id != 0:
                tri_mesh = trimesh.load_mesh(mesh_info[0])
                tri_mesh.apply_scale(mesh_info[1])
                tri_mesh.apply_transform(mesh_info[2])
                z_max = tri_mesh.vertices[:, 2].max()

                if obj_id != tgt_id:
                    occluder_heights.append(z_max)
                else:
                    tgt_height = z_max
                    min_coords = tri_mesh.vertices.min(axis=0)
                    max_coords = tri_mesh.vertices.max(axis=0)
                    length, width, height = max_coords - min_coords

            # Color target red
            if obj_id == tgt_id:
                body.set_color(link_index=-1, rgba_color=(1.0, 0.0, 0.0, 1.0))
                targ_name = urdf_path.split('/')[-1].split('.')[0]
                target_mesh_gt = tri_mesh

        end_time = time.time()

        # Skip if target is in skip_list
        if targ_name in skip_list:
            print(f"Skipping {targ_name} as it's in the skip list")
            continue

        # Acquire data for VGN
        timings = {}
        start_time = time.time()

        # Data acquisition based on model type
        if model_type == 'vgn':
            # VGN specific data acquisition
            tsdf, timings["integration"], scene_grid, targ_grid, targ_mask, occ_level = \
            sim.acquire_single_tsdf_target_grid(
                path_to_npz,
                tgt_id,
                40,
                model_type,  
                curr_mesh_pose_list=scene_name,
            )
            state = argparse.Namespace(
                    tsdf=tsdf,
                    scene_grid=scene_grid,
                        targ_grid=targ_grid,
                    tgt_mask_vol=targ_mask,
                        occ_level=occ_level,
                        type=model_type,
                        timings=timings
                    )
        elif model_type in ['targo', 'targo_full_targ', 'targo_hunyun2', 'targo_ptv3', 'targo_partial', 'targo_full_gt']:
            # TARGO specific data acquisition
            curr_processed_scene_targo_path = f'{processed_scenes_targo_path}/{scene_name}.npz'
            
            # Create scene-specific visualization directory if needed
            scene_vis_dir = None
            if visualize:
                scene_vis_dir = f"{logger.mesh_dir}/{scene_name}"
                os.makedirs(scene_vis_dir, exist_ok=True)
            
            if os.path.exists(curr_processed_scene_targo_path):
                print(f"Loading preprocessed TARGO data for scene {scene_name}")
                processed_data = np.load(curr_processed_scene_targo_path, allow_pickle=True)
                scene_no_targ_pc = processed_data['scene_no_targ_pc'].astype(np.float32)
                targ_pc = processed_data['targ_pc'].astype(np.float32)
                targ_grid = processed_data['targ_grid'].astype(np.float32)
                occ_level = float(processed_data['occ_level'])
                tsdf = targ_grid
                timings["integration"] = 0.0
            else:
                print(f"Generating TARGO input data for scene {scene_name}")
                scene_no_targ_pc, targ_pc, targ_grid, occ_level, tsdf_targ = generate_targo_input_data(
                    sim, path_to_npz, tgt_id, scene_name, processed_scenes_targo_path,
                    use_sam=use_sam, sam_predictor=sam_predictor, scene_vis_dir=scene_vis_dir,
                    sam_results_file=sam_results_file
                )
                
                if scene_no_targ_pc is None:
                    print(f"Failed to generate TARGO data for scene {scene_name}, skipping")
                    continue
                tsdf = tsdf_targ
                timings["integration"] = 0.0
                
            state = argparse.Namespace(
                    tsdf=tsdf,
                    scene_no_targ_pc=scene_no_targ_pc,
                    targ_grid=targ_grid,
                    targ_pc=targ_pc,
                    occ_level=occ_level,
                    type=model_type,
                        timings=timings
                )

        end_time = time.time()

        # Add point cloud visualization for TARGO models (similar to reference file)
        if visualize and model_type in ['targo', 'targo_partial', 'targo_full_gt']:
            # Save target point cloud for visualization
            if hasattr(state, 'targ_pc') and state.targ_pc is not None:
                save_point_cloud_as_ply(state.targ_pc, f"{scene_vis_dir}/targ_pc.ply")
            
            # Save scene point cloud for visualization
            if hasattr(state, 'scene_no_targ_pc') and state.scene_no_targ_pc is not None:
                save_point_cloud_as_ply(state.scene_no_targ_pc, f"{scene_vis_dir}/scene_no_targ_pc.ply")

        # Record occlusion
        occ_level_count_dict = record_occ_level_count(occ_level, occ_level_count_dict)
        offline_occ_level_dict[scene_name] = occ_level

        # Skip if occlusion > 0.9
        if occ_level >= 0.9:
            print("skip")
            skip_dict[scene_name] = occ_level
            continue

        # Get scene mesh for visualization
        mesh_pose_list = get_mesh_pose_list_from_world(sim.world, object_set)
        result = get_scene_from_mesh_pose_list(mesh_pose_list, tgt_id - 1, return_target_mesh=True)
        
        if result is None:
            print(f"Error: Could not get scene and target mesh for scene {scene_name}, skipping")
            continue
        
        scene_mesh, target_mesh = result
        
        # Store target mesh for length_width calculation
        if target_mesh is not None:
            scene_target_mesh_dict[scene_name] = target_mesh
        elif 'target_mesh_gt' in locals() and target_mesh_gt is not None:
            scene_target_mesh_dict[scene_name] = target_mesh_gt
        
        # Planning based on model type
        if visualize:
            # Get scene mesh for visualization (similar to reference file)
            mesh_pose_list = get_mesh_pose_list_from_world(sim.world, object_set)
            scene_mesh = get_scene_from_mesh_pose_list(mesh_pose_list, tgt_id - 1)
            
        if visualize and model_type == 'vgn':
            # VGN planning with visualization
            qual_vol, rot_vol, width_vol = grasp_plan_fn(state, scene_mesh)

            # Convert volumes to grasps
            from vgn.detection_implicit_vgn import select, bound
            qual_vol = bound(qual_vol, state.tsdf.voxel_size)
            
            # Create position grid for grasp selection
            x, y, z = torch.meshgrid(
                torch.linspace(start=-0.5, end=0.5 - 1.0 / 40, steps=40), 
                torch.linspace(start=-0.5, end=0.5 - 1.0 / 40, steps=40), 
                torch.linspace(start=-0.5, end=0.5 - 1.0 / 40, steps=40)
            )
            pos = torch.stack((x, y, z), dim=-1).float()
            center_vol = pos.view(40, 40, 40, 3)
            
            grasps, scores = select(qual_vol.copy(), center_vol, rot_vol, width_vol, threshold=0.9, force_detection=True, max_filter_size=4)
            
            timings["planning"] = 0.0  # VGN doesn't have separate planning time
            visual_mesh = None  # VGN doesn't return visual mesh
            
        elif visualize and model_type in ['targo', 'targo_full_targ', 'targo_hunyun2', 'targo_ptv3', 'targo_partial', 'targo_full_gt']:
            # TARGO planning with visualization
            visual_dict = {}  # Initialize visual_dict
            grasps, scores, timings["planning"], visual_mesh = grasp_plan_fn(
            # grasps, scores, timings["planning"] = grasp_plan_fn(
                state, scene_mesh, visual_dict,
                scene_name=scene_name, 
                cd_iou_measure=True, 
                target_mesh_gt=target_mesh_gt
            )
            
            # Generate and save target shape completion visualization ONLY for targo model
            completion_mesh = None
            
            # First, try to get completed_target_mesh from visual_dict (preferred method)
            if visual_dict and 'completed_target_mesh' in visual_dict:
                completion_mesh = visual_dict['completed_target_mesh']
            
            # Fallback: convert from complete_targ_grid if available
            if (completion_mesh is None and model_type == 'targo' and 
                hasattr(state, 'complete_targ_grid') and state.complete_targ_grid is not None):
                try:
                    completion_mesh = _tsdf_grid_to_mesh(state.complete_targ_grid)
                    if completion_mesh is not None:
                        print(f"Converted completed_target_mesh from complete_targ_grid")
                                
                except Exception as completion_error:
                    print(f"Error generating target completion for {scene_name}: {completion_error}")
            
            # Save completion mesh and calculate metrics if available
            if completion_mesh is not None:
                # Save completion mesh in scene directory
                completion_path = f"{scene_vis_dir}/target_completion.ply"
                # completion_mesh.export(completion_path)
                import open3d as o3d
                o3d.io.write_triangle_mesh(
                    completion_path,
                    completion_mesh,
                    write_ascii=False,          # PLY binary (smaller/faster). Use True for ascii.
                    compressed=False,
                    print_progress=False,
                )
                print(f"Saved target completion mesh: {completion_path}")
                
                # Calculate and save CD and IoU metrics
                if target_mesh_gt is not None:
                    try:
                        from utils_targo import compute_chamfer_and_iou
                        
                        # Calculate CD and IoU
                        cd, iou = compute_chamfer_and_iou(target_mesh_gt, completion_mesh)
                        
                        # Save metrics in scene directory
                        metrics_path = f"{scene_vis_dir}/completion_metrics.txt"
                        with open(metrics_path, 'w') as f:
                            f.write(f"Scene: {scene_name}\n")
                            f.write(f"Model Type: {model_type}\n")
                            f.write(f"Chamfer Distance (CD): {cd:.6f}\n")
                            f.write(f"IoU: {iou:.6f}\n")
                        print(f"Saved completion metrics: CD={cd:.6f}, IoU={iou:.6f}")
                        
                        # Store metrics for later use in meta_evaluations.txt
                        if not hasattr(state, 'completion_metrics'):
                            state.completion_metrics = {}
                        state.completion_metrics['cd'] = cd
                        state.completion_metrics['iou'] = iou
                        
                        # Store in global dictionary
                        completion_metrics_dict[scene_name] = {'cd': cd, 'iou': iou}
                        
                    except Exception as metrics_error:
                        print(f"Error calculating CD/IoU for {scene_name}: {metrics_error}")
                        # Set default values
                        if not hasattr(state, 'completion_metrics'):
                            state.completion_metrics = {}
                        state.completion_metrics['cd'] = 0.0
                        state.completion_metrics['iou'] = 0.0
                        
                        # Store default values in global dictionary
                        completion_metrics_dict[scene_name] = {'cd': 0.0, 'iou': 0.0}
            else:
                # For non-targo models or when no completion available, set default values
                if not hasattr(state, 'completion_metrics'):
                    state.completion_metrics = {}
                state.completion_metrics['cd'] = 0.0
                state.completion_metrics['iou'] = 0.0
                
                # Store default values in global dictionary
                completion_metrics_dict[scene_name] = {'cd': 0.0, 'iou': 0.0}
        
            # visual_mesh = None  # TARGO doesn't return visual mesh
            
            # Add RGB rendering (similar to reference file)
            from vgn.perception import camera_on_sphere
            origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, sim.size / 3])
            r = 2 * sim.size
            theta = np.pi / 3.0
            phi = - np.pi / 2.0
            extrinsic = camera_on_sphere(origin, r, theta, phi)
            rgb, _, _ = sim.camera.render_with_seg(extrinsic)
            output_path = f'{scene_vis_dir}/rgb.png'
            plt.imsave(output_path, rgb)
            
            
        else:
            # Non-visualization planning
            if model_type == 'vgn':
                # VGN planning
                qual_vol, rot_vol, width_vol = grasp_plan_fn(state, scene_mesh)

                # Convert volumes to grasps
                from vgn.detection_implicit_vgn import select, bound
                qual_vol = bound(qual_vol, state.tsdf.voxel_size)
                
                # Create position grid for grasp selection
                x, y, z = torch.meshgrid(
                    torch.linspace(start=-0.5, end=0.5 - 1.0 / 40, steps=40), 
                    torch.linspace(start=-0.5, end=0.5 - 1.0 / 40, steps=40), 
                    torch.linspace(start=-0.5, end=0.5 - 1.0 / 40, steps=40)
                )
                pos = torch.stack((x, y, z), dim=-1).float()
                center_vol = pos.view(40, 40, 40, 3)
                
                grasps, scores = select(qual_vol.copy(), center_vol, rot_vol, width_vol, threshold=0.9, force_detection=True, max_filter_size=4)
                
                timings["planning"] = 0.0  # VGN doesn't have separate planning time
                
            elif model_type in ['targo', 'targo_full_targ', 'targo_hunyun2', 'targo_ptv3', 'targo_partial', 'targo_full_gt']:
                # Non-visualization TARGO planning
                visual_dict = {}  # Initialize visual_dict even for non-visualization
                grasps, scores, timings["planning"], visual_mesh = grasp_plan_fn(
                    state, scene_mesh, visual_dict,
                scene_name=scene_name, 
                cd_iou_measure=True, 
                target_mesh_gt=target_mesh_gt
            )
                
                # Calculate CD and IoU metrics ONLY for targo model (non-visualization mode)
                if model_type == 'targo' and hasattr(state, 'complete_targ_grid') and state.complete_targ_grid is not None:
                    try:
                        completion_mesh = _tsdf_grid_to_mesh(state.complete_targ_grid)
                        if completion_mesh is not None:
                            
                            # Calculate CD and IoU metrics
                            if target_mesh_gt is not None:
                                try:
                                    from utils_targo import compute_chamfer_and_iou
                                    
                                    # Calculate CD and IoU
                                    cd, iou = compute_chamfer_and_iou(target_mesh_gt, completion_mesh)
                                    
                                    # Store metrics for later use in meta_evaluations.txt
                                    if not hasattr(state, 'completion_metrics'):
                                        state.completion_metrics = {}
                                    state.completion_metrics['cd'] = cd
                                    state.completion_metrics['iou'] = iou
                                    
                                    # Store in global dictionary
                                    completion_metrics_dict[scene_name] = {'cd': cd, 'iou': iou}
                                    
                                    print(f"Completion metrics for {scene_name}: CD={cd:.6f}, IoU={iou:.6f}")
                                    
                                except Exception as metrics_error:
                                    # Set default values
                                    if not hasattr(state, 'completion_metrics'):
                                        state.completion_metrics = {}
                                    state.completion_metrics['cd'] = 0.0
                                    state.completion_metrics['iou'] = 0.0
                                    
                                    # Store default values in global dictionary
                                    completion_metrics_dict[scene_name] = {'cd': 0.0, 'iou': 0.0}
                                    
                    except Exception as completion_error:
                        # Set default values
                        if not hasattr(state, 'completion_metrics'):
                            state.completion_metrics = {}
                        state.completion_metrics['cd'] = 0.0
                        state.completion_metrics['iou'] = 0.0
                else:
                    # For non-targo models or when no completion available, set default values
                    if not hasattr(state, 'completion_metrics'):
                        state.completion_metrics = {}
                    state.completion_metrics['cd'] = 0.0
                    state.completion_metrics['iou'] = 0.0
                    
                    # Store default values in global dictionary
                    completion_metrics_dict[scene_name] = {'cd': 0.0, 'iou': 0.0}

        planning_times.append(timings["planning"])
        if isinstance(state.timings, dict):
            total_times.append(timings["planning"] + state.timings["integration"])
        else:
            total_times.append(timings["planning"] + state.timings.item()["integration"])

        if len(grasps) == 0:
            # When no valid grasp found, record as failure instead of skipping
            print(f"No valid grasp found, recording as failure: {scene_name}")
            
            # Set failure state
            label = Label.FAILURE
            plan_fail = 1
            visual_fail = 0
            
            
            # Update records
            count_label_dict[scene_name] = (int(label), occluder_count, occ_level)
            tgt_bbx_label_dict[scene_name] = (int(label), length, width, height, occ_level)
            targ_name_label[targ_name] = int(label)
            
            # Update failure count
            plan_failure_count += 1
                
            # Continue to next scene
            continue

        grasp, score = grasps[0], scores[0]
        grasp.width = sim.gripper.max_opening_width

        # Execute grasp in simulation
        log_id = None
        if video_recording:
            # If target file path is provided, read the list of target names
            target_names_to_record = []
            if target_file_path:
                target_names_to_record = read_target_names_from_file(target_file_path)
                print(f"Read {len(target_names_to_record)} target names for video recording")
            
            # Check if current target should be recorded
            should_record = not target_names_to_record or targ_name in target_names_to_record
            
            if should_record:
                # Create video save path
                video_path = os.path.join(result_path, f"grasping_videos/{targ_name}")
                
                # Create directory if it doesn't exist
                os.makedirs(video_path, exist_ok=True)
                
                # First execute grasp (without recording) to determine success/failure
                sim.save_state()  # Save state for later restoration
                label, plan_fail, visual_fail = sim.execute_grasp(
                    grasp,
                    allow_contact=True,
                    tgt_id=tgt_id,
                    force_targ=True
                )
                
                # Reset the simulator for recording (recreate the scene instead of restoring state)
                sim.world.reset()
                sim.world.set_gravity([0.0, 0.0, -9.81])
                sim.draw_workspace()
                
                # Recreate the scene by loading URDF files again
                for obj_id, mesh_info in enumerate(mp_data.item().values()):
                    pose = Transform.from_matrix(mesh_info[2])
                    if data_type != 'acronym':
                        if mesh_info[0].split('/')[-1] == 'plane.obj':
                            urdf_path = mesh_info[0].replace(".obj", ".urdf")
                        else:
                            urdf_path = mesh_info[0].replace("_textured.obj", ".urdf")
                    
                    elif data_type == 'acronym':
                        # Extract file ID part (without path and extension)
                        file_basename = os.path.basename(mesh_info[0])
                        if file_basename == 'plane.obj':
                            urdf_path = mesh_info[0].replace(".obj", ".urdf")
                            continue
                        else:
                            file_id = file_basename.replace("_textured.obj", "").replace(".obj", "")
                            
                        # Base directory for URDF files
                        urdf_base_dir = "./data//acronym/urdfs_acronym"
                        
                        # Method 1: Directly build path (if no category prefix)
                        urdf_path = f"{urdf_base_dir}/{file_id}.urdf"
                        
                        # Method 2: If category prefix exists, use glob to find matching files
                        if not os.path.exists(urdf_path):
                            import glob
                            matching_files = glob.glob(f"{urdf_base_dir}/*_{file_id}.urdf")
                            if matching_files:
                                urdf_path = matching_files[0]  # Use the first matching file found

                    body = sim.world.load_urdf(
                        urdf_path=urdf_path,
                        pose=pose,
                        scale=mesh_info[1]
                    )
                    
                    # Mark the target object as red
                    if obj_id == tgt_id:
                        body.set_color(link_index=-1, rgba_color=(1.0, 0.0, 0.0, 1.0))
                
                # Create unique video filename with success/failure identifier
                video_filename = f"{'success' if label != Label.FAILURE else 'failure'}_{scene_name}"
                
                # Full path for the video file
                video_file = os.path.join(video_path, f"{video_filename}.mp4")
                
                # Start recording
                log_id = sim.start_video_recording(video_filename, video_path)
                
                # Execute grasp again for recording (result doesn't matter for recording)
                _, _, _ = sim.execute_grasp(
                    grasp,
                    allow_contact=True,
                    tgt_id=tgt_id,
                    force_targ=True
                )
                
                # Stop video recording
                sim.stop_video_recording(log_id)
                
                # Check if video file exists
                if os.path.exists(video_file):
                    print(f"Grasp video successfully saved to: {video_file}")
                else:
                    print(f"Warning: Video recording failed. File not found at: {video_file}")
            else:
                # If current target is not in the recording list, execute grasp without recording
                label, plan_fail, visual_fail = sim.execute_grasp(
                    grasp,
                    allow_contact=True,
                    tgt_id=tgt_id,
                    force_targ=True
                )
        else:
            label, plan_fail, visual_fail = sim.execute_grasp(
                grasp,
                allow_contact=True,
                tgt_id=tgt_id,
                force_targ=True
            )
            
        if plan_fail == 1:
            plan_failure_count += 1
        if visual_fail == 1:
            visual_failure_count += 1
        
        print(f"label: {label}, plan_fail: {plan_fail}, visual_fail: {visual_fail}")
        print(f"dict_keys({list(state.__dict__.keys())})")
        
        
        # Store target name and label
        targ_name_label[targ_name] = int(label)

        count_label_dict[scene_name] = (int(label), occluder_count, occ_level)
        tgt_bbx_label_dict[scene_name] = (int(label), length, width, height, occ_level)
        
        # Create scene information file after label is determined
        if visualize:
            scene_info_path = f"{scene_vis_dir}/scene_info.txt"
            
            # Get IoU and CD values from completion_metrics if available
            iou, cd = 0.0, 0.0  # Default values
            if scene_name in completion_metrics_dict:
                iou = completion_metrics_dict[scene_name]['iou']
                cd = completion_metrics_dict[scene_name]['cd']
            
            with open(scene_info_path, 'w') as f:
                f.write(f"Scene_ID: {scene_name}\n")
                f.write(f"Occlusion_Level: {occ_level:.4f}\n")
                f.write(f"Num_Occluders: {occluder_count}\n")
                f.write(f"Length: {length:.6f}\n")
                f.write(f"Width: {width:.6f}\n")
                f.write(f"Height: {height:.6f}\n")
                f.write(f"Success: {int(label)}\n")
                f.write(f"Model_Type: {model_type}\n")
                f.write(f"Target_Name: {targ_name}\n")
                f.write(f"IoU: {iou:.6f}\n")
                f.write(f"CD: {cd:.6f}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            print(f"Saved scene info: {scene_info_path}")

        # Generate affordance mesh for all scenes (both success and failure)
        if visualize:
            # Get scene mesh for visualization
            mesh_pose_list = get_mesh_pose_list_from_world(sim.world, object_set)
            scene_mesh = get_scene_from_mesh_pose_list(mesh_pose_list, tgt_id - 1)
            
            # Use visual_mesh as affordance mesh if available, otherwise create one
            if 'visual_mesh' in locals() and visual_mesh is not None:
                affordance_mesh = visual_mesh
            else:
                # Create affordance mesh from grasp planning results
                try:
                    from vgn.utils_visualization import create_affordance_mesh
                    affordance_mesh = create_affordance_mesh(state, grasp_plan_fn, model_type)
                except Exception as affordance_error:
                    print(f"Error creating affordance mesh for {scene_name}: {affordance_error}")
                    affordance_mesh = None
            
            # Save meshes using logger.log_mesh (similar to reference file)
            if affordance_mesh is not None:
                logger.log_mesh(scene_mesh, affordance_mesh, f'{occ_level:.4f}_occ_{scene_name}')
                print(f"Saved scene and affordance meshes for {scene_name}")                    
                # Move the generated .obj files to the scene subdirectory
                try:
                    import shutil
                    # Get the generated file paths from logger
                    generated_scene_path = f"{logger.mesh_dir}/{occ_level:.4f}_occ_{scene_name}_scene.obj"
                    generated_aff_path = f"{logger.mesh_dir}/{occ_level:.4f}_occ_{scene_name}_aff.obj"
                    
                    # Move files to scene directory
                    if os.path.exists(generated_scene_path):
                        shutil.move(generated_scene_path, f"{scene_vis_dir}/scene_mesh.obj")
                        print(f"Moved scene mesh to {scene_vis_dir}/scene_mesh.obj")
                    
                    if os.path.exists(generated_aff_path):
                        shutil.move(generated_aff_path, f"{scene_vis_dir}/affordance_mesh.obj")
                        print(f"Moved affordance mesh to {scene_vis_dir}/affordance_mesh.obj")
                        
                except Exception as move_error:
                    print(f"Error moving mesh files: {move_error}")
                    # Keep the original files in scene directory as fallback
                    scene_mesh_path = f"{scene_vis_dir}/scene_mesh.obj"
                    aff_mesh_path = f"{scene_vis_dir}/affordance_mesh.obj"
                    scene_mesh.export(scene_mesh_path)
                    affordance_mesh.export(aff_mesh_path)
                    print(f"Saved fallback meshes in {scene_vis_dir}")
            else:
                # Save only scene mesh if no affordance mesh available
                scene_mesh_path = f"{scene_vis_dir}/scene_mesh.obj"
                scene_mesh.export(scene_mesh_path)
                print(f"Saved scene mesh for {scene_name}")
            
            # Save GT target mesh if requested
            if vis_gt_target:
                try:
                    # Get target mesh from scene
                    result = get_scene_from_mesh_pose_list(mesh_pose_list, tgt_id - 1, return_target_mesh=True)
                    if result is not None:
                        _, target_mesh = result
                        
                        # Set GT target mesh color to red
                        target_mesh.visual.face_colors = [255, 0, 0, 255]  # Red color with full opacity
                        
                        # Save GT target mesh
                        gt_target_path = f"{scene_vis_dir}/gt_target_mesh.obj"
                        target_mesh.export(gt_target_path)
                        print(f"Saved GT target mesh (red) to {gt_target_path}")
                        
                        # Also save as PLY for point cloud visualization
                        gt_target_ply_path = f"{scene_vis_dir}/gt_target_mesh.ply"
                        target_mesh.export(gt_target_ply_path)
                        print(f"Saved GT target mesh (red) as PLY to {gt_target_ply_path}")
                except Exception as gt_error:
                    print(f"Error saving GT target mesh for {scene_name}: {gt_error}")
            
            # Save predicted target mesh if requested
            if vis_target_mesh_pred:
                try:
                    # Get predicted target mesh from visual_dict
                    if 'completed_target_mesh' in visual_dict and visual_dict['completed_target_mesh'] is not None:
                        predicted_target_mesh = visual_dict['completed_target_mesh']
                        
                        # Save predicted target mesh
                        pred_target_path = f"{scene_vis_dir}/target_mesh_pred.obj"
                        save_open3d_mesh(predicted_target_mesh, pred_target_path)
                        print(f"Saved predicted target mesh to {pred_target_path}")
                        
                        # Also save as PLY for point cloud visualization
                        pred_target_ply_path = f"{scene_vis_dir}/target_mesh_pred.ply"
                        save_open3d_mesh(predicted_target_mesh, pred_target_ply_path)
                        print(f"Saved predicted target mesh as PLY to {pred_target_ply_path}")
                    else:
                        print(f"No predicted target mesh available for {scene_name}")
                        
                except Exception as pred_error:
                    print(f"Error saving predicted target mesh for {scene_name}: {pred_error}")
            
            # Save composed_scene (affordance mesh with grasps) if requested
            if vis_grasps and 'composed_scene' in visual_dict and visual_dict['composed_scene'] is not None:
                try:
                    # Use the composed_scene from grasp_plan_fn (similar to target_sample_offline.py)
                    composed_scene = visual_dict['composed_scene']
                    
                    # Save composed_scene as affordance_with_grasps.obj
                    aff_grasps_path = f"{scene_vis_dir}/affordance_with_grasps.obj"
                    composed_scene.export(aff_grasps_path)
                    print(f"Saved composed scene (affordance with grasps) to {aff_grasps_path}")
                    
                except Exception as grasp_error:
                    print(f"Error saving composed scene for {scene_name}: {grasp_error}")
                    # Fallback: save the original affordance mesh
                    if affordance_mesh is not None:
                        aff_grasps_path = f"{scene_vis_dir}/affordance_with_grasps.obj"
                        affordance_mesh.export(aff_grasps_path)
                        print(f"Saved fallback affordance mesh to {aff_grasps_path}")
            elif vis_grasps:
                print(f"No composed_scene available for visualization in {scene_name}")

        # If success, record success in occlusion bin
        if label != Label.FAILURE:
            occ_level_success_dict = record_occ_level_success(occ_level, occ_level_success_dict)

        # Optionally partial stats
        if num_id % 100 == 0:
            occ_level_sr = cal_occ_level_sr(occ_level_count_dict, occ_level_success_dict)
            curr_count = sum(occ_level_count_dict.values())
            intermediate_result_path = f'{result_path}/intermediate_result.txt'
            with open(intermediate_result_path, 'a') as f:
                f.write(f"current total count:{curr_count}\n")
                for key, val in occ_level_sr.items():
                    f.write(f"{key}:{val}\n")
                f.write('\n')

        # Note: max_scenes limit is handled by pre-selecting scenes, so no need to break here

    # Save target name and label dictionary
    with open(f'{result_path}/targ_name_label.json', 'w') as f:
        json.dump(targ_name_label, f)
    
    # Save count and bounding box dictionaries (similar to target_sample_offline_old.py)
    with open(f'{result_path}/count_label_dict.json', 'w') as f:
        json.dump(count_label_dict, f)
    with open(f'{result_path}/tgt_bbx_label_dict.json', 'w') as f:
        json.dump(tgt_bbx_label_dict, f)
        
    # Calculate success rate only for non-zero count levels
    non_zero_levels = [level for level in occ_level_count_dict if occ_level_count_dict[level] > 0]
    
    # scene_target_mesh_dict was already created during the main processing loop
    
    # Save metrics to meta_evaluations.txt
    with open(f'{result_path}/meta_evaluations.txt', 'w') as f:
        f.write("Scene_ID, Occlusion_Level, Num_Occluders, Length, Width, Height, Success, IoU, CD\n")
        total_scenes = len(count_label_dict)
        success_count = 0
        
        for scene_name, (label, occluder_count, occ_level) in count_label_dict.items():
            success = int(label != Label.FAILURE)
            
            # Use the actual occluder count from the scene data
            num_occluders = occluder_count
            
            # Get length, width, height from tgt_bbx_label_dict if available
            length, width, height = 0.04, 0.04, 0.04  # Default fallback values
            if scene_name in tgt_bbx_label_dict:
                try:
                    _, length, width, height, _ = tgt_bbx_label_dict[scene_name]
                except Exception as e:
                    print(f"Warning: Could not get dimensions from tgt_bbx_label_dict for {scene_name}: {e}")
                    # Try to get from scene_target_mesh_dict as fallback
                    if scene_name in scene_target_mesh_dict:
                        try:
                            target_mesh = scene_target_mesh_dict[scene_name]
                            if target_mesh is not None:
                                # Get bounding box dimensions
                                bounds = target_mesh.bounds
                                dims = bounds[1] - bounds[0]  # [width, height, depth]
                                length, width, height = dims[0], dims[1], dims[2]
                        except Exception as e2:
                            print(f"Warning: Could not calculate dimensions for {scene_name}: {e2}")
                            length, width, height = 0.04, 0.04, 0.04  # Use default
            
            # Get IoU and CD values from completion_metrics if available
            iou, cd = 0.0, 0.0  # Default values
            if scene_name in completion_metrics_dict:
                iou = completion_metrics_dict[scene_name]['iou']
                cd = completion_metrics_dict[scene_name]['cd']
            
            f.write(f"{scene_name}, {occ_level:.4f}, {num_occluders}, {length:.6f}, {width:.6f}, {height:.6f}, {success}, {iou:.6f}, {cd:.6f}\n")
            
            # Count successful grasps
            if success == 1:
                success_count += 1
        
        if total_scenes > 0:
            success_rate = success_count / total_scenes * 100
            f.write(f"\nSuccess Rate: {success_rate:.2f}%\n")
            f.write(f"Total scenes evaluated: {total_scenes}\n")
            f.write(f"Successful grasps: {success_count}\n")
        else:
            # If no scene data available, write a message
            f.write("\nNo scene metrics available. Check if model evaluation is correctly configured.\n")
    
    with open(f'{result_path}/sr_rate.txt', 'w') as f:
        # Write overall stats for all non-zero levels
        total_count = sum(occ_level_count_dict[level] for level in non_zero_levels)
        total_success = sum(occ_level_success_dict[level] for level in non_zero_levels)
        overall_sr = total_success / total_count if total_count > 0 else 0
        
        f.write(f"Overall success rate for non-zero levels: {overall_sr:.4f}\n")
        f.write(f"Total count: {total_count}\n")
        f.write(f"Total success: {total_success}\n\n")
        
        # Write individual stats for each non-zero level
        f.write("Success rates by occlusion level:\n")
        for level in non_zero_levels:
            level_count = occ_level_count_dict[level]
            level_success = occ_level_success_dict[level]
            level_sr = level_success / level_count
            f.write(f"Level {level}: {level_sr:.4f} ({level_success}/{level_count})\n")
        
        # Write all occlusion levels data for reference
        f.write("\nAll occlusion levels data:\n")
        f.write(f"occ_level_count_dict: {occ_level_count_dict}\n")
        f.write(f"occ_level_success_dict: {occ_level_success_dict}\n")

    print("done")
    return overall_sr


class Logger(object):
    """
    Logger handles:
    - Writing data about rounds to rounds.csv
    - Writing data about grasps to grasps.csv
    - Storing scene data (TSDF, pc, etc.) to .npz
    - Saving meshes for visualization
    """
    def __init__(self, root, description, tgt_sample=False):
        self.logdir = root / "visualize"
        self.scenes_dir = self.logdir / "visualize" / "scenes"
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.scenes_dir.mkdir(parents=True, exist_ok=True)

        self.mesh_dir = root / "visualize" / "meshes"
        self.mesh_dir.mkdir(parents=True, exist_ok=True)

        self.rounds_csv_path = self.logdir / "rounds.csv"
        self.grasps_csv_path = self.logdir / "grasps.csv"
        self._create_csv_files_if_needed(tgt_sample)
    def _create_csv_files_if_needed(self, tgt_sample):
        """Make rounds.csv and grasps.csv if they do not exist."""
        if not self.rounds_csv_path.exists():
            io.create_csv(self.rounds_csv_path, ["round_id", "object_count"])

        if not self.grasps_csv_path.exists() and not tgt_sample:
            columns = [
                "round_id",
                "scene_id",
                "qx",
                "qy",
                "qz",
                "qw",
                "x",
                "y",
                "z",
                "width",
                "score",
                "label",
                "integration_time",
                "planning_time",
            ]
            io.create_csv(self.grasps_csv_path, columns)

        if not self.grasps_csv_path.exists() and tgt_sample:
            columns = [
                "round_id",
                "scene_id",
                "qx",
                "qy",
                "qz",
                "qw",
                "x",
                "y",
                "z",
                "width",
                "score",
                "label",
                "occ_level",
                "integration_time",
                "planning_time",
            ]
            io.create_csv(self.grasps_csv_path, columns)

    def last_round_id(self):
        """
        Return the last round_id from rounds.csv, or -1 if empty.
        """
        df = pd.read_csv(self.rounds_csv_path)
        return -1 if df.empty else df["round_id"].max()

    def log_round(self, round_id, object_count):
        """Append a row to rounds.csv."""
        io.append_csv(self.rounds_csv_path, round_id, object_count)

    def log_mesh(self, scene_mesh, aff_mesh, name):
        """Save two OBJ meshes (scene + affordance)."""
        scene_mesh.export(self.mesh_dir / (name + "_scene.obj"))
        aff_mesh.export(self.mesh_dir / (name + "_aff.obj"))

    def log_grasp(
        self,
        round_id=None,
        state=None,
        timings=None,
        grasp=None,
        score=None,
        label=None,
        occ_level=None,
        no_valid_grasp=False
    ):
        """
        Save TSDF/points to .npz, then write one row to grasps.csv.
        If no_valid_grasp=True, log placeholder data.
        """
        if not no_valid_grasp:
            tsdf, points = state.tsdf, np.asarray(state.pc.points)
            scene_id = uuid.uuid4().hex
            scene_path = self.scenes_dir / (scene_id + ".npz")
            np.savez_compressed(scene_path, grid=tsdf.get_grid(), points=points)

            qx, qy, qz, qw = grasp.pose.rotation.as_quat()
            x, y, z = grasp.pose.translation
            width = grasp.width
            label = int(label)
        else:
            scene_id = uuid.uuid4().hex
            qx, qy, qz, qw = 0, 0, 0, 0
            x, y, z = 0, 0, 0
            width = 0
            label = 0
            score = 0

        if not occ_level:
            io.append_csv(
                self.grasps_csv_path,
                round_id,
                scene_id,
                qx,
                qy,
                qz,
                qw,
                x,
                y,
                z,
                width,
                score,
                label,
                timings["integration"],
                timings["planning"],
            )
        else:
            io.append_csv(
                self.grasps_csv_path,
                round_id,
                scene_id,
                qx,
                qy,
                qz,
                qw,
                x,
                y,
                z,
                width,
                score,
                label,
                occ_level,
                timings["integration"],
                timings["planning"],
            )


class Data(object):
    """
    Helps load logs from a folder with:
    - rounds.csv
    - grasps.csv
    - a 'scenes' folder (with .npz of scene states).
    """
    def __init__(self, logdir):
        self.logdir = logdir
        self.rounds = pd.read_csv(logdir / "rounds.csv")
        self.grasps = pd.read_csv(logdir / "grasps.csv")

    def num_rounds(self):
        return len(self.rounds.index)

    def num_grasps(self):
        return len(self.grasps.index)

    def success_rate(self):
        """Percent of successful grasps (label == 1)."""
        return self.grasps["label"].mean() * 100

    def percent_cleared(self):
        """
        Ratio of total successful grasps to total objects, aggregated over rounds.
        """
        df = (
            self.grasps[["round_id", "label"]]
            .groupby("round_id")
            .sum()
            .rename(columns={"label": "cleared_count"})
            .merge(self.rounds, on="round_id")
        )
        return df["cleared_count"].sum() / df["object_count"].sum() * 100

    def avg_planning_time(self):
        return self.grasps["planning_time"].mean()

    def read_grasp(self, i):
        """
        Return the i-th grasp, including:
         - scene point cloud
         - grasp pose
         - predicted score
         - success/failure label
        """
        scene_id, grasp, label = io.read_grasp(self.grasps, i)
        score = self.grasps.loc[i, "score"]
        scene_data = np.load(self.logdir / "scenes" / (scene_id + ".npz"))
        return scene_data["points"], grasp, score, label

def validate_point_count(count):
    """Validate point sampling count.
    
    Args:
        count: number of points to sample
        
    Raises:
        ValueError: if count is invalid
    """
    if not isinstance(count, int) or count <= 0:
        raise ValueError(f"Point count must be positive integer, got {count}")



def segment_with_sam(sam_predictor, rgb_image, mask_targ, tgt_id, scene_name, scene_vis_dir=None, sam_results_file=None):
    """
    Use SAM to segment target object from RGB image.
    
    Args:
        sam_predictor: SAM predictor instance
        rgb_image: RGB image array (H, W, 3)
        mask_targ: Ground truth target mask (H, W)
        tgt_id: Target object ID
        scene_name: Scene name for logging
        scene_vis_dir: Directory to save visualization (optional)
        sam_results_file: File path to save segmentation results (optional)
        
    Returns:
        sam_mask: SAM-generated target mask (H, W)
    """
    try:
        # Convert RGB image to the format expected by SAM (0-255 uint8)
        if rgb_image.dtype != np.uint8:
            rgb_image = (rgb_image * 255).astype(np.uint8)
        
        # Set image for SAM
        sam_predictor.set_image(rgb_image)
        
        # Find a random point within the ground truth target mask as prompt
        target_pixels = np.where(mask_targ > 0)
        if len(target_pixels[0]) == 0:
            print(f"Warning: No target pixels found in mask for {scene_name}")
            # Save failure result
            if sam_results_file:
                with open(sam_results_file, 'a') as f:
                    f.write(f"{scene_name}, 0.0000, 0.0000, 0\n")
            return mask_targ  # Return original mask as fallback
        
        # Randomly select a point within the target mask
        random_idx = random.randint(0, len(target_pixels[0]) - 1)
        prompt_point = [target_pixels[1][random_idx], target_pixels[0][random_idx]]  # [x, y]
        
        print(f"Using prompt point for SAM: {prompt_point} for scene {scene_name}")
        
        # Generate SAM mask with point prompt
        sam_masks, scores, logits = sam_predictor.predict(
            point_coords=np.array([prompt_point]),
            point_labels=np.array([1]),  # Positive point
            multimask_output=False
        )
        
        sam_mask = sam_masks[0].astype(np.float32)
        
        # Calculate segmentation metrics
        gt_mask_binary = (mask_targ > 0).astype(np.uint8)
        sam_mask_binary = (sam_mask > 0.5).astype(np.uint8)
        
        # Calculate IoU (Intersection over Union)
        intersection = np.logical_and(gt_mask_binary, sam_mask_binary)
        union = np.logical_or(gt_mask_binary, sam_mask_binary)
        iou = np.sum(intersection) / (np.sum(union) + 1e-8)
        
        # Calculate Dice coefficient
        dice = (2 * np.sum(intersection)) / (np.sum(gt_mask_binary) + np.sum(sam_mask_binary) + 1e-8)
        
        # Determine success (IoU > 0.5 threshold)
        success = 1 if iou > 0.5 else 0
        
        print(f"SAM segmentation metrics for {scene_name}: IoU={iou:.4f}, Dice={dice:.4f}, Success={success}")
        
        # Save results to file
        if sam_results_file:
            with open(sam_results_file, 'a') as f:
                f.write(f"{scene_name}, {iou:.4f}, {dice:.4f}, {success}\n")
        
        # Save visualization if requested
        if scene_vis_dir:
            try:
                import matplotlib.pyplot as plt
                
                # Create visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original RGB image
                axes[0].imshow(rgb_image)
                axes[0].plot(prompt_point[0], prompt_point[1], 'ro', markersize=10)
                axes[0].set_title('RGB Image with Prompt Point')
                axes[0].axis('off')
                
                # Ground truth mask
                axes[1].imshow(mask_targ, cmap='gray')
                axes[1].set_title('Ground Truth Mask')
                axes[1].axis('off')
                
                # SAM prediction
                axes[2].imshow(sam_mask, cmap='gray')
                axes[2].set_title(f'SAM Prediction (IoU: {iou:.3f})')
                axes[2].axis('off')
                
                plt.tight_layout()
                sam_vis_path = f"{scene_vis_dir}/sam_segmentation.png"
                plt.savefig(sam_vis_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"Saved SAM segmentation visualization to {sam_vis_path}")
                
            except Exception as vis_error:
                print(f"Error saving SAM visualization: {vis_error}")
        
        return sam_mask
        
    except Exception as e:
        print(f"Error in SAM segmentation for {scene_name}: {e}")
        # Save error result
        if sam_results_file:
            with open(sam_results_file, 'a') as f:
                f.write(f"{scene_name}, 0.0000, 0.0000, 0\n")
        return mask_targ  # Return original mask as fallback


def generate_targo_input_data(sim, path_to_npz, tgt_id, scene_name, processed_scenes_targo_path, 
                             use_sam=False, sam_predictor=None, scene_vis_dir=None, sam_results_file=None):
    """
    Generate TARGO input data including scene_no_targ_pc, targ_pc, etc.
    Based on generate_scenes_targo_from_mesh_pose.py
    
    Args:
        sim: Simulation instance
        path_to_npz: Path to scene npz file
        tgt_id: Target object ID
        scene_name: Scene name
        processed_scenes_targo_path: Path to save processed data
        use_sam: Whether to use SAM for segmentation
        sam_predictor: SAM predictor instance
        scene_vis_dir: Directory to save visualization
        
    Returns:
        tuple: (scene_no_targ_pc, targ_pc, targ_grid, occ_level, tsdf_targ)
    """
    try:
        base_dir = os.path.dirname(processed_scenes_targo_path)
        occ_level_dict_path = os.path.join(base_dir, 'occ_level_dict.json')
        occ_level_dict = json.load(open(occ_level_dict_path))
        # Load scene data
        scene_data = np.load(path_to_npz, allow_pickle=True)
        
        # Get depth images and segmentation
        depth_imgs = scene_data['depth_imgs']
        extrinsics = scene_data['extrinsics']
        mask_targ = scene_data['mask_targ']
        mask_scene = scene_data['mask_scene']
        
        # Use SAM for segmentation if requested
        if use_sam and sam_predictor is not None:
            try:
                # Generate RGB image for SAM
                from vgn.perception import camera_on_sphere
                origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, sim.size / 3])
                r = 2 * sim.size
                theta = np.pi / 3.0
                phi = - np.pi / 2.0
                extrinsic = camera_on_sphere(origin, r, theta, phi)
                rgb, _, _ = sim.camera.render_with_seg(extrinsic)
                
                # Use SAM to segment target
                sam_mask = segment_with_sam(sam_predictor, rgb, mask_targ[0], tgt_id, scene_name, scene_vis_dir, sam_results_file)
                
                # Replace ground truth mask with SAM mask
                mask_targ = np.expand_dims(sam_mask, axis=0)  # Add batch dimension
                print(f"Using SAM segmentation for scene {scene_name}")
                
            except Exception as sam_error:
                print(f"Error using SAM segmentation for {scene_name}: {sam_error}")
                print("Falling back to ground truth segmentation")
        else:
            print(f"Using ground truth segmentation for scene {scene_name}")
        
        # Generate point clouds for target and scene using reconstruct_40_pc
        from vgn.perception import (
            reconstruct_40_pc, 
            depth_to_point_cloud, 
            depth_to_point_cloud_no_specify,
            remove_A_from_B
        )
        pc_targ = reconstruct_40_pc(sim, (depth_imgs * mask_targ).astype(np.float32), extrinsics)
        pc_scene = reconstruct_40_pc(sim, (depth_imgs * mask_scene).astype(np.float32), extrinsics)
        
        if len(pc_targ.points) == 0:
            print(f"Warning: No target points found for scene {scene_name}")
            return None, None, None, None
            
        # Remove target points from scene to get scene_no_targ_pc
        scene_no_targ_pc = remove_A_from_B(
            np.asarray(pc_targ.points, dtype=np.float32),
            np.asarray(pc_scene.points, dtype=np.float32)
        )
        
        pc_targ_depth_side_c = depth_to_point_cloud(depth_imgs[0], mask_targ[0],
                                                    sim.camera.intrinsic.K, extrinsics[0], 2048)

        pc_targ_depth_side_c = pc_targ_depth_side_c / 0.3 - 0.5
        
        # Convert to numpy arrays and normalize
        targ_pc = np.asarray(pc_targ.points, dtype=np.float32)
        targ_pc = targ_pc / 0.3 - 0.5  # Normalize to [-0.5, 0.5]
        scene_no_targ_pc = scene_no_targ_pc / 0.3 - 0.5  # Normalize to [-0.5, 0.5]
        
        pc_targ_depth_side_c = torch.from_numpy(pc_targ_depth_side_c).unsqueeze(0)   # (1, N, 3)
        pc_targ_depth_side_c = filter_and_pad_point_clouds(pc_targ_depth_side_c)     # (1, 2048, 3)
        pc_targ_depth_side_c = pc_targ_depth_side_c.squeeze(0).numpy()         
        # Generate target grid using point_cloud_to_tsdf from utils_giga
        # from src.utils_giga import point_cloud_to_tsdf
        # try:
            # Convert targ_pc back to world coordinates for TSDF generation
        # targ_pc_world = (targ_pc + 0.5) * 0.3  # Convert from [-0.5, 0.5] to world coordinates
        # Generate TSDF from completed point cloud
        # targ_grid = point_cloud_to_tsdf(targ_pc_world)
        # targ_grid = tsdf_targ
        
        # print(f"Generated TSDF from completed target point cloud. TSDF shape: {targ_grid.shape}")
            

        from vgn.perception import create_tsdf
        tsdf_targ = create_tsdf(sim.size, 40, (depth_imgs * mask_targ).astype(np.float32), sim.camera.intrinsic, extrinsics)
        targ_grid = tsdf_targ.get_grid()
        print(f"Generated TSDF from completed target point cloud. TSDF shape: {targ_grid.shape}")
        print("Using fallback TSDF generation method")
        
        occ_level = occ_level_dict.get(scene_name, 0.0)
        ## [-0.5, 0.5]
        processed_data = {
            'scene_no_targ_pc': scene_no_targ_pc,
            'targ_pc': pc_targ_depth_side_c.astype(np.float32),
            'targ_grid': targ_grid,
            'occ_level': occ_level,
            'scene_name': scene_name,
            'tgt_id': tgt_id,
            # 'targ_depth_pc': pc_targ_depth_side_c.astype(np.float32),
        }
        
        processed_file_path = os.path.join(processed_scenes_targo_path, f'{scene_name}.npz')
        np.savez(processed_file_path, **processed_data)
        print(f"Saved TARGO processed data to {processed_file_path}")
        
        return scene_no_targ_pc, pc_targ_depth_side_c.astype(np.float32), targ_grid, occ_level, tsdf_targ
        
    except Exception as e:
        print(f"Error generating TARGO input data for scene {scene_name}: {e}")
        return None, None, None, None, None
