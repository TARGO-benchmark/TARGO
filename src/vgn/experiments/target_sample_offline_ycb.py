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
from datetime import datetime
from src.vgn import io  # For CSV creation and I/O
from src.vgn.grasp import *
from src.vgn.ConvONets.eval import MeshEvaluator
from src.vgn.simulation import ClutterRemovalSim
from src.vgn.utils.transform import Rotation, Transform
from src.vgn.utils.implicit import (
    get_mesh_pose_list_from_world,
    get_scene_from_mesh_pose_list
)
from src.vgn.perception import camera_on_sphere
# Debug / logging utilities
from src.utils_targo import (
    record_occ_level_count,
    record_occ_level_success,
    cal_occ_level_sr,
    save_point_cloud_as_ply,
    generate_and_transform_grasp_meshes,
    compute_chamfer_and_iou
)

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
    type='targo',
    test_root=None,
    occ_level_dict_path=None,
    model_type=None,
    hunyun2_path=None,
    hunyuan3D_ptv3=False,
    hunyuan3D_path=None,
    video_recording=True,
    target_file_path=None,  
    data_type='ycb',
    max_scenes=0,  # Add parameter to limit number of scenes (0 means no limit)
):
    # Initialize the simulation
    sim = ClutterRemovalSim(
        scene, object_set,
        gui=sim_gui,
        add_noise=add_noise,
        sideview=sideview,
        test_root=test_root,
        egl_mode=False
    )
    logger = Logger(logdir, description, tgt_sample=True)

    # Track planning times
    planning_times, total_times = [], []

    # Paths for test data
    test_mesh_pose_list = f'{test_root}/mesh_pose_dict/'
    test_scenes = f'{test_root}/scenes/'
    processed_scenes_path = f'{test_root}/processed_scenes'
    if not os.path.exists(processed_scenes_path):
        os.makedirs(processed_scenes_path)
    processed_scenes_feat_path = f'{test_root}/processed_scenes_feat'
    if not os.path.exists(processed_scenes_feat_path):
        os.makedirs(processed_scenes_feat_path)

    processed_scenes_ptv3_clip_path = f'{test_root}/processed_scenes_ptv3_clip'
    if not os.path.exists(processed_scenes_ptv3_clip_path):
        os.makedirs(processed_scenes_ptv3_clip_path)
    
    processed_scenes_feat_ptv3_clip_path = f'{test_root}/processed_scenes_ptv3_clip_feat'
    if not os.path.exists(processed_scenes_feat_ptv3_clip_path):
        os.makedirs(processed_scenes_feat_ptv3_clip_path)

    # Dictionaries to track occlusion stats
    occ_level_count_dict = init_occ_level_count_dict()
    occ_level_success_dict = init_occ_level_success_dict()
    offline_occ_level_dict = {}

    
    plan_failure_count = 0
    visual_failure_count = 0

    count_label_dict = {}
    height_label_dict = {}
    tgt_bbx_label_dict = {}
    skip_dict = {}
    targ_name_label = {}  # Dictionary to store target name and label
    #
    # Dictionary to store metrics for each scene
    scene_metrics = {}

    hunyun2_scene_list = []
    # Create a list of all valid scene folder names
    if hunyun2_path and model_type == 'targo_hunyun2':
        for folder_name in os.listdir(hunyun2_path):
            # Check if it's a directory
            if os.path.isdir(os.path.join(hunyun2_path, folder_name)):
                # Check if the reconstructed target object exists
                scene_path = os.path.join(hunyun2_path, folder_name, 'reconstruction', 'gt_targ_obj.ply')
                if os.path.exists(scene_path):
                    hunyun2_scene_list.append(folder_name)
        
        print(f"Found {len(hunyun2_scene_list)} valid scenes in hunyun2 path")

    occ_level_dict = json.load(open(occ_level_dict_path))
    # Loop over the test set
    # occ_level =
    for num_id, curr_mesh_pose_list in enumerate(os.listdir(test_mesh_pose_list)):
        # if num_id == 88:
        #     continue
        # if num_id == 95:
        #     continue
        # if num_id < 95:
        #     continue
        path_to_npz = os.path.join(test_scenes, curr_mesh_pose_list)
        scene_name = curr_mesh_pose_list[:-4]
        occ_level = occ_level_dict[scene_name]
        root_path = scene_name.split('_')[0]
        # curr_processed_scene_ptv3_clip_path = os.path.join(processed_scenes_ptv3_clip_path, scene_name)
        curr_processed_scene_ptv3_clip_path = f'{processed_scenes_ptv3_clip_path}/{scene_name}.npz'
        curr_processed_scene_feat_ptv3_clip_path = f'{processed_scenes_feat_ptv3_clip_path}/{scene_name}.npz'
        # if not os.path.exists(cur_processed_scene_ptv3_clip_path):
        #     os.makedirs(cur_processed_scene_ptv3_clip_path)
        # if not os.path.exists(cur_processed_scene_feat_ptv3_clip_path):
            # os.makedirs(cur_processed_scene_feat_ptv3_clip_path)
        # if scene_name != 'c76d32610ea24da781fb9b7f9beee8d4_c_5':
        #     continue
        if scene_name not in occ_level_dict:
            # os.remove(os.path.join(test_mesh_pose_list, curr_mesh_pose_list))
            # os.remove(path_to_npz)
            continue

        if type == 'targo_hunyun2' and scene_name not in hunyun2_scene_list:
            continue

        # if hunyun2_path:
        #     scene_path = hunyun2_path + '/' + scene_name + '/reconstruction/gt_targ_obj.ply'
        #     if not os.path.exists(scene_path):
        #         continue

        # Prepare simulator
        sim.world.reset()
        sim.world.set_gravity([0.0, 0.0, -9.81])
        sim.draw_workspace()
        sim.save_state()

        # Manually adjust boundaries
        sim.lower = np.array([0.02, 0.02, 0.055])
        sim.upper = np.array([0.28, 0.28, 0.30000000000000004])

        tgt_id = int(scene_name[-1])  
        start_time = time.time()

        occluder_heights = []

        mp_data = np.load(
            os.path.join(test_mesh_pose_list, curr_mesh_pose_list),
            allow_pickle=True
        )["pc"]

        # Place objects
        for obj_id, mesh_info in enumerate(mp_data.item().values()):
            pose = Transform.from_matrix(mesh_info[2])
            if data_type == 'ycb':
                if mesh_info[0].split('/')[-1] == 'plane.obj':
                    urdf_path = mesh_info[0].replace(".obj", ".urdf")
                else:
                    file_id = mesh_info[0].split('/')[-1].replace("_textured.obj", "").replace(".obj", "")
                    urdf_base_dir = "./data/maniskill_ycb/mani_skill2_ycb/collisions"
                    urdf_path = f"{urdf_base_dir}/{file_id}.urdf"
                
                # Fix mesh path from old path to new path
                if mesh_info[0].startswith('./'):
                    mesh_info = list(mesh_info)
                    mesh_info[0] = mesh_info[0].replace('./', './')
                    mesh_info = tuple(mesh_info)
            
            elif data_type == 'acronym':
                # Extract file ID part (without path and extension)
                file_basename = os.path.basename(mesh_info[0])
                if file_basename == 'plane.obj':
                    urdf_path = mesh_info[0].replace(".obj", ".urdf")
                else:
                    file_id = file_basename.replace("_textured.obj", "").replace(".obj", "")
                    
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
                ycb_object_category_dict = json.load(open('./targo_eval_results/stastics_analysis/ycb_prompt_dict.json'))
                target_category = ycb_object_category_dict[targ_name]
                target_mesh_gt = tri_mesh
        
        end_time = time.time()
        if len(occluder_heights) == 0:
            relative_height = tgt_height
        else:
            relative_height = tgt_height - np.max(occluder_heights)

        # Skip if target is in skip_list
        if targ_name in skip_list:
            print(f"Skipping {targ_name} as it's in the skip list")
            continue

        # print(f"load {num_id}-th {scene_name} took {end_time - start_time:.2f}s")

        # Acquire data for shape completion
        timings = {}
        start_time = time.time()

        if model_type == 'FGC-GraspNet' or model_type == 'AnyGrasp':
            tsdf, timings["integration"], scene_pc,target_pc, extrinsic, occ_level = \
            sim.acquire_single_tsdf_target_grid(
                path_to_npz,
                tgt_id,
                40,
                model_type,  
                curr_mesh_pose_list=scene_name,
            )
            state = argparse.Namespace(
                    tsdf=tsdf,
                    scene_pc=scene_pc,
                    target_pc=target_pc,
                    occ_level=occ_level,
                    type=model_type,
                    extrinsic=extrinsic,
                    timings=timings
            )
            
        elif model_type == 'targo' or model_type == 'targo_full_targ' or model_type == 'targo_hunyun2' or model_type == 'targo_ptv3':
            tsdf, timings["integration"], scene_no_targ_pc, targ_pc, targ_grid, occ_level = \
            sim.acquire_single_tsdf_target_grid(
                path_to_npz,
                tgt_id,
                40,
                model_type,  
                curr_mesh_pose_list=scene_name,
            )
            state = argparse.Namespace(
                        tsdf=tsdf,
                        scene_no_targ_pc=scene_no_targ_pc,
                        targ_grid=targ_grid,
                        targ_pc=targ_pc,
                        occ_level=occ_level,
                        type=model_type,
                        timings=timings
                    )
            
        elif model_type == 'ptv3_scene':
            tsdf, timings["integration"], scene_no_targ_pc, complete_targ_pc, complete_targ_tsdf, targ_grid, occ_level, iou_value, cd_value = \
            sim.acquire_single_tsdf_target_grid_ptv3_scene(
                path_to_npz,
                tgt_id,
                40,
                model_type,  
                curr_mesh_pose_list=scene_name,
                hunyuan3D_ptv3=hunyuan3D_ptv3,
                hunyuan3D_path=hunyuan3D_path,
            )
            state = argparse.Namespace(
                    tsdf=tsdf,
                    scene_no_targ_pc=scene_no_targ_pc,
                    complete_targ_pc=complete_targ_pc,
                    complete_targ_tsdf=complete_targ_tsdf,
                    targ_grid=targ_grid,
                    occ_level=occ_level,
                    type=model_type,
                    iou=iou_value,
                    cd=cd_value
                )
            
        elif model_type == 'ptv3_clip':
            if not os.path.exists(curr_processed_scene_ptv3_clip_path):
                try:
                    tsdf, timings["integration"], scene_no_targ_pc, complete_targ_pc, complete_targ_tsdf, targ_grid, occ_level, iou_value, cd_value, vis_dict, target_clip_features, scene_clip_features = \
                    sim.acquire_single_tsdf_target_grid_ptv3_clip(
                        path_to_npz,
                        tgt_id,
                        40,
                        model_type,  
                        curr_mesh_pose_list=scene_name,
                        hunyuan3D_ptv3=hunyuan3D_ptv3,
                        hunyuan3D_path=hunyuan3D_path,
                        target_category=target_category,
                    )
                    vis_path = f'{logdir}/scene_vis/{scene_name}'
                    os.makedirs(vis_path, exist_ok=True)
                    state = argparse.Namespace(
                            # tsdf=tsdf,
                            scene_no_targ_pc=scene_no_targ_pc,
                            complete_targ_pc=complete_targ_pc,
                            complete_targ_tsdf=complete_targ_tsdf,
                            targ_grid=targ_grid,
                            occ_level=occ_level,
                            type=model_type,
                            timings=timings,
                            voxel_size = tsdf.voxel_size,
                            size = tsdf.size,
                            # vis_path=vis_path,
                            iou=iou_value,
                            cd=cd_value,
                            vis_dict=vis_dict,
                            vis_path=vis_path,
                            target_clip_features=target_clip_features,
                            scene_clip_features=scene_clip_features
                        )
                    np.savez(curr_processed_scene_ptv3_clip_path, **state.__dict__)
                except Exception as e:
                    print(f"ERROR: Data corruption in scene {scene_name}: {str(e)}")
                    print(f"Skipping scene {scene_name}...")
                    continue
            else:
                state = np.load(curr_processed_scene_ptv3_clip_path, allow_pickle=True)
                state = argparse.Namespace(**state)
                vis_path = f'{logdir}/scene_vis/{scene_name}'
                os.makedirs(vis_path, exist_ok=True)
                state.vis_path = vis_path

        elif model_type in ("vgn", "giga_aff", "giga", "giga_hr"):
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
        elif model_type == 'AnyGrasp_full_targ' or model_type == 'FGC_full_targ':
            # Get both scene without target and target point clouds for AnyGrasp_full_targ
            tsdf, timings["integration"], scene_no_targ_pc, targ_pc, targ_grid, occ_level = \
            sim.acquire_single_tsdf_target_grid(
                path_to_npz,
                tgt_id,
                40,
                model_type,
                # 'targo',  # Use targo type to get separate scene_no_targ_pc and targ_pc
                curr_mesh_pose_list=scene_name,
            )
            
            # Set scene and target point cloud
            state = argparse.Namespace(
                    tsdf=tsdf,
                    scene_no_targ_pc=scene_no_targ_pc,
                    scene_pc=scene_no_targ_pc,  # For AnyGrasp compatibility
                    targ_grid=targ_grid,
                    targ_pc=targ_pc,
                    target_pc=targ_pc,  # For AnyGrasp compatibility 
                    occ_level=occ_level,
                    type=model_type,
                    timings=timings,
                )
        end_time = time.time()
        # print(f"acquire {num_id}-th {scene_name} took {end_time - start_time:.2f}s for shape-completion input")

        # Record occlusion
        occ_level_count_dict = record_occ_level_count(occ_level, occ_level_count_dict)
        offline_occ_level_dict[scene_name] = occ_level

        # Skip if occlusion > 0.9
        if occ_level >= 0.9:
            print("skip")
            skip_dict[scene_name] = occ_level
            continue

        # Plan the grasp
        # if visualize:
        #     visual_dict = {'mesh_name': scene_name, 'mesh_dir': logger.mesh_dir}
        #     mesh_pose_list = get_mesh_pose_list_from_world(sim.world, object_set)
        #     scene_mesh = get_scene_from_mesh_pose_list(mesh_pose_list, tgt_id - 1)
        #     grasps, scores, timings["planning"], visual_mesh = grasp_plan_fn(state, scene_mesh, visual_dict)

        #     # Render snapshot
        #     origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, sim.size / 3])
        #     r = 2 * sim.size
        #     theta = np.pi / 3.0
        #     phi = - np.pi / 2.0
        #     extrinsic = camera_on_sphere(origin, r, theta, phi)
        #     rgb, _, _ = sim.camera.render_with_seg(extrinsic)
        #     output_path = f'{logger.mesh_dir}/{occ_level}_occ_{scene_name}_rgb.png'
        #     plt.imsave(output_path, rgb)
        # else:
        #     # time_begin = time.time()
        #     grasps, scores, timings["planning"] = grasp_plan_fn(state)
        #     # time_end = time.time()
        #     # print(f"planning time={time_end - time_begin}")
        
        mesh_pose_list = get_mesh_pose_list_from_world(sim.world, object_set)
        result = get_scene_from_mesh_pose_list(mesh_pose_list, tgt_id - 1, return_target_mesh=True)
        
        if result is None:
            print(f"Error: Could not get scene and target mesh for scene {scene_name}, skipping")
            continue
        
        scene_mesh, target_mesh = result
        
        if model_type == 'AnyGrasp_full_targ' or model_type == 'FGC_full_targ':
            targ_full_pc = target_mesh.sample(4096)
            state.targ_full_pc = targ_full_pc
        # if model_type != 'vgn':
        #     grasps, scores, timings["planning"], visual_dict = grasp_plan_fn(state, scene_mesh)
        # else:
        # grasps, scores, timings["planning"] = grasp_plan_fn(state, scene_mesh)

        if model_type == 'targo' or model_type == 'targo_full_targ' or model_type == 'targo_hunyun2' or model_type == 'targo_ptv3' or model_type == 'ptv3_scene':
            grasps, scores, timings["planning"], cd, iou = grasp_plan_fn(state, scene_mesh, hunyun2_path=hunyun2_path, scene_name=scene_name, cd_iou_measure=True, target_mesh_gt=target_mesh_gt)
            # Store metrics for this scene
            scene_metrics[scene_name] = {
                "target_name": targ_name,
                "occlusion_level": float(occ_level),
                "cd": float(cd),
                "iou": float(iou)
            }
        elif model_type == 'ptv3_clip':
            # 专门为 ptv3_clip 模型调用
            grasps, scores, timings["planning"], cd, iou = grasp_plan_fn(state, scene_mesh, clip_feat_path=curr_processed_scene_feat_ptv3_clip_path, hunyun2_path=hunyun2_path, scene_name=scene_name, cd_iou_measure=True, target_mesh_gt=target_mesh_gt)
            # Store metrics for this scene
            scene_metrics[scene_name] = {
                "target_name": targ_name,
                "occlusion_level": float(occ_level),
                "cd": float(cd),
                "iou": float(iou)
            }
        elif model_type == 'FGC-GraspNet' or model_type == 'AnyGrasp' or model_type == 'AnyGrasp_full_targ' or model_type == 'FGC_full_targ': 
            grasps, scores, timings["planning"], g1b_vis_dict, cd, iou = grasp_plan_fn(state, scene_mesh, hunyun2_path=hunyun2_path, scene_name=scene_name, cd_iou_measure=True, target_mesh_gt=target_mesh_gt)
            # Store metrics for this scene
            scene_metrics[scene_name] = {
                "target_name": targ_name,
                "occlusion_level": float(occ_level),
                "cd": float(cd),
                "iou": float(iou)
            }

        elif model_type == 'giga':
            grasps, scores, timings["planning"], cd, iou = grasp_plan_fn(state, scene_mesh, target_mesh_gt=target_mesh_gt)
            # Store metrics for this scene
            scene_metrics[scene_name] = {
                "target_name": targ_name,
                "occlusion_level": float(occ_level),
                "cd": float(cd),
                "iou": float(iou)
            }
        elif model_type == 'vgn':
            grasps, scores, timings["planning"], cd, iou = grasp_plan_fn(state, scene_mesh, cd_iou_measure=True, target_mesh_gt=target_mesh_gt)
            scene_metrics[scene_name] = {
                "target_name": targ_name,
                "occlusion_level": float(occ_level),
                "cd": float(cd),
                "iou": float(iou)
            }   

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
            count_label_dict[scene_name] = (int(label), len(occluder_heights), occ_level)
            height_label_dict[scene_name] = (int(label), relative_height, occ_level)
            tgt_bbx_label_dict[scene_name] = (int(label), length, width, height, occ_level)
            targ_name_label[targ_name] = int(label)
            
            # Update failure count
            plan_failure_count += 1
            
            # Update scene metrics
            if scene_name in scene_metrics:
                scene_metrics[scene_name]["success"] = 0
                
            # Continue to next scene
            continue

        grasp, score = grasps[0], scores[0]

        # Combine scene_no_targ_pc + targ_pc if desired, for local debugging
        # scene_pc = None
        # if hasattr(state, 'scene_no_targ_pc') and hasattr(state, 'targ_pc'):
        #     # Merge for any local inspection
        #     scene_pc = np.concatenate([state.scene_no_targ_pc, state.targ_pc], axis=0)
        #     scene_pc = (scene_pc + 0.5) * 0.3  # revert from [-0.5,0.5] to real world coords

        # if scene_pc is not None:
        #     generate_and_transform_grasp_meshes(
        #         grasp, scene_pc,
        #         './demo_targo'
        #     )

        grasp.width = sim.gripper.max_opening_width

        # Execute grasp in simulation
        ## TODO: start video recording
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
                        
                        # Fix mesh path from old path to new path
                        if mesh_info[0].startswith('./'):
                            mesh_info = list(mesh_info)
                            mesh_info[0] = mesh_info[0].replace('./', './')
                            mesh_info = tuple(mesh_info)
                    
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
                
                # Visualize grasps for FGC-GraspNet or AnyGrasp models
                if model_type in ['FGC-GraspNet', 'AnyGrasp'] and 'g1b_vis_dict' in locals() and g1b_vis_dict is not None:
                    from src.vgn.detection_implicit import vis_grasps_target
                    target_cloud = g1b_vis_dict.get('target_pc')
                    scene_cloud = g1b_vis_dict.get('scene_pc')
                    target_gg = g1b_vis_dict.get('target_gg')
                    if target_cloud is not None and scene_cloud is not None and target_gg is not None:
                        is_anygrasp = model_type == 'AnyGrasp'
                        is_fgc = model_type == 'FGC-GraspNet'
                        # Use video_path and video_filename for output location
                        output_prefix = os.path.join(video_path, video_filename)
                        vis_grasps_target(target_gg, target_cloud, scene_cloud, anygrasp=is_anygrasp, fgc=is_fgc, output_prefix=output_prefix)
                        print(f"Visualized {model_type} grasps in {output_prefix}")
                
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
        
        # Update metrics with grasp success/failure
        if scene_name in scene_metrics:
            scene_metrics[scene_name]["success"] = int(label != Label.FAILURE)

        # if label == Label.FAILURE and visualize:
        # # if label == Label.FAILURE:
        #     ## create a scene
        #     scene_path = logdir / scene_name
        #     scene_path.mkdir(parents=True, exist_ok=True)
        #     metadata_path = scene_path / "scene_metadata.txt"

        #     chamfer_distance, iou = compute_chamfer_and_iou(target_mesh, visual_dict['completed_targ_pc'], mesh_path=scene_path)

        #     meta = {
        #         'scene_id': scene_name,
        #         'cd': float(chamfer_distance) * 1000,  # Convert float32 to Python float
        #         'iou': float(iou) * 100,  # Convert float32 to Python float
        #         'success': 0,
        #         'time': datetime.now().isoformat(),
        #         'occlusion': float(occ_level * 100),  # Convert float32 to Python float
        #     }

        #     # Save to meta_data_path
        #     with open(metadata_path, 'w') as f:
        #         json.dump(meta, f)

        #     # visual_dict = {'mesh_name': scene_name, 'mesh_dir': logger.mesh_dir}
        #     # mesh_pose_list = get_mesh_pose_list_from_world(sim.world, object_set)
        #     # scene_mesh = get_scene_from_mesh_pose_list(mesh_pose_list, tgt_id - 1)
        #     # grasps, scores, timings["planning"], visual_mesh = grasp_plan_fn(state, scene_mesh, visual_dict)

        #     # Render snapshot
        #     origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, sim.size / 3])
        #     r = 2 * sim.size
        #     theta = np.pi / 3.0
        #     phi = - np.pi / 2.0
        #     extrinsic = camera_on_sphere(origin, r, theta, phi)
        #     rgb, _, _ = sim.camera.render_with_seg(extrinsic)
        #     # output_path = f'{logger.mesh_dir}/{occ_level}_occ_{scene_name}_rgb.png'
        #     img_path = scene_path / "scene_rgb.png"
        #     plt.imsave(img_path, rgb)
        #     visual_dict['composed_scene'].export(f'{scene_path}/composed_scene.obj')
        #     visual_dict['affordance_visual'].export(f'{scene_path}/affordance_visual.obj')

        #     save_point_cloud_as_ply(visual_dict['completed_targ_pc'], scene_path / "completed_targ_pc.ply")
        #     transformed_pc = (visual_dict['targ_pc'] + 0.5) * 0.3
        #     save_point_cloud_as_ply(transformed_pc, scene_path / "targ_pc.ply")
        #     # save_point_cloud_as_ply(transformed_pc, scene_path / "targ_pc.ply")

        #     gt_targ_pc, _ = trimesh.sample.sample_surface(target_mesh, count=2048)
        #     gt_targ_pc = (gt_targ_pc / 0.3) - 0.5
        #     save_point_cloud_as_ply(gt_targ_pc, scene_path / "gt_targ_pc.ply")
            

        count_label_dict[scene_name] = (int(label), len(occluder_heights), occ_level)
        height_label_dict[scene_name] = (int(label), relative_height, occ_level)
        tgt_bbx_label_dict[scene_name] = (int(label), length, width, height, occ_level)

        # If success, record success in occlusion bin
        if label != Label.FAILURE:
            occ_level_success_dict = record_occ_level_success(occ_level, occ_level_success_dict)
            # if visualize:
            #     logger.log_mesh(scene_mesh, visual_mesh, f'{occ_level}_occ_{scene_name}')

        # Optionally partial stats
        # if all(v > 0 for v in occ_level_count_dict.values()) and (num_id % 100 == 0):
        if num_id % 100 == 0:
            occ_level_sr = cal_occ_level_sr(occ_level_count_dict, occ_level_success_dict)
            curr_count = sum(occ_level_count_dict.values())
            intermediate_result_path = f'{result_path}/intermediate_result.txt'
            with open(intermediate_result_path, 'a') as f:
                f.write(f"current total count:{curr_count}\n")
                for key, val in occ_level_sr.items():
                    f.write(f"{key}:{val}\n")
                f.write('\n')

        # Check if we've reached the max_scenes limit
        if max_scenes > 0 and num_id >= max_scenes:
            break

    # Save target name and label dictionary
    with open(f'{result_path}/targ_name_label.json', 'w') as f:
        json.dump(targ_name_label, f)
        
    # Calculate success rate only for non-zero count levels
    non_zero_levels = [level for level in occ_level_count_dict if occ_level_count_dict[level] > 0]
    
    # Save metrics to meta_evaluations.txt
    with open(f'{result_path}/meta_evaluations.txt', 'w') as f:
        f.write("Scene_ID, Target_Name, Occlusion_Level, IoU, CD, Success\n")
        avg_cd = 0.0
        avg_iou = 0.0
        total_scenes = len(scene_metrics)
        success_count = 0
        
        for scene_name, metrics in scene_metrics.items():
            target_name = metrics["target_name"]
            occ_level = metrics["occlusion_level"]
            iou = metrics["iou"]
            cd = metrics["cd"]
            success = metrics.get("success", 0)  # Default to 0 if "success" key doesn't exist
            
            f.write(f"{scene_name}, {target_name}, {occ_level:.4f}, {iou:.6f}, {cd:.6f}, {success}\n")
            avg_cd += cd
            avg_iou += iou
            
            # Count successful grasps
            if success == 1:
                success_count += 1
        
        if total_scenes > 0:
            avg_cd /= total_scenes
            avg_iou /= total_scenes
            success_rate = success_count / total_scenes * 100
            f.write(f"\nAverage CD: {avg_cd:.6f}\n")
            f.write(f"Average IoU: {avg_iou:.6f}\n")
            f.write(f"Success Rate: {success_rate:.2f}%\n")
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
        # self.logdir.mkdir(parents=True, exist_ok=True)
        # self.scenes_dir.mkdir(parents=True, exist_ok=True)

        self.mesh_dir = root / "visualize" / "meshes"
        # self.mesh_dir.mkdir(parents=True, exist_ok=True)

        # self.rounds_csv_path = self.logdir / "rounds.csv"
        # self.grasps_csv_path = self.logdir / "grasps.csv"
        # self._create_csv_files_if_needed(tgt_sample)

    def _create_csv_files_if_needed(self, tgt_sample):
        """Make rounds.csv and grasps.csv if they do not exist."""
        if not self.rounds_csv_path.exists():
            io.create_csv(self.rounds_csv_path, ["round_id", "object_count"])

        if not self.grasps_csv_path.exists():
            if not tgt_sample:
                columns = [
                    "round_id", "scene_id",
                    "qx", "qy", "qz", "qw",
                    "x", "y", "z",
                    "width", "score", "label",
                    "integration_time", "planning_time",
                ]
            else:
                columns = [
                    "round_id", "scene_id",
                    "qx", "qy", "qz", "qw",
                    "x", "y", "z",
                    "width", "score", "label",
                    "occ_level",
                    "integration_time", "planning_time",
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

        if occ_level is None:
            io.append_csv(
                self.grasps_csv_path,
                round_id,
                scene_id,
                qx, qy, qz, qw,
                x, y, z,
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
                qx, qy, qz, qw,
                x, y, z,
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
