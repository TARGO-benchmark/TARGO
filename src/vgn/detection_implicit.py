import time

import numpy as np
import trimesh
from scipy import ndimage
import torch
import os
import re
import argparse
import open3d as o3d  # Import Open3D at the file level

#from vgn import vis
from src.vgn.grasp import *
from src.vgn.utils.transform import Transform, Rotation
from src.vgn.networks import load_network
from src.vgn.utils import visual
from src.utils_giga import *
from src.vgn.grasp_conversion import anygrasp_to_vgn, fgc_to_vgn, anygrasp_to_vgn_with_region_filter, fgc_to_vgn_with_region_filter

from src.utils_giga import tsdf_to_ply, point_cloud_to_tsdf
from src.utils_targo import tsdf_to_mesh, filter_grasps_by_target

from src.shape_completion.config import cfg_from_yaml_file
from src.shape_completion import builder

# Remove global imports of FGCGraspNet and AnyGrasp dependencies
# These will be imported conditionally when needed
# from src.shape_completion.models.AdaPoinTr import AdaPoinTr
# from src.transformer.fusion_model import AdaPoinTr
# from src.shape_completion.models.AdaPoinTr import AdaPoinTr
import sys
# sys.path = [
#     '../src/FGCGraspNet',
# ] + sys.path

# Conditional imports - these will be imported inside functions when needed
# from src.shape_completion.models.AdaPoinTr import AdaPoinTr
# from src.FGCGraspNet.models.FGC_graspnet import FGC_graspnet
# from src.FGCGraspNet.dataset.graspnet_dataset import GraspNetDataset
# from src.FGCGraspNet.models.decode import pred_decode
# from graspnetAPI import GraspGroup
# sys.path.append('./src/anygrasp_sdk/grasp_detection')
# from gsnet import AnyGrasp
# from graspnetAPI import GraspGroup

LOW_TH = 0.0

def get_grasps(net, end_points):
    # Conditional imports for FGCGraspNet
    try:
        from src.FGCGraspNet.models.decode import pred_decode
        from graspnetAPI import GraspGroup
    except ImportError as e:
        print(f"Warning: Could not import FGCGraspNet dependencies: {e}")
        return None
    
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    # if save the results, return gg_array
    return gg

def collision_detection(gg, cloud):
    try:
        from src.FGCGraspNet.utils.collision_detector import ModelFreeCollisionDetector
    except ImportError as e:
        print(f"Warning: Could not import collision detector: {e}")
        return gg
    
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=0.01)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=0.01)
    gg = gg[~collision_mask]
    return gg

def vis_grasps_target(target_gg, target_cloud, scene_cloud, anygrasp=False, fgc=False, output_prefix=None):
    # Process grasp group
    target_gg.nms()
    target_gg.sort_by_score()
    target_gg = target_gg[:20]
    grippers = target_gg.to_open3d_geometry_list()
    
    # Create colored point clouds
    # Using o3d (already imported at the top of the file)
    
    # Set target cloud to red color
    target_cloud_colored = o3d.geometry.PointCloud()
    target_cloud_colored.points = target_cloud.points
    target_cloud_colored.colors = o3d.utility.Vector3dVector(np.ones((len(target_cloud.points), 3)) * np.array([1, 0, 0]))  # Red color
    
    # Set scene cloud to green color
    scene_cloud_colored = o3d.geometry.PointCloud()
    scene_cloud_colored.points = scene_cloud.points
    scene_cloud_colored.colors = o3d.utility.Vector3dVector(np.ones((len(scene_cloud.points), 3)) * np.array([0, 1, 0]))  # Green color
    
    # Print statistics for debugging
    target_points = np.asarray(target_cloud.points)
    scene_points = np.asarray(scene_cloud.points)
    print(f"Scene points: {len(scene_points)}, Target points: {len(target_points)}")
    
    # Determine output file names based on grasp type and prefix
    if output_prefix is None:
        # If no output prefix provided, use default
        prefix = 'demo/cloud'
    else:
        # Use the provided output prefix
        prefix = output_prefix
    
    # Add suffixes based on grasp type
    if anygrasp:
        prefix += '_anygrasp'
    elif fgc:
        prefix += '_fgc'
    
    # Save the colored point clouds to PLY files
    o3d.io.write_point_cloud(f'{prefix}_target.ply', target_cloud_colored)
    o3d.io.write_point_cloud(f'{prefix}_scene.ply', scene_cloud_colored)
    print(f'Target point cloud saved to {prefix}_target.ply')
    print(f'Scene point cloud saved to {prefix}_scene.ply')
    
    # Convert point clouds to trimesh
    import trimesh
    target_points = np.asarray(target_cloud_colored.points)
    target_colors = np.asarray(target_cloud_colored.colors)
    target_mesh = trimesh.points.PointCloud(target_points, colors=target_colors)
    
    scene_points = np.asarray(scene_cloud_colored.points)
    scene_colors = np.asarray(scene_cloud_colored.colors)
    scene_mesh = trimesh.points.PointCloud(scene_points, colors=scene_colors)
    
    # Create a scene and add both point clouds
    scene = trimesh.Scene()
    scene.add_geometry(target_mesh)
    scene.add_geometry(scene_mesh)
    
    # Add all grippers to the scene
    for gripper in grippers:
        # Convert Open3D mesh to trimesh
        vertices = np.asarray(gripper.vertices)
        faces = np.asarray(gripper.triangles)
        gripper_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        scene.add_geometry(gripper_mesh)
    
    # Export the scene
    output_file = f'{prefix}_with_target_grasps.glb'
    scene.export(output_file)
    print(f'Scene saved to {output_file}')

def vis_grasps(gg, cloud, anygrasp=False, fgc=False):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:50]
    grippers = gg.to_open3d_geometry_list()
    # o3d.visualization.draw_geometries([cloud, *grippers])
    # save as glb file, cloud and grippers
    # Convert Open3D geometries to trimesh
    # Using trimesh (already imported at the top of the file)
    
    # Save the point cloud to a PLY file
    if anygrasp:
        o3d.io.write_point_cloud('demo/cloud_anygrasp.ply', cloud)
        print('Point cloud saved to demo/cloud_anygrasp.ply')
    elif fgc:
        o3d.io.write_point_cloud('demo/cloud_fgc.ply', cloud)
        print('Point cloud saved to demo/cloud_fgc.ply')
    else:
        o3d.io.write_point_cloud('demo/cloud.ply', cloud)
        print('Point cloud saved to demo/cloud.ply')
    
    # Convert point cloud to trimesh
    cloud_points = np.asarray(cloud.points)
    cloud_colors = np.asarray(cloud.colors)
    cloud_mesh = trimesh.points.PointCloud(cloud_points, colors=cloud_colors)
    
    # Create a scene and add the point cloud
    scene = trimesh.Scene()
    scene.add_geometry(cloud_mesh)
    
    # Add all grippers to the scene
    for gripper in grippers:
        # Convert Open3D mesh to trimesh
        vertices = np.asarray(gripper.vertices)
        faces = np.asarray(gripper.triangles)
        gripper_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        scene.add_geometry(gripper_mesh)
    
    # Export the scene
    if anygrasp:
        scene.export('demo/cloud_anygrasp_with_grasps.glb')
        print('saved to demo/cloud_anygrasp_with_grasps.glb')
    elif fgc:
        scene.export('demo/cloud_fgc_with_grasps.glb')
        print('saved to demo/cloud_fgc_with_grasps.glb')
    else:
        scene.export('demo/cloud_with_grasps.glb')
        print('saved to demo/cloud_with_grasps.glb')

class VGNImplicit(object):
    def __init__(self, model_path, model_type, best=False, force_detection=False, qual_th=0.9, out_th=0.5, visualize=False, resolution=40,cd_iou_measure=False,**kwargs,):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_type != 'FGC-GraspNet' and model_type != 'AnyGrasp' and model_type != 'AnyGrasp_full_targ' and model_type != 'FGC_full_targ':
            self.net = load_network(model_path, self.device, model_type=model_type) 
            self.net = self.net.eval()
        elif model_type == 'FGC-GraspNet' or model_type == 'FGC_full_targ':
            # Conditional import for FGCGraspNet
            try:
                from src.FGCGraspNet.models.FGC_graspnet import FGC_graspnet
            except ImportError as e:
                raise ImportError(f"FGC-GraspNet dependencies not available: {e}")
            
            self.net = FGC_graspnet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=0.055, hmax=0.30, is_training=False, is_demo=True)
            self.net.to(self.device)
            checkpoint = torch.load('./src/FGCGraspNet/checkpoints/realsense_checkpoint.tar')
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.net.eval()
        elif model_type == 'AnyGrasp' or model_type == 'AnyGrasp_full_targ':
            # Conditional import for AnyGrasp
            try:
                import sys
                sys.path.append('./src/anygrasp_sdk/grasp_detection')
                from gsnet import AnyGrasp
            except ImportError as e:
                raise ImportError(f"AnyGrasp dependencies not available: {e}")
            
            # Initialize AnyGrasp with configuration parameters
            parser = argparse.ArgumentParser()
            parser.add_argument('--checkpoint_path', default='./src/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar', help='Model checkpoint path')
            parser.add_argument('--max_gripper_width', type=float, default=0.08, help='Maximum gripper width (<=0.1m)')
            parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
            parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
            parser.add_argument('--debug', action='store_true', help='Enable debug mode')
            cfgs = parser.parse_args([])  # Empty list to avoid reading command line args
            cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))
            
            self.net = AnyGrasp(cfgs)
            self.net.load_net()
            # self.net.eval()
        if model_type != 'AnyGrasp' and model_type != 'AnyGrasp_full_targ':
            net_params_count = sum(p.numel() for p in self.net.parameters())
            print(f"Number of parameters in self.net: {net_params_count}")
        
        # Conditional import for AdaPoinTr
        try:
            from src.shape_completion.models.AdaPoinTr import AdaPoinTr
        except ImportError as e:
            print(f"Warning: Could not import AdaPoinTr: {e}")
            sc_net = None
        else:
            sc_cfg = cfg_from_yaml_file("src/shape_completion/configs/AdaPoinTr.yaml")
            sc_net = AdaPoinTr(sc_cfg.model)
            builder.load_model(sc_net, "checkpoints/adapointr.pth")
            sc_net = sc_net.eval()
            sc_net_params_count = sum(p.numel() for p in sc_net.parameters())
            print(f"Number of parameters in self.sc_net: {sc_net_params_count}")
        
        self.sc_net = sc_net
        self.qual_th = qual_th
        self.best = best
        self.force_detection = force_detection
        self.out_th = out_th
        self.visualize = visualize
        self.resolution = resolution
        self.cd_iou_measure = cd_iou_measure
        if model_type == 'giga_hr':
            self.resolution = 60
        x, y, z = torch.meshgrid(torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution), torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution), torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution))
        pos = torch.stack((x, y, z), dim=-1).float().unsqueeze(0).to(self.device)   
        self.pos = pos.view(1, self.resolution * self.resolution * self.resolution, 3)  ## pos: 1, 64000, -0.5, 0.475

        plane = np.load("./setup/plane_sampled.npy")
        plane = plane.astype(np.float32)
        self.plane = plane / 0.3 - 0.5
        # self.plane = torch.from_numpy(plane).to(self.device)

    def __call__(self, state, scene_mesh=None, visual_dict = None, hunyun2_path = None, scene_name = None, cd_iou_measure=False, target_mesh_gt = None,aff_kwargs={}):
        ## all the keys in the namespace of state
        visual_dict = {}
        scene_name = scene_name
        print(state.__dict__.keys())

        if state.type in ('giga_aff', 'giga', 'giga_hr'):
            if hasattr(state, 'tsdf_process'):
                tsdf_process = state.tsdf_process
            else:
                tsdf_process = state.tgt_mask_vol  

            if not hunyun2_path:
                inputs = state.scene_grid
            else:
                inputs = state.scene_grid - state.targ_grid
            voxel_size, size = state.tsdf.voxel_size, state.tsdf.size 
            if not hunyun2_path:
                if state.type == 'giga':
                    qual_vol, rot_vol, width_vol, cd, iou = predict(inputs, self.pos, self.net,  None, state.type, self.device, target_mesh_gt=target_mesh_gt)
                else:
                    qual_vol, rot_vol, width_vol = predict(inputs, self.pos, self.net,  None, state.type, self.device, target_mesh_gt=target_mesh_gt)
            elif hunyun2_path:
                qual_vol, rot_vol, width_vol, completed_targ_grid = predict(inputs, self.pos, self.net,  None, state.type, self.device, visual_dict, hunyun2_path, scene_name, target_mesh_gt=target_mesh_gt)
            

        elif state.type == 'targo' or state.type == 'targo_full_targ' or state.type == 'targo_hunyun2' or state.type == 'targo_ptv3' or state.type == 'ptv3_scene':
            scene_no_targ_pc = state.scene_no_targ_pc
            scene_no_targ_pc = np.concatenate((scene_no_targ_pc, self.plane), axis=0)
            
            # Handle different input formats for PointTransformerV3 models
            if state.type == 'targo_ptv3':
                # For targo_ptv3: use both scene and target point clouds
                targ_pc = state.targ_pc
                inputs = (scene_no_targ_pc, targ_pc)    # scene_no_targ_pc is tsdf surface points, target pc is the depth backprojected points
            elif state.type == 'ptv3_scene':
                # For ptv3_scene: only use scene point cloud (target is included in scene)
                # state.scene_no_targ_pc should actually be the full scene including target for ptv3_scene
                if hasattr(state, 'targ_pc') and state.targ_pc is not None:
                    # If target PC is available, concatenate it with scene for full scene representation
                    full_scene_pc = np.concatenate((scene_no_targ_pc, state.targ_pc), axis=0)
                    
                    # For ptv3_scene, we need to create a combined point cloud with labels
                    # Create labels: 0 for scene/occluder points, 1 for target points
                    scene_labels = np.zeros((len(scene_no_targ_pc), 1), dtype=np.float32)
                    target_labels = np.ones((len(state.targ_pc), 1), dtype=np.float32)
                    
                    # Combine points and labels
                    scene_with_labels = np.concatenate([scene_no_targ_pc, scene_labels], axis=1)
                    target_with_labels = np.concatenate([state.targ_pc, target_labels], axis=1)
                    full_scene_with_labels = np.concatenate([scene_with_labels, target_with_labels], axis=0)
                    
                    inputs = (full_scene_with_labels, None)  # Only scene input with labels, no separate target
                else:
                    # Use scene_no_targ_pc as is (it should already include target for ptv3_scene)
                    # Assume scene_no_targ_pc already has labels in the 4th column for ptv3_scene
                    full_scene_pc = scene_no_targ_pc
                    inputs = (full_scene_pc, None)  # Only scene input, no separate target
            else:
                # Original targo models
                targ_pc = state.targ_pc
                inputs = (scene_no_targ_pc, targ_pc)    # scene_no_targ_pc is tsdf surface points, target pc is the depth backprojected points

            # Handle both tensor and numpy array inputs
            scene_pc = inputs[0]
            targ_pc = inputs[1] if inputs[1] is not None else np.array([])  # Handle None case for ptv3_scene
            save_point_cloud_as_ply(scene_pc, 'scene_no_targ_pc.ply')
            if targ_pc.size > 0:  # Only save if not empty
                save_point_cloud_as_ply(targ_pc, 'targ_pc.ply')
                visual_dict['targ_pc'] = state.targ_pc
            voxel_size, size = state.tsdf.voxel_size, state.tsdf.size
            with torch.no_grad():
                qual_vol, rot_vol, width_vol, completed_targ_grid, cd, iou = predict(inputs, self.pos, self.net, self.sc_net, state.type, self.device, visual_dict, hunyun2_path, scene_name, cd_iou_measure=True, target_mesh_gt=target_mesh_gt)

        elif state.type == 'FGC-GraspNet' or state.type == 'FGC_full_targ':
            # inputs = state.scene_pc
            if state.type == 'FGC_full_targ':
                inputs = np.concatenate((state.targ_full_pc, state.scene_no_targ_pc), axis=0)
                # plane_hs = np.load('./data//plane_hs.npy')
                # inputs = np.concatenate((inputs, plane_hs), axis=0)
                save_point_cloud_as_ply(inputs, 'scene_pc.ply')
            elif state.type == 'FGC-GraspNet':
                inputs = state.scene_pc
                save_point_cloud_as_ply(inputs, 'scene_pc.ply')
            with torch.no_grad():
                gg = predict(inputs, self.pos, self.net, self.sc_net, state.type, self.device, visual_dict, hunyun2_path, scene_name, cd_iou_measure=True, target_mesh_gt=target_mesh_gt)
            
            ## write target-driven filter here, 
            scene_pc = state.scene_pc
            if state.type == 'FGC_full_targ':
                target_pc = state.targ_full_pc
            elif state.type == 'FGC-GraspNet':
                target_pc = state.target_pc
            print(scene_pc.shape, target_pc.shape)
            print("target_pc")
            print(target_pc)
            print("target_pc.shape")
            print(target_pc.shape)
            
            # Create target point cloud for visualization
            target_pc_np = np.array(target_pc, dtype=np.float32)
            target_pc_o3d = o3d.geometry.PointCloud()
            target_pc_o3d.points = o3d.utility.Vector3dVector(target_pc_np)
            target_pc_o3d.colors = o3d.utility.Vector3dVector(np.ones_like(target_pc_np))
            
            # Filter grasps by target
            target_gg = filter_grasps_by_target(gg, target_pc_np)
            scene_pc_np = np.array(scene_pc, dtype=np.float32)
            scene_pc_o3d = o3d.geometry.PointCloud()
            scene_pc_o3d.points = o3d.utility.Vector3dVector(scene_pc_np)
            scene_pc_o3d.colors = o3d.utility.Vector3dVector(np.ones_like(scene_pc_np))

            vis_grasps_target(target_gg, target_pc_o3d, scene_pc_o3d, anygrasp=False, fgc=True)
            
            # Sort grasps by score and perform NMS before conversion
            # Non-maximum suppression is already applied in the vis_grasps function
            # but we explicitly apply it here to ensure we keep only the best grasps
            if len(target_gg) > 0:
                # Apply non-maximum suppression to remove redundant grasps
                target_gg.nms()
                # Sort grasps by quality score in descending order
                target_gg.sort_by_score()
                print(f"Sorted {len(target_gg)} target grasps by score")
                if len(target_gg) > 0:
                    print(f"Top grasp score: {target_gg[0].score:.3f}, lowest grasp score: {target_gg[-1].score:.3f}")
            
            # Visualize filtered grasps
            # vis_grasps(target_gg, target_pc_o3d, fgc=True)
            # vis_grasps_target(target_gg, target_pc_o3d, scene_pc_o3d, fgc=True) 
            g1b_vis_dict = {}
            g1b_vis_dict['target_pc'] = target_pc_o3d
            g1b_vis_dict['scene_pc'] = scene_pc_o3d
            g1b_vis_dict['target_gg'] = target_gg
            # print("visualized target grasps")
            # Convert scene_pc and target_pc to numpy arrays
            
            # Create open3d point cloud objects
            
        
        elif state.type == 'AnyGrasp' or state.type == 'AnyGrasp_full_targ':
            if state.type == 'AnyGrasp_full_targ':
                inputs = np.concatenate((state.targ_full_pc, state.scene_no_targ_pc), axis=0)
                # plane_hs = np.load('./data//plane_hs.npy')
                # inputs = np.concatenate((inputs, plane_hs), axis=0)
                save_point_cloud_as_ply(inputs, 'scene_pc.ply')
            else:
                inputs = state.scene_pc
                save_point_cloud_as_ply(inputs, 'scene_pc.ply')
            with torch.no_grad():
                gg = predict(inputs, self.pos, self.net, self.sc_net, state.type, self.device, visual_dict, hunyun2_path, scene_name, cd_iou_measure=True, target_mesh_gt=target_mesh_gt)
                
            # Extract target point cloud and filter grasps by target
            scene_pc = state.scene_pc
            if state.type == 'AnyGrasp':
                target_pc = state.targ_pc
            elif state.type == 'AnyGrasp_full_targ':
                target_pc = state.targ_full_pc
            
            # Create target point cloud for visualization
            target_pc_np = np.array(target_pc, dtype=np.float32)
            target_pc_o3d = o3d.geometry.PointCloud()
            target_pc_o3d.points = o3d.utility.Vector3dVector(target_pc_np)
            target_pc_o3d.colors = o3d.utility.Vector3dVector(np.ones_like(target_pc_np))
            
            # Filter grasps by target
            target_gg = filter_grasps_by_target(gg, target_pc_np)
            
            scene_pc_np = np.array(scene_pc, dtype=np.float32)
            scene_pc_o3d = o3d.geometry.PointCloud()
            scene_pc_o3d.points = o3d.utility.Vector3dVector(scene_pc_np)
            scene_pc_o3d.colors = o3d.utility.Vector3dVector(np.ones_like(scene_pc_np))

            vis_grasps_target(target_gg, target_pc_o3d, scene_pc_o3d, anygrasp=True, fgc=False)

            # o3d.io.write_point_cloud(target_pc_o3d.points, 'target_pc.ply')
            o3d.io.write_point_cloud("target_pc.ply", target_pc_o3d)
            
            # Sort grasps by score and perform NMS before conversion
            # This ensures we keep only the best grasps
            if len(target_gg) > 0:
                # Apply non-maximum suppression to remove redundant grasps
                target_gg.nms()
                # Sort grasps by quality score in descending order
                target_gg.sort_by_score()
                print(f"Sorted {len(target_gg)} target grasps by score")
                if len(target_gg) > 0:
                    print(f"Top grasp score: {target_gg[0].score:.3f}, lowest grasp score: {target_gg[-1].score:.3f}")
            
            # Optional: Visualize filtered grasps
            # vis_grasps(target_gg, target_pc_o3d, anygrasp=True)
            # vis_grasps_target(target_gg, target_pc_o3d, scene_pc_o3d, anygrasp=True)
            
            # Create visualization dictionary for AnyGrasp
            g1b_vis_dict = {}
            g1b_vis_dict['target_pc'] = target_pc_o3d
            g1b_vis_dict['scene_pc'] = scene_pc_o3d
            g1b_vis_dict['target_gg'] = target_gg
        begin = time.time()

        if state.type != 'FGC-GraspNet' and state.type != 'AnyGrasp' and state.type != 'AnyGrasp_full_targ' and state.type != 'FGC_full_targ':
            qual_vol = qual_vol.reshape((self.resolution, self.resolution, self.resolution))
            rot_vol = rot_vol.reshape((self.resolution, self.resolution, self.resolution, 4))
            width_vol = width_vol.reshape((self.resolution, self.resolution, self.resolution))

        if state.type == 'FGC-GraspNet' or state.type == 'AnyGrasp' or state.type == 'AnyGrasp_full_targ' or state.type == 'FGC_full_targ':
            # Check if there are valid grasps
            if len(target_gg) == 0 or np.max(target_gg.scores) < 0.0:
                print(f"Warning: {state.type} did not find valid target grasps")
                # Return empty lists instead of raising an error
                return [], [], 0, g1b_vis_dict, 0, 0
            
            # Create camera extrinsic matrix (from camera to world coordinate system)
            # This represents the transformation from camera coordinate system to world coordinate system
            # Note: For simulation environments, this may need adjustment based on your setup
            
            # Get camera extrinsic matrix from state if available
            if hasattr(state, 'extrinsic'):
                extrinsic = state.extrinsic
                print(f"Using camera extrinsic from state: {extrinsic}")
            else:
                # Create a default extrinsic matrix (identity rotation, no translation)
                # This assumes camera is at world origin with no rotation
                # Adjust as needed for your actual camera setup
                print("No camera extrinsics found in state, using default")
                extrinsic = Transform(
                    Rotation.from_matrix(np.eye(3)),
                    np.zeros(3)
                )
            
            # Save visualization files if scene info available
            # This will save point cloud files to the same directory as grasping videos
            if hasattr(state, 'scene_id') and hasattr(state, 'target_name'):
                # Try to get video saving directory from state
                video_dir = None
                if hasattr(state, 'video_dir') and state.video_dir:
                    video_dir = state.video_dir
                elif hasattr(state, 'result_path') and state.result_path:
                    # Construct path similar to what's used for video recording
                    video_dir = os.path.join(state.result_path, 'videos')
                
                if video_dir:
                    # Make sure directory exists
                    if not os.path.exists(video_dir):
                        os.makedirs(video_dir)
                    
                    # Create filename based on scene ID and target name
                    scene_id = state.scene_id
                    target_name = state.target_name
                    grasp_type = "anygrasp" if state.type == "AnyGrasp" else "fgc"
                    
                    # Prepare point clouds for visualization
                    # (using o3d already imported at the top of the file)
                    
                    # Create target point cloud (red)
                    target_pc_np = np.array(state.target_pc, dtype=np.float32)
                    target_pc_o3d = o3d.geometry.PointCloud()
                    target_pc_o3d.points = o3d.utility.Vector3dVector(target_pc_np)
                    target_pc_o3d.colors = o3d.utility.Vector3dVector(np.ones_like(target_pc_np) * np.array([1, 0, 0]))  # Red color
                    
                    # Create scene point cloud (blue)
                    scene_pc_np = np.array(state.scene_pc, dtype=np.float32)
                    scene_pc_o3d = o3d.geometry.PointCloud()
                    scene_pc_o3d.points = o3d.utility.Vector3dVector(scene_pc_np)
                    scene_pc_o3d.colors = o3d.utility.Vector3dVector(np.ones_like(scene_pc_np) * np.array([0, 0, 1]))  # Blue color
                    
                    # Save separate point cloud files
                    base_name = f"{scene_id}_{target_name}_{grasp_type}"
                    target_cloud_path = os.path.join(video_dir, f"{base_name}_target_pc.ply")
                    scene_cloud_path = os.path.join(video_dir, f"{base_name}_scene_pc.ply")
                    
                    # Save point clouds to files
                    o3d.io.write_point_cloud(target_cloud_path, target_pc_o3d)
                    o3d.io.write_point_cloud(scene_cloud_path, scene_pc_o3d)
                    
                    # Also save a visualization with grasps (using trimesh)
                    # (using trimesh already imported at the top of the file)
                    
                    # Create a visualization with top 5 grasps
                    sorted_gg = target_gg.copy()
                    sorted_gg.nms()
                    sorted_gg.sort_by_score()
                    top_grasps = sorted_gg[:5]  # Take top 5 grasps for visualization
                    grippers = top_grasps.to_open3d_geometry_list()
                    
                    # Create point cloud meshes
                    target_mesh = trimesh.points.PointCloud(target_pc_np, colors=np.ones_like(target_pc_np) * np.array([1, 0, 0, 1]))
                    scene_mesh = trimesh.points.PointCloud(scene_pc_np, colors=np.ones_like(scene_pc_np) * np.array([0, 0, 1, 1]))
                    
                    # Create a scene and add point clouds
                    vis_scene = trimesh.Scene()
                    vis_scene.add_geometry(target_mesh)
                    vis_scene.add_geometry(scene_mesh)
                    
                    # Add grippers to the scene
                    for i, gripper in enumerate(grippers):
                        # Convert Open3D mesh to trimesh
                        vertices = np.asarray(gripper.vertices)
                        faces = np.asarray(gripper.triangles)
                        gripper_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                        
                        # Color the gripper based on its rank (best is green, others yellow)
                        if i == 0:
                            # Best grasp is green
                            color = [0, 1, 0, 1]  # RGBA
                        else:
                            # Other grasps are yellow
                            color = [1, 1, 0, 1]  # RGBA
                            
                        # Apply the color to all vertices of the gripper
                        gripper_mesh.visual.vertex_colors = np.tile(np.array(color, dtype=np.float32) * 255, (len(vertices), 1))
                        
                        # Add to the scene
                        vis_scene.add_geometry(gripper_mesh)
                    
                    # Save the scene as GLB file
                    grasp_vis_path = os.path.join(video_dir, f"{base_name}_grasps.glb")
                    vis_scene.export(grasp_vis_path)
                    
                    print(f"Saved visualization files to {video_dir}:")
                    print(f"  - Target point cloud: {os.path.basename(target_cloud_path)}")
                    print(f"  - Scene point cloud: {os.path.basename(scene_cloud_path)}")
                    print(f"  - Grasp visualization: {os.path.basename(grasp_vis_path)}")
            
            # Convert grasps from target_gg to VGN format
            if state.type == 'FGC-GraspNet' or state.type == 'FGC_full_targ':
                print(f"Converting {len(target_gg)} FGC-GraspNet grasps to VGN format")
                # Use workspace_size if available in state
                workspace_size = state.tsdf.size if hasattr(state, 'tsdf') and hasattr(state.tsdf, 'size') else None
                
                # Get region bounds from state if available
                region_lower = state.lower if hasattr(state, 'lower') else np.array([0.02, 0.02, 0.055])
                region_upper = state.upper if hasattr(state, 'upper') else np.array([0.28, 0.28, 0.30])
                
                # Use the region-filtered version of the function
                grasps, scores = fgc_to_vgn_with_region_filter(
                    target_gg, 
                    extrinsic, 
                    workspace_size,
                    region_lower=region_lower,
                    region_upper=region_upper
                )
            else:  # AnyGrasp
                print(f"Converting {len(target_gg)} AnyGrasp grasps to VGN format")
                workspace_size = state.tsdf.size if hasattr(state, 'tsdf') and hasattr(state.tsdf, 'size') else None
                
                # Get region bounds from state if available
                region_lower = state.lower if hasattr(state, 'lower') else np.array([0.02, 0.02, 0.055])
                region_upper = state.upper if hasattr(state, 'upper') else np.array([0.28, 0.28, 0.30])
                
                # Use the region-filtered version of the function
                grasps, scores = anygrasp_to_vgn(
                    target_gg, 
                    workspace_size,
                    grasps_are_in_world=True
                )
            
            # If no valid grasps after conversion, return empty lists
            if len(grasps) == 0:
                print(f"Warning: No valid grasps after conversion")
                return [], [], 0, g1b_vis_dict, 0, 0
            
            # Sort grasps by scores in descending order
            if len(scores) > 1:
                # Get the indices that would sort scores in descending order
                sorted_indices = np.argsort(scores)[::-1]
                
                # Reorder both grasps and scores according to these indices
                grasps = [grasps[i] for i in sorted_indices]
                scores = [scores[i] for i in sorted_indices]
                
                print(f"Sorted {len(grasps)} grasps by score in descending order")
                print(f"Top grasp score: {scores[0]:.3f}, lowest grasp score: {scores[-1]:.3f}")
            
            # 简化版直接存到demo目录，不需要那么正式的路径结构
            # 创建demo目录
            if not os.path.exists('demo'):
                os.makedirs('demo')
            
            # 直接使用模型类型作为文件名前缀
            # model_type = "fgc" if state.type == "FGC-GraspNet" else "anygrasp"
            
            # 使用VGN可视化函数
            from src.vgn.utils.visual import vis_grasps_target_vgn
            vis_grasps_target_vgn(
                grasps=grasps,  # 可视化所有抓取
                scores=scores,
                target_cloud=g1b_vis_dict['target_pc'],
                scene_cloud=g1b_vis_dict['scene_pc'],
                output_prefix=f"demo/{state.type}"
            )
            
            # Print information about the converted grasps
            print(f"Converted {len(grasps)} grasps with scores ranging from {min(scores):.3f} to {max(scores):.3f}")
            
            return grasps, scores, 0, g1b_vis_dict, 0, 0
        
        if state.type == 'targo' or state.type == 'targo_full_targ' or state.type == 'targo_hunyun2' or state.type == 'targo_ptv3' or state.type == 'ptv3_scene':
            # Handle both numpy array and tensor inputs

            if isinstance(completed_targ_grid, np.ndarray):
                # For numpy array: reshape from (40,40,40) to (1,40,40,40)
                completed_targ_grid = np.expand_dims(completed_targ_grid, axis=0)
            else:
                # For torch tensor: use squeeze and unsqueeze
                completed_targ_grid = completed_targ_grid.squeeze().unsqueeze(0)  # from (40,40,40) to (1,40,40,40)
            qual_vol, rot_vol, width_vol = process(completed_targ_grid, qual_vol, rot_vol, width_vol, out_th=self.out_th)
            # qual_vol, rot_vol, width_vol = process(state.targ_grid, qual_vol, rot_vol, width_vol, out_th=self.out_th)
        
        elif state.type in ('giga_aff', 'giga', 'giga_hr'):
            # qual_vol, rot_vol, width_vol = process(state.tsdf.get_grid(), qual_vol, rot_vol, width_vol, out_th=self.out_th)
            # if hunyun2_path:
            #     qual_vol, rot_vol, width_vol = process(completed_targ_grid.squeeze().cpu().numpy(), qual_vol, rot_vol, width_vol, out_th=self.out_th)
            # else:
            qual_vol, rot_vol, width_vol = process(state.targ_grid, qual_vol, rot_vol, width_vol, out_th=self.out_th)
        if len(qual_vol.shape) == 1:
            qual_vol = qual_vol.reshape((self.resolution, self.resolution, self.resolution))
        qual_vol = bound(qual_vol, voxel_size)

        if self.visualize:
            colored_scene_mesh = visual.affordance_visual(qual_vol, rot_vol, scene_mesh, size, self.resolution, **aff_kwargs)
            visual_dict['affordance_visual'] = colored_scene_mesh

        grasps, scores = select(qual_vol.copy(), self.pos.view(self.resolution, self.resolution, self.resolution, 3).cpu(), rot_vol, width_vol, threshold=self.qual_th, force_detection=self.force_detection, max_filter_size=8 if self.visualize else 4)

        toc = time.time()
        grasps, scores = np.asarray(grasps), np.asarray(scores)

        new_grasps = []
        if len(grasps) > 0:
            if self.best:
                p = np.arange(len(grasps))
            else:
                p = np.random.permutation(len(grasps))
            for g in grasps[p]:
                pose = g.pose
                pose.translation = (pose.translation + 0.5) * size
                width = g.width * size
                new_grasps.append(Grasp(pose, width))
            scores = scores[p]
        grasps = new_grasps

        end=time.time()
        print(f"post processing: {end-begin:.3f}s")
        # if self.visualize:
        if visual_dict:
            grasp_mesh_list = [visual.grasp2mesh(g, s) for g, s in zip(grasps, scores)]
            composed_scene = trimesh.Scene(colored_scene_mesh)
            for i, g_mesh in enumerate(grasp_mesh_list):
                composed_scene.add_geometry(g_mesh, node_name=f'grasp_{i}')
            visual_dict['composed_scene'] = composed_scene
            return grasps, scores, toc, composed_scene
        # else:
        # return grasps, scores, toc, visual_dict
        # if cd_iou_measure:
        #     # if not completed_targ_grid:
        #     completed_targ_grid = state.targ_grid
        #     cd = 0
        #     iou = 0
        if state.type == 'targo' or state.type == 'targo_full_targ' or state.type == 'targo_hunyun2' or state.type == 'targo_ptv3' or state.type == 'ptv3_scene':
            return grasps, scores, toc, cd, iou
        elif state.type == 'giga':
            return grasps, scores, toc, cd, iou
        elif state.type == 'vgn':
            return qual_vol, rot_vol, width_vol
        elif state.type == 'AnyGrasp' or state.type == 'FGC-GraspNet':
            # return grasp, score
            return gg

def bound(qual_vol, voxel_size, limit=[0.02, 0.02, 0.055]):
    # avoid grasp out of bound [0.02  0.02  0.055]
    x_lim = int(limit[0] / voxel_size)
    y_lim = int(limit[1] / voxel_size)
    z_lim = int(limit[2] / voxel_size)
    qual_vol[:x_lim] = 0.0
    qual_vol[-x_lim:] = 0.0
    qual_vol[:, :y_lim] = 0.0
    qual_vol[:, -y_lim:] = 0.0
    qual_vol[:, :, :z_lim] = 0.0
    return qual_vol

def predict(inputs, pos, net, sc_net, type, device, visual_dict=None, hunyun2_path = None, scene_name = None, cd_iou_measure = False, target_mesh_gt = None):
    sc_time = 0
    hunyun2_path = hunyun2_path
    scene_name = scene_name

    ## NEW TARGET_MESH_GT
    target_mesh_watertight_path = os.path.join(hunyun2_path, scene_name, 'reconstruction', 'gt_targ_obj.ply')
    if os.path.exists(target_mesh_watertight_path):
        target_mesh_watertight = trimesh.load(target_mesh_watertight_path)
        target_mesh_gt = target_mesh_watertight
    else:
        target_mesh_gt = target_mesh_gt
    # target_mesh_gt.export('target_mesh_gt.obj')
    # target_mesh_gt.export('target_mesh_gt.ply')
    # # target_mesh_watertight.export('target_mesh_watertight.obj')
    # target_mesh_watertight.export('target_mesh_watertight.ply')


    if type in ('giga', 'giga_aff', 'vgn', 'giga_hr'):
        assert sc_net == None
        inputs = torch.from_numpy(inputs).to(device)
        if hunyun2_path:
            # completed_targ_pc = completed_targ_pc.astype(np.float32)    
            # completed_targ_grid = point_cloud_to_tsdf(completed_targ_pc)
            scene_path = hunyun2_path + '/' + scene_name + '/reconstruction/targ_obj_v7.ply'
            # scene_path = hunyun2_path + '/' + scene_name + '/reconstruction/gt_targ_obj.ply'
            targ_mesh = trimesh.load(scene_path)
            targ_grid = mesh_to_tsdf(targ_mesh)
            targ_grid = torch.from_numpy(targ_grid).to(device)
            targ_grid = targ_grid.unsqueeze(0)
            inputs += targ_grid
            completed_targ_grid = targ_grid


    elif type == 'targo' or type == 'targo_full_targ' or type == 'targo_hunyun2' or type == 'targo_ptv3' or type == 'ptv3_scene':
        scene_no_targ_pc = torch.from_numpy(inputs[0]).unsqueeze(0).to(device)
        
        # Handle different input formats for PointTransformerV3 models
        if type == 'targo_ptv3':
            # For targo_ptv3: use both scene and target point clouds
            targ_pc = torch.from_numpy(inputs[1]).unsqueeze(0).to(device)
        elif type == 'ptv3_scene':
            # For ptv3_scene: only use scene point cloud (target is included in scene)
            # state.scene_no_targ_pc should actually be the full scene including target for ptv3_scene
            if hasattr(state, 'targ_pc') and state.targ_pc is not None:
                # If target PC is available, concatenate it with scene for full scene representation
                full_scene_pc = np.concatenate((scene_no_targ_pc, state.targ_pc), axis=0)
                
                # For ptv3_scene, we need to create a combined point cloud with labels
                # Create labels: 0 for scene/occluder points, 1 for target points
                scene_labels = np.zeros((len(scene_no_targ_pc), 1), dtype=np.float32)
                target_labels = np.ones((len(state.targ_pc), 1), dtype=np.float32)
                
                # Combine points and labels
                scene_with_labels = np.concatenate([scene_no_targ_pc, scene_labels], axis=1)
                target_with_labels = np.concatenate([state.targ_pc, target_labels], axis=1)
                full_scene_with_labels = np.concatenate([scene_with_labels, target_with_labels], axis=0)
                
                inputs = (full_scene_with_labels, None)  # Only scene input with labels, no separate target
            else:
                # Use scene_no_targ_pc as is (it should already include target for ptv3_scene)
                # Assume scene_no_targ_pc already has labels in the 4th column for ptv3_scene
                full_scene_pc = scene_no_targ_pc
                inputs = (full_scene_pc, None)  # Only scene input, no separate target
        else:
            # Original targo models
            targ_pc = torch.from_numpy(inputs[1]).unsqueeze(0).to(device)

        with torch.no_grad():
            if type == 'targo':
                sc_net = sc_net.to(device)
                sc_net_params_count = sum(p.numel() for p in sc_net.parameters())
                print(f"Number of parameters in sc_net: {sc_net_params_count}")
                sc_time_start = time.time()
                completed_targ_pc = sc_net(targ_pc)[1]
                visual_dict['completed_targ_pc'] = completed_targ_pc[0].cpu().numpy()
                sc_time = time.time() - sc_time_start
                print(f"Shape completion time: {sc_time:.3f}s")
                start_pc = time.time()
                completed_targ_pc_real_size = (completed_targ_pc+0.5)*0.3
                targ_mesh, completed_targ_grid = point_cloud_to_tsdf(completed_targ_pc_real_size.squeeze().cpu().numpy(), return_mesh=True)
                targ_mesh_vertices = np.asarray(targ_mesh.vertices)
                targ_mesh_triangles = np.asarray(targ_mesh.triangles)
                targ_trimesh = trimesh.Trimesh(vertices=targ_mesh_vertices, faces=targ_mesh_triangles)
                cd, iou = compute_chamfer_and_iou(target_mesh_gt, targ_trimesh)
                completed_targ_pc = filter_and_pad_point_clouds(completed_targ_pc)
            
            elif type == 'targo_full_targ' or type == 'targo_ptv3' or type == 'ptv3_scene':
                # For all these model types: use complete target mesh directly, no shape completion needed
                start_pc = time.time()
                # Sample points from ground truth target mesh (from complete_target preprocessing)
                completed_targ_pc = target_mesh_gt.sample(2048)
                completed_targ_pc = completed_targ_pc.astype(np.float32)
                completed_targ_grid = point_cloud_to_tsdf(completed_targ_pc)
                
                # Convert to tensor and normalize
                completed_targ_pc = torch.from_numpy(completed_targ_pc).to(device).unsqueeze(0)
                completed_targ_pc = completed_targ_pc / 0.3 - 0.5
                
                # Perfect reconstruction means CD and IoU are ideal (since using ground truth)
                cd = 0.0
                iou = 1.0
                
                completed_targ_pc = filter_and_pad_point_clouds(completed_targ_pc)
                
            elif type == 'targo_hunyun2':
                start_pc = time.time()
                # hunyun2_path = 'data/external/Gen3DSR/output_amodal/ycb_amodal_medium_occlusion_icp_v7_only_gt_1000'
                scene_path = hunyun2_path + '/' + scene_name + '/reconstruction/targ_obj_hy3dgen_align.ply'
                gt_scene_path = hunyun2_path + '/' + scene_name + '/reconstruction/gt_targ_obj.ply'
                meta_eval_path = hunyun2_path + '/' + scene_name + '/evaluation/meta_evaluation.txt'
                if os.path.exists(meta_eval_path):
                    with open(meta_eval_path, 'r') as f:
                        meta_content = f.read()
                        # Extract CD and IoU for v6
                        cd_match = re.search(r'Chamfer Distance v7_gt: ([\d\.]+)', meta_content)
                        iou_match = re.search(r'IoU v7_gt: ([\d\.]+)', meta_content)
                        if cd_match and iou_match:
                            cd = float(cd_match.group(1))
                            iou = float(iou_match.group(1))
                            print(f"CD: {cd:.4f}, IoU: {iou:.4f}")
                # scene_mesh = trimesh.load(scene_path)
                completed_targ_mesh = trimesh.load(scene_path)
                gt_scene_mesh = trimesh.load(gt_scene_path)
                completed_targ_mesh.export('completed_targ_mesh.obj')
                gt_scene_mesh.export('gt_scene_mesh.obj')

                # completed_targ_grid =  mesh_to_tsdf(completed_targ_mesh)
                # save_point_cloud_as_ply(completed_targ_mesh, 'completed_targ_mesh.obj')
                # save_point_cloud_as_ply(gt_scene_path, 'gt_scene_mesh.obj')
                ## save obj
                # save_mesh_as_ply(completed_targ_mesh, 'completed_targ_mesh.ply')
                # save_mesh_as_ply(gt_scene_path, 'gt_scene_mesh.ply')
                # Sample points from mesh using FPS (Farthest Point Sampling)
                completed_targ_pc = completed_targ_mesh.sample(2048)
                completed_targ_pc = completed_targ_pc.astype(np.float32)    
                # completed_targ_grid = point_cloud_to_tsdf(completed_targ_pc)
                completed_targ_grid = mesh_to_tsdf(completed_targ_mesh)

                completed_targ_pc = torch.from_numpy(completed_targ_pc).to(device).unsqueeze(0)
                # completed_targ_pc_rs = completed_targ_pc 
                completed_targ_pc = completed_targ_pc / 0.3 - 0.5
                save_point_cloud_as_ply(completed_targ_pc[0].cpu().numpy(), 'completed_targ_pc.ply')
                completed_targ_pc = filter_and_pad_point_clouds(completed_targ_pc)
                # completed_targ_grid = point_cloud_to_tsdf(completed_targ_pc_rs.cpu().numpy())
                # completed_targ_grid =  mesh_to_tsdf(completed_targ_mesh)
                # Print the shape of completed_targ_grid

                
                # print("Saved completed target grid visualization to completed_targ_grid_visualization.png")
                save_point_cloud_as_ply(completed_targ_pc[0].cpu().numpy(), 'completed_targ_pc.ply')
                # Convert mesh vertices and faces to numpy arrays
                # completed_targ_mesh_vertices = np.array(completed_targ_mesh.vertices)
                # completed_targ_mesh_faces = np.array(completed_targ_mesh.faces)
                # completed_targ_mesh.vertices = completed_targ_mesh_vertices
                # completed_targ_mesh.triangles = completed_targ_mesh_faces
                # completed_targ_grid = mesh_to_tsdf(completed_targ_mesh)

                ## make as float
                # completed_targ_grid = completed_targ_grid.astype
                # completed_targ_pc = completed_targ_pc.float()
                # completed_targ_grid = completed_targ_grid.reshape((self.resolution, self.resolution, self.resolution))
            # else:
            #     scene_mesh = None
            # elif type == 'FGC-GraspNet':
                
            # Create combined scene for model input
            if type == 'ptv3_scene':
                # For ptv3_scene, use only the scene point cloud
                targ_completed_scene_pc = scene_no_targ_pc
            else:
                # For other targo variants, combine scene and completed target
                targ_completed_scene_pc = torch.cat([scene_no_targ_pc, completed_targ_pc], dim=1)
            
            save_point_cloud_as_ply(targ_completed_scene_pc[0].cpu().numpy(), 'targ_completed_scene_pc.ply')

            ## save targ_completed_scene_pc
            # save_point_cloud_as_ply(completed_targ_pc[0].cpu().numpy(), 'completed_targ_pc.ply')
            # save_point_cloud_as_ply(targ_completed_scene_pc[0].cpu().numpy(), 'targ_completed_scene_pc.ply')
            

            convert_time = time.time() - start_pc
            print(f"convert: {convert_time:.3f}s")
                
        # if visual_dict is not None:
        #     mesh_dir = visual_dict['mesh_dir']
        #     mesh_name = visual_dict['mesh_name']
        #     path = f'{mesh_dir}/{mesh_name}_completed_targ_pc.ply'
        #     save_point_cloud_as_ply(targ_pc[0].cpu().numpy(), path)
        
        if type == 'targo' or type == 'targo_full_targ' or type == 'targo_hunyun2' or type == 'targo_ptv3' or type == 'ptv3_scene':
            if type == 'ptv3_scene':
                # For ptv3_scene, only pass scene point cloud
                inputs = (targ_completed_scene_pc, None)
            else:
                # For other targo variants, pass both scene and completed target
                inputs = (targ_completed_scene_pc, completed_targ_pc)
    # if type != 'FGC-GraspNet' and type != 'AnyGrasp':
        # targ_completed_scene_pc = inputs[0]
        # completed_targ_pc = inputs[1]
        # save_point_cloud_as_ply(targ_completed_scene_pc[0].cpu().numpy(), 'targ_completed_scene_pc.ply')
        # save_point_cloud_as_ply(completed_targ_pc[0].cpu().numpy(), 'completed_targ_pc.ply')
    time_grasp_start = time.time()
    with torch.no_grad():
        if type == 'FGC-GraspNet' or type == 'FGC_full_targ':
            end_points = {}
            # Convert inputs to torch tensor if it's not already
            if isinstance(inputs, np.ndarray):
                inputs = torch.from_numpy(inputs).float()
            
            # inputs = inputs / 0.3 - 0.5 ## this is important
            
            # Make sure inputs is a batch
            if len(inputs.shape) == 2:
                inputs = inputs.unsqueeze(0)
                
            # Move to CUDA device
            cloud = inputs
            # Convert PyTorch tensor to numpy array
            cloud_np = inputs[0].cpu().numpy()
            plane_hs = np.load('./data//plane_hs.npy')
            cloud_np = np.concatenate((cloud_np, plane_hs), axis=0)
            # Create open3d point cloud object
            cloud = o3d.geometry.PointCloud()
            # Set points for the point cloud
            cloud.points = o3d.utility.Vector3dVector(cloud_np)
            # Keep original tensor for model input
            inputs_tensor = inputs
            inputs = inputs.to(device)
            
            # Print shape for debugging
            print(f"FGC-GraspNet input shape: {inputs.shape}")
            # inputs = inputs.squeeze(0)
            end_points['point_clouds'] = inputs
            gg = get_grasps(net, end_points)
            # mask = gg.scores > 0.0
            # gg = gg[mask]
            # gg = collision_detection(gg, np.array(cloud.points))
            # gg = net(end_points)
            vis_grasps(gg, cloud, anygrasp=False, fgc=True)
            print(gg)
        elif type == 'AnyGrasp' or type == 'AnyGrasp_full_targ':
            points = inputs # [0,0.3]
            # inputs = inputs / 0.3 - 0.5
            colors = np.zeros_like(points) 
            
            # Create an Open3D point cloud object
            cloud_o3d = o3d.geometry.PointCloud()
            # Set the points of the point cloud
            cloud_o3d.points = o3d.utility.Vector3dVector(points)
            # Set the colors of the point cloud
            # cloud_o3d.colors = o3d.utility.Vector3dVector(colors)
            
            # xmin, xmax = 0.0, 1.0
            # ymin, ymax = 0.0, 1.0
            # zmin, zmax = 0.0, 1.0
            lower_bound=torch.tensor([0.02, 0.02, 0.055])    
            upper_bound=torch.tensor([0.28, 0.28, 0.3])
            xmin, xmax = lower_bound[0], upper_bound[0]
            ymin, ymax = lower_bound[1], upper_bound[1]
            zmin, zmax = lower_bound[2], upper_bound[2]
            lims = [xmin, xmax, ymin, ymax, zmin, zmax]
            points = points.astype(np.float32)
            colors = colors.astype(np.float32)
            gg, cloud = net.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=False)
            print(gg)
            # gg = get_grasps(net, end_points)
            vis_grasps(gg, cloud_o3d, anygrasp=True, fgc=False)
            print(gg)
        elif type == 'giga':
            query_points = torch.stack(torch.meshgrid(
                torch.linspace(-0.5, 0.475, 40),
                torch.linspace(-0.5, 0.475, 40),
                torch.linspace(-0.5, 0.475, 40)
            ), dim=-1).view(-1, 3).to(device)
            query_points = query_points.unsqueeze(0)
            qual_vol, rot_vol, width_vol, tsdf_vol = net(inputs, pos, query_points)
            # qual_vol, rot_vol, width_vol = net(inputs, pos)
        else:
            qual_vol, rot_vol, width_vol = net(inputs, pos)
        # ## from tsdf to mesh
        if type == 'giga':
            tsdf_vol = tsdf_vol.cpu().squeeze().numpy()
            tsdf_vol = tsdf_vol.reshape((40, 40, 40))
            scene_mesh = tsdf_to_mesh(tsdf_vol)
            # save_mesh_as_ply(scene_mesh, 'scene_mesh.ply')
            # save_mesh_as_ply(target_mesh_gt, 'target_mesh_gt.ply')
            # scene_mesh.export('scene_mesh.ply')
            # target_mesh_gt.export('target_mesh_gt.ply')
            scene_meshes = scene_mesh.split()
            
            # cd, iou = None, None
            cd, iou = 1, 0
            try:
                if len(scene_meshes) > 0 and target_mesh_gt is not None:
                    target_idx, max_overlap = find_target_mesh_from_scene(scene_meshes, target_mesh=target_mesh_gt, threshold=10)
                    if target_idx < len(scene_meshes):
                        target_mesh = scene_meshes[target_idx]
                        cd, iou = compute_chamfer_and_iou(target_mesh_gt, target_mesh)
                    else:
                        print("No mesh with sufficient match to target was found!")
                else:
                    print("Empty scene meshes or missing target mesh")
            except Exception as e:
                print(f"Error during target mesh processing: {e}")
            # save_point_cloud_as_ply(scene_mesh.vertices, 'scene_mesh.ply')
            # target_mesh.export('target_mesh_pred.ply')
        if not type == 'AnyGrasp' and not type == 'FGC-GraspNet' and not type == 'AnyGrasp_full_targ' and not type == 'FGC_full_targ':
            net_params_count = sum(p.numel() for p in net.parameters())
            print(f"Number of parameters in self.net: {net_params_count:,}")
    time_grasp = time.time() - time_grasp_start
    print(f"Grasp prediction time: {time_grasp:.3f}s")
    total_time = time_grasp + sc_time
    print(f"Total time: {total_time:.3f}s")

    # move output back to the CPU
    if type != 'AnyGrasp' and type != 'FGC-GraspNet' and type != 'AnyGrasp_full_targ' and type != 'FGC_full_targ':
        qual_vol = qual_vol.cpu().squeeze().numpy()
        rot_vol = rot_vol.cpu().squeeze().numpy()
        width_vol = width_vol.cpu().squeeze().numpy()
    # if (sc_net != None nyun2_path) and cd_iou_measure:
    if type == 'targo' or type == 'targo_full_targ' or type == 'targo_hunyun2' or type == 'targo_ptv3' or type == 'ptv3_scene':
        return qual_vol, rot_vol, width_vol, completed_targ_grid, cd, iou
    elif type == 'giga':
        return qual_vol, rot_vol, width_vol, cd, iou
    elif type == 'vgn':
        return qual_vol, rot_vol, width_vol
    elif type == 'AnyGrasp' or type == 'FGC-GraspNet':
        # return grasp, score
        return gg


def process_vgn(
    tsdf_vol,
    qual_vol,
    rot_vol,
    width_vol,
    gaussian_filter_sigma=1.0,
    min_width=1.33,
    max_width=9.33,
    out_th=0.5
):
    tsdf_vol = tsdf_vol.squeeze()

    # smooth quality volume with a Gaussian
    qual_vol = ndimage.gaussian_filter(
        qual_vol, sigma=gaussian_filter_sigma, mode="nearest"
    )

    # mask out voxels too far away from the surface
    outside_voxels = tsdf_vol > out_th
    inside_voxels = np.logical_and(1e-3 < tsdf_vol, tsdf_vol < out_th)
    valid_voxels = ndimage.morphology.binary_dilation(
        outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
    )
    qual_vol[valid_voxels == False] = 0.0

    # reject voxels with predicted widths that are too small or too large
    qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0

    return qual_vol, rot_vol, width_vol

def process(
    tsdf_vol,
    qual_vol,
    rot_vol,
    width_vol,
    gaussian_filter_sigma=1.0,
    min_width=0.033,
    max_width=0.233,
    out_th=0.5
):
    ## check if tsdf_vol is a tuple
    if isinstance(tsdf_vol, tuple):
        if len(tsdf_vol) == 2:
            tsdf_vol = tsdf_vol[0]
    tsdf_vol = tsdf_vol.squeeze()
    
    qual_vol = ndimage.gaussian_filter(
        qual_vol, sigma=gaussian_filter_sigma, mode="nearest"
    )

    # mask out voxels too far away from the surface
    outside_voxels = tsdf_vol > out_th
    inside_voxels = np.logical_and(1e-3 < tsdf_vol, tsdf_vol < out_th)
    valid_voxels = ndimage.morphology.binary_dilation(
        outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
    )
    # if qual_vol.shape == valid_voxels.shape:
    qual_vol[valid_voxels == False] = 0.0
    # else:
        # valid_voxels_flat = valid_voxels.reshape(-1) 
        # qual_vol[~valid_voxels_flat] = 0.0  

    # reject voxels with predicted widths that are too small or too large
    qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0
    # qual_vol = np.where(
    #     np.logical_or(width_vol < min_width, width_vol > max_width),
    #     0.0,
    #     qual_vol
    # )

    return qual_vol, rot_vol, width_vol


def process_dexycb(
    tsdf_vol,
    qual_vol,
    rot_vol,
    width_vol,
    gaussian_filter_sigma=1.0,
    min_width=0.033,
    max_width=0.233,
    out_th=0.5
):
    ## check if tsdf_vol is a tuple
    if isinstance(tsdf_vol, tuple):
        if len(tsdf_vol) == 2:
            tsdf_vol = tsdf_vol[0]
    tsdf_vol = tsdf_vol.squeeze()
    
    qual_vol = ndimage.gaussian_filter(
        qual_vol, sigma=gaussian_filter_sigma, mode="nearest"
    )

    outside_voxels = tsdf_vol > out_th
    qual_vol[outside_voxels == False] = 0.0

    # reject voxels with predicted widths that are too small or too large
    qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0

    return qual_vol, rot_vol, width_vol


def select_vgn(qual_vol, rot_vol, width_vol, threshold=0.90, max_filter_size=4, force_detection=False):
    best_only = False
    qual_vol[qual_vol < LOW_TH] = 0.0
    if force_detection and (qual_vol >= threshold).sum() == 0:
        best_only = True
    else:
        # threshold on grasp quality
        qual_vol[qual_vol < threshold] = 0.0

    # non maximum suppression
    max_vol = ndimage.maximum_filter(qual_vol, size=max_filter_size)
    qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0)
    mask = np.where(qual_vol, 1.0, 0.0)

    # construct grasps
    grasps, scores = [], []
    for index in np.argwhere(mask):
        grasp, score = select_index_vgn(qual_vol, rot_vol, width_vol, index)
        grasps.append(grasp)
        scores.append(score)

    sorted_grasps = [grasps[i] for i in reversed(np.argsort(scores))]
    sorted_scores = [scores[i] for i in reversed(np.argsort(scores))]

    if best_only and len(sorted_grasps) > 0:
        sorted_grasps = [sorted_grasps[0]]
        sorted_scores = [sorted_scores[0]]

    return sorted_grasps, sorted_scores


def select(qual_vol, center_vol, rot_vol, width_vol, threshold=0.90, max_filter_size=4, force_detection=False):
    best_only = False
    qual_vol[qual_vol < LOW_TH] = 0.0
    if force_detection and (qual_vol >= threshold).sum() == 0:
        best_only = True
    else:
        # threshold on grasp quality
        qual_vol[qual_vol < threshold] = 0.0

    # non maximum suppression
    max_vol = ndimage.maximum_filter(qual_vol, size=max_filter_size)
    qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0)
    mask = np.where(qual_vol, 1.0, 0.0)

    # construct grasps
    grasps, scores = [], []
    for index in np.argwhere(mask):
        grasp, score = select_index(qual_vol, center_vol, rot_vol, width_vol, index)
        grasps.append(grasp)
        scores.append(score)

    sorted_grasps = [grasps[i] for i in reversed(np.argsort(scores))]
    sorted_scores = [scores[i] for i in reversed(np.argsort(scores))]

    if best_only and len(sorted_grasps) > 0:
        sorted_grasps = [sorted_grasps[0]]
        sorted_scores = [sorted_scores[0]]
        
    return sorted_grasps, sorted_scores

def select_target(qual_vol, center_vol, rot_vol, width_vol, threshold=0.90, max_filter_size=4, force_detection=False):
    max_vol = ndimage.maximum_filter(qual_vol, size=max_filter_size)
    qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0)
    mask = np.where(qual_vol, 1.0, 0.0)

    # construct grasps
    grasps, scores = [], []
    for index in np.argwhere(mask):
        grasp, score = select_index(qual_vol, center_vol, rot_vol, width_vol, index)
        grasps.append(grasp)
        scores.append(score)

    sorted_grasps = [grasps[i] for i in reversed(np.argsort(scores))]
    sorted_scores = [scores[i] for i in reversed(np.argsort(scores))]
    sorted_grasps = [sorted_grasps[0]]
    sorted_scores = [sorted_scores[0]]
    
    return sorted_grasps, sorted_scores

def select_index(qual_vol, center_vol, rot_vol, width_vol, index):
    i, j, k = index
    score = qual_vol[i, j, k]
    ori = Rotation.from_quat(rot_vol[i, j, k])
    pos = center_vol[i, j, k].numpy()
    width = width_vol[i, j, k]
    return Grasp(Transform(ori, pos), width), score

def select_index_vgn(qual_vol, rot_vol, width_vol, index):
    i, j, k = index
    score = qual_vol[i, j, k]
    ori = Rotation.from_quat(rot_vol[:, i, j, k])
    pos = np.array([i, j, k], dtype=np.float64)
    width = width_vol[i, j, k]
    return Grasp(Transform(ori, pos), width), score

def find_target_mesh_from_scene(scene_meshes, target_mesh, threshold=10):
    """
    Find the mesh from the scene that best matches the target mesh
    
    Args:
        scene_meshes: list[trimesh.Trimesh], all meshes in the scene
        target_mesh: trimesh.Trimesh, target mesh
        threshold: int, minimum overlap point count threshold, default is 10
        
    Returns:
        target_idx: int, index of the target mesh in scene_meshes
        overlap: int, maximum overlap point count
    """
    from vgn.ConvONets.utils.libmesh import check_mesh_contains
    import numpy as np
    
    target_idx = -1
    max_overlap = threshold  # Set a minimum threshold
    
    for i, mesh in enumerate(scene_meshes):
        # Check overlap between each mesh and target mesh vertices
        overlap = np.sum(check_mesh_contains(mesh, target_mesh.vertices))
        
        # Update maximum overlap count and corresponding mesh index
        if overlap > max_overlap:
            max_overlap = overlap
            target_idx = i
    
    if target_idx == -1:
        print('No mesh with sufficient match to target was found!')
    else:
        print(f'Target mesh found, index: {target_idx}, overlap points: {max_overlap}')
    
    return target_idx, max_overlap

def filter_grasps_by_target(gg, target_pc):
    """Filter grasps by target point cloud.
    
    Args:
        gg: GraspGroup from FGC-GraspNet or AnyGrasp
        target_pc: target point cloud, in the same coordinate system as the scene point cloud
        
    Returns:
        target_gg: GraspGroup with only grasps on the target
    """
    # Ensure there are grasps to filter
    if gg is None or len(gg) == 0:
        print("Warning: No grasps to filter")
        # Return an empty GraspGroup
        return GraspGroup()
        
    # Get grasp contact points
    grasp_points = gg.translations
    
    # Create a KDTree for target PC
    import open3d as o3d
    # target_o3d = o3d.geometry.PointCloud()
    # target_o3d.points = o3d.utility.Vector3dVector(target_pc)
    # target_o3d_tree = o3d.geometry.KDTreeFlann(target_o3d)
    
    # Alternative: use scipy KDTree
    from scipy.spatial import KDTree
    target_tree = KDTree(target_pc)
    
    # Filter grasps: keep those close to target PC
    target_indices = []
    target_scores = []
    threshold = 0.02  # 2cm threshold for considering a grasp on the target
    
    # Get distances to nearest target points
    distances, _ = target_tree.query(grasp_points, k=1)
    
    # Print distances for debugging
    print(f"Total grasps: {len(gg)}")
    if len(distances) > 0:
        print(f"Min distance: {distances.min():.4f}")
        print(f"Max distance: {distances.max():.4f}")
    
    # Get indices where distance is below threshold
    target_indices = np.where(distances < threshold)[0]
    
    # If no grasps are close enough to target, return empty GraspGroup
    if len(target_indices) == 0:
        print("Warning: No grasps close to target object")
        return GraspGroup()
    
    # Create new GraspGroup with filtered grasps
    target_gg = gg[target_indices.tolist()]
    print(f"Valid grasps: {len(target_gg)}")
    
    return target_gg