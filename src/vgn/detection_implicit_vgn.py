import time
import numpy as np
import trimesh
from scipy import ndimage
import torch
import os
import re
import argparse
import open3d as o3d

from src.vgn.grasp import *
from src.vgn.utils.transform import Transform, Rotation
from src.vgn.networks import load_network
from src.vgn.utils import visual
from src.utils_giga import *
from src.vgn.grasp_conversion import anygrasp_to_vgn, fgc_to_vgn, anygrasp_to_vgn_with_region_filter, fgc_to_vgn_with_region_filter

from src.utils_giga import tsdf_to_ply, point_cloud_to_tsdf, save_numpy_as_ply, mesh_to_tsdf, point_cloud_to_mesh
from src.utils_targo import tsdf_to_mesh, filter_grasps_by_target

LOW_TH = 0.0

def vis_grasps_target(target_gg, target_cloud, scene_cloud, anygrasp=False, fgc=False, output_prefix=None):
    """Visualize grasps for target object"""
    # Process grasp group
    target_gg.nms()
    target_gg.sort_by_score()
    target_gg = target_gg[:20]
    grippers = target_gg.to_open3d_geometry_list()
    
    # Create colored point clouds
    target_cloud_colored = o3d.geometry.PointCloud()
    target_cloud_colored.points = target_cloud.points
    target_cloud_colored.colors = o3d.utility.Vector3dVector(np.ones((len(target_cloud.points), 3)) * np.array([1, 0, 0]))  # Red color
    
    scene_cloud_colored = o3d.geometry.PointCloud()
    scene_cloud_colored.points = scene_cloud.points
    scene_cloud_colored.colors = o3d.utility.Vector3dVector(np.ones((len(scene_cloud.points), 3)) * np.array([0, 1, 0]))  # Green color
    
    # Print statistics for debugging
    target_points = np.asarray(target_cloud.points)
    scene_points = np.asarray(scene_cloud.points)
    print(f"Scene points: {len(scene_points)}, Target points: {len(target_points)}")
    
    # Determine output file names based on grasp type and prefix
    if output_prefix is None:
        prefix = 'demo/cloud'
    else:
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
    """Visualize grasps for general scene"""
    gg.nms()
    gg.sort_by_score()
    gg = gg[:50]
    grippers = gg.to_open3d_geometry_list()
    
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
    """VGN Implicit grasp planner - simplified for VGN only"""
    def __init__(self, model_path, model_type, best=False, force_detection=False, qual_th=0.9, out_th=0.5, visualize=False, resolution=40, cd_iou_measure=False, select_top=False, sc_model_path=None, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load VGN network
        self.net = load_network(model_path, self.device, model_type=model_type) 
        self.net = self.net.eval()
        
        net_params_count = sum(p.numel() for p in self.net.parameters())
        print(f"Number of parameters in self.net: {net_params_count}")
        
        # Load shape completion network for TARGO models (only for targo model type)
        if model_type == 'targo' and sc_model_path:
            try:
                from src.shape_completion.models.AdaPoinTr import AdaPoinTr
                from src.shape_completion.config import cfg_from_yaml_file
                from src.shape_completion import builder
                
                sc_cfg = cfg_from_yaml_file("src/shape_completion/configs/AdaPoinTr.yaml")
                sc_net = AdaPoinTr(sc_cfg.model)
                builder.load_model(sc_net, sc_model_path)
                sc_net = sc_net.eval()
                sc_net_params_count = sum(p.numel() for p in sc_net.parameters())
                print(f"Number of parameters in sc_net: {sc_net_params_count}")
                self.sc_net = sc_net
            except ImportError as e:
                print(f"Warning: Could not import AdaPoinTr: {e}")
                self.sc_net = None
            except Exception as e:
                print(f"Warning: Could not load shape completion network: {e}")
                self.sc_net = None
        else:
            # No shape completion for VGN, targo_partial, targo_full_gt, or when sc_model_path is not provided
            self.sc_net = None
        
        self.qual_th = qual_th
        self.best = best
        self.force_detection = force_detection
        self.out_th = out_th
        self.visualize = visualize
        self.resolution = resolution
        self.cd_iou_measure = cd_iou_measure
        self.select_top = select_top
        self.sc_model_path = sc_model_path
        
        # Create position grid for VGN
        x, y, z = torch.meshgrid(
            torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution), 
            torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution), 
            torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution)
        )
        pos = torch.stack((x, y, z), dim=-1).float().unsqueeze(0).to(self.device)   
        self.pos = pos.view(1, self.resolution * self.resolution * self.resolution, 3)

        # Load plane points
        plane = np.load("./setup/plane_sampled.npy")
        plane = plane.astype(np.float32)
        self.plane = plane / 0.3 - 0.5

    def __call__(self, state, scene_mesh=None, visual_dict=None, scene_name=None, cd_iou_measure=False, target_mesh_gt=None, aff_kwargs={}):
        """Main inference function for VGN"""
        if visual_dict is None:
            visual_dict = {}
        scene_name = scene_name
        print(state.__dict__.keys())

        # VGN and TARGO specific processing
        if state.type == 'vgn':
            # Get scene grid and target grid
            scene_grid = state.scene_grid
            targ_grid = state.targ_grid
            voxel_size, size = state.tsdf.voxel_size, state.tsdf.size
            
            # VGN prediction
            with torch.no_grad():
                qual_vol, rot_vol, width_vol = predict_vgn(scene_grid, self.pos, self.net, self.device)
        
        elif state.type in ['targo', 'targo_partial', 'targo_full_gt', 'targo_full_targ', 'targo_hunyun2', 'targo_ptv3', 'ptv3_scene']:
            # TARGO specific processing
            scene_no_targ_pc = state.scene_no_targ_pc
            scene_no_targ_pc = np.concatenate((scene_no_targ_pc, self.plane), axis=0)

            if state.type == 'targo':
                if self.sc_net is not None:
                    targ_pc = state.targ_pc
                    print(f"Applying shape completion network to target point cloud with {len(targ_pc)} points")
                    try:
                        # Ensure sc_net is on the same device as input
                        self.sc_net = self.sc_net.to(self.device)
                        
                        # Convert to torch tensor and add batch dimension
                        targ_pc_tensor = torch.from_numpy(targ_pc).float().unsqueeze(0).to(self.device)  # Shape: (1, N, 3)
                        
                        # Apply shape completion network
                        with torch.no_grad():
                            completed_targ_pc = self.sc_net(targ_pc_tensor)  # Shape: (1, N, 3)
                        
                        # Convert back to numpy and remove batch dimension
                        targ_pc = completed_targ_pc[1].cpu().numpy().squeeze()  # Shape: (N, 3)
                        targ_pc = np.nan_to_num(targ_pc, nan=0.0, posinf=0.5 - 1e-6, neginf=-0.5)
                        targ_pc = np.clip(targ_pc, -0.5, 0.5 - 1e-6)
                        targ_pc_world = (targ_pc + 0.5) * 0.3
                        complete_targ_grid = point_cloud_to_tsdf(targ_pc_world)
                        complete_targ_mesh = point_cloud_to_mesh(targ_pc_world)
                        # targ_pc = filter_and_pad_point_clouds(targ_pc)
                        # state.targ_grid = complete_targ_grid
                        state.complete_targ_grid = complete_targ_grid
                        
                        # Add completed target mesh to visualization results
                        if visual_dict is not None:
                            visual_dict['completed_target_mesh'] = complete_targ_mesh
                            print(f"Added completed target mesh to visualization results")
                            print(f"Debug: visual_dict keys after adding: {list(visual_dict.keys())}")
                        else:
                            print(f"Debug: visual_dict is None, cannot add completed_target_mesh")
                        
                        if self.visualize:
                            save_numpy_as_ply(targ_pc, 'targ_pc.ply')
                            save_numpy_as_ply(scene_no_targ_pc, 'scene_no_targ_pc.ply')
                        print(f"Shape completion completed. New target point cloud has {len(targ_pc)} points")
                        
                    except Exception as e:
                        print(f"Error applying shape completion network: {e}")
                        print("Using original target point cloud")
                else:
                    print("No shape completion network available, using original target point cloud")
                
                inputs = (scene_no_targ_pc, targ_pc)
            
            elif state.type == 'targo_partial':
                # targo_partial: 使用原始的 targ_pc，不使用 shape completion
                targ_pc = state.targ_pc
                targ_pc_tensor = torch.from_numpy(targ_pc).float().unsqueeze(0).to(self.device)
                targ_pc_tensor = filter_and_pad_point_clouds(targ_pc_tensor)
                targ_pc = targ_pc_tensor.squeeze(0).cpu().numpy()

                print(f"Using original partial target point cloud with {len(targ_pc)} points")
                inputs = (scene_no_targ_pc, targ_pc)
                
            elif state.type == 'targo_full_gt':
                # targo_full_gt: 从 target_mesh_gt 采样点云，生成完整的 target grid
                # if hasattr(state, 'target_mesh_gt') and state.target_mesh_gt is not None:
                try:
                    print("Sampling points from ground truth target mesh")
                    # 从 mesh 采样点云
                    targ_pc_world, _ = trimesh.sample.sample_surface(target_mesh_gt, 2048)
                    # 转换到 [-0.5, 0.5] 空间
                    targ_pc = (targ_pc_world / 0.3) - 0.5
                    
                    # 生成完整的 target grid
                    complete_targ_grid_gt = mesh_to_tsdf(target_mesh_gt)
                    state.complete_targ_grid_gt = complete_targ_grid_gt
                    
                    print(f"Generated complete target point cloud with {len(targ_pc)} points from ground truth mesh")
                    save_numpy_as_ply(targ_pc, 'targ_pc_gt.ply')
                    save_numpy_as_ply(scene_no_targ_pc, 'scene_no_targ_pc_gt.ply')
                
                except Exception as e:
                    print(f"Error sampling points from ground truth target mesh: {e}")
                    print("Using original target point cloud")
                    targ_pc = state.targ_pc
     
                inputs = (scene_no_targ_pc, targ_pc)
            # save_numpy_as_ply(scene_no_targ_pc, 'scene_no_targ_pc.ply')
            
            # Handle different input formats for PointTransformerV3 models
            if state.type == 'targo_ptv3':
                # For targo_ptv3: use both scene and target point clouds
                targ_pc = state.targ_pc
                
                # Apply shape completion network to targ_pc if available
                if self.sc_net is not None:
                    print(f"Applying shape completion network to target point cloud with {len(targ_pc)} points")
                    try:
                        # Ensure sc_net is on the same device as input
                        self.sc_net = self.sc_net.to(self.device)
                        
                        # Convert to torch tensor and add batch dimension
                        targ_pc_tensor = torch.from_numpy(targ_pc).float().unsqueeze(0).to(self.device)  # Shape: (1, N, 3)
                        
                        # Apply shape completion network
                        with torch.no_grad():
                            completed_targ_pc = self.sc_net(targ_pc_tensor)  # Shape: (1, N, 3)
                        
                        # Convert back to numpy and remove batch dimension
                        targ_pc = completed_targ_pc.squeeze(0).cpu().numpy()  # Shape: (N, 3)
                        print(f"Shape completion completed. New target point cloud has {len(targ_pc)} points")
                        
                    except Exception as e:
                        print(f"Error applying shape completion network: {e}")
                        print("Using original target point cloud")
                else:
                    print("No shape completion network available, using original target point cloud")
                
                inputs = (scene_no_targ_pc, targ_pc)
            elif state.type == 'ptv3_scene':
                # For ptv3_scene: only use scene point cloud (target is included in scene)
                if hasattr(state, 'targ_pc') and state.targ_pc is not None:
                    # If target PC is available, concatenate it with scene for full scene representation
                    full_scene_pc = np.concatenate((scene_no_targ_pc, state.targ_pc), axis=0)
                    
                    # Create labels: 0 for scene/occluder points, 1 for target points
                    scene_labels = np.zeros((len(scene_no_targ_pc), 1), dtype=np.float32)
                    target_labels = np.ones((len(state.targ_pc), 1), dtype=np.float32)
                    
                    # Combine points and labels
                    scene_with_labels = np.concatenate([scene_no_targ_pc, scene_labels], axis=1)
                    target_with_labels = np.concatenate([state.targ_pc, target_labels], axis=1)
                    full_scene_with_labels = np.concatenate([scene_with_labels, target_with_labels], axis=0)
                    
                    inputs = (full_scene_with_labels, None)
                else:
                    # Use scene_no_targ_pc as is
                    inputs = (scene_no_targ_pc, None)
            
            # TARGO prediction
            with torch.no_grad():
                qual_vol, rot_vol, width_vol = predict_targo(inputs, self.net, self.device)
            
            # Get voxel size and size from state
            # voxel_size, size = state.tsdf.voxel_size, state.tsdf.size
            voxel_size, size = 0.0075, 0.3
        
        begin = time.time()

        # Reshape volumes
        qual_vol = qual_vol.reshape((self.resolution, self.resolution, self.resolution))
        rot_vol = rot_vol.reshape((self.resolution, self.resolution, self.resolution, 4))
        width_vol = width_vol.reshape((self.resolution, self.resolution, self.resolution))

        # Process volumes based on model type
        if state.type == 'vgn':
            qual_vol, rot_vol, width_vol = process_vgn(targ_grid, qual_vol, rot_vol, width_vol, out_th=self.out_th)
        elif state.type == 'targo':
            # TARGO with shape completion uses complete_targ_grid
            qual_vol, rot_vol, width_vol = process(state.complete_targ_grid, qual_vol, rot_vol, width_vol, out_th=self.out_th)
        elif state.type == 'targo_partial':
            # targo_partial uses original targ_grid
            qual_vol, rot_vol, width_vol = process(state.targ_grid, qual_vol, rot_vol, width_vol, out_th=self.out_th)
        elif state.type == 'targo_full_gt':
            # targo_full_gt uses complete_targ_grid_gt generated from ground truth mesh
            if hasattr(state, 'complete_targ_grid_gt'):
                qual_vol, rot_vol, width_vol = process(state.complete_targ_grid_gt, qual_vol, rot_vol, width_vol, out_th=self.out_th)
            else:
                # Fallback to original targ_grid if complete_targ_grid_gt is not available
                qual_vol, rot_vol, width_vol = process(state.targ_grid, qual_vol, rot_vol, width_vol, out_th=self.out_th)
        else:
            qual_vol, rot_vol, width_vol = process(state.targ_grid, qual_vol, rot_vol, width_vol, out_th=self.out_th)
        
        # Handle different volume shapes
        if len(qual_vol.shape) == 1:
            qual_vol = qual_vol.reshape((self.resolution, self.resolution, self.resolution))
        qual_vol = bound(qual_vol, voxel_size)

        # Initialize visual_mesh as None
        visual_mesh = None
        
        if self.visualize:
            colored_scene_mesh = visual.affordance_visual(qual_vol, rot_vol, scene_mesh, size, self.resolution, **aff_kwargs)
            visual_dict['affordance_visual'] = colored_scene_mesh
            visual_mesh = colored_scene_mesh

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

        end = time.time()
        print(f"post processing: {end-begin:.3f}s")
        
        # Create composed_scene with grasps if visualize is enabled
        if self.visualize and visual_mesh is not None:
            # Always create composed_scene, even if grasps is empty
            composed_scene = trimesh.Scene(visual_mesh)
            
            # Add grasp meshes if grasps are available
            if len(grasps) > 0:
                grasp_mesh_list = [visual.grasp2mesh(g, s) for g, s in zip(grasps, scores)]
                for i, g_mesh in enumerate(grasp_mesh_list):
                    composed_scene.add_geometry(g_mesh, node_name=f'grasp_{i}')
            
            visual_dict['composed_scene'] = composed_scene
        
        # Return grasps, scores, timing, and visual_mesh for TARGO models
        if state.type in ['targo', 'targo_partial', 'targo_full_gt', 'targo_full_targ', 'targo_hunyun2', 'targo_ptv3', 'ptv3_scene']:
            return grasps, scores, end - begin, visual_mesh
        else:
            # VGN returns volumes for visualization
            return qual_vol, rot_vol, width_vol

def bound(qual_vol, voxel_size, limit=[0.02, 0.02, 0.055]):
    """Avoid grasp out of bound"""
    x_lim = int(limit[0] / voxel_size)
    y_lim = int(limit[1] / voxel_size)
    z_lim = int(limit[2] / voxel_size)
    qual_vol[:x_lim] = 0.0
    qual_vol[-x_lim:] = 0.0
    qual_vol[:, :y_lim] = 0.0
    qual_vol[:, -y_lim:] = 0.0
    qual_vol[:, :, :z_lim] = 0.0
    return qual_vol

def predict_vgn(scene_grid, pos, net, device):
    """VGN prediction function"""
    # Convert to tensor
    scene_grid = torch.from_numpy(scene_grid).to(device)
    
    # VGN forward pass
    with torch.no_grad():
        qual_vol, rot_vol, width_vol = net(scene_grid, pos)
    
    # Move output back to CPU
    qual_vol = qual_vol.cpu().squeeze().numpy()
    rot_vol = rot_vol.cpu().squeeze().numpy()
    width_vol = width_vol.cpu().squeeze().numpy()
    
    return qual_vol, rot_vol, width_vol

def predict_targo(inputs, net, device):
    """TARGO prediction function"""
    # Convert inputs to tensors
    if isinstance(inputs, tuple):
        scene_pc, targ_pc = inputs
        # scene_pc = torch.from_numpy(scene_pc).to(device)
        scene_pc = torch.tensor(scene_pc, dtype=torch.float32, device=device)
        scene_pc = scene_pc.unsqueeze(0)
        if targ_pc is not None:
            # targ_pc = torch.from_numpy(targ_pc).to(device)
            targ_pc = torch.tensor(targ_pc, dtype=torch.float32, device=device)
            targ_pc = targ_pc.unsqueeze(0)
            inputs = (scene_pc, targ_pc)
        else:
            inputs = (scene_pc, None)
    else:
        inputs = torch.from_numpy(inputs).to(device)
    
    # Create position grid for TARGO
    pos = torch.meshgrid(
        torch.linspace(-0.5, 0.475, 40),
        torch.linspace(-0.5, 0.475, 40),
        torch.linspace(-0.5, 0.475, 40)
    )
    pos = torch.stack(pos, dim=-1).view(-1, 3).to(device)
    pos = pos.unsqueeze(0)
    
    # TARGO forward pass
    with torch.no_grad():
        if isinstance(inputs, tuple):
            qual_vol, rot_vol, width_vol = net(inputs, pos)
        else:
            qual_vol, rot_vol, width_vol = net(inputs, pos)
    
    # Move output back to CPU
    qual_vol = qual_vol.cpu().squeeze().numpy()
    rot_vol = rot_vol.cpu().squeeze().numpy()
    width_vol = width_vol.cpu().squeeze().numpy()
    
    return qual_vol, rot_vol, width_vol

def process_vgn(tsdf_vol, qual_vol, rot_vol, width_vol, gaussian_filter_sigma=1.0, min_width=0.033, max_width=0.233, out_th=0.5):
    """Process VGN volumes"""
    tsdf_vol = tsdf_vol.squeeze()
    
    # Smooth quality volume with a Gaussian
    qual_vol = ndimage.gaussian_filter(qual_vol, sigma=gaussian_filter_sigma, mode="nearest")

    # Mask out voxels too far away from the surface
    outside_voxels = tsdf_vol > out_th
    inside_voxels = np.logical_and(1e-3 < tsdf_vol, tsdf_vol < out_th)
    valid_voxels = ndimage.morphology.binary_dilation(
        outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
    )
    qual_vol[valid_voxels == False] = 0.0

    # Reject voxels with predicted widths that are too small or too large
    qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0

    return qual_vol, rot_vol, width_vol

def process(tsdf_vol, qual_vol, rot_vol, width_vol, gaussian_filter_sigma=1.0, min_width=0.033, max_width=0.233, out_th=0.5):
    """Process TARGO volumes"""
    # Check if tsdf_vol is a tuple
    # if isinstance(tsdf_vol, tuple):
    #     if len(tsdf_vol) == 2:
    #         tsdf_vol = tsdf_vol[0]
    tsdf_vol = tsdf_vol.squeeze()
    
    qual_vol = ndimage.gaussian_filter(
        qual_vol, sigma=gaussian_filter_sigma, mode="nearest"
    )

    # Mask out voxels too far away from the surface
    outside_voxels = tsdf_vol > out_th
    inside_voxels = np.logical_and(1e-3 < tsdf_vol, tsdf_vol < out_th)
    valid_voxels = ndimage.morphology.binary_dilation(
        outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
    )
    qual_vol[valid_voxels == False] = 0.0

    # Reject voxels with predicted widths that are too small or too large
    qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0

    return qual_vol, rot_vol, width_vol

def select(qual_vol, center_vol, rot_vol, width_vol, threshold=0.90, max_filter_size=4, force_detection=False):
    """Select grasps from volumes"""
    best_only = False
    qual_vol[qual_vol < LOW_TH] = 0.0
    if force_detection and (qual_vol >= threshold).sum() == 0:
        best_only = True
    else:
        # Threshold on grasp quality
        qual_vol[qual_vol < threshold] = 0.0

    # Non maximum suppression
    max_vol = ndimage.maximum_filter(qual_vol, size=max_filter_size)
    qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0)
    mask = np.where(qual_vol, 1.0, 0.0)

    # Construct grasps
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

def select_index(qual_vol, center_vol, rot_vol, width_vol, index):
    """Select grasp at specific index"""
    i, j, k = index
    score = qual_vol[i, j, k]
    ori = Rotation.from_quat(rot_vol[i, j, k])
    pos = center_vol[i, j, k].numpy()
    width = width_vol[i, j, k]
    return Grasp(Transform(ori, pos), width), score
