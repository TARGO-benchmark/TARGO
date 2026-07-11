import glob
import os
import open3d as o3d
import trimesh
# import mcubes
import numpy as np
import json
import h5py
from src.vgn.utils import visual
from src.vgn.ConvONets.eval import MeshEvaluator
from collections import OrderedDict
# Load and parse URDF files
from urdfpy import URDF
from torch import nn
# from src.vgn.simulation import ClutterRemovalSim
from src.vgn.utils.implicit import *
# from src.vgn.simulation import ClutterRemovalSim
from src.vgn.perception import *
from src.vgn.utils.transform import Rotation, Transform
# from src.vgn.utils.misc import apply_noise, apply_translational_noise
# from acronym_tools import load_mesh, create_gripper_marker
# from src.vgn.simulation import ClutterRemovalSim
from pathlib import Path
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap
# import pymeshfix
# import pyvista as pv 
from pysdf import SDF

# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

# def trimesh_scene2collisio_manager(trimesh_scene):
#     collision_manager = trimesh.collision.CollisionManager()
#     for mesh in trimesh_scene.geometry.values():
#         collision_manager.add_object(mesh.visual, mesh.visual_name)
#     return collision_manager

import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt

import torch.nn.functional as F

import torch
# import MinkowskiEngine as ME

import sys
# print(sys.path)
# from shape_completion.data_transforms import Compose

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial import KDTree


def get_pc_scene_no_targ_kdtree(scene_pc: np.ndarray, targ_pc: np.ndarray, tol=1e-6):
    """
    Remove all points from `scene_pc` that are within a distance `tol` of any point in `targ_pc`.
    This uses a KDTree for efficient nearest neighbor search.
    
    Parameters:
        scene_pc (np.ndarray): The full scene point cloud, shape (N, 3).
        targ_pc (np.ndarray): The target object point cloud to remove, shape (M, 3).
        tol (float): Distance threshold to determine overlap.

    Returns:
        np.ndarray: The filtered point cloud with target points removed.
    """
    # Build a KDTree from the target points
    tree = cKDTree(targ_pc)

    # For each point in the scene, query the nearest neighbor in targ_pc within distance `tol`
    dists, _ = tree.query(scene_pc, distance_upper_bound=tol)

    # Points with no neighbor within `tol` will have distance = inf
    mask = np.isinf(dists)  # Keep only points with no close match in targ_pc

    # Return filtered scene point cloud
    return scene_pc[mask]

def filter_grasps_by_target(gg, target_pc, distance_threshold=0.05):
    """
    Filter grasps that are close to the target object
    
    Args:
        gg: GraspGroup object
        target_pc: Target point cloud, shape (N, 3)
        distance_threshold: Distance threshold, grasps with distances less than this value are considered valid
    
    Returns:
        filtered_gg: Filtered GraspGroup object
    """
    # Build KD-tree for target point cloud
    target_kdtree = KDTree(target_pc)
    
    # Calculate minimum distance from each grasp point to target point cloud
    distances, _ = target_kdtree.query(gg.translations)
    
    # Get indices of grasps with distances less than threshold
    valid_grasp_mask = distances < distance_threshold
    
    # Print debug information
    print(f"Total grasps: {len(gg.translations)}")
    print(f"Valid grasps: {np.sum(valid_grasp_mask)}")
    print(f"Min distance: {np.min(distances):.4f}")
    print(f"Max distance: {np.max(distances):.4f}")
    
    # Return filtered grasps
    return gg[valid_grasp_mask]

def adjust_point_cloud_size(pc, target_size=20000):
    """
    Adjust point cloud to target size (20000, 3).
    If fewer points than target_size, duplicate points and pad with zeros.
    If more points than target_size, randomly sample points.
    
    Args:
        pc: Input point cloud of shape (N, 3)
        target_size: Target number of points (default: 20000)
    
    Returns:
        Adjusted point cloud of shape (target_size, 3)
    """
    print(f"Original point cloud shape: {pc.shape}")
    
    if pc.shape[0] < target_size:
        # If fewer points than target_size, first duplicate existing points
        num_to_duplicate = min(target_size - pc.shape[0], pc.shape[0])
        duplicate_indices = np.random.choice(pc.shape[0], num_to_duplicate, replace=True)
        duplicated_points = pc[duplicate_indices]
        pc = np.vstack([pc, duplicated_points])
        
        # If still not enough, pad with zeros
        if pc.shape[0] < target_size:
            padding = np.zeros((target_size - pc.shape[0], 3), dtype=pc.dtype)
            pc = np.vstack([pc, padding])
        
        print(f"Padded point cloud to shape: {pc.shape}")
    elif pc.shape[0] > target_size:
        # If more points than target_size, randomly sample
        indices = np.random.choice(pc.shape[0], target_size, replace=False)
        pc = pc[indices]
        print(f"Randomly sampled point cloud to shape: {pc.shape}")
    
    assert pc.shape == (target_size, 3), f"Expected shape ({target_size}, 3), got {pc.shape}"
    return pc

def point_cloud_to_mesh(points, alpha=0.5, mesh_fix=False):
    """
    Convert point cloud to mesh using alpha shape reconstruction.
    
    Args:
        points (np.ndarray or torch.Tensor): 
            Point cloud data of shape (N, 3) or (B, N, 3).
        alpha (float, optional): 
            Alpha value for alpha shape reconstruction. Default is 0.5.
        mesh_fix (bool, optional): 
            Whether to fix mesh using pymeshfix. Default is False.
            
    Returns:
        trimesh.Trimesh: Reconstructed mesh.
    """
    # Convert points to numpy if it's a torch tensor
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    
    # Reshape if points is 3D
    if len(points.shape) == 3:
        points = points.reshape(-1, 3)
        
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Create mesh using alpha shape
    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha, tetra_mesh, pt_map
    )

    # Fix mesh if requested
    if mesh_fix:
        try:
            import pymeshfix
            mf = pymeshfix.MeshFix(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
            mf.repair()
            mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(mf.mesh[0]),
                triangles=o3d.utility.Vector3iVector(mf.mesh[1])
            )
        except ImportError:
            print("Warning: pymeshfix not available. Skipping mesh fixing.")
    
    # Convert to trimesh for compatibility
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    return trimesh_mesh

def save_open3d_mesh(mesh, file_path):
    """Save Open3D mesh to file.
    
    Args:
        mesh: Open3D TriangleMesh object
        file_path (str): Path to save the mesh
    """
    try:
        import open3d as o3d
        if isinstance(mesh, o3d.geometry.TriangleMesh):
            # Convert to trimesh for export
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            trimesh_mesh.export(file_path)
        else:
            # Assume it's already a trimesh object
            mesh.export(file_path)
    except Exception as e:
        print(f"Error saving mesh to {file_path}: {e}")

def compute_chamfer_and_iou(target_mesh, completed_pc, mesh_path=None):
    """Compute Chamfer distance and IoU between target mesh and completed point cloud.
    
    Args:
        target_mesh (trimesh.Trimesh or open3d.geometry.TriangleMesh): Ground truth target mesh
        completed_pc (numpy.ndarray): Completed target point cloud
        mesh_path (Path, optional): Path to save visualization meshes
        
    Returns:
        tuple: (chamfer_distance, iou)
    """
    # Convert Open3D mesh to Trimesh if needed
    if hasattr(target_mesh, 'vertices') and hasattr(target_mesh, 'triangles'):
        # This is likely an Open3D mesh, convert to trimesh
        try:
            import open3d as o3d
            if isinstance(target_mesh, o3d.geometry.TriangleMesh):
                # Convert Open3D mesh to trimesh
                vertices = np.asarray(target_mesh.vertices)
                faces = np.asarray(target_mesh.triangles)
                target_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        except Exception as e:
            print(f"Warning: Could not convert mesh type: {e}")
    
    # Sample points from target mesh surface for Chamfer distance
    gt_points, _ = trimesh.sample.sample_surface(target_mesh, 2048)
    
    # Initialize mesh evaluator
    evaluator = MeshEvaluator(n_points=2048)
    
    # Convert GT points to normalized space
    gt_points = (gt_points / 0.3) - 0.5
    
    # Compute Chamfer-L1 using original pointcloud method
    eval_dict = evaluator.eval_pointcloud(
        pointcloud=completed_pc,
        pointcloud_tgt=gt_points,
        normals=None,
        normals_tgt=None
    )
    chamfer_distance = eval_dict['chamfer-L1']
    
    # Convert completed point cloud to mesh using alpha shape
    completed_mesh = point_cloud_to_mesh(completed_pc, alpha=0.5)
    
    # Normalize target mesh vertices to [-0.5, 0.5] space
    target_mesh.vertices = target_mesh.vertices / 0.3 - 0.5
    
    # Save meshes if path is provided
    if mesh_path is not None:
        mesh_path = Path(mesh_path)
        # Create directory if it doesn't exist
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save completed mesh
        completed_mesh_path = mesh_path / f"{mesh_path.stem}_completed.ply"
        completed_mesh.export(str(completed_mesh_path))
        
        # Save target mesh
        target_mesh_path = mesh_path / f"{mesh_path.stem}_target.ply"
        target_mesh.export(str(target_mesh_path))
    
    # target_mesh.vertices = target_mesh.vertices / 0.3 - 0.5
    # Sample points for IoU calculation using uniform sampling
    points_iou, occ_target = sample_iou_points(
        mesh_list=[target_mesh],  # Single target mesh
        bounds=target_mesh.bounds,  # Use mesh bounds
        num_point=100000,  # Number of sample points
        # padding=0.02,  # Add padding around bounds
        uniform=False,  # Use uniform sampling
        size=1.0  # Size=1.0 as mesh is normalized to [-0.5, 0.5]
    )
    
    # completed_mesh.vertices = (completed_mesh.vertices + 0.5) * 0.3 
    # Compute IoU using eval_mesh
    metrics = evaluator.eval_mesh(
        mesh=completed_mesh,
        pointcloud_tgt=gt_points,
        normals_tgt=None,
        points_iou=points_iou,
        occ_tgt=occ_target,
        remove_wall=False
    )
    iou = metrics['iou']
    return chamfer_distance, iou

def trimesh_to_open3d(mesh):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.faces)
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    if hasattr(mesh.visual, 'face_colors'):
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(mesh.visual.face_colors[:, :3] / 255.0)
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh

def combine_meshes(mesh_list):
    combined_mesh = o3d.geometry.TriangleMesh()
    for mesh in mesh_list:
        combined_mesh += mesh
    return combined_mesh

def save_numpy_as_ply(points, ply_file_path):
    """
    Save a numpy array as a PLY file using Open3D.
    
    :param points: A numpy array of shape (N, 3) representing the point cloud.
    :param ply_file_path: The file path to save the PLY file.
    """
    # Convert numpy array to Open3D PointCloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    # Save point cloud to a PLY file
    o3d.io.write_point_cloud(ply_file_path, point_cloud)

def generate_and_transform_grasp_meshes(grasp, pc, save_dir):
    # Convert grasp into a mesh
    grasp_mesh = visual.grasp2mesh_no_score(grasp)  # Assuming grasp2mesh doesn't require a score now

    # Define the transformation function
    def apply_custom_transformation(mesh):
        # Extract the vertices from the mesh
        vertices = np.asarray(mesh.vertices)
        # Apply the transformation
        transformed_vertices = (vertices + 0.5) * 0.3 - np.array([0, 0, 0.02])
        # Update the mesh with transformed vertices
        mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
        return mesh

    # Apply the transformation to the grasp mesh
    o3d_mesh = trimesh_to_open3d(grasp_mesh)  # Assuming trimesh_to_open3d is already defined
    transformed_mesh = apply_custom_transformation(o3d_mesh)

    # Apply color based on a fixed value (since score is no longer used)
    green_intensity = 1.0  # You can adjust this value as needed for the color intensity
    color = [1.0 - green_intensity, 1.0, 1.0 - green_intensity]  # Color range from white to green
    colors = np.tile(color, (len(transformed_mesh.vertices), 1))
    transformed_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # Combine the transformed grasp mesh into a list (even though it's a single mesh for consistency)
    combined_grasp_mesh = combine_meshes([transformed_mesh])

    # Save the scene point cloud and combined grasp mesh
    if not os.path.exists(os.path.join(save_dir, 'icgnet_grasp')):
        os.makedirs(os.path.join(save_dir, 'icgnet_grasp'))
    scene_pc_path = os.path.join(save_dir, 'icgnet_grasp/pc_targo.ply')
    # o3d.io.write_point_cloud(scene_pc_path, pc)
    save_numpy_as_ply(np.asarray(pc), scene_pc_path)
    # print(f"scene_pc is at {scene_pc_path}")

    grasp_path = os.path.join(save_dir, "icgnet_grasp/grasp_targo.ply")
    o3d.io.write_triangle_mesh(grasp_path, combined_grasp_mesh)
    # print(f"grasps are at {grasp_path}")

          
def filter_and_pad_point_clouds_dexycb(points, lower_bound=torch.tensor([0.02, 0.02, 0.055]) , upper_bound=torch.tensor([0.68, 0.68, 0.7]), N=2048):
    """
    Filter each point cloud within a bounding box and pad or truncate to N points.
    
    Args:
        points (torch.Tensor): The point cloud data, shape (BS, N, 3).
        lower_bound (torch.Tensor): Lower limit of the bounding box, shape (3,).
        upper_bound (torch.Tensor): Upper limit of the bounding box, shape (3,).
        N (int): The desired number of points in each point cloud, default is 2048.
    
    Returns:
        torch.Tensor: The processed point cloud data, shape (BS, N, 3). Ensures the processed points are on the same device as the input points.
    """
    BS, _, _ = points.shape
    lower_bound = lower_bound.to(points.device)
    upper_bound = upper_bound.to(points.device)
    processed_points = torch.zeros((BS, N, 3), dtype=points.dtype, device=points.device)  # Ensure the tensor is on the same device as the input points.

    for i in range(BS):
        # Extract a single point cloud
        single_point_cloud = points[i]

        # Apply bounding box filter
        mask = (single_point_cloud >= lower_bound) & (single_point_cloud <= upper_bound)
        mask = mask.all(dim=1)
        filtered_points = single_point_cloud[mask]

        # Pad or truncate to match N points
        filtered_len = filtered_points.size(0)
        if filtered_len < N:
            # If there are fewer points than N after filtering, pad with the last point
            if filtered_len > 0:
                padding = filtered_points[-1].unsqueeze(0).repeat(N - filtered_len, 1)
                processed_points[i] = torch.cat([filtered_points, padding], dim=0)
            else:
                # If no points are left after filtering, you can choose to keep an empty point cloud
                # or fill with zeros. Here, we choose to keep it as all zeros.
                continue
        else:
            # If there are more points than N, truncate
            processed_points[i] = filtered_points[:N]
    
    return processed_points

def filter_and_pad_point_clouds(points, lower_bound=torch.tensor([0.02, 0.02, 0.055]) / 0.3 - 0.5, upper_bound=torch.tensor([0.28, 0.28, 0.3])/ 0.3 - 0.5, N=2048):
    """
    Filter each point cloud within a bounding box and pad or truncate to N points.
    
    Args:
        points (torch.Tensor): The point cloud data, shape (BS, N, 3).
        lower_bound (torch.Tensor): Lower limit of the bounding box, shape (3,).
        upper_bound (torch.Tensor): Upper limit of the bounding box, shape (3,).
        N (int): The desired number of points in each point cloud, default is 2048.
    
    Returns:
        torch.Tensor: The processed point cloud data, shape (BS, N, 3). Ensures the processed points are on the same device as the input points.
    """
    BS, _, _ = points.shape
    lower_bound = lower_bound.to(points.device)
    upper_bound = upper_bound.to(points.device)
    processed_points = torch.zeros((BS, N, 3), dtype=points.dtype, device=points.device)  # Ensure the tensor is on the same device as the input points.

    for i in range(BS):
        # Extract a single point cloud
        single_point_cloud = points[i]

        # Apply bounding box filter
        mask = (single_point_cloud >= lower_bound) & (single_point_cloud <= upper_bound)
        mask = mask.all(dim=1)
        filtered_points = single_point_cloud[mask]

        # Pad or truncate to match N points
        filtered_len = filtered_points.size(0)
        if filtered_len < N:
            # If there are fewer points than N after filtering, pad with the last point
            if filtered_len > 0:
                padding = filtered_points[-1].unsqueeze(0).repeat(N - filtered_len, 1)
                processed_points[i] = torch.cat([filtered_points, padding], dim=0)
            else:
                # If no points are left after filtering, you can choose to keep an empty point cloud
                # or fill with zeros. Here, we choose to keep it as all zeros.
                continue
        else:
            # If there are more points than N, truncate
            processed_points[i] = filtered_points[:N]
    
    return processed_points


def points_within_boundary(points):
    lower_bound = np.array([0.02, 0.02, 0.055])
    upper_bound = np.array([0.28, 0.28, 0.30000000000000004])
    # po
    within_bounding_box = np.all((points >= lower_bound) & (points <= upper_bound), axis=1)
    points = points[within_bounding_box]
    return points

def depth_to_point_cloud(depth_img, mask_targ, intrinsics, extrinsics, num_points):
    """
    Convert a masked and scaled depth image into a point cloud using camera intrinsics and inverse extrinsics.

    Parameters:
    - depth_img: A 2D numpy array containing depth for each pixel.
    - mask_targ: A 2D boolean numpy array where True indicates the target.
    - intrinsics: The camera intrinsic matrix as a 3x3 numpy array.
    - extrinsics: The camera extrinsic matrix as a 4x4 numpy array. This function assumes the matrix is to be inversed for the transformation.
    - scale: Scale factor to apply to the depth values.

    Returns:
    - A numpy array of shape (N, 3) containing the X, Y, Z coordinates of the points in the world coordinate system.
    """

    # Apply the target mask to the depth image, then apply the scale factor
    depth_img_masked_scaled = depth_img * mask_targ
    
    # Get the dimensions of the depth image
    height, width = depth_img_masked_scaled.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    # Flatten the arrays for vectorized operations
    u, v = u.flatten(), v.flatten()
    z = depth_img_masked_scaled.flatten()

    # Convert pixel coordinates (u, v) and depth (z) to camera coordinates
    x = (u - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (v - intrinsics[1, 2]) * z / intrinsics[1, 1]
    
    # Create normal coordinates in the camera frame
    # points_camera_frame = np.array([x, y, z]).T
    points_camera_frame = np.vstack((x, y, z)).T
    points_camera_frame = points_camera_frame[z!=0]
    # Convert the camera coordinates to world coordinate
    # points_camera_frame = specify_num_points(points_camera_frame, num_points)

    extrinsic = Transform.from_list(extrinsics).inverse()
    points_transformed = np.array([extrinsic.transform_point(p) for p in points_camera_frame])


    lower_bound = np.array([0.02, 0.02, 0.055])
    upper_bound = np.array([0.28, 0.28, 0.30000000000000004])
    # po
    within_bounding_box = np.all((points_transformed >= lower_bound) & (points_transformed <= upper_bound), axis=1)
    points_transformed = points_transformed[within_bounding_box]

    points_transformed = specify_num_points(points_transformed, num_points)
    
    return points_transformed


# transform = Compose([{
#         'callback': 'UpSamplePoints',
#         'parameters': {
#             'n_points': 2048
#         },
#         'objects': ['input']
#     }, {
#         'callback': 'ToTensor',
#         'objects': ['input']
#     }])

def transform_pc(pc):
    # device = pc.device
    # pc = pc.cpu().numpy()
    # BS = pc.shape[0]
    # pc_transformed = torch.zeros((BS, 2048, 3), dtype=torch.float32)
    # for i in range(BS):
    points_curr_transformed = transform({'input':pc})
        # pc_transformed[i] = points_curr_transformed['input']
    return points_curr_transformed['input']


def record_occ_level_count(occ_level, occ_level_count_dict):
    if 0 <= occ_level < 0.1:
        occ_level_count_dict['0-0.1'] += 1
    elif 0.1 <= occ_level < 0.2:
        occ_level_count_dict['0.1-0.2'] += 1
    elif 0.2 <= occ_level < 0.3:
        occ_level_count_dict['0.2-0.3'] += 1
    elif 0.3 <= occ_level < 0.4:
        occ_level_count_dict['0.3-0.4'] += 1
    elif 0.4 <= occ_level < 0.5:
        occ_level_count_dict['0.4-0.5'] += 1
    elif 0.5 <= occ_level < 0.6:
        occ_level_count_dict['0.5-0.6'] += 1
    elif 0.6 <= occ_level < 0.7:
        occ_level_count_dict['0.6-0.7'] += 1
    elif 0.7 <= occ_level < 0.8:
        occ_level_count_dict['0.7-0.8'] += 1
    elif 0.8 <= occ_level < 0.9:
        occ_level_count_dict['0.8-0.9'] += 1
    return occ_level_count_dict

def record_occ_level_success(occ_level, occ_level_success_dict):
    if 0 <= occ_level < 0.1:
        occ_level_success_dict['0-0.1'] += 1
    elif 0.1 <= occ_level < 0.2:
        occ_level_success_dict['0.1-0.2'] += 1
    elif 0.2 <= occ_level < 0.3:
        occ_level_success_dict['0.2-0.3'] += 1
    elif 0.3 <= occ_level < 0.4:
        occ_level_success_dict['0.3-0.4'] += 1
    elif 0.4 <= occ_level < 0.5:
        occ_level_success_dict['0.4-0.5'] += 1
    elif 0.5 <= occ_level < 0.6:
        occ_level_success_dict['0.5-0.6'] += 1
    elif 0.6 <= occ_level < 0.7:
        occ_level_success_dict['0.6-0.7'] += 1
    elif 0.7 <= occ_level < 0.8:
        occ_level_success_dict['0.7-0.8'] += 1
    elif 0.8 <= occ_level < 0.9:
        occ_level_success_dict['0.8-0.9'] += 1
    return occ_level_success_dict

def cal_occ_level_sr(occ_level_count_dict, occ_level_success_dict):
    occ_level_sr_dict = {}
    for key in occ_level_count_dict:
        if occ_level_count_dict[key] == 0:
            occ_level_sr_dict[key] = 0
        else:
            occ_level_sr_dict[key] = occ_level_success_dict[key] / occ_level_count_dict[key]
    return occ_level_sr_dict


def pointcloud_to_voxel_indices(points, length, resolution):
    """
    Convert point cloud coordinates to voxel indices within a TSDF volume.

    Parameters:
    - points: torch.Tensor of shape (BS, N, 3) containing point cloud coordinates.
    - length: float, physical side length of the TSDF volume.
    - resolution: int, number of voxels along each dimension of the volume.

    Returns:
    - voxel_indices: torch.Tensor of shape (BS, N, 3) containing voxel indices.
    """
    # Normalize points to [0, 1] based on the TSDF volume's length
    normalized_points = points / length
    
    # Scale normalized points to voxel grid
    scaled_points = normalized_points * resolution
    
    # Convert to integer indices (floor)
    voxel_indices = torch.floor(scaled_points).int()
    
    # Ensure indices are within the bounds of the voxel grid
    voxel_indices = torch.clamp(voxel_indices, 0, resolution - 1)

    return voxel_indices

def points_to_voxel_grid_batch(points, lower_bound, upper_bound, resolution=40):
    """
    Convert a batch of point clouds (BS, N, 3) to a batch of voxel grids (BS, resolution, resolution, resolution).
    Each point cloud in the batch is converted to a voxel grid where occupied voxels are marked as 1.

    Parameters:
    - points: torch tensor of shape (BS, N, 3) containing the batch of point clouds.
    - lower_bound: list or numpy array with 3 elements indicating the lower spatial bound of the point clouds.
    - upper_bound: list or numpy array with 3 elements indicating the upper spatial bound of the point clouds.
    - resolution: int, the resolution of each side of the voxel grid.

    Returns:
    - voxel_grids: torch tensor of shape (BS, resolution, resolution, resolution) representing the batch of voxel grids.
    """
    BS, N, _ = points.shape
    device = points.device

    # Convert bounds to tensors
    lower_bound = torch.tensor(lower_bound, dtype=torch.float32, device=device)
    upper_bound = torch.tensor(upper_bound, dtype=torch.float32, device=device)

    # Calculate the size of each voxel
    voxel_size = (upper_bound - lower_bound) / resolution

    # Normalize points within the bounds
    normalized_points = (points - lower_bound) / (upper_bound - lower_bound)

    # Compute voxel indices
    voxel_indices = (normalized_points * resolution).long()

    # Clamp indices to be within the grid
    voxel_indices = torch.clamp(voxel_indices, 0, resolution - 1)

    # Initialize an empty voxel grid for each point cloud in the batch
    voxel_grids = torch.zeros(BS, resolution, resolution, resolution, dtype=torch.uint8, device=device)

    # Convert voxel indices to linear indices
    linear_indices = voxel_indices[:, :, 0] * resolution**2 + voxel_indices[:, :, 1] * resolution + voxel_indices[:, :, 2]
    linear_indices = linear_indices + torch.arange(BS, device=device).view(BS, 1) * (resolution**3)

    # Flatten voxel grids to use linear indices directly
    voxel_grids_flat = voxel_grids.view(-1)

    # Mark voxels as occupied
    voxel_grids_flat[linear_indices.view(-1)] = 1

    # Reshape to original grid shape
    voxel_grids = voxel_grids_flat.view(BS, resolution, resolution, resolution)

    return voxel_grids

def concat_sparse_tensors(sparse_tensor1, sparse_tensor2):
    """
    Concatenates two SparseTensors along the spatial dimension.

    Args:
    sparse_tensor1 (ME.SparseTensor): The first SparseTensor.
    sparse_tensor2 (ME.SparseTensor): The second SparseTensor.

    Returns:
    ME.SparseTensor: A new SparseTensor containing the concatenated data.
    """

    # Concatenate coordinates and features
    coords1 = sparse_tensor1.C
    coords2 = sparse_tensor2.C
    feats1 = sparse_tensor1.F
    feats2 = sparse_tensor2.F

    combined_coords = torch.cat([coords1, coords2], dim=0)
    combined_feats = torch.cat([feats1, feats2], dim=0)

    # Create a new SparseTensor using the combined coordinates and features
    concatenated_tensor = ME.SparseTensor(features=combined_feats, coordinates=combined_coords)

    return concatenated_tensor

def assert_no_intersection(sparse_tensor1, sparse_tensor2):
    """
    Assert that there is no intersection in the coordinates of two SparseTensors.

    Args:
    sparse_tensor1 (ME.SparseTensor): The first SparseTensor.
    sparse_tensor2 (ME.SparseTensor): The second SparseTensor.

    Raises:
    AssertionError: If there is an intersection in the coordinates.
    """

    # Get coordinates of the SparseTensors
    coords1 = sparse_tensor1.C
    coords2 = sparse_tensor2.C

    # Convert coordinates to sets of tuples
    set_coords1 = set(map(tuple, coords1.tolist()))
    set_coords2 = set(map(tuple, coords2.tolist()))

    # Assert no intersection
    assert set_coords1.isdisjoint(set_coords2), "Coordinates have an intersection"


def pad_to_target(tensor, target_dims):
    """
    Pads a tensor to the target dimensions.

    Parameters:
    tensor (torch.Tensor): The input tensor to be padded.
    target_dims (tuple): A tuple of the target dimensions (BS, Channels, X, Y, Z).

    Returns:
    torch.Tensor: The padded tensor.
    """

    # Get the current dimensions of the tensor
    current_dims = tensor.shape

    # Calculate the padding required for each dimension
    padding = []
    for curr_dim, target_dim in zip(reversed(current_dims), reversed(target_dims)):
        total_pad = target_dim - curr_dim
        pad_one_side = total_pad // 2
        padding.extend([pad_one_side, total_pad - pad_one_side])

    # Apply padding
    padded_tensor = F.pad(tensor, padding)

    return padded_tensor


def pad_sequence(sequences, require_padding_mask=False, require_lens=False,
                 batch_first=False):
    """List of sequences to padded sequences

    Args:
        sequences: List of sequences (N, D)
        require_padding_mask:

    Returns:
        (padded_sequence, padding_mask), where
           padded sequence has shape (N_max, B, D)
           padding_mask will be none if require_padding_mask is False
    """
    padded = nn.utils.rnn.pad_sequence(sequences, batch_first=batch_first)
    padding_mask = None
    padding_lens = None

    if require_padding_mask:
        B = len(sequences)
        seq_lens = list(map(len, sequences))
        padding_mask = torch.zeros((B, padded.shape[0]), dtype=torch.bool, device=padded.device)
        for i, l in enumerate(seq_lens):
            padding_mask[i, l:] = True

    if require_lens:
        padding_lens = [seq.shape[0] for seq in sequences]

    return padded, padding_mask, padding_lens


def unpad_sequences(padded, seq_lens):
    """Reverse of pad_sequence"""
    sequences = [padded[..., :seq_lens[b], b, :] for b in range(len(seq_lens))]
    return sequences


def save_scene_as_ply(scene, file_path):
    """
    Save a trimesh.Scene object as a PLY file.

    Parameters:
    scene (trimesh.Scene): The trimesh scene to save.
    file_path (str): The file path where the PLY file will be saved.
    """
    # Export the scene as a PLY
    ply_data = scene.export(file_type='ply')
    
    # Save the PLY data to a file
    with open(file_path, 'wb') as file:
        file.write(ply_data)
    print(f"Scene saved as '{file_path}'")

def count_unique_scenes(df):
    """
    Count the number of unique clutter and single scenes in a dataframe.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the scene_id column.

    Returns:
    tuple: A tuple containing the counts of unique clutter scenes and single scenes.
    """
    # Count unique clutter scenes (contains '_c_')
    unique_clutter_scenes = df[df['scene_id'].str.contains('_c_')]['scene_id'].nunique()

    # Count unique single scenes (contains '_s_')
    unique_single_scenes = df[df['scene_id'].str.contains('_s_')]['scene_id'].nunique()

    print(f"Unique clutter scenes: {unique_clutter_scenes}")
    print(f"Unique single scenes: {unique_single_scenes}")


    # return unique_clutter_scenes, unique_single_scenes

def remove_target_from_scene(scene_pc, target_pc):
    """
    Removes target_pc from scene_pc.

    Parameters:
    scene_pc (numpy.ndarray): Scene point cloud array of shape (N1, 3).
    target_pc (numpy.ndarray): Target point cloud array of shape (N2, 3).

    Returns:
    numpy.ndarray: Scene point cloud excluding target point cloud.
    """
    # Convert point cloud arrays to strings for comparison
    scene_pc_str = np.array([str(row) for row in scene_pc])
    target_pc_str = np.array([str(row) for row in target_pc])

    # Find indices of scene_pc that are not in target_pc
    result_indices = np.setdiff1d(np.arange(len(scene_pc_str)), np.where(np.isin(scene_pc_str, target_pc_str)))

    # Get the result using the indices
    result = scene_pc[result_indices]

    return result

def specify_num_points(points, target_size):
    # add redundant points if less than target_size
    # if points.shape[0] == 0:
    if points.size == 0:
        print("No points in the scene")
    if points.shape[0] < target_size:
        points_specified_num = duplicate_points(points, target_size)
    # sample farthest points if more than target_size
    elif points.shape[0] > target_size:
        points_specified_num = farthest_point_sampling(points, target_size)
    else:
        points_specified_num = points
    return points_specified_num

def duplicate_points(points, target_size):
    repeated_points = points
    while len(repeated_points) < target_size:
        additional_points = points[:min(len(points), target_size - len(repeated_points))]
        repeated_points = np.vstack((repeated_points, additional_points))
    return repeated_points

def farthest_point_sampling(points, num_samples):
    # Initial farthest point randomly selected
    farthest_pts = np.zeros((num_samples, 3))
    farthest_pts[0] = points[np.random.randint(len(points))]
    distances = np.full(len(points), np.inf)
    
    for i in range(1, num_samples):
        dist = np.sum((points - farthest_pts[i - 1])**2, axis=1)
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest_pts[i] = points[np.argmax(distances)]
    
    return farthest_pts


def print_and_count_patterns(df, unique_id = False):
    # Define the patterns
    pattern_c = r'.*_c_\d'
    pattern_s = r'.*_s_\d'
    pattern_d = r'.*_d_\d_\d'

    # Count the number of rows that match each pattern
    count_c = df['scene_id'].str.contains(pattern_c).sum()
    count_s = df['scene_id'].str.contains(pattern_s).sum()
    count_d = df['scene_id'].str.contains(pattern_d).sum()

    print("sampled data frames stastics")

    # Extract the unique ID part and count the number of unique IDs
    if unique_id:
        unique_id_pattern = r'(.*)(?:_[csd]_\d+)$'  # Modified to capture the unique ID part
        unique_ids = df['scene_id'].str.extract(unique_id_pattern)[0]
        unique_id_count = unique_ids.nunique()
        print("Number of unique IDs: ", unique_id_count)

    # print("sampled data frames stastics")

    # Print the counts with informative messages
    print("Number of cluttered scenes: ", count_c)
    print("Number of single scenes: ", count_s)
    print("Number of double scenes:", count_d)
    # print("Number of unique IDs: ", unique_id_count)


def count_and_sample(df):
    # This pattern matches strings that contain '_c_' followed by one or more digits
    pattern = r'.*_c_\d'
    
    # Count the number of rows that match the pattern
    count_matching_rows = df['scene_id'].str.contains(pattern).sum()
    
    # Randomly select the same number of rows from the dataframe
    sampled_df = df.sample(n=count_matching_rows)
    sampled_df = sampled_df.reset_index(drop=True)
    print_and_count_patterns(sampled_df)
    
    return sampled_df

def load_scene_indices(file_path):
    scene_dict = {}

    with open(file_path, 'r') as file:
        for line in file:
            key, value_str = line.strip().split(': ')
            # Convert the string representation of the list into a NumPy array
            values = np.fromstring(value_str.strip('[]'), sep=',', dtype=int)
            scene_dict[key] = values

    return scene_dict

def filter_rows_by_id_only_clutter(df):
    # This pattern matches strings that end with '_c_' followed by one or more digits
    pattern = r".*_c_\d+$"
    # Apply the regex pattern to filter the dataframe
    filtered_df = df[df['scene_id'].str.contains(pattern, regex=True)]
    # Reset the index of the filtered dataframe and drop the old index
    filtered_df = filtered_df.reset_index(drop=True)
    return filtered_df

def filter_rows_by_id_only_single(df):
    # This pattern matches strings that end with '_c_' followed by one or more digits
    pattern = r".*_s_\d+$"
    # Apply the regex pattern to filter the dataframe
    filtered_df = df[df['scene_id'].str.contains(pattern, regex=True)]
    # Reset the index of the filtered dataframe and drop the old index
    filtered_df = filtered_df.reset_index(drop=True)
    return filtered_df

def filter_rows_by_id_only_single_and_double(df):
    # This pattern matches strings that end with '_c_' followed by one or more digits
    pattern = r'.*_s_\d+$|.*_d_\d+_\d+$'
    # Apply the regex pattern to filter the dataframe
    filtered_df = df[df['scene_id'].str.contains(pattern, regex=True)]
    # Reset the index of the filtered dataframe and drop the old index
    filtered_df = filtered_df.reset_index(drop=True)
    return filtered_df

def filter_rows_by_id_only_clutter_and_double(df):
    # This pattern matches strings that end with '_c_' followed by one or more digits
    pattern = r'.*_c_\d+$|.*_d_\d+_\d+$'
    # Apply the regex pattern to filter the dataframe
    filtered_df = df[df['scene_id'].str.contains(pattern, regex=True)]
    # Reset the index of the filtered dataframe and drop the old index
    filtered_df = filtered_df.reset_index(drop=True)
    return filtered_df

import matplotlib.pyplot as plt
import numpy as np

def visualize_and_save_tsdf(tsdf_data, file_path, edge_color='k', point=None, point_color='r'):
    """
    Visualize a TSDF grid and save the visualization to a file without displaying a GUI. 
    Highlight a specific point in the grid if provided.

    Parameters:
    - tsdf_data: 3D numpy array representing the TSDF grid.
    - file_path: String, path to save the visualization.
    - edge_color: String, color of the edges in the plot.
    - point: Tuple of ints, the (x, y, z) coordinates of the point to highlight.
    - point_color: String, color of the highlighted point.
    """
    # Create a figure for plotting
    fig = plt.figure()

    # Add a 3D subplot
    ax = fig.add_subplot(111, projection='3d')

    # Visualize the data
    ax.voxels(tsdf_data, edgecolor=edge_color)

    # Highlight the specified point if given
    if point is not None:
        # Extract the point coordinates
        x, y, z = point
        # Create a small cube to represent the point
        point_data = np.zeros_like(tsdf_data, dtype=bool)
        point_data[x, y, z] = True
        # Visualize the point
        ax.voxels(point_data, facecolors=point_color, edgecolor='none')

    # Save the plot
    plt.savefig(file_path)


# def visualize_and_save_tsdf(tsdf_data, file_path, edge_color='k', show_plot=False):
#     """
#     Visualize a TSDF grid and save the visualization to a file.

#     Parameters:
#     - tsdf_data: 3D numpy array representing the TSDF grid.
#     - file_path: String, path to save the visualization.
#     - edge_color: String, color of the edges in the plot.
#     - show_plot: Boolean, if True, the plot will be displayed.
#     """
#     # Create a figure for plotting
#     fig = plt.figure()

#     # Add a 3D subplot
#     ax = fig.add_subplot(111, projection='3d')

#     # Visualize the data
#     ax.voxels(tsdf_data, edgecolor=edge_color)

#     # Save the plot
#     plt.savefig(file_path)

#     # Show the plot if requested
#     if show_plot:
#         plt.show()
        
# Function: find_urdf
def find_urdf(file_path):
    base_dir = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    for urdf_path in os.listdir(base_dir):
        if filename in urdf_path:
            return os.path.join(base_dir, urdf_path)

# Function: collect_mesh_pose_list
def collect_mesh_pose_list(sim, exclude_plane=False):
    mesh_pose_list = []
    for uid in sim.world.bodies.keys():
        _, name = sim.world.p.getBodyInfo(uid)
        name = name.decode('utf8')
        if name == 'plane' and exclude_plane:
            continue
        body = sim.world.bodies[uid]
        pose = body.get_pose().as_matrix()
        # scale = body.scale1
        visuals = sim.world.p.getVisualShapeData(uid)
        assert len(visuals) == 1
        _, _, _, scale, mesh_path, _, _, _ = visuals[0]
        mesh_path = mesh_path.decode('utf8')
        if mesh_path == '':
            mesh_path = os.path.join('data/urdfs/pile/train', name + '.urdf')
        mesh_pose_list.append((mesh_path, scale, pose))
    return mesh_pose_list

# # Function: sim_select_single_scene
# def sim_select_single_scene(sim, indices):
#     # urdf_root = sim.urdf_root
#     scene = sim.scene
#     object_set = sim.object_set
#     # size = sim.size
#     # sim_selected = ClutterRemovalSim(urdf_root, size,scene, object_set, gui=False)  ## create a new sim
#     sim_selected = ClutterRemovalSim(scene, object_set, False)
#     sim.urdf_root = Path("data/urdfs")
#     # sim_selected = ClutterRemovalSim(sim.urdf_root, sim.size, sim.scene, sim.object_set, gui=sim.gui)  ## create a new sim
    
#     # set some attributes
#     # sim_selected.gui = False
#     sim_selected.add_noise = sim.add_noise
#     sim_selected.sideview = sim.sideview
#     sim_selected.size = sim.size
#     # sim_selected.intrinsic = sim.intrinsic
#     intrinsics = CameraIntrinsic(640, 480, 540.0, 540.0, 320.0, 240.0)
#     sim_selected.camera = sim_selected.world.add_camera(intrinsics, 0.1, 2.0)
    
#     mesh_pose_list = collect_mesh_pose_list(sim) 
#     for idc in indices:
#         pose = Transform.from_matrix(mesh_pose_list[idc][2])
#         if idc == 0:
#             mesh_path = mesh_pose_list[idc][0].replace(".obj",".urdf")
#         else:
#             mesh_path = find_urdf(mesh_pose_list[idc][0].replace("_visual.obj",".urdf"))
#         sim_selected.world.load_urdf(mesh_path, pose, mesh_pose_list[idc][1][0])
#     return sim_selected
    
    
#     # for idc in indices:
#     #     # if idc == 0:
#     #     pose = Transform.from_matrix(obj_info[idc][2])
#     #     if idc == 0:
#     #         sim_selected.world.load_urdf(obj_info[idc][0].replace(".obj",".urdf"), pose, 0.6)
#     #     else:
#     #         sim_selected.world.load_urdf(find_urdf(obj_info[idc][0].replace(".obj",".urdf").replace('meshes_centered','acronym_urdfs_centered')), pose, 1)
#     # return sim_selected

import open3d as o3d
import numpy as np
import torch

def points_equal(A, B, epsilon=1e-10):
    return abs(A[0] - B[0]) < epsilon and abs(A[1] - B[1]) < epsilon


import open3d as o3d
import numpy as np

# def point_cloud_to_tsdf(points, voxel_size=0.3/40, depth=9, num_points=10000):
def alpha_shape_mesh_reconstruct(np_points, alpha=0.5, mesh_fix=False, visualize=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_points)
    
    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha, tetra_mesh, pt_map
    )

    if mesh_fix:
        mf = pymeshfix.MeshFix(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
        mf.repair()

        mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(mf.mesh[0]), 
                                         triangles=o3d.utility.Vector3iVector(mf.mesh[1]))

    if visualize:
        if mesh_fix:
            plt = pv.Plotter()
            point_cloud = pv.PolyData(np_points)
            plt.add_mesh(point_cloud, color="k", point_size=10)
            plt.add_mesh(mesh)
            plt.add_title("Alpha Shape Reconstruction")
            plt.show()
        else:
            o3d.visualization.draw_geometries([pcd, mesh], title="Alpha Shape Reconstruction")

    return mesh

def point_cloud_to_tsdf_dexycb(points, res):
    ## points -> mesh -> sdf -> tsdf
    ## if the points is pytorch tensor, convert it to numpy
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if len(points.shape) == 3:
        points = points.reshape(-1, 3)
    mesh = alpha_shape_mesh_reconstruct(points, alpha=0.5, mesh_fix=False, visualize=False)
    x, y, z = torch.meshgrid(torch.linspace(start=0, end=0.7 - 0.7 / res, steps=res), torch.linspace(start=0, end=0.7 - 0.7 / res, steps=res), torch.linspace(start=0, end=0.7 - 0.7 / res, steps=res))
    pos = torch.stack((x, y, z), dim=-1).float() # (1, 40, 40, 40, 3)
    pos = pos.view(-1, 3)
    f = SDF(mesh.vertices, mesh.triangles)
    sdf = f(pos)
    sdf_reshaped = sdf.reshape(res, res, res)
    sdf_trunc = 4 * (0.7/res)

    mask = (sdf_reshaped >= sdf_trunc) | (sdf_reshaped <= -sdf_trunc) 

    tsdf = (sdf_reshaped / sdf_trunc + 1) / 2
    # tsdf = tsdf[mask]
    tsdf[mask] = 0
    return tsdf

def point_cloud_to_tsdf(points):
    ## points -> mesh -> sdf -> tsdf
    ## if the points is pytorch tensor, convert it to numpy
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if len(points.shape) == 3:
        points = points.reshape(-1, 3)
    mesh = alpha_shape_mesh_reconstruct(points, alpha=0.5, mesh_fix=False, visualize=False)
    x, y, z = torch.meshgrid(torch.linspace(start=0, end=0.3 - 0.3 / 40, steps=40), torch.linspace(start=0, end=0.3 - 0.3 / 40, steps=40), torch.linspace(start=0, end=0.3 - 0.3 / 40, steps=40))
    pos = torch.stack((x, y, z), dim=-1).float() # (1, 40, 40, 40, 3)
    pos = pos.view(-1, 3)
    f = SDF(mesh.vertices, mesh.triangles)
    sdf = f(pos)
    sdf_reshaped = sdf.reshape(40, 40, 40)
    sdf_trunc = 4 * (0.3/40)

    mask = (sdf_reshaped >= sdf_trunc) | (sdf_reshaped <= -sdf_trunc) 

    tsdf = (sdf_reshaped / sdf_trunc + 1) / 2
    # tsdf = tsdf[mask]
    tsdf[mask] = 0
    return tsdf
    # return tsdf_volume


def point_cloud_to_tsdf_v2(points):
    """
    Convert point cloud to TSDF, working in [-0.5, 0.5] range.
    
    Args:
        points (np.ndarray or torch.Tensor): Point cloud in [-0.5, 0.5] range
        
    Returns:
        torch.Tensor: TSDF values in a 40x40x40 grid
    """
    # Convert to numpy if needed
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if len(points.shape) == 3:
        points = points.reshape(-1, 3)
        
    # Create mesh from normalized points
    mesh = alpha_shape_mesh_reconstruct(points, alpha=0.5, mesh_fix=False, visualize=False)
    
    # Create grid in [-0.5, 0.5] range
    x, y, z = torch.meshgrid(
        torch.linspace(-0.5, 0.5, steps=40),
        torch.linspace(-0.5, 0.5, steps=40),
        torch.linspace(-0.5, 0.5, steps=40)
    )
    pos = torch.stack((x, y, z), dim=-1).float()  # (40, 40, 40, 3)
    pos = pos.view(-1, 3)
    
    # Compute SDF
    f = SDF(mesh.vertices, mesh.triangles)
    sdf = f(pos)
    sdf_reshaped = sdf.reshape(40, 40, 40)
    
    # Compute truncation distance (scaled to new range)
    sdf_trunc = 4 * (1.0/40)  # Adjusted for [-0.5, 0.5] range
    
    # Create mask for truncation
    mask = (sdf_reshaped >= sdf_trunc) | (sdf_reshaped <= -sdf_trunc)
    
    # Convert to TSDF
    tsdf = (sdf_reshaped / sdf_trunc + 1) / 2
    tsdf[mask] = 0
    
    return tsdf

# Example usage
# pcd = o3d.io.read_point_cloud("your_point_cloud.ply")
# tsdf_volume = point_cloud_to_tsdf(pcd)

def point_cloud_to_tsdf_dexycb(points):
    ## points -> mesh -> sdf -> tsdf
    ## if the points is pytorch tensor, convert it to numpy
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if len(points.shape) == 3:
        points = points.reshape(-1, 3)
    mesh = alpha_shape_mesh_reconstruct(points, alpha=0.5, mesh_fix=False, visualize=False)
    x, y, z = torch.meshgrid(torch.linspace(start=0, end=0.7 - 0.7 / 40, steps=40), torch.linspace(start=0, end=0.7 - 0.7 / 40, steps=40), torch.linspace(start=0, end=0.7 - 0.7 / 40, steps=40))
    pos = torch.stack((x, y, z), dim=-1).float() # (1, 40, 40, 40, 3)
    pos = pos.view(-1, 3)
    f = SDF(mesh.vertices, mesh.triangles)
    sdf = f(pos)
    sdf_reshaped = sdf.reshape(40, 40, 40)
    sdf_trunc = 4 * (0.3/40)

    mask = (sdf_reshaped >= sdf_trunc) | (sdf_reshaped <= -sdf_trunc) 

    tsdf = (sdf_reshaped / sdf_trunc + 1) / 2
    # tsdf = tsdf[mask]
    tsdf[mask] = 0
    return tsdf

def tsdf_to_ply(tsdf_voxels, ply_filename):
    """
    Converts TSDF voxels to a PLY file, representing occupied voxels as points,
    with coordinates normalized between 0 and 1.

    Parameters:
        tsdf_voxels (numpy.ndarray): 3D array of TSDF values.
        threshold (float): Threshold to determine occupied voxels.
        ply_filename (str): Path to the output PLY file.
    """
    def write_ply(points, filename):
        with open(filename, 'w') as file:
            file.write('ply\n')
            file.write('format ascii 1.0\n')
            file.write(f'element vertex {len(points)}\n')
            file.write('property float x\n')
            file.write('property float y\n')
            file.write('property float z\n')
            file.write('end_header\n')
            for point in points:
                # point = point* 0.3 /40
                file.write(f'{point[0]} {point[1]} {point[2]}\n')

    # Identify occupied voxels
    # occupied_indices = np.argwhere(np.logical_and(np.abs(tsdf_voxels)  > 0.1, np.abs(tsdf_voxels) <= 0.5) )
    # occupied_indices = 
    occupied_indices = np.argwhere(np.abs(tsdf_voxels)  > 0.15)
    # occupied_indices = np.argwhere(np.logical_and(np.abs(tsdf_voxels)  > 0.1, np.abs(tsdf_voxels) < 0.2))

    # Normalize coordinates to 0-1
    # max_coords = np.array(tsdf_voxels.shape) - 1
    # normalized_points = occupied_indices / max_coords
    normalized_points = occupied_indices * 0.7 /40

    # normalized_points = normalized_points / 0.3  - 0.5

    # Write normalized points to PLY
    write_ply(normalized_points, ply_filename)
    return normalized_points

import open3d as o3d
    
import open3d as o3d
import numpy as np

def save_point_cloud_as_ply(point_cloud_data, ply_file_path):
    """
    Save a numpy array of 3D points as a PLY file.

    Parameters:
    point_cloud_data : numpy.ndarray
        A numpy array with shape (n, 3) where n is the number of points.
    ply_file_path : str or pathlib.Path
        The file path where the PLY file will be saved.
    """
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()

    # Assign the points to the point cloud object
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data)

    # Convert ply_file_path to string if it's a Path object
    if not isinstance(ply_file_path, str):
        ply_file_path = str(ply_file_path)

    # Save the point cloud as a PLY file
    o3d.io.write_point_cloud(ply_file_path, pcd)
    
def visualize_3d_points_and_save(point_cloud_data, path):
    """
    Visualize a numpy array of 3D points using Open3D and save the visualization
    to a specified path without displaying a GUI.

    Parameters:
    point_cloud_data : numpy.ndarray or torch.Tensor
        A numpy array or PyTorch tensor with shape (n, 3) where n is the number of points.
    path : str
        The file path to save the visualization.
    """
    # Check if the point cloud data is a PyTorch tensor and on a CUDA device
    if isinstance(point_cloud_data, torch.Tensor):
        # Move the tensor to CPU and convert it to a numpy array
        point_cloud_data = point_cloud_data.cpu().numpy()
    elif not isinstance(point_cloud_data, np.ndarray):
        # Convert other types (like lists) to numpy arrays
        point_cloud_data = np.asarray(point_cloud_data, dtype=np.float64)

    # Check the shape of the data
    if point_cloud_data.shape[1] != 3:
        raise ValueError("point_cloud_data should be of shape (n, 3)")

    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    # Assign the points to the point cloud object
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data)

    # Set the point size (adjust as needed)
    pcd.paint_uniform_color([1, 0.706, 0])  # Optional: Set a uniform color for all points

    # Set up headless rendering
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # Set visible to False for headless mode
    vis.add_geometry(pcd)

    # Render and update
    vis.poll_events()
    vis.update_renderer()

    # Capture and save the image
    image = vis.capture_screen_float_buffer(False)
    o3d.io.write_image(path, np.asarray(image)*255, quality=100)

    vis.destroy_window()

# Function: visualize_tsdf
def visualize_3d_points(point_cloud_data):
    """
    Visualize a numpy array of 3D points using Open3D.

    Parameters:
    point_cloud_data : numpy.ndarray
        A numpy array with shape (n, 3) where n is the number of points.
    """
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    
    # Assign the points to the point cloud object
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data)
    
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])
    
def visualize_tsdf(tsdf_grid, threshold = 0.5):
    """
    Visualize a 3D mesh from a TSDF grid using Marching Cubes.

    Parameters:
    tsdf_grid (numpy.ndarray): The 3D TSDF grid.
    """
    # Apply Marching Cubes to get the mesh (vertices and faces)
    verts, faces, _, _ = marching_cubes(tsdf_grid, threshold)

    # Create a new figure
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Create a 3D polygon collection
    mesh = Poly3DCollection(verts[faces])
    ax.add_collection3d(mesh)

    # Auto scale to the mesh size
    scale = verts.flatten()
    ax.auto_scale_xyz(scale, scale, scale)

    # Set plot details
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.title('3D Mesh Visualization from TSDF Grid')

    # Show the plot
    plt.show()


# Example usage:
# visualize_tsdf(tsdf_grid)


# Function: render_voxel_scene
def render_voxel_scene(voxel_grid, intrinsic_matrix, extrinsic_matrix):
    """
    Renders a scene based on a 3D numpy array representing an occupancy grid, intrinsic and extrinsic matrices.
    
    :param voxel_array: 3D numpy array representing the occupancy grid
    :param voxel_size: Size of each voxel
    :param intrinsic_matrix: Dictionary representing the intrinsic camera parameters
    :param extrinsic_matrix: 4x4 numpy array representing the camera extrinsic parameters
    :return: Open3D Image object of the rendered scene
    """
    # Extract intrinsic parameters from the dictionary
    width, height = intrinsic_matrix['width'], intrinsic_matrix['height']
    fx, fy = intrinsic_matrix['K'][0], intrinsic_matrix['K'][4]
    cx, cy = intrinsic_matrix['K'][2], intrinsic_matrix['K'][5]

    # Create camera intrinsic object
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    # Create a PinholeCameraParameters object
    camera_parameters = o3d.camera.PinholeCameraParameters()
    camera_parameters.intrinsic = intrinsic
    camera_parameters.extrinsic = extrinsic_matrix

    # Render the image
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.setup_camera(intrinsic, extrinsic_matrix)
    renderer.scene.add_geometry("voxel_grid", voxel_grid, o3d.visualization.rendering.MaterialRecord())
    # renderer.setup_camera(camera_parameters, voxel_grid.get_axis_aligned_bounding_box())
    # o3d.visualization.rendering.OffscreenRenderer.setup_camera(renderer,camera_parameters.intrinsic, extrinsic_matrix)
    image = renderer.render_to_image()
    # o3d.io.write_image('demo/rendered_image.png', image)

    return image

# Function: visualize_voxels
def visualize_voxels(voxel_array):
    """
    Visualizes a 3D voxel array.
    
    :param voxel_array: A 3D numpy array where 1 indicates the presence of a voxel.
    """
    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Prepare the voxels ('True' indicates that the voxel should be drawn)
    voxels = voxel_array == 1

    # Visualize the voxels
    ax.voxels(voxels, edgecolor='k')

    # Set labels and title if needed
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('Voxel Visualization')

    # Show the plot
    plt.show()
    
# Function: visualize_depth_map


def visualize_depth_map(depth_map, colormap=None, save_path=None):
    # Normalize the depth map to be in the range [0, 1]
    normalized_depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))

    # Explicitly create a 2D subplot
    fig, ax = plt.subplots()

    # Show the normalized depth map on 2D axes
    im = ax.imshow(normalized_depth_map, cmap=colormap, aspect='auto')  # 'auto' can be used safely in 2D
    plt.colorbar(im, ax=ax)  # Show a color bar indicating the depth scale
    ax.axis('off')  # Hide the axis

    if save_path:
        if os.path.exists(save_path):
            os.remove(save_path)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    else:
        plt.show()

# Example usage
# visualize_depth_map(your_depth_map_data)


# Function: visualize_mask
def visualize_mask(mask, colormap='jet', save_path=None):
    """
    Visualizes and optionally saves a segmentation mask.

    Parameters:
    - mask: A 2D numpy array where each element is a segmentation label.
    - colormap: The colormap to use for visualizing the segmentation. Defaults to 'jet'.
    - save_path: Path to save the image. If None, the image is displayed.
    """
    # Handle the -1 labels as transparent in the visualization
    mask = mask + 1  # Increment to make -1 become 0, which will be mapped to transparent

    # Create a color map
    cmap = plt.get_cmap(colormap)
    colors = cmap(np.linspace(0, 1, np.max(mask) + 1))  # Generate enough colors
    colors[0] = np.array([0, 0, 0, 0])  # Set the first color (for label -1) as transparent

    # Create a new ListedColormap
    new_cmap = ListedColormap(colors)

    # Show the mask
    plt.imshow(mask, cmap=new_cmap)
    plt.axis('off')  # Hide the axis

    if save_path:
        if os.path.exists(save_path):
            os.remove(save_path)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()


# Function: segmentation_to_colormap
def segmentation_to_colormap(segmentation_mask):
    """
    Converts a segmentation mask to a color map. Pixels with ID 1 are red, others are white.

    :param segmentation_mask: A 2D numpy array representing the segmentation mask.
    :return: A 3D numpy array representing the color map.
    """
    # Initialize a blank (white) color map
    height, width = segmentation_mask.shape
    colormap = np.full((height, width, 3), 0, dtype=np.uint8)

    # Set pixels with ID 1 to red
    colormap[segmentation_mask == 1] = [1,1 , 1]

    return colormap
    
# Function: mesh_pose_list2collision_manager
def mesh_pose_list2collision_manager(mesh_pose_list):
    collision_manager = trimesh.collision.CollisionManager()
    for mesh_path, scale, pose in mesh_pose_list:
        mesh = trimesh.load_mesh(mesh_path)
        mesh.apply_scale(scale)
        mesh.apply_transform(pose)
        mesh_id = os.path.splitext(os.path.basename(mesh_path))[0]
        collision_manager.add_object(name = mesh_id, mesh = mesh, transform = pose)
        
    return collision_manager

# Function: find_urdf
def find_urdf(file_path):
    base_dir = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    for urdf_path in os.listdir(base_dir):
        if filename in urdf_path:
            return os.path.join(base_dir, urdf_path)

# Function: sim_select_indices
def sim_select_indices(sim, indices, obj_info, args):
    sim_selected = ClutterRemovalSim(args.urdf_root, args.size, args.scene, args.object_set, gui=False)  ## create a new sim
    
    sim_selected.gui = False
    sim_selected.add_noise = sim.add_noise
    sim_selected.sideview = sim.sideview
    sim_selected.size = sim.size
    # sim_selected.intrinsic = sim.intrinsic
    intrinsics = CameraIntrinsic(640, 480, 540.0, 540.0, 320.0, 240.0)
    sim_selected.camera = sim_selected.world.add_camera(intrinsics, 0.1, 2.0)
    
    
    for idc in indices:
        # if idc == 0:
        pose = Transform.from_matrix(obj_info[idc][2])
        if idc == 0:
            sim_selected.world.load_urdf(obj_info[idc][0].replace(".obj",".urdf"), pose, 0.6)
        else:
            sim_selected.world.load_urdf(find_urdf(obj_info[idc][0].replace(".obj",".urdf").replace('meshes_centered','acronym_urdfs_centered')), pose, 1)
    return sim_selected

# # Function: visualize_depth_map
# def visualize_depth_map(depth_map, colormap=None):
#     """
#     Visualizes a depth map.

#     Parameters:
#     - depth_map: A 2D numpy array representing the depth map where each element is a depth value.
#     - colormap: A string representing a matplotlib colormap (e.g., 'jet', 'viridis'). If None, will display in grayscale.

#     Shows the depth map using plt.imshow() with the specified colormap.
#     """
#     # Normalize the depth map to be in the range [0, 1]
#     normalized_depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
    
#     # Show the normalized depth map
#     plt.imshow(normalized_depth_map, cmap=colormap)
#     plt.colorbar()  # Optional: show a color bar indicating the depth scale
#     plt.axis('off')  # Hide the axis
#     plt.show()

# Function: render_side_images
def render_side_images(sim, n=1, random=False):
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, sim.size / 3])

    extrinsics = np.empty((n, 7), np.float32)
    depth_imgs = np.empty((n, height, width), np.float32)

    for i in range(n):
        if random:
            r = np.random.uniform(1.6, 2.4) * sim.size
            theta = np.random.uniform(np.pi / 4.0, 5.0 * np.pi / 12.0)
            phi = np.random.uniform(- 5.0 * np.pi / 5, - 3.0 * np.pi / 8.0)
        else:
            r = 2 * sim.size
            theta = np.pi / 3.0
            phi = - np.pi / 2.0

        extrinsic = camera_on_sphere(origin, r, theta, phi)
        depth_img = sim.camera.render(extrinsic)[1]

        extrinsics[i] = extrinsic.to_list()
        depth_imgs[i] = depth_img

    return depth_imgs, extrinsics

# Function: render_side_images_sim_single
def render_side_images_sim_single(sim, n=1, random=False):
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, sim.size / 3])

    extrinsic = np.empty(( 7), np.float32)
    depth_img = np.empty((height, width), np.float32)
    seg_img = np.empty((height, width), np.float32)

    # for i in range(n):
    if random:
        r = np.random.uniform(1.6, 2.4) * sim.size
        theta = np.random.uniform(np.pi / 4.0, 5.0 * np.pi / 12.0)
        phi = np.random.uniform(- 5.0 * np.pi / 5, - 3.0 * np.pi / 8.0)
    else:
        r = 2 * sim.size
        theta = np.pi / 3.0
        phi = - np.pi / 2.0

    extrinsic = camera_on_sphere(origin, r, theta, phi)
    depth_img,seg_img = sim.camera.render_sim(extrinsic)[1], sim.camera.render_sim(extrinsic)[2]

    return depth_img, seg_img, extrinsic


# Function: render_side_images_sim
def render_side_images_sim(sim, n=1, random=False):
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, sim.size / 3])

    extrinsics = np.empty((n, 7), np.float32)
    depth_imgs = np.empty((n, height, width), np.float32)
    seg_imgs = np.empty((n, height, width), np.float32)

    for i in range(n):
        if random:
            r = np.random.uniform(1.6, 2.4) * sim.size
            theta = np.random.uniform(np.pi / 4.0, 5.0 * np.pi / 12.0)
            phi = np.random.uniform(- 5.0 * np.pi / 5, - 3.0 * np.pi / 8.0)
        else:
            r = 2 * sim.size
            theta = np.pi / 3.0
            phi = - np.pi / 2.0

        extrinsic = camera_on_sphere(origin, r, theta, phi)
        depth_img,seg_img = sim.camera.render_sim(extrinsic)[1], sim.camera.render_sim(extrinsic)[2]

        extrinsics[i] = extrinsic.to_list()
        depth_imgs[i] = depth_img
        seg_imgs[i] = seg_img

    return depth_imgs, seg_imgs, extrinsics


# Function: render_images
# def render_images(sim, n):
#     height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
#     origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0])

#     extrinsics = np.empty((n, 7), np.float32)
#     depth_imgs = np.empty((n, height, width), np.float32)
#     seg_imgs = np.empty((n, height, width), np.float32)

#     for i in range(n):
#         r = np.random.uniform(1.6, 2.4) * sim.size
#         theta = np.random.uniform(0.0, np.pi / 4.0)
#         phi = np.random.uniform(0.0, 2.0 * np.pi)

#         extrinsic = camera_on_sphere(origin, r, theta, phi)
#         depth_img, seg_img = sim.camera.render_with_seg(extrinsic)[1], sim.camera.render_with_seg(extrinsic)[2]

#         extrinsics[i] = extrinsic.to_list()
#         depth_imgs[i] = depth_img
#         seg_imgs[i] = seg_img

#     return depth_imgs, extrinsics


def render_images(sim, n,segmentation=False):
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[
                       sim.size / 2, sim.size / 2, 0.0])
    # extrinsics = np.empty((n, 7), np.float32)
    # depth_imgs = np.empty((n, height, width), np.float32)
    # if segmentation:
    #     seg_imgs = np.empty((n, height, width), np.float32)
    extrinsics = np.empty((2*n, 7), np.float32)
    depth_imgs = np.empty((2*n, height, width), np.float32)
    if segmentation:
        seg_imgs = np.empty((2*n, height, width), np.float32)

    for i in range(n):
        for j in range(2):
            r = 2 * sim.size
            theta = (j+1) * np.pi / 6.0
            phi = 2.0 * np.pi * i / n

            extrinsic = camera_on_sphere(origin, r, theta, phi)
            depth_img = sim.camera.render(extrinsic)[1]
            # if segmentation:
            _, depth_img, seg = sim.camera.render_with_seg(extrinsic, segmentation)


            extrinsics[2*i+j] = extrinsic.to_list()
            depth_imgs[2*i+j] = depth_img
            if segmentation:
                seg_imgs[2*i+j] = seg
    
    if segmentation:
        return depth_imgs, extrinsics, seg_imgs
    else:
        return depth_imgs, extrinsics
    

# def render_images(sim, n, seg=False):
#     height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
#     origin = Transform(Rotation.identity(), np.r_[
#                        sim.size / 2, sim.size / 2, 0.0])

#     extrinsics = np.empty((2*n, 7), np.float32)
#     depth_imgs = np.empty((2*n, height, width), np.float32)

#     for i in range(n):
#         for j in range(2):
#             r = 2 * sim.size
#             theta = (j+1) * np.pi / 3.0
#             phi = 2.0 * np.pi * i / n

#             extrinsic = camera_on_sphere(origin, r, theta, phi)
#             depth_img = sim.camera.render(extrinsic)[1]
#             if seg == True:
#                 seg_img = sim.camera.render_with_seg(extrinsic)[2]

#             extrinsics[2*i+j] = extrinsic.to_list()
#             depth_imgs[2*i+j] = depth_img
            
#     if seg == False:
#         return depth_imgs, extrinsics
#     else:
#         return depth_imgs, extrinsics, seg_img

# Function: construct_homogeneous_matrix
def construct_homogeneous_matrix(rotation, translation):
    """
    Constructs a homogeneous transformation matrix.

    Parameters:
    - rotation (numpy array of shape (3,3)): Rotation matrix.
    - translation (numpy array of shape (3,1)): Translation vector.

    Returns:
    - numpy array of shape (4,4): Homogeneous transformation matrix.
    """
    # Create a 4x4 identity matrix
    H = np.eye(4)
    
    # Insert the rotation matrix into H
    H[:3, :3] = rotation
    
    # Insert the translation vector into H
    H[:3, 3] = translation[:, 0]
    
    return H

# Function: as_mesh
def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    The returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces, visual=g.visual)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

# Importing necessary libraries and modules
import trimesh
# Importing necessary libraries and modules
import numpy as np

# Function: color_trimesh_scene
def color_trimesh_scene(trimesh_scene, targ_idc, obj_idcs):
    colored_scene = trimesh.Scene()
    obj_count = 0

    for obj_id, geom_name in trimesh_scene.geometry.items():
        obj_mesh = trimesh_scene.geometry[obj_id]
        
        if obj_count in obj_idcs:
            color = None
            if obj_count == 0:
                color = np.array([128, 128, 128, 255])  # grey
            elif obj_count == targ_idc and obj_count != 0:
                color = np.array([255, 0, 0, 255])  # red
            elif obj_count != targ_idc and obj_id != 'table':
                color = np.array([0, 0, 255, 255])  # blue
            
            # If obj_mesh doesn't have vertex colors or you want to overwrite them, assign new color
            if color is not None:
                obj_mesh.visual.vertex_colors = color

            # Add the colored mesh to the new scene
            colored_scene.add_geometry(
                obj_mesh,
                node_name=obj_id,
                geom_name=obj_id
            )
        
        obj_count += 1

    return colored_scene



# Function: get_scene_from_mesh_pose_list_color
def get_scene_from_mesh_pose_list_color(mesh_pose_list, targ_idc, scene_as_mesh=True, return_list=False):
    # create scene from meshes
    scene = trimesh.Scene()
    mesh_list = []
    count = 0
    for mesh_path, scale, pose in mesh_pose_list:
        if os.path.splitext(mesh_path)[1] == '.urdf':
# Load and parse URDF files
            obj = URDF.load(mesh_path)
            assert len(obj.links) == 1
            assert len(obj.links[0].visuals) == 1
            assert len(obj.links[0].visuals[0].geometry.meshes) == 1
            mesh = obj.links[0].visuals[0].geometry.meshes[0].copy()
        else:
            mesh = trimesh.load(mesh_path)

        mesh.apply_scale(scale)
        mesh.apply_transform(pose)
        if count == 0:
            color = np.array([128, 128, 128, 255]) 
        if count == targ_idc:
            color = np.array([255, 0, 0, 255])
        elif count != targ_idc and count != 0:
            color = np.array([0, 0, 255, 255])
        mesh.visual.vertex_colors = color
        scene.add_geometry(mesh)
        mesh_list.append(mesh)
        count += 1
    if scene_as_mesh:
        scene = as_mesh(scene)
    if return_list:
        return scene, mesh_list
    else:
        return scene
# Importing necessary libraries and modules
import os

# import os
# Function: find_grasp_path
def find_grasp_path(grasp_root, grasp_id):

    for filename in os.listdir(grasp_root):
        if grasp_id in filename:
            grasp_path = os.path.join(grasp_root, filename)
            # print(f'Found grasp path: {grasp_path}')
    return grasp_path

# Function: visualize_mask
# def visualize_mask(mask, colormap='jet'):
#     """
#     Visualizes a segmentation mask as an image with each segment shown in a different color.

#     Parameters:
#     - mask: A 2D numpy array where each element is a segmentation label.
#     - colormap: The colormap to use for visualizing the segmentation. Defaults to 'jet'.
#     """
#     # Handle the -1 labels as transparent in the visualization
#     mask = mask + 1  # Increment to make -1 become 0, which will be mapped to transparent

#     # Create a color map
#     cmap = plt.get_cmap(colormap)
#     colors = cmap(np.linspace(0, 1, np.max(mask) + 1))  # Generate enough colors
#     colors[0] = np.array([0, 0, 0, 0])  # Set the first color (for label -1) as transparent

#     # Create a new ListedColormap
#     new_cmap = ListedColormap(colors)

#     # Show the mask
#     plt.imshow(mask, cmap=new_cmap)
#     plt.axis('off')  # Hide the axis
#     plt.show()

# Example usage:
# Assuming 'mask' is your 2D numpy array with segmentation labels starting from -1
# visualize_mask(mask)

def collect_mesh_pose_dict(sim, exclude_plane=False):
    mesh_pose_dict = {}
    for uid in sim.world.bodies.keys():
        _, name = sim.world.p.getBodyInfo(uid)
        name = name.decode('utf8')
        if name == 'plane' and exclude_plane:
            continue
        body = sim.world.bodies[uid]
        pose = body.get_pose().as_matrix()
        # scale = body.scale1
        visuals = sim.world.p.getVisualShapeData(uid)
        assert len(visuals) == 1
        _, _, _, scale, mesh_path, _, _, _ = visuals[0]
        mesh_path = mesh_path.decode('utf8')
        if mesh_path == '':
            mesh_path = os.path.join('/home/ding/ran-gr/GraspInClutter/data/urdfs/pile/train', name + '.urdf')
        mesh_pose_dict[uid] = (mesh_path, scale, pose)
    return mesh_pose_dict


# Function: collect_mesh_pose_list
def collect_mesh_pose_list(sim, exclude_plane=False):
    mesh_pose_list = []
    for uid in sim.world.bodies.keys():
        _, name = sim.world.p.getBodyInfo(uid)
        name = name.decode('utf8')
        if name == 'plane' and exclude_plane:
            continue
        body = sim.world.bodies[uid]
        pose = body.get_pose().as_matrix()
        # scale = body.scale1
        visuals = sim.world.p.getVisualShapeData(uid)
        assert len(visuals) == 1
        _, _, _, scale, mesh_path, _, _, _ = visuals[0]
        mesh_path = mesh_path.decode('utf8')
        if mesh_path == '':
            mesh_path = os.path.join('/home/ding/ran-gr/GraspInClutter/data/urdfs/pile/train', name + '.urdf')
        mesh_pose_list.append((mesh_path, scale, pose))
    return mesh_pose_list

# Function: extract_mesh_id
def extract_mesh_id(path):
    """
    Extracts the mesh_id from the given path.

    Parameters:
    - path (str): Input path string.

    Returns:
    - str: Extracted mesh_id.
    """
    # Split the path using the delimiter '/'
    parts = path.split('/')
    
    # Get the last part, which should be in the format 'mesh_id_collision.obj'
    # filename = parts[-1]
    mesh_id = os.path.splitext(parts[-1])[0]
    
    # Split the filename using the delimiter '_'
    # mesh_id = filename.split('_')[0]
    
    return mesh_id

# Function: get_occlusion_from_hdf5
def get_occlusion_from_hdf5(filename):
    with h5py.File(filename, 'r') as f:
        cluttered_occ_level = f['cluttered_occ_level'][()]
    return cluttered_occ_level

# Function: count_cluttered_bins
def count_cluttered_bins(root_folder):
    scene_info_path = os.path.join(root_folder)                                                                     
    cluttered_bin_counts = [0] * 10


    for file in os.listdir(scene_info_path):
        if file.endswith(".h5"):
            full_path = os.path.join(scene_info_path, file)
            cluttered_occ = get_occlusion_from_hdf5(full_path)
            
            # Count for cluttered_bin_counts
            index = int(cluttered_occ * 10)
            index = min(index, 9)  # To make sure index stays within bounds
            cluttered_bin_counts[index] += 1
    return cluttered_bin_counts


# Function: find_unique_grasps
def find_unique_grasps(pairwise_grasps, cluttered_grasps):
    # Step 1: Reshape to 2D
    cluttered_flat = cluttered_grasps.reshape(cluttered_grasps.shape[0], -1)
    pairwise_flat = pairwise_grasps.reshape(pairwise_grasps.shape[0], -1)

    # Step 2: View as structured array
    dtype = [('f{}'.format(i), cluttered_flat.dtype) for i in range(cluttered_flat.shape[1])]
    cluttered_struct = cluttered_flat.view(dtype=dtype)
    pairwise_struct = pairwise_flat.view(dtype=dtype)

    # Step 3: Use np.setdiff1d
    result_struct = np.setdiff1d(pairwise_struct, cluttered_struct)

    # Reshape result back to 3D
    result = result_struct.view(pairwise_grasps.dtype).reshape(-1, pairwise_grasps.shape[1], pairwise_grasps.shape[2])

    return result


# Function: create_grippers_scene
# def create_grippers_scene(scene_grasps):
#     """
#     Visualizes table top scene with grasps and saves the visualization to an image file.

#     Arguments:
#         scene_grasps {np.ndarray} -- Nx4x4 grasp transforms
#         save_path {str} -- Path to save the visualization image
#     """
#     print('Visualizing scene and grasps.. takes time')
    
#     gripper_marker = create_gripper_marker(color=[0, 255, 0])
#     gripper_markers = [gripper_marker.copy().apply_transform(t) for t in scene_grasps]

#     return trimesh.Scene(gripper_markers)


# Function: create_pairwise_grippers_scene
# def create_pairwise_grippers_scene(pairwise_grasps, cluttered_grasps):
#     """
#     Visualizes table top scene with grasps and saves the visualization to an image file.

#     Arguments:
#         scene_grasps {np.ndarray} -- Nx4x4 grasp transforms
#         save_path {str} -- Path to save the visualization image
#     """
#     print('Visualizing scene and grasps.. takes time')
    
#     gripper_marker = create_gripper_marker(color=[255, 255, 0]) # yellow
#     scene_grasps = find_unique_grasps(pairwise_grasps, cluttered_grasps)
#     gripper_markers = [gripper_marker.copy().apply_transform(t) for t in scene_grasps]

    

    return trimesh.Scene(gripper_markers)

# def create_mesh_from_tsdf(tsdf_grid, threshold, save_path):
#     """
#     Create a mesh from a TSDF grid and save it to a file.

#     :param tsdf_grid: A 3D numpy array representing the TSDF grid.
#     :param threshold: The threshold value to use for the surface extraction.
#     :param save_path: Path where the mesh will be saved.
#     """
#     # Use Marching Cubes algorithm to extract the surface
#     vertices, faces = mcubes.marching_cubes(tsdf_grid, threshold)

#     # Create a mesh
#     mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

#     # Save the mesh to the specified path
#     mesh.export(save_path)

# Function: read_from_hdf5
def read_from_hdf5(filename):
    with h5py.File(filename, 'r') as f:
        grasp_paths = [n.decode('ascii') for n in f['grasp_paths']]
        obj_transforms = f['obj_transforms'][:]
        obj_scales = f['obj_scales'][:]
        obj_num = f['obj_num'][()]
        targ_idc = f['targ_idc'][()]
        pairwise_scene_objects_idcs = f['pairwise_scene_objects_idcs'][:]
        pairwise_scene_idcs = f['pairwise_scene_idcs'][:]
        cluttered_scene_filtered_grasps = f['cluttered_scene_filtered_grasps'][:]
        single_object_grasps_no_scene = f['single_object_grasps_no_scene'][:]
        single_object_grasps_scene = f['single_object_grasps_scene'][:]
        pairwise_scene_filtered_grasps = f['pairwise_scene_filtered_grasps'][:]
        cluttered_occ_level = f['cluttered_occ_level'][()]
        # pairwise_occ_level =f['pairwise_occ_level'][:]
        pairwise_occ_level = np.ones(obj_num)
        # fail_ratio = f['fail_ratio']
        # fail_ratio = f['fail_ratio'][()]
                # self._table_dims = [0.5, 0.6, 0.6]
        # self._table_support = [0.6, 0.6, 0.6]
        # self._table_pose = np.eye(4)
        # self._lower_table = 0.02
        return SceneInfo(grasp_paths, obj_transforms, obj_scales, obj_num, 
                        targ_idc, pairwise_scene_objects_idcs, pairwise_scene_idcs,
                        cluttered_scene_filtered_grasps, 
                        single_object_grasps_no_scene, 
                        single_object_grasps_scene,
                        pairwise_scene_filtered_grasps,
                        cluttered_occ_level ,
                        pairwise_occ_level,
                        [0.5, 0.6, 0.6], [0.6, 0.6, 0.6], np.eye(4), 0.02)
        

# Function: convert_mesh_file_path
def convert_mesh_file_path(file_path):
    """
    Convert a file path from 'meshes/PictureFrame/1049af17ad48aaeb6d41c42f7ade8c8.obj' to 'meshes/1049af17ad48aaeb6d41c42f7ade8c8.obj'.

    Arguments:
        file_path {str} -- file path to convert

    Returns:
        str -- converted file path
    """
    # Split the file path into directory and filename components
    directory, filename = os.path.split(file_path)

    # Split the directory component into subdirectories
    # subdirectories = directory.split(os.path.sep)

    # # If the first subdirectory is 'meshes', remove it
    # if subdirectories[0] == 'meshes':
    #     subdirectories.pop(0)

    # Join the subdirectories and filename components to form the new file path
    new_file_path = os.path.join('meshes',filename)

    return new_file_path

# Function: load_grasps_h5
def load_grasps_h5(root_folder):
    """
    Load grasps into memory

    Arguments:
        root_folder {str} -- path to acronym data

    Keyword Arguments:
        splits {list} -- object data split(s) to use for scene generation
        min_pos_grasps {int} -- minimum successful grasps to consider object

    Returns:
        [dict] -- h5 file names as keys and grasp transforms as values
    """
    grasp_infos = {}
    grasp_paths = glob.glob(os.path.join(root_folder,'grasps', '*.h5'))
    grasp_contact_paths = glob.glob(os.path.join(root_folder,'mesh_contacts', '*.npz'))
    for grasp_path in grasp_paths:
        with h5py.File(grasp_path, 'r') as f:
            grasp_contact_path = grasp_path.replace('grasps', 'mesh_contacts').replace('.h5', '.npz')
            if os.path.exists(grasp_contact_path):
                all_grasp_suc = f['grasps']['qualities']['flex']['object_in_gripper'][:].reshape(-1)
                pos_idcs = np.where(all_grasp_suc > 0)[0]
                if len(pos_idcs) > 0 and os.path.exists(os.path.join(root_folder,  convert_mesh_file_path(f['object']['file'][()].decode('utf-8')))):
                    grasp_contact = np.load(grasp_contact_path)
                    valid_idc = np.where(grasp_contact['valid_locations.npy'] == 1)
                    grasp_succ_label = np.where(grasp_contact['successful.npy'][valid_idc] == 1)
                    grasp_transform = grasp_contact['grasp_transform.npy'][valid_idc]
                    # grasp_succ_label = np.where(grasp_contact['successful.npy'][valid_idc] == 1)
                    grasp_contact_points = grasp_contact['contact_points.npy'][valid_idc]
                    grasp_width =   np.linalg.norm(grasp_contact_points[:, 1, :] - grasp_contact_points[:, 0, :], axis=1)
                    grasp_id = os.path.basename(grasp_path).split('_')[0] + '_' + os.path.basename(grasp_path).split('_')[1]
                    grasp_infos[grasp_id] = {}
                    # grasp_infos[grasp_id]['grasp_transform'] = f['grasps']['transforms'][:]
                    # grasp_infos[grasp_id]['successful'] = f['grasps']['qualities']['flex']['object_in_gripper'][:]
                    grasp_infos[grasp_id]['grasp_transform'] = grasp_transform
                    grasp_infos[grasp_id]['successful'] = grasp_succ_label
                    grasp_infos[grasp_id]['grasp_width'] = grasp_width
                    grasp_infos[grasp_id]['mesh_file'] = convert_mesh_file_path(f['object']['file'][()].decode('utf-8'))
                    grasp_infos[grasp_id]['scale'] = f['object']['scale'][()]
                    grasp_infos[grasp_id]['inertia'] = f['object']['inertia'][:]
                    grasp_infos[grasp_id]['mass'] = f['object']['mass'][()]
                    # uccess_rate = len(pos_idcs) / len(all_grasp_suc)
                    grasp_infos[grasp_id]['com'] =  f['object']['com'][:]
    return grasp_infos

# Function: generate_robot_xml
def generate_robot_xml(name, visual_mesh_filename, collision_mesh_filename, mass, inertia, scale):
    """
    Generate an XML string for a robot with a single link.

    Arguments:
        name {str} -- name of the robot
        visual_mesh_filename {str} -- filename of the visual mesh
        collision_mesh_filename {str} -- filename of the collision mesh
        mass {float} -- mass of the link
        inertia {tuple} -- tuple containing the moments of inertia (ixx, ixy, ixz, iyy, iyz, izz)

    Returns:
        str -- XML string for the robot
    """
    xml = f'<?xml version="1.0"?>\n'
    xml += f'<robot name="{name}">\n'
    xml += f'  <link name="base_link">\n'
    xml += f'    <contact>\n'
    xml += f'      <lateral_friction value="1.0"/>\n'
    xml += f'      <rolling_friction value="0.0"/>\n'
    xml += f'      <contact_cfm value="0.0"/>\n'
    xml += f'      <contact_erp value="1.0"/>\n'
    xml += f'    </contact>\n'
    xml += f'    <inertial>\n'
    xml += f'      <mass value="{mass}"/>\n'
    xml += f'      <inertia ixx="{inertia[0]}" ixy="{inertia[1]}" ixz="{inertia[2]}" iyy="{inertia[3]}" iyz="{inertia[4]}" izz="{inertia[5]}"/>\n'
    xml += f'    </inertial>\n'
    xml += f'    <visual>\n'
    xml += f'      <geometry>\n'
    xml += f'        <mesh filename="{visual_mesh_filename}" scale="{scale} {scale} {scale}"/>\n'
    xml += f'      </geometry>\n'
    xml += f'    </visual>\n'
    xml += f'    <collision>\n'
    xml += f'      <geometry>\n'
    xml += f'        <mesh filename="{collision_mesh_filename}" scale="{scale} {scale} {scale}"/>\n'
    xml += f'      </geometry>\n'
    xml += f'    </collision>\n'
    xml += f'  </link>\n'
    xml += f'</robot>\n'

    return xml

# Function: save_robot_xml
def save_robot_xml(xml_string, directory, filename):
    """
    Save an XML string to a file in a directory.

    Arguments:
        xml_string {str} -- XML string to save
        directory {str} -- directory to save the file in
        filename {str} -- name of the file to save
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, f'{filename}.urdf')
    with open(file_path, 'w') as f:
        f.write(xml_string)
# Function: load_grasps
def load_grasps(root_folder, data_splits, splits=['train'], min_pos_grasps=1):
    """
    Load grasps into memory

    Arguments:
        root_folder {str} -- path to acronym data
        data_splits {dict} -- dict of categories of train/test object grasp files

    Keyword Arguments:
        splits {list} -- object data split(s) to use for scene generation
        min_pos_grasps {int} -- minimum successful grasps to consider object

    Returns:
        [dict] -- h5 file names as keys and grasp transforms as values
    """
    grasp_infos = {}
    for category_paths in data_splits.values():
        for split in splits:
            for grasp_path in category_paths[split]:
                grasp_file_path = os.path.join(root_folder, 'grasps', grasp_path)
                if os.path.exists(grasp_file_path):
                    with h5py.File(grasp_file_path, 'r') as f:
                        all_grasp_suc =  f['grasps']['qualities']['flex']['object_in_gripper'][:].reshape(-1)
                        pos_idcs = np.where(all_grasp_suc>0)[0]
                        if len(pos_idcs) > min_pos_grasps:
                            grasp_infos[grasp_path] = {}
                            grasp_infos[grasp_path]['grasp_transform'] = f['grasps']['transforms'][:]
                            grasp_infos[grasp_path]['successful'] = f['grasps']['qualities']['flex']['object_in_gripper'][:]
    if not grasp_infos:
        print('Warning: No grasps found. Please ensure the grasp data is present!')

    return grasp_infos

# Function: load_splits
def load_splits(root_folder):
    """
    Load splits of training and test objects

    Arguments:
        root_folder {str} -- path to acronym data

    Returns:
        [dict] -- dict of category-wise train/test object grasp files
    """
    split_dict = {}
    split_paths = glob.glob(os.path.join(root_folder, 'splits/*.json'))
    for split_p in split_paths:
        category = os.path.basename(split_p).split('.json')[0]
        splits = json.load(open(split_p,'r'))
        split_dict[category] = {}
        split_dict[category]['train'] = [obj_p.replace('.json', '.h5') for obj_p in splits['train']]
        split_dict[category]['test'] = [obj_p.replace('.json', '.h5') for obj_p in splits['test']]
    return split_dict



# class SceneInfo:
# # Function: __init__
#     def __init__(self, grasp_paths, obj_transforms, obj_scales, obj_num, targ_idc, pairwise_scene_objects_idcs, pairwise_scene_idcs,
#                  cluttered_scene_filtered_grasps, 
#                  single_object_grasps_no_scene, 
#                  single_object_grasps_scene,
#                 pairwise_scene_filtered_grasps,
#                 cluttered_occ_level,
#                 pairwise_occ_level,
#                 supp_obj_path, supp_pose, supp_scale,
#                 # table_dims, table_support, table_pose, lower_table,
#                 depth_cluttered, depth_single, depth_pairwise_list,camera_pose
                
#                 ):

#         self.grasp_paths = grasp_paths
#         self.obj_transforms = obj_transforms
#         self.obj_scales = obj_scales
#         self.obj_num = obj_num
#         self.targ_idc = targ_idc
#         self.pairwise_scene_objects_idcs = pairwise_scene_objects_idcs
#         self.pairwise_scene_idcs = pairwise_scene_idcs
#         self.cluttered_scene_filtered_grasps = cluttered_scene_filtered_grasps
#         self.single_object_grasps_no_scene = single_object_grasps_no_scene
#         self.single_object_grasps_scene = single_object_grasps_scene
#         self.pairwise_scene_filtered_grasps = pairwise_scene_filtered_grasps
#         self.cluttered_occ_level = cluttered_occ_level
#         self.pairwise_occ_level = pairwise_occ_level


#         # self._table_dims = [0.5, 0.6, 0.6]
#         # self._table_support = [0.6, 0.6, 0.6]
#         # self._table_pose = np.eye(4)
#         # self._lower_table = 0.02
        
#         # self._table_dims = table_dims
#         # self._table_support = table_support
#         # self._table_pose = table_pose
#         # self._lower_table = lower_table
        
        
#         # self._table_pose[2, 3] -= self._lower_table
#         # self.table_mesh = trimesh.creation.box(self._table_dims)
#         # self.table_mesh.apply_transform(self._table_pose)
#         # self.table_support = trimesh.creation.box(self._table_support)
#         # self.table_support.apply_transform(self._table_pose)

#         self.depth_cluttered = depth_cluttered
#         self.depth_single = depth_single
#         self.depth_pairwise_list = depth_pairwise_list
#         self.camera_pose = camera_pose

#         self._objects = OrderedDict()
#         self._poses = OrderedDict()

#         plane_mesh = trimesh.load(supp_obj_path.replace('.urdf', '.obj'))
#         plane_mesh.apply_scale(supp_scale)
    
#         self._objects['plane'] = plane_mesh
#         self._poses['plane'] = supp_pose
        
#         for i in range(self.obj_num):
#             self._objects[self.grasp_paths[i]] = load_mesh(os.path.join('/mnt/hdd/GraspInClutter/acronym', 'grasps', self.grasp_paths[i]), '/mnt/hdd/GraspInClutter/acronym')
#             mesh_mean =  np.mean(self._objects[self.grasp_paths[i]].vertices, 0, keepdims=True)
#             self._objects[self.grasp_paths[i]].vertices -= mesh_mean    ## make the objects centroid at the origin
#             self._poses[self.grasp_paths[i]] = self.obj_transforms[i] 
        
# # Function: scene_info_as_trimesh_scene
#     def scene_info_as_trimesh_scene(self, targ_idc, obj_idcs):
#         """Return trimesh scene representation.

#         Returns:
#             trimesh.Scene: Scene representation.
#         """
#         trimesh_scene = trimesh.scene.Scene()
#         obj_count = 0
#         for obj_id, obj_mesh in self._objects.items():
#             if obj_count in obj_idcs:
#                 if obj_id == 'table':
#                     obj_mesh.visual.vertex_colors = [128, 128, 128, 255]    ## grey
#                     trimesh_scene.add_geometry(
#                         obj_mesh,
#                         node_name=obj_id,
#                         geom_name=obj_id,
#                         transform=self._poses[obj_id],
#                     )
#                 if obj_count == targ_idc  and obj_count != 0:
#                     obj_mesh.visual.vertex_colors = [255, 0, 0, 255] ## red
#                     trimesh_scene.add_geometry(
#                         obj_mesh,
#                         node_name=obj_id,
#                         geom_name=obj_id,
#                         transform=self._poses[obj_id],
#                     )
#                 if obj_count != targ_idc and obj_id != 'table':
#                     obj_mesh.visual.vertex_colors = [0, 0, 255, 255]    ## blue
#                     trimesh_scene.add_geometry(
#                         obj_mesh,
#                         node_name=obj_id,
#                         geom_name=obj_id,
#                         transform=self._poses[obj_id],
#                     )
#             obj_count += 1
            
#         return trimesh_scene
    
# # Function: save_to_hdf5
#     def save_to_hdf5(self, file_name):
#         with h5py.File(file_name, 'w') as f:
#             # Strings and arrays need special treatment
#             # string_dt = h5py.special_dtype(vlen=str)
#             # f.create_dataset("obj_paths", (len(self.obj_paths),), dtype=string_dt, data=self.obj_paths)
#             f.create_dataset("grasp_paths", data = [n.encode("ascii", "ignore") for n in self.grasp_paths])
#             f.create_dataset("obj_transforms", data=self.obj_transforms)
#             f.create_dataset("obj_scales", data=self.obj_scales)
#             f.create_dataset("obj_num", data=self.obj_num)
#             f.create_dataset("targ_idc", data=self.targ_idc)
#             f.create_dataset("pairwise_scene_objects_idcs", data=self.pairwise_scene_objects_idcs)
#             f.create_dataset("cluttered_scene_filtered_grasps", data=self.cluttered_scene_filtered_grasps)
            
#             f.create_dataset("single_object_grasps_no_scene", data=self.single_object_grasps_no_scene)

#             f.create_dataset("single_object_grasps_scene", data=self.single_object_grasps_scene)

#             # f.create_dataset("fail_idcs",data= self.fail_idcs)

#             f.create_dataset("pairwise_scene_filtered_grasps", data=self.pairwise_scene_filtered_grasps)

#             # f.create_dataset("fail_ratio", data=self.fail_ratio)
#             f.create_dataset("pairwise_scene_idcs",data= self.pairwise_scene_idcs)


#             f.create_dataset("cluttered_occ_level",data= self.cluttered_occ_level)
#             f.create_dataset("pairwise_occ_level",data= self.pairwise_occ_level)

#             f.create_dataset("depth_cluttered", data =self.depth_cluttered )
#             f.create_dataset("depth_single", data =self.depth_single )
#             f.create_dataset("depth_pairwise_list", data =self.depth_pairwise_list )
#             f.create_dataset("camera_pose", data =self.camera_pose )

    
# # Function: visualize_SceneInfo
#     def visualize_SceneInfo(self):
#         print("1")
#         env_scene = trimesh.scene.Scene()
#         base_dir = '/mnt/hdd/GraspInClutter/acronym'
#         # print(self.obj_transforms)
#         # print(self.grasp_paths)

#         env_scene.add_geometry(
#                         self.table_mesh,
#                         node_name = 'table',
#                         geom_name = 'table',
#                         transform=self._table_pose,
#                     )
        
#         obj_count = 0
#         for i in range(self.obj_num):
#             # Load mesh from the object path
#             mesh = load_mesh(os.path.join('/mnt/hdd/GraspInClutter/acronym', 'grasps', self.grasp_paths[i]), '/mnt/hdd/GraspInClutter/acronym')
            
#             if obj_count != self.targ_idc:
#                 mesh.visual.vertex_colors = [0, 0, 255, 255]
#                 env_scene.add_geometry(
#                             mesh,
#                             node_name = self.grasp_paths[i],
#                             geom_name =self.grasp_paths[i] ,
#                             transform=self.obj_transforms[i],
#                         )
            
#             elif obj_count == self.targ_idc:
#                 mesh.visual.vertex_colors = [255, 0, 0, 255]
#                 env_scene.add_geometry(
#                             mesh,
#                             node_name = self.grasp_paths[i],
#                             geom_name =self.grasp_paths[i] ,
#                             transform=self.obj_transforms[i],
#                         )
#             obj_count += 1

#         env_scene = self.scene_info_as_trimesh_scene(self.targ_idc+1, range(self.obj_num + 1))
#         env_scene.show()
#         cluttered_scene = trimesh.scene.scene.append_scenes([env_scene,create_grippers_scene(self.cluttered_scene_filtered_grasps) ])
#         cluttered_scene.show()
#         print("1")


def mesh_to_occupancy(mesh: trimesh.Trimesh, 
                     resolution: int = 64, 
                     bounds: tuple = None) -> np.ndarray:
    """
    Convert a mesh to an occupancy grid using ray casting.
    
    Args:
        mesh (trimesh.Trimesh): Input mesh to convert
        resolution (int): Resolution of occupancy grid (N x N x N)
        bounds (tuple): Optional (min_bound, max_bound) to specify volume bounds.
                       If None, uses mesh bounds with small padding.
                       
    Returns:
        np.ndarray: Boolean occupancy grid of shape (resolution, resolution, resolution)
    """
    if bounds is None:
        # Add small padding to mesh bounds
        bounds = mesh.bounds
        padding = (bounds[1] - bounds[0]) * 0.05  # 5% padding
        bounds = (bounds[0] - padding, bounds[1] + padding)
    
    min_bound, max_bound = bounds
    
    # Create grid points
    x = np.linspace(min_bound[0], max_bound[0], resolution)
    y = np.linspace(min_bound[1], max_bound[1], resolution)
    z = np.linspace(min_bound[2], max_bound[2], resolution)
    
    # Create 3D grid of points
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    
    # Ensure the mesh is watertight for accurate inside/outside testing
    if not mesh.is_watertight:
        print("Warning: Mesh is not watertight, attempting to fix...")
        # Try to fix the mesh
        mesh.fill_holes()
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.remove_infinite_values()
        mesh.remove_unreferenced_vertices()
        
        if not mesh.is_watertight:
            print("Warning: Mesh could not be made watertight. Results may be inaccurate.")
    
    # Use ray casting method for occupancy check
    occupancy = np.zeros(len(points), dtype=bool)
    
    # Create rays along x-axis
    ray_origins = np.zeros_like(points)
    ray_origins[:, 0] = min_bound[0] - 1  # Start rays from outside the bounds
    ray_directions = np.zeros_like(points)
    ray_directions[:, 0] = 1  # Ray direction along x-axis
    
    # Process points in batches to avoid memory issues
    batch_size = 1000
    for i in range(0, len(points), batch_size):
        batch_end = min(i + batch_size, len(points))
        batch_points = points[i:batch_end]
        batch_origins = ray_origins[i:batch_end]
        batch_directions = ray_directions[i:batch_end]
        
        # Get ray intersections
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
        locations = intersector.intersects_location(
            ray_origins=batch_origins,
            ray_directions=batch_directions
        )
        
        # For each point in the batch
        for j, point in enumerate(batch_points):
            # Get intersections for this ray
            ray_locations = locations[0][locations[1] == j]
            # Sort intersections by x coordinate
            if len(ray_locations) > 0:
                ray_locations = ray_locations[ray_locations[:, 0].argsort()]
                # Count intersections before the point
                intersections = ray_locations[ray_locations[:, 0] <= point[0]]
                # Point is inside if number of intersections is odd
                occupancy[i + j] = len(intersections) % 2 == 1
    
    # Reshape back to 3D grid
    occupancy_grid = occupancy.reshape(resolution, resolution, resolution)
    
    return occupancy_grid

def visualize_occupancy_grid(occupancy_grid: np.ndarray, 
                           save_path: str = None,
                           title: str = "Occupancy Grid Visualization"):
    """
    Visualize a 3D occupancy grid.
    
    Args:
        occupancy_grid (np.ndarray): 3D boolean array of occupancy values
        save_path (str): Optional path to save visualization
        title (str): Title for the visualization
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get occupied voxel coordinates
    occupied = np.where(occupancy_grid)
    
    if len(occupied[0]) > 0:
        # Create scatter plot with small alpha for better visibility
        scatter = ax.scatter(occupied[0], occupied[1], occupied[2],
                           c='blue', marker='s', alpha=0.1)
        
        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Add grid
        ax.grid(True)
        
        # Add title
        plt.title(title)
        
        # Add colorbar
        plt.colorbar(scatter)
    else:
        plt.title("No occupied voxels found")
    
    if save_path:
        # Create directory if needed
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# Example usage:
"""
# Load a mesh
mesh = trimesh.load('path/to/mesh.obj')

# Convert to occupancy grid
occupancy = mesh_to_occupancy(mesh, resolution=64)

# Visualize
visualize_occupancy_grid(occupancy, 'occupancy.png', "Mesh Occupancy")
"""

def tsdf_to_mesh(tsdf_volume, threshold=0.0, step_size=1, voxel_size=0.3/40, origin=None):
    """
    Convert TSDF voxel grid to a triangle mesh with real-world size
    
    Parameters:
        tsdf_volume: numpy array, TSDF voxel grid
        threshold: float, isosurface extraction threshold, default is 0
        step_size: int, sampling step size, can be used to control the detail level of the output mesh
        voxel_size: float, size of each voxel in real-world units (default: 0.3/40 meters)
        origin: numpy array, origin point of the TSDF volume in real-world coordinates
        
    Returns:
        trimesh.Trimesh: triangle mesh extracted from the TSDF with real-world scale
    """
    from skimage.measure import marching_cubes
    import trimesh
    import numpy as np
    
    # Check input type
    if not isinstance(tsdf_volume, np.ndarray):
        tsdf_volume = np.array(tsdf_volume)
    
    # Set default origin if not provided
    if origin is None:
        origin = np.array([0, 0, 0])
    
    # Extract isosurface using marching cubes algorithm
    try:
        vertices, faces, normals, _ = marching_cubes(
            tsdf_volume, 
            level=threshold,
            step_size=step_size,
            allow_degenerate=False
        )
    except Exception as e:
        print(f"Marching cubes algorithm error: {e}")
        # Return an empty mesh if processing fails
        return trimesh.Trimesh()
    
    # Scale vertices to real-world coordinates
    vertices = vertices * voxel_size + origin
    
    # Create trimesh object
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=normals
    )
    
    # Return empty mesh if the mesh is invalid or empty
    if mesh.vertices.shape[0] == 0:
        print("Warning: Generated mesh is empty.")
        return trimesh.Trimesh()
    
    return mesh

# Example usage:
"""
# Generate mesh from TSDF
tsdf_data = point_cloud_to_tsdf(points)
mesh = tsdf_to_mesh(tsdf_data)

# Save the mesh
mesh.export('output_mesh.obj')

# Or visualize directly
mesh.show()
"""
