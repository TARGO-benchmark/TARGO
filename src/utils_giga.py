import glob
import os
import open3d as o3d
import trimesh
import mcubes
import numpy as np
import json
import h5py
import sys
from vgn.utils import visual
from collections import OrderedDict
# Load and parse URDF files
from urdfpy import URDF
from torch import nn
# from vgn.simulation import ClutterRemovalSim

from vgn.utils.implicit import *
# from vgn.simulation import ClutterRemovalSim
from vgn.perception import *
# from src.vgn.eval import *
from vgn.ConvONets.eval import *
from vgn.utils.transform import Rotation, Transform
# from vgn.utils.misc import apply_noise, apply_translational_noise
# from acronym_tools import load_mesh, create_gripper_marker
# from vgn.simulation import ClutterRemovalSim
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
    print(f"Point cloud saved to {ply_file_path}")

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
    print(f"grasps are at {grasp_path}")

          
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

# def filter_and_pad_point_clouds(points, lower_bound=torch.tensor([0.02, 0.02, 0.055]) / 0.3 - 0.5, upper_bound=torch.tensor([0.28, 0.28, 0.3])/ 0.3 - 0.5, N=2048):
#     """
#     Filter each point cloud within a bounding box and pad or truncate to N points.
    
#     Args:
#         points (torch.Tensor): The point cloud data, shape (BS, N, 3).
#         lower_bound (torch.Tensor): Lower limit of the bounding box, shape (3,).
#         upper_bound (torch.Tensor): Upper limit of the bounding box, shape (3,).
#         N (int): The desired number of points in each point cloud, default is 2048.
    
#     Returns:
#         torch.Tensor: The processed point cloud data, shape (BS, N, 3). Ensures the processed points are on the same device as the input points.
#     """
#     BS, _, _ = points.shape
#     lower_bound = lower_bound.to(points.device)
#     upper_bound = upper_bound.to(points.device)
#     processed_points = torch.zeros((BS, N, 3), dtype=points.dtype, device=points.device)  # Ensure the tensor is on the same device as the input points.

#     for i in range(BS):
#         # Extract a single point cloud
#         single_point_cloud = points[i]

#         # Apply bounding box filter
#         mask = (single_point_cloud >= lower_bound) & (single_point_cloud <= upper_bound)
#         mask = mask.all(dim=1)
#         filtered_points = single_point_cloud[mask]

#         # Pad or truncate to match N points
#         filtered_len = filtered_points.size(0)
#         if filtered_len < N:
#             # If there are fewer points than N after filtering, pad with the last point
#             if filtered_len > 0:
#                 padding = filtered_points[-1].unsqueeze(0).repeat(N - filtered_len, 1)
#                 processed_points[i] = torch.cat([filtered_points, padding], dim=0)
#             else:
#                 # If no points are left after filtering, you can choose to keep an empty point cloud
#                 # or fill with zeros. Here, we choose to keep it as all zeros.
#                 continue
#         else:
#             # If there are more points than N, truncate
#             processed_points[i] = filtered_points[:N]
    
#     return processed_points

def filter_and_pad_point_clouds_rs(
    points,
    lower_bound=torch.tensor([0.02, 0.02, 0.055]),
    upper_bound=torch.tensor([0.28, 0.28, 0.3]),
    N=2048
):
    """
    Filter each point cloud within a bounding box and pad or truncate to N points, using a more
    vectorized approach for bounding-box filtering.

    Args:
        points (torch.Tensor): The point cloud data, shape (BS, M, 3). 
                               M can be any number of points (not necessarily 2048).
        lower_bound (torch.Tensor): Lower limit of the bounding box, shape (3,).
        upper_bound (torch.Tensor): Upper limit of the bounding box, shape (3,).
        N (int): The desired number of points in each point cloud. Default is 2048.
    
    Returns:
        torch.Tensor: The processed point cloud data, shape (BS, N, 3). The output is on
                      the same device as the input points.
    """
    # Number of batches (BS) and original number of points (M)
    BS, M, _ = points.shape

    # Move bounds to the same device as points
    lower_bound = lower_bound.to(points.device)
    upper_bound = upper_bound.to(points.device)

    # Create a mask for all batches at once: shape will be (BS, M)
    # We check whether each point is within the bounding box
    # mask = (points >= lower_bound) & (points <= upper_bound)  # shape: (BS, M, 3)
    # mask = mask.all(dim=2)  # reduce across the coordinate dimension, resulting in (BS, M)

    # Prepare a storage tensor for the final processed point clouds
    processed_points = torch.zeros(
        (BS, N, 3),
        dtype=points.dtype,
        device=points.device
    )

    # For each batch, pad or truncate after we've already identified which points are in the box
    for i in range(BS):
        # valid_points = points[i][mask[i]]  # shape: (num_in_box, 3)
        valid_points = points[i]
        valid_len = valid_points.size(0)

        if valid_len == 0:
            # If no points remain after filtering, we leave it as zeros
            # (You could also fill it with some default points if desired)
            continue
        elif valid_len < N:
            # Pad with the last valid point if we have fewer than N points
            padding = valid_points[-1].unsqueeze(0).repeat(N - valid_len, 1)
            processed_points[i] = torch.cat([valid_points, padding], dim=0)
        else:
            # If we have more than N points, truncate
            processed_points[i] = valid_points[:N]

    return processed_points

def filter_and_pad_point_clouds(
    points,
    lower_bound=torch.tensor([0.02, 0.02, 0.055]) / 0.3 - 0.5,
    upper_bound=torch.tensor([0.28, 0.28, 0.3]) / 0.3 - 0.5,
    N=2048
):
    """
    Filter each point cloud within a bounding box and pad or truncate to N points, using a more
    vectorized approach for bounding-box filtering.

    Args:
        points (torch.Tensor): The point cloud data, shape (BS, M, 3). 
                               M can be any number of points (not necessarily 2048).
        lower_bound (torch.Tensor): Lower limit of the bounding box, shape (3,).
        upper_bound (torch.Tensor): Upper limit of the bounding box, shape (3,).
        N (int): The desired number of points in each point cloud. Default is 2048.
    
    Returns:
        torch.Tensor: The processed point cloud data, shape (BS, N, 3). The output is on
                      the same device as the input points.
    """
    # Number of batches (BS) and original number of points (M)
    BS, M, _ = points.shape

    # Move bounds to the same device as points
    lower_bound = lower_bound.to(points.device)
    upper_bound = upper_bound.to(points.device)

    # Create a mask for all batches at once: shape will be (BS, M)
    # We check whether each point is within the bounding box
    mask = (points >= lower_bound) & (points <= upper_bound)  # shape: (BS, M, 3)
    mask = mask.all(dim=2)  # reduce across the coordinate dimension, resulting in (BS, M)

    # Prepare a storage tensor for the final processed point clouds
    processed_points = torch.zeros(
        (BS, N, 3),
        dtype=points.dtype,
        device=points.device
    )

    # For each batch, pad or truncate after we've already identified which points are in the box
    for i in range(BS):
        valid_points = points[i][mask[i]]  # shape: (num_in_box, 3)
        valid_len = valid_points.size(0)

        if valid_len == 0:
            # If no points remain after filtering, we leave it as zeros
            # (You could also fill it with some default points if desired)
            continue
        elif valid_len < N:
            # Pad with the last valid point if we have fewer than N points
            padding = valid_points[-1].unsqueeze(0).repeat(N - valid_len, 1)
            processed_points[i] = torch.cat([valid_points, padding], dim=0)
        else:
            # If we have more than N points, truncate
            processed_points[i] = valid_points[:N]

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

def point_cloud_to_mesh(points):
    ## points -> mesh -> sdf -> tsdf
    ## if the points is pytorch tensor, convert it to numpy
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if len(points.shape) == 3: 
        points = points.reshape(-1, 3)
    mesh = alpha_shape_mesh_reconstruct(points, alpha=0.5, mesh_fix=False, visualize=False)
    return mesh

def point_cloud_to_tsdf(points, return_mesh=False):
    ## points -> mesh -> sdf -> tsdf
    ## if the points is pytorch tensor, convert it to numpy
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if len(points.shape) == 3: 
        points = points.reshape(-1, 3)
    mesh = alpha_shape_mesh_reconstruct(points, alpha=0.5, mesh_fix=False, visualize=False)
    # o3d.io.write_triangle_mesh("mesh.ply", mesh)
    if return_mesh:
        return mesh, mesh_to_tsdf(mesh)
    else:
        return mesh_to_tsdf(mesh)

def compute_chamfer_and_iou(target_mesh, completed_mesh):
    """Compute Chamfer distance and IoU between target mesh and completed mesh.
    
    Args:
        target_mesh (trimesh.Trimesh): Ground truth target mesh
        completed_mesh (trimesh.Trimesh): Completed mesh
        mesh_path (Path, optional): Path to save visualization meshes
        
    Returns:
        tuple: (chamfer_distance, iou)
    """
    # Sample points from mesh surfaces for Chamfer distance
    gt_points, _ = trimesh.sample.sample_surface(target_mesh, 2048)
    completed_points, _ = trimesh.sample.sample_surface(completed_mesh, 2048)
    
    evaluator = MeshEvaluator(n_points=2048)
    
    # Normalize points to [-0.5, 0.5] space
    gt_points = (gt_points / 0.3) - 0.5
    completed_points = (completed_points / 0.3) - 0.5
    
    # Compute Chamfer distance
    eval_dict = evaluator.eval_pointcloud(
        pointcloud=completed_points,
        pointcloud_tgt=gt_points,
        normals=None,
        normals_tgt=None
    )
    chamfer_distance = eval_dict['chamfer-L1']
    
    # Normalize meshes to [-0.5, 0.5] space
    target_mesh.vertices = target_mesh.vertices / 0.3 - 0.5
    completed_mesh.vertices = completed_mesh.vertices / 0.3 - 0.5
    
    # Sample points for IoU calculation
    points_iou, occ_target = sample_iou_points(
        mesh_list=[target_mesh],
        bounds=target_mesh.bounds,
        num_point=100000,
        uniform=False,
        size=1.0
    )
    
    # Compute IoU
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

# def point_cloud_to_tsdf(points):
#     ## points -> mesh -> sdf -> tsdf
#     ## if the points is pytorch tensor, convert it to numpy
#     if isinstance(points, torch.Tensor):
#         points = points.detach().cpu().numpy()
#     if len(points.shape) == 3: 
#         points = points.reshape(-1, 3)
#     mesh = alpha_shape_mesh_reconstruct(points, alpha=0.5, mesh_fix=False, visualize=False)
#     # o3d.io.write_triangle_mesh("mesh.ply", mesh)
#     return mesh_to_tsdf(mesh)

def mesh_to_tsdf(mesh, size=0.3, resolution=40):
    """
    Convert mesh to TSDF using SDF sampling.

    Args:
        mesh: trimesh.Trimesh object
        size: Size of the workspace (float, default 0.3)
        resolution: TSDF grid resolution (int, default 40)

    Returns:
        tsdf: TSDF grid (numpy array of shape [resolution, resolution, resolution])
    """
    # Create grid coordinates in [0, size)
    lin = torch.linspace(start=0, end=size - size / resolution, steps=resolution)
    x, y, z = torch.meshgrid(lin, lin, lin, indexing='ij')
    pos = torch.stack((x, y, z), dim=-1).float()  # (resolution, resolution, resolution, 3)
    pos = pos.view(-1, 3)

    # Convert mesh vertices and faces to correct format
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.uint32)

    # Ensure faces are 2D array
    if faces.ndim == 3:
        faces = np.asarray(mesh.faces, dtype=np.uint32)

    f = SDF(vertices, faces)
    sdf = f(pos)
    sdf_reshaped = sdf.reshape(resolution, resolution, resolution)
    sdf_trunc = 4 * (size / resolution)

    mask = (sdf_reshaped >= sdf_trunc) | (sdf_reshaped <= -sdf_trunc)

    tsdf = (sdf_reshaped / sdf_trunc + 1) / 2
    tsdf[mask] = 0
    return tsdf

# def mesh_to_tsdf(mesh, size=0.3, resolution=40):
#     """
#     Convert mesh to TSDF using the same method as in training.
    
#     Args:
#         mesh: trimesh.Trimesh object
#         size: Size of the workspace
#         resolution: TSDF resolution
        
#     Returns:
#         tsdf: TSDF grid
#     """
#     try:
#         # Create camera and simulation setup
#         camera = Camera.default()
        
#         # Create TSDF
#         tsdf = create_tsdf(
#             size=size,
#             resolution=resolution,
#             depth_imgs=None,  # We don't have depth images, using mesh directly
#             intrinsic=camera.intrinsic,
#             extrinsics=None
#         )
        
#         # For mesh-based TSDF, we need to convert mesh to depth images
#         # This is a simplified approach - in practice, you might want to render from multiple viewpoints
        
#         # Alternative: use mesh voxelization
#         voxel_grid = mesh.voxelized(pitch=size/resolution)
#         voxel_array = voxel_grid.matrix.astype(np.float32)
        
#         # Convert to TSDF format (distance field)
#         # This is a simplified conversion - proper TSDF would need signed distance
#         tsdf_grid = np.where(voxel_array > 0, 1.0, -1.0)
        
#         return tsdf_grid
        
#     except Exception as e:
#         print(f"Error converting mesh to TSDF: {e}")
#         return None



    # return tsdf_volume


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
    
def save_point_cloud_as_ply(point_cloud_data, ply_file_path):
    """
    Save a numpy array of 3D points as a PLY file.

    Parameters:
    point_cloud_data : numpy.ndarray
        A numpy array with shape (n, 3) where n is the number of points.
    ply_file_path : str
        The file path where the PLY file will be saved.
    """
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()

    # Assign the points to the point cloud object
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data)

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
            _, depth_img, seg = sim.camera.render_qseg(extrinsic, segmentation)


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
            mesh_path = os.path.join(str(getattr(sim, 'urdf_root', 'data/urdfs')), getattr(sim, 'object_set', 'packed/train'), name + '.urdf')
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
            mesh_path = os.path.join(str(getattr(sim, 'urdf_root', 'data/urdfs')), getattr(sim, 'object_set', 'packed/train'), name + '.urdf')
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

def create_mesh_from_tsdf(tsdf_grid, threshold, save_path):
    """
    Create a mesh from a TSDF grid and save it to a file.

    :param tsdf_grid: A 3D numpy array representing the TSDF grid.
    :param threshold: The threshold value to use for the surface extraction.
    :param save_path: Path where the mesh will be saved.
    """
    # Use Marching Cubes algorithm to extract the surface
    vertices, faces = mcubes.marching_cubes(tsdf_grid, threshold)

    # Create a mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Save the mesh to the specified path
    mesh.export(save_path)

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
def compute_multi_object_completion_metrics(gt_meshes, pred_meshes, num_samples=10000):
    """
    Compute evaluation metrics for multi-object reconstruction using the existing compute_chamfer_and_iou function
    
    Args:
        gt_meshes: List of Ground Truth meshes (reference)
        pred_meshes: List of predicted meshes (already aligned if using GAPS)
        num_samples: Number of sample points for metric computation
        
    Returns:
        Dictionary containing various metrics and aligned scenes
    """
    print(f"GT scene contains {len(gt_meshes)} objects")
    print(f"Predicted scene contains {len(pred_meshes)} objects")
    
    if len(gt_meshes) == 0 or len(pred_meshes) == 0:
        print("Warning: No valid meshes found")
        return {
            'chamfer_distance': float('inf'),
            'iou': 0.0,
            'chamfer_distance_std': 0.0,
            'iou_std': 0.0,
            'per_object_cds': [],
            'per_object_ious': [],
            'per_object_b_ious': [],
            'scene_iou': 0.0
        }
    
    # Use the provided meshes directly (alignment already done in eval.py)
    aligned_pred_meshes = pred_meshes
    
    # Step 4: Compute per-object metrics
    print("Step 4: Computing per-object metrics...")
    per_object_cds = []
    per_object_ious = []
    per_object_b_ious = []
    
    # For each predicted object, find best matching GT object and compute metrics
    for i, pred_mesh in enumerate(aligned_pred_meshes):
        best_cd = float('inf')
        best_iou = 0.0
        best_b_iou = 0.0
        best_gt_idx = -1
        
        for j, gt_mesh in enumerate(gt_meshes):
            try:
                # Use the existing compute_chamfer_and_iou function
                cd, iou = compute_chamfer_and_iou(gt_mesh, pred_mesh)
                
                # Optional: Save meshes for debugging
                # gt_mesh.export(f"gt_{j}.ply", include_attributes=False)
                # pred_mesh.export(f"pred_{i}.ply", include_attributes=False)
                
                if cd < best_cd:
                    best_cd = cd
                    best_iou = iou
                    best_gt_idx = j
                    
            except Exception as e:
                print(f"Error computing metrics for pred object {i} and GT object {j}: {e}")
                continue
        
        if best_gt_idx >= 0:
            per_object_cds.append(best_cd)
            per_object_ious.append(best_iou)
            per_object_b_ious.append(best_b_iou)
            print(f"Pred object {i} matched with GT object {best_gt_idx}: CD={best_cd:.6f}, IoU={best_iou:.6f}, B-IoU={best_b_iou:.6f}")
        else:
            per_object_cds.append(float('inf'))
            per_object_ious.append(0.0)
            per_object_b_ious.append(0.0)
            print(f"No match found for pred object {i}")
    
    # Compute scene-level metrics
    if per_object_cds:
        scene_cd = np.mean(per_object_cds)
        scene_iou = np.mean(per_object_ious)
        scene_cd_std = np.std(per_object_cds)
        scene_iou_std = np.std(per_object_ious)
    else:
        scene_cd = float('inf')
        scene_iou = 0.0
        scene_cd_std = 0.0
        scene_iou_std = 0.0
    
    # Create metrics dictionary
    metrics = {
        'chamfer_distance': float(scene_cd),
        'iou': float(scene_iou),
        'chamfer_distance_std': float(scene_cd_std),
        'iou_std': float(scene_iou_std),
        'per_object_cds': per_object_cds,
        'per_object_ious': per_object_ious,
        'per_object_b_ious': per_object_b_ious,
        'scene_iou': float(scene_iou)
    }
    
    # Create aligned scenes for compatibility
    aligned_gt_scene = trimesh.Scene(gt_meshes)
    aligned_pred_scene = trimesh.Scene(aligned_pred_meshes)
    
    # Store aligned scenes for later use
    metrics['aligned_gt_scene'] = aligned_gt_scene
    metrics['aligned_pred_scene'] = aligned_pred_scene
    
    # Create merged meshes for compatibility
    if len(gt_meshes) > 1:
        metrics['aligned_gt_merged'] = trimesh.util.concatenate(gt_meshes)
    else:
        metrics['aligned_gt_merged'] = gt_meshes[0] if gt_meshes else None
    
    if len(aligned_pred_meshes) > 1:
        metrics['aligned_pred_merged'] = trimesh.util.concatenate(aligned_pred_meshes)
    else:
        metrics['aligned_pred_merged'] = aligned_pred_meshes[0] if aligned_pred_meshes else None
    
    print(f"Final scene metrics: CD={metrics['chamfer_distance']:.6f}±{metrics['chamfer_distance_std']:.6f}, "
          f"IoU={metrics['iou']:.6f}±{metrics['iou_std']:.6f}")
    
    return metrics

def compute_penetration_level(mesh_list, merged_mesh, size=0.3, resolution=40):
    """
    Compute penetration level between merged mesh and individual meshes.
    
    Penetration level = 1 - (merged_mesh_internal_points / sum_of_individual_mesh_internal_points)
    
    Args:
        mesh_list: List of trimesh.Trimesh objects
        merged_mesh: Single merged trimesh.Trimesh object
        size: Size of the workspace (float, default 0.3)
        resolution: TSDF grid resolution (int, default 40)
        
    Returns:
        float: Penetration level (0 = no penetration, 1 = complete overlap)
    """
    if not mesh_list or merged_mesh is None:
        print("Warning: Invalid input meshes")
        return 0.0
    
    print(f"Computing penetration level for {len(mesh_list)} individual meshes vs 1 merged mesh")
    
    try:
        # Convert merged mesh to TSDF
        merged_tsdf = mesh_to_tsdf(merged_mesh, size, resolution)
        
        # Count internal points in merged mesh (TSDF > 0.5 indicates inside)
        merged_internal_points = np.sum(merged_tsdf > 0.5)
        print(f"Merged mesh internal points: {merged_internal_points}")
        
        # Convert each individual mesh to TSDF and count internal points
        individual_internal_points_sum = 0
        
        for i, mesh in enumerate(mesh_list):
            try:
                individual_tsdf = mesh_to_tsdf(mesh, size, resolution)
                individual_internal_points = np.sum(individual_tsdf > 0.5)
                individual_internal_points_sum += individual_internal_points
                print(f"Mesh {i} internal points: {individual_internal_points}")
            except Exception as e:
                print(f"Error processing mesh {i}: {e}")
                continue
        
        print(f"Sum of individual mesh internal points: {individual_internal_points_sum}")
        
        # Calculate penetration level
        if individual_internal_points_sum > 0:
            penetration_level = 1 - (merged_internal_points / individual_internal_points_sum)
            penetration_level = max(0.0, min(1.0, penetration_level))  # Clamp to [0, 1]
        else:
            penetration_level = 0.0
            print("Warning: No internal points found in individual meshes")
        
        print(f"Penetration level: {penetration_level:.6f}")
        return float(penetration_level)
        
    except Exception as e:
        print(f"Error computing penetration level: {e}")
        return 0.0


def compute_penetration_level_detailed(mesh_list, merged_mesh, size=0.3, resolution=40):
    """
    Compute detailed penetration analysis between merged mesh and individual meshes.
    
    Args:
        mesh_list: List of trimesh.Trimesh objects
        merged_mesh: Single merged trimesh.Trimesh object
        size: Size of the workspace (float, default 0.3)
        resolution: TSDF grid resolution (int, default 40)
        
    Returns:
        dict: Detailed penetration analysis including per-mesh breakdown
    """
    if not mesh_list or merged_mesh is None:
        print("Warning: Invalid input meshes")
        return {
            'penetration_level': 0.0,
            'merged_internal_points': 0,
            'individual_internal_points_sum': 0,
            'per_mesh_internal_points': [],
            'per_mesh_penetration': [],
            'overlap_ratio': 0.0
        }
    
    print(f"Computing detailed penetration analysis for {len(mesh_list)} individual meshes vs 1 merged mesh")
    
    try:
        # Convert merged mesh to TSDF
        merged_tsdf = mesh_to_tsdf(merged_mesh, size, resolution)
        merged_internal_points = np.sum(merged_tsdf > 0.5)
        
        # Process individual meshes
        per_mesh_internal_points = []
        per_mesh_penetration = []
        
        for i, mesh in enumerate(mesh_list):
            try:
                individual_tsdf = mesh_to_tsdf(mesh, size, resolution)
                individual_internal_points = np.sum(individual_tsdf > 0.5)
                per_mesh_internal_points.append(individual_internal_points)
                
                # Calculate overlap with merged mesh
                overlap = np.sum((individual_tsdf > 0.5) & (merged_tsdf > 0.5))
                if individual_internal_points > 0:
                    mesh_penetration = 1 - (overlap / individual_internal_points)
                else:
                    mesh_penetration = 0.0
                per_mesh_penetration.append(mesh_penetration)
                
                print(f"Mesh {i}: internal={individual_internal_points}, overlap={overlap}, penetration={mesh_penetration:.6f}")
                
            except Exception as e:
                print(f"Error processing mesh {i}: {e}")
                per_mesh_internal_points.append(0)
                per_mesh_penetration.append(0.0)
        
        individual_internal_points_sum = sum(per_mesh_internal_points)
        
        # Calculate overall penetration level
        if individual_internal_points_sum > 0:
            penetration_level = 1 - (merged_internal_points / individual_internal_points_sum)
            penetration_level = max(0.0, min(1.0, penetration_level))
        else:
            penetration_level = 0.0
        
        # Calculate overlap ratio
        if individual_internal_points_sum > 0:
            overlap_ratio = merged_internal_points / individual_internal_points_sum
        else:
            overlap_ratio = 0.0
        
        results = {
            'penetration_level': float(penetration_level),
            'merged_internal_points': int(merged_internal_points),
            'individual_internal_points_sum': int(individual_internal_points_sum),
            'per_mesh_internal_points': per_mesh_internal_points,
            'per_mesh_penetration': per_mesh_penetration,
            'overlap_ratio': float(overlap_ratio)
        }
        
        print(f"Overall penetration level: {penetration_level:.6f}")
        print(f"Overlap ratio: {overlap_ratio:.6f}")
        
        return results
        
    except Exception as e:
        print(f"Error computing detailed penetration analysis: {e}")
        return {
            'penetration_level': 0.0,
            'merged_internal_points': 0,
            'individual_internal_points_sum': 0,
            'per_mesh_internal_points': [],
            'per_mesh_penetration': [],
            'overlap_ratio': 0.0
        }

def mesh_to_tsdf_messy_kitchen(mesh, size=0.7, resolution=40):
    """
    Convert mesh to TSDF for messy kitchen scenarios with different normalization.
    
    This function uses a larger workspace size (0.7) and different normalization
    approach compared to the original mesh_to_tsdf function.
    
    Args:
        mesh: trimesh.Trimesh object
        size: Size of the workspace (float, default 0.7 for messy kitchen)
        resolution: TSDF grid resolution (int, default 40)

    Returns:
        tsdf: TSDF grid (numpy array of shape [resolution, resolution, resolution])
    """
    # Create grid coordinates in [0, size) for messy kitchen
    lin = torch.linspace(start=0, end=size - size / resolution, steps=resolution)
    x, y, z = torch.meshgrid(lin, lin, lin, indexing='ij')
    pos = torch.stack((x, y, z), dim=-1).float()  # (resolution, resolution, resolution, 3)
    pos = pos.view(-1, 3)

    # Convert mesh vertices and faces to correct format
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.uint32)

    # Ensure faces are 2D array
    if faces.ndim == 3:
        faces = np.asarray(mesh.faces, dtype=np.uint32)

    f = SDF(vertices, faces)
    sdf = f(pos)
    sdf_reshaped = sdf.reshape(resolution, resolution, resolution)
    sdf_trunc = 4 * (size / resolution)

    mask = (sdf_reshaped >= sdf_trunc) | (sdf_reshaped <= -sdf_trunc)

    tsdf = (sdf_reshaped / sdf_trunc + 1) / 2
    tsdf[mask] = 0
    return tsdf


def compute_chamfer_and_iou_messy_kitchen(target_mesh, completed_mesh):
    """Compute Chamfer distance, IoU, and B-IoU between target mesh and completed mesh for messy kitchen scenarios.
    
    This function directly uses the input meshes without any normalization.
    
    Args:
        target_mesh (trimesh.Trimesh): Ground truth target mesh
        completed_mesh (trimesh.Trimesh): Completed mesh
        
    Returns:
        tuple: (chamfer_distance, iou, b_iou)
    """
    # Sample points from mesh surfaces for Chamfer distance
    gt_points, _ = trimesh.sample.sample_surface(target_mesh, 2048)
    completed_points, _ = trimesh.sample.sample_surface(completed_mesh, 2048)
    
    evaluator = MeshEvaluator(n_points=2048)
    
    # Compute Chamfer distance
    eval_dict = evaluator.eval_pointcloud(
        pointcloud=completed_points,
        pointcloud_tgt=gt_points,
        normals=None,
        normals_tgt=None
    )
    chamfer_distance = eval_dict['chamfer-L1']
    fscore = eval_dict['f-score']
    
    # Sample points for IoU calculation
    points_iou, occ_target = sample_iou_points(
        mesh_list=[target_mesh],
        bounds=target_mesh.bounds,
        num_point=100000,
        uniform=False,
        size=1.0
    )
    
    # Compute IoU
    metrics = evaluator.eval_mesh(
        mesh=completed_mesh,
        pointcloud_tgt=gt_points,
        normals_tgt=None,
        points_iou=points_iou,
        occ_tgt=occ_target,
        remove_wall=False
    )
    iou = metrics['iou']
    
    # Compute B-IoU
    b_iou = compute_bounding_box_iou(target_mesh, completed_mesh)
    
    return chamfer_distance, iou, b_iou, fscore


def compute_multi_object_completion_metrics_messy_kitchen(gt_meshes, pred_meshes, num_samples=10000):
    """
    Compute evaluation metrics for multi-object reconstruction in messy kitchen scenarios.
    
    This function uses scene-level normalization and the messy kitchen specific evaluation functions.
    
    Args:
        gt_meshes: List of Ground Truth meshes (reference)
        pred_meshes: List of predicted meshes (already aligned if using GAPS)
        num_samples: Number of sample points for metric computation
        
    Returns:
        Dictionary containing various metrics and aligned scenes
    """
    print(f"GT scene contains {len(gt_meshes)} objects")
    print(f"Predicted scene contains {len(pred_meshes)} objects")
    
    if len(gt_meshes) == 0 or len(pred_meshes) == 0:
        print("Warning: No valid meshes found")
        return {
            'chamfer_distance': float('inf'),
            'iou': 0.0,
            'chamfer_distance_std': 0.0,
            'iou_std': 0.0,
            'per_object_cds': [],
            'per_object_ious': [],
            'per_object_b_ious': [],
            'per_object_fscores': [],
            'scene_iou': 0.0
        }
    
    # Use the provided meshes directly (alignment already done in eval.py)
    aligned_pred_meshes = pred_meshes
    
    # Step 4: Compute per-object metrics using messy kitchen specific functions
    print("Step 4: Computing per-object metrics for messy kitchen scenario...")
    per_object_cds = []
    per_object_ious = []
    per_object_b_ious = []
    per_object_fscores = []
    
    # For each predicted object, find best matching GT object and compute metrics
    for i, pred_mesh in enumerate(aligned_pred_meshes):
        best_cd = float('inf')
        best_iou = 0.0
        best_b_iou = 0.0
        best_fscore = 0.0
        best_gt_idx = -1
        
        for j, gt_mesh in enumerate(gt_meshes):
            try:
                # Use the messy kitchen specific compute_chamfer_and_iou function
                cd, iou, b_iou, fscore = compute_chamfer_and_iou_messy_kitchen(gt_mesh, pred_mesh)
                
                # Optional: Save meshes for debugging
                # gt_mesh.export(f"gt_{j}.ply", include_attributes=False)
                # pred_mesh.export(f"pred_{i}.ply", include_attributes=False)
                
                if cd < best_cd:
                    best_cd = cd
                    best_iou = iou
                    best_b_iou = b_iou
                    best_fscore = fscore
                    best_gt_idx = j
                    
            except Exception as e:
                print(f"Error computing metrics for pred object {i} and GT object {j}: {e}")
                continue
        
        if best_gt_idx >= 0:
            per_object_cds.append(best_cd)
            per_object_ious.append(best_iou)
            per_object_b_ious.append(best_b_iou)
            per_object_fscores.append(best_fscore)
            print(f"Pred object {i} matched with GT object {best_gt_idx}: CD={best_cd:.6f}, IoU={best_iou:.6f}, B-IoU={best_b_iou:.6f}, F-score={best_fscore:.6f}")
        else:
            per_object_cds.append(float('inf'))
            per_object_ious.append(0.0)
            per_object_b_ious.append(0.0)
            per_object_fscores.append(0.0)
            print(f"No match found for pred object {i}")
    
    # Compute scene-level metrics
    if per_object_cds:
        scene_cd = np.mean(per_object_cds)
        scene_iou = np.mean(per_object_ious)
        scene_b_iou = np.mean(per_object_b_ious)
        scene_fscore = np.mean(per_object_fscores)
        scene_cd_std = np.std(per_object_cds)
        scene_iou_std = np.std(per_object_ious)
        scene_b_iou_std = np.std(per_object_b_ious)
        scene_fscore_std = np.std(per_object_fscores)
    else:
        scene_cd = float('inf')
        scene_iou = 0.0
        scene_b_iou = 0.0
        scene_fscore = 0.0
        scene_cd_std = 0.0
        scene_iou_std = 0.0
        scene_b_iou_std = 0.0
        scene_fscore_std = 0.0
    
    # Create metrics dictionary
    metrics = {
        'chamfer_distance': float(scene_cd),
        'iou': float(scene_iou),
        'chamfer_distance_std': float(scene_cd_std),
        'iou_std': float(scene_iou_std),
        'per_object_cds': per_object_cds,
        'per_object_ious': per_object_ious,
        'per_object_b_ious': per_object_b_ious,
        'per_object_fscores': per_object_fscores,
        'scene_iou': float(scene_iou)
    }
    
    # Create aligned scenes for compatibility
    aligned_gt_scene = trimesh.Scene(gt_meshes)
    aligned_pred_scene = trimesh.Scene(aligned_pred_meshes)
    
    # Store aligned scenes for later use
    metrics['aligned_gt_scene'] = aligned_gt_scene
    metrics['aligned_pred_scene'] = aligned_pred_scene
    
    # Create merged meshes for compatibility
    if len(gt_meshes) > 1:
        metrics['aligned_gt_merged'] = trimesh.util.concatenate(gt_meshes)
    else:
        metrics['aligned_gt_merged'] = gt_meshes[0] if gt_meshes else None
    
    if len(aligned_pred_meshes) > 1:
        metrics['aligned_pred_merged'] = aligned_pred_meshes[0] if aligned_pred_meshes else None
    else:
        metrics['aligned_pred_merged'] = aligned_pred_meshes[0] if aligned_pred_meshes else None
    
    # Calculate B-IoU and F-score statistics
    if per_object_b_ious:
        scene_b_iou = np.mean(per_object_b_ious)
        scene_b_iou_std = np.std(per_object_b_ious)
    else:
        scene_b_iou = 0.0
        scene_b_iou_std = 0.0
    
    if per_object_fscores:
        scene_fscore = np.mean(per_object_fscores)
        scene_fscore_std = np.std(per_object_fscores)
    else:
        scene_fscore = 0.0
        scene_fscore_std = 0.0
    
    # Add B-IoU and F-score to metrics dictionary
    metrics['b_iou'] = float(scene_b_iou)
    metrics['b_iou_std'] = float(scene_b_iou_std)
    metrics['f_score'] = float(scene_fscore)
    metrics['f_score_std'] = float(scene_fscore_std)
    
    print(f"Final scene metrics (messy kitchen): CD={metrics['chamfer_distance']:.6f}±{metrics['chamfer_distance_std']:.6f}, "
          f"IoU={metrics['iou']:.6f}±{metrics['iou_std']:.6f}, B-IoU={metrics['b_iou']:.6f}±{metrics['b_iou_std']:.6f}, "
          f"F-score={metrics['f_score']:.6f}±{metrics['f_score_std']:.6f}")
    
    return metrics

def compute_bounding_box_iou(mesh1, mesh2):
    """
    Compute Bounding-box IoU (B-IoU) between two meshes.
    
    B-IoU measures the overlap between the bounding boxes of two meshes,
    which is useful for evaluating spatial alignment and coverage.
    
    Args:
        mesh1 (trimesh.Trimesh): First mesh
        mesh2 (trimesh.Trimesh): Second mesh
        
    Returns:
        float: Bounding-box IoU value between 0 and 1
    """
    try:
        # Get bounding boxes of both meshes
        bbox1 = mesh1.bounds  # Shape: (2, 3) - (min_coords, max_coords)
        bbox2 = mesh2.bounds
        
        # Calculate intersection bounds
        intersection_min = np.maximum(bbox1[0], bbox2[0])
        intersection_max = np.minimum(bbox1[1], bbox2[1])
        
        # Check if there's an intersection
        if np.any(intersection_min > intersection_max):
            return 0.0
        
        # Calculate intersection volume
        intersection_dims = intersection_max - intersection_min
        intersection_volume = np.prod(intersection_dims)
        
        # Calculate individual volumes
        bbox1_dims = bbox1[1] - bbox1[0]
        bbox2_dims = bbox2[1] - bbox2[0]
        bbox1_volume = np.prod(bbox1_dims)
        bbox2_volume = np.prod(bbox2_dims)
        
        # Calculate union volume
        union_volume = bbox1_volume + bbox2_volume - intersection_volume
        
        # Calculate B-IoU
        b_iou = intersection_volume / union_volume if union_volume > 0 else 0.0
        
        return float(b_iou)
        
    except Exception as e:
        print(f"Error computing B-IoU: {e}")
        return 0.0


# def compute_chamfer_and_iou_messy_kitchen(target_mesh, completed_mesh):
#     """Compute Chamfer distance and IoU between target mesh and completed mesh for messy kitchen scenarios.
    
#     This function directly uses the input meshes without any normalization.
    
#     Args:
#         target_mesh (trimesh.Trimesh): Ground truth target mesh
#         completed_mesh (trimesh.Trimesh): Completed mesh
        
#     Returns:
#         tuple: (chamfer_distance, iou)
#     """
#     # Sample points from mesh surfaces for Chamfer distance
#     gt_points, _ = trimesh.sample.sample_surface(target_mesh, 2048)
#     completed_points, _ = trimesh.sample.sample_surface(completed_mesh, 2048)
    
#     evaluator = MeshEvaluator(n_points=2048)
    
#     # Compute Chamfer distance
#     eval_dict = evaluator.eval_pointcloud(
#         pointcloud=completed_points,
#         pointcloud_tgt=gt_points,
#         normals=None,
#         normals_tgt=None
#     )
#     chamfer_distance = eval_dict['chamfer-L1']
#     # fscore = eval_dict['fscore']
    
#     # Sample points for IoU calculation
#     points_iou, occ_target = sample_iou_points(
#         mesh_list=[target_mesh],
#         bounds=target_mesh.bounds,
#         num_point=100000,
#         uniform=False,
#         size=1.0
#     )
    
#     # Compute IoU
#     metrics = evaluator.eval_mesh(
#         mesh=completed_mesh,
#         pointcloud_tgt=gt_points,
#         normals_tgt=None,
#         points_iou=points_iou,
#         occ_tgt=occ_target,
#         remove_wall=False
#     )
#     iou = metrics['iou']
    
#     return chamfer_distance, iou
