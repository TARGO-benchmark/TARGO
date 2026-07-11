from math import cos, sin
import time

import numpy as np
import open3d as o3d

from src.vgn.utils.transform import Transform


class CameraIntrinsic(object):
    """Intrinsic parameters of a pinhole camera model.

    Attributes:
        width (int): The width in pixels of the camera.
        height(int): The height in pixels of the camera.
        K: The intrinsic camera matrix.
    """

    def __init__(self, width, height, fx, fy, cx, cy):
        self.width = width
        self.height = height
        self.K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    @property
    def fx(self):
        return self.K[0, 0]

    @property
    def fy(self):
        return self.K[1, 1]

    @property
    def cx(self):
        return self.K[0, 2]

    @property
    def cy(self):
        return self.K[1, 2]

    def to_dict(self):
        """Serialize intrinsic parameters to a dict object."""
        data = {
            "width": self.width,
            "height": self.height,
            "K": self.K.flatten().tolist(),
        }
        return data

    @classmethod
    def from_dict(cls, data):
        """Deserialize intrinisic parameters from a dict object."""
        intrinsic = cls(
            width=data["width"],
            height=data["height"],
            fx=data["K"][0],
            fy=data["K"][4],
            cx=data["K"][2],
            cy=data["K"][5],
        )
        return intrinsic


class TSDFVolume(object):
    """Integration of multiple depth images using a TSDF."""

    def __init__(self, size, resolution):
        self.size = size
        self.resolution = resolution
        self.voxel_size = self.size / self.resolution
        self.sdf_trunc = 4 * self.voxel_size

        self._volume = o3d.pipelines.integration.UniformTSDFVolume(
            length=self.size,
            resolution=self.resolution,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
        )

    def integrate(self, depth_img, intrinsic, extrinsic):
        """
        Args:
            depth_img: The depth image.
            intrinsic: The intrinsic parameters of a pinhole camera model.
            extrinsics: The transform from the TSDF to camera coordinates, T_eye_task.
        """
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.empty_like(depth_img)),
            o3d.geometry.Image(depth_img),
            depth_scale=1.0,
            depth_trunc=2.0,
            convert_rgb_to_intensity=False,
        )

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic.width,
            height=intrinsic.height,
            fx=intrinsic.fx,
            fy=intrinsic.fy,
            cx=intrinsic.cx,
            cy=intrinsic.cy,
        )

        extrinsic = extrinsic.as_matrix()

        self._volume.integrate(rgbd, intrinsic, extrinsic)

    def get_grid(self):
        # TODO(mbreyer) very slow (~35 ms / 50 ms of the whole pipeline)
        # shape = (1, self.resolution, self.resolution, self.resolution)
        # tsdf_grid = np.zeros(shape, dtype=np.float32)
        # voxels = self._volume.extract_voxel_grid().get_voxels()
        # for voxel in voxels:
        #     i, j, k = voxel.grid_index
        #     tsdf_grid[0, i, j, k] = voxel.color[0]
        shape = (1, self.resolution, self.resolution, self.resolution)
        tsdf_grid = np.zeros(shape, dtype=np.float32)
        tsdf_vol = np.asarray(self._volume.extract_volume_tsdf()).astype(np.float32)
        tsdf_vals = np.copy(tsdf_vol[:, 0]).reshape(shape)
        tsdf_weights = np.copy(tsdf_vol[:, 1]).reshape(shape)
        tsdf_grid = np.where(
            (tsdf_weights != 0.0) & (tsdf_vals < 0.98) & (tsdf_vals >= -0.98),
            (tsdf_vals + 1.0) * 0.5,
            tsdf_grid,
        )
        return tsdf_grid

    def get_cloud(self):
        return self._volume.extract_point_cloud()


def create_tsdf(size, resolution, depth_imgs, intrinsic, extrinsics):
    tsdf = TSDFVolume(size, resolution)
    for i in range(depth_imgs.shape[0]):
        # extrinsic = Transform.from_matrix(extrinsics[i])
        extrinsic = Transform.from_list(extrinsics[i])
        tsdf.integrate(depth_imgs[i], intrinsic, extrinsic)
    return tsdf


def camera_on_sphere(origin, radius, theta, phi):
    eye = np.r_[
        radius * sin(theta) * cos(phi),
        radius * sin(theta) * sin(phi),
        radius * cos(theta),
    ]
    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])  # this breaks when looking straight down
    return Transform.look_at(eye, target, up) * origin.inverse()
    

def reconstruct_40_pc(sim, depth_imgs, extrinsics):
    tsdf = create_tsdf(sim.size, 40, depth_imgs, sim.camera.intrinsic, extrinsics)
    pc = tsdf.get_cloud()

    # crop surface and borders from point cloud
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(sim.lower, sim.upper)
    pc = pc.crop(bounding_box)
    if False:
        o3d.visualization.draw_geometries([pc])
    return pc


def duplicate_points(points, target_size):
    """Duplicate points to reach target size"""
    repeated_points = points
    while len(repeated_points) < target_size:
        additional_points = points[:min(len(points), target_size - len(repeated_points))]
        repeated_points = np.vstack((repeated_points, additional_points))
    return repeated_points


def farthest_point_sampling(points, num_samples):
    """Farthest point sampling"""
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


def specify_num_points(points, target_size):
    """Specify number of points by duplicating or sampling"""
    if points.size == 0:
        print("No points in the scene")
        return points
    if points.shape[0] < target_size:
        points_specified_num = duplicate_points(points, target_size)
    elif points.shape[0] > target_size:
        points_specified_num = farthest_point_sampling(points, target_size)
    else:
        points_specified_num = points
    return points_specified_num


def depth_to_point_cloud(depth_img, mask_targ, intrinsics, extrinsics, num_points):
    """
    Convert a masked depth image into a point cloud using camera intrinsics and inverse extrinsics.
    """
    # Apply the target mask to the depth image
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
    points_camera_frame = np.vstack((x, y, z)).T
    points_camera_frame = points_camera_frame[z!=0]
    
    # Specify number of points
    points_camera_frame = specify_num_points(points_camera_frame, num_points)

    # Convert the camera coordinates to world coordinate
    extrinsic = Transform.from_list(extrinsics).inverse()
    points_transformed = np.array([extrinsic.transform_point(p) for p in points_camera_frame])
    
    return points_transformed


def depth_to_point_cloud_no_specify(depth_img, mask_targ, intrinsics, extrinsics):
    """
    Convert a masked depth image into a point cloud without specifying number of points.
    """
    # Apply the target mask to the depth image
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
    points_camera_frame = np.vstack((x, y, z)).T
    points_camera_frame = points_camera_frame[z!=0]

    # Convert the camera coordinates to world coordinate
    extrinsic = Transform.from_list(extrinsics).inverse()
    points_transformed = np.array([extrinsic.transform_point(p) for p in points_camera_frame])
    
    return points_transformed


def remove_A_from_B(A, B):
    """Remove points in A from B"""
    # Step 1: Use broadcasting to find matching points
    matches = np.all(A[:, np.newaxis] == B, axis=2)
    # Step 2: Identify points in B that are not in A
    unique_to_B = ~np.any(matches, axis=0)
    # Step 3: Filter B to keep only unique points
    B_unique = B[unique_to_B]
    return B_unique