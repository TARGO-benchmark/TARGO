import enum
import time
import torch
import dataclasses
import pathlib
import numpy as np
import rerun as rr
import spatialmath as sm
import open3d as o3d
from src.vgn.detection_implicit import VGNImplicit  # TARGO
from src.vgn.perception import TSDFVolume, create_tsdf
from src.vgn.ConvONets.conv_onet.generation import Generator3D

O_RESOLUTION = 40
O_SIZE = 0.3  # Meters --> I saw this somewhere in the repo
O_VOXEL_SIZE = O_SIZE / O_RESOLUTION


@dataclasses.dataclass
class State:
    tsdf: TSDFVolume


@dataclasses.dataclass
class CameraIntrinsic:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


class ModelType(enum.Enum):
    packed = enum.auto()  # objects are placed on the table at their canonical poses.
    pile = enum.auto()  # objects are dropped on the workspace with random poses.


class TARGOInference:
    def __init__(
        self,
        model_dir: pathlib.Path,
        camera_intrinsic: CameraIntrinsic,
        model_type: ModelType = ModelType.packed,
        vis: bool = False,
    ):
        """
        ## Additional install stuff
        pip install .
        pip install cython
        pip install torch-scatter -f https://data.pyg.org/whl/{$TORCH}+${CUDA}.html # (check with conda list | grep torch)
        python scripts/convonet_setup.py build_ext --inplace
        pip install urdfpy
        pip install networkx==2.5
        """
        model_name = f"giga_{model_type.name}.pt"
        model_path = model_dir / model_name
        self.giga_model = VGNImplicit(
            model_path=model_path,
            model_type="giga",
            visualize=vis,
            # All arguments below are copied from the default list/arguments form the readme
            best=True,
            qual_th=0.75,
            force_detection=True,
            out_th=0.1,
            select_top=False,
        )
        self.camera_intrinsic = camera_intrinsic
        self.giga_model.net.eval()
        self.generator = Generator3D(
            self.giga_model.net,
            device=self.giga_model.device,
            input_type="pointcloud",
            padding=0,
        )
        self.O_SIZE = O_SIZE
        return

    def predict(self, depth: np.ndarray, camTtask_np: np.ndarray, reconstruction: bool = False):
        assert camTtask_np.shape == (4, 4)

        start_time = time.time()
        tsdf_volume = create_tsdf(
            size=O_SIZE,
            resolution=O_RESOLUTION,
            depth_imgs=np.expand_dims(depth, axis=0),
            intrinsic=self.camera_intrinsic,
            extrinsics=np.expand_dims(camTtask_np, axis=0),
        )

        state = State(tsdf=tsdf_volume)
        grasps, scores, toc = self.giga_model(state)
        inference_time = time.time() - start_time

        if reconstruction:
            pc_torch = torch.tensor(tsdf_volume.get_grid())
            pred_mesh, _ = self.generator.generate_mesh({"inputs": pc_torch})
        else:
            pred_mesh = None
        return grasps, scores, inference_time, tsdf_volume.get_cloud(), pred_mesh

    def visualize(self, grasp_mesh_list, wTcam, wTtask, tsdf_pc, rgb, depth, pred_mesh):
        rr.log("giga", rr.Clear(recursive=True))
        rr.log("rgb", rr.Clear(recursive=True))
        rr.log("depth", rr.Clear(recursive=True))

        # Add images
        rr.log("rgb", rr.Image(rgb))
        rr.log("depth", rr.DepthImage(depth))

        # add_o3d_pointcloud
        o3d_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=self.camera_intrinsic.width,
            height=self.camera_intrinsic.height,
            fx=self.camera_intrinsic.fx,
            fy=self.camera_intrinsic.fy,
            cx=self.camera_intrinsic.cx,
            cy=self.camera_intrinsic.cy,
        )
        rgb_o3d = o3d.geometry.Image(rgb)
        depth_o3d = o3d.geometry.Image((depth * 1000).astype(np.uint16))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, convert_rgb_to_intensity=False
        )
        full_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d_camera_intrinsic,
        )
        full_pcd_w = full_pcd.transform(wTcam.A)
        points = np.asanyarray(full_pcd_w.points)
        colors = np.asanyarray(full_pcd_w.colors) if full_pcd_w.has_colors() else None
        colors_uint8 = (colors * 255).astype(np.uint8) if full_pcd_w.has_colors() else None
        rr.log("giga/full_pcd_w", rr.Points3D(positions=points, colors=colors_uint8, radii=0.001))

        # Add origin
        rr.log("giga/origin", rr.Arrows3D(
            origins=np.zeros((3, 3)),
            vectors=np.eye(3) * 0.1,
            colors=np.eye(3, dtype=int) * 255,
        ))

        # add input_tsdf
        w_T_input_tsdf_pc = tsdf_pc.transform(wTtask.A)
        rr.log(
            "giga/input_tsdf_pc", rr.Points3D(positions=np.asanyarray(w_T_input_tsdf_pc.points), radii=0.001)
        )

        # add workspace obb
        ws_half_size = np.array([O_SIZE, O_SIZE, O_SIZE]) / 2
        rr.log("giga/workspace", rr.Boxes3D(
            centers=wTtask.t + ws_half_size,
            half_sizes=ws_half_size
            )
        )

        # Add grasps
        for i, grasp_mesh in enumerate(grasp_mesh_list):
            grasp_mesh_w = grasp_mesh.apply_transform(wTtask.A)
            rr.log(
                f"giga/w_grasp_{i}",
                rr.Mesh3D(
                    vertex_positions=grasp_mesh_w.vertices,
                    vertex_normals=grasp_mesh_w.vertex_normals,
                    vertex_colors=grasp_mesh_w.visual.vertex_colors,
                    indices=grasp_mesh_w.faces,
                )
            )

        if pred_mesh is None:
            return
        # Add shape
        rr.log(
            "giga/shape",
            rr.Mesh3D(
                vertex_positions=pred_mesh.vertices,
                vertex_normals=pred_mesh.vertex_normals,
                vertex_colors=pred_mesh.visual.vertex_colors,
                indices=pred_mesh.faces,
            )
        )
        return


class TARGOInference:
    def __init__(
        self,
        model_dir: pathlib.Path,
        camera_intrinsic: CameraIntrinsic,
        model_type: ModelType = ModelType.packed,
        vis: bool = False,
    ):
        """
        ## Additional install stuff
        pip install .
        pip install cython
        pip install torch-scatter -f https://data.pyg.org/whl/{$TORCH}+${CUDA}.html # (check with conda list | grep torch)
        python scripts/convonet_setup.py build_ext --inplace
        pip install urdfpy
        pip install networkx==2.5
        """
        # model_name = f"giga_{model_type.name}.pt"
        model_name = "targonet.pt"
        model_path = model_dir / model_name
        self.targo_model = VGNImplicit(
            model_path=model_path,
            model_type="targo",
            visualize=vis,
            # All arguments below are copied from the default list/arguments form the readme
            best=True,
            qual_th=0.75,
            force_detection=True,
            out_th=0.1,
            select_top=False,
        )
        self.camera_intrinsic = camera_intrinsic
        self.targo_model.net.eval()
        self.generator = Generator3D(
            self.targo_model.net,
            device=self.targo_model.device,
            input_type="pointcloud",
            padding=0,
        )
        self.O_SIZE = O_SIZE
        return

    def predict(self, depth: np.ndarray, camTtask_np: np.ndarray, reconstruction: bool = False):
        assert camTtask_np.shape == (4, 4)

        start_time = time.time()
        tsdf_volume = create_tsdf(
            size=O_SIZE,
            resolution=O_RESOLUTION,
            depth_imgs=np.expand_dims(depth, axis=0),
            intrinsic=self.camera_intrinsic,
            extrinsics=np.expand_dims(camTtask_np, axis=0),
        )

        state = State(tsdf=tsdf_volume)
        grasps, scores, toc = self.targo_model(state)
        inference_time = time.time() - start_time

        if reconstruction:
            pc_torch = torch.tensor(tsdf_volume.get_grid())
            pred_mesh, _ = self.generator.generate_mesh({"inputs": pc_torch})
        else:
            pred_mesh = None
        return grasps, scores, inference_time, tsdf_volume.get_cloud(), pred_mesh

    def visualize(self, grasp_mesh_list, wTcam, wTtask, tsdf_pc, rgb, depth, pred_mesh):
        rr.log("giga", rr.Clear(recursive=True))
        rr.log("rgb", rr.Clear(recursive=True))
        rr.log("depth", rr.Clear(recursive=True))

        # Add images
        rr.log("rgb", rr.Image(rgb))
        rr.log("depth", rr.DepthImage(depth))

        # add_o3d_pointcloud
        o3d_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=self.camera_intrinsic.width,
            height=self.camera_intrinsic.height,
            fx=self.camera_intrinsic.fx,
            fy=self.camera_intrinsic.fy,
            cx=self.camera_intrinsic.cx,
            cy=self.camera_intrinsic.cy,
        )
        rgb_o3d = o3d.geometry.Image(rgb)
        depth_o3d = o3d.geometry.Image((depth * 1000).astype(np.uint16))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, convert_rgb_to_intensity=False
        )
        full_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d_camera_intrinsic,
        )
        full_pcd_w = full_pcd.transform(wTcam.A)
        points = np.asanyarray(full_pcd_w.points)
        colors = np.asanyarray(full_pcd_w.colors) if full_pcd_w.has_colors() else None
        colors_uint8 = (colors * 255).astype(np.uint8) if full_pcd_w.has_colors() else None
        rr.log("giga/full_pcd_w", rr.Points3D(positions=points, colors=colors_uint8, radii=0.001))

        # Add origin
        rr.log("giga/origin", rr.Arrows3D(
            origins=np.zeros((3, 3)),
            vectors=np.eye(3) * 0.1,
            colors=np.eye(3, dtype=int) * 255,
        ))

        # add input_tsdf
        w_T_input_tsdf_pc = tsdf_pc.transform(wTtask.A)
        rr.log(
            "giga/input_tsdf_pc", rr.Points3D(positions=np.asanyarray(w_T_input_tsdf_pc.points), radii=0.001)
        )

        # add workspace obb
        ws_half_size = np.array([O_SIZE, O_SIZE, O_SIZE]) / 2
        rr.log("giga/workspace", rr.Boxes3D(
            centers=wTtask.t + ws_half_size,
            half_sizes=ws_half_size
            )
        )

        # Add grasps
        for i, grasp_mesh in enumerate(grasp_mesh_list):
            grasp_mesh_w = grasp_mesh.apply_transform(wTtask.A)
            rr.log(
                f"giga/w_grasp_{i}",
                rr.Mesh3D(
                    vertex_positions=grasp_mesh_w.vertices,
                    vertex_normals=grasp_mesh_w.vertex_normals,
                    vertex_colors=grasp_mesh_w.visual.vertex_colors,
                    indices=grasp_mesh_w.faces,
                )
            )

        if pred_mesh is None:
            return
        # Add shape
        rr.log(
            "giga/shape",
            rr.Mesh3D(
                vertex_positions=pred_mesh.vertices,
                vertex_normals=pred_mesh.vertex_normals,
                vertex_colors=pred_mesh.visual.vertex_colors,
                indices=pred_mesh.faces,
            )
        )
        return