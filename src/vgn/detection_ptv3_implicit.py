import time
import numpy as np
import trimesh
from scipy import ndimage
import torch
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.vgn.grasp import *
from src.vgn.utils.transform import Transform, Rotation
from src.vgn.networks import load_network
from src.vgn.utils import visual
from src.utils_giga import *
from src.utils_targo import tsdf_to_mesh, filter_and_pad_point_clouds, save_point_cloud_as_ply
import pyvista as pv  

# Import pyrender for rendering
try:
    import pyrender
    from PIL import Image
    PYRENDER_AVAILABLE = True
    print("✓ pyrender package available for rendering")
except ImportError:
    PYRENDER_AVAILABLE = False
    print("✗ pyrender package not available, rendering will be skipped")

LOW_TH = 0.0

# import os
# import numpy as np
# import trimesh
# import pyvista as pv

def render_colored_scene_mesh_with_pyvista(colored_scene_mesh,
                                           output_path="demo/ptv3_scene_affordance_visual.png",
                                           width=640, height=480):
    """
    Render a trimesh object using PyVista with original color, white background, no text or colorbar.
    """
    try:
        if colored_scene_mesh.vertices.size == 0 or colored_scene_mesh.faces.size == 0:
            print("Mesh is empty!")
            return False

        # 获取 mesh 原始颜色
        face_colors = getattr(colored_scene_mesh.visual, "face_colors", None)

        # 转为 PyVista mesh
        faces_flat = np.hstack(
            np.c_[np.full(len(colored_scene_mesh.faces), 3),
                  colored_scene_mesh.faces]
        ).astype(np.int64).ravel()
        pv_mesh = pv.PolyData(colored_scene_mesh.vertices, faces_flat)

        if face_colors is not None:
            pv_mesh.cell_data["colors"] = face_colors
            pv_mesh.cell_data.active_scalars_name = "colors"

        # 创建渲染器
        plotter = pv.Plotter(off_screen=True, window_size=(width, height))
        plotter.add_mesh(
            pv_mesh,
            show_edges=False,
            show_scalar_bar=False  # ✅ 关闭颜色条
        )
        plotter.set_background("white")

        # 设置相机（远视角）
        center = pv_mesh.center
        bounds = pv_mesh.bounds
        extent = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
        iso_dir = np.array([1, 1, 1]) / np.sqrt(3)
        camera_pos = center + iso_dir * extent * 4

        plotter.camera.position = camera_pos.tolist()
        plotter.camera.focal_point = list(center)
        plotter.camera.up = [0, 0, 1]

        # 渲染并保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plotter.screenshot(output_path)
        plotter.close()

        return True

    except Exception as e:
        print(f"Error rendering with PyVista: {e}")
        import traceback; traceback.print_exc()
        return False

    
# def render_colored_scene_mesh_with_pyrender(colored_scene_mesh, output_path="demo/ptv3_scene_affordance_visual.png", 
#                                           width=800, height=600, camera_distance=0.5):
#     """
#     Render colored scene mesh using pyrender package and save it as an image.
    
#     Args:
#         colored_scene_mesh: trimesh object
#         output_path: path to save the rendered image
#         width: image width
#         height: image height
#         camera_distance: distance from camera to scene center
#     """
#     if not PYRENDER_AVAILABLE:
#         print("pyrender not available, skipping rendering")
#         return False
    
#     try:
#         # Ensure the mesh has visual properties
#         if not hasattr(colored_scene_mesh, 'visual') or colored_scene_mesh.visual is None:
#             colored_scene_mesh.visual = trimesh.visual.ColorVisuals()
        
#         # Set all mesh faces to blue color (RGBA) if not already colored
#         if not hasattr(colored_scene_mesh.visual, 'face_colors') or colored_scene_mesh.visual.face_colors is None:
#             colored_scene_mesh.visual.face_colors = np.full((len(colored_scene_mesh.faces), 4), [0, 0, 255, 255], dtype=np.uint8)

#         # Create pyrender scene
#         scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[255, 255, 255])

#         # Add the colored mesh to the scene
#         mesh_render = pyrender.Mesh.from_trimesh(colored_scene_mesh, smooth=False)
#         scene.add(mesh_render)

#         # Set up camera with better positioning
#         camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=float(width) / height)
        
#         # Calculate camera position based on mesh bounds
#         bounds = colored_scene_mesh.bounds
#         center = (bounds[0] + bounds[1]) / 2
#         extent = np.linalg.norm(bounds[1] - bounds[0])
#         camera_distance = max(camera_distance, extent * 1.5)
        
#         camera_pose = np.eye(4)
#         camera_pose[:3, 3] = center + [camera_distance, camera_distance, camera_distance]
#         scene.add(camera, pose=camera_pose)

#         # Set up lighting
#         light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
#         light_pose = np.eye(4)
#         light_pose[:3, 3] = center + [0, -camera_distance, camera_distance]
#         scene.add(light, pose=light_pose)
        
#         # Add ambient light
#         ambient_light = pyrender.AmbientLight(color=[0.5, 0.5, 0.5], intensity=0.5)
#         scene.add(ambient_light)

#         # Render the scene
#         renderer = pyrender.OffscreenRenderer(width, height)
#         color, depth = renderer.render(scene)
        
#         # Clean up
#         renderer.delete()

#         # Ensure output directory exists
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)

#         # Save the rendered image
#         image = Image.fromarray(color)
#         image.save(output_path)
#         print(f"Rendered image saved to: {output_path}")
#         print(f"Image size: {image.size}, Color range: {color.min()}-{color.max()}")
#         return True

#     except Exception as e:
#         print(f"Error rendering with pyrender: {e}")
#         import traceback
#         traceback.print_exc()
#         return False

def predict_ptv3_clip(inputs, pos, net, device, visual_dict=None, state=None, target_mesh_gt=None):
    """
    Predict function specialized for ptv3_clip model type.
    
    Args:
        inputs: tuple of (scene_point_cloud, None) where scene_point_cloud includes labels
        pos: position tensor for queries
        net: ptv3_scene network model
        device: torch device
        visual_dict: dictionary for visualization data
        state: state object containing preprocessed complete target data
        target_mesh_gt: ground truth target mesh (fallback option)
        
    Returns:
        qual_vol, rot_vol, width_vol, completed_targ_grid, cd, iou
    """
    # scene_no_targ_pc = torch.from_numpy(inputs[0]).unsqueeze(0).to(device)
    # full_scene_with_labels = inputs[0]

    with torch.no_grad():
        # For ptv3_scene: directly use preprocessed complete target data
        start_pc = time.time()
        # Read complete target TSDF if available
        # if hasattr(state, 'complete_target_tsdf') and state.complete_target_tsdf is not None:
        #     # Use preprocessed complete target TSDF
        completed_targ_grid = state.complete_targ_tsdf
    
        # Perfect reconstruction means CD and IoU are ideal (since using ground truth/preprocessed data)
        cd = state.cd
        iou = state.iou
        
        # completed_targ_pc  = state.complete_targ_pc
        
        
        # save_point_cloud_as_ply(targ_completed_scene_pc[0].cpu().numpy(), 'targ_completed_scene_pc.ply')
        
        convert_time = time.time() - start_pc
        print(f"convert: {convert_time:.3f}s")

        scene_pc_tensor = torch.from_numpy(inputs[0]).unsqueeze(0).to(device)
        model_inputs = scene_pc_tensor
        scene_pc_feat_tensor = torch.from_numpy(inputs[1]).unsqueeze(0).to(device)
        model_inputs = (scene_pc_tensor, scene_pc_feat_tensor)
        # save_point_cloud_as_ply(scene_pc_tensor_cpu[:, :,:3][0], 'scene_pc_tensor.ply')
        
        # For ptv3_scene, only pass scene point cloud
        # model_inputs = (full_scene_with_labels, None)
        # full_scene_with_labels_tensor = torch.from_numpy(full_scene_with_labels).unsqueeze(0).to(device)
        # model_inputs = full_scene_with_labels_tensor
        # save_point_cloud_as_ply(full_scene_with_labels_tensor[:, :,:3][0].cpu().numpy(), 'full_scene_with_labels_tensor.ply')
    
    time_grasp_start = time.time()
    with torch.no_grad():
        qual_vol, rot_vol, width_vol = net(model_inputs, pos)
        
        net_params_count = sum(p.numel() for p in net.parameters())
        print(f"Number of parameters in self.net: {net_params_count:,}")
    
    time_grasp = time.time() - time_grasp_start
    print(f"Grasp prediction time: {time_grasp:.3f}s")
    
    # Move output back to the CPU
    qual_vol = qual_vol.cpu().squeeze().numpy()
    rot_vol = rot_vol.cpu().squeeze().numpy()
    width_vol = width_vol.cpu().squeeze().numpy()
    
    return qual_vol, rot_vol, width_vol, completed_targ_grid, cd, iou


def predict_ptv3_scene(inputs, pos, net, device, visual_dict=None, state=None, target_mesh_gt=None):
    """
    Predict function specialized for ptv3_scene model type.
    
    Args:
        inputs: tuple of (scene_point_cloud, None) where scene_point_cloud includes labels
        pos: position tensor for queries
        net: ptv3_scene network model
        device: torch device
        visual_dict: dictionary for visualization data
        state: state object containing preprocessed complete target data
        target_mesh_gt: ground truth target mesh (fallback option)
        
    Returns:
        qual_vol, rot_vol, width_vol, completed_targ_grid, cd, iou
    """
    # scene_no_targ_pc = torch.from_numpy(inputs[0]).unsqueeze(0).to(device)
    full_scene_with_labels = inputs[0]

    with torch.no_grad():
        # For ptv3_scene: directly use preprocessed complete target data
        start_pc = time.time()
        # Read complete target TSDF if available
        # if hasattr(state, 'complete_target_tsdf') and state.complete_target_tsdf is not None:
        #     # Use preprocessed complete target TSDF
        completed_targ_grid = state.complete_targ_tsdf
    
        # Perfect reconstruction means CD and IoU are ideal (since using ground truth/preprocessed data)
        cd = state.cd
        iou = state.iou
        
        # completed_targ_pc  = state.complete_targ_pc
        
        
        # save_point_cloud_as_ply(targ_completed_scene_pc[0].cpu().numpy(), 'targ_completed_scene_pc.ply')
        
        convert_time = time.time() - start_pc
        print(f"convert: {convert_time:.3f}s")
        
        # For ptv3_scene, only pass scene point cloud
        # model_inputs = (full_scene_with_labels, None)
        full_scene_with_labels_tensor = torch.from_numpy(full_scene_with_labels).unsqueeze(0).to(device)
        model_inputs = full_scene_with_labels_tensor
        save_point_cloud_as_ply(full_scene_with_labels_tensor[:, :,:3][0].cpu().numpy(), 'full_scene_with_labels_tensor.ply')
    
    time_grasp_start = time.time()
    with torch.no_grad():
        qual_vol, rot_vol, width_vol = net(model_inputs, pos)
        
        net_params_count = sum(p.numel() for p in net.parameters())
        print(f"Number of parameters in self.net: {net_params_count:,}")
    
    time_grasp = time.time() - time_grasp_start
    print(f"Grasp prediction time: {time_grasp:.3f}s")
    
    # Move output back to the CPU
    qual_vol = qual_vol.cpu().squeeze().numpy()
    rot_vol = rot_vol.cpu().squeeze().numpy()
    width_vol = width_vol.cpu().squeeze().numpy()
    
    return qual_vol, rot_vol, width_vol, completed_targ_grid, cd, iou


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
    """Process prediction volumes for ptv3_scene model."""
    # Check if tsdf_vol is a tuple
    if isinstance(tsdf_vol, tuple):
        if len(tsdf_vol) == 2:
            tsdf_vol = tsdf_vol[0]
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
    """Select grasps from prediction volumes."""
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
    """Select grasp at specific index."""
    i, j, k = index
    score = qual_vol[i, j, k]
    ori = Rotation.from_quat(rot_vol[i, j, k])
    pos = center_vol[i, j, k].numpy()
    width = width_vol[i, j, k]
    return Grasp(Transform(ori, pos), width), score


def bound(qual_vol, voxel_size, limit=[0.02, 0.02, 0.055]):
    """Avoid grasp out of bound."""
    # avoid grasp out of bound
    x, y, z = np.mgrid[:qual_vol.shape[0], :qual_vol.shape[1], :qual_vol.shape[2]]
    xyz = np.vstack([x.reshape(-1), y.reshape(-1), z.reshape(-1)]).T 
    xyz = xyz * voxel_size  + limit
    # xyz = xyz * voxel_size  + [0.02, 0.02, 0.055]
    # filter out all grasps outside of the boundary
    # mask = np.logical_and(0.02 <= xyz[:, 0], xyz[:, 0] <= 0.28)
    # mask = np.logical_and(mask, np.logical_and(0.02 <= xyz[:, 1], xyz[:, 1] <= 0.28))
    # mask = np.logical_and(mask, np.logical_and(0.055 <= xyz[:, 2], xyz[:, 2] <= 0.30))
    mask = np.logical_and(limit[0] <= xyz[:, 0], xyz[:, 0] <= 0.28)
    mask = np.logical_and(mask, np.logical_and(limit[1] <= xyz[:, 1], xyz[:, 1] <= 0.28))
    mask = np.logical_and(mask, np.logical_and(limit[2] <= xyz[:, 2], xyz[:, 2] <= 0.30))
    mask = mask.reshape(qual_vol.shape)
    qual_vol[~mask] = 0.0
    return qual_vol


def render_scene_multiview(
        scene,
        output_dir="demo/multiview",          # 目录而不是单一路径
        views=("iso", "front", "right", "top"),
        width=640, height=480,
        combine=True,                         # True → 最后拼成一张
):
    """
    渲染 trimesh.Scene 到多张视角图像。
    views 取值:
        'iso'   : 等距视角 (1,1,1)
        'front' : -Z 方向看
        'back'  : +Z
        'left'  : +X
        'right' : -X
        'top'   : +Y
        'bottom': -Y
    """
    if len(scene.geometry) == 0:
        raise ValueError("Scene is empty")

    # ---------- 集中整理所有顶点 ----------
    all_vertices = [g.vertices for g in scene.geometry.values()
                    if hasattr(g, 'vertices') and g.vertices.size > 0]
    all_vertices = np.vstack(all_vertices)
    center = all_vertices.mean(axis=0)
    extent = (all_vertices.max(axis=0) - all_vertices.min(axis=0)).max()

    # ---------- 准备输出 ----------
    os.makedirs(output_dir, exist_ok=True)
    out_paths = []

    # ---------- 视角到方向向量 ----------
    dir_map = dict(
        iso=np.array([1, 1, 1]),
        front=np.array([0, 0, -1]),
        back=np.array([0, 0, 1]),
        left=np.array([1, 0, 0]),
        right=np.array([-1, 0, 0]),
        top=np.array([0, 1, 0]),
        bottom=np.array([0, -1, 0]),
    )

    # ---------- 渲染循环 ----------
    for v in views:
        direction = dir_map[v] / np.linalg.norm(dir_map[v])
        cam_pos = center + direction * extent * 4

        plotter = pv.Plotter(off_screen=True, window_size=(width, height))
        _add_scene_to_plotter(scene, plotter)   # 复用你原来的函数片段
        plotter.set_background("white")
        plotter.camera.position = cam_pos.tolist()
        plotter.camera.focal_point = center.tolist()
        plotter.camera.up = [0, 0, 1]

        out_path = os.path.join(output_dir, f"{v}.png")
        plotter.screenshot(out_path)
        plotter.close()

        out_paths.append(out_path)

    # ---------- 可选：拼图 ----------
    if combine and out_paths:
        imgs = [Image.open(p) for p in out_paths]
        # 这里示例横向拼接，你可改成 2×2 九宫格
        combined = Image.new("RGB", (width * len(imgs), height), "white")
        for i, im in enumerate(imgs):
            combined.paste(im, (i * width, 0))
        combined_path = os.path.join(output_dir, "combined.png")
        combined.save(combined_path)
        out_paths.append(combined_path)

    return out_paths


# ---- 把你原先加 mesh 的片段抽成一个私有函数，便于复用 ----
def _add_scene_to_plotter(scene, plotter):
    for geometry in scene.geometry.values():
        if not (hasattr(geometry, "vertices") and hasattr(geometry, "faces")):
            continue
        if geometry.vertices.size == 0 or geometry.faces.size == 0:
            continue

        faces_flat = np.hstack([
            np.full(len(geometry.faces), 3, dtype=np.int64)[:, None],
            geometry.faces.astype(np.int64)
        ]).ravel()

        pv_mesh = pv.PolyData(geometry.vertices, faces_flat)

        fc = getattr(geometry.visual, "face_colors", None)
        if fc is not None:
            # 确保颜色值在正确范围内 [0, 255]
            if fc.dtype != np.uint8:
                fc = fc.astype(np.uint8)
            pv_mesh.cell_data["colors"] = fc
            pv_mesh.cell_data.active_scalars_name = "colors"
            print(f"Added mesh with colors: {fc[0] if len(fc) > 0 else 'No colors'}")
            
            # 使用正确的PyVista参数来渲染颜色
            plotter.add_mesh(pv_mesh, show_edges=False, show_scalar_bar=False, scalars="colors")
        else:
            # 如果没有颜色，使用默认渲染
            plotter.add_mesh(pv_mesh, show_edges=False, show_scalar_bar=False)

def render_scene_with_pyvista(scene,
                             output_path="demo/ptv3_scene_composed_visual.png",
                             width=640, height=480):
    """
    Render a trimesh.Scene object using PyVista with white background, no text or colorbar.
    """
    try:
        if len(scene.geometry) == 0:
            print("Scene is empty!")
            return False

        # 创建渲染器
        plotter = pv.Plotter(off_screen=True, window_size=(width, height))
        
        # 添加场景中的所有几何体
        for name, geometry in scene.geometry.items():
            if hasattr(geometry, 'vertices') and hasattr(geometry, 'faces'):
                if geometry.vertices.size > 0 and geometry.faces.size > 0:
                    # 获取 mesh 颜色
                    face_colors = getattr(geometry.visual, "face_colors", None)
                    
                    # 转为 PyVista mesh
                    faces_flat = np.hstack(
                        np.c_[np.full(len(geometry.faces), 3),
                              geometry.faces]
                    ).astype(np.int64).ravel()
                    pv_mesh = pv.PolyData(geometry.vertices, faces_flat)
                    
                    if face_colors is not None:
                        pv_mesh.cell_data["colors"] = face_colors
                        pv_mesh.cell_data.active_scalars_name = "colors"
                    
                    plotter.add_mesh(
                        pv_mesh,
                        show_edges=False,
                        show_scalar_bar=False,
                        scalars="colors" if face_colors is not None else None
                    )

        plotter.set_background("white")

        # 设置相机（远视角）
        # 计算场景的边界框
        all_vertices = []
        for geometry in scene.geometry.values():
            if hasattr(geometry, 'vertices') and geometry.vertices.size > 0:
                all_vertices.append(geometry.vertices)
        
        if all_vertices:
            all_vertices = np.vstack(all_vertices)
            center = np.mean(all_vertices, axis=0)
            bounds = [all_vertices[:, i].min() for i in range(3)] + [all_vertices[:, i].max() for i in range(3)]
            extent = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
            iso_dir = np.array([1, 1, 1]) / np.sqrt(3)
            camera_pos = center + iso_dir * extent * 4

            plotter.camera.position = camera_pos.tolist()
            plotter.camera.focal_point = list(center)
            plotter.camera.up = [0, 0, 1]

        # 渲染并保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plotter.screenshot(output_path)
        plotter.close()

        return True

    except Exception as e:
        print(f"Error rendering scene with PyVista: {e}")
        import traceback; traceback.print_exc()
        return False


class PTV3SceneImplicit(object):
    """
    Implicit grasp detection class specialized for ptv3_scene model type.
    This class directly uses preprocessed complete target point clouds and TSDF data.
    """
    
    def __init__(self, model_path, model_type, best=False, force_detection=False, 
                 qual_th=0.9, out_th=0.5, visualize=False, resolution=40, 
                 cd_iou_measure=False, **kwargs):
        """
        Initialize PTV3SceneImplicit detector.
        
        Args:
            model_path: path to the trained model
            model_type: must be 'ptv3_scene'
            best: whether to return only the best grasp
            force_detection: whether to force detection even with low quality
            qual_th: quality threshold for grasp selection
            out_th: output threshold for processing
            visualize: whether to enable visualization
            resolution: voxel grid resolution
            cd_iou_measure: whether to measure Chamfer Distance and IoU
        """
        assert model_type == 'ptv3_scene', f"PTV3SceneImplicit only supports ptv3_scene model type, got {model_type}"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_network(model_path, self.device, model_type=model_type) 
        self.net = self.net.eval()
        
        net_params_count = sum(p.numel() for p in self.net.parameters())
        print(f"Number of parameters in self.net: {net_params_count}")
        
        self.qual_th = qual_th
        self.best = best
        self.force_detection = force_detection
        self.out_th = out_th
        self.visualize = visualize
        self.resolution = resolution
        self.cd_iou_measure = cd_iou_measure
        
        # Create position tensor for query points
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

    def __call__(self, state, scene_mesh=None, visual_dict=None, hunyun2_path=None, 
                 scene_name=None, cd_iou_measure=False, target_mesh_gt=None, aff_kwargs={}):
        """
        Perform grasp detection on the given state.
        
        Args:
            state: state object containing scene data and preprocessed complete target data
                   Expected attributes: scene_no_targ_pc, targ_pc, complete_target_pc, complete_target_tsdf
            scene_mesh: scene mesh for visualization (optional)
            visual_dict: dictionary for visualization data (optional)
            hunyun2_path: path for hunyun2 data (unused for ptv3_scene)
            scene_name: scene name (optional)
            cd_iou_measure: whether to measure CD and IoU
            target_mesh_gt: ground truth target mesh (optional, used as fallback)
            aff_kwargs: affordance visualization arguments
            
        Returns:
            grasps, scores, inference_time, cd, iou
        """
        assert state.type == 'ptv3_scene', f"PTV3SceneImplicit only supports ptv3_scene model type, got {state.type}"
        
        visual_dict = {} if visual_dict is None else visual_dict
        print(state.__dict__.keys())

        # Handle ptv3_scene input format
        scene_no_targ_pc = state.scene_no_targ_pc
        scene_no_targ_pc = np.concatenate((scene_no_targ_pc, self.plane), axis=0)

        scene_no_targ_pc =specify_num_points(scene_no_targ_pc, 512)

        complete_targ_pc = state.complete_targ_pc
        complete_targ_pc = specify_num_points(complete_targ_pc, 512)

        full_scene_pc = np.concatenate((scene_no_targ_pc, complete_targ_pc), axis=0)
        scene_labels = np.zeros((len(scene_no_targ_pc), 1), dtype=np.float32)
        target_labels = np.ones((len(complete_targ_pc), 1), dtype=np.float32)
        scene_with_labels = np.concatenate([scene_no_targ_pc, scene_labels], axis=1)
        target_with_labels = np.concatenate([complete_targ_pc, target_labels], axis=1)
        full_scene_with_labels = np.concatenate([scene_with_labels, target_with_labels], axis=0)
        inputs = (full_scene_with_labels, None)

        # save_point_cloud_as_ply(full_scene_with_labels[:, :3], 'full_scene_with_labels.ply')
        
        voxel_size, size = state.tsdf.voxel_size, state.tsdf.size
        
        # Predict using ptv3_scene model with preprocessed complete target data
        with torch.no_grad():
            qual_vol, rot_vol, width_vol, completed_targ_grid, cd, iou = predict_ptv3_scene(
                inputs, self.pos, self.net, self.device, 
                visual_dict, state, target_mesh_gt
            )

        begin = time.time()

        # Reshape prediction volumes
        qual_vol = qual_vol.reshape((self.resolution, self.resolution, self.resolution))
        rot_vol = rot_vol.reshape((self.resolution, self.resolution, self.resolution, 4))
        width_vol = width_vol.reshape((self.resolution, self.resolution, self.resolution))

        # Process predictions with completed target grid
        if isinstance(completed_targ_grid, np.ndarray):
            # For numpy array: reshape from (40,40,40) to (1,40,40,40)
            completed_targ_grid = np.expand_dims(completed_targ_grid, axis=0)
        else:
            # For torch tensor: use squeeze and unsqueeze
            completed_targ_grid = completed_targ_grid.squeeze().unsqueeze(0)  # from (40,40,40) to (1,40,40,40)
        
        qual_vol, rot_vol, width_vol = process(completed_targ_grid, qual_vol, rot_vol, width_vol, out_th=self.out_th)
        
        if len(qual_vol.shape) == 1:
            qual_vol = qual_vol.reshape((self.resolution, self.resolution, self.resolution))
        qual_vol = bound(qual_vol, voxel_size)

        if self.visualize:
            colored_scene_mesh = visual.affordance_visual(qual_vol, rot_vol, scene_mesh, size, self.resolution, **aff_kwargs)
            visual_dict['affordance_visual'] = colored_scene_mesh

        # Select grasps from prediction volumes
        grasps, scores = select(
            qual_vol.copy(), 
            self.pos.view(self.resolution, self.resolution, self.resolution, 3).cpu(), 
            rot_vol, width_vol, 
            threshold=self.qual_th, 
            force_detection=self.force_detection, 
            max_filter_size=8 if self.visualize else 4
        )

        toc = time.time()
        grasps, scores = np.asarray(grasps), np.asarray(scores)

        # Transform grasps to world coordinates
        new_grasps = []
        if len(grasps) > 0:
            # Define random offsets for grasps
            offset_list = [
                Transform(Rotation.identity(), [0.0, 0.0, -0.02]),
                Transform(Rotation.identity(), [0.0, 0.0, -0.04]),
                Transform(Rotation.identity(), [0.0, 0.0, -0.03]),
                Transform(Rotation.identity(), [0.0, 0.0, -0.01]),
                Transform(Rotation.identity(), [0.0, 0.0, 0.01]),
                Transform(Rotation.identity(), [0.0, 0.0, 0.02]),
                Transform(Rotation.identity(), [0.0, 0.0, 0.03]),
                Transform(Rotation.identity(), [0.0, 0.0, 0.04])
            ]
            
            if self.best:
                p = np.arange(len(grasps))
            else:
                p = np.random.permutation(len(grasps))
            for g in grasps[p]:
                pose = g.pose
                pose.translation = (pose.translation + 0.5) * size
                width = g.width * size
                
                # Apply random offset
                random_offset = np.random.choice(offset_list)
                pose = pose * random_offset  # Apply offset transform
                
                new_grasps.append(Grasp(pose, width))
            scores = scores[p]
        grasps = new_grasps

        end = time.time()
        print(f"post processing: {end-begin:.3f}s")

        if visual_dict:
            grasp_mesh_list = [visual.grasp2mesh(g, s) for g, s in zip(grasps, scores)]
            
            # Create pre-grasp visualization for each grasp
            pre_grasp_mesh_list = []
            for i, grasp in enumerate(grasps):
                # Calculate pre-grasp position (5cm above grasp position)
                T_world_grasp = grasp.pose
                T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
                T_world_pregrasp = T_world_grasp * T_grasp_pregrasp
                
                # Create a Grasp object for pre-grasp position using the same width
                pregrasp_grasp = Grasp(T_world_pregrasp, grasp.width)
                
                # Create pre-grasp mesh using grasp2mesh with a dummy score
                pre_grasp_mesh = visual.grasp2mesh(pregrasp_grasp, 0.5)  # Use 0.5 as dummy score
                
                # Override color to yellow for pre-grasp
                yellow_color = np.array([255, 255, 0, 180]).astype(np.uint8)  # Yellow with transparency
                colors = np.repeat(yellow_color[np.newaxis, :], len(pre_grasp_mesh.faces), axis=0)
                pre_grasp_mesh.visual.face_colors = colors
                
                pre_grasp_mesh_list.append(pre_grasp_mesh)
            
            composed_scene = trimesh.Scene(colored_scene_mesh)
            
            # Add grasp meshes (green)
            for i, g_mesh in enumerate(grasp_mesh_list):
                composed_scene.add_geometry(g_mesh, node_name=f'grasp_{i}')
            
            # Add pre-grasp visualization (yellow)
            for i, pre_grasp_mesh in enumerate(pre_grasp_mesh_list):
                composed_scene.add_geometry(pre_grasp_mesh, node_name=f'pregrasp_{i}')
            
            visual_dict['composed_scene'] = composed_scene
            
            # Render composed scene with PyVista
            render_composed_success = render_scene_with_pyvista(
                composed_scene, 
                output_path=f"{state.vis_path}/ptv3_scene_composed_visual.png",
                width=800, 
                height=600
            )
            
            if render_composed_success:
                print("✓ Successfully rendered composed scene with PyVista")
            else:
                print("✗ Failed to render composed scene with PyVista")
            
            # Generate colored scene point cloud from RGB image if available
            if hasattr(state, 'vis_dict') and state.vis_dict is not None:
                try:
                    # Get RGB image and depth from vis_dict
                    if isinstance(state.vis_dict, np.ndarray) and state.vis_dict.dtype == object:
                        vis_data = state.vis_dict.item()
                    else:
                        vis_data = state.vis_dict
                    
                    # Check if we have RGB image (either direct or as file path)
                    rgb_img = None
                    depth_img = None
                    
                    if 'rgb_img' in vis_data:
                        rgb_img = vis_data['rgb_img']
                    elif 'scene_rgba_path' in vis_data:
                        # Load RGB image from file path
                        import cv2
                        rgb_img = cv2.imread(vis_data['scene_rgba_path'])
                        if rgb_img is not None:
                            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                    
                    if 'depth_img' in vis_data:
                        depth_img = vis_data['depth_img']
                        # Handle batch dimension if present
                        if len(depth_img.shape) == 3 and depth_img.shape[0] == 1:
                            depth_img = depth_img[0]  # Remove batch dimension
                    
                    if rgb_img is not None and depth_img is not None:
                        # scene_id = os.path.basename(state.vis_path).split('_')[0]
                        scene_id = os.path.basename(state.vis_path)
                        root_path = 'data_scenes/acronym/acronym-middle-occlusion-1000/scenes'
                        scene_path = os.path.join(root_path, f'{scene_id}.npz')
                        scene_file = np.load(scene_path)
                        extrinsics = scene_file['extrinsics']
                        mask_targ = scene_file['mask_targ']
                        mask_scene = scene_file['mask_scene']
                        camera_extrinsic = Transform.from_list(extrinsics[0]).as_matrix()
                        
                        # Get camera parameters from simulation or use defaults
                        # You may need to adjust these based on your actual camera setup
                        # Try to get camera intrinsics from vis_data if available, else use default
                        # if 'camera_intrinsic' in vis_data:
                        #     camera_intrinsic = np.array(vis_data['camera_intrinsic'])
                        # else:
                        camera_intrinsic = np.array([
                            [540.0, 0.0, 320.0],
                            [0.0, 540.0, 240.0],
                            [0.0, 0.0, 1.0]
                        ])
                        
                        # Generate full colored point cloud without sampling limit
                        full_colored_scene_pc = rgb_depth_to_colored_point_cloud(
                            rgb_img, depth_img, camera_intrinsic, camera_extrinsic, num_points=None
                        )
                        
                        # Use masks to separate target and scene_no_targ point clouds
                        if len(full_colored_scene_pc) > 0:
                            # Get image dimensions
                            height, width = depth_img.shape
                            
                            # Create pixel coordinate grids
                            u, v = np.meshgrid(np.arange(width), np.arange(height))
                            u, v = u.flatten(), v.flatten()
                            
                            # Apply masks to separate target and scene_no_targ
                            mask_targ_flat = mask_targ.flatten()
                            mask_scene_flat = mask_scene.flatten()
                            
                            # Get valid depth pixels
                            z = depth_img.flatten()
                            valid_mask = (z > 0) & (z < np.inf) & (z < 2.0)
                            
                            # Apply valid mask to coordinates and masks
                            u_valid = u[valid_mask]
                            v_valid = v[valid_mask]
                            mask_targ_valid = mask_targ_flat[valid_mask]
                            mask_scene_valid = mask_scene_flat[valid_mask]
                            
                            # Separate target and scene_no_targ points
                            target_indices = mask_targ_valid > 0
                            scene_no_targ_indices = (mask_scene_valid > 0) & (mask_targ_valid == 0)
                            
                            # Extract target point cloud
                            if target_indices.any():
                                target_colored_pc = full_colored_scene_pc[target_indices]
                                target_pc_path = f"{state.vis_path}/colored_target_point_cloud.ply"
                                save_colored_point_cloud_as_ply(target_colored_pc, target_pc_path)
                                print(f"✓ Colored target point cloud saved to: {target_pc_path}")
                                print(f"✓ Target point cloud has {len(target_colored_pc)} points")
                                
                                # Also save as numpy array
                                np.save(f"{state.vis_path}/colored_target_point_cloud.npy", target_colored_pc)
                                visual_dict['colored_target_point_cloud'] = target_colored_pc
                            
                            # Extract scene_no_targ point cloud
                            if scene_no_targ_indices.any():
                                scene_no_targ_colored_pc = full_colored_scene_pc[scene_no_targ_indices]
                                scene_no_targ_pc_path = f"{state.vis_path}/colored_scene_no_targ_point_cloud.ply"
                                save_colored_point_cloud_as_ply(scene_no_targ_colored_pc, scene_no_targ_pc_path)
                                print(f"✓ Colored scene_no_targ point cloud saved to: {scene_no_targ_pc_path}")
                                print(f"✓ Scene_no_targ point cloud has {len(scene_no_targ_colored_pc)} points")
                                
                                # Also save as numpy array
                                np.save(f"{state.vis_path}/colored_scene_no_targ_point_cloud.npy", scene_no_targ_colored_pc)
                                visual_dict['colored_scene_no_targ_point_cloud'] = scene_no_targ_colored_pc
                            
                            # Save full scene point cloud
                            colored_pc_path = f"{state.vis_path}/colored_scene_point_cloud.ply"
                            save_colored_point_cloud_as_ply(full_colored_scene_pc, colored_pc_path)
                            print(f"✓ Full colored scene point cloud saved to: {colored_pc_path}")
                            
                            # Also save as numpy array for further processing
                            np.save(f"{state.vis_path}/colored_scene_point_cloud.npy", full_colored_scene_pc)
                            
                            # Add to visual_dict for external access
                            visual_dict['colored_scene_point_cloud'] = full_colored_scene_pc
                            
                            # Print point cloud statistics
                            print(f"✓ Generated full colored point cloud with {len(full_colored_scene_pc)} points")
                            print(f"✓ Point cloud range: X[{full_colored_scene_pc[:, 0].min():.3f}, {full_colored_scene_pc[:, 0].max():.3f}], "
                                  f"Y[{full_colored_scene_pc[:, 1].min():.3f}, {full_colored_scene_pc[:, 1].max():.3f}], "
                                  f"Z[{full_colored_scene_pc[:, 2].min():.3f}, {full_colored_scene_pc[:, 2].max():.3f}]")
                        else:
                            print("⚠ No valid points found in the colored point cloud")
                except Exception as e:
                    print(f"✗ Error generating colored point cloud: {e}")
                    import traceback
                    traceback.print_exc()
            
        return grasps, scores, toc, cd, iou



class PTV3ClipImplicit(object):
    """
    Implicit grasp detection class specialized for ptv3_scene model type.
    This class directly uses preprocessed complete target point clouds and TSDF data.
    """
    
    def __init__(self, model_path, model_type, best=False, force_detection=False, 
                 qual_th=0.9, out_th=0.5, visualize=False, resolution=40, 
                 cd_iou_measure=False, **kwargs):
        """
        Initialize PTV3SceneImplicit detector.
        
        Args:
            model_path: path to the trained model
            model_type: must be 'ptv3_clip'
            best: whether to return only the best grasp
            force_detection: whether to force detection even with low quality
            qual_th: quality threshold for grasp selection
            out_th: output threshold for processing
            visualize: whether to enable visualization
            resolution: voxel grid resolution
            cd_iou_measure: whether to measure Chamfer Distance and IoU
        """
        assert model_type == 'ptv3_clip', f"PTV3ClipImplicit only supports ptv3_clip model type, got {model_type}"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_network(model_path, self.device, model_type=model_type) 
        self.net = self.net.eval()
        
        net_params_count = sum(p.numel() for p in self.net.parameters())
        print(f"Number of parameters in self.net: {net_params_count}")
        
        self.qual_th = qual_th
        self.best = best
        self.force_detection = force_detection
        self.out_th = out_th
        self.visualize = visualize
        self.resolution = resolution
        self.cd_iou_measure = cd_iou_measure
        
        # Create position tensor for query points
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

    def __call__(self, state, scene_mesh=None, clip_feat_path = None, visual_dict=None, hunyun2_path=None, 
                 scene_name=None, cd_iou_measure=False, target_mesh_gt=None, aff_kwargs={}):
        """
        Perform grasp detection on the given state.
        
        Args:
            state: state object containing scene data and preprocessed complete target data
                   Expected attributes: scene_no_targ_pc, targ_pc, complete_target_pc, complete_target_tsdf
            scene_mesh: scene mesh for visualization (optional)
            visual_dict: dictionary for visualization data (optional)
            hunyun2_path: path for hunyun2 data (unused for ptv3_scene)
            scene_name: scene name (optional)
            cd_iou_measure: whether to measure CD and IoU
            target_mesh_gt: ground truth target mesh (optional, used as fallback)
            aff_kwargs: affordance visualization arguments
            
        Returns:
            grasps, scores, inference_time, cd, iou
        """
        assert state.type == 'ptv3_clip', f"PTV3ClipImplicit only supports ptv3_clip model type, got {state.type}"
        
        visual_dict = {} if visual_dict is None else visual_dict
        print(state.__dict__.keys())

        # Handle ptv3_scene input format
        if not os.path.exists(clip_feat_path):

            scene_no_targ_pc = state.scene_no_targ_pc
            scene_no_targ_pc = np.concatenate((scene_no_targ_pc, self.plane), axis=0)

            scene_no_targ_pc =specify_num_points(scene_no_targ_pc, 512)

            complete_targ_pc = state.complete_targ_pc
            complete_targ_pc = specify_num_points(complete_targ_pc, 512)

            occluder_clip_features = np.tile(state.scene_clip_features, (512, 1))
            target_clip_features = np.tile(state.target_clip_features, (512, 1))

            scene_clip_features = np.concatenate((target_clip_features, occluder_clip_features), axis=0)
            scene_pc = np.concatenate((complete_targ_pc, scene_no_targ_pc), axis=0)
            inputs = (scene_pc, scene_clip_features)
            # np.savez(clip_feat_path, **inputs)
            np.savez(clip_feat_path, scene_pc=inputs[0], scene_clip_features=inputs[1])
            
            # Visualize CLIP features
            if hasattr(state, 'vis_path') and state.vis_path:
                print("🎨 Visualizing CLIP features...")
                clip_vis_dir = os.path.join(state.vis_path, 'clip_features')
                os.makedirs(clip_vis_dir, exist_ok=True)
                
                # PCA visualization
                visualize_clip_features_pca(
                    scene_pc, scene_clip_features,
                    os.path.join(clip_vis_dir, 'clip_features_pca.ply'),
                    "CLIP Features PCA Visualization"
                )
                
                # Heatmap visualization
                visualize_clip_features_heatmap(
                    scene_clip_features,
                    os.path.join(clip_vis_dir, 'clip_features_heatmap.png'),
                    "CLIP Features Heatmap"
                )
                
                # Statistical analysis
                visualize_clip_features_statistics(
                    scene_clip_features, target_clip_features, occluder_clip_features,
                    clip_vis_dir
                )
                
                # Save raw data for further analysis
                save_clip_features_data(
                    scene_pc, scene_clip_features,
                    os.path.join(clip_vis_dir, 'clip_features_data.npz')
                )
                
                # Add to visual dict for tracking
                visual_dict['clip_features_vis_dir'] = clip_vis_dir
                visual_dict['clip_features_pca_path'] = os.path.join(clip_vis_dir, 'clip_features_pca.ply')
                visual_dict['clip_features_heatmap_path'] = os.path.join(clip_vis_dir, 'clip_features_heatmap.png')
                
        else:
            inputs = np.load(clip_feat_path, allow_pickle=True)
            inputs = (inputs['scene_pc'], inputs['scene_clip_features'])
            
            # Also visualize loaded CLIP features
            if hasattr(state, 'vis_path') and state.vis_path:
                print("🎨 Visualizing loaded CLIP features...")
                scene_pc, scene_clip_features = inputs
                clip_vis_dir = os.path.join(state.vis_path, 'clip_features')
                os.makedirs(clip_vis_dir, exist_ok=True)
                
                # PCA visualization
                visualize_clip_features_pca(
                    scene_pc, scene_clip_features,
                    os.path.join(clip_vis_dir, 'clip_features_pca_loaded.ply'),
                    "Loaded CLIP Features PCA Visualization"
                )
                
                # Heatmap visualization  
                visualize_clip_features_heatmap(
                    scene_clip_features,
                    os.path.join(clip_vis_dir, 'clip_features_heatmap_loaded.png'),
                    "Loaded CLIP Features Heatmap"
                )
        
        voxel_size, size = state.voxel_size, state.size
        
        # Predict using ptv3_scene model with preprocessed complete target data
        with torch.no_grad():
            qual_vol, rot_vol, width_vol, completed_targ_grid, cd, iou = predict_ptv3_clip(
                inputs, self.pos, self.net, self.device, 
                visual_dict, state, target_mesh_gt
            )

        begin = time.time()

        # Reshape prediction volumes
        qual_vol = qual_vol.reshape((self.resolution, self.resolution, self.resolution))
        rot_vol = rot_vol.reshape((self.resolution, self.resolution, self.resolution, 4))
        width_vol = width_vol.reshape((self.resolution, self.resolution, self.resolution))

        # Process predictions with completed target grid
        if isinstance(completed_targ_grid, np.ndarray):
            # For numpy array: reshape from (40,40,40) to (1,40,40,40)
            completed_targ_grid = np.expand_dims(completed_targ_grid, axis=0)
        else:
            # For torch tensor: use squeeze and unsqueeze
            completed_targ_grid = completed_targ_grid.squeeze().unsqueeze(0)  # from (40,40,40) to (1,40,40,40)
        
        qual_vol, rot_vol, width_vol = process(completed_targ_grid, qual_vol, rot_vol, width_vol, out_th=self.out_th)
        
        if len(qual_vol.shape) == 1:
            qual_vol = qual_vol.reshape((self.resolution, self.resolution, self.resolution))
        qual_vol = bound(qual_vol, voxel_size)

        if self.visualize:
            colored_scene_mesh = visual.affordance_visual(qual_vol, rot_vol, scene_mesh, size, self.resolution, **aff_kwargs)
            visual_dict['affordance_visual'] = colored_scene_mesh

        # Select grasps from prediction volumes
        grasps, scores = select(
            qual_vol.copy(), 
            self.pos.view(self.resolution, self.resolution, self.resolution, 3).cpu(), 
            rot_vol, width_vol, 
            threshold=self.qual_th, 
            force_detection=self.force_detection, 
            max_filter_size=8 if self.visualize else 4
        )

        toc = time.time()
        grasps, scores = np.asarray(grasps), np.asarray(scores)

        # Transform grasps to world coordinates
        new_grasps = []
        if len(grasps) > 0:
            if self.best:
                p = np.arange(len(grasps))
            else:
                p = np.random.permutation(len(grasps))
            # Define random offsets for grasps
            offset_list = [
                Transform(Rotation.identity(), [0.0, 0.0, -0.02]),
                Transform(Rotation.identity(), [0.0, 0.0, -0.04]),
                Transform(Rotation.identity(), [0.0, 0.0, -0.03]),
                Transform(Rotation.identity(), [0.0, 0.0, -0.01]),
                Transform(Rotation.identity(), [0.0, 0.0, 0.01]),
                Transform(Rotation.identity(), [0.0, 0.0, 0.02]),
                Transform(Rotation.identity(), [0.0, 0.0, 0.03]),
                Transform(Rotation.identity(), [0.0, 0.0, 0.04])
            ]
            
            for g in grasps[p]:
                pose = g.pose
                pose.translation = (pose.translation + 0.5) * size
                width = g.width * size
                
                # Apply random offset
                random_offset = np.random.choice(offset_list)
                pose = pose * random_offset  # Apply offset transform
                
                new_grasps.append(Grasp(pose, width))
            scores = scores[p]
        grasps = new_grasps

        end = time.time()
        print(f"post processing: {end-begin:.3f}s")

        if visual_dict:
            grasp_mesh_list = [visual.grasp2mesh(g, s) for g, s in zip(grasps, scores)]
            
            # Create pre-grasp visualization for each grasp
            pre_grasp_mesh_list = []
            for i, grasp in enumerate(grasps):
                # Calculate pre-grasp position (5cm above grasp position)
                T_world_grasp = grasp.pose
                T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.07])
                T_world_pregrasp = T_world_grasp * T_grasp_pregrasp
                
                # Create a Grasp object for pre-grasp position using the same width
                pregrasp_grasp = Grasp(T_world_pregrasp, 0.05)
                
                # Create pre-grasp mesh using grasp2mesh with a dummy score
                pre_grasp_mesh = visual.grasp2mesh(pregrasp_grasp, 0.5)  # Use 0.5 as dummy score
                
                # Override color to yellow for pre-grasp
                yellow_color = np.array([255, 255, 0, 180]).astype(np.uint8)  # Yellow with transparency
                colors = np.repeat(yellow_color[np.newaxis, :], len(pre_grasp_mesh.faces), axis=0)
                pre_grasp_mesh.visual.face_colors = colors
                
                pre_grasp_mesh_list.append(pre_grasp_mesh)
            
            composed_scene = trimesh.Scene(colored_scene_mesh)
            
            # Add grasp meshes (green)
            # for i, g_mesh in enumerate(grasp_mesh_list):
            #     composed_scene.add_geometry(g_mesh, node_name=f'grasp_{i}')
            
            # Add pre-grasp visualization (yellow)
            for i, pre_grasp_mesh in enumerate(pre_grasp_mesh_list):
                composed_scene.add_geometry(pre_grasp_mesh, node_name=f'pregrasp_{i}')

            # Save original composed scene GLB
            composed_scene_glb = composed_scene.export(file_type='glb')
            with open(f"{state.vis_path}/ptv3_scene_composed_visual.glb", "wb") as f:
                f.write(composed_scene_glb)
            
            # Generate colored scene point cloud from RGB image if available
            if hasattr(state, 'vis_dict') and state.vis_dict is not None:
                try:
                    # Get RGB image and depth from vis_dict
                    if isinstance(state.vis_dict, np.ndarray) and state.vis_dict.dtype == object:
                        vis_data = state.vis_dict.item()
                    else:
                        vis_data = state.vis_dict
                    
                    # Check if we have RGB image (either direct or as file path)
                    rgb_img = None
                    depth_img = None
                    
                    if 'rgb_img' in vis_data:
                        rgb_img = vis_data['rgb_img']
                    elif 'scene_rgba_path' in vis_data:
                        # Load RGB image from file path
                        import cv2
                        rgb_img = cv2.imread(vis_data['scene_rgba_path'])
                        if rgb_img is not None:
                            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                    
                    if 'depth_img' in vis_data:
                        depth_img = vis_data['depth_img']
                        # Handle batch dimension if present
                        if len(depth_img.shape) == 3 and depth_img.shape[0] == 1:
                            depth_img = depth_img[0]  # Remove batch dimension
                    
                    if rgb_img is not None and depth_img is not None:
                        # scene_id = os.path.basename(state.vis_path).split('_')[0]
                        scene_id = os.path.basename(state.vis_path)
                        root_path = 'data_scenes/acronym/acronym-middle-occlusion-1000/scenes'
                        scene_path = os.path.join(root_path, f'{scene_id}.npz')
                        scene_file = np.load(scene_path)
                        extrinsics = scene_file['extrinsics']
                        mask_targ = scene_file['mask_targ']
                        mask_scene = scene_file['mask_scene']
                        camera_extrinsic = Transform.from_list(extrinsics[0]).as_matrix()
                        
                        # Get camera parameters from simulation or use defaults
                        # You may need to adjust these based on your actual camera setup
                        # Try to get camera intrinsics from vis_data if available, else use default
                        # if 'camera_intrinsic' in vis_data:
                        #     camera_intrinsic = np.array(vis_data['camera_intrinsic'])
                        # else:
                        camera_intrinsic = np.array([
                            [540.0, 0.0, 320.0],
                            [0.0, 540.0, 240.0],
                            [0.0, 0.0, 1.0]
                        ])
                        
                        # Generate full colored point cloud without sampling limit
                        full_colored_scene_pc = rgb_depth_to_colored_point_cloud(
                            rgb_img, depth_img, camera_intrinsic, camera_extrinsic, num_points=None
                        )
                        
                        # Use masks to separate target and scene_no_targ point clouds
                        if len(full_colored_scene_pc) > 0:
                            # Get image dimensions
                            height, width = depth_img.shape
                            
                            # Create pixel coordinate grids
                            u, v = np.meshgrid(np.arange(width), np.arange(height))
                            u, v = u.flatten(), v.flatten()
                            
                            # Apply masks to separate target and scene_no_targ
                            mask_targ_flat = mask_targ.flatten()
                            mask_scene_flat = mask_scene.flatten()
                            
                            # Get valid depth pixels
                            z = depth_img.flatten()
                            valid_mask = (z > 0) & (z < np.inf) & (z < 2.0)
                            
                            # Apply valid mask to coordinates and masks
                            u_valid = u[valid_mask]
                            v_valid = v[valid_mask]
                            mask_targ_valid = mask_targ_flat[valid_mask]
                            mask_scene_valid = mask_scene_flat[valid_mask]
                            
                            # Separate target and scene_no_targ points
                            target_indices = mask_targ_valid > 0
                            scene_no_targ_indices = (mask_scene_valid > 0) & (mask_targ_valid == 0)
                            
                            # Extract target point cloud
                            if target_indices.any():
                                target_colored_pc = full_colored_scene_pc[target_indices]
                                target_pc_path = f"{state.vis_path}/colored_target_point_cloud.ply"
                                save_colored_point_cloud_as_ply(target_colored_pc, target_pc_path)
                                print(f"✓ Colored target point cloud saved to: {target_pc_path}")
                                print(f"✓ Target point cloud has {len(target_colored_pc)} points")
                                
                                # Also save as numpy array
                                np.save(f"{state.vis_path}/colored_target_point_cloud.npy", target_colored_pc)
                                visual_dict['colored_target_point_cloud'] = target_colored_pc
                            
                            # Extract scene_no_targ point cloud
                            if scene_no_targ_indices.any():
                                scene_no_targ_colored_pc = full_colored_scene_pc[scene_no_targ_indices]
                                scene_no_targ_pc_path = f"{state.vis_path}/colored_scene_no_targ_point_cloud.ply"
                                save_colored_point_cloud_as_ply(scene_no_targ_colored_pc, scene_no_targ_pc_path)
                                print(f"✓ Colored scene_no_targ point cloud saved to: {scene_no_targ_pc_path}")
                                print(f"✓ Scene_no_targ point cloud has {len(scene_no_targ_colored_pc)} points")
                                
                                # Also save as numpy array
                                np.save(f"{state.vis_path}/colored_scene_no_targ_point_cloud.npy", scene_no_targ_colored_pc)
                                visual_dict['colored_scene_no_targ_point_cloud'] = scene_no_targ_colored_pc
                            
                            # Save full scene point cloud
                            colored_pc_path = f"{state.vis_path}/colored_scene_point_cloud.ply"
                            save_colored_point_cloud_as_ply(full_colored_scene_pc, colored_pc_path)
                            print(f"✓ Full colored scene point cloud saved to: {colored_pc_path}")
                            
                            # Also save as numpy array for further processing
                            np.save(f"{state.vis_path}/colored_scene_point_cloud.npy", full_colored_scene_pc)
                            
                            # Add to visual_dict for external access
                            visual_dict['colored_scene_point_cloud'] = full_colored_scene_pc
                            
                            # Print point cloud statistics
                            print(f"✓ Generated full colored point cloud with {len(full_colored_scene_pc)} points")
                            print(f"✓ Point cloud range: X[{full_colored_scene_pc[:, 0].min():.3f}, {full_colored_scene_pc[:, 0].max():.3f}], "
                                  f"Y[{full_colored_scene_pc[:, 1].min():.3f}, {full_colored_scene_pc[:, 1].max():.3f}], "
                                  f"Z[{full_colored_scene_pc[:, 2].min():.3f}, {full_colored_scene_pc[:, 2].max():.3f}]")
                        else:
                            print("⚠ No valid points found in the colored point cloud")
                except Exception as e:
                    print(f"✗ Error generating colored point cloud: {e}")
                    import traceback
                    traceback.print_exc()
            
        return grasps, scores, toc, cd, iou



class PTV3ClipGTImplicit(object):
    """
    Implicit grasp detection class specialized for ptv3_scene model type.
    This class directly uses preprocessed complete target point clouds and TSDF data.
    """
    
    def __init__(self, model_path, model_type, best=False, force_detection=False, 
                 qual_th=0.9, out_th=0.5, visualize=False, resolution=40, 
                 cd_iou_measure=False, **kwargs):
        """
        Initialize PTV3SceneImplicit detector.
        
        Args:
            model_path: path to the trained model
            model_type: must be 'ptv3_clip_gt'
            best: whether to return only the best grasp
            force_detection: whether to force detection even with low quality
            qual_th: quality threshold for grasp selection
            out_th: output threshold for processing
            visualize: whether to enable visualization
            resolution: voxel grid resolution
            cd_iou_measure: whether to measure Chamfer Distance and IoU
        """
        assert model_type == 'ptv3_clip_gt', f"PTV3ClipImplicit only supports ptv3_clip model type, got {model_type}"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_network(model_path, self.device, model_type=model_type) 
        self.net = self.net.eval()
        
        net_params_count = sum(p.numel() for p in self.net.parameters())
        print(f"Number of parameters in self.net: {net_params_count}")
        
        self.qual_th = qual_th
        self.best = best
        self.force_detection = force_detection
        self.out_th = out_th
        self.visualize = visualize
        self.resolution = resolution
        self.cd_iou_measure = cd_iou_measure
        
        # Create position tensor for query points
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

    def __call__(self, state, scene_mesh=None, clip_feat_gt_path = None, visual_dict=None, hunyun2_path=None, 
                 scene_name=None, cd_iou_measure=False, target_mesh_gt=None, aff_kwargs={}):
        """
        Perform grasp detection on the given state.
        
        Args:
            state: state object containing scene data and preprocessed complete target data
                   Expected attributes: scene_no_targ_pc, targ_pc, complete_target_pc, complete_target_tsdf
            scene_mesh: scene mesh for visualization (optional)
            visual_dict: dictionary for visualization data (optional)
            hunyun2_path: path for hunyun2 data (unused for ptv3_scene)
            scene_name: scene name (optional)
            cd_iou_measure: whether to measure CD and IoU
            target_mesh_gt: ground truth target mesh (optional, used as fallback)
            aff_kwargs: affordance visualization arguments
            
        Returns:
            grasps, scores, inference_time, cd, iou
        """
        assert state.type == 'ptv3_clip_gt', f"PTV3ClipImplicit only supports ptv3_clip model type, got {state.type}"
        
        visual_dict = {} if visual_dict is None else visual_dict
        print(state.__dict__.keys())

        # Handle ptv3_scene input format
        if not os.path.exists(clip_feat_gt_path):

            scene_no_targ_pc = state.scene_no_targ_pc
            scene_no_targ_pc = np.concatenate((scene_no_targ_pc, self.plane), axis=0)

            scene_no_targ_pc =specify_num_points(scene_no_targ_pc, 512)

            complete_targ_pc = state.complete_targ_pc
            complete_targ_pc = specify_num_points(complete_targ_pc, 512)

            occluder_clip_features = np.tile(state.scene_clip_features, (512, 1))
            target_clip_features = np.tile(state.target_clip_features, (512, 1))

            scene_clip_features = np.concatenate((target_clip_features, occluder_clip_features), axis=0)
            scene_pc = np.concatenate((complete_targ_pc, scene_no_targ_pc), axis=0)
            inputs = (scene_pc, scene_clip_features)
            # np.savez(clip_feat_path, **inputs)
            np.savez(clip_feat_gt_path, scene_pc=inputs[0], scene_clip_features=inputs[1])
            
            # Visualize CLIP features for GT model
            if hasattr(state, 'vis_path') and state.vis_path:
                print("🎨 Visualizing CLIP GT features...")
                clip_vis_dir = os.path.join(state.vis_path, 'clip_features_gt')
                os.makedirs(clip_vis_dir, exist_ok=True)
                
                # PCA visualization
                visualize_clip_features_pca(
                    scene_pc, scene_clip_features,
                    os.path.join(clip_vis_dir, 'clip_features_gt_pca.ply'),
                    "CLIP GT Features PCA Visualization"
                )
                
                # Heatmap visualization
                visualize_clip_features_heatmap(
                    scene_clip_features,
                    os.path.join(clip_vis_dir, 'clip_features_gt_heatmap.png'),
                    "CLIP GT Features Heatmap"
                )
                
                # Statistical analysis
                visualize_clip_features_statistics(
                    scene_clip_features, target_clip_features, occluder_clip_features,
                    clip_vis_dir
                )
                
                # Save raw data for further analysis
                save_clip_features_data(
                    scene_pc, scene_clip_features,
                    os.path.join(clip_vis_dir, 'clip_features_gt_data.npz')
                )
                
                # Add to visual dict for tracking
                visual_dict['clip_features_gt_vis_dir'] = clip_vis_dir
                visual_dict['clip_features_gt_pca_path'] = os.path.join(clip_vis_dir, 'clip_features_gt_pca.ply')
                visual_dict['clip_features_gt_heatmap_path'] = os.path.join(clip_vis_dir, 'clip_features_gt_heatmap.png')
                
        else:
            inputs = np.load(clip_feat_gt_path, allow_pickle=True)
            inputs = (inputs['scene_pc'], inputs['scene_clip_features'])
            
            # Also visualize loaded CLIP GT features
            if hasattr(state, 'vis_path') and state.vis_path:
                print("🎨 Visualizing loaded CLIP GT features...")
                scene_pc, scene_clip_features = inputs
                clip_vis_dir = os.path.join(state.vis_path, 'clip_features_gt')
                os.makedirs(clip_vis_dir, exist_ok=True)
                
                # PCA visualization
                visualize_clip_features_pca(
                    scene_pc, scene_clip_features,
                    os.path.join(clip_vis_dir, 'clip_features_gt_pca_loaded.ply'),
                    "Loaded CLIP GT Features PCA Visualization"
                )
                
                # Heatmap visualization
                visualize_clip_features_heatmap(
                    scene_clip_features,
                    os.path.join(clip_vis_dir, 'clip_features_gt_heatmap_loaded.png'),
                    "Loaded CLIP GT Features Heatmap"
                )
        
        voxel_size, size = state.voxel_size, state.size
        
        # Predict using ptv3_scene model with preprocessed complete target data
        with torch.no_grad():
            qual_vol, rot_vol, width_vol, completed_targ_grid, cd, iou = predict_ptv3_clip(
                inputs, self.pos, self.net, self.device, 
                visual_dict, state, target_mesh_gt
            )

        begin = time.time()

        # Reshape prediction volumes
        qual_vol = qual_vol.reshape((self.resolution, self.resolution, self.resolution))
        rot_vol = rot_vol.reshape((self.resolution, self.resolution, self.resolution, 4))
        width_vol = width_vol.reshape((self.resolution, self.resolution, self.resolution))

        # Process predictions with completed target grid
        if isinstance(completed_targ_grid, np.ndarray):
            # For numpy array: reshape from (40,40,40) to (1,40,40,40)
            completed_targ_grid = np.expand_dims(completed_targ_grid, axis=0)
        else:
            # For torch tensor: use squeeze and unsqueeze
            completed_targ_grid = completed_targ_grid.squeeze().unsqueeze(0)  # from (40,40,40) to (1,40,40,40)
        
        qual_vol, rot_vol, width_vol = process(completed_targ_grid, qual_vol, rot_vol, width_vol, out_th=self.out_th)
        
        if len(qual_vol.shape) == 1:
            qual_vol = qual_vol.reshape((self.resolution, self.resolution, self.resolution))
        qual_vol = bound(qual_vol, voxel_size)

        if self.visualize:
            colored_scene_mesh = visual.affordance_visual(qual_vol, rot_vol, scene_mesh, size, self.resolution, **aff_kwargs)
            visual_dict['affordance_visual'] = colored_scene_mesh

        # Select grasps from prediction volumes
        grasps, scores = select(
            qual_vol.copy(), 
            self.pos.view(self.resolution, self.resolution, self.resolution, 3).cpu(), 
            rot_vol, width_vol, 
            threshold=self.qual_th, 
            force_detection=self.force_detection, 
            max_filter_size=8 if self.visualize else 4
        )

        toc = time.time()
        grasps, scores = np.asarray(grasps), np.asarray(scores)

        # Transform grasps to world coordinates
        new_grasps = []
        if len(grasps) > 0:
            # Define random offsets for grasps
            offset_list = [
                Transform(Rotation.identity(), [0.0, 0.0, -0.02]),
                Transform(Rotation.identity(), [0.0, 0.0, -0.04]),
                Transform(Rotation.identity(), [0.0, 0.0, -0.03]),
                Transform(Rotation.identity(), [0.0, 0.0, -0.01]),
                Transform(Rotation.identity(), [0.0, 0.0, 0.01]),
                Transform(Rotation.identity(), [0.0, 0.0, 0.02]),
                Transform(Rotation.identity(), [0.0, 0.0, 0.03]),
                Transform(Rotation.identity(), [0.0, 0.0, 0.04])
            ]
            
            if self.best:
                p = np.arange(len(grasps))
            else:
                p = np.random.permutation(len(grasps))
            for g in grasps[p]:
                pose = g.pose
                pose.translation = (pose.translation + 0.5) * size
                width = g.width * size
                
                # Apply random offset
                random_offset = np.random.choice(offset_list)
                pose = pose * random_offset  # Apply offset transform
                
                new_grasps.append(Grasp(pose, width))
            scores = scores[p]
        grasps = new_grasps

        end = time.time()
        print(f"post processing: {end-begin:.3f}s")

        if visual_dict:
            grasp_mesh_list = [visual.grasp2mesh(g, s) for g, s in zip(grasps, scores)]
            
            # Create pre-grasp visualization for each grasp
            pre_grasp_mesh_list = []
            for i, grasp in enumerate(grasps):
                # Calculate pre-grasp position (5cm above grasp position)
                T_world_grasp = grasp.pose
                T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
                T_world_pregrasp = T_world_grasp * T_grasp_pregrasp
                
                # Create a Grasp object for pre-grasp position using the same width
                pregrasp_grasp = Grasp(T_world_pregrasp, grasp.width)
                
                # Create pre-grasp mesh using grasp2mesh with a dummy score
                pre_grasp_mesh = visual.grasp2mesh(pregrasp_grasp, 0.5)  # Use 0.5 as dummy score
                
                # Override color to yellow for pre-grasp
                yellow_color = np.array([255, 255, 0, 180]).astype(np.uint8)  # Yellow with transparency
                colors = np.repeat(yellow_color[np.newaxis, :], len(pre_grasp_mesh.faces), axis=0)
                pre_grasp_mesh.visual.face_colors = colors
                
                pre_grasp_mesh_list.append(pre_grasp_mesh)
            
            composed_scene = trimesh.Scene(colored_scene_mesh)
            
            # Add grasp meshes (green)
            for i, g_mesh in enumerate(grasp_mesh_list):
                composed_scene.add_geometry(g_mesh, node_name=f'grasp_{i}')
            
            # Add pre-grasp visualization (yellow)
            for i, pre_grasp_mesh in enumerate(pre_grasp_mesh_list):
                composed_scene.add_geometry(pre_grasp_mesh, node_name=f'pregrasp_{i}')
            
            visual_dict['composed_scene'] = composed_scene
            
            # Render composed scene with PyVista
            render_composed_success = render_scene_with_pyvista(
                composed_scene, 
                output_path=f"{state.vis_path}/ptv3_scene_composed_visual.png",
                width=800, 
                height=600
            )
            
            if render_composed_success:
                print("✓ Successfully rendered composed scene with PyVista")
            else:
                print("✗ Failed to render composed scene with PyVista")
            
            # Generate colored scene point cloud from RGB image if available
            if hasattr(state, 'vis_dict') and state.vis_dict is not None:
                try:
                    # Get RGB image and depth from vis_dict
                    if isinstance(state.vis_dict, np.ndarray) and state.vis_dict.dtype == object:
                        vis_data = state.vis_dict.item()
                    else:
                        vis_data = state.vis_dict
                    
                    # Check if we have RGB image (either direct or as file path)
                    rgb_img = None
                    depth_img = None
                    
                    if 'rgb_img' in vis_data:
                        rgb_img = vis_data['rgb_img']
                    elif 'scene_rgba_path' in vis_data:
                        # Load RGB image from file path
                        import cv2
                        rgb_img = cv2.imread(vis_data['scene_rgba_path'])
                        if rgb_img is not None:
                            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                    
                    if 'depth_img' in vis_data:
                        depth_img = vis_data['depth_img']
                        # Handle batch dimension if present
                        if len(depth_img.shape) == 3 and depth_img.shape[0] == 1:
                            depth_img = depth_img[0]  # Remove batch dimension
                    
                    if rgb_img is not None and depth_img is not None:
                        # scene_id = os.path.basename(state.vis_path).split('_')[0]
                        scene_id = os.path.basename(state.vis_path)
                        root_path = 'data_scenes/acronym/acronym-middle-occlusion-1000/scenes'
                        scene_path = os.path.join(root_path, f'{scene_id}.npz')
                        scene_file = np.load(scene_path)
                        extrinsics = scene_file['extrinsics']
                        mask_targ = scene_file['mask_targ']
                        mask_scene = scene_file['mask_scene']
                        camera_extrinsic = Transform.from_list(extrinsics[0]).as_matrix()
                        
                        # Get camera parameters from simulation or use defaults
                        # You may need to adjust these based on your actual camera setup
                        # Try to get camera intrinsics from vis_data if available, else use default
                        # if 'camera_intrinsic' in vis_data:
                        #     camera_intrinsic = np.array(vis_data['camera_intrinsic'])
                        # else:
                        camera_intrinsic = np.array([
                            [540.0, 0.0, 320.0],
                            [0.0, 540.0, 240.0],
                            [0.0, 0.0, 1.0]
                        ])
                        
                        # Generate full colored point cloud without sampling limit
                        full_colored_scene_pc = rgb_depth_to_colored_point_cloud(
                            rgb_img, depth_img, camera_intrinsic, camera_extrinsic, num_points=None
                        )
                        
                        # Use masks to separate target and scene_no_targ point clouds
                        if len(full_colored_scene_pc) > 0:
                            # Get image dimensions
                            height, width = depth_img.shape
                            
                            # Create pixel coordinate grids
                            u, v = np.meshgrid(np.arange(width), np.arange(height))
                            u, v = u.flatten(), v.flatten()
                            
                            # Apply masks to separate target and scene_no_targ
                            mask_targ_flat = mask_targ.flatten()
                            mask_scene_flat = mask_scene.flatten()
                            
                            # Get valid depth pixels
                            z = depth_img.flatten()
                            valid_mask = (z > 0) & (z < np.inf) & (z < 2.0)
                            
                            # Apply valid mask to coordinates and masks
                            u_valid = u[valid_mask]
                            v_valid = v[valid_mask]
                            mask_targ_valid = mask_targ_flat[valid_mask]
                            mask_scene_valid = mask_scene_flat[valid_mask]
                            
                            # Separate target and scene_no_targ points
                            target_indices = mask_targ_valid > 0
                            scene_no_targ_indices = (mask_scene_valid > 0) & (mask_targ_valid == 0)
                            
                            # Extract target point cloud
                            if target_indices.any():
                                target_colored_pc = full_colored_scene_pc[target_indices]
                                target_pc_path = f"{state.vis_path}/colored_target_point_cloud.ply"
                                save_colored_point_cloud_as_ply(target_colored_pc, target_pc_path)
                                print(f"✓ Colored target point cloud saved to: {target_pc_path}")
                                print(f"✓ Target point cloud has {len(target_colored_pc)} points")
                                
                                # Also save as numpy array
                                np.save(f"{state.vis_path}/colored_target_point_cloud.npy", target_colored_pc)
                                visual_dict['colored_target_point_cloud'] = target_colored_pc
                            
                            # Extract scene_no_targ point cloud
                            if scene_no_targ_indices.any():
                                scene_no_targ_colored_pc = full_colored_scene_pc[scene_no_targ_indices]
                                scene_no_targ_pc_path = f"{state.vis_path}/colored_scene_no_targ_point_cloud.ply"
                                save_colored_point_cloud_as_ply(scene_no_targ_colored_pc, scene_no_targ_pc_path)
                                print(f"✓ Colored scene_no_targ point cloud saved to: {scene_no_targ_pc_path}")
                                print(f"✓ Scene_no_targ point cloud has {len(scene_no_targ_colored_pc)} points")
                                
                                # Also save as numpy array
                                np.save(f"{state.vis_path}/colored_scene_no_targ_point_cloud.npy", scene_no_targ_colored_pc)
                                visual_dict['colored_scene_no_targ_point_cloud'] = scene_no_targ_colored_pc
                            
                            # Save full scene point cloud
                            colored_pc_path = f"{state.vis_path}/colored_scene_point_cloud.ply"
                            save_colored_point_cloud_as_ply(full_colored_scene_pc, colored_pc_path)
                            print(f"✓ Full colored scene point cloud saved to: {colored_pc_path}")
                            
                            # Also save as numpy array for further processing
                            np.save(f"{state.vis_path}/colored_scene_point_cloud.npy", full_colored_scene_pc)
                            
                            # Add to visual_dict for external access
                            visual_dict['colored_scene_point_cloud'] = full_colored_scene_pc
                            
                            # Print point cloud statistics
                            print(f"✓ Generated full colored point cloud with {len(full_colored_scene_pc)} points")
                            print(f"✓ Point cloud range: X[{full_colored_scene_pc[:, 0].min():.3f}, {full_colored_scene_pc[:, 0].max():.3f}], "
                                  f"Y[{full_colored_scene_pc[:, 1].min():.3f}, {full_colored_scene_pc[:, 1].max():.3f}], "
                                  f"Z[{full_colored_scene_pc[:, 2].min():.3f}, {full_colored_scene_pc[:, 2].max():.3f}]")
                        else:
                            print("⚠ No valid points found in the colored point cloud")
                except Exception as e:
                    print(f"✗ Error generating colored point cloud: {e}")
                    import traceback
                    traceback.print_exc()
            
        return grasps, scores, toc, cd, iou


class PTV3SceneGTImplicit(object):
    """
    Implicit grasp detection class specialized for ptv3_scene model type.
    This class directly uses preprocessed complete target point clouds and TSDF data.
    """
    
    def __init__(self, model_path, model_type, best=False, force_detection=False, 
                 qual_th=0.9, out_th=0.5, visualize=False, resolution=40, 
                 cd_iou_measure=False, **kwargs):
        """
        Initialize PTV3SceneImplicit detector.
        
        Args:
            model_path: path to the trained model
            model_type: must be 'ptv3_scene_gt'
            best: whether to return only the best grasp
            force_detection: whether to force detection even with low quality
            qual_th: quality threshold for grasp selection
            out_th: output threshold for processing
            visualize: whether to enable visualization
            resolution: voxel grid resolution
            cd_iou_measure: whether to measure Chamfer Distance and IoU
        """
        assert model_type == 'ptv3_scene_gt', f"PTV3SceneImplicit only supports ptv3_scene model type, got {model_type}"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_network(model_path, self.device, model_type=model_type) 
        self.net = self.net.eval()
        
        net_params_count = sum(p.numel() for p in self.net.parameters())
        print(f"Number of parameters in self.net: {net_params_count}")
        
        self.qual_th = qual_th
        self.best = best
        self.force_detection = force_detection
        self.out_th = out_th
        self.visualize = visualize
        self.resolution = resolution
        self.cd_iou_measure = cd_iou_measure
        
        # Create position tensor for query points
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

    def __call__(self, state, scene_mesh=None, visual_dict=None, hunyun2_path=None, 
                 scene_name=None, cd_iou_measure=False, target_mesh_gt=None, aff_kwargs={}):
        """
        Perform grasp detection on the given state.
        
        Args:
            state: state object containing scene data and preprocessed complete target data
                   Expected attributes: scene_no_targ_pc, targ_pc, complete_target_pc, complete_target_tsdf
            scene_mesh: scene mesh for visualization (optional)
            visual_dict: dictionary for visualization data (optional)
            hunyun2_path: path for hunyun2 data (unused for ptv3_scene)
            scene_name: scene name (optional)
            cd_iou_measure: whether to measure CD and IoU
            target_mesh_gt: ground truth target mesh (optional, used as fallback)
            aff_kwargs: affordance visualization arguments
            
        Returns:
            grasps, scores, inference_time, cd, iou
        """
        assert state.type == 'ptv3_scene_gt', f"PTV3SceneImplicit only supports ptv3_scene model type, got {state.type}"
        
        visual_dict = {} if visual_dict is None else visual_dict
        print(state.__dict__.keys())

        # Handle ptv3_scene input format
        scene_no_targ_pc = state.scene_no_targ_pc
        scene_no_targ_pc = np.concatenate((scene_no_targ_pc, self.plane), axis=0)

        scene_no_targ_pc =specify_num_points(scene_no_targ_pc, 512)

        complete_targ_pc = state.complete_targ_pc
        complete_targ_pc = specify_num_points(complete_targ_pc, 512)

        full_scene_pc = np.concatenate((scene_no_targ_pc, complete_targ_pc), axis=0)
        scene_labels = np.zeros((len(scene_no_targ_pc), 1), dtype=np.float32)
        target_labels = np.ones((len(complete_targ_pc), 1), dtype=np.float32)
        scene_with_labels = np.concatenate([scene_no_targ_pc, scene_labels], axis=1)
        target_with_labels = np.concatenate([complete_targ_pc, target_labels], axis=1)
        full_scene_with_labels = np.concatenate([scene_with_labels, target_with_labels], axis=0)
        inputs = (full_scene_with_labels, None)

        # save_point_cloud_as_ply(full_scene_with_labels[:, :3], 'full_scene_with_labels.ply')
        
        voxel_size, size = state.voxel_size, state.size
        
        # Predict using ptv3_scene model with preprocessed complete target data
        with torch.no_grad():
            qual_vol, rot_vol, width_vol, completed_targ_grid, cd, iou = predict_ptv3_scene(
                inputs, self.pos, self.net, self.device, 
                visual_dict, state, target_mesh_gt
            )

        begin = time.time()

        # Reshape prediction volumes
        qual_vol = qual_vol.reshape((self.resolution, self.resolution, self.resolution))
        rot_vol = rot_vol.reshape((self.resolution, self.resolution, self.resolution, 4))
        width_vol = width_vol.reshape((self.resolution, self.resolution, self.resolution))

        # Process predictions with completed target grid
        if isinstance(completed_targ_grid, np.ndarray):
            # For numpy array: reshape from (40,40,40) to (1,40,40,40)
            completed_targ_grid = np.expand_dims(completed_targ_grid, axis=0)
        else:
            # For torch tensor: use squeeze and unsqueeze
            completed_targ_grid = completed_targ_grid.squeeze().unsqueeze(0)  # from (40,40,40) to (1,40,40,40)
        
        qual_vol, rot_vol, width_vol = process(completed_targ_grid, qual_vol, rot_vol, width_vol, out_th=self.out_th)
        
        if len(qual_vol.shape) == 1:
            qual_vol = qual_vol.reshape((self.resolution, self.resolution, self.resolution))
        qual_vol = bound(qual_vol, voxel_size)

        if self.visualize:
            colored_scene_mesh = visual.affordance_visual(qual_vol, rot_vol, scene_mesh, size, self.resolution, **aff_kwargs)
            visual_dict['affordance_visual'] = colored_scene_mesh

        # Select grasps from prediction volumes
        grasps, scores = select(
            qual_vol.copy(), 
            self.pos.view(self.resolution, self.resolution, self.resolution, 3).cpu(), 
            rot_vol, width_vol, 
            threshold=self.qual_th, 
            force_detection=self.force_detection, 
            max_filter_size=8 if self.visualize else 4
        )

        toc = time.time()
        grasps, scores = np.asarray(grasps), np.asarray(scores)

        # Transform grasps to world coordinates
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

        if visual_dict:
            grasp_mesh_list = [visual.grasp2mesh(g, s) for g, s in zip(grasps, scores)]
            
            # Create pre-grasp visualization for each grasp
            pre_grasp_mesh_list = []
            for i, grasp in enumerate(grasps):
                # Calculate pre-grasp position (5cm above grasp position)
                T_world_grasp = grasp.pose
                T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
                T_world_pregrasp = T_world_grasp * T_grasp_pregrasp
                
                # Create a Grasp object for pre-grasp position using the same width
                pregrasp_grasp = Grasp(T_world_pregrasp, grasp.width)
                
                # Create pre-grasp mesh using grasp2mesh with a dummy score
                pre_grasp_mesh = visual.grasp2mesh(pregrasp_grasp, 0.5)  # Use 0.5 as dummy score
                
                # Override color to yellow for pre-grasp
                yellow_color = np.array([255, 255, 0, 180]).astype(np.uint8)  # Yellow with transparency
                colors = np.repeat(yellow_color[np.newaxis, :], len(pre_grasp_mesh.faces), axis=0)
                pre_grasp_mesh.visual.face_colors = colors
                
                pre_grasp_mesh_list.append(pre_grasp_mesh)
            
            composed_scene = trimesh.Scene(colored_scene_mesh)
            
            # Add grasp meshes (green)
            for i, g_mesh in enumerate(grasp_mesh_list):
                composed_scene.add_geometry(g_mesh, node_name=f'grasp_{i}')
            
            # Add pre-grasp visualization (yellow)
            for i, pre_grasp_mesh in enumerate(pre_grasp_mesh_list):
                composed_scene.add_geometry(pre_grasp_mesh, node_name=f'pregrasp_{i}')
            
            visual_dict['composed_scene'] = composed_scene
            
            # Render composed scene with PyVista
            render_composed_success = render_scene_with_pyvista(
                composed_scene, 
                output_path=f"{state.vis_path}/ptv3_scene_composed_visual.png",
                width=800, 
                height=600
            )
            
            if render_composed_success:
                print("✓ Successfully rendered composed scene with PyVista")
            else:
                print("✗ Failed to render composed scene with PyVista")
            
            # Generate colored scene point cloud from RGB image if available
            if hasattr(state, 'vis_dict') and state.vis_dict is not None:
                try:
                    # Get RGB image and depth from vis_dict
                    if isinstance(state.vis_dict, np.ndarray) and state.vis_dict.dtype == object:
                        vis_data = state.vis_dict.item()
                    else:
                        vis_data = state.vis_dict
                    
                    # Check if we have RGB image (either direct or as file path)
                    rgb_img = None
                    depth_img = None
                    
                    if 'rgb_img' in vis_data:
                        rgb_img = vis_data['rgb_img']
                    elif 'scene_rgba_path' in vis_data:
                        # Load RGB image from file path
                        import cv2
                        rgb_img = cv2.imread(vis_data['scene_rgba_path'])
                        if rgb_img is not None:
                            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                    
                    if 'depth_img' in vis_data:
                        depth_img = vis_data['depth_img']
                        # Handle batch dimension if present
                        if len(depth_img.shape) == 3 and depth_img.shape[0] == 1:
                            depth_img = depth_img[0]  # Remove batch dimension
                    
                    if rgb_img is not None and depth_img is not None:
                        # scene_id = os.path.basename(state.vis_path).split('_')[0]
                        scene_id = os.path.basename(state.vis_path)
                        root_path = 'data_scenes/acronym/acronym-middle-occlusion-1000/scenes'
                        scene_path = os.path.join(root_path, f'{scene_id}.npz')
                        scene_file = np.load(scene_path)
                        extrinsics = scene_file['extrinsics']
                        mask_targ = scene_file['mask_targ']
                        mask_scene = scene_file['mask_scene']
                        camera_extrinsic = Transform.from_list(extrinsics[0]).as_matrix()
                        
                        # Get camera parameters from simulation or use defaults
                        # You may need to adjust these based on your actual camera setup
                        # Try to get camera intrinsics from vis_data if available, else use default
                        # if 'camera_intrinsic' in vis_data:
                        #     camera_intrinsic = np.array(vis_data['camera_intrinsic'])
                        # else:
                        camera_intrinsic = np.array([
                            [540.0, 0.0, 320.0],
                            [0.0, 540.0, 240.0],
                            [0.0, 0.0, 1.0]
                        ])
                        
                        # Generate full colored point cloud without sampling limit
                        full_colored_scene_pc = rgb_depth_to_colored_point_cloud(
                            rgb_img, depth_img, camera_intrinsic, camera_extrinsic, num_points=None
                        )
                        
                        # Use masks to separate target and scene_no_targ point clouds
                        if len(full_colored_scene_pc) > 0:
                            # Get image dimensions
                            height, width = depth_img.shape
                            
                            # Create pixel coordinate grids
                            u, v = np.meshgrid(np.arange(width), np.arange(height))
                            u, v = u.flatten(), v.flatten()
                            
                            # Apply masks to separate target and scene_no_targ
                            mask_targ_flat = mask_targ.flatten()
                            mask_scene_flat = mask_scene.flatten()
                            
                            # Get valid depth pixels
                            z = depth_img.flatten()
                            valid_mask = (z > 0) & (z < np.inf) & (z < 2.0)
                            
                            # Apply valid mask to coordinates and masks
                            u_valid = u[valid_mask]
                            v_valid = v[valid_mask]
                            mask_targ_valid = mask_targ_flat[valid_mask]
                            mask_scene_valid = mask_scene_flat[valid_mask]
                            
                            # Separate target and scene_no_targ points
                            target_indices = mask_targ_valid > 0
                            scene_no_targ_indices = (mask_scene_valid > 0) & (mask_targ_valid == 0)
                            
                            # Extract target point cloud
                            if target_indices.any():
                                target_colored_pc = full_colored_scene_pc[target_indices]
                                target_pc_path = f"{state.vis_path}/colored_target_point_cloud.ply"
                                save_colored_point_cloud_as_ply(target_colored_pc, target_pc_path)
                                print(f"✓ Colored target point cloud saved to: {target_pc_path}")
                                print(f"✓ Target point cloud has {len(target_colored_pc)} points")
                                
                                # Also save as numpy array
                                np.save(f"{state.vis_path}/colored_target_point_cloud.npy", target_colored_pc)
                                visual_dict['colored_target_point_cloud'] = target_colored_pc
                            
                            # Extract scene_no_targ point cloud
                            if scene_no_targ_indices.any():
                                scene_no_targ_colored_pc = full_colored_scene_pc[scene_no_targ_indices]
                                scene_no_targ_pc_path = f"{state.vis_path}/colored_scene_no_targ_point_cloud.ply"
                                save_colored_point_cloud_as_ply(scene_no_targ_colored_pc, scene_no_targ_pc_path)
                                print(f"✓ Colored scene_no_targ point cloud saved to: {scene_no_targ_pc_path}")
                                print(f"✓ Scene_no_targ point cloud has {len(scene_no_targ_colored_pc)} points")
                                
                                # Also save as numpy array
                                np.save(f"{state.vis_path}/colored_scene_no_targ_point_cloud.npy", scene_no_targ_colored_pc)
                                visual_dict['colored_scene_no_targ_point_cloud'] = scene_no_targ_colored_pc
                            
                            # Save full scene point cloud
                            colored_pc_path = f"{state.vis_path}/colored_scene_point_cloud.ply"
                            save_colored_point_cloud_as_ply(full_colored_scene_pc, colored_pc_path)
                            print(f"✓ Full colored scene point cloud saved to: {colored_pc_path}")
                            
                            # Also save as numpy array for further processing
                            np.save(f"{state.vis_path}/colored_scene_point_cloud.npy", full_colored_scene_pc)
                            
                            # Add to visual_dict for external access
                            visual_dict['colored_scene_point_cloud'] = full_colored_scene_pc
                            
                            # Print point cloud statistics
                            print(f"✓ Generated full colored point cloud with {len(full_colored_scene_pc)} points")
                            print(f"✓ Point cloud range: X[{full_colored_scene_pc[:, 0].min():.3f}, {full_colored_scene_pc[:, 0].max():.3f}], "
                                  f"Y[{full_colored_scene_pc[:, 1].min():.3f}, {full_colored_scene_pc[:, 1].max():.3f}], "
                                  f"Z[{full_colored_scene_pc[:, 2].min():.3f}, {full_colored_scene_pc[:, 2].max():.3f}]")
                        else:
                            print("⚠ No valid points found in the colored point cloud")
                except Exception as e:
                    print(f"✗ Error generating colored point cloud: {e}")
                    import traceback
                    traceback.print_exc()
            
        return grasps, scores, toc, cd, iou

def test_color_rendering():
    """
    测试颜色渲染功能
    """
    import trimesh
    import pyvista as pv
    
    # 创建一个简单的测试mesh
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
    
    # 创建trimesh对象
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # 设置绿色
    green_color = np.array([0, 255, 0, 255]).astype(np.uint8)
    colors = np.repeat(green_color[np.newaxis, :], len(mesh.faces), axis=0)
    mesh.visual.face_colors = colors
    
    print(f"Test mesh colors: {mesh.visual.face_colors[0]}")
    
    # 创建PyVista mesh
    faces_flat = np.hstack([
        np.full(len(mesh.faces), 3, dtype=np.int64)[:, None],
        mesh.faces.astype(np.int64)
    ]).ravel()
    
    pv_mesh = pv.PolyData(mesh.vertices, faces_flat)
    
    # 设置颜色
    fc = getattr(mesh.visual, "face_colors", None)
    if fc is not None:
        pv_mesh.cell_data["colors"] = fc
        pv_mesh.cell_data.active_scalars_name = "colors"
        print(f"PyVista mesh colors: {fc[0]}")
    
    # 创建plotter并渲染
    plotter = pv.Plotter(off_screen=True, window_size=(400, 300))
    plotter.add_mesh(pv_mesh, show_edges=False, show_scalar_bar=False, scalars="colors")
    plotter.set_background("white")
    
    # 保存测试图像
    test_path = "test_color_rendering.png"
    plotter.screenshot(test_path)
    plotter.close()
    
    print(f"Test image saved to: {test_path}")
    return True

def rgb_depth_to_colored_point_cloud(rgb_img, depth_img, camera_intrinsic, camera_extrinsic, num_points=2048):
    """
    Convert RGB and depth images to colored point cloud.
    
    Args:
        rgb_img: RGB image (H, W, 3) with values in [0, 255]
        depth_img: Depth image (H, W) with depth values in meters
        camera_intrinsic: Camera intrinsic matrix (3x3)
        camera_extrinsic: Camera extrinsic matrix (4x4) - world to camera transform
        num_points: Number of points to sample from the point cloud (None for all points)
        
    Returns:
        colored_point_cloud: numpy array (N, 6) with [x, y, z, r, g, b] format
    """
    import cv2
    
    # Get image dimensions
    height, width = depth_img.shape
    
    # Create pixel coordinate grids
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u, v = u.flatten(), v.flatten()
    z = depth_img.flatten()
    
    # Filter out invalid depth values
    valid_mask = (z > 0) & (z < np.inf) & (z < 2.0)  # Filter depth range
    u, v, z = u[valid_mask], v[valid_mask], z[valid_mask]
    
    # Convert pixel coordinates to camera coordinates
    x = (u - camera_intrinsic[0, 2]) * z / camera_intrinsic[0, 0]
    y = (v - camera_intrinsic[1, 2]) * z / camera_intrinsic[1, 1]
    
    # Stack camera coordinates
    points_camera = np.vstack((x, y, z)).T
    
    # Transform to world coordinates
    camera_extrinsic_inv = np.linalg.inv(camera_extrinsic)
    points_world = np.hstack((points_camera, np.ones((len(points_camera), 1))))
    points_world = (camera_extrinsic_inv @ points_world.T).T[:, :3]
    
    # Get corresponding RGB colors
    rgb_colors = rgb_img[v, u]  # Get colors for valid pixels
    
    # Combine points and colors
    colored_point_cloud = np.hstack((points_world, rgb_colors))
    
    # Sample points if needed (only if num_points is not None)
    if num_points is not None and len(colored_point_cloud) > num_points:
        indices = np.random.choice(len(colored_point_cloud), num_points, replace=False)
        colored_point_cloud = colored_point_cloud[indices]
    
    return colored_point_cloud


def save_colored_point_cloud_as_ply(points_with_colors, output_path):
    """
    Save colored point cloud to PLY file.
    
    Args:
        points_with_colors: numpy array (N, 6) with [x, y, z, r, g, b] format
        output_path: path to save the PLY file
    """
    with open(output_path, 'w') as f:
        # Write PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_with_colors)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write point data
        for point in points_with_colors:
            x, y, z, r, g, b = point
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def visualize_clip_features_pca(scene_pc, scene_clip_features, output_path, title="CLIP Features PCA Visualization"):
    """
    Visualize CLIP features using PCA to map to RGB colors.
    
    Args:
        scene_pc: Point cloud coordinates (N, 3)
        scene_clip_features: CLIP features (N, feature_dim)
        output_path: Path to save the visualization
        title: Title for the visualization
    """
    try:
        # Apply PCA to reduce CLIP features to 3 dimensions (RGB)
        pca = PCA(n_components=3)
        features_3d = pca.fit_transform(scene_clip_features)
        
        # Normalize to [0, 1] range
        features_3d = (features_3d - features_3d.min(axis=0)) / (features_3d.max(axis=0) - features_3d.min(axis=0) + 1e-8)
        
        # Convert to RGB colors (0-255)
        rgb_colors = (features_3d * 255).astype(np.uint8)
        
        # Combine point cloud with colors
        colored_pc = np.hstack([scene_pc, rgb_colors])
        
        # Save as PLY file
        save_colored_point_cloud_as_ply(colored_pc, output_path)
        
        print(f"✓ CLIP features PCA visualization saved to: {output_path}")
        print(f"✓ PCA explained variance ratio: {pca.explained_variance_ratio_}")
        
        return colored_pc, pca.explained_variance_ratio_
        
    except Exception as e:
        print(f"✗ Error visualizing CLIP features with PCA: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def visualize_clip_features_heatmap(scene_clip_features, output_path, title="CLIP Features Heatmap"):
    """
    Create a heatmap visualization of CLIP features.
    
    Args:
        scene_clip_features: CLIP features (N, feature_dim)
        output_path: Path to save the heatmap
        title: Title for the heatmap
    """
    try:
        plt.figure(figsize=(12, 8))
        
        # Create heatmap of features (sample subset if too many points)
        max_points = 1000
        if len(scene_clip_features) > max_points:
            indices = np.random.choice(len(scene_clip_features), max_points, replace=False)
            features_subset = scene_clip_features[indices]
        else:
            features_subset = scene_clip_features
        
        # Plot heatmap
        im = plt.imshow(features_subset.T, aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(im, label='Feature Value')
        plt.xlabel('Point Index')
        plt.ylabel('Feature Dimension')
        plt.title(title)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ CLIP features heatmap saved to: {output_path}")
        
    except Exception as e:
        print(f"✗ Error creating CLIP features heatmap: {e}")
        import traceback
        traceback.print_exc()


def visualize_clip_features_statistics(scene_clip_features, target_clip_features, occluder_clip_features, output_dir):
    """
    Create statistical visualizations of CLIP features.
    
    Args:
        scene_clip_features: Combined scene CLIP features
        target_clip_features: Target object CLIP features  
        occluder_clip_features: Occluder CLIP features
        output_dir: Directory to save visualizations
    """
    try:
        # Feature statistics
        target_mean = np.mean(target_clip_features, axis=0)
        occluder_mean = np.mean(occluder_clip_features, axis=0)
        target_std = np.std(target_clip_features, axis=0)
        occluder_std = np.std(occluder_clip_features, axis=0)
        
        # Plot feature means comparison
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        feature_indices = np.arange(len(target_mean))
        plt.plot(feature_indices, target_mean, label='Target', color='red', alpha=0.7)
        plt.plot(feature_indices, occluder_mean, label='Occluder', color='blue', alpha=0.7)
        plt.xlabel('Feature Dimension')
        plt.ylabel('Mean Value')
        plt.title('CLIP Features: Mean Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot feature standard deviations comparison
        plt.subplot(1, 2, 2)
        plt.plot(feature_indices, target_std, label='Target', color='red', alpha=0.7)
        plt.plot(feature_indices, occluder_std, label='Occluder', color='blue', alpha=0.7)
        plt.xlabel('Feature Dimension')
        plt.ylabel('Standard Deviation')
        plt.title('CLIP Features: Std Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        stats_path = os.path.join(output_dir, 'clip_features_statistics.png')
        plt.savefig(stats_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate and save feature similarity
        similarity = np.dot(target_mean, occluder_mean) / (np.linalg.norm(target_mean) * np.linalg.norm(occluder_mean))
        
        # Save statistics to text file
        stats_text_path = os.path.join(output_dir, 'clip_features_statistics.txt')
        with open(stats_text_path, 'w') as f:
            f.write("CLIP Features Statistics\n")
            f.write("=" * 30 + "\n")
            f.write(f"Target features shape: {target_clip_features.shape}\n")
            f.write(f"Occluder features shape: {occluder_clip_features.shape}\n")
            f.write(f"Scene features shape: {scene_clip_features.shape}\n")
            f.write(f"Target mean norm: {np.linalg.norm(target_mean):.6f}\n")
            f.write(f"Occluder mean norm: {np.linalg.norm(occluder_mean):.6f}\n")
            f.write(f"Feature similarity (cosine): {similarity:.6f}\n")
            f.write(f"Target features range: [{target_clip_features.min():.6f}, {target_clip_features.max():.6f}]\n")
            f.write(f"Occluder features range: [{occluder_clip_features.min():.6f}, {occluder_clip_features.max():.6f}]\n")
        
        print(f"✓ CLIP features statistics saved to: {stats_path}")
        print(f"✓ CLIP features statistics text saved to: {stats_text_path}")
        print(f"✓ Target-Occluder feature similarity: {similarity:.6f}")
        
    except Exception as e:
        print(f"✗ Error creating CLIP features statistics: {e}")
        import traceback
        traceback.print_exc()


def save_clip_features_data(scene_pc, scene_clip_features, output_path):
    """
    Save CLIP features and point cloud data as numpy arrays.
    
    Args:
        scene_pc: Point cloud coordinates (N, 3)
        scene_clip_features: CLIP features (N, feature_dim)
        output_path: Path to save the data
    """
    try:
        np.savez(
            output_path,
            scene_pc=scene_pc,
            scene_clip_features=scene_clip_features
        )
        print(f"✓ CLIP features data saved to: {output_path}")
    except Exception as e:
        print(f"✗ Error saving CLIP features data: {e}")
        import traceback
        traceback.print_exc()

# 在文件末尾添加测试调用
if __name__ == "__main__":
    test_color_rendering() 