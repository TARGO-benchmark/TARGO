import os
import trimesh
import numpy as np
from urdfpy import URDF
try:
    # from vgn.ConvONets.utils.libmesh import check_mesh_contains

    from vgn.ConvONets.utils.libmesh.inside_mesh import check_mesh_contains
except:
    print('import libmesh failed!')

n_iou_points = 100000
n_iou_points_files = 10

def create_grid(resolution=40):
    # 直接在 [0, 1] 区间创建等间距点
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    z = np.linspace(0, 1, resolution)
    grid = np.meshgrid(x, y, z, indexing='ij')
    points = np.vstack(map(np.ravel, grid)).T
    return points

def sample_grid_points_bounds(bounds, resolution=40):
    # 提取每个维度的最小和最大值
    x_min, y_min, z_min = bounds[0]
    x_max, y_max, z_max = bounds[1]
    
    # 在每个维度上生成均匀分布的点
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    z = np.linspace(z_min, z_max, resolution)
    
    # 使用 meshgrid 创建三维网格
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    
    # 将三维网格的坐标转换为点的集合
    points = np.column_stack([xv.ravel(), yv.ravel(), zv.ravel()])
    
    return points

def generate_grid(space_min=0, space_max=0.3, resolution=40):
    """
    Generate a grid of points within the specified space boundaries.

    Parameters:
    space_min (float): The minimum boundary of the space.
    space_max (float): The maximum boundary of the space.
    resolution (int): The number of points to generate along each axis.

    Returns:
    numpy.ndarray: An array of points within the specified space.
    """
    # Generate a grid of points within the specified space
    x = np.linspace(space_min, space_max, resolution)
    y = np.linspace(space_min, space_max, resolution)
    z = np.linspace(space_min, space_max, resolution)
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')

    # Reshape the grid to have a list of points
    grid_points = np.column_stack((xv.ravel(), yv.ravel(), zv.ravel()))

    return grid_points

def sample_iou_points_grid(mesh_list, bounds, padding = 0.02, uniform = False, size = 0.3,resolution=40):
    # points = create_grid(resolution=resolution)
    # points  *= size
    # points = sample_grid_points_bounds(bounds)
    points = generate_grid()
    # if uniform:
    #     points *= size + 2 * padding
    #     points -= padding
    # points = points * (bounds[[1]] + 2 * padding - bounds[[0]]) + bounds[[0]] - padding
    # points = points/size - 0.5

    occ = np.zeros(points.shape[0]).astype(bool)
    for mesh in mesh_list:
        occi = check_mesh_contains(mesh, points)
        occ = occ | occi

    return points, occ

## occupancy related code
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

def get_mesh_pose_list_from_world(world, object_set, exclude_plane=True):
    mesh_pose_list = []
    # collect object mesh paths and poses
    for uid in world.bodies.keys():
        _, name = world.p.getBodyInfo(uid)
        name = name.decode('utf8')
        if name == 'plane' and exclude_plane:
            continue
        body = world.bodies[uid]
        pose = body.get_pose().as_matrix()
        scale = body.scale
        visuals = world.p.getVisualShapeData(uid)
        assert len(visuals) == 1
        _, _, _, _, mesh_path, _, _, _ = visuals[0]
        mesh_path = mesh_path.decode('utf8')
        if mesh_path == '':
            mesh_path = os.path.join('./data/urdfs', object_set, name + '.urdf')
        mesh_pose_list.append((mesh_path, scale, pose))
    return mesh_pose_list

def get_mesh_pose_dict_from_world(world, object_set, exclude_plane=True):
    mesh_pose_dict = {}
    # collect object mesh paths and poses
    for uid in world.bodies.keys():
        _, name = world.p.getBodyInfo(uid)
        name = name.decode('utf8')
        if name == 'plane' and exclude_plane:
            continue
        body = world.bodies[uid]
        pose = body.get_pose().as_matrix()
        scale = body.scale
        visuals = world.p.getVisualShapeData(uid)
        assert len(visuals) == 1
        _, _, _, _, mesh_path, _, _, _ = visuals[0]
        mesh_path = mesh_path.decode('utf8')
        if mesh_path == '':
            mesh_path = os.path.join('./data/urdfs', object_set, name + '.urdf')
        # mesh_pose_list.append((mesh_path, scale, pose))
        mesh_pose_dict[uid] = (mesh_path, scale, pose)
    return mesh_pose_dict

def get_scene_from_mesh_pose_dict(mesh_pose_list, target_id=None, scene_as_mesh=True, return_list=False):
    scene = trimesh.Scene()
    mesh_list = []
    for idx, (mesh_path, scale, pose) in enumerate(mesh_pose_list.items()):
        if os.path.splitext(mesh_path)[1] == '.urdf':
            obj = URDF.load(mesh_path)
            assert len(obj.links) == 1, "Assumption that URDF has exactly one link might not hold true."
            assert len(obj.links[0].visuals) == 1, "Assumption that each link has exactly one visual might not hold true."
            assert len(obj.links[0].visuals[0].geometry.meshes) == 1, "Assumption that each visual has exactly one mesh might not hold true."
            mesh = obj.links[0].visuals[0].geometry.meshes[0].copy()
        else:
            mesh = trimesh.load(mesh_path)

        mesh.apply_scale(scale)
        mesh.apply_transform(pose)

        # Check if current mesh is the target and color it red
        if idx == target_id:
            mesh.visual.face_colors = [255, 0, 0, 255]  # Apply red color to faces

        scene.add_geometry(mesh)
        mesh_list.append(mesh)

    if scene_as_mesh:
        # Attempt to convert the entire scene into a single mesh, if applicable
        try:
            scene = scene.dump(concatenate=True)
        except Exception as e:
            print(f"Error converting scene to mesh: {e}")

    if return_list:
        return scene, mesh_list
    else:
        return scene
    
def get_scene_from_mesh_pose_list_and_color(mesh_pose_list, target_id=None, scene_as_mesh=True, return_list=False, targ_id = None):
    scene = trimesh.Scene()
    mesh_list = []
    for idx, (mesh_path, scale, pose) in enumerate(mesh_pose_list):
        
        if os.path.splitext(mesh_path)[1] == '.urdf':
            obj = URDF.load(mesh_path)
            assert len(obj.links) == 1, "Assumption that URDF has exactly one link might not hold true."
            assert len(obj.links[0].visuals) == 1, "Assumption that each link has exactly one visual might not hold true."
            assert len(obj.links[0].visuals[0].geometry.meshes) == 1, "Assumption that each visual has exactly one mesh might not hold true."
            mesh = obj.links[0].visuals[0].geometry.meshes[0].copy()
        else:
            mesh = trimesh.load(mesh_path)
        if idx == targ_id:
            mesh.visual.face_colors = [255, 0, 0, 255] ## red

        mesh.apply_scale(scale)
        mesh.apply_transform(pose)

        # Check if current mesh is the target and color it red
        if idx == target_id:
            mesh.visual.face_colors = [255, 0, 0, 255]  # Apply red color to faces

        scene.add_geometry(mesh)
        mesh_list.append(mesh)

    if scene_as_mesh:
        # Attempt to convert the entire scene into a single mesh, if applicable
        try:
            scene = scene.dump(concatenate=True)
        except Exception as e:
            print(f"Error converting scene to mesh: {e}")

    if return_list:
        return scene, mesh_list
    else:
        return scene
    
def get_scene_from_mesh_pose_list(mesh_pose_list, target_id=None, scene_as_mesh=True, return_list=False):
    scene = trimesh.Scene()
    mesh_list = []
    for idx, (mesh_path, scale, pose) in enumerate(mesh_pose_list):
        if os.path.splitext(mesh_path)[1] == '.urdf':
            obj = URDF.load(mesh_path)
            assert len(obj.links) == 1, "Assumption that URDF has exactly one link might not hold true."
            assert len(obj.links[0].visuals) == 1, "Assumption that each link has exactly one visual might not hold true."
            assert len(obj.links[0].visuals[0].geometry.meshes) == 1, "Assumption that each visual has exactly one mesh might not hold true."
            mesh = obj.links[0].visuals[0].geometry.meshes[0].copy()
        else:
            mesh = trimesh.load(mesh_path)

        mesh.apply_scale(scale)
        mesh.apply_transform(pose)

        # Check if current mesh is the target and color it red
        if idx == target_id:
            mesh.visual.face_colors = [255, 0, 0, 255]  # Apply red color to faces

        scene.add_geometry(mesh)
        mesh_list.append(mesh)

    if scene_as_mesh:
        # Attempt to convert the entire scene into a single mesh, if applicable
        try:
            scene = scene.dump(concatenate=True)
        except Exception as e:
            print(f"Error converting scene to mesh: {e}")

    if return_list:
        return scene, mesh_list
    else:
        return scene

# def get_scene_from_mesh_pose_list(mesh_pose_list, target_id, scene_as_mesh=True, return_list=False):
#     # create scene from meshes
#     scene = trimesh.Scene()
#     mesh_list = []
#     for mesh_path, scale, pose in mesh_pose_list:
#         if os.path.splitext(mesh_path)[1] == '.urdf':
#             obj = URDF.load(mesh_path)
#             assert len(obj.links) == 1
#             assert len(obj.links[0].visuals) == 1
#             assert len(obj.links[0].visuals[0].geometry.meshes) == 1
#             mesh = obj.links[0].visuals[0].geometry.meshes[0].copy()
#         else:
#             mesh = trimesh.load(mesh_path)

#         mesh.apply_scale(scale)
#         mesh.apply_transform(pose)
#         scene.add_geometry(mesh)
#         mesh_list.append(mesh)
#     if scene_as_mesh:
#         scene = as_mesh(scene)
#     if return_list:
#         return scene, mesh_list
#     else:
#         return scene

def sample_iou_points(mesh_list, bounds, num_point, padding=0.02, uniform=False, size=0.3):
    points = np.random.rand(num_point, 3).astype(np.float32)
    if uniform:
        points *= size + 2 * padding
        points -= padding
    else:
        points = points * (bounds[[1]] + 2 * padding - bounds[[0]]) + bounds[[0]] - padding
    occ = np.zeros(num_point).astype(bool)
    for mesh in mesh_list:
        occi = check_mesh_contains(mesh, points)
        occ = occ | occi

    return points, occ

def get_occ_from_world(world, object_set):
    mesh_pose_list = get_mesh_pose_list_from_world(world, object_set)
    scene, mesh_list = get_scene_from_mesh_pose_list(mesh_pose_list, return_list=True)
    points, occ = sample_iou_points(mesh_list, scene.bounds, n_iou_points * n_iou_points_files)
    return points, occ


# def get_occ_from_mesh(scene_mesh, world_size, object_count, voxel_resolution=120):
#     # voxelize scene
#     voxel_length = world_size / voxel_resolution
#     scene_voxel = scene_mesh.voxelized(voxel_length)

#     # collect near surface points occupancy
#     surface_points, _ = trimesh.sample.sample_surface(scene_mesh, object_count * 2048)
#     surface_points += np.random.randn(*surface_points.shape) * 0.002
#     occ_surface = scene_voxel.is_filled(surface_points)
#     # collect randomly distributed points occupancy
#     random_points = np.random.rand(object_count * 2048, 3)
#     random_points = random_points * (scene_voxel.bounds[[1]] - scene_voxel.bounds[[0]]) + scene_voxel.bounds[[0]]
#     occ_random = scene_voxel.is_filled(random_points)
#     surface_p_occ = np.concatenate((surface_points, occ_surface[:, np.newaxis]), axis=-1)
#     random_p_occ = np.concatenate((random_points, occ_random[:, np.newaxis]), axis=-1)
#     return surface_p_occ, random_p_occ
