import json
import uuid

import numpy as np
import pandas as pd
from pathlib import Path
from vgn.grasp import Grasp
from vgn.perception import *
from vgn.utils.transform import Rotation, Transform


_SCENE_INDEX_CACHE = {}


def scene_file_path(root, scene_id, subdir="scenes"):
    """Return the .npz file of ``scene_id`` below ``<root>/<subdir>``.

    Local datasets store ``<root>/<subdir>/<scene_id>.npz``. The Hugging Face copy of the
    training set is sharded into numbered sub-directories (``<root>/<subdir>/000/<scene_id>.npz``).
    The sharded layout is indexed once per process so both layouts work without renaming files.
    """
    root = Path(root)
    flat = root / subdir / f"{scene_id}.npz"
    if flat.exists():
        return flat
    key = (str(root), subdir)
    index = _SCENE_INDEX_CACHE.get(key)
    if index is None:
        index = {path.stem: path for path in (root / subdir).glob("*/*.npz")}
        _SCENE_INDEX_CACHE[key] = index
    return index.get(scene_id, flat)



def write_setup(root, size, intrinsic, max_opening_width, finger_depth):
    path = root / "setup.json"
    if path.exists():
        return
    data = {
        "size": size,
        "intrinsic": intrinsic.to_dict(),
        "max_opening_width": max_opening_width,
        "finger_depth": finger_depth,
    }
    write_json(data, path)


def read_setup(root):
    data = read_json(root / "setup.json")
    size = data["size"]
    intrinsic = CameraIntrinsic.from_dict(data["intrinsic"])
    max_opening_width = data["max_opening_width"]
    finger_depth = data["finger_depth"]
    return size, intrinsic, max_opening_width, finger_depth


def write_sensor_data(root, depth_imgs, extrinsics):
    scene_id = uuid.uuid4().hex
    path = scene_file_path(root, scene_id)
    assert not path.exists()
    np.savez_compressed(path, depth_imgs=depth_imgs, extrinsics=extrinsics)
    return scene_id

def write_raw_sensor_data(root, scene_id, depth_imgs, extrinsics, seg, depth_imgs_side, extrinsics_side, seg_side, mesh_pose_dict, pc_part, tgt_id):
    path = scene_file_path(root, scene_id)
    assert not path.exists()
    np.savez_compressed(path, depth_imgs=depth_imgs, extrinsics=extrinsics, seg=seg, depth_imgs_side=depth_imgs_side, extrinsics_side=extrinsics_side, seg_side=seg_side, mesh_pose_dict=mesh_pose_dict, pc_part=pc_part, tgt_id=tgt_id)
    return scene_id

def write_sensor_data_TSDF(root, depth_imgs, extrinsics, tsdf_grid):
    scene_id = uuid.uuid4().hex
    path = scene_file_path(root, scene_id)
    assert not path.exists()
    np.savez_compressed(path, depth_imgs=depth_imgs, extrinsics=extrinsics, tsdf_grid=tsdf_grid)
    return scene_id

def write_other_sensor_data(root, scene_id, depth_imgs, extrinsics, mask_targ,  mask_occ=None, segmentation_map = None,grid_scene = None, grid_targ = None, pc_scene = None, pc_targ=None, occ_targ=None, complete_target_tsdf=None, complete_target_pc=None):
    path = scene_file_path(root, scene_id)
    assert not path.exists()
    if '_c_' in scene_id:
        np.savez_compressed(path, depth_imgs=depth_imgs, extrinsics=extrinsics, mask_targ=mask_targ, mask_occ = mask_occ,\
                            segmentation_map = segmentation_map, grid_scene = grid_scene, grid_targ = grid_targ, pc_scene = pc_scene, pc_targ = pc_targ,occ_targ=occ_targ)
    elif '_s_' in scene_id:
        np.savez_compressed(path, depth_imgs=depth_imgs, extrinsics=extrinsics, 
            mask_targ=mask_targ, mask_occ = mask_occ, grid_scene = grid_scene, grid_targ = grid_targ, pc_scene = pc_scene,pc_targ = pc_targ,occ_targ=occ_targ, complete_target_tsdf = complete_target_tsdf, complete_target_pc = complete_target_pc)
    elif '_d_' in scene_id:
        np.savez_compressed(path, depth_imgs=depth_imgs, extrinsics=extrinsics, mask_targ=mask_targ, mask_occ = mask_occ, \
                            grid_scene = grid_scene, grid_targ = grid_targ, pc_scene = pc_scene, pc_targ = pc_targ,occ_targ=occ_targ)
    
def write_clutter_sensor_data(root, scene_id, depth_imgs, extrinsics, mask_targ,mask_scene,  \
                               segmentation_map = None,grid_scene = None, grid_targ = None,\
                                  pc_depth_scene = None, pc_depth_targ = None, pc_depth_scene_no_targ = None,
                                  pc_scene = None, pc_targ=None, pc_scene_no_targ = None, occ_targ=None):
    path = scene_file_path(root, scene_id)
    assert not path.exists()
    assert '_c_' in scene_id, 'scene_id should have _c_ in it'
    np.savez_compressed(path, depth_imgs=depth_imgs, extrinsics=extrinsics, mask_targ=mask_targ,mask_scene=mask_scene, \
                            segmentation_map = segmentation_map, grid_scene = grid_scene, grid_targ = grid_targ,\
                            pc_depth_scene = pc_depth_scene, pc_depth_targ = pc_depth_targ, pc_depth_scene_no_targ = pc_depth_scene_no_targ,\
                            pc_scene = pc_scene, pc_targ = pc_targ, pc_scene_no_targ = pc_scene_no_targ, occ_targ=occ_targ)

def write_single_scene_data(root, scene_id, depth_imgs, extrinsics, mask_targ,  grid_scene = None, grid_targ = None,\
                             pc_depth_scene = None, pc_depth_targ = None,
                             pc_scene = None, pc_targ=None, occ_targ=None, complete_target_tsdf=None, complete_target_pc=None):
    path = scene_file_path(root, scene_id)
    assert not path.exists()
    assert '_s_' in scene_id, 'scene_id should have _s_ in it'
    np.savez_compressed(path, depth_imgs=depth_imgs, extrinsics=extrinsics,
            mask_targ=mask_targ, grid_scene = grid_scene, grid_targ = grid_targ, \
                pc_depth_scene = pc_depth_scene, pc_depth_targ = pc_depth_targ, \
                pc_scene = pc_scene,pc_targ = pc_targ,occ_targ=occ_targ, complete_target_tsdf = complete_target_tsdf, complete_target_pc = complete_target_pc)

def write_double_scene_data(root, scene_id, depth_imgs, extrinsics, mask_targ, grid_scene = None, grid_targ = None, pc_scene = None, pc_targ=None, occ_targ=None):
    path = scene_file_path(root, scene_id)
    assert not path.exists()
    assert '_d_' in scene_id, 'scene_id should have _d_ in it'
    np.savez_compressed(path, depth_imgs=depth_imgs, extrinsics=extrinsics, mask_targ=mask_targ,\
                            grid_scene = grid_scene, grid_targ = grid_targ, pc_scene = pc_scene, pc_targ = pc_targ,occ_targ=occ_targ)


def write_full_sensor_data(root, scene_id, depth_imgs, extrinsics, segmentation):
    path = root / "shape_completion" / (scene_id + ".npz")
    assert not path.exists()
    np.savez_compressed(path, depth_imgs=depth_imgs, extrinsics=extrinsics, segmentation=segmentation)
    return scene_id

def read_set_theory_sensor_data(root, scene_id):
    data = np.load(scene_file_path(root, scene_id), allow_pickle=True)
    return data["depth_imgs"], data["extrinsics"], data["mask_targ"]


def read_set_theory_occluder_sensor_data(root, scene_id):
    data = np.load(scene_file_path(root, scene_id), allow_pickle=True)
    return data["depth_imgs"], data["extrinsics"], data["mask_targ"], data["mask_occ"]

def read_sensor_data(root, scene_id):
    data = np.load(scene_file_path(root, scene_id), allow_pickle=True)
    return data["depth_imgs"], data["extrinsics"]

def read_full_sensor_data(root, scene_id):
    data = np.load(root / "full_scenes" / (scene_id + ".npz"), allow_pickle=True)
    return data["depth_imgs"], data["extrinsics"], data["segmentation"]


def write_grasp(root, scene_id, grasp, label):
    csv_path = root / "grasps.csv"
    if not csv_path.exists():
        create_csv(
            csv_path,
            ["scene_id", "qx", "qy", "qz", "qw", "x", "y", "z", "width", "label"],
        )
    qx, qy, qz, qw = grasp.pose.rotation.as_quat()
    x, y, z = grasp.pose.translation
    width = grasp.width
    append_csv(csv_path, scene_id, qx, qy, qz, qw, x, y, z, width, label)


def read_grasp(df, i):
    scene_id = df.loc[i, "scene_id"]
    orientation = Rotation.from_quat(df.loc[i, "qx":"qw"].to_numpy(np.double))
    position = df.loc[i, "x":"z"].to_numpy(np.double)
    width = df.loc[i, "width"]
    label = df.loc[i, "label"]
    grasp = Grasp(Transform(orientation, position), width)
    return scene_id, grasp, label


def read_df_filtered(root):
    # return pd.read_csv(root / "grasps_cropped.csv",)
    return pd.read_csv(root / "grasps_processed.csv",)

def read_df_resized(root):
    # return pd.read_csv(root / "grasps_cropped.csv",)
    return pd.read_csv(root / "grasps_resized.csv",)

def read_df(root):
    return pd.read_csv(root / "grasps.csv",)

def read_df_prev(root):
    return pd.read_csv(root / "grasps_prev.csv",)

def read_df_no_double(root):
    return pd.read_csv(root / "grasps_no_double.csv",)

def write_df(df, root):
    df.to_csv(root / "grasps.csv", index=False)

def write_df_processed(df, root):
    df.to_csv(root / "grasps_processed.csv", index=False)

def write_df_resized(df, root):
    df.to_csv(root / "grasps_resized.csv", index=False)

def write_df_no_double(df, root):
    df.to_csv(root / "grasps_no_double.csv", index=False)

def write_df_filtered(df, root):
    df.to_csv(root / "grasps_cropped.csv", index=False)

def write_set_theory_voxel_grid(root, scene_id, voxel_grid, target_mask_grid):
    path = scene_file_path(root, scene_id)
    np.savez_compressed(path, grid=voxel_grid, mask = target_mask_grid)

def write_set_theory_occluder_voxel_grid(root, scene_id, voxel_grid, target_mask_grid, occ_mask_grid):
    path = scene_file_path(root, scene_id)
    np.savez_compressed(path, grid=voxel_grid, targ_mask = target_mask_grid, occ_mask = occ_mask_grid)

def write_voxel_grid(root, scene_id, voxel_grid):
    path = scene_file_path(root, scene_id)
    np.savez_compressed(path, grid=voxel_grid)


def write_scene_targ_pc(root, scene_id, scene_pc, targ_pc, name="point_clouds"):
    path = root / name / (scene_id + ".npz")
    # point_cloud = np.array(point_cloud, dtype=object)
    # path = scene_file_path(root, scene_id)
    scene_pc = np.array(scene_pc, dtype=object)
    targ_pc = np.array(targ_pc, dtype=object)
    np.savez_compressed(path, scene_pc=scene_pc, targ_pc=targ_pc)

def write_point_cloud(root, scene_id, point_cloud, name="point_clouds"):
    path = root / name / (scene_id + ".npz")
    point_cloud = np.array(point_cloud, dtype=object)
    np.savez_compressed(path, pc=point_cloud)

def write_test_set_point_cloud(root, scene_id, point_cloud, name="test_set"):
    path = root / name /(scene_id + ".npz")
    point_cloud = np.array(point_cloud, dtype=object)
    np.savez_compressed(path, pc=point_cloud)

def read_voxel_grid(root, scene_id):
    path = scene_file_path(root, scene_id)
    return np.load(path, allow_pickle=True)["grid"]

def read_targ_depth_pc(root, scene_id):
    # path = root / "point_clouds" / (scene_id + ".npz")
    # path = root / "depth_pc" / (scene_id + ".npz")
    path = scene_file_path(root, scene_id)
    # targ_pc = np.load(path)["targ_pc"]
    # targ_pc = np.load(path)["targ_pc.npy"]
    # targ_pc = np.load(path, allow_pickle=True)["targ_pc"]
    targ_pc = np.load(path, allow_pickle=True)["pc_depth_targ"]

    return targ_pc

def read_scene_depth_pc(root, scene_id):
    # path = root / "point_clouds" / (scene_id + ".npz")
    # path = root / "depth_pc" / (scene_id + ".npz")
    path = scene_file_path(root, scene_id)
    # targ_pc = np.load(path)["targ_pc"]
    # targ_pc = np.load(path)["targ_pc.npy"]
    # scene_pc = np.load(path, allow_pickle=True)["scene_pc"]
    scene_pc = np.load(path, allow_pickle=True)["pc_depth_scene"]
    return scene_pc

# def read_targ_pc(root, scene_id):
#     # path = root / "point_clouds" / (scene_id + ".npz")
#     path = scene_file_path(root, scene_id)
#     # targ_pc = np.load(path)["targ_pc"]
#     # targ_pc = np.load(path)["targ_pc.npy"]
#     # targ_pc = np.load(path, allow_pickle=True)["targ_pc"]
#     ## assert when no pc_targ
#     assert 'pc_targ' in np.load(path, allow_pickle=True), f"No pc_targ found in {scene_id}.npz"
#     targ_pc = np.load(path, allow_pickle=True)["pc_targ"]

#     return targ_pc

def read_targ_pc(root, scene_id):
    """
    Try to read 'pc_targ' from <root>/scenes/<scene_id>.npz.
    If not found, append scene_id to invalid_scene_id.txt and return None.
    """
    from pathlib import Path

    path = scene_file_path(root, scene_id)
    invalid_file = "./scripts/invalid_scene_id.txt"

    try:
        with np.load(path, allow_pickle=True) as data:
            return data["pc_targ"]
    except (FileNotFoundError, KeyError):
        # Append scene_id to invalid_scene_id.txt (create if not exists)
        with open(invalid_file, "a", encoding="utf-8") as f:
            f.write(f"{scene_id}\n")
        return None

def read_scene_no_targ_pc(root, scene_id):
    # path = root / "point_clouds" / (scene_id + ".npz")
    # path = root / "scene_no_targ_pc" / (scene_id + ".npz")
    # targ_pc = np.load(path)["targ_pc"]
    path = scene_file_path(root, scene_id)
    # targ_pc = np.load(path)["targ_pc.npy"]
    # scene_pc = np.load(path, allow_pickle=True)["scene_pc"]
    
    try:
        data = np.load(path, allow_pickle=True)
        
        # For clutter scenes (_c_), use pc_scene_no_targ
        if '_c_' in scene_id:
            if 'pc_scene_no_targ' in data:
                scene_no_targ_pc = data["pc_scene_no_targ"]
            else:
                # Fallback: use pc_depth_scene_no_targ if available
                if 'pc_depth_scene_no_targ' in data:
                    scene_no_targ_pc = data["pc_depth_scene_no_targ"]
                else:
                    available_keys = list(data.keys())
                    data.close()
                    raise KeyError(f"Neither 'pc_scene_no_targ' nor 'pc_depth_scene_no_targ' found in clutter scene {scene_id}.npz. Available keys: {available_keys}")
        
        # For single scenes (_s_), use pc_scene (since there's no "scene without target" concept)
        elif '_s_' in scene_id:
            if 'pc_scene' in data:
                scene_no_targ_pc = data["pc_scene"]
            else:
                # Fallback: use pc_depth_scene if available
                if 'pc_depth_scene' in data:
                    scene_no_targ_pc = data["pc_depth_scene"]
                else:
                    available_keys = list(data.keys())
                    data.close()
                    raise KeyError(f"Neither 'pc_scene' nor 'pc_depth_scene' found in single scene {scene_id}.npz. Available keys: {available_keys}")
        
        # For other scene types, try pc_scene_no_targ first
        else:
            if 'pc_scene_no_targ' in data:
                scene_no_targ_pc = data["pc_scene_no_targ"]
            elif 'pc_scene' in data:
                scene_no_targ_pc = data["pc_scene"]
            else:
                available_keys = list(data.keys())
                data.close()
                raise KeyError(f"No suitable scene point cloud found in {scene_id}.npz. Available keys: {available_keys}")
        
        data.close()
        return scene_no_targ_pc
        
    except Exception as e:
        print(f"Error loading scene {scene_id} from {path}: {e}")
        # Check if file exists
        if not path.exists():
            print(f"File does not exist: {path}")
        else:
            print(f"File exists but cannot be loaded properly")
        raise

def read_scene_pc(root, scene_id):
    # path = root / "point_clouds" / (scene_id + ".npz")
    path = scene_file_path(root, scene_id)
    # targ_pc = np.load(path)["targ_pc"]
    # targ_pc = np.load(path)["targ_pc.npy"]
    # scene_pc = np.load(path, allow_pickle=True)["scene_pc"]
    scene_pc = np.load(path, allow_pickle=True)["pc_scene"]
    return scene_pc


def read_voxel_grid_set_theory(root, scene_id):
    path = scene_file_path(root, scene_id)
    depth_imgs = np.load(path)["depth_imgs.npy"]
    extrinsics = np.load(path)["extrinsics.npy"]
    mask_targ =  np.load(path)["mask_targ.npy"]
    return depth_imgs, mask_targ, extrinsics

def read_voxel_and_mask(root, scene_id):
    path = scene_file_path(root, scene_id)
    depth_imgs = np.load(path)["grid.npy"]
    mask_targ =  np.load(path)["mask.npy"]
    return depth_imgs, mask_targ

def read_voxel_and_mask_occluder(root, scene_id):
    path = scene_file_path(root, scene_id)
    # depth_img = np.load(path)["grid"]
    # mask_targ =  np.load(path)["targ_mask"]
    # mask_occ = np.load(path)["occ_mask"]
    # return depth_img, mask_targ, mask_occ
    try:
        data = np.load(path, allow_pickle=True)
        
        # Check if required keys exist
        if "grid_scene" not in data:
            available_keys = list(data.keys())
            print(f"Warning: Skipping {scene_id}.npz - 'grid_scene' not found. Available keys: {available_keys}")
            data.close()
            return None, None
        
        if "grid_targ" not in data:
            available_keys = list(data.keys())
            print(f"Warning: Skipping {scene_id}.npz - 'grid_targ' not found. Available keys: {available_keys}")
            data.close()
            return None, None
        
        grid_scene = data["grid_scene"]
        grid_targ = data["grid_targ"]
        data.close()
        return grid_scene, grid_targ
        
    except Exception as e:
        print(f"Warning: Error loading {scene_id}.npz - {e}")
        return None, None

def read_depth_mask(root, scene_id):
    path = scene_file_path(root, scene_id)
    depth_img = np.load(path)["depth_imgs.npy"]
    mask_targ =  np.load(path)["mask_targ.npy"]
    return depth_img, mask_targ
    # return depth_img, mask_targ, mask_occ

def read_single_complete_target(root, scene_id):
    path = scene_file_path(root, scene_id)
    targ_grid = np.load(path)["complete_target_tsdf"]
    targ_pc = np.load(path)["complete_target_pc"]
    return targ_grid, targ_pc

def read_voxel_grid_complete_shape(root, scene_id):
    path = root / "complete_tsdf" / (scene_id + ".npy")
    return np.load(path)

def read_point_cloud(root, scene_id, name="point_clouds"):
    path = root / name / (scene_id + ".npz")
    return np.load(path)["pc"]

def read_json(path):
    with path.open("r") as f:
        data = json.load(f)
    return data


def write_json(data, path):
    with path.open("w") as f:
        json.dump(data, f, indent=4)


def create_csv(path, columns):
    with path.open("w") as f:
        f.write(",".join(columns))
        f.write("\n")


def append_csv(path, *args):
    row = ",".join([str(arg) for arg in args])
    with path.open("a") as f:
        f.write(row)
        f.write("\n")

def read_complete_target_pc(root, scene_id):
    """Read complete target point cloud from preprocessed data."""
    path = scene_file_path(root, scene_id)
    try:
        data = np.load(path, allow_pickle=True)
        if 'complete_target_pc' in data:
            return data['complete_target_pc']
        else:
            return None
    except:
        return None

def read_complete_target_mesh(root, scene_id):
    """Read complete target mesh from preprocessed data."""
    path = scene_file_path(root, scene_id)
    try:
        data = np.load(path, allow_pickle=True)
        if 'complete_target_mesh_vertices' in data and 'complete_target_mesh_faces' in data:
            import trimesh
            vertices = data['complete_target_mesh_vertices']
            faces = data['complete_target_mesh_faces']
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            return mesh
        else:
            return None
    except:
        return None

def check_complete_target_available(root, scene_id):
    """Check if complete target data is available for a scene."""
    path = scene_file_path(root, scene_id)
    try:
        data = np.load(path, allow_pickle=True)
        return 'complete_target_pc' in data
    except:
        return False
