import numpy as np
from scipy import ndimage
import torch.utils.data
import torch
import random
import os
from collections import defaultdict
from pathlib import Path
import collections
import argparse
import json
import math
import pyrender
import time
import tqdm
import trimesh
import uuid
import shutil
import matplotlib.pyplot as plt
import clip
from datetime import datetime
from sys import path
path.append('.')

from src.vgn.io import *
from src.vgn.perception import *
from src.vgn.utils.transform import Rotation, Transform
from src.vgn.utils.implicit import get_scene_from_mesh_pose_list
from utils_giga import *

from vgn.io import *
from vgn.perception import *
from vgn.utils.transform import Rotation, Transform
from vgn.utils.implicit import get_scene_from_mesh_pose_list
import re
from shape_completion.data_transforms import Compose

transform = Compose([{
        'callback': 'UpSamplePoints',
        'parameters': {
            'n_points': 2048
        },
        'objects': ['input']
    }, {
        'callback': 'ToTensor',
        'objects': ['input']
    }])

# Global CLIP model for feature extraction
_clip_model = None
_clip_preprocess = None
_clip_device = None

def get_clip_model():
    """Get or initialize CLIP model for feature extraction."""
    global _clip_model, _clip_preprocess, _clip_device
    
    if _clip_model is None:
        _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=_clip_device)
        print(f"CLIP model loaded on {_clip_device}")
    
    return _clip_model, _clip_preprocess, _clip_device

def extract_clip_features(text_prompt, model, device):
    """Extract CLIP features for a given text prompt."""
    text_inputs = torch.cat([clip.tokenize(text_prompt)]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        return text_features.cpu().numpy().flatten()

def transform_pc(pc):
    # device = pc.device
    # pc = pc.cpu().numpy()
    # BS = pc.shape[0]
    # pc_transformed = torch.zeros((BS, 2048, 3), dtype=torch.float32)
    # for i in range(BS):
    points_curr_transformed = transform({'input':pc})
        # pc_transformed[i] = points_curr_transformed['input']
    return points_curr_transformed['input']

# Global error log file for recording problematic scene_ids
ERROR_LOG_FILE_TARGO_FULL = Path("./data/_check_results/full_train/dataset_error_scenes_targo_full.txt")
ERROR_LOG_FILE_PTV3_SCENE = Path("./data/_check_results/full_train/dataset_error_scenes_ptv3_scene.txt")

def safe_specify_num_points(points, target_size, scene_id, point_type="unknown"):
    """
    Safe wrapper for specify_num_points that handles empty point clouds.
    
    Args:
        points: Input point cloud array
        target_size: Target number of points
        scene_id: Scene identifier for logging
        point_type: Type of points (e.g., "target", "scene", "occluder")
    
    Returns:
        Processed point cloud or None if error
    """
    try:
        if points.size == 0:
            raise ValueError(f"No points in the scene for {point_type}")
        if points.shape[0] == 0:
            raise ValueError(f"Empty {point_type} point cloud")
        
        # Import here to avoid circular imports
        from utils_giga import specify_num_points
        return specify_num_points(points, target_size)
        
    except Exception as e:
        # Log error to file
        ERROR_LOG_FILE_TARGO_FULL.parent.mkdir(parents=True, exist_ok=True)
        with ERROR_LOG_FILE_TARGO_FULL.open("a", encoding="utf-8") as f:
            f.write(f"{scene_id},{point_type},\"{str(e)}\"\n")
        
        print(f"[ERROR] Scene {scene_id}: {point_type} point cloud error - {e}")
        return None

class DatasetVoxel_Target(torch.utils.data.Dataset):
    def __init__(self, root, raw_root, num_point=2048, augment=False, ablation_dataset="",  model_type="giga_aff",
                 data_contain="pc", add_single_supervision=False, decouple = False,use_complete_targ = False,\
                input_points = 'tsdf_points', shape_completion = False, vis_data = False, logdir = None):
        self.root = root
        self.augment = augment
        self.num_point = num_point
        self.raw_root = raw_root
        self.num_th = 32
        self.use_complete_targ = use_complete_targ
        self.model_type = model_type
        if model_type == "vgn":
            self.df = read_df_filtered(raw_root)
        else:
            self.df = read_df(raw_root)
        # self.df = read_df_filtered(raw_root)
        # self.df = self.df[:300]
        if ablation_dataset == 'only_cluttered':
            self.df = filter_rows_by_id_only_clutter(self.df)

        if ablation_dataset == '1_10':
            self.df = self.df.sample(frac=0.1)
            self.df = self.df.reset_index(drop=True)
        
        if ablation_dataset == '1_100':
            self.df = self.df.sample(frac=0.01)
            self.df = self.df.reset_index(drop=True)
        if ablation_dataset == '1_100000':
            self.df = self.df.sample(frac=0.00001)
            self.df = self.df.reset_index(drop=True)
        
        skip_scene_ids = ['b960209b0cbd406d98dac25aeccd3c71_s_2','5522fdbe62d9450687a195cfd2bbbac3_c_1', 'b960209b0cbd406d98dac25aeccd3c71_c_2']
        self.df = self.df[~self.df['scene_id'].isin(skip_scene_ids)]
        # self.df = self.df[~self.df.scene_id.isin(skip_scene_ids)]
        self.df = self.df.reset_index(drop=True)


        print("data frames stastics")
        print_and_count_patterns(self.df,False)

        self.size, _, _, _ = read_setup(raw_root)
        self.data_contain = data_contain
        self.add_single_supervision = add_single_supervision
        self.decouple = decouple
        self.input_points = input_points
        self.shape_completion = shape_completion
        self.vis_data = vis_data

        if self.vis_data:
            vis_logdir = logdir / 'vis_data'
            if not vis_logdir.exists():
                vis_logdir.mkdir(exist_ok=True, parents=True)
            self.vis_logdir = vis_logdir


    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, i):
        scene_id = self.df.loc[i, "scene_id"]
        
        # Add retry mechanism for incomplete scenes
        max_retries = 10
        for attempt in range(max_retries):
            try:
                return self._get_item_safe(i, scene_id)
            except Exception as e:
                # Log the error
                ERROR_LOG_FILE_TARGO_FULL.parent.mkdir(parents=True, exist_ok=True)
                with ERROR_LOG_FILE_TARGO_FULL.open("a", encoding="utf-8") as f:
                    f.write(f"{scene_id},dataset_loading,\"attempt_{attempt+1}: {str(e)}\"\n")
                
                print(f"[WARNING] Scene {scene_id} loading failed (attempt {attempt+1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    # Try next scene
                    i = (i + 1) % len(self.df)
                    scene_id = self.df.loc[i, "scene_id"]
                else:
                    # Final attempt failed, raise the error
                    raise e
    
    def _get_item_safe(self, i, scene_id):
        """Safe version of __getitem__ with proper error handling."""
        # if not os.path.exists(os.path.join(self.root, 'scenes', scene_id)):
        #     print("Error")
        ori = Rotation.from_quat(self.df.loc[i, "qx":"qw"].to_numpy(np.single))
        if not self.model_type == "vgn":
            pos = self.df.loc[i, "x":"z"].to_numpy(np.single)
            width =  np.float32(self.df.loc[i, "width"])
            label = self.df.loc[i, "label"].astype(np.int64)
        else:
            pos = self.df.loc[i, "i":"k"].to_numpy(np.single)
            width = self.df.loc[i, "width"].astype(np.single)
            label = self.df.loc[i, "label"].astype(np.int64)

        if self.use_complete_targ:
            single_scene_id = (scene_id.split('_')[0]) + '_s_' + scene_id.split('_')[2]

        
        if self.data_contain == "pc and targ_grid":
            voxel_grid, targ_grid = read_voxel_and_mask_occluder(self.raw_root, scene_id)
            if self.vis_data:
                vis_path = str(self.vis_logdir / 'voxel_grid.png')
                visualize_and_save_tsdf(voxel_grid[0], vis_path)
                vis_path = str(self.vis_logdir / 'targ_grid.png')
                visualize_and_save_tsdf(targ_grid[0], vis_path)

            if not self.shape_completion:
                if self.input_points == "tsdf_points":
                    # assert os.path.exists(self.raw_root / "scenes" / (scene_id + ".npz")), f"Scene {scene_id} not found in {self.raw_root}"

                    targ_pc = read_targ_pc(self.raw_root, scene_id).astype(np.float32)
                    scene_pc = read_scene_pc(self.raw_root, scene_id).astype(np.float32)
                    targ_pc = points_within_boundary(targ_pc)
                    scene_pc = points_within_boundary(scene_pc)
                
                elif self.input_points == "depth_target_others_tsdf":
                    if '_c_' in scene_id:
                        targ_pc = read_targ_depth_pc(self.raw_root, scene_id).astype(np.float32)
                        scene_no_targ_pc = read_scene_no_targ_pc(self.raw_root, scene_id).astype(np.float32)
                        # scene_no_targ_pc = np.concatenate((scene_no_targ_pc, np.load('./setup/plane_sampled.npy')), axis=0)
                        targ_pc = points_within_boundary(targ_pc)
                        
                        # Safe point cloud processing
                        targ_pc = safe_specify_num_points(targ_pc, 2048, scene_id, "target")
                        if targ_pc is None:
                            raise ValueError(f"Failed to process target point cloud for scene {scene_id}")
                        
                        scene_no_targ_pc = points_within_boundary(scene_no_targ_pc)
                        scene_no_targ_pc = np.concatenate((scene_no_targ_pc, np.load('./setup/plane_sampled.npy')), axis=0)
                        scene_no_targ_pc = safe_specify_num_points(scene_no_targ_pc, 2048, scene_id, "scene_no_target")
                        if scene_no_targ_pc is None:
                            raise ValueError(f"Failed to process scene_no_target point cloud for scene {scene_id}")
                        
                        scene_pc = np.concatenate((scene_no_targ_pc, targ_pc))
                    elif '_s_' in scene_id:
                        targ_pc = read_targ_depth_pc(self.raw_root, scene_id).astype(np.float32)
                        targ_pc = points_within_boundary(targ_pc)
                        
                        targ_pc = safe_specify_num_points(targ_pc, 2048, scene_id, "target")
                        if targ_pc is None:
                            raise ValueError(f"Failed to process target point cloud for scene {scene_id}")
                        
                        scene_pc = targ_pc
                        scene_pc = safe_specify_num_points(scene_pc, 4096, scene_id, "scene")
                        if scene_pc is None:
                            raise ValueError(f"Failed to process scene point cloud for scene {scene_id}")


            if self.shape_completion:
                targ_pc = read_targ_pc(self.raw_root, scene_id).astype(np.float32)
                targ_pc = points_within_boundary(targ_pc)
                
                targ_pc = safe_specify_num_points(targ_pc, 2048, scene_id, "target")
                if targ_pc is None:
                    raise ValueError(f"Failed to process target point cloud for scene {scene_id}")
                
                scene_pc = np.load('./setup/plane_sampled.npy')
                num_scene = scene_pc.shape[0] + 2048
                if '_c_' in scene_id:
                    scene_no_targ_pc = read_scene_no_targ_pc(self.raw_root, scene_id).astype(np.float32)
                    scene_no_targ_pc = points_within_boundary(scene_no_targ_pc)
                    scene_pc = np.concatenate((scene_no_targ_pc, scene_pc), axis=0)
                
                scene_pc = safe_specify_num_points(scene_pc, num_scene, scene_id, "scene_with_plane")
                if scene_pc is None:
                    raise ValueError(f"Failed to process scene_with_plane point cloud for scene {scene_id}")

            elif self.input_points == "depth_bp": 
                targ_pc = read_targ_depth_pc(self.raw_root, scene_id).astype(np.float32)
                scene_pc = read_scene_depth_pc(self.raw_root, scene_id).astype(np.float32)
                targ_pc = points_within_boundary(targ_pc)
                scene_pc = points_within_boundary(scene_pc)
         
            if self.vis_data:
                vis_path = str(self.vis_logdir / 'targ_pc.ply')
                save_point_cloud_as_ply(targ_pc, vis_path)
                vis_path = str(self.vis_logdir / 'scene_pc.ply')
                save_point_cloud_as_ply(scene_pc, vis_path)


            # if self.decouple:
            #     voxel_grid = voxel_grid - targ_grid

            elif self.use_complete_targ:
                # Read complete target data directly from current scene file
                scene_single_path = self.raw_root / "scenes" / f"{single_scene_id}.npz"
                # miss_log = Path("./miss_complete_target_tsdf.txt")

                # try:
                with np.load(scene_single_path, allow_pickle=True) as data_single:
                    targ_grid = data_single["complete_target_tsdf"]
                # except (FileNotFoundError, KeyError) as e:
                #     miss_log.parent.mkdir(parents=True, exist_ok=True)
                #     with miss_log.open("a", encoding="utf-8") as f:
                #         f.write(f"{scene_id}\n")
                #     # 出错的话，跳过当前循环
                #     print(f"Warning: skip scene {scene_id} (missing complete_target_tsdf): {e}")
                    # return None
            
            # Check point cloud validity
            if hasattr(self, 'targ_pc') and targ_pc is not None:
                if targ_pc.shape[0] == 0:
                    raise ValueError(f"Empty target point cloud for scene {scene_id}")
            if hasattr(self, 'scene_pc') and scene_pc is not None:
                if scene_pc.shape[0] == 0:
                    raise ValueError(f"Empty scene point cloud for scene {scene_id}")


            if self.model_type == "targo_full" or self.model_type == "targo":
                targ_pc = safe_specify_num_points(targ_pc, 2048, scene_id, "final_target")
                if targ_pc is None:
                    raise ValueError(f"Failed to process final target point cloud for scene {scene_id}")

                scene_pc = safe_specify_num_points(scene_pc, 2048, scene_id, "final_scene")
                if scene_pc is None:
                    raise ValueError(f"Failed to process final scene point cloud for scene {scene_id}")

        if self.data_contain == "pc":
            voxel_grid, _, = read_voxel_and_mask_occluder(self.root, scene_id)
            if self.use_complete_targ == True:
                voxel_grid, targ_grid = read_voxel_and_mask_occluder(self.raw_root, scene_id)
                targ_complete_grid, _ = read_single_complete_target(self.raw_root, single_scene_id)
                voxel_no_targ_grid = voxel_grid - targ_grid
                voxel_grid = voxel_no_targ_grid + targ_complete_grid
            if self.vis_data:
                vis_path = str(self.vis_logdir / 'voxel_grid.png')
                visualize_and_save_tsdf(voxel_grid[0], vis_path)
                
        if self.augment:
            voxel_grid, ori, pos = apply_transform(voxel_grid, ori, pos)

        if self.model_type != "vgn":
        
            pos = pos / self.size - 0.5
            width = width / self.size

            rotations = np.empty((2, 4), dtype=np.single)
            R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
            rotations[0] = ori.as_quat()
            rotations[1] = (ori * R).as_quat()
        else:
            index = np.round(pos).astype(np.int64)
            rotations = np.empty((2, 4), dtype=np.single)
            R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
            rotations[0] = ori.as_quat()
            rotations[1] = (ori * R).as_quat()

        if self.data_contain == "pc and targ_grid":
            if not self.shape_completion:
                plane = np.load('./setup/plane_sampled.npy')
                if not ('_s_' in scene_id and self.shape_completion):
                    scene_pc = np.concatenate((scene_pc, plane), axis=0)
                    targ_pc = targ_pc /0.3- 0.5
                    scene_pc = scene_pc /0.3- 0.5
                elif '_s_' in scene_id and self.shape_completion:
                    scene_pc = plane
                    scene_pc = safe_specify_num_points(scene_pc, 2048 + plane.shape[0], scene_id, "plane_scene")
                    if scene_pc is None:
                        raise ValueError(f"Failed to process plane scene point cloud for scene {scene_id}")
                    targ_pc = targ_pc /0.3- 0.5
                    scene_pc = scene_pc /0.3- 0.5
            elif self.shape_completion:
                scene_pc = scene_pc /0.3- 0.5
                targ_pc = targ_pc /0.3- 0.5

            # Apply filter_and_pad_point_clouds to ensure coordinates are within valid range
            # This prevents MinkowskiEngine negative coordinate errors
            targ_pc_tensor = torch.from_numpy(targ_pc).unsqueeze(0).float()
            scene_pc_tensor = torch.from_numpy(scene_pc).unsqueeze(0).float()
            
            # if self.model_type == "targo_full_targ" or self.model_type == "targo":
            #     targ_pc_filtered = filter_and_pad_point_clouds(targ_pc_tensor)
            #     scene_pc_filtered = filter_and_pad_point_clouds(scene_pc_tensor)
            
            targ_pc = targ_pc_tensor.squeeze(0).numpy()
            scene_pc = scene_pc_tensor.squeeze(0).numpy()

            if self.model_type != "ptv3_scene":
                x = (voxel_grid[0], targ_grid[0], targ_pc, scene_pc)
            else:
                # For ptv3_scene, combine scene and target into one point cloud with labels
                combined_pc = np.concatenate([scene_pc, targ_pc], axis=0)
                x = (voxel_grid[0], targ_grid[0], targ_pc, combined_pc)

        if self.data_contain == "pc":
            if self.model_type == "vgn":
                x = (voxel_grid)
            else:
                x = (voxel_grid[0])


        y = (label, rotations, width)
        
        if self.model_type == "vgn":
            return x, y, index
        else:
            return x, y, pos

    def get_mesh(self, idx):
        scene_id = self.df.loc[idx, "scene_id"]
        mesh_pose_list_path = self.raw_root / 'mesh_pose_list' / (scene_id + '.npz')
        mesh_pose_list = np.load(mesh_pose_list_path, allow_pickle=True)['pc']
        scene = get_scene_from_mesh_pose_list(mesh_pose_list, return_list=False)
        return scene

class DatasetVoxel_PTV3_Clip(torch.utils.data.Dataset):
    def __init__(self, root, raw_root, num_point=2048, augment=False, ablation_dataset="",  model_type="giga_aff",
            use_complete_targ = False, debug = False, logdir = None):
        assert model_type == "ptv3_clip"
        self.root = root
        self.augment = augment
        self.num_point = num_point
        self.raw_root = raw_root
        self.num_th = 32
        self.use_complete_targ = use_complete_targ
        self.model_type = model_type
        self.debug = debug

        scenes_ptv3_clip_path = raw_root / 'scenes_ptv3_clip'
        if not scenes_ptv3_clip_path.exists():
            scenes_ptv3_clip_path.mkdir(parents=True, exist_ok=True)
        self.scenes_ptv3_clip_path = scenes_ptv3_clip_path

        scene_ptv3_clip_feat_path = raw_root / 'scene_ptv3_clip_feat'
        if not scene_ptv3_clip_feat_path.exists():
            scene_ptv3_clip_feat_path.mkdir(parents=True, exist_ok=True)
        self.scene_ptv3_clip_feat_path = scene_ptv3_clip_feat_path

        scene_ptv3_clip_final = raw_root / 'scene_ptv3_clip_final'
        if not scene_ptv3_clip_final.exists():
            scene_ptv3_clip_final.mkdir(parents=True, exist_ok=True)
        self.scene_ptv3_clip_final = scene_ptv3_clip_final
        
        self.df = read_df(raw_root)
        # self.df = read_df_filtered(raw_root)
        # self.df = self.df[:300]
        if ablation_dataset == 'only_cluttered':
            self.df = filter_rows_by_id_only_clutter(self.df)

        if ablation_dataset == '1_10':
            self.df = self.df.sample(frac=0.1)
            self.df = self.df.reset_index(drop=True)
        
        if ablation_dataset == '1_100':
            self.df = self.df.sample(frac=0.01)
            self.df = self.df.reset_index(drop=True)
        if ablation_dataset == '1_100000':
            self.df = self.df.sample(frac=0.00001)
            self.df = self.df.reset_index(drop=True)

        print("data frames stastics")
        print_and_count_patterns(self.df,False)
        self.scene_category_dict = json.load(open(f'{self.raw_root}/category_scene_dict.json'))
        ## filter dirty data
        skip_scene_ids = ['b960209b0cbd406d98dac25aeccd3c71_s_2','5522fdbe62d9450687a195cfd2bbbac3_c_1', 'b960209b0cbd406d98dac25aeccd3c71_c_2']
        self.df = self.df[~self.df['scene_id'].isin(skip_scene_ids)]
        # self.df = self.df[~self.df.scene_id.isin(skip_scene_ids)]
        self.df = self.df.reset_index(drop=True)

        self.size, _, _, _ = read_setup(raw_root)

        # Add visualization support
        if self.debug:
            vis_logdir = logdir / 'vis_data'
            if not vis_logdir.exists():
                vis_logdir.mkdir(exist_ok=True, parents=True)
            self.vis_logdir = vis_logdir

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, i):
        scene_id = self.df.loc[i, "scene_id"]
        self.curr_scene_path = self.scenes_ptv3_clip_path / f"{scene_id}.npz"
        self.curr_scene_clip_feat_path = self.scene_ptv3_clip_feat_path / f"{scene_id}.npz"
        self.curr_scene_clip_final_path = self.scene_ptv3_clip_final / f"{scene_id}.npz"
        # Add retry mechanism for incomplete scenes
        # max_retries = 10
        # for attempt in range(max_retries):
        try:
            return self._get_item_safe_ptv3_clip(i, scene_id)
        except Exception as e:
            # Log the error
            ERROR_LOG_FILE_PTV3_SCENE.parent.mkdir(parents=True, exist_ok=True)
            with ERROR_LOG_FILE_PTV3_SCENE.open("a", encoding="utf-8") as f:
                f.write(f"{scene_id},ptv3_dataset_loading,\"{str(e)}\"\n")
            
            print(f"[WARNING] PTV3 Scene {scene_id} loading failed: {e}")
            
            # if attempt < max_retries - 1:
            #     # Try next scene
            #     i = (i + 1) % len(self.df)
            #     scene_id = self.df.loc[i, "scene_id"]
            # else:
            #     # Final attempt failed, raise the error
            #     raise e
    
    def _get_item_safe_ptv3_clip(self, i, scene_id):
        # if self.curr_scene_path.exists():
        #     with np.load(self.curr_scene_path, allow_pickle=True) as data:
        #         targ_grid = data["complete_target_tsdf"]
        #         if targ_grid.ndim == 3:
        #             targ_grid = np.expand_dims(targ_grid, axis=0)
        # else:
        """Safe version of __getitem__ for PTV3_Scene with proper error handling."""
        ori = Rotation.from_quat(self.df.loc[i, "qx":"qw"].to_numpy(np.single))
        pos = self.df.loc[i, "x":"z"].to_numpy(np.single)
        width =  np.float32(self.df.loc[i, "width"])
        label = self.df.loc[i, "label"].astype(np.int64)


        rotations = np.empty((2, 4), dtype=np.single)
        R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
        rotations[0] = ori.as_quat()
        rotations[1] = (ori * R).as_quat()
        y = (label, rotations, width)

        if not self.curr_scene_clip_final_path.exists():
            if self.curr_scene_path.exists():
                scene_data = np.load(self.curr_scene_path, allow_pickle=True)
                voxel_grid = scene_data['voxel_grid']
                targ_grid = scene_data['targ_grid']
                targ_pc_with_labels = scene_data['targ_pc_with_labels']
                scene_pc_with_labels = scene_data['scene_pc_with_labels']
                # targ_pc = targ_pc_with_labels[:, :3]
                # scene_pc = scene_pc_with_labels[:, :3]
                x = (voxel_grid[0], targ_grid[0], targ_pc_with_labels, scene_pc_with_labels)

                # targ_grid = scene_data["complete_target_tsdf"]
                # if targ_grid.ndim == 3:
                #     targ_grid = np.expand_dims(targ_grid, axis=0)
                # voxel_grid = scene_data["voxel_grid"]
                # occluder_grid = voxel_grid - targ_grid
                # targ_pc = scene_data["targ_pc"]
                # with np.load(self.curr_scene_path, allow_pickle=True) as data:
                #     targ_grid = data["complete_target_tsdf"]
                #     if targ_grid.ndim == 3:
                #         targ_grid = np.expand_dims(targ_grid, axis=0)
            else:
                single_scene_id = (scene_id.split('_')[0]) + '_s_' + scene_id.split('_')[2]
                voxel_grid, targ_grid = read_voxel_and_mask_occluder(self.raw_root, scene_id)
                occluder_grid = voxel_grid - targ_grid
                targ_pc = read_complete_target_pc(self.raw_root, scene_id).astype(np.float32)
                scene_path = self.raw_root / "scenes" / f"{scene_id}.npz"
                
                # Handle missing complete_target_tsdf with error logging and fallback
                scene_single_path = self.raw_root / "scenes" / f"{single_scene_id}.npz"
                try:
                    # with np.load(scene_single_path, allow_pickle=True) as data_single:
                        # targ_grid = data_single["complete_target_tsdf"]
                    with np.load(scene_path) as data:
                        targ_grid = data["complete_target_tsdf"]
                        if targ_grid.ndim == 3:
                            targ_grid = np.expand_dims(targ_grid, axis=0)
                except KeyError as e:
                    # Log the error to file
                    ERROR_LOG_FILE_PTV3_SCENE.parent.mkdir(parents=True, exist_ok=True)
                    with ERROR_LOG_FILE_PTV3_SCENE.open("a", encoding="utf-8") as f:
                        f.write(f"{scene_id},complete_target_tsdf_missing,\"KeyError: 'complete_target_tsdf' not found in {single_scene_id}.npz\"\n")
                    
                    print(f"[WARNING] Scene {scene_id}: complete_target_tsdf missing in {single_scene_id}.npz, using fallback strategy")
                    
                    # Fallback: use original targ_grid from read_voxel_and_mask_occluder
                    # This maintains training continuity but may have lower quality complete target data
                    pass  # targ_grid already set from read_voxel_and_mask_occluder above
                    
                except Exception as e:
                    # Log other errors
                    ERROR_LOG_FILE_PTV3_SCENE.parent.mkdir(parents=True, exist_ok=True)
                    with ERROR_LOG_FILE_PTV3_SCENE.open("a", encoding="utf-8") as f:
                        f.write(f"{scene_id},scene_file_error,\"Error loading {single_scene_id}.npz: {str(e)}\"\n")
                    
                    print(f"[WARNING] Scene {scene_id}: Error loading {single_scene_id}.npz: {e}, using fallback strategy")
                    
                    # Fallback: use original targ_grid
                    pass  # targ_grid already set from read_voxel_and_mask_occluder above
                
                voxel_grid = occluder_grid + targ_grid

                plane = np.load('./setup/plane_sampled.npy')
                if '_c_' in scene_id:
                    scene_no_targ_pc = read_scene_no_targ_pc(self.raw_root, scene_id).astype(np.float32)
                    # scene_no_targ_pc = np.concatenate((scene_no_targ_pc, plane), axis=0)
                    targ_pc = points_within_boundary(targ_pc)
                    
                    targ_pc = safe_specify_num_points(targ_pc, 512, scene_id, "ptv3_target")
                    if targ_pc is None:
                        raise ValueError(f"Failed to process PTV3 target point cloud for scene {scene_id}")
                    
                    scene_no_targ_pc = points_within_boundary(scene_no_targ_pc)
                    scene_no_targ_pc = np.concatenate((scene_no_targ_pc, plane), axis=0)
                    scene_no_targ_pc = safe_specify_num_points(scene_no_targ_pc, 512, scene_id, "ptv3_scene_no_target")
                    if scene_no_targ_pc is None:
                        raise ValueError(f"Failed to process PTV3 scene_no_target point cloud for scene {scene_id}")
                    # scene_pc = np.concatenate((scene_no_targ_pc, targ_pc))
                elif '_s_' in scene_id:
                    scene_no_targ_pc = plane
                    targ_pc = points_within_boundary(targ_pc)
                    
                    targ_pc = safe_specify_num_points(targ_pc, 512, scene_id, "ptv3_target")
                    if targ_pc is None:
                        raise ValueError(f"Failed to process PTV3 target point cloud for scene {scene_id}")
                    
                    scene_no_targ_pc = safe_specify_num_points(scene_no_targ_pc, 512, scene_id, "ptv3_plane")
                    if scene_no_targ_pc is None:
                        raise ValueError(f"Failed to process PTV3 plane point cloud for scene {scene_id}")
                    # scene_pc = targ_pc
                    # scene_pc = specify_num_points(scene_pc, 512)
                if not self.scene_ptv3_clip_feat_path.exists():
                    # scene_pc = scene_pc_with_labels[:, :3]
                    targ_pc = targ_pc_with_labels[:, :3]
                    # scene_no_targ_pc = scene_no_targ_pc_with_labels[:, :3]
                    scene_no_targ_pc = scene_pc_with_labels[scene_pc_with_labels[:, 3] == 0, :3]
                    target_category = self.scene_category_dict[scene_id]

                #     self.scene_ptv3_clip_feat_path.mkdir(parents=True, exist_ok=True)
                # scene_ptv3_clip_feat_path = self.scene_ptv3_clip_feat_path / f"{scene_id}.npz"
                # if not scene_ptv3_clip_feat_path.exists():
                #     np.savez(scene_ptv3_clip_feat_path, targ_pc_with_labels=targ_pc_with_labels, scene_pc_with_labels=scene_pc_with_labels)

                targ_pc = targ_pc /0.3- 0.5
                scene_no_targ_pc = scene_no_targ_pc /0.3- 0.5
                targ_pc = torch.from_numpy(targ_pc).float()
                scene_no_targ_pc = torch.from_numpy(scene_no_targ_pc).float()

                targ_labels = torch.ones((targ_pc.shape[0], 1), dtype=torch.float32)  # Target points labeled as 1
                occluder_labels = torch.zeros((scene_no_targ_pc.shape[0], 1), dtype=torch.float32)  # Scene points labeled as 0
                targ_pc_with_labels = torch.cat((targ_pc, targ_labels), dim=1)
                scene_no_targ_pc_with_labels = torch.cat((scene_no_targ_pc, occluder_labels), dim=1)
                scene_pc_with_labels = torch.cat((scene_no_targ_pc_with_labels, targ_pc_with_labels), dim=0)
                
                pos = pos / self.size - 0.5
                width = width / self.size

                # rotations = np.empty((2, 4), dtype=np.single)
                # R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
                # rotations[0] = ori.as_quat()
                # rotations[1] = (ori * R).as_quat()

                targ_pc_with_labels = targ_pc_with_labels.squeeze(0).numpy()
                scene_pc_with_labels = scene_pc_with_labels.squeeze(0).numpy()

                x = (voxel_grid[0], targ_grid[0], targ_pc_with_labels, scene_pc_with_labels)

                # save path
                save_path = self.scenes_ptv3_clip_path / f"{scene_id}.npz"
                np.savez(save_path, voxel_grid=voxel_grid, targ_grid=targ_grid, targ_pc_with_labels=targ_pc_with_labels, scene_pc_with_labels=scene_pc_with_labels)
            
            # if not self.curr_scene_clip_feat_path.exists():
            #     # Use the already processed point clouds
            #     # targ_pc and scene_no_targ_pc are already processed and have the correct shapes

            #     target_category = self.scene_category_dict[scene_id]

            #     # Add CLIP features to point clouds
            #     # Get CLIP model
            #     clip_model, clip_preprocess, clip_device = get_clip_model()
                
            #     # Extract CLIP features for target and occluders
            #     target_text_prompt = f"a {target_category} to grasp"
            #     occluder_text_prompt = "occluders"
                
            #     target_clip_features = extract_clip_features(target_text_prompt, clip_model, clip_device)
            #     occluder_clip_features = extract_clip_features(occluder_text_prompt, clip_model, clip_device)
                
            #     # Add CLIP features to each point in the point clouds
            #     # target_pc: (512, 3) -> (512, 515) [xyz + 512 CLIP features]
            #     # scene_no_targ_pc: (512, 3) -> (512, 515) [xyz + 512 CLIP features]
            #     target_clip_features_expanded = np.tile(target_clip_features, (targ_pc.shape[0], 1))  # (512, 512)
            #     occluder_clip_features_expanded = np.tile(occluder_clip_features, (scene_no_targ_pc.shape[0], 1))  # (512, 512)
            #     scene_clip_features_expanded = np.concatenate([target_clip_features_expanded, occluder_clip_features_expanded], axis=1)  # (512, 515)
                
            #     targ_pc_with_clip = np.concatenate([targ_pc, target_clip_features_expanded], axis=1)  # (512, 515)
            #     scene_no_targ_pc_with_clip = np.concatenate([scene_no_targ_pc, occluder_clip_features_expanded], axis=1)  # (512, 515)
                
            #     print(f"Added CLIP features - target_pc shape: {targ_pc_with_clip.shape}, scene_no_targ_pc shape: {scene_no_targ_pc_with_clip.shape}")
            #     print(f"Target category: {target_category}, text prompt: '{target_text_prompt}'")
            #     np.savez(targ_clip_features = target_clip_features_expanded, scene_clip_features_expanded = scene_clip_features_expanded)
            
            if self.curr_scene_clip_feat_path.exists():
                targ_clip_features = np.load(self.scene_ptv3_clip_feat_path / f"{scene_id}.npz")['targ_clip_features']
                occluder_clip_features = np.load(self.scene_ptv3_clip_feat_path / f"{scene_id}.npz")['scene_clip_features_expanded']
                # targ_pc_with_labels = scene_data['targ_pc_with_labels']
                # scene_pc_with_labels = scene_data['scene_pc_with_labels']
                occluder_pc = scene_pc_with_labels[scene_pc_with_labels[:, 3] == 0, :3]
                target_pc = scene_pc_with_labels[scene_pc_with_labels[:, 3] == 1, :3]
                assert np.allclose(target_pc, targ_pc_with_labels[:, :3])
                scene_pc = np.concatenate([target_pc, occluder_pc], axis=0)
                scene_clip_feat = np.concatenate([targ_clip_features, occluder_clip_features], axis=0)

                x = (voxel_grid[0], targ_grid[0], scene_pc, scene_clip_feat)
                np.savez(self.curr_scene_clip_final_path, voxel_grid=voxel_grid, targ_grid=targ_grid, scene_pc=scene_pc, scene_clip_feat=scene_clip_feat)
                
        elif self.curr_scene_clip_final_path.exists():
            scene_data = np.load(self.curr_scene_clip_final_path, allow_pickle=True)
            voxel_grid = scene_data['voxel_grid']
            targ_grid = scene_data['targ_grid']
            scene_pc = scene_data['scene_pc']
            scene_clip_feat = scene_data['scene_clip_feat']
            x = (voxel_grid[0], targ_grid[0], scene_pc, scene_clip_feat)

            # scene_clip_features_expanded = np.load(self.scene_ptv3_clip_feat_path / f"{scene_id}.npz")['scene_clip_features_expanded']
            # targ_pc_with_clip = np.concatenate([targ_pc, targ_clip_features], axis=1)
            # scene_no_targ_pc_with_clip = np.concatenate([scene_no_targ_pc, scene_clip_features_expanded], axis=1)
            # scene_pc_with_labels = np.concatenate([scene_no_targ_pc_with_clip, targ_pc_with_clip], axis=0)
            # x = (voxel_grid[0], targ_grid[0], targ_pc_with_labels, scene_pc_with_labels)
        

        # if self.debug:
        #     vis_path = str(self.vis_logdir / 'complete_targ_grid.png')
        #     visualize_and_save_tsdf(targ_grid[0], vis_path)

        # y = (label, rotations, width)

        return x, y, pos

    def get_mesh(self, idx):
        scene_id = self.df.loc[idx, "scene_id"]
        mesh_pose_list_path = self.raw_root / 'mesh_pose_list' / (scene_id + '.npz')
        mesh_pose_list = np.load(mesh_pose_list_path, allow_pickle=True)['pc']
        scene = get_scene_from_mesh_pose_list(mesh_pose_list, return_list=False)
        return scene

class DatasetVoxel_PTV3_Scene(torch.utils.data.Dataset):
    def __init__(self, root, raw_root, num_point=2048, augment=False, ablation_dataset="",  model_type="giga_aff",
            use_complete_targ = False, debug = False, logdir = None):
        assert model_type == "ptv3_scene"
        self.root = root
        self.augment = augment
        self.num_point = num_point
        self.raw_root = raw_root
        self.num_th = 32
        self.use_complete_targ = use_complete_targ
        self.model_type = model_type
        self.debug = debug
        
        self.df = read_df(raw_root)
        # self.df = read_df_filtered(raw_root)
        # self.df = self.df[:300]
        if ablation_dataset == 'only_cluttered':
            self.df = filter_rows_by_id_only_clutter(self.df)

        if ablation_dataset == '1_10':
            self.df = self.df.sample(frac=0.1)
            self.df = self.df.reset_index(drop=True)
        
        if ablation_dataset == '1_100':
            self.df = self.df.sample(frac=0.01)
            self.df = self.df.reset_index(drop=True)
        if ablation_dataset == '1_100000':
            self.df = self.df.sample(frac=0.00001)
            self.df = self.df.reset_index(drop=True)

        print("data frames stastics")
        print_and_count_patterns(self.df,False)

        ## filter dirty data
        skip_scene_ids = ['b960209b0cbd406d98dac25aeccd3c71_s_2','5522fdbe62d9450687a195cfd2bbbac3_c_1', 'b960209b0cbd406d98dac25aeccd3c71_c_2']
        self.df = self.df[~self.df['scene_id'].isin(skip_scene_ids)]
        # self.df = self.df[~self.df.scene_id.isin(skip_scene_ids)]
        self.df = self.df.reset_index(drop=True)

        self.size, _, _, _ = read_setup(raw_root)

        # Add visualization support
        if self.debug:
            vis_logdir = logdir / 'vis_data'
            if not vis_logdir.exists():
                vis_logdir.mkdir(exist_ok=True, parents=True)
            self.vis_logdir = vis_logdir

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, i):
        scene_id = self.df.loc[i, "scene_id"]
        
        # Add retry mechanism for incomplete scenes
        max_retries = 10
        for attempt in range(max_retries):
            try:
                return self._get_item_safe_ptv3(i, scene_id)
            except Exception as e:
                # Log the error
                ERROR_LOG_FILE_PTV3_SCENE.parent.mkdir(parents=True, exist_ok=True)
                with ERROR_LOG_FILE_PTV3_SCENE.open("a", encoding="utf-8") as f:
                    f.write(f"{scene_id},ptv3_dataset_loading,\"attempt_{attempt+1}: {str(e)}\"\n")
                
                print(f"[WARNING] PTV3 Scene {scene_id} loading failed (attempt {attempt+1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    # Try next scene
                    i = (i + 1) % len(self.df)
                    scene_id = self.df.loc[i, "scene_id"]
                else:
                    # Final attempt failed, raise the error
                    raise e
    
    def _get_item_safe_ptv3(self, i, scene_id):
        """Safe version of __getitem__ for PTV3_Scene with proper error handling."""
        ori = Rotation.from_quat(self.df.loc[i, "qx":"qw"].to_numpy(np.single))
        pos = self.df.loc[i, "x":"z"].to_numpy(np.single)
        width =  np.float32(self.df.loc[i, "width"])
        label = self.df.loc[i, "label"].astype(np.int64)

        single_scene_id = (scene_id.split('_')[0]) + '_s_' + scene_id.split('_')[2]
        voxel_grid, targ_grid = read_voxel_and_mask_occluder(self.raw_root, scene_id)
        occluder_grid = voxel_grid - targ_grid
        targ_pc = read_complete_target_pc(self.raw_root, scene_id).astype(np.float32)
        scene_path = self.raw_root / "scenes" / f"{scene_id}.npz"
        
        # Handle missing complete_target_tsdf with error logging and fallback
        scene_single_path = self.raw_root / "scenes" / f"{single_scene_id}.npz"
        try:
            # with np.load(scene_single_path, allow_pickle=True) as data_single:
                # targ_grid = data_single["complete_target_tsdf"]
            with np.load(scene_path) as data:
                targ_grid = data["complete_target_tsdf"]
                if targ_grid.ndim == 3:
                    targ_grid = np.expand_dims(targ_grid, axis=0)
        except KeyError as e:
            # Log the error to file
            ERROR_LOG_FILE_PTV3_SCENE.parent.mkdir(parents=True, exist_ok=True)
            with ERROR_LOG_FILE_PTV3_SCENE.open("a", encoding="utf-8") as f:
                f.write(f"{scene_id},complete_target_tsdf_missing,\"KeyError: 'complete_target_tsdf' not found in {single_scene_id}.npz\"\n")
            
            print(f"[WARNING] Scene {scene_id}: complete_target_tsdf missing in {single_scene_id}.npz, using fallback strategy")
            
            # Fallback: use original targ_grid from read_voxel_and_mask_occluder
            # This maintains training continuity but may have lower quality complete target data
            pass  # targ_grid already set from read_voxel_and_mask_occluder above
            
        except Exception as e:
            # Log other errors
            ERROR_LOG_FILE_PTV3_SCENE.parent.mkdir(parents=True, exist_ok=True)
            with ERROR_LOG_FILE_PTV3_SCENE.open("a", encoding="utf-8") as f:
                f.write(f"{scene_id},scene_file_error,\"Error loading {single_scene_id}.npz: {str(e)}\"\n")
            
            print(f"[WARNING] Scene {scene_id}: Error loading {single_scene_id}.npz: {e}, using fallback strategy")
            
            # Fallback: use original targ_grid
            pass  # targ_grid already set from read_voxel_and_mask_occluder above
        
        voxel_grid = occluder_grid + targ_grid

        plane = np.load('./setup/plane_sampled.npy')
        if '_c_' in scene_id:
            scene_no_targ_pc = read_scene_no_targ_pc(self.raw_root, scene_id).astype(np.float32)
            # scene_no_targ_pc = np.concatenate((scene_no_targ_pc, plane), axis=0)
            targ_pc = points_within_boundary(targ_pc)
            
            targ_pc = safe_specify_num_points(targ_pc, 512, scene_id, "ptv3_target")
            if targ_pc is None:
                raise ValueError(f"Failed to process PTV3 target point cloud for scene {scene_id}")
            
            scene_no_targ_pc = points_within_boundary(scene_no_targ_pc)
            scene_no_targ_pc = np.concatenate((scene_no_targ_pc, plane), axis=0)
            scene_no_targ_pc = safe_specify_num_points(scene_no_targ_pc, 512, scene_id, "ptv3_scene_no_target")
            if scene_no_targ_pc is None:
                raise ValueError(f"Failed to process PTV3 scene_no_target point cloud for scene {scene_id}")
            # scene_pc = np.concatenate((scene_no_targ_pc, targ_pc))
        elif '_s_' in scene_id:
            scene_no_targ_pc = plane
            targ_pc = points_within_boundary(targ_pc)
            
            targ_pc = safe_specify_num_points(targ_pc, 512, scene_id, "ptv3_target")
            if targ_pc is None:
                raise ValueError(f"Failed to process PTV3 target point cloud for scene {scene_id}")
            
            scene_no_targ_pc = safe_specify_num_points(scene_no_targ_pc, 512, scene_id, "ptv3_plane")
            if scene_no_targ_pc is None:
                raise ValueError(f"Failed to process PTV3 plane point cloud for scene {scene_id}")
            # scene_pc = targ_pc
            # scene_pc = specify_num_points(scene_pc, 512)

        targ_pc = targ_pc /0.3- 0.5
        scene_no_targ_pc = scene_no_targ_pc /0.3- 0.5
        targ_pc = torch.from_numpy(targ_pc).float()
        scene_no_targ_pc = torch.from_numpy(scene_no_targ_pc).float()

        targ_labels = torch.ones((targ_pc.shape[0], 1), dtype=torch.float32)  # Target points labeled as 1
        occluder_labels = torch.zeros((scene_no_targ_pc.shape[0], 1), dtype=torch.float32)  # Scene points labeled as 0
        targ_pc_with_labels = torch.cat((targ_pc, targ_labels), dim=1)
        scene_no_targ_pc_with_labels = torch.cat((scene_no_targ_pc, occluder_labels), dim=1)
        scene_pc_with_labels = torch.cat((scene_no_targ_pc_with_labels, targ_pc_with_labels), dim=0)
           
        pos = pos / self.size - 0.5
        width = width / self.size

        rotations = np.empty((2, 4), dtype=np.single)
        R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
        rotations[0] = ori.as_quat()
        rotations[1] = (ori * R).as_quat()

        targ_pc_with_labels = targ_pc_with_labels.squeeze(0).numpy()
        scene_pc_with_labels = scene_pc_with_labels.squeeze(0).numpy()

        x = (voxel_grid[0], targ_grid[0], targ_pc_with_labels, scene_pc_with_labels)

        # if self.debug:
        #     vis_path = str(self.vis_logdir / 'complete_targ_grid.png')
        #     visualize_and_save_tsdf(targ_grid[0], vis_path)

        y = (label, rotations, width)

        return x, y, pos

    def get_mesh(self, idx):
        scene_id = self.df.loc[idx, "scene_id"]
        mesh_pose_list_path = self.raw_root / 'mesh_pose_list' / (scene_id + '.npz')
        mesh_pose_list = np.load(mesh_pose_list_path, allow_pickle=True)['pc']
        scene = get_scene_from_mesh_pose_list(mesh_pose_list, return_list=False)
        return scene
    
def apply_transform(voxel_grid, orientation, position):
    angle = np.pi / 2.0 * np.random.choice(4)
    R_augment = Rotation.from_rotvec(np.r_[0.0, 0.0, angle])

    z_offset = np.random.uniform(6, 34) - position[2]

    t_augment = np.r_[0.0, 0.0, z_offset]
    T_augment = Transform(R_augment, t_augment)

    T_center = Transform(Rotation.identity(), np.r_[20.0, 20.0, 20.0])
    T = T_center * T_augment * T_center.inverse()

    # transform voxel grid
    T_inv = T.inverse()
    matrix, offset = T_inv.rotation.as_matrix(), T_inv.translation
    voxel_grid[0] = ndimage.affine_transform(voxel_grid[0], matrix, offset, order=0)

    # transform grasp pose
    position = T.transform_point(position)
    orientation = T.rotation * orientation

    return voxel_grid, orientation, position
    
class DatasetVoxel(torch.utils.data.Dataset):
    def __init__(self, root, raw_root, num_point=2048, augment=False):
        self.root = root
        self.augment = augment
        self.num_point = num_point
        self.raw_root = raw_root
        self.num_th = 32
        self.df = read_df(raw_root)
        self.size, _, _, _ = read_setup(raw_root)

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, i):
        scene_id = self.df.loc[i, "scene_id"]
        ori = Rotation.from_quat(self.df.loc[i, "qx":"qw"].to_numpy(np.single))
        pos = self.df.loc[i, "x":"z"].to_numpy(np.single)
        width = self.df.loc[i, "width"].astype(np.single)
        # label = self.df.loc[i, "label"].astype(np.long)
        label = self.df.loc[i, "label"].astype(np.int64)
        voxel_grid = read_voxel_grid(self.root, scene_id)
        
        if self.augment:
            voxel_grid, ori, pos = apply_transform(voxel_grid, ori, pos)
        
        pos = pos / self.size - 0.5
        width = width / self.size

        rotations = np.empty((2, 4), dtype=np.single)
        R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
        rotations[0] = ori.as_quat()
        rotations[1] = (ori * R).as_quat()

        x, y = voxel_grid[0], (label, rotations, width)

        return x, y, pos

    def get_mesh(self, idx):
        scene_id = self.df.loc[idx, "scene_id"]
        mesh_pose_list_path = self.raw_root / 'mesh_pose_list' / (scene_id + '.npz')
        mesh_pose_list = np.load(mesh_pose_list_path, allow_pickle=True)['pc']
        scene = get_scene_from_mesh_pose_list(mesh_pose_list, return_list=False)
        return scene


class DatasetVoxelOccFile(torch.utils.data.Dataset):
    def __init__(self, root, raw_root, num_point=2048, num_point_occ=2048, augment=False):
        self.root = root
        self.augment = augment
        self.num_point = num_point
        self.num_point_occ = num_point_occ
        self.raw_root = raw_root
        self.num_th = 32
        self.df = read_df(raw_root)
        self.size, _, _, _ = read_setup(raw_root)

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, i):
        scene_id = self.df.loc[i, "scene_id"]
        ori = Rotation.from_quat(self.df.loc[i, "qx":"qw"].to_numpy(np.single))
        pos = self.df.loc[i, "x":"z"].to_numpy(np.single)
        width = self.df.loc[i, "width"].astype(np.single)
        label = self.df.loc[i, "label"].astype(np.long)
        voxel_grid = read_voxel_grid(self.root, scene_id)
        
        if self.augment:
            voxel_grid, ori, pos = apply_transform(voxel_grid, ori, pos)
        
        pos = pos / self.size - 0.5
        width = width / self.size

        rotations = np.empty((2, 4), dtype=np.single)
        R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
        rotations[0] = ori.as_quat()
        rotations[1] = (ori * R).as_quat()

        x, y = voxel_grid[0], (label, rotations, width)

        occ_points, occ = self.read_occ(scene_id, self.num_point_occ)
        occ_points = occ_points / self.size - 0.5

        return x, y, pos, occ_points, occ

    def read_occ(self, scene_id, num_point):
        occ_paths = list((self.raw_root / 'occ' / scene_id).glob('*.npz'))
        path_idx = torch.randint(high=len(occ_paths), size=(1,), dtype=int).item()
        occ_path = occ_paths[path_idx]
        occ_data = np.load(occ_path)
        points = occ_data['points']
        occ = occ_data['occ']
        points, idxs = sample_point_cloud(points, num_point, return_idx=True)
        occ = occ[idxs]
        return points, occ

    def get_mesh(self, idx):
        scene_id = self.df.loc[idx, "scene_id"]
        mesh_pose_list_path = self.raw_root / 'mesh_pose_list' / (scene_id + '.npz')
        mesh_pose_list = np.load(mesh_pose_list_path, allow_pickle=True)['pc']
        scene = get_scene_from_mesh_pose_list(mesh_pose_list, return_list=False)
        return scene


def apply_transform(voxel_grid, orientation, position):
    angle = np.pi / 2.0 * np.random.choice(4)
    R_augment = Rotation.from_rotvec(np.r_[0.0, 0.0, angle])

    z_offset = np.random.uniform(6, 34) - position[2]

    t_augment = np.r_[0.0, 0.0, z_offset]
    T_augment = Transform(R_augment, t_augment)

    T_center = Transform(Rotation.identity(), np.r_[20.0, 20.0, 20.0])
    T = T_center * T_augment * T_center.inverse()

    # transform voxel grid
    T_inv = T.inverse()
    matrix, offset = T_inv.rotation.as_matrix(), T_inv.translation
    voxel_grid[0] = ndimage.affine_transform(voxel_grid[0], matrix, offset, order=0)

    # transform grasp pose
    position = T.transform_point(position)
    orientation = T.rotation * orientation

    return voxel_grid, orientation, position

def sample_point_cloud(pc, num_point, return_idx=False):
    num_point_all = pc.shape[0]
    idxs = np.random.choice(np.arange(num_point_all), size=(num_point,), replace=num_point > num_point_all)
    if return_idx:
        return pc[idxs], idxs
    else:
        return pc[idxs]
