import numpy as np
from scipy import ndimage
import torch.utils.data
from pathlib import Path

from vgn.io import *
from vgn.perception import *
from vgn.utils.transform import Rotation, Transform
from vgn.utils.implicit import get_scene_from_mesh_pose_list
import random
import os
# import utils_giga
from utils_giga import visualize_depth_map,tsdf_to_ply, visualize_and_save_tsdf, filter_rows_by_id_only_clutter, filter_rows_by_id_only_single_and_double, count_and_sample, print_and_count_patterns, filter_rows_by_id_only_clutter_and_double, filter_rows_by_id_only_single, load_scene_indices, specify_num_points
from utils_giga import save_point_cloud_as_ply, points_within_boundary
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

def transform_pc(pc):
    # device = pc.device
    # pc = pc.cpu().numpy()
    # BS = pc.shape[0]
    # pc_transformed = torch.zeros((BS, 2048, 3), dtype=torch.float32)
    # for i in range(BS):
    points_curr_transformed = transform({'input':pc})
        # pc_transformed[i] = points_curr_transformed['input']
    return points_curr_transformed['input']

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
        # if ablation_dataset == 'only_single_double':
        #     self.df = filter_rows_by_id_only_single_and_double(self.df)
        # if ablation_dataset == 'resized_set_theory':
        #     self.df = count_and_sample(self.df)

        
        if add_single_supervision == True:
            path_single_scene_indices = os.path.join(raw_root, "single_scene_indices.txt")
            self.single_scene_indices = load_scene_indices(path_single_scene_indices)
            # self.df_s = filter_rows_by_id_only_single(self.df)
            self.df_prev = self.df.copy()
            self.df = filter_rows_by_id_only_clutter_and_double(self.df)
        
        # if use_complete_targ == True:
        #     # self.df = filter_rows_by_id_only_single(self.df)
        #     path_single_scene_indices = os.path.join(raw_root, "single_scene_indices.txt")
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
        # if not os.path.exists(os.path.join(self.root, 'scenes', scene_id)):
        #     print("Error")
        ori = Rotation.from_quat(self.df.loc[i, "qx":"qw"].to_numpy(np.single))
        if not self.model_type == "vgn":
            pos = self.df.loc[i, "x":"z"].to_numpy(np.single)
            width =  np.float32(self.df.loc[i, "width"])
            label = self.df.loc[i, "label"].astype(np.long)
        else:
            pos = self.df.loc[i, "i":"k"].to_numpy(np.single)
            width = self.df.loc[i, "width"].astype(np.single)
            label = self.df.loc[i, "label"].astype(np.long)

        if self.use_complete_targ:
            single_scene_id = (scene_id.split('_')[0]) + '_s_' + scene_id.split('_')[2]

        if self.add_single_supervision:
            single_scene_id = (scene_id.split('_')[0]) + '_s_' + scene_id.split('_')[2]
            single_scene_index =  random.choice(self.single_scene_indices[single_scene_id])
            assert self.df_prev.loc[single_scene_index, "scene_id"] == single_scene_id
            ori_s = Rotation.from_quat(self.df_prev.loc[single_scene_index, "qx":"qw"].to_numpy(np.single))
            pos_s = self.df_prev.loc[single_scene_index, "x":"z"].to_numpy(np.single)
            width_s =  np.float32(self.df_prev.loc[single_scene_index, "width"])
            label_s = self.df_prev.loc[single_scene_index, "label"].astype(np.long)
        
        if self.data_contain == "pc and targ_grid":
            voxel_grid, targ_grid = read_voxel_and_mask_occluder(self.raw_root, scene_id)
            if self.vis_data:
                vis_path = str(self.vis_logdir / 'voxel_grid.png')
                visualize_and_save_tsdf(voxel_grid[0], vis_path)
                vis_path = str(self.vis_logdir / 'targ_grid.png')
                visualize_and_save_tsdf(targ_grid[0], vis_path)

            if not self.shape_completion:
                if self.input_points == "tsdf_points":
                    targ_pc = read_targ_pc(self.raw_root, scene_id).astype(np.float32)
                    scene_pc = read_scene_pc(self.raw_root, scene_id).astype(np.float32)
                    targ_pc = points_within_boundary(targ_pc)
                    scene_pc = points_within_boundary(scene_pc)
                
                elif self.input_points == "depth_target_others_tsdf":
                    if '_c_' in scene_id:
                        targ_pc = read_targ_depth_pc(self.raw_root, scene_id).astype(np.float32)
                        scene_no_targ_pc = read_scene_no_targ_pc(self.raw_root, scene_id).astype(np.float32)
                        targ_pc = points_within_boundary(targ_pc)
                        targ_pc = specify_num_points(targ_pc, 2048)
                        scene_no_targ_pc = points_within_boundary(scene_no_targ_pc)
                        scene_no_targ_pc = specify_num_points(scene_no_targ_pc, 2048)
                        scene_pc = np.concatenate((scene_no_targ_pc, targ_pc))
                    elif '_s_' in scene_id:
                        targ_pc = read_targ_depth_pc(self.raw_root, scene_id).astype(np.float32)
                        targ_pc = points_within_boundary(targ_pc)
                        targ_pc = specify_num_points(targ_pc, 2048)
                        scene_pc = targ_pc
                        scene_pc = specify_num_points(scene_pc, 4096)


            if self.shape_completion:
                targ_pc = read_targ_pc(self.raw_root, scene_id).astype(np.float32)
                targ_pc = points_within_boundary(targ_pc)
                targ_pc = specify_num_points(targ_pc, 2048)
                scene_pc = np.load('/usr/stud/dira/GraspInClutter/grasping/setup/plane_sampled.npy')
                num_scene = scene_pc.shape[0] + 2048
                if '_c_' in scene_id:
                    scene_no_targ_pc = read_scene_no_targ_pc(self.raw_root, scene_id).astype(np.float32)
                    scene_no_targ_pc = points_within_boundary(scene_no_targ_pc)
                    scene_pc = np.concatenate((scene_no_targ_pc, scene_pc), axis=0)
                scene_pc = specify_num_points(scene_pc, num_scene)

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


            if self.decouple:
                voxel_grid = voxel_grid - targ_grid
            if self.add_single_supervision:
                targ_grid = read_single_complete_target(self.raw_root, single_scene_id)
            elif self.use_complete_targ:
                targ_grid, targ_pc = read_single_complete_target(self.raw_root, single_scene_id)
                scene_pc = read_scene_no_targ_pc(self.raw_root, scene_id).astype(np.float32)
                scene_pc = np.concatenate((scene_pc, targ_pc), axis=0)

            if not (self.shape_completion == False and self.input_points == "depth_target_others_tsdf") \
                and not self.shape_completion:
                targ_pc = specify_num_points(targ_pc, 2048)
                if not ('_s_' in scene_id and self.shape_completion):
                    scene_pc = specify_num_points(scene_pc, 2048)

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
            index = np.round(pos).astype(np.long)
            rotations = np.empty((2, 4), dtype=np.single)
            R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
            rotations[0] = ori.as_quat()
            rotations[1] = (ori * R).as_quat()


        if self.add_single_supervision:
            pos_s = pos_s / self.size - 0.5
            width_s = width_s / self.size
            rotations_s = np.empty((2, 4), dtype=np.single)
            R_s =  Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
            rotations_s[0] = ori_s.as_quat()
            rotations_s[1] = (ori_s * R_s).as_quat()

        if self.data_contain == "pc and targ_grid":
            if not self.shape_completion:
                plane = np.load('/usr/stud/dira/GraspInClutter/grasping/setup/plane_sampled.npy')
                if not ('_s_' in scene_id and self.shape_completion):
                    scene_pc = np.concatenate((scene_pc, plane), axis=0)
                    targ_pc = targ_pc /0.3- 0.5
                    scene_pc = scene_pc /0.3- 0.5
                elif '_s_' in scene_id and self.shape_completion:
                    scene_pc = plane
                    scene_pc = specify_num_points(scene_pc, 2048 + plane.shape[0])
                    targ_pc = targ_pc /0.3- 0.5
                    scene_pc = scene_pc /0.3- 0.5
            elif self.shape_completion:
                scene_pc = scene_pc /0.3- 0.5
                targ_pc = targ_pc /0.3- 0.5

            x = (voxel_grid[0], targ_grid[0], targ_pc, scene_pc)

        if self.data_contain == "pc":
            if self.model_type == "vgn":
                x = (voxel_grid)
            else:
                x = (voxel_grid[0])


        if self.add_single_supervision:
            y = (label, rotations, width, label_s, rotations_s, width_s)
        else:
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