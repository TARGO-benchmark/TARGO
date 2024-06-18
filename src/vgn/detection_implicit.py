import time

import numpy as np
import trimesh
from scipy import ndimage
import torch

#from vgn import vis
from vgn.grasp import *
from vgn.utils.transform import Transform, Rotation
from vgn.networks import load_network
from vgn.utils import visual
from vgn.utils.implicit import as_mesh
from utils_giga import *
from utils_giga import tsdf_to_ply, point_cloud_to_tsdf

from shape_completion.config import cfg_from_yaml_file
from shape_completion import builder
from shape_completion.models.AdaPoinTr import AdaPoinTr

import MinkowskiEngine as ME
# LOW_TH = 0.5
LOW_TH = 0.0

class VGNImplicit(object):
    def __init__(self, model_path, model_type, best=False, force_detection=False, qual_th=0.9, out_th=0.5, visualize=False, resolution=40, shared_weights = False,
                                    add_single_supervision = False, fusion_type = 'MLP_Fusion', feat_type = 'plane',shape_completion=True,num_encoder_layers = 1,**kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_network(model_path, self.device, model_type=model_type, shared_weights=shared_weights, add_single_supervision=add_single_supervision, fusion_type=fusion_type, feat_type= feat_type, num_encoder_layers=num_encoder_layers) 
        # self.sc_net = 
        self.net = self.net.eval()
        if shape_completion == True:
            sc_cfg = cfg_from_yaml_file("/usr/stud/dira/GraspInClutter/grasping/src/shape_completion/configs/stso/AdaPoinTr.yaml")
            sc_net = AdaPoinTr(sc_cfg.model)
            builder.load_model(sc_net, "/usr/stud/dira/GraspInClutter/grasping/checkpoints_noisy/sc_net/ckpt-best_0425.pth")
            sc_net = sc_net.eval()
            self.sc_net = sc_net
        self.qual_th = qual_th
        self.best = best
        self.force_detection = force_detection
        self.out_th = out_th
        self.visualize = visualize
        self.fusion_type = fusion_type
        self.shape_completion = shape_completion
        self.resolution = resolution
        if model_type == 'giga_hr':
            self.resolution = 60
        x, y, z = torch.meshgrid(torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution), torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution), torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution))
        pos = torch.stack((x, y, z), dim=-1).float().unsqueeze(0).to(self.device)   
        self.pos = pos.view(1, self.resolution * self.resolution * self.resolution, 3)  ## pos: 1, 64000, -0.5, 0.475

    def __call__(self, state, scene_mesh=None, visual_dict = None, aff_kwargs={}):
        ## all the keys in the namespace of state
        print(state.__dict__.keys())
        if state.type == 'afford_scene_pc':
            if self.shape_completion:
                inputs = (state.scene_no_targ_pc, state.targ_pc)    # scene_no_targ_pc is tsdf surface points, target pc is the depth backprojected points
            voxel_size, size = state.tsdf.voxel_size, state.tsdf.size

        elif state.type in ('giga_aff', 'giga', 'giga_hr'):
            if hasattr(state, 'tsdf_process'):
                tsdf_process = state.tsdf_process
            else:
                tsdf_process = state.tgt_mask_vol    
            
            inputs = state.scene_grid
            voxel_size, size = state.tsdf.voxel_size, state.tsdf.size        

        elif state.type == 'afford_scene_targ_pc':
            if self.fusion_type in ('transformer_query_scene','transformer_query_target', 'transformer_concat'):
                if self.shape_completion:
                    inputs = (state.scene_no_targ_pc, state.targ_pc)    # scene_no_targ_pc is tsdf surface points, target pc is the depth backprojected points
                elif not self.shape_completion:
                    inputs = (state.scene_pc, state.targ_pc) 
            elif self.fusion_type in ('CNN_concat', 'CNN_add', 'MLP_fusion'):
                inputs = (state.tsdf.get_grid(), state.tgt_grid)
            ## save inputs to npz, scene_no_targ_pc, targ_pc
            # np.savez('/usr/stud/dira/GraspInClutter/grasping/scripts/real_world/data/demo_ours.npz', scene_no_targ_pc= inputs[0], targ_pc=inputs[1])
            # np.savez('/usr/stud/dira/GraspInClutter/grasping/scripts/real_world/data/demo_ours.npz', inputs[0], inputs[1])
            voxel_size, size = state.tsdf.voxel_size, state.tsdf.size

        tic = time.time()
        if self.shape_completion:
            qual_vol, rot_vol, width_vol, completed_targ_grid = predict(inputs, self.pos, self.net, self.sc_net, state.type, state.fusion_type, self.device, visual_dict if self.visualize else None,)
        elif not self.shape_completion:
            qual_vol, rot_vol, width_vol = predict(inputs, self.pos, self.net,  None, state.type, state.fusion_type, self.device)

        qual_vol = qual_vol.reshape((self.resolution, self.resolution, self.resolution))
        rot_vol = rot_vol.reshape((self.resolution, self.resolution, self.resolution, 4))
        width_vol = width_vol.reshape((self.resolution, self.resolution, self.resolution))

        if self.shape_completion:
            qual_vol, rot_vol, width_vol = process(completed_targ_grid, qual_vol, rot_vol, width_vol, out_th=self.out_th)
        if not self.shape_completion:
            ## TODO affordance filter
            tsdf_to_ply(state.tgt_grid[0], 'tgt_tsdf.ply')
            qual_vol, rot_vol, width_vol = process(state.tgt_grid[0], qual_vol, rot_vol, width_vol, out_th=self.out_th)
            
        qual_vol = bound(qual_vol, voxel_size)
        if self.visualize:
            colored_scene_mesh = visual.affordance_visual(qual_vol, rot_vol, scene_mesh, size, self.resolution, **aff_kwargs)

        grasps, scores = select(qual_vol.copy(), self.pos.view(self.resolution, self.resolution, self.resolution, 3).cpu(), rot_vol, width_vol, threshold=self.qual_th, force_detection=self.force_detection, max_filter_size=8 if self.visualize else 4)
        toc = time.time() - tic

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

        if self.visualize:
            grasp_mesh_list = [visual.grasp2mesh(g, s) for g, s in zip(grasps, scores)]
            composed_scene = trimesh.Scene(colored_scene_mesh)
            for i, g_mesh in enumerate(grasp_mesh_list):
                composed_scene.add_geometry(g_mesh, node_name=f'grasp_{i}')
            return grasps, scores, toc, composed_scene
        else:
            return grasps, scores, toc

def bound(qual_vol, voxel_size, limit=[0.02, 0.02, 0.055]):
    # avoid grasp out of bound [0.02  0.02  0.055]
    x_lim = int(limit[0] / voxel_size)
    y_lim = int(limit[1] / voxel_size)
    z_lim = int(limit[2] / voxel_size)
    qual_vol[:x_lim] = 0.0
    qual_vol[-x_lim:] = 0.0
    qual_vol[:, :y_lim] = 0.0
    qual_vol[:, -y_lim:] = 0.0
    qual_vol[:, :, :z_lim] = 0.0
    return qual_vol

def predict(inputs, pos, net, sc_net, type, fusion_type, device, visual_dict=None,):
    if type in ('giga', 'giga_aff', 'vgn', 'giga_hr'):
        assert sc_net == None
        inputs = torch.from_numpy(inputs).to(device)

    elif type == 'afford_scene_targ_pc' and fusion_type in ('CNN_concat, CNN_add, MLP_fusion'):
        inputs = (torch.from_numpy(inputs[0]).to(device), torch.from_numpy(inputs[1]).to(device))

    elif type == 'afford_scene_targ_pc' and sc_net is None and fusion_type == 'transformer_concat':
        scene_pc = torch.from_numpy(inputs[0]).unsqueeze(0).to(device)
        targ_pc = torch.from_numpy(inputs[1]).unsqueeze(0).to(device)
        inputs = (scene_pc, targ_pc)

    
    elif type in ('afford_scene_pc', 'afford_scene_targ_pc') and sc_net is not None and fusion_type == 'transformer_concat':
        scene_no_targ_pc = torch.from_numpy(inputs[0]).unsqueeze(0).to(device)
        targ_pc = torch.from_numpy(inputs[1]).unsqueeze(0).to(device)

        with torch.no_grad():
            
            # sc_net = sc_net.to(device)
            # data = np.load('/usr/stud/dira/Desktop/dex-ycb-toolkit/demo_sc2/20201015-subject-09_20201015_143455_836212060125_s_11.npz')
            # pc, gt = data['pc_targ_cam.npy'][:,:3], data['pc_targ_cam_com.npy']
            # indices = np.random.choice(pc.shape[0], 2048, replace=True)
            # pc = pc[indices].astype(np.float32)
            # pc = pc / 0.3
            # pc = torch.from_numpy(pc).unsqueeze(0).to(device)
            # gt = gt / 0.3
            # completed_pc = sc_net(pc)[1]
            # save_point_cloud_as_ply(pc.cpu().numpy()[0], 'demo_sc2/pc.ply')
            # save_point_cloud_as_ply(completed_pc.cpu().numpy()[0], 'demo_sc2/completed_pc.ply')
            # save_point_cloud_as_ply(gt, 'demo_sc2/gt.ply')
            ## sample 2048 points from the target point cloud
            sc_net = sc_net.to(device)
            completed_targ_pc = sc_net(targ_pc)[1]
            # save_point_cloud_as_ply(targ_pc.cpu().numpy()[0], 'targ_pc.ply')
            # save_point_cloud_as_ply(completed_targ_pc.cpu().numpy()[0], 'completed_targ_pc.ply')
            completed_targ_pc_real_size = (completed_targ_pc+0.5)*0.3
            completed_targ_grid = point_cloud_to_tsdf(completed_targ_pc_real_size.squeeze().cpu().numpy())
            completed_targ_pc = filter_and_pad_point_clouds(completed_targ_pc)
            targ_completed_scene_pc = torch.cat([scene_no_targ_pc, completed_targ_pc], dim=1)
            targ_scene_pc = torch.cat([scene_no_targ_pc, targ_pc], dim=1)
                
        if visual_dict is not None:
            mesh_dir = visual_dict['mesh_dir']
            mesh_name = visual_dict['mesh_name']
            path = f'{mesh_dir}/{mesh_name}_completed_targ_pc.ply'
            save_point_cloud_as_ply(targ_pc[0].cpu().numpy(), path)
        
        if type == 'afford_scene_targ_pc':
            ## TODO TODO
            # inputs = (targ_completed_scene_pc, completed_targ_pc)
            # inputs = (targ_scene_pc, completed_targ_pc)
            inputs = (targ_completed_scene_pc, targ_pc)
        elif type == 'afford_scene_pc':
            inputs = targ_completed_scene_pc
        

    with torch.no_grad():
        # tsdf_to_ply(inputs[0].cpu().numpy(), 'tsdf_input.ply')
        qual_vol, rot_vol, width_vol = net(inputs, pos)

    # move output back to the CPU
    qual_vol = qual_vol.cpu().squeeze().numpy()
    rot_vol = rot_vol.cpu().squeeze().numpy()
    width_vol = width_vol.cpu().squeeze().numpy()
    if sc_net != None:
        return qual_vol, rot_vol, width_vol, completed_targ_grid
    else:
        return qual_vol, rot_vol, width_vol

def process_vgn(
    tsdf_vol,
    qual_vol,
    rot_vol,
    width_vol,
    gaussian_filter_sigma=1.0,
    min_width=1.33,
    max_width=9.33,
    out_th=0.5
):
    tsdf_vol = tsdf_vol.squeeze()

    # smooth quality volume with a Gaussian
    qual_vol = ndimage.gaussian_filter(
        qual_vol, sigma=gaussian_filter_sigma, mode="nearest"
    )

    # mask out voxels too far away from the surface
    outside_voxels = tsdf_vol > out_th
    inside_voxels = np.logical_and(1e-3 < tsdf_vol, tsdf_vol < out_th)
    valid_voxels = ndimage.morphology.binary_dilation(
        outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
    )
    qual_vol[valid_voxels == False] = 0.0

    # reject voxels with predicted widths that are too small or too large
    qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0

    # if VIS:
    #     voxel_dict = {'tsdf': tsdf_vol, 'quality': qual_vol.squeeze()}
    #     fig = visual.plot_3d_voxel_cloud_dict(voxel_dict)
    #     plt.show(block=True)
    #     plt.close(fig)

    return qual_vol, rot_vol, width_vol

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
    ## check if tsdf_vol is a tuple
    if isinstance(tsdf_vol, tuple):
        if len(tsdf_vol) == 2:
            tsdf_vol = tsdf_vol[0]
    tsdf_vol = tsdf_vol.squeeze()
    
    qual_vol = ndimage.gaussian_filter(
        qual_vol, sigma=gaussian_filter_sigma, mode="nearest"
    )

    # mask out voxels too far away from the surface
    outside_voxels = tsdf_vol > out_th
    inside_voxels = np.logical_and(1e-3 < tsdf_vol, tsdf_vol < out_th)
    valid_voxels = ndimage.morphology.binary_dilation(
        outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
    )
    qual_vol[valid_voxels == False] = 0.0

    # reject voxels with predicted widths that are too small or too large
    qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0

    return qual_vol, rot_vol, width_vol


def process_dexycb(
    tsdf_vol,
    qual_vol,
    rot_vol,
    width_vol,
    gaussian_filter_sigma=1.0,
    min_width=0.033,
    max_width=0.233,
    out_th=0.5
):
    ## check if tsdf_vol is a tuple
    if isinstance(tsdf_vol, tuple):
        if len(tsdf_vol) == 2:
            tsdf_vol = tsdf_vol[0]
    tsdf_vol = tsdf_vol.squeeze()
    
    qual_vol = ndimage.gaussian_filter(
        qual_vol, sigma=gaussian_filter_sigma, mode="nearest"
    )

    # mask out voxels too far away from the surface
    outside_voxels = tsdf_vol > out_th
    # inside_voxels = np.logical_and(1e-3 < tsdf_vol, tsdf_vol < out_th)
    # valid_voxels = ndimage.morphology.binary_dilation(
    #     outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
    # )
    qual_vol[outside_voxels == False] = 0.0

    # reject voxels with predicted widths that are too small or too large
    qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0

    return qual_vol, rot_vol, width_vol


def select_vgn(qual_vol, rot_vol, width_vol, threshold=0.90, max_filter_size=4, force_detection=False):
    best_only = False
    qual_vol[qual_vol < LOW_TH] = 0.0
    if force_detection and (qual_vol >= threshold).sum() == 0:
        best_only = True
    else:
        # threshold on grasp quality
        qual_vol[qual_vol < threshold] = 0.0

    # non maximum suppression
    max_vol = ndimage.maximum_filter(qual_vol, size=max_filter_size)
    qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0)
    mask = np.where(qual_vol, 1.0, 0.0)

    # construct grasps
    grasps, scores = [], []
    for index in np.argwhere(mask):
        grasp, score = select_index_vgn(qual_vol, rot_vol, width_vol, index)
        grasps.append(grasp)
        scores.append(score)

    sorted_grasps = [grasps[i] for i in reversed(np.argsort(scores))]
    sorted_scores = [scores[i] for i in reversed(np.argsort(scores))]

    if best_only and len(sorted_grasps) > 0:
        sorted_grasps = [sorted_grasps[0]]
        sorted_scores = [sorted_scores[0]]

    return sorted_grasps, sorted_scores


def select(qual_vol, center_vol, rot_vol, width_vol, threshold=0.90, max_filter_size=4, force_detection=False):
    best_only = False
    qual_vol[qual_vol < LOW_TH] = 0.0
    if force_detection and (qual_vol >= threshold).sum() == 0:
        best_only = True
    else:
        # threshold on grasp quality
        qual_vol[qual_vol < threshold] = 0.0

    # non maximum suppression
    max_vol = ndimage.maximum_filter(qual_vol, size=max_filter_size)
    qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0)
    mask = np.where(qual_vol, 1.0, 0.0)

    # construct grasps
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

def select_target(qual_vol, center_vol, rot_vol, width_vol, threshold=0.90, max_filter_size=4, force_detection=False):
    # best_only = False
    # qual_vol[qual_vol < LOW_TH] = 0.0
    # if force_detection and (qual_vol >= threshold).sum() == 0:
    #     best_only = True
    # else:
    #     # threshold on grasp quality
    #     qual_vol[qual_vol < threshold] = 0.0

    # best_only = False
    # qual_vol[qual_vol < LOW_TH] = 0.0
    # if force_detection and (qual_vol >= threshold).sum() == 0:
    #     best_only = True
    # else:
    #     # threshold on grasp quality
    #     qual_vol[qual_vol < threshold] = 0.0

    # non maximum suppression
    max_vol = ndimage.maximum_filter(qual_vol, size=max_filter_size)
    qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0)
    mask = np.where(qual_vol, 1.0, 0.0)

    # construct grasps
    grasps, scores = [], []
    for index in np.argwhere(mask):
        grasp, score = select_index(qual_vol, center_vol, rot_vol, width_vol, index)
        grasps.append(grasp)
        scores.append(score)

    sorted_grasps = [grasps[i] for i in reversed(np.argsort(scores))]
    sorted_scores = [scores[i] for i in reversed(np.argsort(scores))]

    # if best_only and len(sorted_grasps) > 0:
    sorted_grasps = [sorted_grasps[0]]
    sorted_scores = [sorted_scores[0]]
    
    return sorted_grasps, sorted_scores



def select_index(qual_vol, center_vol, rot_vol, width_vol, index):
    i, j, k = index
    score = qual_vol[i, j, k]
    ori = Rotation.from_quat(rot_vol[i, j, k])
    #pos = np.array([i, j, k], dtype=np.float64)
    pos = center_vol[i, j, k].numpy()
    width = width_vol[i, j, k]
    return Grasp(Transform(ori, pos), width), score

def select_index_vgn(qual_vol, rot_vol, width_vol, index):
    i, j, k = index
    score = qual_vol[i, j, k]
    ori = Rotation.from_quat(rot_vol[:, i, j, k])
    pos = np.array([i, j, k], dtype=np.float64)
    width = width_vol[i, j, k]
    return Grasp(Transform(ori, pos), width), score