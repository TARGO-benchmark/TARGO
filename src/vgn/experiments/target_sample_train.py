import collections
import argparse
from datetime import datetime
import uuid
import json
import numpy as np
import pandas as pd
import tqdm
import math
import time
import pyrender
from vgn import io#, vis
import matplotlib.pyplot as plt
from vgn.grasp import *
from vgn.simulation import ClutterRemovalSim
from vgn.utils.transform import Rotation, Transform
from vgn.utils.implicit import get_mesh_pose_list_from_world, get_scene_from_mesh_pose_list
from utils_giga import record_occ_level_count, record_occ_level_success, cal_occ_level_sr, save_point_cloud_as_ply
import trimesh
from vgn.perception import camera_on_sphere
# from MinkowskiEngine as ME
# import MinkowskiEngine as ME
MAX_CONSECUTIVE_FAILURES = 2
import os
State = collections.namedtuple("State", ["tsdf", "pc"])


def init_occ_level_count_dict():
    occ_level_count_dict = {
        '0-0.1': 0,
        '0.1-0.2': 0,
        '0.2-0.3': 0,
        '0.3-0.4': 0,
        '0.4-0.5': 0,
        '0.5-0.6': 0,
        '0.6-0.7': 0,
        '0.7-0.8': 0,
        '0.8-0.9': 0,
    }
    return occ_level_count_dict

def init_occ_level_success_dict():
    occ_level_success_dict = {
        '0-0.1': 0,
        '0.1-0.2': 0,
        '0.2-0.3': 0,
        '0.3-0.4': 0,
        '0.4-0.5': 0,
        '0.5-0.6': 0,
        '0.6-0.7': 0,
        '0.7-0.8': 0,
        '0.8-0.9': 0,
    }
    return occ_level_success_dict


def run(
    grasp_plan_fn,
    logdir,
    description,
    scene,
    object_set,
    num_objects=5,
    n=6,
    N=None,
    num_rounds=40,
    # seed=1,
    sim_gui=False,
    result_path=None,
    add_noise=False,
    sideview=False,
    resolution=40,
    silence=False,
    visualize=False,
    # tgt_sample=False,
    task_eval = 'occ_level',
    complete_shape=False,
    type = 'giga_aff',
    fusion_type = 'MLP_fusion',
    # test_mesh_pose_list = None,
    test_root = None,
    input_points="tsdf_points",
    shape_completion = False,
):
    """Run several rounds of simulated clutter removal experiments.

    Each round, m objects are randomly placed in a tray. Then, the grasping pipeline is
    run until (a) no objects remain, (b) the planner failed to find a grasp hypothesis,
    or (c) maximum number of consecutive failed grasp attempts.
    """
    #sideview=False
    #n = 6
    sim = ClutterRemovalSim(scene, object_set, gui=sim_gui, add_noise=add_noise, sideview=sideview, test_root=test_root)
    logger = Logger(logdir, description,tgt_sample=True)
    planning_times, total_times = [], []
    # occ_level_count_dict = init_occ_level_count_dict()
    # occ_level_success_dict = init_occ_level_success_dict()


    test_mesh_pose_list = f'{test_root}/mesh_pose_dict/'

    test_scenes = f'{test_root}/scenes/'
    occ_level_count_dict = {}
    offline_occ_level_dict = {}
    occ_level_success_dict = {}
    occ_level_count_dict = {
        '0-0.1': 0,
        '0.1-0.2': 0,
        '0.2-0.3': 0,
        '0.3-0.4': 0,
        '0.4-0.5': 0,
        '0.5-0.6': 0,
        '0.6-0.7': 0,
        '0.7-0.8': 0,
        '0.8-0.9': 0,
    }
    occ_level_success_dict = {
        '0-0.1': 0,
        '0.1-0.2': 0,
        '0.2-0.3': 0,
        '0.3-0.4': 0,
        '0.4-0.5': 0,
        '0.5-0.6': 0,
        '0.6-0.7': 0,
        '0.7-0.8': 0,
        '0.8-0.9': 0,
    }
    skip_dict = {}
    for num_id, curr_mesh_pose_list in enumerate(os.listdir(test_mesh_pose_list)):
        scene_name_single = curr_mesh_pose_list[:-4]
        if '_c' in scene_name_single:
            continue
        scene_name = scene_name_single.replace('_s_', '_c_')

    # for num_id, curr_mesh_pose_list in enumerate(os.listdir(test_scenes)):
        # if num_id < 3259:
        #     continue
        # scene_name = curr_mesh_pose_list[:-4]
        # if scene_name
        # if scene_name not in sim.occ_level_dict:
            # ## delete the mesh_pose_list
            # if os.path.exists(test_mesh_pose_list + curr_mesh_pose_list):
            #     os.remove(test_mesh_pose_list + curr_mesh_pose_list)
            # if os.path.exists(test_root + 'scenes/' + curr_mesh_pose_list):
            #     os.remove(test_root + 'scenes/' + curr_mesh_pose_list)
            # continue

        # if scene_name == '98b6d1ebb41341608a74bc10eedb5c93_c_1':
        #     continue
        # if scene_name == '72bb0e42b98a4e78a03b3ef822a38932_c_1':
        #     continue
        # if scene_name == '9fbc07bfeafe4ef6b0ca1fc8b2bf4751_c_1':
        #     continue
            
        # if scene_name == '3cc0152feff0451d8cdba14525bfb0b9_c_1':
        #     continue

        curr_scene_path = f'{test_root}scenes/{curr_mesh_pose_list}'.replace('_s_', '_c_')
        if not os.path.exists(curr_scene_path):
            continue
        curr_clutter_mesh_pose = curr_mesh_pose_list.replace('_s_', '_c_')[:-6] + '.npz'
        if visualize:
            visual_dict = {'mesh_name':scene_name, 'mesh_dir':logger.mesh_dir}
        mesh_pos = np.load(test_mesh_pose_list + curr_clutter_mesh_pose, allow_pickle=True)['pc']
        sim.world.reset()
        sim.world.set_gravity([0.0, 0.0, -9.81])
        sim.draw_workspace()
        sim.save_state()
        sim.lower = np.array([0.02, 0.02, 0.055])
        sim.upper = np.array([0.28, 0.28, 0.30000000000000004])
        tgt_id = int(scene_name[-1])
        start_time = time.time() 

        ## load the mesh and construct the sim
        for id, mesh in enumerate(mesh_pos.item().values()):
            pose = Transform.from_matrix(mesh[2])
            if mesh[0].split('/')[-1] == 'plane.obj':
                urdf_path = mesh[0].replace(".obj", ".urdf")
            else:
                urdf_path = mesh[0].replace("_visual.obj", ".urdf")
            body = sim.world.load_urdf(urdf_path = urdf_path,pose=pose,scale=mesh[1])
            if id == tgt_id:
                body.set_color(link_index=-1, rgba_color=(1.0, 0.0, 0.0, 1.0))  # Set the base color to red
                tgt_id = body.uid
            
        end_time = time.time() 
        print(f"load {num_id}-th {scene_name} took {end_time - start_time:.2f} s")

        # print(f"comstruct {scene_name} took {time.duration:.2f} s")
        timings = {}
        start_time = time.time()
        if type in ('vgn', 'giga_aff', 'giga', 'giga_hr'):
            res = 40 if type != 'giga_hr' else 60
            tsdf, timings["integration"], scene_grid, tgt_grid, tgt_mask_vol, occ_level = sim.acquire_single_tsdf_target_grid_train(curr_scene_path, tgt_id, res, type, complete_shape = complete_shape, curr_mesh_pose_list = scene_name, input_points=input_points, shape_completion= shape_completion)
            state = argparse.Namespace(tsdf=tsdf, tgt_grid = tgt_grid,tgt_mask_vol=tgt_mask_vol, scene_grid= scene_grid,occ_level=occ_level,type = type, fusion_type = fusion_type)
            
        elif type == 'afford_scene_targ_pc':
            if fusion_type in ('CNN_concat','CNN_add', 'MLP_fusion'):
                tsdf, timings["integration"], scene_grid, tgt_grid, tgt_mask_vol, occ_level = sim.acquire_single_tsdf_target_grid_train(curr_scene_path, tgt_id, 40, type, fusion_type = fusion_type, complete_shape = complete_shape, curr_mesh_pose_list = scene_name, input_points=input_points, shape_completion= shape_completion)
                state = argparse.Namespace(tsdf=tsdf, tgt_grid = tgt_grid,tgt_mask_vol=tgt_mask_vol, scene_grid= scene_grid,occ_level=occ_level,type = type, fusion_type = fusion_type)
            elif fusion_type in ('transformer_query_scene','transformer_query_target', 'transformer_concat'):
                if shape_completion:
                    tsdf, timings["integration"], scene_no_targ_pc, targ_pc, occ_level = sim.acquire_single_tsdf_target_grid_train(curr_scene_path, tgt_id, 40, type, fusion_type = fusion_type, complete_shape=complete_shape, curr_mesh_pose_list = scene_name, input_points=input_points, shape_completion= shape_completion)
                    state = argparse.Namespace(tsdf=tsdf,scene_no_targ_pc = scene_no_targ_pc, targ_pc = targ_pc, occ_level=occ_level,type = type, fusion_type = fusion_type)
                else:
                    tsdf, timings["integration"], scene_pc, targ_pc, tgt_grid, occ_level = sim.acquire_single_tsdf_target_grid_train(curr_scene_path, tgt_id, 40, type, fusion_type = fusion_type, complete_shape=complete_shape, curr_mesh_pose_list = scene_name, input_points=input_points, shape_completion= shape_completion)
                    # tsdf, timings["integration"], scene_no_targ_pc, targ_pc, occ_level  = sim.acquire_single_tsdf_target_grid(tgt_id, 40, type, fusion_type = fusion_type, complete_shape=complete_shape, curr_mesh_pose_list = curr_mesh_pose_list, input_points=input_points, shape_completion= shape_completion)
                    state = argparse.Namespace(tsdf=tsdf,targ_pc = targ_pc, tgt_grid = tgt_grid, occ_level=occ_level,scene_pc = scene_pc,type = type, fusion_type = fusion_type)

        elif type == 'afford_scene_pc':
            tsdf, timings["integration"],  scene_no_targ_pc, targ_pc, occ_level= sim.acquire_single_tsdf_target_grid_train(curr_scene_path, tgt_id, 40, type, fusion_type = fusion_type, complete_shape=complete_shape, curr_mesh_pose_list = scene_name, input_points=input_points, shape_completion= shape_completion)
            if shape_completion:
                state = argparse.Namespace(tsdf=tsdf,scene_no_targ_pc = scene_no_targ_pc, targ_pc = targ_pc, occ_level=occ_level,type = type, fusion_type = fusion_type)

        end_time = time.time()
        print(f"acquire {num_id}-th {scene_name} took {end_time - start_time:.2f} to construct inputs")

        if visualize and type == 'afford_scene_targ_pc' and fusion_type in ('transformer_query_scene','transformer_query_target', 'transformer_concat'):
            # save_point_cloud_as_ply(scene_no_targ_pc, f"{logger.mesh_dir}/{scene_name}_scene_no_targ_pc.ply")
            save_point_cloud_as_ply(targ_pc, f"{logger.mesh_dir}/{scene_name}_targ_pc.ply")
            
        occ_level_count_dict = record_occ_level_count(occ_level, occ_level_count_dict)
        if resolution != 40:
            extra_tsdf, _, _ = sim.acquire_tsdf(n=n, N=N, resolution=resolution)
            state.tsdf_process = extra_tsdf
        offline_occ_level_dict[scene_name] = occ_level

        ## TODO: occ_level = 0.9
        if occ_level  >= 0.9:
            print("skip")
            skip_dict[scene_name] = occ_level
            continue
        #----------------------##
        # plan grasps
        #----------------------##
        if visualize:
            mesh_pose_list = get_mesh_pose_list_from_world(sim.world, object_set)
            # ## tgt_id - 1, because no plane
            scene_mesh = get_scene_from_mesh_pose_list(mesh_pose_list,tgt_id-1)
            grasps, scores, timings["planning"], visual_mesh = grasp_plan_fn(state, scene_mesh, visual_dict)

            origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, sim.size / 3])
            r = 2 * sim.size
            theta = np.pi / 3.0
            phi = - np.pi / 2.0
            extrinsic = camera_on_sphere(origin, r, theta, phi)
            rgb, _, _ = sim.camera.render_with_seg(extrinsic )
            output_path = f'{logger.mesh_dir}/{occ_level}_occ_{scene_name}_rgb.png'
            plt.imsave(output_path, rgb)
            ## save rgb to output_path
            # save_point_cloud_as_ply(state.targ_pc, f"{logger.mesh_dir}/{scene_name}_targ_pc.ply")

            # render_mesh_with_camera(mesh_pose_list , extrinsic.as_matrix(), resolution=(640, 480), output_path=f'{logger.mesh_dir}/{scene_name}_occ_{occ_level}.png')
            # render_mesh_with_camera(scene_mesh, extrinsic.as_matrix(), resolution=(640, 480), output_path=f'{logger.mesh_dir}/{scene_name}_scene.png')
            # logger.log_mesh(scene_mesh, visual_mesh, f'{scene_name}_occ_{occ_level}')

        else:
            grasps, scores, timings["planning"] = grasp_plan_fn(state)
        planning_times.append(timings["planning"])
        total_times.append(timings["planning"] + timings["integration"])

        if len(grasps) == 0:
            continue

        # execute grasp
        grasp, score = grasps[0], scores[0]
        # grasp.width *= 1.5
        grasp.width = sim.gripper.max_opening_width
        label, _ = sim.execute_grasp(grasp, allow_contact=True, tgt_id = tgt_id, force_targ = True)
        if label != Label.FAILURE:
            occ_level_success_dict = record_occ_level_success(occ_level, occ_level_success_dict)
            if visualize:   ## visualize the scene_mesh and visual_mesh
                logger.log_mesh(scene_mesh, visual_mesh, f'{occ_level}_occ_{scene_name}')
                                
        if all(value > 0 for value in occ_level_count_dict.values()) and num_id % 100 == 0:
            occ_level_sr = cal_occ_level_sr(occ_level_count_dict, occ_level_success_dict)

            curr_count = 0
            for _, value in occ_level_count_dict.items():
                curr_count += value
            
            ## write result_path 
            intermediate_result_path = f'{result_path}/intermediate_result.txt'
            with open(intermediate_result_path, 'a') as f:
                f.write(f"current total count:{curr_count}\n")
                for key, value in occ_level_sr.items():
                    f.write(f"{key}:{value}\n")
                f.write('\n')
    
    if sim.save_occ_level_dict:
        with open(sim.occ_level_dict_path, 'w') as f:
            json.dump(sim.occ_level_dict, f)
    with open(f'{result_path}/occ_level_sr.json', 'w') as f:
        json.dump(occ_level_sr, f)
    with open(f'{result_path}/occ_level_count_dict.json', 'w') as f:
        json.dump(occ_level_count_dict, f)
    print("done")

    return occ_level_sr

    
    # return success_rate, declutter_rate
    


class Logger(object):
    def __init__(self, root, description, tgt_sample=False):
        # time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        self.logdir = root/ "visualize" 
        self.scenes_dir = self.logdir / "visualize" / "scenes"
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.scenes_dir.mkdir(parents=True, exist_ok=True)

        self.mesh_dir = root / "visualize" / "meshes"
        self.mesh_dir.mkdir(parents=True, exist_ok=True)


        self.rounds_csv_path = self.logdir / "rounds.csv"
        self.grasps_csv_path = self.logdir / "grasps.csv"
        self._create_csv_files_if_needed(tgt_sample)

    def _create_csv_files_if_needed(self, tgt_sample ):
        if not self.rounds_csv_path.exists():
            io.create_csv(self.rounds_csv_path, ["round_id", "object_count"])

        if not self.grasps_csv_path.exists() and not tgt_sample:
            # if not tgt_sample:
            columns = [
                "round_id",
                "scene_id",
                "qx",
                "qy",
                "qz",
                "qw",
                "x",
                "y",
                "z",
                "width",
                "score",
                "label",
                "integration_time",
                "planning_time",
            ]
            
        if not self.grasps_csv_path.exists() and tgt_sample:
            columns = [
                "round_id",
                "scene_id",
                "qx",
                "qy",
                "qz",
                "qw",
                "x",
                "y",
                "z",
                "width",
                "score",
                "label",
                "occ_level",
                "integration_time",
                "planning_time",
            ]
            io.create_csv(self.grasps_csv_path, columns)
            

    def last_round_id(self):
        df = pd.read_csv(self.rounds_csv_path)
        return -1 if df.empty else df["round_id"].max()

    def log_round(self, round_id, object_count):
        io.append_csv(self.rounds_csv_path, round_id, object_count)

    def log_mesh(self, scene_mesh, aff_mesh, name):
        scene_mesh.export(self.mesh_dir / (name + "_scene.obj"))
        aff_mesh.export(self.mesh_dir / (name + "_aff.obj"))

    def log_grasp(self, round_id=None, state=None, timings=None, grasp=None, score=None, label=None, occ_level= None, no_valid_grasp=False):
        # log scene
        if not no_valid_grasp:
            tsdf, points = state.tsdf, np.asarray(state.pc.points)
            scene_id = uuid.uuid4().hex
            scene_path = self.scenes_dir / (scene_id + ".npz")
            np.savez_compressed(scene_path, grid=tsdf.get_grid(), points=points)

            # log grasp
            qx, qy, qz, qw = grasp.pose.rotation.as_quat()
            x, y, z = grasp.pose.translation
            width = grasp.width
            label = int(label)
        else:
            scene_id = uuid.uuid4().hex
            qx, qy, qz, qw = 0, 0, 0, 0
            x, y, z = 0, 0, 0
            width = 0
            label = 0
            score = 0
        if not occ_level:
            io.append_csv(
                self.grasps_csv_path,
                round_id,
                scene_id,
                qx,
                qy,
                qz,
                qw,
                x,
                y,
                z,
                width,
                score,
                label,
                timings["integration"],
                timings["planning"],
            )
        else:
            io.append_csv(
                self.grasps_csv_path,
                round_id,
                scene_id,
                qx,
                qy,
                qz,
                qw,
                x,
                y,
                z,
                width,
                score,
                label,
                occ_level,
                timings["integration"],
                timings["planning"],
            )


class Data(object):
    """Object for loading and analyzing experimental data."""

    def __init__(self, logdir):
        self.logdir = logdir
        self.rounds = pd.read_csv(logdir / "rounds.csv")
        self.grasps = pd.read_csv(logdir / "grasps.csv")

    def num_rounds(self):
        return len(self.rounds.index)

    def num_grasps(self):
        return len(self.grasps.index)

    def success_rate(self):
        return self.grasps["label"].mean() * 100

    def percent_cleared(self):
        df = (
            self.grasps[["round_id", "label"]]
            .groupby("round_id")
            .sum()
            .rename(columns={"label": "cleared_count"})
            .merge(self.rounds, on="round_id")
        )
        return df["cleared_count"].sum() / df["object_count"].sum() * 100

    def avg_planning_time(self):
        return self.grasps["planning_time"].mean()

    def read_grasp(self, i):
        scene_id, grasp, label = io.read_grasp(self.grasps, i)
        score = self.grasps.loc[i, "score"]
        scene_data = np.load(self.logdir / "scenes" / (scene_id + ".npz"))

        return scene_data["points"], grasp, score, label
