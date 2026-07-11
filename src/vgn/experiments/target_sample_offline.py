import collections
import argparse
import json
import math
import numpy as np
import os
import pandas as pd
import pyrender
import time
import tqdm
import trimesh
import uuid
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from src.vgn import io  # For CSV creation and I/O
from src.vgn.grasp import *
from src.vgn.simulation import ClutterRemovalSim
from src.vgn.utils.transform import Rotation, Transform
from src.vgn.utils.implicit import (
    get_mesh_pose_list_from_world,
    get_scene_from_mesh_pose_list
)
from src.vgn.perception import camera_on_sphere
# Debug / logging utilities
from src.utils_targo import (
    record_occ_level_count,
    record_occ_level_success,
    cal_occ_level_sr,
    save_point_cloud_as_ply,
    generate_and_transform_grasp_meshes,
    compute_chamfer_and_iou
)

MAX_CONSECUTIVE_FAILURES = 2
State = collections.namedtuple("State", ["tsdf", "pc"])


def init_occ_level_count_dict():
    """
    Create a dictionary to count the number of scenes in each occlusion bin.
    """
    return {
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


def init_occ_level_success_dict():
    """
    Create a dictionary to record successful grasps in each occlusion bin.
    """
    return {
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


def run(
    grasp_plan_fn,
    logdir,
    description,
    scene,
    object_set,
    sim_gui=False,
    result_path=None,
    add_noise=False,
    sideview=False,
    resolution=40,
    visualize=False,
    type='targo',
    test_root=None,
):
 
    # Initialize the simulation
    sim = ClutterRemovalSim(
        scene, object_set,
        gui=sim_gui,
        add_noise=add_noise,
        sideview=sideview,
        test_root=test_root
    )
    logger = Logger(logdir, description, tgt_sample=True)

    # Track planning times
    planning_times, total_times = [], []

    # Paths for test data
    test_mesh_pose_list = f'{test_root}/mesh_pose_dict/'
    test_scenes = f'{test_root}/scenes/'

    # Dictionaries to track occlusion stats
    occ_level_count_dict = init_occ_level_count_dict()
    occ_level_success_dict = init_occ_level_success_dict()
    offline_occ_level_dict = {}

    plan_failure_count = 0
    visual_failure_count = 0

    count_label_dict = {}
    height_label_dict = {}
    tgt_bbx_label_dict = {}
    skip_dict = {}

    # Loop over the test set
    for num_id, curr_mesh_pose_list in enumerate(os.listdir(test_mesh_pose_list)):
        path_to_npz = os.path.join(test_scenes, curr_mesh_pose_list)
        scene_name = curr_mesh_pose_list[:-4]
        if scene_name not in sim.occ_level_dict:
            # os.remove(os.path.join(test_mesh_pose_list, curr_mesh_pose_list))
            # os.remove(path_to_npz)
            continue

        # Prepare simulator
        sim.world.reset()
        sim.world.set_gravity([0.0, 0.0, -9.81])
        sim.draw_workspace()
        sim.save_state()

        # Manually adjust boundaries
        sim.lower = np.array([0.02, 0.02, 0.055])
        sim.upper = np.array([0.28, 0.28, 0.30000000000000004])

        tgt_id = int(scene_name[-1])  
        start_time = time.time()

        occluder_heights = []

        mp_data = np.load(
            os.path.join(test_mesh_pose_list, curr_mesh_pose_list),
            allow_pickle=True
        )["pc"]

        # Place objects
        for obj_id, mesh_info in enumerate(mp_data.item().values()):
            pose = Transform.from_matrix(mesh_info[2])
            if mesh_info[0].split('/')[-1] == 'plane.obj':
                urdf_path = mesh_info[0].replace(".obj", ".urdf")
            else:
                urdf_path = mesh_info[0].replace("_visual.obj", ".urdf")

            body = sim.world.load_urdf(
                urdf_path=urdf_path,
                pose=pose,
                scale=mesh_info[1]
            )

            # Get bounding heights
            if obj_id != 0:
                tri_mesh = trimesh.load_mesh(mesh_info[0])
                tri_mesh.apply_scale(mesh_info[1])
                tri_mesh.apply_transform(mesh_info[2])
                z_max = tri_mesh.vertices[:, 2].max()

                if obj_id != tgt_id:
                    occluder_heights.append(z_max)
                else:
                    tgt_height = z_max
                    min_coords = tri_mesh.vertices.min(axis=0)
                    max_coords = tri_mesh.vertices.max(axis=0)
                    length, width, height = max_coords - min_coords

            # Color target red
            if obj_id == tgt_id:
                body.set_color(link_index=-1, rgba_color=(1.0, 0.0, 0.0, 1.0))

        end_time = time.time()
        if len(occluder_heights) == 0:
            relative_height = tgt_height
        else:
            relative_height = tgt_height - np.max(occluder_heights)

        # print(f"load {num_id}-th {scene_name} took {end_time - start_time:.2f}s")

        # Acquire data for shape completion
        timings = {}
        start_time = time.time()

        tsdf, timings["integration"], scene_no_targ_pc, targ_pc, occ_level = \
            sim.acquire_single_tsdf_target_grid(
                path_to_npz,
                tgt_id,
                40,
                'targo',  
                curr_mesh_pose_list=scene_name,
            )
        state = argparse.Namespace(
            tsdf=tsdf,
            scene_no_targ_pc=scene_no_targ_pc,
            targ_pc=targ_pc,
            occ_level=occ_level,
            type='targo'
        )

        end_time = time.time()
        # print(f"acquire {num_id}-th {scene_name} took {end_time - start_time:.2f}s for shape-completion input")

        # Record occlusion
        occ_level_count_dict = record_occ_level_count(occ_level, occ_level_count_dict)
        offline_occ_level_dict[scene_name] = occ_level

        # Skip if occlusion > 0.9
        if occ_level >= 0.9:
            print("skip")
            skip_dict[scene_name] = occ_level
            continue

        # Plan the grasp
        # if visualize:
        #     visual_dict = {'mesh_name': scene_name, 'mesh_dir': logger.mesh_dir}
        #     mesh_pose_list = get_mesh_pose_list_from_world(sim.world, object_set)
        #     scene_mesh = get_scene_from_mesh_pose_list(mesh_pose_list, tgt_id - 1)
        #     grasps, scores, timings["planning"], visual_mesh = grasp_plan_fn(state, scene_mesh, visual_dict)

        #     # Render snapshot
        #     origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, sim.size / 3])
        #     r = 2 * sim.size
        #     theta = np.pi / 3.0
        #     phi = - np.pi / 2.0
        #     extrinsic = camera_on_sphere(origin, r, theta, phi)
        #     rgb, _, _ = sim.camera.render_with_seg(extrinsic)
        #     output_path = f'{logger.mesh_dir}/{occ_level}_occ_{scene_name}_rgb.png'
        #     plt.imsave(output_path, rgb)
        # else:
        #     # time_begin = time.time()
        #     grasps, scores, timings["planning"] = grasp_plan_fn(state)
        #     # time_end = time.time()
        #     # print(f"planning time={time_end - time_begin}")
        
        mesh_pose_list = get_mesh_pose_list_from_world(sim.world, object_set)
        result = get_scene_from_mesh_pose_list(mesh_pose_list, tgt_id - 1, return_target_mesh=True)
        
        if result is None:
            print(f"Error: Could not get scene and target mesh for scene {scene_name}, skipping")
            continue
            
        scene_mesh, target_mesh = result
        grasps, scores, timings["planning"], visual_dict = grasp_plan_fn(state, scene_mesh)

        planning_times.append(timings["planning"])
        total_times.append(timings["planning"] + timings["integration"])

        if len(grasps) == 0:
            continue

        grasp, score = grasps[0], scores[0]

        # Combine scene_no_targ_pc + targ_pc if desired, for local debugging
        scene_pc = None
        if hasattr(state, 'scene_no_targ_pc') and hasattr(state, 'targ_pc'):
            # Merge for any local inspection
            scene_pc = np.concatenate([state.scene_no_targ_pc, state.targ_pc], axis=0)
            scene_pc = (scene_pc + 0.5) * 0.3  # revert from [-0.5,0.5] to real world coords

        if scene_pc is not None:
            generate_and_transform_grasp_meshes(
                grasp, scene_pc,
                './demo_targo'
            )

        grasp.width = sim.gripper.max_opening_width

        # Execute grasp in simulation
        label, plan_fail, visual_fail = sim.execute_grasp(
            grasp,
            allow_contact=True,
            tgt_id=tgt_id,
            force_targ=True
        )
        if plan_fail == 1:
            plan_failure_count += 1
        if visual_fail == 1:
            visual_failure_count += 1

        if label == Label.FAILURE:
            ## create a scene
            scene_path = logdir / scene_name
            scene_path.mkdir(parents=True, exist_ok=True)
            metadata_path = scene_path / "scene_metadata.txt"

            chamfer_distance, iou_approx = compute_chamfer_and_iou(target_mesh, visual_dict['completed_targ_pc'], mesh_path=scene_path)

            meta = {
                'scene_id': scene_name,
                'cd': chamfer_distance * 1000,  # Scale Chamfer Distance by 1000
                'iou': iou_approx * 100,  # Scale IoU by 100
                'success': 0,
                'time': datetime.now().isoformat(),
                'occlusion': occ_level * 100,  # Scale occlusion level by 100 to show as percentage
            }

            # 保存到meta_data_path
            with open(metadata_path, 'w') as f:
                json.dump(meta, f)

            # visual_dict = {'mesh_name': scene_name, 'mesh_dir': logger.mesh_dir}
            # mesh_pose_list = get_mesh_pose_list_from_world(sim.world, object_set)
            # scene_mesh = get_scene_from_mesh_pose_list(mesh_pose_list, tgt_id - 1)
            # grasps, scores, timings["planning"], visual_mesh = grasp_plan_fn(state, scene_mesh, visual_dict)

            # Render snapshot
            origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, sim.size / 3])
            r = 2 * sim.size
            theta = np.pi / 3.0
            phi = - np.pi / 2.0
            extrinsic = camera_on_sphere(origin, r, theta, phi)
            rgb, _, _ = sim.camera.render_with_seg(extrinsic)
            # output_path = f'{logger.mesh_dir}/{occ_level}_occ_{scene_name}_rgb.png'
            img_path = scene_path / "scene_rgb.png"
            plt.imsave(img_path, rgb)
            visual_dict['composed_scene'].export(f'{scene_path}/composed_scene.obj')
            visual_dict['affordance_visual'].export(f'{scene_path}/affordance_visual.obj')

            save_point_cloud_as_ply(visual_dict['completed_targ_pc'], scene_path / "completed_targ_pc.ply")
            save_point_cloud_as_ply(visual_dict['targ_pc'], scene_path / "targ_pc.ply")

            gt_targ_pc, _ = trimesh.sample.sample_surface(target_mesh, count=2048)
            gt_targ_pc = (gt_targ_pc + 0.5) * 0.3
            save_point_cloud_as_ply(gt_targ_pc, scene_path / "gt_targ_pc.ply")
            




        count_label_dict[scene_name] = (int(label), len(occluder_heights), occ_level)
        height_label_dict[scene_name] = (int(label), relative_height, occ_level)
        tgt_bbx_label_dict[scene_name] = (int(label), length, width, height, occ_level)

        # If success, record success in occlusion bin
        if label != Label.FAILURE:
            occ_level_success_dict = record_occ_level_success(occ_level, occ_level_success_dict)
            # if visualize:
            #     logger.log_mesh(scene_mesh, visual_mesh, f'{occ_level}_occ_{scene_name}')

        # Optionally partial stats
        if all(v > 0 for v in occ_level_count_dict.values()) and (num_id % 100 == 0):
            occ_level_sr = cal_occ_level_sr(occ_level_count_dict, occ_level_success_dict)
            curr_count = sum(occ_level_count_dict.values())
            intermediate_result_path = f'{result_path}/intermediate_result.txt'
            with open(intermediate_result_path, 'a') as f:
                f.write(f"current total count:{curr_count}\n")
                for key, val in occ_level_sr.items():
                    f.write(f"{key}:{val}\n")
                f.write('\n')

    # Possibly save updated occlusion dictionary
    if sim.save_occ_level_dict:
        with open(sim.occ_level_dict_path, 'w') as f:
            json.dump(sim.occ_level_dict, f)

    # Final occlusion stats
    final_sr = cal_occ_level_sr(occ_level_count_dict, occ_level_success_dict)

    with open(f'{result_path}/occ_level_sr.json', 'w') as f:
        json.dump(final_sr, f)
    with open(f'{result_path}/occ_level_count_dict.json', 'w') as f:
        json.dump(occ_level_count_dict, f)
    with open(f'{result_path}/count_label_dict_0.json', 'w') as f:
        json.dump(count_label_dict, f)
    with open(f'{result_path}/height_label_dict_0.json', 'w') as f:
        json.dump(height_label_dict, f)
    with open(f'{result_path}/tgt_bbx_label_dict_0.json', 'w') as f:
        json.dump(tgt_bbx_label_dict, f)

    with open(f'{result_path}/visual_failure_count.txt', 'w') as f:
        f.write(f"visual_failure_count:{visual_failure_count}\n")
        f.write(f"plan_failure_count:{plan_failure_count}\n")

    print("done")
    return final_sr


class Logger(object):
    """
    Logger handles:
    - Writing data about rounds to rounds.csv
    - Writing data about grasps to grasps.csv
    - Storing scene data (TSDF, pc, etc.) to .npz
    - Saving meshes for visualization
    """
    def __init__(self, root, description, tgt_sample=False):
        self.logdir = root / "visualize"
        self.scenes_dir = self.logdir / "visualize" / "scenes"
        # self.logdir.mkdir(parents=True, exist_ok=True)
        # self.scenes_dir.mkdir(parents=True, exist_ok=True)

        self.mesh_dir = root / "visualize" / "meshes"
        # self.mesh_dir.mkdir(parents=True, exist_ok=True)

        # self.rounds_csv_path = self.logdir / "rounds.csv"
        # self.grasps_csv_path = self.logdir / "grasps.csv"
        # self._create_csv_files_if_needed(tgt_sample)

    def _create_csv_files_if_needed(self, tgt_sample):
        """Make rounds.csv and grasps.csv if they do not exist."""
        if not self.rounds_csv_path.exists():
            io.create_csv(self.rounds_csv_path, ["round_id", "object_count"])

        if not self.grasps_csv_path.exists():
            if not tgt_sample:
                columns = [
                    "round_id", "scene_id",
                    "qx", "qy", "qz", "qw",
                    "x", "y", "z",
                    "width", "score", "label",
                    "integration_time", "planning_time",
                ]
            else:
                columns = [
                    "round_id", "scene_id",
                    "qx", "qy", "qz", "qw",
                    "x", "y", "z",
                    "width", "score", "label",
                    "occ_level",
                    "integration_time", "planning_time",
                ]
            io.create_csv(self.grasps_csv_path, columns)

    def last_round_id(self):
        """
        Return the last round_id from rounds.csv, or -1 if empty.
        """
        df = pd.read_csv(self.rounds_csv_path)
        return -1 if df.empty else df["round_id"].max()

    def log_round(self, round_id, object_count):
        """Append a row to rounds.csv."""
        io.append_csv(self.rounds_csv_path, round_id, object_count)

    def log_mesh(self, scene_mesh, aff_mesh, name):
        """Save two OBJ meshes (scene + affordance)."""
        scene_mesh.export(self.mesh_dir / (name + "_scene.obj"))
        aff_mesh.export(self.mesh_dir / (name + "_aff.obj"))

    def log_grasp(
        self,
        round_id=None,
        state=None,
        timings=None,
        grasp=None,
        score=None,
        label=None,
        occ_level=None,
        no_valid_grasp=False
    ):
        """
        Save TSDF/points to .npz, then write one row to grasps.csv.
        If no_valid_grasp=True, log placeholder data.
        """
        if not no_valid_grasp:
            tsdf, points = state.tsdf, np.asarray(state.pc.points)
            scene_id = uuid.uuid4().hex
            scene_path = self.scenes_dir / (scene_id + ".npz")
            np.savez_compressed(scene_path, grid=tsdf.get_grid(), points=points)

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

        if occ_level is None:
            io.append_csv(
                self.grasps_csv_path,
                round_id,
                scene_id,
                qx, qy, qz, qw,
                x, y, z,
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
                qx, qy, qz, qw,
                x, y, z,
                width,
                score,
                label,
                occ_level,
                timings["integration"],
                timings["planning"],
            )


class Data(object):
    """
    Helps load logs from a folder with:
    - rounds.csv
    - grasps.csv
    - a 'scenes' folder (with .npz of scene states).
    """
    def __init__(self, logdir):
        self.logdir = logdir
        self.rounds = pd.read_csv(logdir / "rounds.csv")
        self.grasps = pd.read_csv(logdir / "grasps.csv")

    def num_rounds(self):
        return len(self.rounds.index)

    def num_grasps(self):
        return len(self.grasps.index)

    def success_rate(self):
        """Percent of successful grasps (label == 1)."""
        return self.grasps["label"].mean() * 100

    def percent_cleared(self):
        """
        Ratio of total successful grasps to total objects, aggregated over rounds.
        """
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
        """
        Return the i-th grasp, including:
         - scene point cloud
         - grasp pose
         - predicted score
         - success/failure label
        """
        scene_id, grasp, label = io.read_grasp(self.grasps, i)
        score = self.grasps.loc[i, "score"]
        scene_data = np.load(self.logdir / "scenes" / (scene_id + ".npz"))
        return scene_data["points"], grasp, score, label

def validate_point_count(count):
    """Validate point sampling count.
    
    Args:
        count: number of points to sample
        
    Raises:
        ValueError: if count is invalid
    """
    if not isinstance(count, int) or count <= 0:
        raise ValueError(f"Point count must be positive integer, got {count}")
