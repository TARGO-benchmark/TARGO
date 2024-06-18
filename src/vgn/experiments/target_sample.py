import collections
import argparse
from datetime import datetime
import uuid

import numpy as np
import pandas as pd
import tqdm
import math
from vgn import io#, vis
from vgn.grasp import *
from vgn.simulation import ClutterRemovalSim
from vgn.utils.transform import Rotation, Transform
from vgn.utils.implicit import get_mesh_pose_list_from_world, get_scene_from_mesh_pose_list
# from MinkowskiEngine as ME
import MinkowskiEngine as ME
MAX_CONSECUTIVE_FAILURES = 2


State = collections.namedtuple("State", ["tsdf", "pc"])


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
    seed=1,
    sim_gui=False,
    result_path=None,
    add_noise=False,
    sideview=False,
    resolution=40,
    silence=False,
    visualize=False,
    tgt_sample=False,
    complete_shape=False,
    type = 'giga_aff',
    fusion_type = 'MLP_fusion',
):
    """Run several rounds of simulated clutter removal experiments.

    Each round, m objects are randomly placed in a tray. Then, the grasping pipeline is
    run until (a) no objects remain, (b) the planner failed to find a grasp hypothesis,
    or (c) maximum number of consecutive failed grasp attempts.
    """
    #sideview=False
    #n = 6
    sim = ClutterRemovalSim(scene, object_set, gui=sim_gui, seed=seed, add_noise=add_noise, sideview=sideview)
    logger = Logger(logdir, description,tgt_sample=tgt_sample)
    cnt, success, left_objs, total_objs, cons_fail, no_grasp = 0, 0, 0, 0, 0, 0
    planning_times, total_times = [], []

    for _ in tqdm.tqdm(range(num_rounds), disable=silence):
        sim.reset(num_objects)
        round_id = logger.last_round_id() + 1
        logger.log_round(round_id, sim.num_objects)
        total_objs += sim.num_objects
        trial_id, consecutive_failures, last_label = -1, 1, None
        trial_id += 1
        timings = {}

        obj_id = [key for key in sim.world.bodies.keys() if key != 0]
        tgt_id = np.random.choice(obj_id)
        if type in ('vgn', 'giga_aff', 'giga'):
            if complete_shape:
                tsdf_complete, tsdf, pc, timings["integration"], tgt_mask_vol, occ_level = sim.acquire_complete_tsdf_target_mask(tgt_id, 40)
                state = argparse.Namespace(tsdf=tsdf_complete, pc=pc, tgt_mask_vol=tgt_mask_vol, occ_level=occ_level,type = type)
            else:
                # tsdf, pc, timings["integration"], tgt_mask_vol, occ_level, targ_pc = sim.acquire_single_tsdf_target_mask(tgt_id, 40)
                # state = argparse.Namespace(tsdf=tsdf, pc=pc, tgt_mask_vol=tgt_mask_vol, occ_level=occ_level,type = type)
                tsdf, pc, timings["integration"], tgt_grid, tgt_mask_vol, occ_level, targ_pc = sim.acquire_single_tsdf_target_grid(tgt_id, 40, complete_shape = complete_shape)
                state = argparse.Namespace(tsdf=tsdf, pc=pc, tgt_mask_vol=tgt_mask_vol, occ_level=occ_level,type = type)
        # else:
        # if type in 'giga_aff_plus_target_input':
        if type in ('afford_scene_targ_pc',):
            if fusion_type not in ('transformer_query_scene','transformer_query_target', 'transformer_concat'):
                tsdf, pc, timings["integration"], tgt_grid, tgt_mask_vol, occ_level, targ_pc = sim.acquire_single_tsdf_target_grid(tgt_id, 40,fusion_type = fusion_type, complete_shape=complete_shape)
                state = argparse.Namespace(tsdf=tsdf, pc=pc,tgt_grid = tgt_grid, tgt_mask_vol=tgt_mask_vol, occ_level=occ_level,type = type)
            else:
                tsdf, pc, timings["integration"], tgt_grid, tgt_mask_vol, occ_level, scene_pc, targ_pc = sim.acquire_single_tsdf_target_grid(tgt_id, 40,fusion_type = fusion_type, complete_shape=complete_shape)
                state = argparse.Namespace(tsdf=tsdf, pc=pc,tgt_grid = tgt_grid, tgt_mask_vol=tgt_mask_vol, occ_level=occ_level,scene_pc = scene_pc, targ_pc = targ_pc,type = type)
        
        if resolution != 40:
            extra_tsdf, _, _ = sim.acquire_tsdf(n=n, N=N, resolution=resolution)
            state.tsdf_process = extra_tsdf
        
        if occ_level  == 1 or pc.is_empty() or targ_pc.shape[0] == 0:
            print("skip")
        # if occ_level > 0.95:
            continue
        targ_quantized_pc = ME.utils.sparse_quantize(targ_pc, quantization_size=0.0075)
        if targ_quantized_pc.size(0) <= 2:
            continue

        # plan grasps
        if visualize:
            mesh_pose_list = get_mesh_pose_list_from_world(sim.world, object_set)
            scene_mesh = get_scene_from_mesh_pose_list(mesh_pose_list,tgt_id)
            grasps, scores, timings["planning"], visual_mesh = grasp_plan_fn(state, scene_mesh)
            logger.log_mesh(scene_mesh, visual_mesh, f'round_{round_id:03d}_trial_{trial_id:03d}')
        else:
            grasps, scores, timings["planning"] = grasp_plan_fn(state)
        planning_times.append(timings["planning"])
        total_times.append(timings["planning"] + timings["integration"])

        if len(grasps) == 0:
            no_grasp += 1
            logger.log_grasp(round_id, state, timings, None, None, 0, occ_level, no_valid_grasp=True)

            # break  # no detections found, abort this round
            continue

        # execute grasp
        grasp, score = grasps[0], scores[0]
        # grasp.width *= 1.5
        grasp.width = sim.gripper.max_opening_width
        label, _ = sim.execute_grasp(grasp, allow_contact=True)
        cnt += 1
        if label != Label.FAILURE:
            success += 1

        # log the grasp
        logger.log_grasp(round_id, state, timings, grasp, score, label, occ_level)

        if last_label == Label.FAILURE and label == Label.FAILURE:
            consecutive_failures += 1
        else:
            consecutive_failures = 1
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            cons_fail += 1
        last_label = label
        left_objs += sim.num_objects

    success_rate = 100.0 * success / cnt
    declutter_rate = 100.0 * success / total_objs
    print('Grasp success rate: %.2f %%, Declutter rate: %.2f %%' % (success_rate, declutter_rate))
    print(f'Average planning time: {np.mean(planning_times)}, total time: {np.mean(total_times)}')
    #print('Consecutive failures and no detections: %d, %d' % (cons_fail, no_grasp))
    if result_path is not None:
        with open(result_path, 'w') as f:
            f.write('%.2f%%, %.2f%%; %d, %d\n' % (success_rate, declutter_rate, cons_fail, no_grasp))
    return success_rate, declutter_rate
    


class Logger(object):
    def __init__(self, root, description, tgt_sample=False):
        time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        description = "{}_{}".format(time_stamp, description).strip("_")

        self.logdir = root / description
        self.scenes_dir = self.logdir / "scenes"
        self.scenes_dir.mkdir(parents=True, exist_ok=True)

        self.mesh_dir = self.logdir / "meshes"
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
