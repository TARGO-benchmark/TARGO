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
    validate=True,
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
    cnt = 0
    success = 0
    left_objs = 0
    # total_objs = 0
    cons_fail = 0
    no_grasp = 0
    planning_times = []
    total_times = []

    for _ in tqdm.tqdm(range(num_rounds), disable=silence):
        sim.reset(num_objects)

        round_id = logger.last_round_id() + 1
        logger.log_round(round_id, sim.num_objects)
        # total_objs += sim.num_objects
        consecutive_failures = 1
        last_label = None
        trial_id = -1

        ## target removal
        trial_id += 1
        timings = {}
        obj_id = [key for key in sim.world.bodies.keys() if key != 0]
        tgt_id = np.random.choice(obj_id)   ## randomly choose a target object
        
        if not validate:
            clutter_tsdf, pc, timings["integration"], tgt_mask_vol, occ_level = sim.acquire_single_tsdf_target_mask(tgt_id, 40)
            state = argparse.Namespace(tsdf= clutter_tsdf, pc=pc, tgt_mask_vol=tgt_mask_vol, occ_level=occ_level)
        else:
            clutter_tsdf, single_tsdf, double_tsdfs, sim_single, sims_double, pc, timings["integration"], tgt_mask_vol, occ_level = sim.acquire_clutter_single_tsdf_double_tsdfs_target_mask(tgt_id, 40)
        state = argparse.Namespace(tsdf= clutter_tsdf, pc=pc, tgt_mask_vol=tgt_mask_vol, occ_level=occ_level)
        
        if resolution != 40:
            extra_tsdf, _, _ = sim.acquire_tsdf(n=n, N=N, resolution=resolution)
            state.tsdf_process = extra_tsdf

        if pc.is_empty():
            continue
        
        # plan grasps
        if visualize:
            mesh_pose_list = get_mesh_pose_list_from_world(sim.world, object_set)
            scene_mesh = get_scene_from_mesh_pose_list(mesh_pose_list)
            grasps, scores, timings["planning"], visual_mesh = grasp_plan_fn(state, scene_mesh)
            logger.log_mesh(scene_mesh, visual_mesh, f'round_{round_id:03d}_trial_{trial_id:03d}')
        else:
            grasps, scores, timings["planning"] = grasp_plan_fn(state)
        
        ## validate set theory 
            ## union of double scenes' labels should the same as the single scene's label
            ## intersection of double scenes' labels should the same as the clutter scene's label   
        if validate:
            ## single scene
            grasps_single, scores_single, timings_single = grasp_plan_fn(argparse.Namespace(tsdf= single_tsdf, pc=pc, tgt_mask_vol=tgt_mask_vol, occ_level=occ_level))
            double_scores, double_grasps = {}, {}
            for key in double_tsdfs.keys():
                double_grasps[key], double_scores[key],_ = grasp_plan_fn(argparse.Namespace(tsdf= double_tsdfs[key], pc=pc, tgt_mask_vol=tgt_mask_vol, occ_level=occ_level))
            
            
            ## union of double scenes' labels should the same as the single scene's label
            
            
        planning_times.append(timings["planning"])
        total_times.append(timings["planning"] + timings["integration"])

        if len(grasps) == 0:
            no_grasp += 1
            # break  # no detections found, abort this round
            continue

        # execute grasp
        grasp, score = grasps[0], scores[0]
        label, _ = sim.execute_grasp(grasp, allow_contact=True)
        if validate:
            label_single, _ = sim_single.execute_grasp(grasps_single[0], allow_contact=True)
            labels_double = {}
            # for key in double_tsdfs.keys():
            for key, value in sims_double.items():
                labels_double[key], _ = value.execute_grasp(double_grasps[key][0], allow_contact=True)
            union_double_label = 1
            intersection_double_label = 0
            for key, value in labels_double.items():
                # intersec_double_label  *= value
                union_double_label  *= int(value)
                intersection_double_label  += int(value)
            
            validate = (int(label_single)== int(union_double_label)) and (int(label) == int(intersection_double_label>0))
            
            
                # if value != label:
                    # print('error')
            
        cnt += 1
        if label != Label.FAILURE:
            success += 1

        # log the grasp
        logger.log_grasp(round_id, state, timings, grasp, score, label, occ_level, validate=validate)

        if last_label == Label.FAILURE and label == Label.FAILURE:
            consecutive_failures += 1
        else:
            consecutive_failures = 1
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            cons_fail += 1
        last_label = label
    left_objs += sim.num_objects
    success_rate = 100.0 * success / cnt
    print(f'Average planning time: {np.mean(planning_times)}, total time: {np.mean(total_times)}')
    if result_path is not None:
        with open(result_path, 'w') as f:
            f.write('%.2f%%, %.2f%%; %d, %d\n' % (success_rate, cons_fail, no_grasp))
    return success_rate
    


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

    def log_grasp(self, round_id, state, timings, grasp, score, label, occ_level= None, validate=None):
        # log scene
        tsdf, points = state.tsdf, np.asarray(state.pc.points)
        scene_id = uuid.uuid4().hex
        scene_path = self.scenes_dir / (scene_id + ".npz")
        np.savez_compressed(scene_path, grid=tsdf.get_grid(), points=points)

        # log grasp
        qx, qy, qz, qw = grasp.pose.rotation.as_quat()
        x, y, z = grasp.pose.translation
        width = grasp.width
        label = int(label)
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
