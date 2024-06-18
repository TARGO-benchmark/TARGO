from pathlib import Path
import time

import numpy as np
import pybullet

from utils_giga import points_equal
from vgn.grasp import Label
from vgn.perception import *
from vgn.utils import btsim, workspace_lines
from vgn.utils.transform import Rotation, Transform
from vgn.utils.misc import apply_noise, apply_translational_noise
from utils_giga import *
import math
from utils_giga import depth_to_point_cloud
import json

# Function: sim_select_scene
def sim_select_scene(sim, indices):
    # urdf_root = sim.urdf_root
    scene = sim.scene
    object_set = sim.object_set
    # size = sim.size
    # sim_selected = ClutterRemovalSim(urdf_root, size,scene, object_set, gui=False)  ## create a new sim
    sim_selected = ClutterRemovalSim(scene, object_set, False)
    sim.urdf_root = Path("data/urdfs")
    # sim_selected = ClutterRemovalSim(sim.urdf_root, sim.size, sim.scene, sim.object_set, gui=sim.gui)  ## create a new sim
    
    # set some attributes
    # sim_selected.gui = False
    sim_selected.add_noise = sim.add_noise
    sim_selected.sideview = sim.sideview
    sim_selected.size = sim.size
    # sim_selected.intrinsic = sim.intrinsic
    intrinsics = CameraIntrinsic(640, 480, 540.0, 540.0, 320.0, 240.0)
    sim_selected.camera = sim_selected.world.add_camera(intrinsics, 0.1, 2.0)
    
    # mesh_pose_list = collect_mesh_pose_list(sim) 
    mesh_pose_dict = collect_mesh_pose_dict(sim)
    for idc in indices:
        pose = Transform.from_matrix(mesh_pose_dict[idc][2])
        if idc == 0:
            mesh_path = mesh_pose_dict[idc][0].replace(".obj",".urdf")
        else:
            mesh_path = find_urdf(mesh_pose_dict[idc][0].replace("_visual.obj",".urdf"))
        sim_selected.world.load_urdf(mesh_path, pose, mesh_pose_dict[idc][1][0])
    return sim_selected
    
    
    # for idc in indices:
    #     # if idc == 0:
    #     pose = Transform.from_matrix(obj_info[idc][2])
    #     if idc == 0:
    #         sim_selected.world.load_urdf(obj_info[idc][0].replace(".obj",".urdf"), pose, 0.6)
    #     else:
    #         sim_selected.world.load_urdf(find_urdf(obj_info[idc][0].replace(".obj",".urdf").replace('meshes_centered','acronym_urdfs_centered')), pose, 1)
    # return sim_selected

class ClutterRemovalSim(object):
    def __init__(self, scene, object_set, size=None, gui=False, seed=None, add_noise=False, sideview=False, save_dir=None, save_freq=8, test_root = None):
        assert scene in ["pile", "packed", "dex-ycb"]

        self.urdf_root = Path("data/urdfs")
        self.scene = scene
        self.object_set = object_set
        self.discover_objects()

        self.global_scaling = {
            "blocks": 1.67,
            "google": 0.7,
            'google_pile': 0.7,
            'google_packed': 0.7,
            
        }.get(object_set, 1.0)
        self.gui = gui
        self.add_noise = add_noise
        self.sideview = sideview

        self.rng = np.random.RandomState(seed) if seed else np.random
        self.world = btsim.BtWorld(self.gui, save_dir, save_freq)
        self.gripper = Gripper(self.world)
        if size:
            self.size = size
        else:
            self.size = 6 * self.gripper.finger_depth
        intrinsic = CameraIntrinsic(640, 480, 540.0, 540.0, 320.0, 240.0)
        self.camera = self.world.add_camera(intrinsic, 0.1, 2.0)

        ## whether exists a directory
        if test_root is not None:
            self.occ_level_dict_path = Path(test_root) / 'occ_level_dict.json'
            if not self.occ_level_dict_path.exists():
                self.occ_level_dict = {}
                self.save_occ_level_dict = True
            else:
                self.occ_level_dict = json.loads(self.occ_level_dict_path.read_text())
                self.save_occ_level_dict = False
        
        ycb_grasps_path = 'data/ycb_farthest_100_grasps.json'
        with open(ycb_grasps_path, 'r') as file:
            ycb_grasps_dict = json.load(file)
        self.ycb_grasps_dict = ycb_grasps_dict

        urdf_path2ycb_id_path = 'data/urdf_path2ycb_id.json'
        with open(urdf_path2ycb_id_path, 'r') as file:
            urdf_path2ycb_id_dict = json.load(file)        

        ## keys change to Path
        self.urdf_path2ycb_id_dict = {Path(k):v for k,v in urdf_path2ycb_id_dict.items()}

    @property
    def num_objects(self):
        return max(0, self.world.p.getNumBodies() - 1)  # remove table from body count

    def discover_objects(self):
        root = self.urdf_root / self.object_set
        self.object_urdfs = [f for f in root.iterdir() if f.suffix == ".urdf"]

    def save_state(self):
        self._snapshot_id = self.world.save_state()

    def restore_state(self):
        self.world.restore_state(self._snapshot_id)

    def reset(self, object_count, target_id=None):
        self.world.reset()
        self.world.set_gravity([0.0, 0.0, -9.81])
        self.draw_workspace()

        if self.gui:
            self.world.p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=0.0,
                cameraPitch=-45,
                cameraTargetPosition=[0.15, 0.50, -0.3],
            )

        table_height = self.gripper.finger_depth
        self.place_table(table_height)

        if self.scene == "pile":
            self.generate_pile_scene(object_count, table_height, target_id=target_id)
        elif self.scene == "packed":
            self.generate_packed_scene(object_count, table_height, target_id=target_id)
        elif self.scene == "dex-ycb":
                self.generate_dex_ycb_scene(object_count, table_height, target_id=target_id)
        else:
            raise ValueError("Invalid scene argument")

    def draw_workspace(self):
        points = workspace_lines(self.size)
        color = [0.5, 0.5, 0.5]
        for i in range(0, len(points), 2):
            self.world.p.addUserDebugLine(
                lineFromXYZ=points[i], lineToXYZ=points[i + 1], lineColorRGB=color
            )

    def place_table(self, height):
        urdf = self.urdf_root / "setup" / "plane.urdf"
        pose = Transform(Rotation.identity(), [0.15, 0.15, height])
        self.world.load_urdf(urdf, pose, scale=0.6)

        # define valid volume for sampling grasps
        lx, ux = 0.02, self.size - 0.02
        ly, uy = 0.02, self.size - 0.02
        lz, uz = height + 0.005, self.size
        self.lower = np.r_[lx, ly, lz]
        self.upper = np.r_[ux, uy, uz]

    def generate_dex_ycb_scene(self, object_count, table_height, target_id=None):
        attempts = 0
        max_attempts = 50

        while self.num_objects < object_count and attempts < max_attempts:
            self.save_state()
            if self.num_objects == 0 and target_id is not None:
                urdf = self.object_urdfs[target_id]
            else:
                urdf = self.rng.choice(self.object_urdfs)
            x = self.rng.uniform(0.08, 0.22)
            y = self.rng.uniform(0.08, 0.22)
            z = 1.0
            # half the chance for packed, the other half for pile
            if np.random.rand() < 0.5:
                angle = self.rng.uniform(0.0, 2.0 * np.pi)
                rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
            else:
                rotation = Rotation.random(random_state=self.rng)
            pose = Transform(rotation, np.r_[x, y, z])
            scale = 1.0
            body = self.world.load_urdf(urdf, pose, scale=scale)
            lower, upper = self.world.p.getAABB(body.uid)
            z = table_height + 0.5 * (upper[2] - lower[2]) + 0.002
            body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))
            self.wait_for_objects_to_rest()
            contacts = self.world.get_contacts(body)
            
            if not all(contact.bodyB.name == "plane" for contact in contacts):
                self.world.remove_body(body)
                self.restore_state()
            else:
                lower, upper = self.world.p.getAABB(body.uid)
                if not self.object_within_bounds(lower, upper):
                    self.world.remove_body(body)
                    self.restore_state()
                else:
                    self.remove_and_wait()
            attempts += 1

    def object_within_bounds(self, lower, upper):
        return (lower[0] >= self.lower[0] and upper[0] <= self.upper[0] and
                lower[1] >= self.lower[1] and upper[1] <= self.upper[1])

    def generate_pile_scene(self, object_count, table_height, target_id=None):
        # place box
        urdf = self.urdf_root / "setup" / "box.urdf"
        pose = Transform(Rotation.identity(), np.r_[0.02, 0.02, table_height])
        box = self.world.load_urdf(urdf, pose, scale=1.3)

        # drop objects
        urdfs = self.rng.choice(self.object_urdfs, size=object_count)
        if target_id is not None:
            random_index = np.random.randint(0, len(urdfs))
            urdfs[random_index] = self.object_urdfs[target_id]
        for urdf in urdfs:
            rotation = Rotation.random(random_state=self.rng)
            xy = self.rng.uniform(1.0 / 3.0 * self.size, 2.0 / 3.0 * self.size, 2)
            pose = Transform(rotation, np.r_[xy, table_height + 0.2])
            scale = self.rng.uniform(0.8, 1.0)
            self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
            self.wait_for_objects_to_rest(timeout=1.0)

        # remove box
        self.world.remove_body(box)
        self.remove_and_wait()

    def generate_packed_scene(self, object_count, table_height, target_id=None):
        attempts = 0
        max_attempts = 12

        while self.num_objects < object_count and attempts < max_attempts:
            self.save_state()
            if self.num_objects == 0 and target_id is not None:
                urdf = self.object_urdfs[target_id]
            else:
                urdf = self.rng.choice(self.object_urdfs)
            x = self.rng.uniform(0.08, 0.22)
            y = self.rng.uniform(0.08, 0.22)
            z = 1.0
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
            pose = Transform(rotation, np.r_[x, y, z])
            scale = self.rng.uniform(0.7, 0.9)
            body = self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
            lower, upper = self.world.p.getAABB(body.uid)
            z = table_height + 0.5 * (upper[2] - lower[2]) + 0.002
            body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))
            self.world.step()

            if self.world.get_contacts(body):
                self.world.remove_body(body)
                self.restore_state()
            else:
                self.remove_and_wait()
            attempts += 1
    
    def acquire_single_tsdf_target_grid_train(self, curr_scene_path = None, target_id=None, resolution=40, model = 'vgn',target_complete = False, fusion_type = None, complete_shape = False, curr_mesh_pose_list = None, input_points = 'tsdf_points', shape_completion = False):
        """
        Render synthetic depth images from viewpoints and integrate into a TSDF.
        If N is None, the viewpoints are equally distributed on a circular trajectory.
        If N is given, the first n viewpoints on a circular trajectory of N points are rendered.
        """
        if model == 'giga_hr':
            resolution = 60

        tsdf = TSDFVolume(self.size, resolution)
        tgt_mask_tsdf = TSDFVolume(self.size, resolution)
        plane = np.load('/usr/stud/dira/GraspInClutter/grasping/setup/plane_sampled.npy')
        plane = plane.astype(np.float32)

        half_size = self.size / 2
        origin_yz = np.r_[half_size, half_size]

        if self.sideview:
            origin = Transform(Rotation.identity(), np.r_[origin_yz, self.size / 3])
            theta, phi = np.pi / 3.0, - np.pi / 2.0
        else:
            origin = Transform(Rotation.identity(), np.r_[origin_yz, 0])
            theta, phi = np.pi / 6.0, 0

        # Position camera
        r = 2.0 * self.size
        extrinsic = camera_on_sphere(origin, r, theta, phi)

        # Render images
        _, _, seg_img = self.camera.render_with_seg(extrinsic)

        depth_img = np.load(curr_scene_path)['depth_imgs']
        tgt_mask, scene_mask = np.load(curr_scene_path)['mask_targ'], seg_img > 0
        
        assert  np.all(scene_mask == (seg_img > 0))
        assert  np.all(tgt_mask == (seg_img == target_id))
        # for i in np.unique(seg_img):
        #     if np.all(tgt_mask[0] == (seg_img == i)):
        #         target_id = i
        # if curr_mesh_pose_list is not None:
        #     if not self.save_occ_level_dict:
        #         occ_level = self.occ_level_dict[curr_mesh_pose_list]
        #     else:
        sim_single = sim_select_scene(self, [0, target_id])
        _, seg_img_single = sim_single.camera.render_with_seg(extrinsic)[1:3]
        # Calculate occlusion level
        occ_level = 1 - np.sum(seg_img == target_id) / np.sum(seg_img_single == 1)
        self.occ_level_dict[curr_mesh_pose_list] = occ_level

        if occ_level > 0.9:
            print("high occlusion level")

        # Print diagnostics
        print("Occlusion level: ", occ_level)
        print("Number of objects: ", np.max(seg_img))

        scene_mask = scene_mask.astype(np.uint8) 
        tgt_mask = tgt_mask.astype(np.uint8) 
        if model in ('vgn', 'giga_aff', 'giga', 'giga_hr') \
            or (model == 'afford_scene_targ_pc' and fusion_type in ('CNN_concat','CNN_add', 'MLP_fusion')) :
            scene_mask = (seg_img >= 0).astype(np.uint8)    ## it should include plane

        tic = time.time()
        tsdf.integrate((depth_img * scene_mask)[0], self.camera.intrinsic, extrinsic)
        tgt_mask_tsdf.integrate((depth_img * tgt_mask)[0], self.camera.intrinsic, extrinsic)
        timing = time.time() - tic
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(self.lower, self.upper)
        
        if model in ('vgn', 'giga_aff', 'giga', 'giga_hr'):
            if not model == 'giga_hr':
                targ_grid, scene_grid = np.load(curr_scene_path)['grid_targ'], np.load(curr_scene_path)['grid_scene']
            elif model == 'giga_hr':
                targ_grid = create_tsdf(self.size, 60, depth_img * tgt_mask, self.camera.intrinsic, np.array(extrinsic.to_list()).reshape(1,7)).get_grid()
                scene_grid = create_tsdf(self.size, 40, depth_img * scene_mask, self.camera.intrinsic, np.array(extrinsic.to_list()).reshape(1,7)).get_grid()
            targ_mask = targ_grid > 0
            return tsdf, timing, scene_grid, targ_grid, targ_mask, occ_level
        
        elif model == 'afford_scene_pc':
            targ_depth_pc = np.load(curr_scene_path)['pc_depth_targ']
            scene_no_targ_pc = np.load(curr_scene_path)['pc_scene_no_targ']
            scene_no_targ_pc = np.concatenate((scene_no_targ_pc, plane), axis=0)
            targ_depth_pc = targ_depth_pc.astype(np.float32)
            targ_depth_pc = targ_depth_pc / 0.3 - 0.5
            scene_no_targ_pc = scene_no_targ_pc / 0.3 - 0.5
            return tsdf, timing,  scene_no_targ_pc, targ_depth_pc, occ_level 
        
        elif model == 'afford_scene_targ_pc':
            if fusion_type in ('CNN_concat','CNN_add', 'MLP_fusion'):
                targ_grid, scene_grid = np.load(curr_scene_path)['grid_targ'], np.load(curr_scene_path)['grid_scene']
                targ_mask = targ_grid > 0
                return tsdf, timing, scene_grid, targ_grid,targ_mask, occ_level
                
            if fusion_type in  ("transformer_concat","transformer_query_scene", "transformer_query_target"):
                if shape_completion:
                    scene_no_targ_pc, targ_depth_pc = np.load(curr_scene_path)['pc_scene_no_targ'], np.load(curr_scene_path)['pc_depth_targ']
                    targ_depth_pc = targ_depth_pc.astype(np.float32)
                    scene_no_targ_pc = np.concatenate((scene_no_targ_pc, plane), axis=0)
                    scene_no_targ_pc = scene_no_targ_pc /0.3 - 0.5
                    targ_depth_pc = targ_depth_pc / 0.3 - 0.5
                    return tsdf, timing, scene_no_targ_pc,  targ_depth_pc, occ_level
                elif not shape_completion:
                    scene_pc, targ_pc = np.load(curr_scene_path)['pc_scene'], np.load(curr_scene_path)['pc_targ']
                    scene_pc = np.concatenate((scene_pc, plane), axis=0)
                    scene_pc = scene_pc / 0.3 - 0.5
                    targ_grid = np.load(curr_scene_path)['grid_targ']
                    targ_pc = targ_pc / 0.3 - 0.5
                    return tsdf, timing, scene_pc, targ_pc, targ_grid, occ_level
                
    def acquire_single_tsdf_target_grid(self, curr_scene_path = None, target_id=None, resolution=40, model = 'vgn',target_complete = False, fusion_type = None, complete_shape = False, curr_mesh_pose_list = None, input_points = 'tsdf_points', shape_completion = False):
        """
        Render synthetic depth images from viewpoints and integrate into a TSDF.
        If N is None, the viewpoints are equally distributed on a circular trajectory.
        If N is given, the first n viewpoints on a circular trajectory of N points are rendered.
        """
        if model == 'giga_hr':
            resolution = 60

        tsdf = TSDFVolume(self.size, resolution)
        tgt_mask_tsdf = TSDFVolume(self.size, resolution)
        plane = np.load('/usr/stud/dira/GraspInClutter/grasping/setup/plane_sampled.npy')
        plane = plane.astype(np.float32)

        half_size = self.size / 2
        origin_yz = np.r_[half_size, half_size]

        if self.sideview:
            origin = Transform(Rotation.identity(), np.r_[origin_yz, self.size / 3])
            theta, phi = np.pi / 3.0, - np.pi / 2.0
        else:
            origin = Transform(Rotation.identity(), np.r_[origin_yz, 0])
            theta, phi = np.pi / 6.0, 0

        # Position camera
        r = 2.0 * self.size
        extrinsic = camera_on_sphere(origin, r, theta, phi)

        # Render images
        _, _, seg_img = self.camera.render_with_seg(extrinsic)

        depth_img = np.load(curr_scene_path)['depth_imgs']
        tgt_mask, scene_mask = np.load(curr_scene_path)['mask_targ'], np.load(curr_scene_path)['mask_scene']
        
        assert  np.all(scene_mask == (seg_img > 0))
        assert  np.all(tgt_mask == (seg_img == target_id))
        
        if curr_mesh_pose_list is not None:
            if not self.save_occ_level_dict:
                occ_level = self.occ_level_dict[curr_mesh_pose_list]
            else:
                sim_single = sim_select_scene(self, [0, target_id])
                _, seg_img_single = sim_single.camera.render_with_seg(extrinsic)[1:3]
                # Calculate occlusion level
                occ_level = 1 - np.sum(seg_img == target_id) / np.sum(seg_img_single == 1)
                self.occ_level_dict[curr_mesh_pose_list] = occ_level

        if occ_level > 0.9:
            print("high occlusion level")

        # Print diagnostics
        print("Occlusion level: ", occ_level)
        print("Number of objects: ", np.max(seg_img))

        scene_mask = scene_mask.astype(np.uint8) 
        tgt_mask = tgt_mask.astype(np.uint8) 
        if model in ('vgn', 'giga_aff', 'giga', 'giga_hr') \
            or (model == 'afford_scene_targ_pc' and fusion_type in ('CNN_concat','CNN_add', 'MLP_fusion')) :
            scene_mask = (seg_img >= 0).astype(np.uint8)    ## it should include plane

        tic = time.time()
        tsdf.integrate((depth_img * scene_mask)[0], self.camera.intrinsic, extrinsic)
        tgt_mask_tsdf.integrate((depth_img * tgt_mask)[0], self.camera.intrinsic, extrinsic)
        timing = time.time() - tic
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(self.lower, self.upper)
        
        if model in ('vgn', 'giga_aff', 'giga', 'giga_hr'):
            if not model == 'giga_hr':
                targ_grid, scene_grid = np.load(curr_scene_path)['grid_targ'], np.load(curr_scene_path)['grid_scene']
            elif model == 'giga_hr':
                targ_grid = create_tsdf(self.size, 60, depth_img * tgt_mask, self.camera.intrinsic, np.array(extrinsic.to_list()).reshape(1,7)).get_grid()
                scene_grid = create_tsdf(self.size, 40, depth_img * scene_mask, self.camera.intrinsic, np.array(extrinsic.to_list()).reshape(1,7)).get_grid()
            targ_mask = targ_grid > 0
            return tsdf, timing, scene_grid, targ_grid, targ_mask, occ_level
        
        elif model == 'afford_scene_pc':
            targ_depth_pc = np.load(curr_scene_path)['pc_depth_targ']
            scene_no_targ_pc = np.load(curr_scene_path)['pc_scene_no_targ']
            scene_no_targ_pc = np.concatenate((scene_no_targ_pc, plane), axis=0)
            targ_depth_pc = targ_depth_pc.astype(np.float32)
            targ_depth_pc = targ_depth_pc / 0.3 - 0.5
            scene_no_targ_pc = scene_no_targ_pc / 0.3 - 0.5
            return tsdf, timing,  scene_no_targ_pc, targ_depth_pc, occ_level            
        
        elif model == 'afford_scene_targ_pc':
            if fusion_type in ('CNN_concat','CNN_add', 'MLP_fusion'):
                targ_grid, scene_grid = np.load(curr_scene_path)['grid_targ'], np.load(curr_scene_path)['grid_scene']
                targ_mask = targ_grid > 0
                return tsdf, timing, scene_grid, targ_grid,targ_mask, occ_level
                
            if fusion_type in  ("transformer_concat","transformer_query_scene", "transformer_query_target"):
                if shape_completion:
                    scene_no_targ_pc, targ_depth_pc = np.load(curr_scene_path)['pc_scene_no_targ'], np.load(curr_scene_path)['pc_depth_targ']
                    targ_depth_pc = targ_depth_pc.astype(np.float32)
                    scene_no_targ_pc = np.concatenate((scene_no_targ_pc, plane), axis=0)
                    scene_no_targ_pc = scene_no_targ_pc /0.3 - 0.5
                    targ_depth_pc = targ_depth_pc / 0.3 - 0.5
                    return tsdf, timing, scene_no_targ_pc,  targ_depth_pc, occ_level
                elif not shape_completion:
                    scene_pc, targ_pc = np.load(curr_scene_path)['pc_scene'], np.load(curr_scene_path)['pc_targ']
                    scene_pc = np.concatenate((scene_pc, plane), axis=0)
                    scene_pc = scene_pc / 0.3 - 0.5
                    targ_grid = np.load(curr_scene_path)['grid_targ']
                    targ_pc = targ_pc / 0.3 - 0.5
                    return tsdf, timing, scene_pc, targ_pc, targ_grid, occ_level
                        
                        
    def execute_grasp(self, grasp, remove=True, allow_contact=False, tgt_id=0, force_targ = False):
        T_world_grasp = grasp.pose
        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp

        approach = T_world_grasp.rotation.as_matrix()[:, 2]
        angle = np.arccos(np.dot(approach, np.r_[0.0, 0.0, -1.0]))
        if angle > np.pi / 3.0:
            # side grasp, lift the object after establishing a grasp
            T_grasp_pregrasp_world = Transform(Rotation.identity(), [0.0, 0.0, 0.1])
            T_world_retreat = T_grasp_pregrasp_world * T_world_grasp
        else:
            T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.1])
            T_world_retreat = T_world_grasp * T_grasp_retreat

        self.gripper.reset(T_world_pregrasp)

        if self.gripper.detect_contact():
            result = Label.FAILURE, self.gripper.max_opening_width
        else:
            self.gripper.move_tcp_xyz(T_world_grasp, abort_on_contact=True)
            if self.gripper.detect_contact() and not allow_contact:
                result = Label.FAILURE, self.gripper.max_opening_width
            else:
                self.gripper.move(0.0)
                self.gripper.move_tcp_xyz(T_world_retreat, abort_on_contact=False)
                if not force_targ:
                    if self.check_success(self.gripper):
                        result = Label.SUCCESS, self.gripper.read()
                        if remove:
                            contacts = self.world.get_contacts(self.gripper.body)
                            self.world.remove_body(contacts[0].bodyB)
                    else:
                        result = Label.FAILURE, self.gripper.max_opening_width
                if force_targ:
                    res, contacts_targ = self.check_success_target_grasp(self.gripper, tgt_id)
                    if res:
                        result = Label.SUCCESS, self.gripper.read()
                        if remove:
                            self.world.remove_body(contacts_targ[0].bodyB)
                    else:
                        result = Label.FAILURE, self.gripper.max_opening_width

        self.world.remove_body(self.gripper.body)

        if remove:
            self.remove_and_wait()

        return result

    def remove_and_wait(self):
        # wait for objects to rest while removing bodies that fell outside the workspace
        removed_object = True
        while removed_object:
            self.wait_for_objects_to_rest()
            removed_object = self.remove_objects_outside_workspace()

    def wait_for_objects_to_rest(self, timeout=2.0, tol=0.01):
        timeout = self.world.sim_time + timeout
        objects_resting = False
        while not objects_resting and self.world.sim_time < timeout:
            # simulate a quarter of a second
            for _ in range(60):
                self.world.step()
            # check whether all objects are resting
            objects_resting = True
            for _, body in self.world.bodies.items():
                if np.linalg.norm(body.get_velocity()) > tol:
                    objects_resting = False
                    break

    def remove_objects_outside_workspace(self):
        removed_object = False
        for body in list(self.world.bodies.values()):
            xyz = body.get_pose().translation
            if np.any(xyz < 0.0) or np.any(xyz > self.size):
                self.world.remove_body(body)
                removed_object = True
        return removed_object

    def check_success(self, gripper):
        # check that the fingers are in contact with some object and not fully closed
        contacts = self.world.get_contacts(gripper.body)
        res = len(contacts) > 0 and gripper.read() > 0.1 * gripper.max_opening_width
        return res
    
    def check_success_target_grasp(self, gripper, tgt_id=0):
        # check that the fingers are in contact with some object and not fully closed
        contacts = self.world.get_contacts(gripper.body)
        contacts_targ = []
        for contact in contacts:
            assert contact.bodyA.uid == gripper.body.uid
            if contact.bodyB.uid == tgt_id:
                contacts_targ.append(contact)
            if contact.bodyB.uid != tgt_id:
                print("1")

        res = len(contacts_targ) > 0 and gripper.read() > 0.1 * gripper.max_opening_width
        return res, contacts_targ

    def check_success_valid(self, gripper, tgt_id=0):
        # check that the fingers are in contact with some object and not fully closed
        # contacts = self.world.get_contacts(gripper.body)
        contacts = self.world.get_contacts_valid(gripper.body, tgt_id)
        res = len(contacts) > 0 and gripper.read() > 0.1 * gripper.max_opening_width
        return res


class Gripper(object):
    """Simulated Panda hand."""

    def __init__(self, world):
        self.world = world
        self.urdf_path = Path("data/urdfs/panda/hand.urdf")

        self.max_opening_width = 0.08
        self.finger_depth = 0.05
        self.T_body_tcp = Transform(Rotation.identity(), [0.0, 0.0, 0.022])
        self.T_tcp_body = self.T_body_tcp.inverse()

    def reset(self, T_world_tcp):
        T_world_body = T_world_tcp * self.T_tcp_body
        self.body = self.world.load_urdf(self.urdf_path, T_world_body)
        self.body.set_pose(T_world_body)  # sets the position of the COM, not URDF link
        self.constraint = self.world.add_constraint(
            self.body,
            None,
            None,
            None,
            pybullet.JOINT_FIXED,
            [0.0, 0.0, 0.0],
            Transform.identity(),
            T_world_body,
        )
        self.update_tcp_constraint(T_world_tcp)
        # constraint to keep fingers centered
        self.world.add_constraint(
            self.body,
            self.body.links["panda_leftfinger"],
            self.body,
            self.body.links["panda_rightfinger"],
            pybullet.JOINT_GEAR,
            [1.0, 0.0, 0.0],
            Transform.identity(),
            Transform.identity(),
        ).change(gearRatio=-1, erp=0.1, maxForce=50)
        self.joint1 = self.body.joints["panda_finger_joint1"]
        self.joint1.set_position(0.5 * self.max_opening_width, kinematics=True)
        self.joint2 = self.body.joints["panda_finger_joint2"]
        self.joint2.set_position(0.5 * self.max_opening_width, kinematics=True)

    def update_tcp_constraint(self, T_world_tcp):
        T_world_body = T_world_tcp * self.T_tcp_body
        self.constraint.change(
            jointChildPivot=T_world_body.translation,
            jointChildFrameOrientation=T_world_body.rotation.as_quat(),
            maxForce=300,
        )

    def set_tcp(self, T_world_tcp):
        T_word_body = T_world_tcp * self.T_tcp_body
        self.body.set_pose(T_word_body)
        self.update_tcp_constraint(T_world_tcp)

    def move_tcp_xyz(self, target, eef_step=0.002, vel=0.10, abort_on_contact=True):
        T_world_body = self.body.get_pose()
        T_world_tcp = T_world_body * self.T_body_tcp

        diff = target.translation - T_world_tcp.translation
        n_steps = int(np.linalg.norm(diff) / eef_step)
        dist_step = diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel

        for _ in range(n_steps):
            T_world_tcp.translation += dist_step
            self.update_tcp_constraint(T_world_tcp)
            for _ in range(int(dur_step / self.world.dt)):
                self.world.step()
            if abort_on_contact and self.detect_contact():
                return

    def detect_contact(self, threshold=5):
        if self.world.get_contacts(self.body):
            return True
        else:
            return False

    def move(self, width):
        self.joint1.set_position(0.5 * width)
        self.joint2.set_position(0.5 * width)
        for _ in range(int(0.5 / self.world.dt)):
            self.world.step()

    def read(self):
        width = self.joint1.get_position() + self.joint2.get_position()
        return width
