from pathlib import Path
import time
import numpy as np
import pybullet
import json
import math
import open3d as o3d
import gzip
import matplotlib.pyplot as plt
from typing import Union
from utils_targo import points_equal, collect_mesh_pose_dict, find_urdf, get_pc_scene_no_targ_kdtree
from vgn.grasp import Label
from vgn.perception import TSDFVolume, CameraIntrinsic, create_tsdf, camera_on_sphere
from vgn.utils import btsim, workspace_lines
from utils_targo import save_point_cloud_as_ply
from vgn.utils.transform import Rotation, Transform
import pathlib
from utils_giga import point_cloud_to_tsdf, mesh_to_tsdf
import random
import trimesh
import trimesh.transformations as tra
import os
import torch

# CLIP imports
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not available. CLIP features will not be generated.")

class SceneObject:
    visual_fpath: pathlib.Path
    collision_fpath: pathlib.Path
    pose4x4: np.ndarray = np.eye(4)
    scale: np.ndarray = np.ones(3)
    density: float = 1000.0
    name: str = ""

def load_json(filename: Union[str, pathlib.Path]):
    filename = str(filename)
    if filename.endswith(".gz"):
        f = gzip.open(filename, "rt")
    elif filename.endswith(".json"):
        f = open(filename, "rt")
    else:
        raise RuntimeError(f"Unsupported extension: {filename}")
    ret = json.loads(f.read())
    f.close()
    return ret


def load_json_or_default(filename: Union[str, pathlib.Path], default):
    filename = Path(filename)
    if not filename.exists():
        return default
    return load_json(filename)


class YCBPathsLoader:
    ycb_root = Path("./data//maniskill_ycb")
    MODEL_JSON = ycb_root / "mani_skill2_ycb/info_pick_v0.json"
    HEAVY_OBJECTS = ["006_mustard_bottle"]


    def __init__(self) -> None:
        if not self.MODEL_JSON.exists():
            raise FileNotFoundError(
                f"json file ({self.MODEL_JSON}) is not found."
                "To download default json:"
                "`python -m mani_skill2.utils.download_asset pick_clutter_ycb`."
            )
        self.model_db: dict[str, dict] = load_json(self.MODEL_JSON)
        self.mesh_names = sorted(list(self.model_db.keys()))
        self.mesh_paths = [
            self.ycb_root / f"mani_skill2_ycb/models/{n}/textured.obj" for n in self.mesh_names   
        ]


    def __len__(self):
        return len(self.model_db)


    def get_random(self) -> pathlib.Path:
        return random.choice(self.mesh_paths)


    def meshpath_to_sceneobj(
        self, meshpath: pathlib.Path, pose: np.ndarray = np.eye(4)
    ) -> SceneObject:
        name = meshpath.parent.stem
        visual_path = str(meshpath)
        collision_path = visual_path.replace("textured.obj", "collision.obj")
        scale = self.model_db[name].get("scales", 1) * np.ones(3)
        density = self.model_db[name].get("density", 1000)
        if name in self.HEAVY_OBJECTS:
            density /= 3
        return SceneObject(
            pathlib.Path(visual_path), pathlib.Path(collision_path), pose, scale, density, name
        )
    

def sim_select_vgn_scene(sim, indices):
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

# Simple function to get a single-scene simulator
def sim_select_scene(sim, indices):
    """
    Create a new simulator with only the objects whose indices are specified.
    """
    scene = sim.scene
    object_set = sim.object_set
    sim_selected = ClutterRemovalSim(scene, object_set, False)
    sim.urdf_root = Path("data/urdfs")
    sim.ycb_root = Path("./data//urdfs/maniskill_ycb/mani_skill2_ycb/models")
    sim.g1b_root = Path("./data//urdfs/g1b/models")
    sim.acroym_root = Path("./data//urdfs/acronym/collisions_tabletop")
    # sim.object_urdfs_g1b = [f for f in sim.urdf_root.iterdir() if f.suffix == ".urdf"]

    sim_selected.add_noise = sim.add_noise
    sim_selected.sideview = sim.sideview
    sim_selected.size = sim.size
    intrinsics = CameraIntrinsic(640, 480, 540.0, 540.0, 320.0, 240.0)
    sim_selected.camera = sim_selected.world.add_camera(intrinsics, 0.1, 2.0)

    mesh_pose_dict = collect_mesh_pose_dict(sim)
    for idc in indices:
        pose = Transform.from_matrix(mesh_pose_dict[idc][2])
        if idc == 0:
            mesh_path = mesh_pose_dict[idc][0].replace(".obj", ".urdf")
        else:
            mesh_path = find_urdf(mesh_pose_dict[idc][0].replace("_textured.obj", ".urdf"))
        sim_selected.world.load_urdf(mesh_path, pose, mesh_pose_dict[idc][1][0])
    return sim_selected


class ClutterRemovalSim(object):
    """
    A simple class to simulate objects on a table and execute grasps.
    Scenes: 'pile' or 'packed'.
    Models: 'vgn', 'giga_aff', 'giga', 'giga_hr', 'afford_scene_targ_pc' (with fusion_type='targo').
    """

    def __init__(
        self,
        scene,
        object_set,
        size=None,
        gui=False,
        seed=None,
        add_noise=False,
        sideview=False,
        save_dir=None,
        save_freq=8,
        test_root=None,
        is_acronym=False,
        egl_mode=False,
        sc_net=None,
    ):
        # Only allow 'pile' or 'packed' scenes now
        assert scene in ["pile", "packed"]

        self.urdf_root = Path("data/urdfs")
        self.ycb_root = Path("./data//maniskill_ycb/mani_skill2_ycb/collisions")
        self.g1b_root = Path("./data//g1b/collisions")
        self.egad_root = Path("./data//egad/collisions")
        if object_set == "mess_kitchen/train":
            self.gso_root = Path("./data/urdfs/mess_kitchen/train")
        elif object_set == "mess_kitchen/test":
            self.gso_root = Path("./data/urdfs/mess_kitchen/test")
        self.acroym_root = Path("data/urdfs/mess_kitchen/train")
        self.acroym_scales = load_json_or_default("data/acronym/acronym_scales.json", {})
        self.ycb_loader = YCBPathsLoader() if YCBPathsLoader.MODEL_JSON.exists() else None
        self.scene = scene
        self.object_set = object_set
        self.discover_objects()

        self.global_scaling = {
            "blocks": 1.67,
            "google": 0.7,
            "google_pile": 0.7,
            "google_packed": 0.7,
            "acroym": 0.7,
        }.get(object_set, 1.0)

        if is_acronym:
            self.global_scaling = 0.5
        self.gui = gui
        self.add_noise = add_noise
        self.sideview = sideview

        self.rng = np.random.RandomState(seed) if seed else np.random
        self.world = btsim.BtWorld(self.gui, save_dir, save_freq, egl_mode)
        self.gripper = Gripper(self.world)
        self.sc_net = sc_net  # Shape completion network
        if size:
            self.size = size
        else:
            self.size = 6 * self.gripper.finger_depth

        intrinsic = CameraIntrinsic(640, 480, 540.0, 540.0, 320.0, 240.0)
        self.camera = self.world.add_camera(intrinsic, 0.1, 2.0)

        # For optional occlusion-level bookkeeping
        self.occ_level_dict_path = None
        self.occ_level_dict = {}
        self.save_occ_level_dict = False
        if test_root is not None:
            self.occ_level_dict_path = Path(test_root) / 'test_set' / 'occlusion_level_dict.json'
            if not self.occ_level_dict_path.exists():
                self.occ_level_dict = {}
                self.save_occ_level_dict = True
            else:
                self.occ_level_dict = json.loads(self.occ_level_dict_path.read_text())
                self.save_occ_level_dict = False

    @property
    def num_objects(self):
        """
        Number of objects (excluding the table).
        """
        return max(0, self.world.p.getNumBodies() - 1)

    def discover_objects(self):
        """
        Collect all the URDF files in the specified folder.
        """
        def list_urdfs(root):
            root = Path(root)
            if not root.exists():
                return []
            return [f for f in root.iterdir() if f.suffix == ".urdf" or f.name.endswith(".urdf")]

        root = self.urdf_root / self.object_set
        self.object_urdfs = list_urdfs(root)
        self.object_urdfs_ycb = list_urdfs(self.ycb_root)
        # self.object_urdfs_g1b = [f for f in self.g1b_root.iterdir() if f.name.endswith(".urdf")]
        self.object_urdfs_g1b = list_urdfs(self.egad_root)
        self.object_urdfs_egad = list_urdfs(self.egad_root)
        self.object_urdfs_acroym = list_urdfs(self.acroym_root)
        # self.object_urdfs_gso = [f for f in self.gso_root.iterdir() if f.name.endswith(".urdf")]

    def load_clip_model(self):
        """
        Load CLIP model for feature extraction.
        """
        if not CLIP_AVAILABLE:
            return None, None, None
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        return model, preprocess, device

    def load_targo_category_mapping(self):
        """
        Load TARGO category mapping from files.
        """
        # Load class names with indices
        class_names_path = Path("data/targo_category/class_names.json")
        if not class_names_path.exists():
            print(f"Warning: {class_names_path} not found. Using default categories.")
            return {"others": 0}, {}
            
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
        
        # Load object to category mapping
        vgn_objects_path = Path("data/targo_category/vgn_objects_category.json")
        if not vgn_objects_path.exists():
            print(f"Warning: {vgn_objects_path} not found. Using empty mapping.")
            return class_names, {}
            
        with open(vgn_objects_path, 'r') as f:
            object_category_mapping = json.load(f)
        
        return class_names, object_category_mapping

    def extract_object_category_from_path(self, mesh_path, object_category_mapping):
        """
        Extract object category from mesh file path using TARGO mapping.
        """
        # Extract filename without extension
        filename = os.path.basename(mesh_path).replace("_textured.obj", "").replace(".obj", "")
        
        # Try to get category from TARGO mapping
        if filename in object_category_mapping:
            return object_category_mapping[filename]
        
        # Fallback: try to extract category from filename
        parts = filename.split("_")
        if len(parts) > 1:
            potential_category = parts[0].lower()
            return potential_category
        
        return "others"

    def extract_clip_features(self, text_prompt, model, device):
        """Extract CLIP features for a given text prompt."""
        if not CLIP_AVAILABLE or model is None:
            return np.zeros(512, dtype=np.float32)
            
        text_inputs = torch.cat([clip.tokenize(text_prompt)]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
            return text_features.cpu().numpy().flatten()

    def generate_clip_features_for_scene(self, scene_name, target_category, model, device):
        """
        Generate CLIP features for a scene based on target category.
        """
        # Extract CLIP features for target and occluders
        target_text_prompt = f"a {target_category} to grasp"
        occluder_text_prompt = "occluders"
        
        target_clip_features = self.extract_clip_features(target_text_prompt, model, device)
        occluder_clip_features = self.extract_clip_features(occluder_text_prompt, model, device)
        
        return target_clip_features, occluder_clip_features

    def load_clip_features_from_file(self, scene_name, clip_feat_dir):
        """
        Load CLIP features from pre-computed file.
        """
        clip_feat_path = Path(clip_feat_dir) / f"{scene_name}.npz"
        if clip_feat_path.exists():
            clip_data = np.load(clip_feat_path)
            target_clip_features = clip_data['targ_clip_features']
            scene_clip_features = clip_data['scene_clip_features_expanded']
            return target_clip_features, scene_clip_features
        return None, None

    def save_state(self):
        """
        Save the simulator state so it can be restored if needed.
        """
        self._snapshot_id = self.world.save_state()

    def restore_state(self):
        """
        Restore the simulator to the previously saved state.
        """
        self.world.restore_state(self._snapshot_id)

    def reset(self, object_count, target_id=None, is_ycb=False, is_g1b=False, is_egad=False, is_acronym=False, is_gso=False):
        """
        Reset the world, place the table, and generate a scene with objects.
        """
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

        # Generate either pile or packed scene
        if not is_gso:
            if self.scene == "pile":
                self.generate_pile_scene(object_count, table_height, target_id=target_id)
            elif self.scene == "packed":
                self.generate_packed_scene(object_count, table_height, target_id=target_id, is_ycb=is_ycb, is_g1b=is_g1b, is_egad=is_egad, is_acronym=is_acronym)
        elif is_gso:
            if self.scene == "pile":
                self.generate_gso_pile_scene(object_count, table_height, target_id=target_id)
            elif self.scene == "packed":
                self.generate_gso_packed_scene(object_count, table_height, target_id=target_id)

    def draw_workspace(self):
        """
        Draw lines marking the valid workspace region in the PyBullet GUI.
        """
        points = workspace_lines(self.size)
        color = [0.5, 0.5, 0.5]
        for i in range(0, len(points), 2):
            self.world.p.addUserDebugLine(
                lineFromXYZ=points[i], lineToXYZ=points[i + 1], lineColorRGB=color
            )

    def place_table(self, height):
        """
        Load a simple plane for the table and define the valid volume.
        """
        urdf = self.urdf_root / "setup" / "plane.urdf"
        pose = Transform(Rotation.identity(), [0.15, 0.15, height])
        self.world.load_urdf(urdf, pose, scale=0.6)

        lx, ux = 0.02, self.size - 0.02
        ly, uy = 0.02, self.size - 0.02
        lz, uz = height + 0.005, self.size
        self.lower = np.r_[lx, ly, lz]
        self.upper = np.r_[ux, uy, uz]

    def generate_gso_pile_scene(self, object_count, table_height, target_id=None):
        """
        Generate a 'pile' scene by dropping objects within a box, then removing the box.
        """
        self.object_urdfs = self.object_urdfs_gso
        urdf = self.urdf_root / "setup" / "box.urdf"
        pose = Transform(Rotation.identity(), np.r_[0.02, 0.02, table_height])
        box = self.world.load_urdf(urdf, pose, scale=1.3)

        urdfs = self.rng.choice(self.object_urdfs, size=object_count)
        if target_id is not None:
            random_index = np.random.randint(0, len(urdfs))
            urdfs[random_index] = self.object_urdfs[target_id]
        for urdf_file in urdfs:
            rotation = Rotation.random(random_state=self.rng)
            xy = self.rng.uniform(1.0 / 3.0 * self.size, 2.0 / 3.0 * self.size, 2)
            pose = Transform(rotation, np.r_[xy, table_height + 0.2])
            scale = self.rng.uniform(0.18, 0.22)
            scale *= 0.3
            self.world.load_urdf(urdf_file, pose, scale=self.global_scaling * scale)
            self.wait_for_objects_to_rest(timeout=1.0)

        self.world.remove_body(box)
        self.remove_and_wait()

    def generate_gso_packed_scene(self, object_count, table_height, target_id=None):
        """
        Generate a 'packed' scene by placing objects next to each other without overlap.
        
        For Acronym objects, this method follows a similar approach to TableScene.arrange() in the
        acronym_tools library, placing objects in stable poses without collisions.
        """
        attempts = 0
        max_attempts = 12

        while self.num_objects < object_count and attempts < max_attempts:
            self.save_state()
            object_urdfs = self.object_urdfs_gso
   
            # if self.num_objects == 0 and target_id is not None:
            #     urdf_file = object_urdfs[target_id]
            # else:
            urdf_file = self.rng.choice(object_urdfs)

            x = self.rng.uniform(0.08, 0.22)
            y = self.rng.uniform(0.08, 0.22)
            z = 1.0
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
            pose = Transform(rotation, np.r_[x, y, z])
            
            scale = self.rng.uniform(0.7, 0.9)
            scale *= 0.1
            body = self.world.load_urdf(
                urdf_file, pose, scale=self.global_scaling * scale
            )
            
            lower, upper = self.world.p.getAABB(body.uid)
            z = table_height + 0.5 * (upper[2] - lower[2]) + 0.002
            body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))
            self.world.step()

            # If it collides or overlaps, remove it
            if self.world.get_contacts(body):
                self.world.remove_body(body)
                self.restore_state()
            else:
                self.remove_and_wait()
            attempts += 1
            
    def generate_packed_scene(self, object_count, table_height, target_id=None, is_ycb=False, is_g1b=False, is_egad=False, is_acronym=False):
        """
        Generate a 'packed' scene by placing objects next to each other without overlap.
        
        For Acronym objects, this method follows a similar approach to TableScene.arrange() in the
        acronym_tools library, placing objects in stable poses without collisions.
        """
        attempts = 0
        max_attempts = 12

        while self.num_objects < object_count and attempts < max_attempts:
            self.save_state()
            if not is_ycb and not is_g1b and not is_egad and not is_acronym:
                object_urdfs = self.object_urdfs
            elif is_ycb:
                object_urdfs = self.object_urdfs_ycb
            elif is_g1b:
                object_urdfs = self.object_urdfs_g1b
            elif is_egad:
                object_urdfs = self.object_urdfs_egad
            elif is_acronym:
                object_urdfs = self.object_urdfs_acroym
                
            if self.num_objects == 0 and target_id is not None:
                urdf_file = object_urdfs[target_id]
            else:
                urdf_file = self.rng.choice(object_urdfs)

            # For Acronym objects, we try to find stable poses
            if is_acronym:
                x = self.rng.uniform(0.08, 0.22)
                y = self.rng.uniform(0.08, 0.22)
                z = table_height + 0.2
                # z = 1.0
                angle = self.rng.uniform(0.0, 2.0 * np.pi)
                rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
                pose = Transform(rotation, np.r_[x, y, z])
                
                # Get scale from acronym scales if available
                urdf_id = Path(urdf_file).name
                scale = self.acroym_scales.get(urdf_id, 1.0)
                
                body = self.world.load_urdf(
                    urdf_file, pose, scale=self.global_scaling * scale
                )
                
                lower, upper = self.world.p.getAABB(body.uid)
                z = table_height + 0.5 * (upper[2] - lower[2]) + 0.002
                body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))
                self.world.step()
            elif is_ycb:
                # Original method for non-Acronym objects
                x = self.rng.uniform(0.08, 0.22)
                y = self.rng.uniform(0.08, 0.22)
                z = 1.0
                angle = self.rng.uniform(0.0, 2.0 * np.pi)
                rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
                pose = Transform(rotation, np.r_[x, y, z])
                
                # scale = self.rng.uniform(0.7, 0.9)
                scale = 1.0
                body = self.world.load_urdf(
                    urdf_file, pose, scale=self.global_scaling * scale
                )
                
                lower, upper = self.world.p.getAABB(body.uid)
                z = table_height + 0.5 * (upper[2] - lower[2]) + 0.002
                body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))
                self.world.step()
            else:
                # Original method for non-Acronym objects
                x = self.rng.uniform(0.08, 0.22)
                y = self.rng.uniform(0.08, 0.22)
                z = 1.0
                angle = self.rng.uniform(0.0, 2.0 * np.pi)
                rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
                pose = Transform(rotation, np.r_[x, y, z])
                
                scale = self.rng.uniform(0.7, 0.9)
                body = self.world.load_urdf(
                    urdf_file, pose, scale=self.global_scaling * scale
                )
                
                lower, upper = self.world.p.getAABB(body.uid)
                z = table_height + 0.5 * (upper[2] - lower[2]) + 0.002
                body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))
                self.world.step()

            # If it collides or overlaps, remove it
            if self.world.get_contacts(body):
                self.world.remove_body(body)
                self.restore_state()
            else:
                self.remove_and_wait()
            attempts += 1
    
    def acquire_single_tsdf_target_grid_ptv3_clip(
        self,
        curr_scene_path,
        target_id,
        resolution=40,
        model_type=None,
        curr_mesh_pose_list=None,
        hunyuan3D_ptv3=False,
        hunyuan3D_path=None,
        target_category=None,
        clip_feat_dir=None,
        use_precomputed_clip=True,
    ):
        """
        Acquire TSDF and point clouds for 'ptv3_clip' model.
        Returns:
            tsdf: TSDFVolume object
            timing: integration time
            scene_pc_full: normalized + filtered full scene point cloud
            complete_targ_pc: complete target point cloud
            complete_targ_tsdf: complete target TSDF grid
            grid_targ: partial target TSDF grid
            occ_level: occlusion level
            target_clip_features: CLIP features for target (512,)
            scene_clip_features: CLIP features for scene (512,)
        """
        # Setup TSDF volume
        vis_dict = {}
        assert model_type == "ptv3_clip", f"PTV3ClipImplicit only supports ptv3_clip model type, got {model_type}"
        tsdf = TSDFVolume(self.size, resolution)

        # Camera pose configuration
        half_size = self.size / 2
        origin = Transform(Rotation.identity(), np.r_[half_size, half_size, 0])
        theta, phi = np.pi / 6.0, 0
        r = 2.0 * self.size
        extrinsic = camera_on_sphere(origin, r, theta, phi)

        # Load data from .npz
        data = np.load(curr_scene_path, allow_pickle=True)
        depth_img = data["depth_imgs"]
        vis_dict["depth_img"] = depth_img
        seg_img = data["segmentation_map"]
        tgt_mask = data["mask_targ"]
        scene_mask = data["mask_scene"]

        # Validate masks
        assert np.all(scene_mask == (seg_img > 0))
        assert np.all(tgt_mask == (seg_img == target_id))

        # Compute occlusion level
        occ_level = 0.0
        if curr_mesh_pose_list is not None:
            if not self.save_occ_level_dict:
                occ_level = self.occ_level_dict[curr_mesh_pose_list]
            else:
                sim_single = sim_select_scene(self, [0, target_id])
                _, seg_img_single = sim_single.camera.render_with_seg(extrinsic)[1:3]
                occ_level = 1 - np.sum(seg_img == target_id) / np.sum(seg_img_single == 1)
                self.occ_level_dict[curr_mesh_pose_list] = occ_level
        if occ_level > 0.9:
            print("High occlusion level")

        # TSDF integration
        tic = time.time()
        depth_input = (depth_img * scene_mask)[0].astype(np.float32)
        tsdf.integrate(depth_input, self.camera.intrinsic, extrinsic)

        # tsdf.integrate((depth_img * scene_mask)[0], self.camera.intrinsic, extrinsic)
        timing = time.time() - tic

        # Load and process point clouds
        ## load plane.npy
        plane = np.load("./setup/plane_sampled.npy")
        plane = plane.astype(np.float32)
        pc_scene_no_targ = np.concatenate([data["pc_scene_no_targ"], plane], axis=0)
        # scene_pc_scene_no_targ = pc_scene_no_targ / 0.3 - 0.5
        from utils_giga import filter_and_pad_point_clouds
        pc_scene_no_targ = filter_and_pad_point_clouds(
            torch.from_numpy(pc_scene_no_targ).unsqueeze(0).float(),
            lower_bound=torch.tensor([0, 0, 0]),
             upper_bound=torch.tensor([0.3, 0.3, 0.3]),
        ).squeeze(0).numpy()
        ## concate with plane
        pc_scene_no_targ = np.concatenate([pc_scene_no_targ, plane], axis=0)
        pc_scene_no_targ = pc_scene_no_targ / 0.3 - 0.5

        # Load complete target data based on hunyuan3D_ptv3 flag
        iou_value = 1.0
        cd_value = 0.0
        if hunyuan3D_ptv3 and hunyuan3D_path is not None:
            # Load from hunyuan3D path
            scene_name = curr_mesh_pose_list if curr_mesh_pose_list is not None else "unknown"
            hunyuan3D_scene_path = os.path.join(hunyuan3D_path, scene_name, "scenes")
            # hunyuan3D_crop_path = os.path.join(hunyuan3D_path, scene_name, "crops")
            scene_rgba_path = os.path.join(hunyuan3D_path, scene_name, "crops", "scene_rgba.png")
            vis_dict["scene_rgba_path"] = scene_rgba_path

            eval_recon_file = os.path.join(hunyuan3D_path, scene_name, "evaluation", "meta_evaluation.txt")
            # Read CD (Chamfer Distance) and IoU from meta_evaluation.txt
            # cd_value, iou_value = None, None
            if not os.path.exists(eval_recon_file):
                cd_value = -1
                iou_value = -1
            elif os.path.exists(eval_recon_file):
                with open(eval_recon_file, 'r') as f:
                    for line in f:
                        if 'Chamfer Distance v7_gt' in line:
                            cd_value = float(line.strip().split(':')[-1])
                        if 'Chamfer_watertight' in line:
                            cd_value = float(line.strip().split(':')[-1])
                        if 'IoU v7_gt' in line:
                            iou_value = float(line.strip().split(':')[-1])
                        if 'IoU_watertight' in line:
                            iou_value = float(line.strip().split(':')[-1])
                        # elif 'IoU v7_gt' in line:
                        #     iou_value = float(line.strip().split(':')[-1])
            # with open(eval_recon_file, "r") as f:
            
            # Try to load hunyuan3D reconstructed data
            pc_file_path = os.path.join(hunyuan3D_scene_path, "complete_targ_hunyuan_pc.npy")
            tsdf_file_path = os.path.join(hunyuan3D_scene_path, "complete_targ_hunyuan_tsdf.npy")
            
            if os.path.exists(pc_file_path):
                complete_targ_pc = np.load(pc_file_path).astype(np.float32)
                complete_targ_tsdf = np.load(tsdf_file_path).astype(np.float32)
            else:
                if not os.path.exists(hunyuan3D_scene_path):
                    os.makedirs(hunyuan3D_scene_path, exist_ok=True)
                reconstructed_mesh = o3d.io.read_triangle_mesh(os.path.join(hunyuan3D_path, scene_name, "reconstruction", "targ_obj_hy3dgen_align.ply"))
                reconstructed_mesh.compute_vertex_normals()
                pcd = reconstructed_mesh.sample_points_poisson_disk(number_of_points=512, init_factor=5)
                complete_targ_pc = np.asarray(pcd.points)
                complete_targ_tsdf = point_cloud_to_tsdf(complete_targ_pc)
                ## save to pc_file_path and tsdf_file_path
                np.save(pc_file_path, complete_targ_pc)
                np.save(tsdf_file_path, complete_targ_tsdf)
       

        # Process complete target point cloud
        complete_targ_pc = filter_and_pad_point_clouds(
            torch.from_numpy(complete_targ_pc).unsqueeze(0).float(),
            lower_bound=torch.tensor([0, 0, 0]),
            upper_bound=torch.tensor([0.3, 0.3, 0.3]),
        ).squeeze(0).numpy()
        complete_targ_pc = complete_targ_pc / 0.3 - 0.5
        
        grid_targ = data["grid_targ"]

        # Load or generate CLIP features
        target_clip_features = None
        scene_clip_features = None
        
        if curr_mesh_pose_list is not None:
            scene_name = curr_mesh_pose_list
            
            if use_precomputed_clip and clip_feat_dir is not None:
                # Try to load pre-computed CLIP features
                target_clip_features, scene_clip_features = self.load_clip_features_from_file(scene_name, clip_feat_dir)
                
            if target_clip_features is None:
                # Generate CLIP features on-the-fly
                # if target_category is not None:
                assert target_category is not None, "Target category is not provided"
                    # Use provided target category
                model, preprocess, device = self.load_clip_model()
                target_clip_features, scene_clip_features = self.generate_clip_features_for_scene(
                    scene_name, target_category, model, device
                )
                # else:
                #     # Try to determine target category from mesh_pose_dict
                #     try:
                #         # Load mesh_pose_dict to determine target category
                #         mesh_pose_path = curr_scene_path.parent.parent / "mesh_pose_dict" / f"{scene_name}.npz"
                #         if mesh_pose_path.exists():
                #             mesh_pose_data = np.load(mesh_pose_path, allow_pickle=True)
                #             mesh_pose_dict = mesh_pose_data["pc"].item()
                            
                #             # Load category mapping
                #             class_names, object_category_mapping = self.load_targo_category_mapping()
                            
                #             # Extract target category
                #             if target_id in mesh_pose_dict:
                #                 mesh_path = mesh_pose_dict[target_id][0]
                #                 target_category = self.extract_object_category_from_path(mesh_path, object_category_mapping)
                                
                #                 # Generate CLIP features
                #                 model, preprocess, device = self.load_clip_model()
                #                 target_clip_features, scene_clip_features = self.generate_clip_features_for_scene(
                #                     scene_name, target_category, model, device
                #                 )
                    # except Exception as e:
                    #     print(f"Warning: Could not determine target category for {scene_name}: {e}")
                    #     # Use default features
                    #     target_clip_features = np.zeros(512, dtype=np.float32)
                    #     scene_clip_features = np.zeros(512, dtype=np.float32)
        
        # If still no CLIP features, use default
        if target_clip_features is None:
            target_clip_features = np.zeros(512, dtype=np.float32)
            scene_clip_features = np.zeros(512, dtype=np.float32)

        return tsdf, timing, pc_scene_no_targ, complete_targ_pc, complete_targ_tsdf, grid_targ, occ_level, iou_value, cd_value, vis_dict, target_clip_features, scene_clip_features
    
    def acquire_single_tsdf_target_grid_ptv3_scene_gt(
        self,
        curr_scene_path,
        target_id,
        resolution=40,
        model_type=None,
        curr_mesh_pose_list=None,
        hunyuan3D_ptv3=False,
        hunyuan3D_path=None,
        target_mesh_gt=None,
        target_category=None,
        clip_feat_dir=None,
        use_precomputed_clip=True,
    ):
        """
        Acquire TSDF and point clouds for 'ptv3_clip' model.
        Returns:
            tsdf: TSDFVolume object
            timing: integration time
            scene_pc_full: normalized + filtered full scene point cloud
            complete_targ_pc: complete target point cloud
            complete_targ_tsdf: complete target TSDF grid
            grid_targ: partial target TSDF grid
            occ_level: occlusion level
            target_clip_features: CLIP features for target (512,)
            scene_clip_features: CLIP features for scene (512,)
        """
        # Setup TSDF volume
        vis_dict = {}
        assert model_type == "ptv3_scene_gt", f"PTV3ClipImplicit only supports ptv3_clip_gt model type, got {model_type}"
        tsdf = TSDFVolume(self.size, resolution)

        # Camera pose configuration
        half_size = self.size / 2
        origin = Transform(Rotation.identity(), np.r_[half_size, half_size, 0])
        theta, phi = np.pi / 6.0, 0
        r = 2.0 * self.size
        extrinsic = camera_on_sphere(origin, r, theta, phi)

        # Load data from .npz
        data = np.load(curr_scene_path, allow_pickle=True)
        depth_img = data["depth_imgs"]
        vis_dict["depth_img"] = depth_img
        seg_img = data["segmentation_map"]
        tgt_mask = data["mask_targ"]
        scene_mask = data["mask_scene"]

        # Validate masks
        assert np.all(scene_mask == (seg_img > 0))
        assert np.all(tgt_mask == (seg_img == target_id))

        # Compute occlusion level
        occ_level = 0.0
        if curr_mesh_pose_list is not None:
            if not self.save_occ_level_dict:
                occ_level = self.occ_level_dict[curr_mesh_pose_list]
            else:
                sim_single = sim_select_scene(self, [0, target_id])
                _, seg_img_single = sim_single.camera.render_with_seg(extrinsic)[1:3]
                occ_level = 1 - np.sum(seg_img == target_id) / np.sum(seg_img_single == 1)
                self.occ_level_dict[curr_mesh_pose_list] = occ_level
        if occ_level > 0.9:
            print("High occlusion level")

        # TSDF integration
        tic = time.time()
        depth_input = (depth_img * scene_mask)[0].astype(np.float32)
        tsdf.integrate(depth_input, self.camera.intrinsic, extrinsic)

        timing = time.time() - tic

        ## load plane.npy
        plane = np.load("./setup/plane_sampled.npy")
        plane = plane.astype(np.float32)
        pc_scene_no_targ = np.concatenate([data["pc_scene_no_targ"], plane], axis=0)
        # scene_pc_scene_no_targ = pc_scene_no_targ / 0.3 - 0.5
        from utils_giga import filter_and_pad_point_clouds
        pc_scene_no_targ = filter_and_pad_point_clouds(
            torch.from_numpy(pc_scene_no_targ).unsqueeze(0).float(),
            lower_bound=torch.tensor([0, 0, 0]),
             upper_bound=torch.tensor([0.3, 0.3, 0.3]),
        ).squeeze(0).numpy()
        ## concate with plane
        pc_scene_no_targ = np.concatenate([pc_scene_no_targ, plane], axis=0)
        pc_scene_no_targ = pc_scene_no_targ / 0.3 - 0.5

        # Load complete target data based on hunyuan3D_ptv3 flag
        iou_value = 1.0
        cd_value = 0.0
        # complete_targ_pc = data["complete_target_pc"]
        ## generate complete_target_tsdf
        # try:
        # complete_target_mesh_vertices = data['complete_target_mesh_vertices']
        # complete_target_mesh_faces = data['complete_target_mesh_faces']
        # complete_target_mesh = o3d.geometry.TriangleMesh()
        # complete_target_mesh.vertices = o3d.utility.Vector3dVector(complete_target_mesh_vertices)
        # complete_target_mesh.triangles = o3d.utility.Vector3iVector(complete_target_mesh_faces)
        complete_target_mesh = o3d.geometry.TriangleMesh()
        complete_target_mesh.vertices = o3d.utility.Vector3dVector(target_mesh_gt.vertices)
        complete_target_mesh.triangles = o3d.utility.Vector3iVector(target_mesh_gt.faces)
        ## sample
        complete_targ_pc = complete_target_mesh.sample_points_poisson_disk(number_of_points=512, init_factor=5)
        complete_targ_pc = np.asarray(complete_targ_pc.points)
        # complete_targ_pc = 

        try:
            complete_target_mesh.compute_vertex_normals()
            complete_targ_tsdf = mesh_to_tsdf(complete_target_mesh)
        except:
            complete_targ_tsdf = point_cloud_to_tsdf(complete_targ_pc)

        complete_targ_tsdf = complete_targ_tsdf
        # Process complete target point cloud
        complete_targ_pc = filter_and_pad_point_clouds(
            torch.from_numpy(complete_targ_pc).unsqueeze(0).float(),
            lower_bound=torch.tensor([0, 0, 0]),
            upper_bound=torch.tensor([0.3, 0.3, 0.3]),
        ).squeeze(0).numpy()
        complete_targ_pc = complete_targ_pc / 0.3 - 0.5
        
        grid_targ = data["grid_targ"]

        return tsdf, timing, pc_scene_no_targ, complete_targ_pc, complete_targ_tsdf, grid_targ, occ_level, iou_value, cd_value, vis_dict
        

    def acquire_single_tsdf_target_grid_ptv3_clip_gt(
        self,
        curr_scene_path,
        target_id,
        resolution=40,
        model_type=None,
        curr_mesh_pose_list=None,
        hunyuan3D_ptv3=False,
        hunyuan3D_path=None,
        target_mesh_gt=None,
        target_category=None,
        clip_feat_dir=None,
        use_precomputed_clip=True,
    ):
        """
        Acquire TSDF and point clouds for 'ptv3_clip' model.
        Returns:
            tsdf: TSDFVolume object
            timing: integration time
            scene_pc_full: normalized + filtered full scene point cloud
            complete_targ_pc: complete target point cloud
            complete_targ_tsdf: complete target TSDF grid
            grid_targ: partial target TSDF grid
            occ_level: occlusion level
            target_clip_features: CLIP features for target (512,)
            scene_clip_features: CLIP features for scene (512,)
        """
        # Setup TSDF volume
        vis_dict = {}
        assert model_type == "ptv3_clip_gt", f"PTV3ClipImplicit only supports ptv3_clip_gt model type, got {model_type}"
        tsdf = TSDFVolume(self.size, resolution)

        # Camera pose configuration
        half_size = self.size / 2
        origin = Transform(Rotation.identity(), np.r_[half_size, half_size, 0])
        theta, phi = np.pi / 6.0, 0
        r = 2.0 * self.size
        extrinsic = camera_on_sphere(origin, r, theta, phi)

        # Load data from .npz
        data = np.load(curr_scene_path, allow_pickle=True)
        depth_img = data["depth_imgs"]
        vis_dict["depth_img"] = depth_img
        seg_img = data["segmentation_map"]
        tgt_mask = data["mask_targ"]
        scene_mask = data["mask_scene"]

        # Validate masks
        assert np.all(scene_mask == (seg_img > 0))
        assert np.all(tgt_mask == (seg_img == target_id))

        # Compute occlusion level
        occ_level = 0.0
        if curr_mesh_pose_list is not None:
            if not self.save_occ_level_dict:
                occ_level = self.occ_level_dict[curr_mesh_pose_list]
            else:
                sim_single = sim_select_scene(self, [0, target_id])
                _, seg_img_single = sim_single.camera.render_with_seg(extrinsic)[1:3]
                occ_level = 1 - np.sum(seg_img == target_id) / np.sum(seg_img_single == 1)
                self.occ_level_dict[curr_mesh_pose_list] = occ_level
        if occ_level > 0.9:
            print("High occlusion level")

        # TSDF integration
        tic = time.time()
        depth_input = (depth_img * scene_mask)[0].astype(np.float32)
        tsdf.integrate(depth_input, self.camera.intrinsic, extrinsic)

        timing = time.time() - tic

        ## load plane.npy
        plane = np.load("./setup/plane_sampled.npy")
        plane = plane.astype(np.float32)
        pc_scene_no_targ = np.concatenate([data["pc_scene_no_targ"], plane], axis=0)
        # scene_pc_scene_no_targ = pc_scene_no_targ / 0.3 - 0.5
        from utils_giga import filter_and_pad_point_clouds
        pc_scene_no_targ = filter_and_pad_point_clouds(
            torch.from_numpy(pc_scene_no_targ).unsqueeze(0).float(),
            lower_bound=torch.tensor([0, 0, 0]),
             upper_bound=torch.tensor([0.3, 0.3, 0.3]),
        ).squeeze(0).numpy()
        ## concate with plane
        pc_scene_no_targ = np.concatenate([pc_scene_no_targ, plane], axis=0)
        pc_scene_no_targ = pc_scene_no_targ / 0.3 - 0.5

        # Load complete target data based on hunyuan3D_ptv3 flag
        iou_value = 1.0
        cd_value = 0.0
        # complete_targ_pc = data["complete_target_pc"]
        ## generate complete_target_tsdf
        # try:
        # complete_target_mesh_vertices = data['complete_target_mesh_vertices']
        # complete_target_mesh_faces = data['complete_target_mesh_faces']
        # complete_target_mesh = o3d.geometry.TriangleMesh()
        # complete_target_mesh.vertices = o3d.utility.Vector3dVector(complete_target_mesh_vertices)
        # complete_target_mesh.triangles = o3d.utility.Vector3iVector(complete_target_mesh_faces)
        complete_target_mesh = o3d.geometry.TriangleMesh()
        complete_target_mesh.vertices = o3d.utility.Vector3dVector(target_mesh_gt.vertices)
        complete_target_mesh.triangles = o3d.utility.Vector3iVector(target_mesh_gt.faces)
        ## sample
        complete_targ_pc = complete_target_mesh.sample_points_poisson_disk(number_of_points=512, init_factor=5)
        complete_targ_pc = np.asarray(complete_targ_pc.points)
        # complete_targ_pc = 

        try:
            complete_target_mesh.compute_vertex_normals()
            complete_targ_tsdf = mesh_to_tsdf(complete_target_mesh)
        except:
            complete_targ_tsdf = point_cloud_to_tsdf(complete_targ_pc)

        complete_targ_tsdf = complete_targ_tsdf
        # Process complete target point cloud
        complete_targ_pc = filter_and_pad_point_clouds(
            torch.from_numpy(complete_targ_pc).unsqueeze(0).float(),
            lower_bound=torch.tensor([0, 0, 0]),
            upper_bound=torch.tensor([0.3, 0.3, 0.3]),
        ).squeeze(0).numpy()
        complete_targ_pc = complete_targ_pc / 0.3 - 0.5
        
        grid_targ = data["grid_targ"]

        # Load or generate CLIP features
        target_clip_features = None
        scene_clip_features = None
        
        if curr_mesh_pose_list is not None:
            scene_name = curr_mesh_pose_list
            
            if use_precomputed_clip and clip_feat_dir is not None:
                # Try to load pre-computed CLIP features
                target_clip_features, scene_clip_features = self.load_clip_features_from_file(scene_name, clip_feat_dir)
                
            if target_clip_features is None:
                # Generate CLIP features on-the-fly
                # if target_category is not None:
                assert target_category is not None, "Target category is not provided"
                    # Use provided target category
                model, preprocess, device = self.load_clip_model()
                target_clip_features, scene_clip_features = self.generate_clip_features_for_scene(
                    scene_name, target_category, model, device
                )
        
        # If still no CLIP features, use default
        if target_clip_features is None:
            target_clip_features = np.zeros(512, dtype=np.float32)
            scene_clip_features = np.zeros(512, dtype=np.float32)

        return tsdf, timing, pc_scene_no_targ, complete_targ_pc, complete_targ_tsdf, grid_targ, occ_level, iou_value, cd_value, vis_dict, target_clip_features, scene_clip_features        
    def acquire_single_tsdf_target_grid_ptv3_scene(
        self,
        curr_scene_path,
        target_id,
        resolution=40,
        model_type=None,
        curr_mesh_pose_list=None,
        hunyuan3D_ptv3=False,
        hunyuan3D_path=None,
    ):
        """
        Acquire TSDF and point clouds for 'ptv3_scene' model.
        Returns:
            tsdf: TSDFVolume object
            timing: integration time
            scene_pc_full: normalized + filtered full scene point cloud
            complete_targ_pc: complete target point cloud
            complete_targ_tsdf: complete target TSDF grid
            grid_targ: partial target TSDF grid
            occ_level: occlusion level
        """
        # Setup TSDF volume
        vis_dict = {}
        assert model_type == "ptv3_scene", f"PTV3SceneImplicit only supports ptv3_scene model type, got {model_type}"
        tsdf = TSDFVolume(self.size, resolution)

        # Camera pose configuration
        half_size = self.size / 2
        origin = Transform(Rotation.identity(), np.r_[half_size, half_size, 0])
        theta, phi = np.pi / 6.0, 0
        r = 2.0 * self.size
        extrinsic = camera_on_sphere(origin, r, theta, phi)

        # Load data from .npz
        data = np.load(curr_scene_path, allow_pickle=True)
        depth_img = data["depth_imgs"]
        vis_dict["depth_img"] = depth_img
        seg_img = data["segmentation_map"]
        tgt_mask = data["mask_targ"]
        scene_mask = data["mask_scene"]

        # Validate masks
        assert np.all(scene_mask == (seg_img > 0))
        assert np.all(tgt_mask == (seg_img == target_id))

        # Compute occlusion level
        occ_level = 0.0
        if curr_mesh_pose_list is not None:
            if not self.save_occ_level_dict:
                occ_level = self.occ_level_dict[curr_mesh_pose_list]
            else:
                sim_single = sim_select_scene(self, [0, target_id])
                _, seg_img_single = sim_single.camera.render_with_seg(extrinsic)[1:3]
                occ_level = 1 - np.sum(seg_img == target_id) / np.sum(seg_img_single == 1)
                self.occ_level_dict[curr_mesh_pose_list] = occ_level
        if occ_level > 0.9:
            print("High occlusion level")

        # TSDF integration
        tic = time.time()
        depth_input = (depth_img * scene_mask)[0].astype(np.float32)
        tsdf.integrate(depth_input, self.camera.intrinsic, extrinsic)

        # tsdf.integrate((depth_img * scene_mask)[0], self.camera.intrinsic, extrinsic)
        timing = time.time() - tic

        # Load and process point clouds
        ## load plane.npy
        plane = np.load("./setup/plane_sampled.npy")
        plane = plane.astype(np.float32)
        pc_scene_no_targ = np.concatenate([data["pc_scene_no_targ"], plane], axis=0)
        # scene_pc_scene_no_targ = pc_scene_no_targ / 0.3 - 0.5
        from utils_giga import filter_and_pad_point_clouds
        pc_scene_no_targ = filter_and_pad_point_clouds(
            torch.from_numpy(pc_scene_no_targ).unsqueeze(0).float(),
            lower_bound=torch.tensor([0, 0, 0]),
             upper_bound=torch.tensor([0.3, 0.3, 0.3]),
        ).squeeze(0).numpy()
        ## concate with plane
        pc_scene_no_targ = np.concatenate([pc_scene_no_targ, plane], axis=0)
        pc_scene_no_targ = pc_scene_no_targ / 0.3 - 0.5

        # Load complete target data based on hunyuan3D_ptv3 flag
        iou_value = 1.0
        cd_value = 0.0
        if hunyuan3D_ptv3 and hunyuan3D_path is not None:
            # Load from hunyuan3D path
            scene_name = curr_mesh_pose_list if curr_mesh_pose_list is not None else "unknown"
            hunyuan3D_scene_path = os.path.join(hunyuan3D_path, scene_name, "scenes")
            # hunyuan3D_crop_path = os.path.join(hunyuan3D_path, scene_name, "crops")
            scene_rgba_path = os.path.join(hunyuan3D_path, scene_name, "crops", "scene_rgba.png")
            vis_dict["scene_rgba_path"] = scene_rgba_path

            eval_recon_file = os.path.join(hunyuan3D_path, scene_name, "evaluation", "meta_evaluation.txt")
            # Read CD (Chamfer Distance) and IoU from meta_evaluation.txt
            # cd_value, iou_value = None, None
            if not os.path.exists(eval_recon_file):
                cd_value = -1
                iou_value = -1
            elif os.path.exists(eval_recon_file):
                with open(eval_recon_file, 'r') as f:
                    for line in f:
                        if 'Chamfer Distance v7_gt' in line:
                            cd_value = float(line.strip().split(':')[-1])
                        if 'Chamfer_watertight' in line:
                            cd_value = float(line.strip().split(':')[-1])
                        if 'IoU v7_gt' in line:
                            iou_value = float(line.strip().split(':')[-1])
                        if 'IoU_watertight' in line:
                            iou_value = float(line.strip().split(':')[-1])
                        # elif 'IoU v7_gt' in line:
                        #     iou_value = float(line.strip().split(':')[-1])
            # with open(eval_recon_file, "r") as f:
            
            # Try to load hunyuan3D reconstructed data
            pc_file_path = os.path.join(hunyuan3D_scene_path, "complete_targ_hunyuan_pc.npy")
            tsdf_file_path = os.path.join(hunyuan3D_scene_path, "complete_targ_hunyuan_tsdf.npy")
            
            if os.path.exists(pc_file_path):
                complete_targ_pc = np.load(pc_file_path).astype(np.float32)
                complete_targ_tsdf = np.load(tsdf_file_path).astype(np.float32)
            else:
                if not os.path.exists(hunyuan3D_scene_path):
                    os.makedirs(hunyuan3D_scene_path, exist_ok=True)
                reconstructed_mesh = o3d.io.read_triangle_mesh(os.path.join(hunyuan3D_path, scene_name, "reconstruction", "targ_obj_hy3dgen_align.ply"))
                reconstructed_mesh.compute_vertex_normals()
                pcd = reconstructed_mesh.sample_points_poisson_disk(number_of_points=512, init_factor=5)
                complete_targ_pc = np.asarray(pcd.points)
                complete_targ_tsdf = point_cloud_to_tsdf(complete_targ_pc)
                ## save to pc_file_path and tsdf_file_path
                np.save(pc_file_path, complete_targ_pc)
                np.save(tsdf_file_path, complete_targ_tsdf)
       

        # Process complete target point cloud
        complete_targ_pc = filter_and_pad_point_clouds(
            torch.from_numpy(complete_targ_pc).unsqueeze(0).float(),
            lower_bound=torch.tensor([0, 0, 0]),
            upper_bound=torch.tensor([0.3, 0.3, 0.3]),
        ).squeeze(0).numpy()
        complete_targ_pc = complete_targ_pc / 0.3 - 0.5
        
        grid_targ = data["grid_targ"]

        return tsdf, timing, pc_scene_no_targ, complete_targ_pc, complete_targ_tsdf, grid_targ, occ_level, iou_value, cd_value, vis_dict


    def acquire_single_tsdf_target_grid(
        self,
        curr_scene_path=None,
        target_id=None,
        resolution=40,
        is_vgn=False,
        model="vgn",
        curr_mesh_pose_list=None,
    ):
        """
        Same as acquire_single_tsdf_target_grid_train, but possibly for inference or other usage.
        """
        if model == "giga_hr":
            resolution = 60

        tsdf = TSDFVolume(self.size, resolution)
        tgt_mask_tsdf = TSDFVolume(self.size, resolution)
        plane = np.load("./setup/plane_sampled.npy")
        plane = plane.astype(np.float32)

        half_size = self.size / 2
        origin_yz = np.r_[half_size, half_size]

        if self.sideview:
            origin = Transform(Rotation.identity(), np.r_[origin_yz, self.size / 3])
            theta, phi = np.pi / 3.0, -np.pi / 2.0
        else:
            origin = Transform(Rotation.identity(), np.r_[origin_yz, 0])
            theta, phi = np.pi / 6.0, 0

        r = 2.0 * self.size
        extrinsic = camera_on_sphere(origin, r, theta, phi)

        _, _, seg_img = self.camera.render_with_seg(extrinsic)

        depth_img = np.load(curr_scene_path)["depth_imgs"]
        tgt_mask = np.load(curr_scene_path)["mask_targ"]
        scene_mask = np.load(curr_scene_path)["mask_scene"]

        seg_img = np.load(curr_scene_path)["segmentation_map"]

        assert np.all(scene_mask == (seg_img > 0))
        assert np.all(tgt_mask == (seg_img == target_id))

        occ_level = 0.0
        if curr_mesh_pose_list is not None:
            if not self.save_occ_level_dict:
                occ_level = self.occ_level_dict[curr_mesh_pose_list]
            else:
                # if model == "targo" :
                if is_vgn:
                    sim_single = sim_select_vgn_scene(self, [0, target_id])
                else:
                    sim_single = sim_select_scene(self, [0, target_id])
                _, seg_img_single = sim_single.camera.render_with_seg(extrinsic)[1:3]
                occ_level = 1 - np.sum(seg_img == target_id) / np.sum(seg_img_single == 1)
                self.occ_level_dict[curr_mesh_pose_list] = occ_level

        if occ_level > 0.9:
            print("high occlusion level")

        # print("Occlusion level: ", occ_level)
        # print("Number of objects: ", np.max(seg_img))

        scene_mask = scene_mask.astype(np.uint8)
        tgt_mask = tgt_mask.astype(np.uint8)

        # Keep only these models
        if model in ("vgn", "giga_aff", "giga", "giga_hr"):
            scene_mask = (seg_img >= 0).astype(np.uint8)
        elif model ==  "targo" or model == "targo_full_targ" or model == "targo_hunyun2" :
            scene_mask = (seg_img >= 0).astype(np.uint8)
        elif model == "targo_ptv3" or model == "ptv3_scene":
            scene_mask = (seg_img >= 0).astype(np.uint8)

        tic = time.time()
        tsdf.integrate((depth_img * scene_mask)[0], self.camera.intrinsic, extrinsic)
        # tsdf.integrate((depth_img * scene_mask), self.camera.intrinsic, extrinsic)
        tgt_mask_tsdf.integrate((depth_img * tgt_mask)[0], self.camera.intrinsic, extrinsic)
        # tgt_mask_tsdf.integrate((depth_img * tgt_mask), self.camera.intrinsic, extrinsic)
        timing = time.time() - tic

        targ_grid = np.load(curr_scene_path)["grid_targ"]

        # targ_grid = create_tsdf(
        #             self.size,
        #             60,
        #             depth_img * tgt_mask,
        #             self.camera.intrinsic,
        #             np.array(extrinsic.to_list()).reshape(1, 7),
        #         ).get_grid()

        if model in ("vgn", "giga_aff", "giga", "giga_hr"):
            if model != "giga_hr":
                targ_grid = np.load(curr_scene_path)["grid_targ"]
                scene_grid = np.load(curr_scene_path)["grid_scene"]
            else:
                targ_grid = create_tsdf(
                    self.size,
                    60,
                    depth_img * tgt_mask,
                    self.camera.intrinsic,
                    np.array(extrinsic.to_list()).reshape(1, 7),
                ).get_grid()
                scene_grid = create_tsdf(
                    self.size,
                    40,
                    depth_img * scene_mask,
                    self.camera.intrinsic,
                    np.array(extrinsic.to_list()).reshape(1, 7),
                ).get_grid()

            targ_mask = targ_grid > 0
            return tsdf, timing, scene_grid, targ_grid, targ_mask, occ_level

        elif model == "targo" or model == "targo_full_targ" or model == "targo_hunyun2":
            scene_no_targ_pc = np.load(curr_scene_path)["pc_scene_no_targ"]
            targ_depth_pc = np.load(curr_scene_path)["pc_depth_targ"].astype(np.float32)
            save_point_cloud_as_ply(targ_depth_pc, 'targ_depth_pc.ply')
            # scene_no_targ_pc = np.concatenate((scene_no_targ_pc, plane), axis=0)
            scene_no_targ_pc = scene_no_targ_pc / 0.3 - 0.5
            targ_depth_pc = targ_depth_pc / 0.3 - 0.5
            return tsdf, timing, scene_no_targ_pc, targ_depth_pc,targ_grid, occ_level
        
        elif model == "targo_ptv3":
            # For targo_ptv3: return both scene point cloud and target point cloud
            # Similar to original targo but with specific processing for PointTransformerV3
            scene_no_targ_pc = np.load(curr_scene_path)["pc_scene_no_targ"]
            targ_depth_pc = np.load(curr_scene_path)["pc_depth_targ"].astype(np.float32)
            
            # Normalize point clouds to [-0.5, 0.5] range as expected by PointTransformerV3
            scene_no_targ_pc = scene_no_targ_pc / 0.3 - 0.5
            targ_depth_pc = targ_depth_pc / 0.3 - 0.5
            
            # Apply coordinate filtering to ensure valid ranges for MinkowskiEngine
            from utils_giga import filter_and_pad_point_clouds
            scene_no_targ_pc = filter_and_pad_point_clouds(
                torch.from_numpy(scene_no_targ_pc).unsqueeze(0).float()
            ).squeeze(0).numpy()
            targ_depth_pc = filter_and_pad_point_clouds(
                torch.from_numpy(targ_depth_pc).unsqueeze(0).float()
            ).squeeze(0).numpy()
            
            return tsdf, timing, scene_no_targ_pc, targ_depth_pc, targ_grid, occ_level
            
        elif model == "ptv3_scene":
            # For ptv3_scene: only return scene point cloud (no target point cloud needed)
            # Load scene point cloud including target (complete scene)
            scene_pc_full = np.load(curr_scene_path)["pc_depth_scene_no_targ"]
            
            # Add target point cloud to scene for complete scene representation
            if "pc_depth_targ" in np.load(curr_scene_path).files:
                targ_depth_pc = np.load(curr_scene_path)["pc_depth_targ"].astype(np.float32)
                scene_pc_full = np.concatenate([scene_pc_full, targ_depth_pc], axis=0)
            
            complete_targ_pc = np.load(curr_scene_path)['complete_target_pc'].astype(np.float32)
            complete_targ_tsdf = np.load(curr_scene_path)['complete_target_tsdf']
            
            # Normalize to [-0.5, 0.5] range
            scene_pc_full = scene_pc_full / 0.3 - 0.5
            
            # Apply coordinate filtering 
            from utils_giga import filter_and_pad_point_clouds
            scene_pc_full = filter_and_pad_point_clouds(
                torch.from_numpy(scene_pc_full).unsqueeze(0).float()
            ).squeeze(0).numpy()
            
            # For ptv3_scene, we return the full scene point cloud as the main input
            # No separate target point cloud needed
            return tsdf, timing, scene_pc_full, complete_targ_pc, complete_targ_tsdf, targ_grid, occ_level
            
        elif model == "AnyGrasp_full_targ" or model == "FGC_full_targ":
            # Similar to targo type, but get both scene_no_targ_pc and targ_pc 
            # to be used later for concatenation as input
            pc_scene_depth_no_specify = np.load(curr_scene_path)["pc_scene_depth_no_specify"]
            pc_targ_depth_no_specify = np.load(curr_scene_path)["pc_targ_depth_no_specify"]
            pc_scene_no_targ_pc = get_pc_scene_no_targ_kdtree(pc_scene_depth_no_specify, pc_targ_depth_no_specify)
            # scene_no_targ_pc = np.load(curr_scene_path)["pc_scene_no_targ"]
            targ_depth_pc = np.load(curr_scene_path)["pc_depth_targ"].astype(np.float32)
            
            
            # Return all necessary data for AnyGrasp_full_targ processing
            return tsdf, timing, pc_scene_no_targ_pc, targ_depth_pc, targ_grid, occ_level
        
        elif model == "FGC-GraspNet" or model == "AnyGrasp":
            # Load point clouds in world coordinates
            scene_pc = np.load(curr_scene_path)['pc_scene_depth_no_specify']
            plane = np.load('./data//plane.npy')
            plane_hs = np.load('./data//plane_hs.npy')
            scene_pc = np.concatenate((scene_pc, plane_hs), axis=0)

            
            # Visualize scene point cloud using Open3D and save as PLY
            scene_pc_o3d = o3d.geometry.PointCloud()
            scene_pc_o3d.points = o3d.utility.Vector3dVector(scene_pc)

            # plane_hs_o3d = o3d.geometry.PointCloud()
            # plane_hs_o3d.points = o3d.utility.Vector3dVector(plane_hs)
            # o3d.io.write_point_cloud('plane_hs.ply', plane_hs_o3d)
            
            # Save the point cloud as PLY file
            output_path = 'scene_point_cloud.ply'
            o3d.io.write_point_cloud(output_path, scene_pc_o3d)
            print(f"Scene point cloud shape: {scene_pc.shape}")
            print(f"Saved scene point cloud to {output_path}")
            target_pc = np.load(curr_scene_path)['pc_targ_depth_no_specify']

            
            # Convert point clouds from world to camera view
            # Create transformation matrix from world to camera
            # world_to_camera = extrinsic.as_matrix()
            
            
            # # Apply transformation to scene point cloud
            # scene_pc_homogeneous = np.hstack((scene_pc, np.ones((scene_pc.shape[0], 1))))
            # scene_pc = (world_to_camera @ scene_pc_homogeneous.T).T[:, :3]
            
            # # Apply transformation to target point cloud
            # target_pc_homogeneous = np.hstack((target_pc, np.ones((target_pc.shape[0], 1))))
            # target_pc = (world_to_camera @ target_pc_homogeneous.T).T[:, :3]
            # Define a function to adjust point cloud to target size
            # Adjust scene_pc to target size
            # scene_pc = adjust_point_cloud_size(scene_pc)
            # plane = np.load('./data//plane.npy')
            # scene_pc = np.concatenate((scene_pc, plane), axis=0)
            # Convert depth image to point cloud for FGC-GraspNet
            # Load depth image and extrinsic parameters
            # depth_imgs = np.load(curr_scene_path)['depth_imgs']
            # scene_mask = np.load(curr_scene_path)['mask_scene']
            # extrinsics = np.load(curr_scene_path)['extrinsics']
            
            # Convert depth to point cloud
            # depth_img = depth_imgs[0]  # (480, 640)
            # extrinsic_tf = Transform.from_list(extrinsics[0])
            # scene_pc = depth_to_point_cloud(depth_img, self.camera.intrinsic, extrinsic_tf)
            # scene_pc shape is (2048, 3)
            # save_point_cloud_as_ply(scene_pc, 'scene_pc_with_plane.ply')  # Visualize using open3d
            ## save
            return tsdf, timing, scene_pc, target_pc, extrinsic, occ_level

        # # Load data
        # depth_img = np.load(curr_scene_path)["depth_imgs"]
        # tgt_mask = np.load(curr_scene_path)["mask_targ"]
        # scene_mask = np.load(curr_scene_path)["mask_scene"]
        
        # # Remove first dimension if needed
        # if len(depth_img.shape) == 3:
        #     depth_img = depth_img[0]  # Convert from (1, 480, 640) to (480, 640)
        # if len(tgt_mask.shape) == 3:
        #     tgt_mask = tgt_mask[0]
        # if len(scene_mask.shape) == 3:
        #     scene_mask = scene_mask[0]
        
        # # Save segmentation image
        # _, _, seg_img = self.camera.render_with_seg(extrinsic)

    def execute_grasp(self, grasp, remove=True, allow_contact=False, tgt_id=0, force_targ=False):
        """
        Execute the given grasp.
        """
        T_world_grasp = grasp.pose
        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp

        plan_failure = 0
        visual_failure = 0

        approach = T_world_grasp.rotation.as_matrix()[:, 2]
        angle = np.arccos(np.dot(approach, np.r_[0.0, 0.0, -1.0]))
        if angle > np.pi / 3.0:
            T_grasp_pregrasp_world = Transform(Rotation.identity(), [0.0, 0.0, 0.1])
            T_world_retreat = T_grasp_pregrasp_world * T_world_grasp
        else:
            T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.1])
            T_world_retreat = T_world_grasp * T_grasp_retreat

        self.gripper.reset(T_world_pregrasp)
        
        # For video recording, capture some frames at initial position
        if hasattr(self.world, 'recording') and self.world.recording:
            for _ in range(10):  # Capture more frames at the start for better visualization
                self.world.capture_frame()

        if self.gripper.detect_contact():
            result = Label.FAILURE, self.gripper.max_opening_width
            plan_failure = 1
        else:
            # Move to target position
            self.gripper.move_tcp_xyz(T_world_grasp, abort_on_contact=True)
            
            # Capture frames before closing the gripper
            if hasattr(self.world, 'recording') and self.world.recording:
                for _ in range(10):  # More frames for better visualization
                    self.world.capture_frame()
                    
            if self.gripper.detect_contact() and not allow_contact:
                result = Label.FAILURE, self.gripper.max_opening_width
            else:
                # Close the gripper
                self.gripper.move(0.0)
                
                # Capture frames after closing the gripper
                if hasattr(self.world, 'recording') and self.world.recording:
                    for _ in range(10):  # More frames for better visualization
                        self.world.capture_frame()
                        
                # Retreat gripper
                self.gripper.move_tcp_xyz(T_world_retreat, abort_on_contact=False)
                
                # Capture frames after retreating
                if hasattr(self.world, 'recording') and self.world.recording:
                    for _ in range(10):  # More frames for better visualization
                        self.world.capture_frame()
                        
                if not force_targ:
                    if self.check_success(self.gripper):
                        result = Label.SUCCESS, self.gripper.read()
                        if remove:
                            contacts = self.world.get_contacts(self.gripper.body)
                            self.world.remove_body(contacts[0].bodyB)
                    else:
                        result = Label.FAILURE, self.gripper.max_opening_width
                        visual_failure = 1
                else:
                    res, contacts_targ = self.check_success_target_grasp(self.gripper, tgt_id)
                    if res:
                        result = Label.SUCCESS, self.gripper.read()
                        if remove and contacts_targ:
                            self.world.remove_body(contacts_targ[0].bodyB)
                    else:
                        result = Label.FAILURE, self.gripper.max_opening_width
                        visual_failure = 1

        # # Capture some final frames
        # if hasattr(self.world, 'recording') and self.world.recording:
        #     for _ in range(10):  # More frames at the end for better visualization
        #         self.world.capture_frame()

        self.world.remove_body(self.gripper.body)
        if remove:
            self.remove_and_wait()

        return result[0], plan_failure, visual_failure

    def remove_and_wait(self):
        """
        Remove any objects that fell outside the workspace and wait for new rest state.
        """
        removed_object = True
        while removed_object:
            self.wait_for_objects_to_rest()
            removed_object = self.remove_objects_outside_workspace()

    def wait_for_objects_to_rest(self, timeout=2.0, tol=0.01):
        """
        Wait until objects are below a velocity threshold or a timeout occurs.
        """
        timeout = self.world.sim_time + timeout
        objects_resting = False
        while not objects_resting and self.world.sim_time < timeout:
            for _ in range(60):
                self.world.step()
            objects_resting = True
            for _, body in self.world.bodies.items():
                if np.linalg.norm(body.get_velocity()) > tol:
                    objects_resting = False
                    break

    def remove_objects_outside_workspace(self):
        """
        Remove objects that are out of the bounding box.
        """
        removed_object = False
        for body in list(self.world.bodies.values()):
            xyz = body.get_pose().translation
            if np.any(xyz < 0.0) or np.any(xyz > self.size):
                self.world.remove_body(body)
                removed_object = True
        return removed_object

    def check_success(self, gripper):
        """
        Check if fingers are in contact with any object and not fully closed.
        """
        contacts = self.world.get_contacts(gripper.body)
        res = len(contacts) > 0 and gripper.read() > 0.1 * gripper.max_opening_width
        return res

    def check_success_target_grasp(self, gripper, tgt_id=0):
        """
        Check if fingers contact the specific target object.
        """
        contacts = self.world.get_contacts(gripper.body)
        contacts_targ = []
        for contact in contacts:
            if contact.bodyB.uid == tgt_id:
                contacts_targ.append(contact)
        res = len(contacts_targ) > 0 and gripper.read() > 0.1 * gripper.max_opening_width
        return res, contacts_targ

    def check_success_valid(self, gripper, tgt_id=0):
        """
        Optional function for checking valid contact with specific object ID.
        """
        contacts = self.world.get_contacts_valid(gripper.body, tgt_id)
        res = len(contacts) > 0 and gripper.read() > 0.1 * gripper.max_opening_width
        return res

    def start_video_recording(self, filename, video_path):
        """
        Start recording a video of the simulation using OpenCV.
        
        Args:
            filename (str): The name of the video file (without extension)
            video_path (str): Path to the directory where the video will be saved
            
        Returns:
            int: Log ID to be used when stopping the recording
        """
        # Use BtWorld's OpenCV recording method
        log_id = self.world.start_video_recording(filename, video_path)
        print(f"Starting OpenCV video recording: {video_path}/{filename}.mp4")
        return log_id
    
    def stop_video_recording(self, log_id):
        """
        Stop an active video recording.
        
        Args:
            log_id (int): The log ID returned by start_video_recording
        """
        # Use BtWorld's OpenCV recording method to stop recording
        self.world.stop_video_recording(log_id)
        print("Video recording completed")


class Gripper(object):
    """
    Simple simulated Panda hand with open/close motion.
    """

    def __init__(self, world):
        self.world = world
        self.urdf_path = Path("data/urdfs/panda/hand.urdf")

        self.max_opening_width = 0.08
        self.finger_depth = 0.05
        self.T_body_tcp = Transform(Rotation.identity(), [0.0, 0.0, 0.022])
        self.T_tcp_body = self.T_body_tcp.inverse()

    def reset(self, T_world_tcp):
        """
        Spawn the gripper in the given pose.
        """
        T_world_body = T_world_tcp * self.T_tcp_body
        self.body = self.world.load_urdf(self.urdf_path, T_world_body)
        self.body.set_pose(T_world_body)
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

        # Keep fingers centered
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
        """
        Update the constraint that holds the gripper's body in place.
        """
        T_world_body = T_world_tcp * self.T_tcp_body
        self.constraint.change(
            jointChildPivot=T_world_body.translation,
            jointChildFrameOrientation=T_world_body.rotation.as_quat(),
            maxForce=300,
        )

    def set_tcp(self, T_world_tcp):
        """
        Directly set the pose of the gripper's Tool Center Point (TCP).
        """
        T_word_body = T_world_tcp * self.T_tcp_body
        self.body.set_pose(T_word_body)
        self.update_tcp_constraint(T_world_tcp)

    def move_tcp_xyz(self, target, eef_step=0.002, vel=0.10, abort_on_contact=True):
        """
        Move the gripper in small steps toward a target pose.
        """
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
                # Capture video frame
                if hasattr(self.world, 'recording') and self.world.recording:
                    self.world.capture_frame()
            if abort_on_contact and self.detect_contact():
                return

    def detect_contact(self, threshold=5):
        """
        Check contact with any object. If any contact is found, return True.
        """
        if self.world.get_contacts(self.body):
            return True
        else:
            return False

    def move(self, width):
        """
        Move fingers to the specified width.
        """
        self.joint1.set_position(0.5 * width)
        self.joint2.set_position(0.5 * width)
        
        # Take more steps for smoother video when recording
        steps = 12 if hasattr(self.world, 'recording') and self.world.recording else int(0.5 / self.world.dt)
        
        for _ in range(steps):
            self.world.step()
            # Capture video frame
            if hasattr(self.world, 'recording') and self.world.recording:
                self.world.capture_frame()

    def read(self):
        """
        Current opening width = sum of the two finger joint positions.
        """
        width = self.joint1.get_position() + self.joint2.get_position()
        return width
