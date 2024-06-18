import os
import argparse
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import h5py
import logging

from vgn.grasp import *
from vgn.simulation import ClutterRemovalSim
from vgn.utils.transform import Transform
from vgn.utils.implicit import get_scene_from_mesh_pose_list
from generate_data_soco import depth_imgs_to_point_clouds, reconstruct_pc, filter_boundary, xy_within_bounds, specify_num_points, NUM_POINTS, render_images, render_side_images, MAX_VIEWPOINT_COUNT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# read in given VGN dataset, and generate the corresponding dataset for shape completion
def main(args):
    sim = ClutterRemovalSim(args.scene, args.object_set, size=args.size, gui=args.sim_gui, add_noise=False, sideview=True)
    all_mpl = os.listdir(args.root_mpl)[args.min_idx : args.max_idx]
    clutter_mpl = [mpl for mpl in all_mpl if '_c.npz' in mpl]
    for i, curr_mesh_pose_list in enumerate(clutter_mpl):
        scene_name = curr_mesh_pose_list[:-4]
        logger.info(f"starting {i}-th scene: {scene_name}")
        mesh_pose_dict = np.load(os.path.join(args.root_mpl, curr_mesh_pose_list), allow_pickle=True)['pc'].item()
        mesh_pose_dict.pop(0)
        mesh_pose_list = list(mesh_pose_dict.values())

        sim.world.reset()
        sim.world.set_gravity([0.0, 0.0, -9.81])
        sim.draw_workspace()
        table_height = sim.gripper.finger_depth
        sim.place_table(table_height)
        sim.save_state()
        sim.lower = np.array([0.02, 0.02, 0.055])
        sim.upper = np.array([sim.size-0.02, sim.size-0.02, sim.size])
        
        for id, mesh in enumerate(mesh_pose_list):
            pose = Transform.from_matrix(mesh[2])
            urdf_path = mesh[0].replace(".obj", ".urdf")
            body = sim.world.load_urdf(urdf_path=urdf_path, pose=pose, scale=mesh[1])

        depth_side_c, extr_side_c, seg_side_c = render_side_images(sim, 1, random=False, segmentation=True)
        depth_c, extr_c, _ = render_images(sim, MAX_VIEWPOINT_COUNT, segmentation=True)
        pc_c = reconstruct_pc(sim, depth_c, extr_c) # point cloud of the clutter scene
        assert not pc_c.is_empty(), "Clutter scene point cloud is empty"
        
        # get object poses
        target_poses = {}
        target_bodies = {}
        count_cluttered = {}    # count_cluttered is a dictionary that stores the counts of target object
        
        body_ids = deepcopy(list(sim.world.bodies.keys()))
        body_ids.remove(0)  # remove the plane
        
        for target_id in body_ids:
            assert target_id != 0
            target_body = sim.world.bodies[target_id]   # get the target object
            target_poses[target_id] = target_body.get_pose()
            target_bodies[target_id] = target_body
            count_cluttered[target_id] = np.count_nonzero(seg_side_c[0] == target_id)   # count the number of pixels of the target object in the cluttered scene
        
        # remove all objects first except the plane
        for body_id in body_ids:
            body = sim.world.bodies[body_id]
            sim.world.remove_body(body)
        
        # cnt = 0
        for target_id in body_ids:
            if count_cluttered[target_id] == 0:  # if the target object is not in the cluttered scene, skip
                continue
            
            assert target_id != 0 # the plane should not be a target object
            body = target_bodies[target_id]
            
            #--------------------------------- single scene ---------------------------------##
            target_body = sim.world.load_urdf(body.urdf_path, target_poses[target_id], scale=body.scale)
            sim.save_state()    # single scene: target object only and plane

            depth_side_s, extr_side_s, seg_side_s = render_side_images(sim, 1, random=False, segmentation=True) # side view is fixed
            assert np.unique(seg_side_s[seg_side_s > 0]).shape[0] == 1, "Single scene should have only 1 object"

            count_single = np.count_nonzero(seg_side_s[0] == target_body.uid)

            #--------------------------------- cluttered scene ---------------------------------##
            sim.world.remove_body(target_body)
            sim.save_state()
            
            occ_level_c = 1 - count_cluttered[target_id] / count_single # calculate occlusion level for the target object in cluttered scene
            if occ_level_c > 0.95:
                continue
            
            pc_part_c = depth_imgs_to_point_clouds(sim, depth_side_c, extr_side_c, seg_side_c, target_id, args.noise_type)[0]
            pc_part_c = filter_boundary(pc_part_c, sim.size)
            if pc_part_c.shape[0] < 25:
                continue
            
            gt_mesh = get_scene_from_mesh_pose_list([mesh_pose_list[target_id-1]])

            pc_com = np.asarray(gt_mesh.sample(NUM_POINTS, return_index=False), dtype=np.float32)
            if not xy_within_bounds(pc_com, sim.size):
                continue
            # will not filter boundary for gt pc
            pc_complete.append(pc_com)
            pc_part_c = specify_num_points(pc_part_c, NUM_POINTS)
            pc_partial.append(pc_part_c)
            labels.append(name_to_id[target_body.name])
        
        logger.info(f"ending {i}-th scene: {scene_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sc dataset from given packed dataset")
    parser.add_argument("--root_mpl", type=str, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--scene", type=str, choices=["pile", "packed", "dex-ycb"], default="packed")
    parser.add_argument("--object-set", type=str, default="packed/train")
    parser.add_argument("--size", type=float, default=0.6)
    parser.add_argument("--sim-gui", default=False)
    parser.add_argument("--visualize", action="store_true", default=True)
    parser.add_argument("--noise-type", type=str, default="norm")
    parser.add_argument("--min-idx", type=int, default=0)
    parser.add_argument("--max-idx", type=int, default=17000)
    args = parser.parse_args()
    
    os.makedirs(args.dest, exist_ok=True)
    obj_set = str(args.object_set).replace("/", "_")
    h5_file = f"{args.dest}/pc_{obj_set}_{args.min_idx}_{args.max_idx}.h5"

    urdf_root = Path("data/urdfs")
    urdf_path = urdf_root / args.object_set
    urdf_files = [f for f in urdf_path.iterdir() if f.suffix == ".urdf"]
    num_total_obj = len(urdf_files)

    name_to_id = {}
    # Iterate through each URDF file
    for idx, urdf_file in enumerate(urdf_files):
        # parse the URDF file
        tree = ET.parse(urdf_file)
        root = tree.getroot()
        
        # extract the name attribute from the <robot> tag
        robot_name = root.attrib['name']
        name_to_id[robot_name] = idx

    pc_partial = []
    pc_complete = []
    labels = []
    
    logger.info("Start")
    main(args)
    
    # turn data into numpy array
    pc_partial = np.array(pc_partial, dtype=np.float32)
    pc_complete = np.array(pc_complete, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    logger.info("Completed data generation")

    # save to h5 file
    with h5py.File(h5_file, "w") as f:
        f.create_dataset('incomplete_pcds', data=pc_partial)
        f.create_dataset('complete_pcds', data=pc_complete)
        f.create_dataset('labels', data=labels)

    logger.info("End")
