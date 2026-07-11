import numpy as np
from scipy.spatial.transform import Rotation
from vgn.grasp import Grasp
from vgn.utils.transform import Transform

def anygrasp_to_vgn(grasp_group,
                    extrinsic=None,
                    workspace_size=None,
                    gripper_offset=np.array([0, 0, -0.02]),
                    # gripper_offset=np.array([0, 0, 0]),
                    grasps_are_in_world=True):
    """
    Convert AnyGrasp/GraspNet grasps to VGN format
    
    Args:
        grasp_group: GraspGroup from AnyGrasp/GraspNet
        extrinsic: Camera extrinsic matrix (camera→world), required when grasps are not in world frame
        workspace_size: Workspace size limit for filtering
        gripper_offset: Offset for the Franka gripper
        grasps_are_in_world: If True, skip camera extrinsic transformation
    """
    vgn_grasps = []
    scores = []
    
    for grasp in grasp_group:
        # First add a 90° rotation around Y-axis in the grasp's local coordinate system
        grasp_R = Rotation.from_matrix(grasp.rotation_matrix) \
                  * Rotation.from_euler('Y', np.pi/2)
        grasp_tf = Transform(grasp_R, grasp.translation)
        
        # If input grasps are already in world frame, use them directly
        if grasps_are_in_world:
            grasp_world_mat = grasp_tf.as_matrix()
        else:
            # Otherwise transform using extrinsic (camera→world)
            if extrinsic is None:
                raise ValueError("extrinsic cannot be None when grasps are not in world frame")
            # If extrinsic is camera→world
            grasp_world_mat = extrinsic.as_matrix() @ grasp_tf.as_matrix()
            # If your extrinsic is actually world→camera, use np.linalg.inv(extrinsic.as_matrix()) instead
        
        # Filter out grasps outside the workspace
        if workspace_size is not None:
            x, y = grasp_world_mat[0,3], grasp_world_mat[1,3]
            if not (0.0 <= x <= workspace_size and 0.0 <= y <= workspace_size):
                continue
        
        # Apply gripper_offset (in world frame along Z direction)
        if gripper_offset is not None:
            offset_T = Transform(Rotation.identity(), gripper_offset)
            # Note: Right multiply to apply offset in local frame
            grasp_world_mat = grasp_world_mat @ offset_T.as_matrix()
        
        # Construct VGN Grasp
        curr_grasp = Grasp(Transform.from_matrix(grasp_world_mat),
                           grasp.width)
        vgn_grasps.append(curr_grasp)
        scores.append(grasp.score)
    
    return vgn_grasps, scores


def anygrasp_to_vgn_with_region_filter(grasp_group,
                    extrinsic=None,
                    workspace_size=None,
                    region_lower=np.array([0.02, 0.02, 0.055]),
                    region_upper=np.array([0.28, 0.28, 0.30]),
                    gripper_offset=np.array([0, 0, -0.02]),
                    grasps_are_in_world=True):
    """
    Convert AnyGrasp/GraspNet grasps to VGN format with region filtering
    
    Args:
        grasp_group: GraspGroup from AnyGrasp/GraspNet
        extrinsic: Camera extrinsic matrix (camera→world), required when grasps are not in world frame
        workspace_size: Workspace size limit for filtering
        region_lower: Lower bounds of the valid region [x_min, y_min, z_min]
        region_upper: Upper bounds of the valid region [x_max, y_max, z_max]
        gripper_offset: Offset for the Franka gripper
        grasps_are_in_world: If True, skip camera extrinsic transformation
    """
    vgn_grasps = []
    scores = []
    
    # Print region bounds for debugging
    print(f"Filtering grasps to region: lower={region_lower}, upper={region_upper}")
    
    total_grasps = len(grasp_group)
    filtered_count = 0
    
    for grasp in grasp_group:
        # First add a 90° rotation around Y-axis in the grasp's local coordinate system
        grasp_R = Rotation.from_matrix(grasp.rotation_matrix) \
                  * Rotation.from_euler('Y', np.pi/2)
        grasp_tf = Transform(grasp_R, grasp.translation)
        
        # If input grasps are already in world frame, use them directly
        if grasps_are_in_world:
            grasp_world_mat = grasp_tf.as_matrix()
        else:
            # Otherwise transform using extrinsic (camera→world)
            if extrinsic is None:
                raise ValueError("extrinsic cannot be None when grasps are not in world frame")
            # If extrinsic is camera→world
            grasp_world_mat = extrinsic.as_matrix() @ grasp_tf.as_matrix()
        
        # Get grasp position in world coordinates
        grasp_pos = grasp_world_mat[:3, 3]
        
        # Filter out grasps outside the workspace size if specified
        if workspace_size is not None:
            x, y = grasp_pos[0], grasp_pos[1]
            if not (0.0 <= x <= workspace_size and 0.0 <= y <= workspace_size):
                filtered_count += 1
                continue
        
        # Filter out grasps outside the specified region bounds
        if not (region_lower[0] <= grasp_pos[0] <= region_upper[0] and
                region_lower[1] <= grasp_pos[1] <= region_upper[1] and
                region_lower[2] <= grasp_pos[2] <= region_upper[2]):
            filtered_count += 1
            continue
        
        # Apply gripper_offset (in world frame along Z direction)
        if gripper_offset is not None:
            offset_T = Transform(Rotation.identity(), gripper_offset)
            # Note: Right multiply to apply offset in local frame
            grasp_world_mat = grasp_world_mat @ offset_T.as_matrix()
        
        # Construct VGN Grasp
        curr_grasp = Grasp(Transform.from_matrix(grasp_world_mat),
                           grasp.width)
        vgn_grasps.append(curr_grasp)
        scores.append(grasp.score)
    
    # Print filtering statistics
    print(f"Filtered {filtered_count} out of {total_grasps} grasps based on region bounds")
    print(f"Remaining grasps: {len(vgn_grasps)}")
    
    return vgn_grasps, scores


def fgc_to_vgn(grasp_group, extrinsic, workspace_size=None, gripper_offset=np.array([0, 0, -0.02])):
    """
    Convert grasps from FGC-Grasp format to VGN format.
    This uses the same conversion as AnyGrasp as they have similar formats.
    
    Args:
        grasp_group: GraspGroup from FGC-Grasp
        extrinsic: Camera extrinsic matrix (camera to world transform)
        workspace_size: Size limits of the workspace for filtering [optional]
        gripper_offset: Offset to apply for the Franka gripper [default: 2cm in -Z direction]
    
    Returns:
        vgn_grasps: List of VGN Grasp objects
        scores: List of corresponding grasp scores
    """
    # FGC-Grasp uses the same format as AnyGrasp for grasp representation
    return anygrasp_to_vgn(grasp_group, extrinsic, workspace_size, gripper_offset)


def fgc_to_vgn_with_region_filter(grasp_group, extrinsic, workspace_size=None, 
                                  region_lower=np.array([0.02, 0.02, 0.055]),
                                  region_upper=np.array([0.28, 0.28, 0.30]),
                                  gripper_offset=np.array([0, 0, -0.02])):
    """
    Convert grasps from FGC-Grasp format to VGN format with region filtering.
    
    Args:
        grasp_group: GraspGroup from FGC-Grasp
        extrinsic: Camera extrinsic matrix (camera to world transform)
        workspace_size: Size limits of the workspace for filtering [optional]
        region_lower: Lower bounds of the valid region [x_min, y_min, z_min]
        region_upper: Upper bounds of the valid region [x_max, y_max, z_max]
        gripper_offset: Offset to apply for the Franka gripper [default: 2cm in -Z direction]
    
    Returns:
        vgn_grasps: List of VGN Grasp objects
        scores: List of corresponding grasp scores
    """
    # FGC-Grasp uses the same format as AnyGrasp for grasp representation
    return anygrasp_to_vgn_with_region_filter(grasp_group, extrinsic, workspace_size, 
                                            region_lower, region_upper, gripper_offset)


def vgn_to_anygrasp(vgn_grasps, scores, extrinsic):
    """
    Convert grasps from VGN format to AnyGrasp/GraspNet format
    
    Args:
        vgn_grasps: List of VGN Grasp objects
        scores: List of grasp scores
        extrinsic: Camera extrinsic matrix (world to camera transform)
    
    Returns:
        grasp_group: GraspGroup for AnyGrasp/GraspNet
    
    Note: 
        This function requires graspnetAPI's GraspGroup class to be imported.
        Import it only when this function is called to avoid dependency issues.
    """
    try:
        from graspnetAPI.grasp import GraspGroup, Grasp as GraspnetGrasp
    except ImportError:
        raise ImportError("graspnetAPI is required for vgn_to_anygrasp conversion. "
                         "Please install it with: pip install graspnetAPI")
    
    grasp_list = []
    
    for i, vgn_grasp in enumerate(vgn_grasps):
        # Get the grasp pose in world frame
        grasp_tf_world = vgn_grasp.pose.as_matrix()
        
        # Remove the gripper offset
        offset = np.array([0, 0, 0.02])  # Reverse the offset applied in anygrasp_to_vgn
        offset_tf = Transform(Rotation.from_matrix(np.eye(3)), offset)
        grasp_tf_world = grasp_tf_world @ offset_tf.as_matrix()
        
        # Transform to camera frame
        grasp_tf_camera = extrinsic.as_matrix() @ grasp_tf_world
        
        # Remove the Y-axis 90 degree rotation
        grasp_R = Rotation.from_matrix(grasp_tf_camera[:3, :3]) * Rotation.from_euler('Y', -np.pi/2)
        
        # Create AnyGrasp/GraspNet grasp
        score = scores[i] if i < len(scores) else 0.0
        grasp = GraspnetGrasp(
            rotation_matrix=grasp_R.as_matrix(),
            translation=grasp_tf_camera[:3, 3],
            width=vgn_grasp.width,
            score=score
        )
        grasp_list.append(grasp)
    
    return GraspGroup(grasp_list) 