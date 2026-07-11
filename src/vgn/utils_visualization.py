import os
import numpy as np
import trimesh
from datetime import datetime
from src.vgn.utils.implicit import (
    get_mesh_pose_list_from_world,
    get_scene_from_mesh_pose_list
)

def visualize_failure_case(sim, scene_name, result_path, vis_rgb=False, grasp=None, score=None, state=None, grasp_plan_fn=None, 
                          model_type='targo', target_mesh_gt=None, scene_target_mesh_dict=None):
    """
    Visualize failure cases with comprehensive analysis:
    1. Input Scene - 输入场景
    2. GT Target - 真实目标物体  
    3. Target Completion - 目标补全结果（仅对targo模型）
    4. Affordance Visualization - 抓取可能性可视化
    5. CD和IoU计算 - 仅对targo模型
    
    Args:
        sim: Simulation instance
        scene_name: Name of the scene
        result_path: Path to save visualizations (should be scene-specific directory)
        vis_rgb: Whether to visualize RGB images
        grasp: Grasp object (if available)
        score: Grasp score (if available)
        state: State object for affordance visualization
        grasp_plan_fn: Grasp planning function for affordance generation
        model_type: Type of model ('targo', 'targo_partial', 'targo_full_gt')
        target_mesh_gt: Ground truth target mesh
        scene_target_mesh_dict: Dictionary containing target meshes for each scene
    """
    try:
        # Use the provided result_path directly (should be scene-specific directory)
        failure_vis_dir = result_path
        os.makedirs(failure_vis_dir, exist_ok=True)
        
        # Create scene information file for failure case
        scene_info_path = os.path.join(failure_vis_dir, 'scene_info.txt')
        with open(scene_info_path, 'w') as f:
            f.write(f"Scene_ID: {scene_name}\n")
            f.write(f"Model_Type: {model_type}\n")
            f.write(f"Failure_Type: {'No valid grasp found' if grasp is None else 'Grasp execution failed'}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if grasp is not None:
                f.write(f"Attempted_Grasp_Position: {grasp.pose.translation}\n")
                f.write(f"Attempted_Grasp_Score: {score if score is not None else 'N/A'}\n")
        print(f"Saved failure scene info: {scene_info_path}")
        
        # 1. Save Input Scene
        scene_mesh = None
        scene_mesh_path = os.path.join(failure_vis_dir, 'input_scene.ply')
        if hasattr(sim, 'world') and sim.world:
            # Get scene mesh from simulation
            mesh_pose_list = get_mesh_pose_list_from_world(sim.world, sim.object_set)
            if mesh_pose_list:
                scene_mesh = get_scene_from_mesh_pose_list(mesh_pose_list, -1, return_target_mesh=False)
                if scene_mesh is not None:
                    scene_mesh.export(scene_mesh_path)
                    print(f"Saved input scene mesh: {scene_mesh_path}")
        
        # 2. Save GT Target
        if target_mesh_gt is not None:
            gt_target_path = os.path.join(failure_vis_dir, 'gt_target.ply')
            target_mesh_gt.export(gt_target_path)
            print(f"Saved GT target mesh: {gt_target_path}")
        elif scene_target_mesh_dict and scene_name in scene_target_mesh_dict:
            gt_target_path = os.path.join(failure_vis_dir, 'gt_target.ply')
            scene_target_mesh_dict[scene_name].export(gt_target_path)
            print(f"Saved GT target mesh from dict: {gt_target_path}")
        
        # Save RGB images if requested
        if vis_rgb:
            # Save RGB directly in scene directory
            rgb_path = os.path.join(failure_vis_dir, 'rgb.png')
            
            try:
                # Try different methods to capture RGB image
                rgb_img = None
                
                # Method 1: Direct camera render with extrinsic
                if hasattr(sim, 'camera') and sim.camera is not None:
                    try:
                        # Set up camera pose similar to acquire_single_tsdf_target_grid
                        from src.vgn.utils.transform import Transform, Rotation
                        from src.vgn.perception import camera_on_sphere
                        
                        half_size = getattr(sim, 'size', 0.3) / 2
                        origin_yz = np.r_[half_size, half_size]
                        
                        if getattr(sim, 'sideview', False):
                            origin = Transform(Rotation.identity(), np.r_[origin_yz, getattr(sim, 'size', 0.3) / 3])
                            theta, phi = np.pi / 3.0, -np.pi / 2.0
                        else:
                            origin = Transform(Rotation.identity(), np.r_[origin_yz, 0])
                            theta, phi = np.pi / 6.0, 0
                        
                        r = 2.0 * getattr(sim, 'size', 0.3)
                        extrinsic = camera_on_sphere(origin, r, theta, phi)
                        
                        rgb_img, depth_img, seg_img = sim.camera.render_with_seg(extrinsic)
                        print(f"Camera render method 1 successful for {scene_name}")
                    except Exception as e:
                        print(f"Camera render method 1 failed for {scene_name}: {e}")
                
                # Method 2: Try render method if available
                if rgb_img is None and hasattr(sim, 'render'):
                    try:
                        rgb_img = sim.render()
                        print(f"Render method successful for {scene_name}")
                    except Exception as e:
                        print(f"Render method failed for {scene_name}: {e}")
                
                # Method 3: Try world render if available
                if rgb_img is None and hasattr(sim, 'world') and hasattr(sim.world, 'render'):
                    try:
                        rgb_img = sim.world.render()
                        print(f"World render method successful for {scene_name}")
                    except Exception as e:
                        print(f"World render method failed for {scene_name}: {e}")
                
                # Method 4: Try pyrender if available
                if rgb_img is None:
                    try:
                        import pyrender
                        # Create a simple scene for rendering
                        scene = pyrender.Scene()
                        # Add basic lighting
                        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0))
                        
                        # Try to get camera from sim
                        if hasattr(sim, 'camera') and sim.camera is not None:
                            camera = sim.camera
                            if hasattr(camera, 'intrinsic'):
                                # Create pyrender camera
                                fx, fy = camera.intrinsic.K[0, 0], camera.intrinsic.K[1, 1]
                                cx, cy = camera.intrinsic.K[0, 2], camera.intrinsic.K[1, 2]
                                camera_pyrender = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
                                scene.add(camera_pyrender)
                                
                                # Render
                                renderer = pyrender.OffscreenRenderer(640, 480)
                                rgb_img, _ = renderer.render(scene)
                                print(f"Pyrender method successful for {scene_name}")
                    except Exception as e:
                        print(f"Pyrender method failed for {scene_name}: {e}")
                
                # Save the captured image as PNG
                if rgb_img is not None:
                    try:
                        # Convert to PIL Image and save as PNG
                        from PIL import Image
                        if isinstance(rgb_img, np.ndarray):
                            if rgb_img.dtype != np.uint8:
                                rgb_img = (rgb_img * 255).astype(np.uint8)
                            pil_img = Image.fromarray(rgb_img)
                            pil_img.save(rgb_path)
                            print(f"Saved RGB image as PNG: {rgb_path}")
                        else:
                            # If it's already a PIL Image
                            rgb_img.save(rgb_path)
                            print(f"Saved RGB image as PNG: {rgb_path}")
                    except Exception as e:
                        print(f"Error saving RGB image as PNG for {scene_name}: {e}")
                        # Fallback to matplotlib
                        import matplotlib.pyplot as plt
                        plt.imsave(rgb_path, rgb_img)
                        print(f"Saved RGB image using matplotlib: {rgb_path}")
                else:
                    # Save a placeholder image with error message
                    error_path = os.path.join(failure_vis_dir, 'rgb_error.txt')
                    with open(error_path, 'w') as f:
                        f.write(f"Failed to capture RGB image for {scene_name}\n")
                        f.write("Available methods tried:\n")
                        f.write("1. sim.camera.render(extrinsic)\n")
                        f.write("2. sim.render()\n")
                        f.write("3. sim.world.render()\n")
                        f.write("4. pyrender\n")
                    print(f"Could not capture RGB image for {scene_name}, saved error info")
                    
            except Exception as e:
                print(f"Error in RGB capture for {scene_name}: {e}")
                import traceback
                error_traceback = traceback.format_exc()
                print(f"RGB capture traceback: {error_traceback}")
                # Save error info
                error_path = os.path.join(failure_vis_dir, 'rgb_error.txt')
                with open(error_path, 'w') as f:
                    f.write(f"RGB capture error: {e}\n")
                    f.write(f"Traceback:\n{error_traceback}\n")
        
        # 3. Generate and save Target Completion (only for targo model)
        completion_mesh = None
        if model_type == 'targo' and state is not None:
            try:
                # Generate completion using shape completion network
                from src.vgn.utils_giga import mesh_to_tsdf, tsdf_to_mesh
                import torch
                
                # Get target point cloud from state
                if hasattr(state, 'targ_pc') and state.targ_pc is not None:
                    targ_pc = state.targ_pc
                    if isinstance(targ_pc, torch.Tensor):
                        targ_pc = targ_pc.cpu().numpy()
                    
                    # Convert to TSDF
                    targ_tsdf = mesh_to_tsdf(targ_pc, resolution=40)
                    
                    # Convert back to mesh (this simulates completion)
                    completion_vertices, completion_faces = tsdf_to_mesh(targ_tsdf)
                    if completion_vertices is not None and completion_faces is not None:
                        import trimesh
                        completion_mesh = trimesh.Trimesh(vertices=completion_vertices, faces=completion_faces)
                        
                        completion_path = os.path.join(failure_vis_dir, 'target_completion.ply')
                        completion_mesh.export(completion_path)
                        print(f"Saved target completion mesh: {completion_path}")
                        
                        # 5. Calculate CD and IoU for targo model
                        if target_mesh_gt is not None and completion_mesh is not None:
                            try:
                                from src.vgn.utils_giga import compute_chamfer_distance, compute_iou
                                
                                # Sample points from both meshes
                                gt_points = target_mesh_gt.sample(2048)
                                completion_points = completion_mesh.sample(2048)
                                
                                # Calculate CD
                                cd = compute_chamfer_distance(gt_points, completion_points)
                                
                                # Calculate IoU
                                iou = compute_iou(target_mesh_gt, completion_mesh)
                                
                                # Save CD and IoU results
                                metrics_path = os.path.join(failure_vis_dir, 'completion_metrics.txt')
                                with open(metrics_path, 'w') as f:
                                    f.write(f"Scene: {scene_name}\n")
                                    f.write(f"Model Type: {model_type}\n")
                                    f.write(f"Chamfer Distance (CD): {cd:.6f}\n")
                                    f.write(f"IoU: {iou:.6f}\n")
                                    f.write(f"GT Target Points: {len(gt_points)}\n")
                                    f.write(f"Completion Points: {len(completion_points)}\n")
                                print(f"Saved completion metrics: {metrics_path}")
                                
                            except Exception as metrics_error:
                                print(f"Error calculating CD/IoU for {scene_name}: {metrics_error}")
                                
            except Exception as completion_error:
                print(f"Error generating target completion for {scene_name}: {completion_error}")
        
        # 4. Generate and save affordance visualization using colored_scene_mesh
        if state is not None and grasp_plan_fn is not None and scene_mesh is not None:
            # Save affordance directly in scene directory
            
            try:
                # Generate affordance visualization using the same method as detection_implicit_vgn.py
                from src.vgn.utils import visual
                
                # Create a mock grasp planner call to get the affordance volumes
                if hasattr(grasp_plan_fn, '__call__'):
                    try:
                        # Call the grasp planner to get affordance volumes
                        grasps, scores, _ = grasp_plan_fn(
                            state, scene_mesh, 
                            scene_name=scene_name, 
                            cd_iou_measure=False, 
                            target_mesh_gt=None
                        )
                        
                        # Generate colored scene mesh for affordance visualization
                        # This mimics the logic from detection_implicit_vgn.py lines 378-380
                        if hasattr(grasp_plan_fn, 'net') and hasattr(grasp_plan_fn, 'pos'):
                            # Get affordance volumes by calling the network directly
                            from src.vgn.detection_implicit_vgn import predict_targo
                            
                            # Prepare inputs based on model type
                            if state.type in ['targo', 'targo_partial', 'targo_full_gt']:
                                scene_no_targ_pc = state.scene_no_targ_pc
                                if state.type == 'targo':
                                    targ_pc = state.targ_pc
                                elif state.type == 'targo_partial':
                                    targ_pc = state.targ_pc
                                elif state.type == 'targo_full_gt':
                                    targ_pc = state.targ_pc
                                
                                inputs = (scene_no_targ_pc, targ_pc)
                                
                                # Get affordance volumes
                                qual_vol, rot_vol, width_vol = predict_targo(inputs, grasp_plan_fn.net, grasp_plan_fn.device)
                                
                                # Reshape volumes
                                resolution = 40
                                qual_vol = qual_vol.reshape((resolution, resolution, resolution))
                                rot_vol = rot_vol.reshape((resolution, resolution, resolution, 4))
                                width_vol = width_vol.reshape((resolution, resolution, resolution))
                                
                                # Generate colored scene mesh
                                colored_scene_mesh = visual.affordance_visual(
                                    qual_vol, rot_vol, scene_mesh, 0.3, resolution
                                )
                                
                                # Save colored scene mesh (use .obj format to match successful cases)
                                affordance_path = os.path.join(failure_vis_dir, 'affordance_mesh.obj')
                                colored_scene_mesh.export(affordance_path)
                                print(f"Saved affordance visualization: {affordance_path}")
                                
                                # Also save affordance info
                                info_path = os.path.join(failure_vis_dir, 'affordance_info.txt')
                                with open(info_path, 'w') as f:
                                    f.write(f"Scene: {scene_name}\n")
                                    f.write(f"Model Type: {model_type}\n")
                                    f.write(f"Number of grasps found: {len(grasps) if grasps else 0}\n")
                                    if grasps:
                                        f.write(f"Best grasp score: {max(scores) if scores else 'N/A'}\n")
                                        f.write(f"Average grasp score: {sum(scores)/len(scores) if scores else 'N/A'}\n")
                                        f.write(f"Score range: [{min(scores):.4f}, {max(scores):.4f}]\n")
                                    else:
                                        f.write("No valid grasps found - this explains the failure\n")
                                print(f"Saved affordance info: {info_path}")
                                
                    except Exception as grasp_error:
                        print(f"Error in grasp planning for {scene_name}: {grasp_error}")
                        
                        # Fallback: save basic affordance info without colored mesh
                        info_path = os.path.join(failure_vis_dir, 'affordance_info.txt')
                        with open(info_path, 'w') as f:
                            f.write(f"Scene: {scene_name}\n")
                            f.write(f"Model Type: {model_type}\n")
                            f.write(f"Error generating affordance visualization: {grasp_error}\n")
                        print(f"Saved basic affordance info: {info_path}")
                        
            except Exception as e:
                print(f"Error generating affordance map for {scene_name}: {e}")
                import traceback
                error_traceback = traceback.format_exc()
                print(f"Affordance generation traceback: {error_traceback}")
                # Save error info
                error_path = os.path.join(failure_vis_dir, 'affordance_error.txt')
                with open(error_path, 'w') as f:
                    f.write(f"Error generating affordance map: {e}\n")
                    f.write(f"Traceback:\n{error_traceback}\n")
        
        # Save grasp information if available
        if grasp is not None:
            grasp_info_path = os.path.join(failure_vis_dir, 'grasp_info.txt')
            with open(grasp_info_path, 'w') as f:
                f.write(f"Scene: {scene_name}\n")
                f.write(f"Model Type: {model_type}\n")
                f.write(f"Grasp Position: {grasp.pose.translation}\n")
                f.write(f"Grasp Rotation: {grasp.pose.rotation.as_quat()}\n")
                f.write(f"Grasp Width: {grasp.width}\n")
                if score is not None:
                    f.write(f"Grasp Score: {score}\n")
                f.write(f"Failure Type: Grasp execution failed\n")
            print(f"Saved failure grasp info: {grasp_info_path}")
        
        # Save comprehensive failure summary
        failure_summary_path = os.path.join(failure_vis_dir, 'failure_summary.txt')
        with open(failure_summary_path, 'w') as f:
            f.write(f"=== COMPREHENSIVE FAILURE CASE ANALYSIS ===\n")
            f.write(f"Scene ID: {scene_name}\n")
            f.write(f"Model Type: {model_type}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Failure Reason: {'No valid grasp found' if grasp is None else 'Grasp execution failed'}\n")
            if grasp is not None:
                f.write(f"Attempted Grasp Position: {grasp.pose.translation}\n")
                f.write(f"Attempted Grasp Score: {score if score is not None else 'N/A'}\n")
            f.write(f"\nVisualization Components:\n")
            f.write(f"1. Input Scene: input_scene.ply\n")
            f.write(f"2. GT Target: gt_target.ply\n")
            if model_type == 'targo':
                f.write(f"3. Target Completion: target_completion.ply\n")
                f.write(f"4. Completion Metrics: completion_metrics.txt\n")
            f.write(f"5. RGB Image: rgb.png\n")
            f.write(f"6. Affordance Map: affordance_mesh.obj\n")
            f.write(f"7. Affordance Info: affordance_info.txt\n")
            if grasp is not None:
                f.write(f"8. Grasp Info: grasp_info.txt\n")
        
        print(f"Failure case visualization completed for {scene_name}")
        
    except Exception as e:
        print(f"Error visualizing failure case {scene_name}: {e}")
        import traceback
        traceback.print_exc()


def create_affordance_mesh(state, grasp_plan_fn, model_type):
    """
    Create affordance mesh for visualization
    
    Args:
        state: Simulation state
        grasp_plan_fn: Grasp planning function
        model_type: Type of model ('targo', 'vgn', etc.)
    
    Returns:
        trimesh.Trimesh: Affordance mesh or None if failed
    """
    try:
        import torch
        
        if model_type == 'vgn':
            # VGN affordance generation
            qual_vol, rot_vol, width_vol = grasp_plan_fn(state, None)
            
            # Convert to numpy and create mesh
            qual_vol_np = qual_vol.detach().cpu().numpy()
            
            # Create mesh from volume
            from skimage import measure
            vertices, faces, _, _ = measure.marching_cubes(qual_vol_np, level=0.5)
            
            # Scale vertices to world coordinates
            voxel_size = state.tsdf.voxel_size
            vertices = vertices * voxel_size - 0.5
            
            # Create colored mesh based on affordance values
            affordance_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Color mesh based on affordance values
            face_colors = np.ones((len(faces), 4)) * [1.0, 0.96, 0.94, 1.0]  # Light pink
            affordance_mesh.visual.face_colors = face_colors
            
            return affordance_mesh
            
        elif model_type in ['targo', 'targo_full', 'targo_ptv3', 'ptv3_scene']:
            # TARGO affordance generation
            grasps, scores, _, _ = grasp_plan_fn(state, None)
            
            if len(grasps) > 0:
                # Create mesh from grasp positions
                positions = grasps[:, :3].detach().cpu().numpy()
                scores_np = scores.detach().cpu().numpy()
                
                # Create point cloud mesh
                affordance_mesh = trimesh.points.PointCloud(positions)
                
                # Convert to mesh with colored vertices
                vertices = positions
                faces = []
                for i in range(0, len(vertices), 3):
                    if i + 2 < len(vertices):
                        faces.append([i, i+1, i+2])
                
                if len(faces) > 0:
                    affordance_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                    
                    # Color based on scores
                    colors = np.ones((len(vertices), 4))
                    for i, score in enumerate(scores_np):
                        # Color from light pink to red based on score
                        intensity = min(1.0, max(0.0, score))
                        colors[i] = [1.0, 1.0 - intensity * 0.4, 0.94 - intensity * 0.4, 1.0]
                    
                    affordance_mesh.visual.vertex_colors = colors
                    return affordance_mesh
            
        return None
        
    except Exception as e:
        print(f"Error creating affordance mesh: {e}")
        return None


