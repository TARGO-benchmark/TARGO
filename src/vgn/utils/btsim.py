import os
# Set EGL platform for headless OpenGL rendering (must be before pybullet import)
os.environ["PYOPENGL_PLATFORM"] = "egl"

import time
import pickle
import numpy as np
import pybullet
from pybullet_utils import bullet_client

from vgn.utils.transform import Rotation, Transform
from vgn.utils.saver import get_mesh_pose_dict_from_world

assert pybullet.isNumpyEnabled(), "Pybullet needs to be built with NumPy"


class BtWorld(object):
    """Interface to a PyBullet physics server.

    Attributes:
        dt: Time step of the physics simulation.
        rtf: Real time factor. If negative, the simulation is run as fast as possible.
        sim_time: Virtual time elpased since the last simulation reset.
    """

    def __init__(self, gui=True, save_dir=None, save_freq=8, egl_mode=False):
        # Force EGL mode for headless rendering when gui=False
        if not gui:
            egl_mode = True
            
        if egl_mode or not gui:
            # Use EGL for headless OpenGL rendering (same quality as GUI)
            self.p = bullet_client.BulletClient(pybullet.DIRECT)
            
            # Load EGL renderer plugin for hardware-accelerated rendering
            plugin_status = self.p.loadPlugin("eglRendererPlugin")
            if plugin_status >= 0:
                print("✓ EGL renderer plugin loaded successfully - headless OpenGL rendering enabled")
                self.use_hardware_renderer = True
            else:
                print("⚠ Warning: EGL renderer plugin failed to load, falling back to software renderer")
                print("  This will result in poor video quality with black background")
                self.use_hardware_renderer = False
        else:
            # Original GUI mode
            connection_mode = pybullet.GUI if gui else pybullet.DIRECT
            self.p = bullet_client.BulletClient(connection_mode)
            self.use_hardware_renderer = gui  # GUI mode uses hardware rendering

        self.gui = gui
        self.egl_mode = egl_mode
        self.dt = 1.0 / 240.0
        self.solver_iterations = 150
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.sim_step = 0

        self.reset()

    def set_gravity(self, gravity):
        self.p.setGravity(*gravity)

    def load_urdf(self, urdf_path, pose, scale=1.0):
        body = Body.from_urdf(self.p, urdf_path, pose, scale)
        self.bodies[body.uid] = body
        return body

    def remove_body(self, body):
        self.p.removeBody(body.uid)
        del self.bodies[body.uid]

    def add_constraint(self, *argv, **kwargs):
        """See `Constraint` below."""
        constraint = Constraint(self.p, *argv, **kwargs)
        return constraint

    def add_camera(self, intrinsic, near, far):
        camera = Camera(self.p, intrinsic, near, far, use_hardware_renderer=self.use_hardware_renderer)
        return camera
    
    def get_contacts_valid(self, bodyA, tgt_id):
    #   def get_contacts(self, bodyA):
        points = self.p.getContactPoints(bodyA.uid)
        contacts = []
        # assert
        contact_object = []
        for point in points:
            contact_object.append(point[2])
            # assert point[2] == tgt_id
            contact = Contact(
                bodyA=self.bodies[point[1]],
                bodyB=self.bodies[point[2]],
                point=point[5],
                normal=point[7],
                depth=point[8],
                force=point[9],
            )
            contacts.append(contact)
        # if len(contacts) > 0:
        #     assert tgt_id in contact_object
        return contacts

    def get_contacts(self, bodyA):
        points = self.p.getContactPoints(bodyA.uid)
        contacts = []
        for point in points:
            contact = Contact(
                bodyA=self.bodies[point[1]],
                bodyB=self.bodies[point[2]],
                point=point[5],
                normal=point[7],
                depth=point[8],
                force=point[9],
            )
            contacts.append(contact)
        return contacts

    def reset(self):
        self.p.resetSimulation()
        self.p.setPhysicsEngineParameter(
            fixedTimeStep=self.dt, numSolverIterations=self.solver_iterations
        )
        self.bodies = {}
        self.sim_time = 0.0

    def step(self):
        self.p.stepSimulation()
        
        
        if self.gui:
            time.sleep(self.dt)
        if self.save_dir:
            if self.sim_step % self.save_freq == 0:
                mesh_pose_dict = get_mesh_pose_dict_from_world(self, self.p._client)
                with open(os.path.join(self.save_dir, f'{self.sim_step:08d}.pkl'), 'wb') as f:
                    pickle.dump(mesh_pose_dict, f)

        self.sim_time += self.dt
        self.sim_step += 1

    def save_state(self):
        return self.p.saveState()

    def restore_state(self, state_uid):
        self.p.restoreState(stateId=state_uid)

    def close(self):
        self.p.disconnect()

    def start_video_recording(self, filename, video_path):
        """Start video recording using OpenCV for manual video capture"""
        import cv2
        import numpy as np
        
        os.makedirs(video_path, exist_ok=True)
        video_file = os.path.join(video_path, f"{filename}.mp4")
        
        # Reduce video resolution to improve performance
        self.width, self.height = 640, 480  # Lower resolution
        self.frames = []
        self.video_file = video_file
        self.recording = True
        self.frame_count = 0  # Frame counter for skipping frames
        self.capture_interval = 3  # Capture every nth frame
        
        print(f"Starting OpenCV video recording: {video_file} (optimized mode)")
        return 1  # Return virtual ID
    
    def capture_frame(self):
        """Capture frames during simulation steps, using frame intervals to reduce capture frequency"""
        if not hasattr(self, 'recording') or not self.recording:
            return
        
        # Use interval sampling to reduce frame count    
        self.frame_count += 1
        if self.frame_count % self.capture_interval != 0:
            return
            
        # Set camera view
        viewMatrix = self.p.computeViewMatrix(
            cameraEyePosition=[0.5, -0.2, 0.5],
            cameraTargetPosition=[0.15, 0.15, 0.15],
            cameraUpVector=[0, 0, 1]
        )
        projectionMatrix = self.p.computeProjectionMatrixFOV(
            fov=60, aspect=self.width/self.height, nearVal=0.1, farVal=100
        )
        
        # Choose renderer based on hardware availability
        if self.use_hardware_renderer:
            # Use OpenGL hardware renderer for high-quality rendering (same as GUI)
            renderer = self.p.ER_BULLET_HARDWARE_OPENGL
            # Only print renderer info every 100 frames to reduce output spam
            if self.frame_count % 100 == 1:
                print(f"📹 Frame {self.frame_count}: Using OpenGL hardware renderer")
        else:
            # Fall back to software renderer
            renderer = self.p.ER_TINY_RENDERER
            # Only print renderer info every 100 frames to reduce output spam
            if self.frame_count % 100 == 1:
                print(f"📹 Frame {self.frame_count}: Using software renderer")
        
        # Get image with selected renderer
        img = self.p.getCameraImage(
            self.width, self.height, viewMatrix, projectionMatrix,
            renderer=renderer
        )
        
        # Save image
        import numpy as np
        rgb_array = np.array(img[2], dtype=np.uint8).reshape(self.height, self.width, 4)
        rgb_array = rgb_array[:, :, :3]  # Remove alpha channel
        self.frames.append(rgb_array)
    
    def stop_video_recording(self, log_id):
        """End video recording and generate video file using more efficient encoder"""
        if not hasattr(self, 'recording') or not self.recording:
            print("❌ No active recording in progress")
            return
            
        print(f"🎬 Stopping video recording...")
        print(f"📊 Total frames captured: {len(self.frames)}")
        print(f"📁 Video file path: {self.video_file}")
        
        import cv2
        
        if len(self.frames) == 0:
            print("⚠️ Warning: No frames captured, video is empty")
            return
            
        # Create video - using H.264 encoding
        try:
            # First try using H.264 encoder
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            out = cv2.VideoWriter(self.video_file, fourcc, 30, (self.width, self.height))
            
            # Test if video writer is working properly
            if not out.isOpened():
                raise Exception("H264 encoder not available")
                
            print("✓ Using H.264 encoder")
                
        except Exception as e:
            print(f"⚠️ H264 encoder not available: {e}, trying MP4V...")
            # Fall back to MP4V encoder
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.video_file, fourcc, 30, (self.width, self.height))
            
            if not out.isOpened():
                print("❌ Error: Could not open video writer")
                return
                
            print("✓ Using MP4V encoder")
        
        # Reduce frame count to speed up processing
        total_frames = len(self.frames)
        if total_frames > 150:  # If too many frames, reduce further
            step = max(1, total_frames // 150)
            print(f"🔄 Too many frames ({total_frames}), reducing to ~150 frames with step {step}")
            selected_frames = self.frames[::step]
        else:
            selected_frames = self.frames
            
        print(f"🎞️ Processing {len(selected_frames)} frames...")
        
        # Write frames to video
        frame_count = 0
        for frame in selected_frames:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)
            frame_count += 1
            
            # Progress indicator every 50 frames
            if frame_count % 50 == 0:
                print(f"  Processed {frame_count}/{len(selected_frames)} frames...")
        
        out.release()
        self.recording = False
        self.frames = []
        
        # Verify video file was created
        import os
        if os.path.exists(self.video_file):
            file_size = os.path.getsize(self.video_file)
            print(f"✅ Video recording completed successfully!")
            print(f"📁 File: {self.video_file}")
            print(f"📏 Size: {file_size} bytes")
        else:
            print(f"❌ Error: Video file was not created at {self.video_file}")
        
        return self.video_file


class Body(object):
    """Interface to a multibody simulated in PyBullet.

    Attributes:
        uid: The unique id of the body within the physics server.
        name: The name of the body.
        joints: A dict mapping joint names to Joint objects.
        links: A dict mapping link names to Link objects.
    """

    def __init__(self, physics_client, body_uid, scale, urdf_path=None):
        self.p = physics_client
        self.uid = body_uid
        self.scale = scale
        self.urdf_path = urdf_path
        self.name = self.p.getBodyInfo(self.uid)[1].decode("utf-8")
        self.joints, self.links = {}, {}
        for i in range(self.p.getNumJoints(self.uid)):
            joint_info = self.p.getJointInfo(self.uid, i)
            joint_name = joint_info[1].decode("utf8")
            self.joints[joint_name] = Joint(self.p, self.uid, i)
            link_name = joint_info[12].decode("utf8")
            self.links[link_name] = Link(self.p, self.uid, i)

    @classmethod
    def from_urdf(cls, physics_client, urdf_path, pose, scale):
        body_uid = physics_client.loadURDF(
            str(urdf_path),
            pose.translation,
            pose.rotation.as_quat(),
            globalScaling=scale,
        )
        # self.urdf_path = urdf_path
        return cls(physics_client, body_uid, scale, urdf_path)

    def get_pose(self):
        pos, ori = self.p.getBasePositionAndOrientation(self.uid)
        return Transform(Rotation.from_quat(ori), np.asarray(pos))

    def set_pose(self, pose):
        self.p.resetBasePositionAndOrientation(
            self.uid, pose.translation, pose.rotation.as_quat()
        )
    def set_color(self, link_index, rgba_color):
        """
        Sets the color of a specific link in this body.
        
        Args:
            link_index: The index of the link to change the color of. Use -1 for the base.
            rgba_color: A tuple of 4 floats representing the RGBA color.
        """
        self.p.changeVisualShape(self.uid, link_index, rgbaColor=rgba_color)

    def get_velocity(self):
        linear, angular = self.p.getBaseVelocity(self.uid)
        return linear, angular


class Link(object):
    """Interface to a link simulated in Pybullet.

    Attributes:
        link_index: The index of the joint.
    """

    def __init__(self, physics_client, body_uid, link_index):
        self.p = physics_client
        self.body_uid = body_uid
        self.link_index = link_index

    def get_pose(self):
        link_state = self.p.getLinkState(self.body_uid, self.link_index)
        pos, ori = link_state[0], link_state[1]
        return Transform(Rotation.from_quat(ori), pos)


class Joint(object):
    """Interface to a joint simulated in PyBullet.

    Attributes:
        joint_index: The index of the joint.
        lower_limit: Lower position limit of the joint.
        upper_limit: Upper position limit of the joint.
        effort: The maximum joint effort.
    """

    def __init__(self, physics_client, body_uid, joint_index):
        self.p = physics_client
        self.body_uid = body_uid
        self.joint_index = joint_index

        joint_info = self.p.getJointInfo(body_uid, joint_index)
        self.lower_limit = joint_info[8]
        self.upper_limit = joint_info[9]
        self.effort = joint_info[10]

    def get_position(self):
        joint_state = self.p.getJointState(self.body_uid, self.joint_index)
        return joint_state[0]

    def set_position(self, position, kinematics=False):
        if kinematics:
            self.p.resetJointState(self.body_uid, self.joint_index, position)
        self.p.setJointMotorControl2(
            self.body_uid,
            self.joint_index,
            pybullet.POSITION_CONTROL,
            targetPosition=position,
            force=self.effort,
        )


class Constraint(object):
    """Interface to a constraint in PyBullet.

    Attributes:
        uid: The unique id of the constraint within the physics server.
    """

    def __init__(
        self,
        physics_client,
        parent,
        parent_link,
        child,
        child_link,
        joint_type,
        joint_axis,
        parent_frame,
        child_frame,
    ):
        """
        Create a new constraint between links of bodies.

        Args:
            parent:
            parent_link: None for the base.
            child: None for a fixed frame in world coordinates.

        """
        self.p = physics_client
        parent_body_uid = parent.uid
        parent_link_index = parent_link.link_index if parent_link else -1
        child_body_uid = child.uid if child else -1
        child_link_index = child_link.link_index if child_link else -1

        self.uid = self.p.createConstraint(
            parentBodyUniqueId=parent_body_uid,
            parentLinkIndex=parent_link_index,
            childBodyUniqueId=child_body_uid,
            childLinkIndex=child_link_index,
            jointType=joint_type,
            jointAxis=joint_axis,
            parentFramePosition=parent_frame.translation,
            parentFrameOrientation=parent_frame.rotation.as_quat(),
            childFramePosition=child_frame.translation,
            childFrameOrientation=child_frame.rotation.as_quat(),
        )

    def change(self, **kwargs):
        self.p.changeConstraint(self.uid, **kwargs)


class Contact(object):
    """Contact point between two multibodies.

    Attributes:
        point: Contact point.
        normal: Normal vector from ... to ...
        depth: Penetration depth
        force: Contact force acting on body ...
    """

    def __init__(self, bodyA, bodyB, point, normal, depth, force):
        self.bodyA = bodyA
        self.bodyB = bodyB
        self.point = point
        self.normal = normal
        self.depth = depth
        self.force = force


class Camera(object):
    """Virtual RGB-D camera based on the PyBullet camera interface.

    Attributes:
        intrinsic: The camera intrinsic parameters.
    """

    def __init__(self, physics_client, intrinsic, near, far, use_hardware_renderer=True):
        self.intrinsic = intrinsic
        self.near = near
        self.far = far
        self.proj_matrix = _build_projection_matrix(intrinsic, near, far)
        self.p = physics_client
        self.use_hardware_renderer = use_hardware_renderer

    def render(self, extrinsic):
        """Render synthetic RGB and depth images.

        Args:
            extrinsic: Extrinsic parameters, T_cam_ref.
        """
        # Construct OpenGL compatible view and projection matrices.
        gl_view_matrix = extrinsic.as_matrix()
        gl_view_matrix[2, :] *= -1  # flip the Z axis
        gl_view_matrix = gl_view_matrix.flatten(order="F")
        gl_proj_matrix = self.proj_matrix.flatten(order="F")

        # Choose renderer based on hardware availability
        renderer = pybullet.ER_BULLET_HARDWARE_OPENGL if self.use_hardware_renderer else pybullet.ER_TINY_RENDERER

        result = self.p.getCameraImage(
            width=self.intrinsic.width,
            height=self.intrinsic.height,
            viewMatrix=gl_view_matrix,
            projectionMatrix=gl_proj_matrix,
            renderer=renderer,
        )

        rgb, z_buffer = result[2][:, :, :3], result[3]
        depth = (
            1.0 * self.far * self.near / (self.far - (self.far - self.near) * z_buffer)
        )
        return rgb, depth

    def render_with_seg(self, extrinsic, segmentation=True):
        """Render synthetic RGB and depth images.

        Args:
            extrinsic: Extrinsic parameters, T_cam_ref.
            segmentation: Boolean flag indicating if segmentation mask is required.
        """
        # Construct OpenGL compatible view and projection matrices.
        gl_view_matrix = extrinsic.as_matrix()
        gl_view_matrix[2, :] *= -1  # flip the Z axis
        gl_view_matrix = gl_view_matrix.flatten(order="F")
        gl_proj_matrix = self.proj_matrix.flatten(order="F")

        # Choose renderer based on hardware availability
        # renderer = pybullet.ER_BULLET_HARDWARE_OPENGL if self.use_hardware_renderer else pybullet.ER_TINY_RENDERER

        result = self.p.getCameraImage(
            width=self.intrinsic.width,
            height=self.intrinsic.height,
            viewMatrix=gl_view_matrix,
            projectionMatrix=gl_proj_matrix,
            renderer=pybullet.ER_TINY_RENDERER,
        )
        w, h, rgba, depth, seg = result
        rgba = np.array(rgba, dtype=np.uint8).reshape(h, w, 4)
        rgb = rgba[..., :3]
        depth = np.array(depth, dtype=np.float32)
        seg = np.array(seg, dtype=np.uint8).reshape(h, w)

        # rgb, z_buffer = result[2][:, :, :3], result[3]
        depth = (
            1.0 * self.far * self.near / (self.far - (self.far - self.near) * depth)
        )

        # seg = None
        if segmentation:
            # seg = result[4]
            seg = seg
        else:
            seg = None
        return rgb, depth, seg
    

    def render(self, extrinsic):
        """Render synthetic RGB and depth images.

        Args:
            extrinsic: Extrinsic parameters, T_cam_ref.
        """
        # Construct OpenGL compatible view and projection matrices.
        gl_view_matrix = extrinsic.as_matrix()
        gl_view_matrix[2, :] *= -1  # flip the Z axis
        gl_view_matrix = gl_view_matrix.flatten(order="F")
        gl_proj_matrix = self.proj_matrix.flatten(order="F")

        result = self.p.getCameraImage(
            width=self.intrinsic.width,
            height=self.intrinsic.height,
            viewMatrix=gl_view_matrix,
            projectionMatrix=gl_proj_matrix,
            renderer=pybullet.ER_TINY_RENDERER,
        )

        rgb, z_buffer = result[2][:, :, :3], result[3]
        depth = (
            1.0 * self.far * self.near / (self.far - (self.far - self.near) * z_buffer)
        )
        return rgb, depth


def _build_projection_matrix(intrinsic, near, far):
    perspective = np.array(
        [
            [intrinsic.fx, 0.0, -intrinsic.cx, 0.0],
            [0.0, intrinsic.fy, -intrinsic.cy, 0.0],
            [0.0, 0.0, near + far, near * far],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )
    ortho = _gl_ortho(0.0, intrinsic.width, intrinsic.height, 0.0, near, far)
    return np.matmul(ortho, perspective)


def _gl_ortho(left, right, bottom, top, near, far):
    ortho = np.diag(
        [2.0 / (right - left), 2.0 / (top - bottom), -2.0 / (far - near), 1.0]
    )
    ortho[0, 3] = -(right + left) / (right - left)
    ortho[1, 3] = -(top + bottom) / (top - bottom)
    ortho[2, 3] = -(far + near) / (far - near)
    return ortho
