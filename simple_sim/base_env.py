from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import DoorObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.controllers import load_composite_controller_config

def array_to_string(array):
    """
    Converts a numeric array into the string format in mujoco.

    Examples:
        [0, 1, 2] => "0 1 2"

    Args:
        array (n-array): Array to convert to a string

    Returns:
        str: String equivalent of @array
    """
    return " ".join(["{}".format(x) for x in array])

class SimpleEnv(ManipulationEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        use_latch=True,
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
    ):
        # settings for table top (hardcoded since it's not an essential part of the environment)
        self.table_full_size = (0.8, 0.3, 0.05)
        self.table_offset = (-0.2, -0.35, 0.8)

        # reward configuration
        self.use_latch = use_latch
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):
        reward = 0.0

        # # sparse completion reward
        # if self._check_success():
        #     reward = 1.0

        # # else, we consider only the case if we're using shaped rewards
        # elif self.reward_shaping:
        #     # Add reaching component
        #     dist = np.linalg.norm(self._gripper_to_handle)
        #     reaching_reward = 0.25 * (1 - np.tanh(10.0 * dist))
        #     reward += reaching_reward
        #     # Add rotating component if we're using a locked door
        #     if self.use_latch:
        #         handle_qpos = self.sim.data.qpos[self.handle_qpos_addr]
        #         reward += np.clip(0.25 * np.abs(handle_qpos / (0.5 * np.pi)), -0.25, 0.25)

        # # Scale reward if requested
        # if self.reward_scale is not None:
        #     reward *= self.reward_scale / 1.0

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        self.robots[0].set_base_xpos([0.01652802 , -0.254359790, 1.12428212])
        quaternion = np.array([0.50203721, 0.77966656,  0.31712895, -0.19876602])
        self.robots[0]._elements["root_body"].set("quat", array_to_string(quaternion))


        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5986131746834771, -4.392035683362857e-09, 1.5903500240372423],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349],
        )

        # initialize objects of interest
        self.door = DoorObject(
            name="Door",
            friction=0.0,
            damping=0.1,
            lock=self.use_latch,
        )

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.door)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.door,
                x_range=[0.07, 0.09],
                y_range=[-0.01, 0.01],
                rotation=(-np.pi / 2.0 - 0.25, -np.pi / 2.0),
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.door,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.object_body_ids = dict()
        self.object_body_ids["door"] = self.sim.model.body_name2id(self.door.door_body)
        self.object_body_ids["frame"] = self.sim.model.body_name2id(self.door.frame_body)
        self.object_body_ids["latch"] = self.sim.model.body_name2id(self.door.latch_body)
        self.door_handle_site_id = self.sim.model.site_name2id(self.door.important_sites["handle"])
        self.hinge_qpos_addr = self.sim.model.get_joint_qpos_addr(self.door.joints[0])
        if self.use_latch:
            self.handle_qpos_addr = self.sim.model.get_joint_qpos_addr(self.door.joints[1])

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            modality = "object"

            # Define sensor callbacks
            @sensor(modality=modality)
            def door_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.object_body_ids["door"]])

            @sensor(modality=modality)
            def handle_pos(obs_cache):
                return self._handle_xpos

            @sensor(modality=modality)
            def hinge_qpos(obs_cache):
                return np.array([self.sim.data.qpos[self.hinge_qpos_addr]])

            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])
            sensors = [door_pos, handle_pos, hinge_qpos]

            # create variable number of sensors for each arm representing distance from specified arm to handle/door
            sensors += [
                self._get_obj_eef_sensor(full_pf, "door_pos", f"door_to_{arm_pf}eef_pos", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]
            sensors += [
                self._get_obj_eef_sensor(full_pf, "handle_pos", f"handle_to_{arm_pf}eef_pos", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]

            names = [s.__name__ for s in sensors]

            # Also append handle qpos if we're using a locked door version with rotatable handle
            if self.use_latch:

                @sensor(modality=modality)
                def handle_qpos(obs_cache):
                    return np.array([self.sim.data.qpos[self.handle_qpos_addr]])

                sensors.append(handle_qpos)
                names.append("handle_qpos")

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # We know we're only setting a single object (the door), so specifically set its pose
            door_pos, door_quat, _ = object_placements[self.door.name]
            door_body_id = self.sim.model.body_name2id(self.door.root_body)
            self.sim.model.body_pos[door_body_id] = door_pos
            self.sim.model.body_quat[door_body_id] = door_quat

    def _check_success(self):
        """
        Check if door has been opened.

        Returns:
            bool: True if door has been opened
        """
        hinge_qpos = self.sim.data.qpos[self.hinge_qpos_addr]
        return hinge_qpos > 0.3

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the door handle.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the door handle
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper, target=self.door.important_sites["handle"], target_type="site"
            )
    @property
    def _handle_xpos(self):
        """
        Grabs the position of the door handle handle.

        Returns:
            np.array: Door handle (x,y,z)
        """
        return self.sim.data.site_xpos[self.door_handle_site_id]

    @property
    def _gripper_to_handle(self):
        """
        Calculates distance from the gripper to the door handle.
        If multiple arms, returns the minimum distance.

        Returns:
            np.array: (x,y,z) distance between handle and eef
        """
        dists = []
        for arm in self.robots[0].arms:
            diff = self._handle_xpos - np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]])
            dists.append(np.linalg.norm(diff))
        return min(dists)

if __name__ == "__main__":
    env = SimpleEnv(
        robots='UR5e',
        has_renderer=True,  # no on-screen renderer
        has_offscreen_renderer=False,  # no off-screen renderer
        ignore_done=True,
        use_camera_obs=False,  # no camera observations
        control_freq=20,
        renderer="mjviewer",)
    env.reset()
    dof = env.action_dim
    for t in range(11000):
        action = np.random.randn(dof)
        env.step(action)
        env.render()