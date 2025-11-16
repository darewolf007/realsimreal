import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import xml.etree.ElementTree as ET
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string, string_to_array, xml_path_completion
from robosuite.utils.mjcf_utils import (
    ENVIRONMENT_COLLISION_COLOR,
    array_to_string,
    find_elements,
    new_body,
    new_element,
    new_geom,
    new_joint,
    recolor_collision_geoms,
    string_to_array,
)
from simple_sim.know_obj.xml_obj import (
    CanObject, KettleObject, 
    CupObject, BananaObject, 
    AppleObject, BowlObject, 
    MustardObject, MarkerObject,
    PenHolderObject, BlueCanObject,
    StopButtonObject
)
from simple_sim.sim_utils.external_area import ExternalArea
from simple_sim.sim_utils.sim_util import add_noise_to_rotation_z
KNOW_OBJ = {"can": CanObject,
             "kettle": KettleObject,
              "cup": CupObject,
              "banana": BananaObject,
              "apple": AppleObject,
              "bowl": BowlObject,
              "mustard": MustardObject,
              "marker": MarkerObject,
              "pen_holder": PenHolderObject,
              "blue_can": BlueCanObject,
              "stop_button": StopButtonObject
              }

class SimpleEnv(ManipulationEnv):
    def __init__(
        self,
        robots,
        env_info,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        use_camera_obs=True,
        reward_scale=1.0,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="sceneview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="frontview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=True,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
        schedule_random = False,
    ):
        self.env_info = env_info
        self.reward_scale = reward_scale
        self.scene_objects = []
        self.obj_body_id = {}
        self.schedule_random = schedule_random
        self.reset_time = 0
        self.init_scene_info()
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

    def init_scene_info(self):
        self.init_qpos = np.array(self.env_info['robot_init_qpos'])
        scene_dict = self.env_info['obj_info']
        for i, label in enumerate(scene_dict["labels"]):
            if label == "table":
                continue
            if label in KNOW_OBJ.keys():
                obj = KNOW_OBJ[label](name=label)
                self.scene_objects.append(obj)

    def reward(self, action=None):
        reward = 0.0
        return reward

    def _load_model(self):
        super()._load_model()
        robot_xpos = self.env_info['robot_base_pose'][:3]
        robot_quat = self.env_info['robot_base_pose'][3:]
        self.robots[0].robot_model.set_base_xpos(robot_xpos.tolist())
        self.robots[0].robot_model._elements["root_body"].set("quat", array_to_string(robot_quat))
        self.robots[0].init_qpos = self.init_qpos
        base_path = os.path.dirname(os.path.realpath(__file__))
        obj_xml = os.path.join(base_path, "../asset/external_area.xml")
        mujoco_arena = ExternalArea(xml_path_completion(obj_xml))
        self.mujoco_arena = mujoco_arena
        for view_name, view_info in self.env_info['camera_info'].items():
            mujoco_arena.set_camera(
                camera_name=view_name,
                pos=view_info["pos"],
                quat=view_info["quat"],
                fovy = np.array([76.22424707826806])
            )
            mujoco_arena.set_camera(
                camera_name="moveview",
                pos=view_info["pos"],
                quat=view_info["quat"],
                fovy = np.array([76.22424707826806])
            )
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.scene_objects,
        )
        self.modify_xml_for_camera_movement(self.model.tree, "moveview")
    
    def object_initializer(self):
        for obj in self.scene_objects:
            idx = self.env_info['obj_info']['labels'].index(obj.name)
            obj_position = np.array(self.env_info['obj_info']['poses'][idx][:3].copy())
            obj_quat = np.array(self.env_info['obj_info']['poses'][idx][3:].copy())
            if self.env_info['init_noise']:
                obj_position[:2] += np.random.uniform(self.env_info['init_translation_noise_bounds'][0], self.env_info['init_translation_noise_bounds'][1], size=2)
                obj_quat = add_noise_to_rotation_z(obj_quat, self.env_info['init_rotation_noise_bounds'])
            if self.env_info['obj_info']['grasp_obj'][idx]:
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([obj_position, obj_quat]))
            else:
                self.sim.model.body_pos[self.obj_body_id[obj.name]] = obj_position
                self.sim.model.body_quat[self.obj_body_id[obj.name]] = obj_quat

    def get_mujoco_model(self):
        return self.model.get_model(mode="mujoco")

    def _setup_references(self):
        super()._setup_references()
        for obj in self.scene_objects:
            self.obj_body_id[obj.name] = self.sim.model.body_name2id(obj.root_body)

    def _setup_observables(self):
        observables = super()._setup_observables()
        return observables

    def _reset_internal(self):
        self.init_qpos = np.array(self.env_info['robot_init_qpos'])
        if self.schedule_random:
            self.reset_time += 1
            self.schedule_scale = self.get_schedule_scale()
            robot_xpos = self.env_info['robot_base_pose'][:3]
            robot_quat = self.env_info['robot_base_pose'][3:]
            self.robots[0].robot_model.set_base_xpos(robot_xpos.tolist())
            self.robots[0].robot_model._elements["root_body"].set("quat", array_to_string(robot_quat))
            self.robots[0].init_qpos = self.init_qpos + self.schedule_scale * (np.random.uniform(low=-0.01, high=0.01, size=self.init_qpos.shape))
            # self.init_qpos = self.robots[0].init_qpos
            base_path = os.path.dirname(os.path.realpath(__file__))
            obj_xml = os.path.join(base_path, "../asset/external_area.xml")
            mujoco_arena = ExternalArea(xml_path_completion(obj_xml))
            for view_name, view_info in self.env_info['camera_info'].items():
                mujoco_arena.set_camera(
                    camera_name=view_name,
                    pos=view_info["pos"],
                    quat=view_info["quat"],
                    fovy = np.array([76.22424707826806])
                )
            # self.model = ManipulationTask(
            #     mujoco_arena=mujoco_arena,
            #     mujoco_robots=[robot.robot_model for robot in self.robots],
            #     mujoco_objects=self.scene_objects,
            # )
            super()._reset_internal()
            # TODO: rewrite move camera base on fix scene view
            self.sim.data.set_mocap_pos(self.mover_camera_name, view_info["pos"]  + self.schedule_scale * (np.random.uniform(low=-0.003, high=0.003, size=view_info["pos"].shape)))
            self.sim.data.set_mocap_quat(self.mover_camera_name, view_info["quat"]  + self.schedule_scale * (np.random.uniform(low=-0.001, high=0.001, size=view_info["quat"].shape)))
            for obj in self.scene_objects:
                idx = self.env_info['obj_info']['labels'].index(obj.name)
                obj_position = np.array(self.env_info['obj_info']['poses'][idx][:3].copy())
                obj_quat = np.array(self.env_info['obj_info']['poses'][idx][3:].copy())
                if self.env_info['init_noise']:
                    obj_position[:2] += self.schedule_scale * np.random.uniform(self.env_info['init_translation_noise_bounds'][0], self.env_info['init_translation_noise_bounds'][1], size=2)
                    obj_quat = add_noise_to_rotation_z(obj_quat, self.schedule_scale * np.array(self.env_info['init_rotation_noise_bounds']))
                if self.env_info['obj_info']['grasp_obj'][idx]:
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([obj_position, obj_quat]))
                else:
                    self.sim.model.body_pos[self.obj_body_id[obj.name]] = obj_position
                    self.sim.model.body_quat[self.obj_body_id[obj.name]] = obj_quat
        else:
            super()._reset_internal()
            self.object_initializer()
        # for obj in self.scene_objects:
        #     idx = self.env_info['obj_info']['labels'].index(obj.name)
        #     if self.env_info['obj_info']['grasp_obj'][idx]:
        #         self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([self.env_info['obj_info']['poses'][idx][:3], self.env_info['obj_info']['poses'][idx][3:]]))
        #     else:
        #         self.sim.model.body_pos[self.obj_body_id[obj.name]] = self.env_info['obj_info']['poses'][idx][:3]
        #         self.sim.model.body_quat[self.obj_body_id[obj.name]] = self.env_info['obj_info']['poses'][idx][3:]
            # item_indices = [index for index, value in enumerate(self.env_info['obj_info']['labels']) if value == obj.name]

    def _check_success(self):
        return 0

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)

    def update_env_info(self, new_env_info):
        self.env_info = new_env_info

    def robot_collisions(self):
        if self.check_contact(self.robots[0].robot_model, "ur_table"):
            return True
        elif self.robots[0].check_q_limits():
            # this is too sensitive, so we just return False
            return False
        else:
            return False
        
    def get_schedule_scale(self):
        scale = 1.0 - np.exp(-0.0001 * self.reset_time)
        return np.clip(scale, 0.0, 1.0)

    def modify_xml_for_camera_movement(self, tree, camera_name):
        wb = tree.find("worldbody")
        assert wb is not None, "Cannot find <worldbody> in XML tree."
        self.mover_camera_name = f"{camera_name}_cameramover"
        camera_elem = None
        for camera in wb.findall("camera"):
            if camera.get("name") == camera_name:
                camera_elem = camera
                break
        assert camera_elem is not None, f"Camera '{camera_name}' not found in <worldbody>."
        mocap = ET.SubElement(wb, "body")
        mocap.set("name", self.mover_camera_name)
        mocap.set("mocap", "true")
        mocap.set("pos", camera.get("pos"))
        mocap.set("quat", camera.get("quat"))
        new_camera = ET.SubElement(mocap, "camera")
        new_camera.set("mode", "fixed")
        new_camera.set("name", camera.get("name"))
        new_camera.set("pos", "0 0 0")
        wb.remove(camera_elem)