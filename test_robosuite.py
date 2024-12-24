from robosuite.models.robots import UR5e
from robosuite.models.grippers import gripper_factory
import robosuite.macros as macros
from robosuite.models.base import MujocoXML
from robosuite.utils.mjcf_utils import convert_to_string, find_elements, xml_path_completion
import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class UR5e(ManipulatorModel):
    """
    UR5e is a sleek and elegant new robot created by Universal Robots

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    arms = ["right"]

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/ur5e/robot.xml"), idn=idn)

    @property
    def default_base(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return {"right": "Robotiq85Gripper"}

    @property
    def default_controller_config(self):
        return {"right": "default_ur5e"}

    @property
    def init_qpos(self):
        return np.array([-1.169, -1.19, 1.332, -1.699, -1.518, 1.2])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (0, 0, 0),
            "empty": (0, 0, 0),
            "table": (0, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"

class MujocoWorldBase(MujocoXML):
    """Base class to inherit all mujoco worlds from."""

    def __init__(self):
        super().__init__(xml_path_completion("/home/haowen/hw_sim_real/depth_to_mujoco/test.xml"))
        # Modify the simulation timestep to be the requested value
        options = find_elements(root=self.root, tags="option", attribs=None, return_first=True)
        options.set("timestep", convert_to_string(macros.SIMULATION_TIMESTEP))
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

import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Robotiq140GripperBase(GripperModel):
    """
    Gripper with 140mm Jaw width from Robotiq (has two fingers).

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance

    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/robotiq_gripper_140.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.012, 0.065, 0.065, -0.012, 0.065, 0.065])

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                "left_outer_finger_collision",
                "left_inner_finger_collision",
                "left_fingertip_collision",
                "left_fingerpad_collision",
            ],
            "right_finger": [
                "right_outer_finger_collision",
                "right_inner_finger_collision",
                "right_fingertip_collision",
                "right_fingerpad_collision",
            ],
            "left_fingerpad": ["left_fingerpad_collision"],
            "right_fingerpad": ["right_fingerpad_collision"],
        }


class Robotiq140Gripper(Robotiq140GripperBase):
    """
    Modifies Robotiq140GripperBase to only take one action.
    """

    def format_action(self, action):
        """
        Maps continuous action into binary output
        -1 => open, 1 => closed

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == 1
        self.current_action = np.clip(
            self.current_action + np.array([1.0, -1.0]) * self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action

    @property
    def speed(self):
        return 0.2

    @property
    def dof(self):
        return 1


world = MujocoWorldBase()
mujoco_robot = UR5e()
gripper = gripper_factory('Robotiq140Gripper')
mujoco_robot.add_gripper(gripper)
# Translation: [ 0.01652802 -0.25435979  1.12428212]
# [-2.13639129 -0.67944774 -2.75273527]
mujoco_robot.set_base_xpos([0.01652802 , -0.254359790, 1.12428212])
# [-0.31712895  0.77966656 -0.50203721 -0.19876602]
quaternion = np.array([0.50203721, 0.77966656,  0.31712895, -0.19876602])
mujoco_robot._elements["root_body"].set("quat", array_to_string(quaternion))
world.merge(mujoco_robot)
model = world.get_model(mode="mujoco")
import mujoco
from robosuite.utils.binding_utils import MjRenderContextOffscreen, MjSim
data = mujoco.MjData(model)
data.qpos[:6] = mujoco_robot.init_qpos
# mujoco.viewer.launch(model, data)
sim = MjSim(model)
sim_state = sim.get_state()
sim_state.qpos[:6] = mujoco_robot.init_qpos
sim.set_state(sim_state)
from robosuite.utils import OpenCVRenderer
viewer = OpenCVRenderer(sim)
render_context = MjRenderContextOffscreen(sim, device_id=2)
sim.add_render_context(render_context)
# viewer = MjViewer(sim)
while data.time < 1:
    sim_state = sim.get_state()
    sim_state.qpos[:6] = mujoco_robot.init_qpos
    sim_state.qpos[6:] = np.zeros(6)
    sim.set_state(sim_state)
    sim.step()
    print(data.qpos)
    viewer.render()
    
# mujoco.mj_step(model, data)
# mujoco.viewer.launch(model, data)

# while data.time < 1:
#     mujoco.mj_step(model, data)