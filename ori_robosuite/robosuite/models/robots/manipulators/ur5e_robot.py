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
        return {"right": "Robotiq140Gripper"}

    @property
    def default_controller_config(self):
        return {"right": "default_ur5e"}

    @property
    def init_qpos(self):
        return np.array(np.array([-1.169, -1.19, 1.332, -1.699, -1.518, -0.52]))

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
