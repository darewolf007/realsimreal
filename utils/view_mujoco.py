from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path("/home/haowen/hw_mine/Real_Sim_Real/simple_sim/asset/external_area.xml")
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        mujoco.mj_forward(model, data)
        viewer.sync()
        # Initialize the mocap target at the end-effector site.
        rate = RateLimiter(frequency=5.0, warn=False)
        while viewer.is_running():
            for _ in range(10):
                mujoco.mj_step(model, data)
            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()

