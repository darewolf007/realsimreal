import numpy as np

class motion_planning:
    def __init__(self):
        pass

    def linear_interpolation(self, start_pose, goal_pose, step):
        path = np.linspace(start_pose, goal_pose, step)
        return path