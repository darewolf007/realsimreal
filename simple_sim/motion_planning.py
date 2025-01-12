import numpy as np

class MotionPlanning:
    def __init__(self, algorithm = "linear_interpolation"):
        self.algorithm = algorithm
        

    def linear_interpolation(self, start_pose, goal_pose, step):
        path = np.linspace(start_pose, goal_pose, step)
        return path
    
    def plan(self, start_pose, goal_pose, step = 20):
        if self.algorithm == "linear_interpolation":
            return self.linear_interpolation(start_pose, goal_pose, step)
        else:
            raise NotImplementedError("Algorithm not implemented")