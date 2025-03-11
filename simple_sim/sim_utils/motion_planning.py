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
        
if __name__ == "__main__":
    motion_planning = MotionPlanning()
    start_pose = np.array([0, 0, 0])
    goal_pose = np.array([1, 1, 1])
    path = motion_planning.plan(start_pose, goal_pose, 5)
    relative_trajectory = np.diff(path, axis=0, prepend=start_pose.reshape(1, -1))[1:]

    print(path)