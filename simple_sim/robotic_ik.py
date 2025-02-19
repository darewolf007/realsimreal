import mink
import numpy as np
import logging
from typing import Optional
import mujoco
from mink.configuration import Configuration

class ReturnConfig(Configuration):
    def new_check_limits(self, tol: float = 1e-6, safety_break: bool = True) -> None:
        for jnt in range(self.model.njnt):
            jnt_type = self.model.jnt_type[jnt]
            if (
                jnt_type == mujoco.mjtJoint.mjJNT_FREE
                or not self.model.jnt_limited[jnt]
            ):
                continue
            padr = self.model.jnt_qposadr[jnt]
            qval = self.q[padr]
            qmin = self.model.jnt_range[jnt, 0]
            qmax = self.model.jnt_range[jnt, 1]
            if qval < qmin - tol or qval > qmax + tol:
                return True
        return False

class mink_ik:
    def __init__(self, model, prefix="robot0_", collision_pairs=[]):
        self.model = model
        self.prefix = prefix
        self.configuration = ReturnConfig(model)
        self.end_effector_task = mink.FrameTask(
            frame_name=prefix + "attachment_site",
            frame_type="site",
            position_cost=0.5,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        self.tasks = [self.end_effector_task]
        self.limits = [
            mink.ConfigurationLimit(model=self.configuration.model),
        ]
        if len(collision_pairs) > 0:
            self.add_collision_pairs(collision_pairs)
        self.max_velocities = {
            prefix + "shoulder_pan_joint": np.pi,
            prefix + "shoulder_lift_joint": np.pi,
            prefix + "elbow_joint": np.pi,
            prefix + "wrist_1_joint": np.pi,
            prefix + "wrist_2_joint": np.pi,
            prefix + "wrist_3_joint": np.pi,
        }
        self.velocity_limit = mink.VelocityLimit(model, self.max_velocities)
        self.limits.append(self.velocity_limit)
        self.mocap_id = self.model.body("target").mocapid[0]
        self.solver = "quadprog"
        self.pos_threshold = 0.005
        self.ori_threshold = 1e-3
        self.max_iters = 3000

    #TODO
    def add_collision_pairs(self,end_effector, collision_pairs):
        wrist_3_geoms = mink.get_body_geom_ids(self.model, self.model.body(self.prefix + "wrist_3_link").id)
        # collision_pairs = [
        #     (wrist_3_geoms, ["floor", "wall"]),
        # ]
        # self.limits = [
        #     mink.ConfigurationLimit(model=self.configuration.model),
        #     mink.CollisionAvoidanceLimit(
        #         model=self.configuration.model,
        #         geom_pairs=collision_pairs,
        #     ),
        # ]
        # collection_pair = mink.CollisionAvoidanceLimit(
        #         model=self.configuration.model,
        #         geom_pairs=collision_pairs,)
        # self.limits.append(collection_pair)

    def ik(self, data, target_translation, target_rotation, dt=0.002):
        self.configuration.update(data.qpos)
        data.mocap_pos[self.mocap_id] = target_translation
        data.mocap_quat[self.mocap_id] = target_rotation
        # T_wt = mink.SE3.from_mocap_name(self.model, data, "target")
        T_wt = mink.SE3(np.concatenate([target_rotation, target_translation]))
        self.end_effector_task.set_target(T_wt)
        for _ in range(self.max_iters):
            vel = mink.solve_ik(
                self.configuration, self.tasks, dt, self.solver, damping=1e-3, limits=self.limits
            )
            self.configuration.integrate_inplace(vel, dt)
            err = self.end_effector_task.compute_error(self.configuration)
            pos_achieved = np.linalg.norm(err[:3]) <= self.pos_threshold
            ori_achieved = np.linalg.norm(err[3:]) <= self.ori_threshold
            if pos_achieved and ori_achieved:
                break
        return self.configuration.q