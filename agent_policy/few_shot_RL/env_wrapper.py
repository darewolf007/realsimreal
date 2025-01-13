import numpy as np
import gymnasium as gym


def make(
    domain_name,
    task_name,
    seed,
    from_pixels,
    height,
    width,
    cameras=range(1),
):
    if "robosuite" in domain_name:
        env = RobosuiteWrapper(
            domain_name,
            from_pixels=from_pixels,
            cameras=cameras,
            height=height,
            width=width,
        )
    elif "Franka" in domain_name:
        import gym_franka

        env = gym.make(domain_name)
    else:
        raise ValueError(f"Domain name {domain_name} not supported")
    return env


class RobosuiteWrapper(gym.Env):
    def __init__(
        self,
        env_name,
        cameras=(0, 1),
        reward_type="sparse",
        from_pixels=True,
        height=100,
        width=100,
        channels_first=True,
        control=None,
        set_done_at_success=True,
    ):
        self.camera_names = ["frontview", "robot0_eye_in_hand"]
        # self.camera_names = ["frontview"]

        import robosuite as suite
        from robosuite import load_controller_config

        config = load_controller_config(default_controller="OSC_POSE")
        # initial_qpos  array([ 0.        ,  0.19634954,  0.        , -2.61799388,  0.        ,2.94159265,  0.78539816])
        # config["initial_qpos"] = np.array([ 0.        ,  0.29634954,  0.        , -1.61799388,  0.        ,1.94159265,  1.78539816])
        if "lift" in env_name:
            env = suite.make(
                env_name="Lift",
                robots="Panda",
                controller_configs=config,
                camera_names=["frontview", "robot0_eye_in_hand"],
                camera_heights=height,
                camera_widths=width,
                control_freq=10,
                horizon=40,
            )
            self.horizon = 40
        elif "stack" in env_name:
            env = suite.make(
                env_name="Stack",
                robots="Panda",
                controller_configs=config,
                camera_names=["frontview", "robot0_eye_in_hand"],
                camera_heights=height,
                camera_widths=width,
                control_freq=10,
                horizon=80,
            )
            self.horizon = 80
        elif "door" in env_name:
            # create environment instance
            env = suite.make(
                env_name="Door",
                robots="Panda",
                controller_configs=config,
                camera_names=["frontview", "robot0_eye_in_hand"],
                camera_heights=height,
                camera_widths=width,
                control_freq=10,
                horizon=80,
            )
            self.horizon = 80
        elif "pick_place_can" in env_name:
            env = suite.make(
                env_name="PickPlace",
                robots="Panda",
                controller_configs=config,
                camera_names=["frontview", "robot0_eye_in_hand"],
                camera_heights=height,
                camera_widths=width,
                control_freq=10,
                horizon=120,
                single_object_mode=2,
                object_type="can",
                # bin1_pos=(0.1, -0.27, 0.8),
                # bin2_pos=(0.1, 0.27, 0.8),
            )
            self.horizon = 120

        self._env = env
        self.cameras = cameras
        self.reward_type = reward_type
        self.from_pixels = from_pixels
        self.height = height
        self.width = width
        self.channels_first = channels_first
        self.control = control
        self.set_done_at_success = set_done_at_success
        self.domain_name = env_name

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=float)

        shape = (
            [3 * len(cameras), height, width]
            if channels_first
            else [height, width, 3 * len(cameras)]
        )
        if self.from_pixels:
            self._observation_space = gym.spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
        else:
            if isinstance(self._env.observation_space, gym.spaces.Dict):
                self._observation_space = self._env.observation_space["state"]
            else:
                self._observation_space = self._env.observation_space

    def _get_obs(self):
        obs = self._env._get_observations()
        return self._unpack_obs(obs)

    def _unpack_obs(self, obs):
        if self.from_pixels:
            images = []
            for c in self.cameras:
                frame = obs[self.camera_names[c] + "_image"][::-1]
                images.append(frame)
            pixel_obs = np.concatenate(images, axis=-1)
            if self.channels_first:
                pixel_obs = pixel_obs.transpose((2, 0, 1))
            return pixel_obs.astype("uint8")
        else:
            return obs["state"]

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = self._unpack_obs(obs)
        reward = -1 if reward <= 0 else 100
        if self.set_done_at_success and reward > 0:
            done = True
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self._env.reset()
        return self._unpack_obs(obs)

    def render(self, mode="rgb_array", **kwargs):
        obs = self._env._get_observations()
        return obs["frontview_image"][::-1]

    @property
    def _max_episode_steps(self):
        return self.horizon

    @property
    def observation_space(self):
        return self._observation_space
