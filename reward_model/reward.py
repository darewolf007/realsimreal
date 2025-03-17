from reward_model.utils.image_util import resize_image, save_image_pkl

class RewardModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.subtask_num = len(cfg.subtask_language_info)
        self.subtask_language_info = cfg.subtask_language_info
        self.subtask_object_info = cfg.subtask_object_info
        self.reward_fn_map = {
            "test": self.test_online_reward
        }
        self.last_action = None
        self.last_observation = None
        self.last_info = None
        self.sum_reward = 0
        self.subtask_idx = 0
        self.subtask_pre_flag = False if self.cfg.use_pre_reward else True
        self.subtask_done_flag = False

    def reset(self):
        self.last_action = None
        self.last_observation = None
        self.last_info = None
        self.sum_reward = 0
        self.subtask_idx = 0
        self.subtask_pre_flag = False if self.cfg.use_pre_reward else True
        self.subtask_done_flag = False

    def update(self, action, observation, info):
        if self.subtask_done_flag and self.subtask_pre_flag:
            self.subtask_idx += 1
            self.subtask_pre_flag = False if self.cfg.use_pre_reward else True
            self.subtask_done_flag = False
        self.last_action = action
        self.last_observation = observation
        self.last_info = info

    def save_reward_data(self, image_dict, reward):
        image_dict['result'] = reward
        if reward > 0:
            save_image_pkl(image_dict, self.cfg.agent.online_data_save_path + "/done/", save_ori_image=False)
        else:   
            save_image_pkl(image_dict, self.cfg.agent.online_data_save_path + "/fail/", save_ori_image=False)

    def test_online_reward(self, observations, action, info, reward = -1, is_save=False, is_train=True):
        image_dict = {
            "front_view": resize_image(observations["frontview_image"], 0.25),
            "right_view": resize_image(observations["rightview_image"], 0.25),
            "bird_view": resize_image(observations["birdview_image"], 0.25),
            "sceneview_depth": observations["sceneview_depth"],
            "sceneview_rgb": observations["sceneview_image"],
        }
        if self.subtask_pre_flag == False:
            # pre_reward function
            if reward == -1:
                pass
            else:
                self.subtask_pre_flag = True
                self.sum_reward += reward
        else:
            # done_reward function
            if reward == -1:
                pass
            else:
                self.subtask_done_flag = True
                self.sum_reward += reward
        reward_info = {self.subtask_language_info[self.subtask_idx]: {"pre_reward": self.subtask_pre_flag, "done_reward": self.subtask_done_flag}}
        self.update(action, observations, info)
        done = True if self.subtask_idx == self.subtask_num else False
        if info["truncation"] == True or done:
            self.reset()
        if is_save:
            self.save_reward_data(image_dict, reward)
        return reward, done, reward_info

    def __call__(self, observations, action, info, reward = -1, reward_type="test", is_save=False, is_train=True):
        reward_fn = self.reward_fn_map[reward_type]
        return reward_fn(observations, action, info, reward, is_save, is_train)