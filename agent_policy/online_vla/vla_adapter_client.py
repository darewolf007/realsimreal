import numpy as np
from agent_policy.online_vla.websocket import websocket_client_policy

SERVER_URL = "10.184.17.177"
class VlaAdapterClient:
    def __init__(self, server_url=SERVER_URL, port=5001):
        self.client = websocket_client_policy.WebsocketClientPolicy(host=server_url, port=port)

    def get_action(self, obsrvations, use_wrist_camera=False, state_info = "eff"):
        if use_wrist_camera:
            wrist_image = obsrvations['hand_image']
        else:
            wrist_image = np.zeros((256, 256, 3), dtype=np.uint8)
        if state_info == "eff":
            state = obsrvations['eff']
        elif state_info == "joint":
            state = obsrvations['joint']
        observation_dict = {"agentview_image": obsrvations['scene_image'],
                            "robot0_eye_in_hand_image": wrist_image,
                            "state":  state}
        task_description = obsrvations['task_prompt']
        obs = {"observation": observation_dict, "task_description": task_description}
        action = self.client.infer(obs)["actions"]
        return action