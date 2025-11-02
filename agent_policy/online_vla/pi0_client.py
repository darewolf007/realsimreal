import requests
import numpy as np
import io
import cv2
import time
SERVER_URL = "http://10.184.17.177:8000/pi0"

class Pi0Client:
    def __init__(self, server_url=SERVER_URL):
        self.server_url = server_url
    
    def get_action(self, obsrvations):
        scene_image = obsrvations['scene_image']
        hand_image = obsrvations['hand_image']
        joint = obsrvations['joint']
        gripper = obsrvations['gripper']
        task_prompt = obsrvations['task_prompt']
        action_list = self.send_observation(scene_image, hand_image, joint, gripper, task_prompt)
        return np.array(action_list).astype(np.float32)

    def send_observation(self, scene_image: np.ndarray, hand_image: np.ndarray, joint: np.ndarray, gripper: np.ndarray, task_prompt):
        _, scene_image_encoded = cv2.imencode('.jpg', scene_image)
        scene_image_bytes = io.BytesIO(scene_image_encoded.tobytes())
        _, hand_image_encoded = cv2.imencode('.jpg', hand_image)
        hand_image_bytes = io.BytesIO(hand_image_encoded.tobytes())
        joint_bytes = io.BytesIO()
        np.save(joint_bytes, joint)
        joint_bytes.seek(0)
        gripper_bytes = io.BytesIO()
        np.save(gripper_bytes, gripper)
        gripper_bytes.seek(0)


        files = {
            "scene_image_file": ("scene.jpg", scene_image_bytes, "image/jpeg"),
            "hand_image_file": ("hand.jpg", hand_image_bytes, "image/jpeg"),
            "joint_file": joint_bytes,
            "gripper_file": gripper_bytes,
        }
        
        data = {
            "task_prompt": task_prompt
        }

        response = requests.post(SERVER_URL, files=files, data=data)

        if response.status_code == 200:
            action = response.json()
            return action['action']
        else:
            print("Failed to get a valid response from server. Status code:", response.status_code)