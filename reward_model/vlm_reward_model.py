import base64
import numpy as np
import os
import io
import cv2
import json
import requests
import PIL.Image as Image
from collections import Counter
from openai import OpenAI
os.environ['ALL_PROXY'] = ''
os.environ['all_proxy'] = ''
PROMPT_TEMPLATES = {
    "pick":{
        "check_occluded": ["Is the {target_obj} in the current perspective significantly occluded by the {moving_obj}? no represents no occlusion, and yes represents occlusion. Directly output no or yes."],
        "no_occluded": ["If the {target_obj} is not occluded, the projection of the object must largely fall within the {moving_obj}'s area, without requiring complete enclosure or coverage. no represents not meeting the requirement, and yes represents meeting the requirement. Directly output no or yes."],
        "is_occluded": ["If the {target_obj} is occluded, at least one of the {moving_obj} must have a projection overlapping with the object and occlude it. Complete enclosure or coverage is not required. no represents not meeting the requirement, and yes represents meeting the requirement. Directly output no or yes."]
    }
}

class VLMClient:
    def __init__(self, model = "Qwen3VL-8B", api_key="EMPTY", base_url="http://10.184.17.177:22002/v1", timeout=3600, max_tokens = 2048):
        self.model = model
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def encode_image_from_path(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def encode_image_from_numpy(self, np_img, convert_img = True):
        if convert_img:
            np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode(".png", np_img)
        return base64.b64encode(buffer).decode("utf-8")

    def _build_img_prompt(self, image, instruction):
        if isinstance(image, np.ndarray):
            img_base64 = self.encode_image_from_numpy(image)
        elif isinstance(image, str):
            img_base64 = self.encode_image_from_path(image)
        else:
            raise TypeError(
                f"Unsupported image type: {type(image)}, expected numpy array or str"
            )
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": instruction
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    },
                ]
            }
        ]
        return messages

    def send_messages(self, message):
        stream = self.client.chat.completions.create(model=self.model, messages=message, max_tokens=self.max_tokens)
        return stream

    def ask_human_provided_projection_relationships(self, image_dict, task, moving_obj, target_obj, threadshold = 1):
        def postprocess_message(stream):
            output = stream.choices[0].message.content
            return output.strip().lower() == "yes"
        answer = []
        if moving_obj == "gripper":
            moving_obj = "two-finger gripper"
        task_promt = PROMPT_TEMPLATES[task]
        formatted_prompts = {key: [t.format(target_obj=target_obj,moving_obj=moving_obj) for t in templates]for key, templates in task_promt.items()}
        for view, image in image_dict.items():
            message = self._build_img_prompt(image, formatted_prompts["check_occluded"][0])
            occluded_model = "is_occluded" if postprocess_message(self.send_messages(message)) else "no_occluded"
            for instruction in formatted_prompts[occluded_model]:
                message = self._build_img_prompt(image, instruction)
                answer.append(postprocess_message(self.send_messages(message)))
        true_ratio = sum(answer) / len(answer)
        return true_ratio >= threadshold

if __name__ == "__main__":
    import PIL.Image as Image
    front_view_image = Image.open("/home/haowen/hw_mine/Real_Sim_Real/data/sim_data/pick_up_apple_and_place_it_to_the_bowl/Pick up apple and place it to the bowl4/rgb_frontview/9.png").convert("RGB")
    right_view_image = Image.open("/home/haowen/hw_mine/Real_Sim_Real/data/sim_data/pick_up_apple_and_place_it_to_the_bowl/Pick up apple and place it to the bowl4/rgb_rightview/9.png").convert("RGB")
    bird_view_image = Image.open("/home/haowen/hw_mine/Real_Sim_Real/data/sim_data/pick_up_apple_and_place_it_to_the_bowl/Pick up apple and place it to the bowl4/rgb_birdview/9.png").convert("RGB")
    image_dict = {
        "front_view": np.array(front_view_image),
        "right_view": np.array(right_view_image),
        "bird_view": np.array(bird_view_image),
    }
    moving_obj = "gripper"
    target_obj = "apple"
    task = "pick"
    client = VLMClient(model="Qwen3VL-8B")
    client.ask_human_provided_projection_relationships(image_dict, task, moving_obj, target_obj)
