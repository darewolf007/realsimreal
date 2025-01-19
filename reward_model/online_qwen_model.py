from openai import OpenAI
import base64
import numpy as np
import os
import io
import cv2
import json
import requests
import PIL.Image as Image
from collections import Counter
os.environ['ALL_PROXY'] = ''
os.environ['all_proxy'] = ''

def ask_claude(image_data, prompt, other_image = None, model_choice = "claude-3-5-sonnet-20240620", max_retries=5):
    client = OpenAI(
        api_key="sk-QyHM6JgtRWhaM33UHyx7ovfhSzXn07GdO4klT7nl4CaR6XzK",
        base_url="https://api.aikeji.vip/v1"
    )
    if isinstance(image_data, np.ndarray):
        image = Image.fromarray(image_data.astype('uint8'))  # 确保数据格式正确
    else:
        raise ValueError("Input image_data must be a numpy array.")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.read()).decode('utf-8')
    # encoded_image = base64.b64encode(image_data).decode('utf-8')
    if other_image is not None:
        other_image = Image.fromarray(other_image.astype('uint8'))
        other_buffer = io.BytesIO()
        other_image.save(other_buffer, format="PNG")
        other_buffer.seek(0)
        other_encoded_image = base64.b64encode(other_buffer.read()).decode('utf-8')
    retries = 0
    while retries < max_retries:
        print("promot", prompt)
        try:
            if other_image is not None:
                response = client.chat.completions.create(
                model=model_choice,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text",
                            "text": prompt},
                            {"type": "text",
                            "text": "0 view"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                            },
                            {"type": "text",
                            "text": "1 view"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{other_encoded_image}"},
                            },
                        ],
                    }
                    ],
                    max_tokens=300,
                    stream=False
                )
                print(response.choices[0].message.content)
                content = response.choices[0].message.content
                if content == "0" or content == "1":
                    return content
                else:
                    return "0"
                # return response.choices[0].message.content
            else:
                response = client.chat.completions.create(
                    model=model_choice,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text",
                                "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                                },
                            ],
                        }
                    ],
                    max_tokens=300,
                    stream=False
                )
                print(response.choices[0].message.content)
                content = response.choices[0].message.content
                if content == "0" or content == "1":
                    return content
                else:
                    return "0"
                # return response.choices[0].message.content    
        except Exception as e:
            print("network error, retrying...")
    return "0"

def count_yes_no_from_end(text, num_words=50):
    words = text.split()
    last_words = words[-num_words:]
    word_counts = Counter(last_words)
    no_num = word_counts['no']
    yes_num = word_counts['yes']
    if no_num > yes_num:
        return "no"
    else:
        return "yes"

def check_qwen_results(results):
    start_idx = results.find("Final Answer")
    if start_idx != -1:
        final_answer_section = results[start_idx + len("Final Answer"):].strip()
        if "yes" in final_answer_section:
            return "yes"
        elif "no" in final_answer_section:
            return "no"
    return count_yes_no_from_end(results)

def ask_qwen_online(image_data, prompt, url = 'http://10.184.17.177:8001/predict'):
    images_list = []
    # images_list.append(image_data.tolist())
    # _, image_encoded = cv2.imencode('.jpg', image_data)
    # image_bytes = image_encoded.tobytes()
    # image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    # images_list.append(image_base64)
    _, image_encoded = cv2.imencode('.jpg', image_data)
    image_base64 = base64.b64encode(image_encoded.tobytes()).decode('utf-8')
    images_list.append(image_base64)
    text_list = []
    text_list.append(prompt)
    data = json.dumps({"texts": text_list, "images": images_list})
    response = requests.post(url, data=data, headers={'Content-Type': 'application/json'}, timeout=1000)
    response.raise_for_status()
    results = response.json()
    print(results)
    return check_qwen_results(results['response'][0])

def ask_grasp_subtask(image_dict, moving_obj = "", target_obj = "can", ask_qwen = True):
    if ask_qwen:
        ask_online = ask_qwen_online
    else:
        ask_online = ask_claude
    promot_1 = f"Is the object {target_obj} in the current perspective significantly occluded by the two-finger gripper? no represents no occlusion, and yes represents occlusion. Directly output no or yes."
    promot_2_0 = f"If the object {target_obj} is not occluded, the projection of the object must largely fall within the gripper's area, without requiring complete enclosure or coverage. no represents not meeting the requirement, and yes represents meeting the requirement. Directly output no or yes."
    promot_2_1 = f"If the object {target_obj} is occluded, at least one finger of the two-finger gripper must have a projection overlapping with the object and occlude it. Complete enclosure or coverage is not required. no represents not meeting the requirement, and yes represents meeting the requirement. Directly output no or yes."
    image_state_list = []
    for key in image_dict.keys():
        image_data = image_dict[key]
        result = ask_online(image_data, promot_1)
        if result == "no":
            image_state = ask_online(image_data, promot_2_0)
        else:
            image_state = ask_online(image_data, promot_2_1)
        image_state_list.append(int(image_state))
        if image_state == "no":
            return False
    return True
    
def ask_pour_subtask(image_dict, moving_obj = "", target_obj = "", ask_qwen = False):
    promot_1 = f"Which of these two perspectives better shows the tilt angle between the {moving_obj} and the {target_obj} in the vertical direction relative to the table? Directly output 0 or 1."
    promot_2 = f"Let's think step by step. The tilt angle of the can relative to the vertical direction of the {target_obj} should be greater than 40 degrees. Does this perspective meet the requirement? 0 represents not meeting the requirement, and 1 represents meeting the requirement. Directly only output 0 or 1."
    promot_3 = f"Let's think step by step. The edge of the {moving_obj}'s opening should be close to the upper edge of the {target_obj}. Does this perspective meet the requirement? 0 represents not meeting the requirement, and 1 represents meeting the requirement. Directly only output 0 or 1."
    promot_4 = f"Let's think step by step. The edge of the {moving_obj}'s opening should be positioned above the {target_obj}. Does this perspective meet the requirement? 0 represents not meeting the requirement, and 1 represents meeting the requirement. Directly only output 0 or 1."
    promot_5 = f"When pouring the {moving_obj}, the projection of the {moving_obj} should be above the {target_obj}. Does this perspective meet the requirement? Directly output 0 or 1."
    promot_6 = f"When pouring the {moving_obj}, the projection of the {moving_obj} should not block the opening of the {target_obj}. Does this perspective meet the requirement? Directly output 0 or 1."
    front_view = image_dict["front_view"]
    right_view = image_dict["right_view"]
    bird_view = image_dict["bird_view"]
    result_1 = ask_online(front_view, promot_1, right_view)
    # result_1 = "0"
    if result_1 == "0":
        ask_image = front_view
        other_image = right_view
    else:
        ask_image = right_view
        other_image = front_view
    result_2 = ask_online(ask_image, promot_2)
    result_3 = ask_online(ask_image, promot_3)
    result_4 = ask_online(ask_image, promot_4)
    flag_1 = (np.array([int(result_2), int(result_3), int(result_4)]).sum()>=2)
    if not flag_1:
        return False
    result_5_1 = ask_online(other_image, promot_5)
    result_6_1 = ask_online(other_image, promot_6)
    flag_2 = (np.array([int(result_5_1), int(result_6_1)]).sum()>=2)
    if flag_2:
        return True
    else:
        result_5_2 = ask_online(bird_view, promot_5)
        result_6_2 = ask_online(bird_view, promot_6)
        flag_3 = (np.array([int(result_5_2), int(result_6_2)]).sum()>=2)
        if flag_3:
            return True
        else:
            return False

if __name__ == "__main__":
    import PIL.Image as Image
    front_view_image = Image.open("/data1/user/sunhaowen_dataset/realsimreal/data/sim_data/Pour can into a cup4/rgb_frontview/32.png").convert("RGB")
    right_view_image = Image.open("/data1/user/sunhaowen_dataset/realsimreal/data/sim_data/Pour can into a cup4/rgb_rightview/32.png").convert("RGB")
    bird_view_image = Image.open("/data1/user/sunhaowen_dataset/realsimreal/data/sim_data/Pour can into a cup4/rgb_birdview/32.png").convert("RGB")
    # front_view_image = Image.open("/home/haowen/hw_mine/Real_Sim_Real/data/sim_data/pour_can_new/Pour can into a cup4/rgb_frontview/169.png").convert("RGB")
    # right_view_image = Image.open("/home/haowen/hw_mine/Real_Sim_Real/data/sim_data/pour_can_new/Pour can into a cup4/rgb_rightview/169.png").convert("RGB")
    # bird_view_image = Image.open("/home/haowen/hw_mine/Real_Sim_Real/data/sim_data/pour_can_new/Pour can into a cup4/rgb_birdview/169.png").convert("RGB")
    image_dict = {
        "front_view": np.array(front_view_image),
        "right_view": np.array(right_view_image),
        "bird_view": np.array(bird_view_image),
    }
    grasp_state = ask_grasp_subtask(image_dict, moving_obj = "gripper", target_obj = "can")
    # pour_state = ask_pour_subtask(image_dict, moving_obj = "can", target_obj = "cup")
    print(grasp_state)
