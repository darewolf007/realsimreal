import requests
import numpy as np
import io
import cv2
import time
# 定义服务端地址（例如运行在本地服务器上）
SERVER_URL = "http://10.184.17.177:8000/act"

proprio_array = np.random.rand(1,).astype(np.float32)
instr = "can"
timestep = 0

rgb_img = "/home/haowen/hw_mine/Real_Sim_Real/data/4/scene_rgb_image/scene_1.jpg"
depth_img = '/home/haowen/hw_mine/Real_Sim_Real/data/4/scene_depth_image/scene_1mat.png'
mask = "/home/haowen/hw_mine/Real_Sim_Real/data/a can._0.84.png"
rgb_image = cv2.imread(rgb_img)
mask_image = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
depth_image = cv2.imread(depth_img, cv2.IMREAD_UNCHANGED)
depth_image = depth_image / 1000.0
def send_data(rgb_image: np.ndarray, depth_image: np.ndarray, mask_image: np.ndarray, instr):
    # 将图像数组转换为字节流
    _, image_encoded = cv2.imencode('.jpg', rgb_image)
    image_bytes = io.BytesIO(image_encoded.tobytes())

    # 将深度数组转换为字节流（使用 .npy 格式）
    depth_npy = np.load("/home/haowen/hw_mine/Real_Sim_Real/data/4/scene_depth_image/scene_1.npy")
    depth_bytes = io.BytesIO()
    np.save(depth_bytes, depth_npy)
    depth_bytes.seek(0)
    
    mask_bytes = io.BytesIO()
    np.save(mask_bytes, mask_image)
    mask_bytes.seek(0)
    # proprio_bytes = io.BytesIO()
    # np.save(proprio_bytes, proprio_array)
    # proprio_bytes.seek(0)
    camera_intrinsics = np.array([
        [978.735788085938, 0.0, 1030.94287109375],
        [0.0, 979.0402221679688, 766.4556274414062],
        [0.0, 0.0, 1.0]
    ])
    camera_intrinsics_bytes = io.BytesIO()
    np.save(camera_intrinsics_bytes, camera_intrinsics)
    camera_intrinsics_bytes.seek(0)
    # 定义要发送的数据
    files = {
        "image_file": ("image.jpg", image_bytes, "image/jpeg"),
        "depth_file": ("depth.npy", depth_bytes, "application/octet-stream"),
        "mask_file": ("mask.npy", mask_bytes, "application/octet-stream"),
        "camera_intrinsics_file": camera_intrinsics_bytes,
    }
    
    data = {
        "label": instr
    }

    # 发送 POST 请求到服务端
    response = requests.post(SERVER_URL, files=files, data=data)

    # 打印服务端返回的结果
    if response.status_code == 200:
        task_info = response.json()
        print(f"Task started: {task_info}")
        return task_info["task_id"]
    else:
        print("Failed to get a valid response from server. Status code:", response.status_code)
def poll_task_status(task_id):
    start_time = time.time()
    while time.time() - start_time < 100:  # 最多等待 60 秒
        response = requests.get(f"{SERVER_URL}/task_status/{task_id}")
        if response.status_code == 200:
            task_status = response.json()
            print(f"Task status: {task_status}")
            if task_status["status"] == "completed":
                print("Task completed. Result:", task_status["pose"])
                return task_status
            elif task_status["status"] == "error":
                print("Task failed. Error:", task_status["error"])
                return task_status
        else:
            print(f"Task is processing")
            
        time.sleep(1)  # 每秒轮询一次
    raise TimeoutError("Timeout: Task did not complete within 60 seconds")


if __name__ == "__main__":
    task_id = send_data(rgb_image, depth_image, mask_image, instr)
    if task_id:
        response = poll_task_status(task_id)
        if response["status"] == "completed":
            result = response["pose"]
            print(result)