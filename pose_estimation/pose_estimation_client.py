import requests
import numpy as np
import io
import cv2
import time
def send_data(rgb_image, depth_npy, mask_image, label, SERVER_URL, camera_intrinsics):
    np_rgb_image = np.array(rgb_image)
    _, image_encoded = cv2.imencode('.jpg', np_rgb_image)
    image_bytes = io.BytesIO(image_encoded.tobytes())

    depth_bytes = io.BytesIO()
    np.save(depth_bytes, depth_npy)
    depth_bytes.seek(0)

    mask_bytes = io.BytesIO()
    np.save(mask_bytes, mask_image)
    mask_bytes.seek(0)

    camera_intrinsics_bytes = io.BytesIO()
    np.save(camera_intrinsics_bytes, camera_intrinsics)
    camera_intrinsics_bytes.seek(0)

    files = {
        "image_file": ("image.jpg", image_bytes, "image/jpeg"),
        "depth_file": ("depth.npy", depth_bytes, "application/octet-stream"),
        "mask_file": ("mask.npy", mask_bytes, "application/octet-stream"),
        "camera_intrinsics_file": camera_intrinsics_bytes,
    }

    data = {
        "label": label,
        "camera_intrinsics": camera_intrinsics
    }

    response = requests.post(SERVER_URL, files=files, data=data)

    if response.status_code == 200:
        task_info = response.json()
        print(f"Task started: {task_info}")
        return task_info["task_id"]
    else:
        print("Failed to get a valid response from server. Status code:", response.status_code)


def poll_task_status(task_id, SERVER_URL):
    start_time = time.time()
    while time.time() - start_time < 100:
        response = requests.get(f"{SERVER_URL}/task_status/{task_id}")
        if response.status_code == 200:
            task_status = response.json()
            if task_status["status"] == "completed":
                return task_status
            elif task_status["status"] == "error":
                print("Task failed. Error:", task_status["error"])
                return task_status
        time.sleep(5)
    raise TimeoutError("Timeout: Task did not complete within 100 seconds")

def get_pose(server_url, rgb_image, depth_image, mask_image, label, camera_intrinsics):
    task_id = send_data(rgb_image, depth_image, mask_image, label, server_url, camera_intrinsics)
    if task_id:
        response = poll_task_status(task_id, server_url)
        if response["status"] == "completed":
            result = response["pose"]
            return result
    else:
        return None