import os
import cv2
import imageio
import numpy as np
from pathlib import Path
import re

def images_to_video(image_dir, output_filename="output.mp4", fps=24):
    image_dir = Path(image_dir)

    def extract_number(file):
        match = re.search(r"scene_(\d+)\.jpg", file.name)
        return int(match.group(1)) if match else -1

    image_files = sorted([
        f for f in image_dir.iterdir()
        if f.suffix.lower() == ".jpg" and f.name.startswith("scene_")
    ], key=extract_number)

    if not image_files:
        raise ValueError(f"No scene_*.jpg files found in {image_dir}")

    frames = []
    for file in image_files:
        img = cv2.imread(str(file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)

    output_path = image_dir / output_filename
    imageio.mimsave(str(output_path), frames, fps=fps, codec="libx264")

    print(f"Video saved to: {output_path}")

# 示例调用
images_to_video("/home/haowen/hw_mine/Real_Sim_Real/data/realexperiment/can_1/scene_rgb_image", output_filename="video.mp4", fps=10)
