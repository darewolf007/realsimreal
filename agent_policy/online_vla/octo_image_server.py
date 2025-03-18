from fastapi import FastAPI, UploadFile, File, Form
from fastapi.logger import logger
from fastapi.background import BackgroundTasks
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from octo.model.octo_model import OctoModel
model = OctoModel.load_pretrained("/data1/user/zhangshaolong/IL_RL/octo/octo_model")
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
import jax
import nest_asyncio
import cv2
nest_asyncio.apply()
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
app = FastAPI()

@app.post("/vla")
async def vla(
    image_file: UploadFile = File(...), 
    label: str = Form(...),
):
    image_bytes = await image_file.read()
    nparr_image = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr_image, cv2.IMREAD_COLOR)
    print(image.shape)
    height, width = image.shape[:2]
    min_dim = min(height, width)
    start_y = (height - min_dim) // 2
    start_x = (width - min_dim) // 2
    cropped = image[start_y:start_y+min_dim, start_x:start_x+min_dim]
    resized_img = cv2.resize(cropped, (256, 256), interpolation=cv2.INTER_AREA)
    print(resized_img.shape)
    logger.info(f"instruction received: {label}")

    img = resized_img[np.newaxis,np.newaxis,...]
    print(img.shape)
    observation = {"image_primary": img, "timestep_pad_mask": np.array([[True]])}
    task = model.create_tasks(texts=["pick up banana"])
    action = model.sample_actions(
        observation, 
        task, 
        unnormalization_statistics=model.dataset_statistics["bridge_dataset"]["action"], 
        rng=jax.random.PRNGKey(0)
    )
    print(action)
    return {"action": action.tolist()}

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app="image_server:app", host="0.0.0.0", port=8000, reload=False)