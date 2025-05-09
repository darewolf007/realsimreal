import os
import torch
import cv2
from PIL import Image
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.logger import logger
from agent_policy.agent_policy import BaseAgentPolicy
from train_agent import make_policy_agent, make_imageprocess_fn
from omegaconf import OmegaConf
cfg = OmegaConf.load("configs/pick_maniwhere.yaml")
app = FastAPI()
obs_shape = (128,128,3)
action_shape = (7,)
agent = make_policy_agent(cfg, cfg.agent_name, cfg.device, obs_shape, action_shape, is_train=True)
image_process_fn = make_imageprocess_fn(cfg, cfg.agent_name, obs_shape)
@app.post("/vla")
async def agent(
    image_file: UploadFile = File(...), 
    label: str = Form(...),
):
    image_bytes = await image_file.read()
    nparr_image = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr_image, cv2.IMREAD_COLOR)
    if cfg.agent_name == "maniwhere":
        step_image = image_process_fn(image, reward = -1, info = {"truncation": False}, action = np.zeros(action_shape), is_reset = False, is_train = False)
    else:
        step_image = image_process_fn(image)
    action = agent.get_action(step_image)
    return {"action": action.tolist()}

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app="image_server:app", host="0.0.0.0", port=8000, reload=False)