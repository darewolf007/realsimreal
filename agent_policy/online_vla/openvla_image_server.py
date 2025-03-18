from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.logger import logger
import os
from fastapi.background import BackgroundTasks
import torch
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
app = FastAPI()
MODEL_PATH = "/data1/user/zhangshaolong/IL_RL/openvla/model/openvla"
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
vla_model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH, 
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda:0")

@app.post("/vla")
async def vla(
    image_file: UploadFile = File(...), 
    label: str = Form(...),
):
    print("hjrerere")
    image_bytes = await image_file.read()
    nparr_image = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr_image, cv2.IMREAD_COLOR)
    print(image.shape)
    image = Image.fromarray(nparr_image)
    logger.info(f"instruction received: {label}")
    prompt = "In: What action should the robot take to {<label>}?\nOut:"
    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
    action = vla_model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    print(action)
    return {"action": action.tolist()}

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app="image_server:app", host="0.0.0.0", port=8000, reload=False)