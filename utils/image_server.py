from typing import Union
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.logger import logger
import numpy as np
import logging
import cv2
import io
import os, json

import clip
import torch
from scipy.spatial.transform import Rotation as R

logging.basicConfig(level=logging.INFO)
# --------------------------------- Camera Parameters ---------------------------------

# instrinsic
K = np.array([
    [978.735788085938, 0.0, 1030.94287109375],
    [0.0, 979.0402221679688, 766.4556274414062],
    [0.0, 0.0, 1.0]
])
# x, y, z, qx, qy, qz, qw
# camera_parameters = np.array([
#     -0.21951864924879808,
#     -1.2729284299978574,
#     0.3962307642102433,
#     -0.8550775097388451,
#     0.05841229797447445,
#     -0.013242663058440563,
#     0.5150292104912855
# ])
camera_parameters = np.array([
    -2.220946620485160505e-01,
    -1.264790385858366450e+00,
    2.973679281265694785e-01,
    -8.607659875045721165e-01,
    5.898920282536856963e-02,
    -1.602337205464063677e-02,
    5.053171679779984160e-01 
])


# ------------------------------------ Load Model -------------------------------------

# device = torch.device("cuda")
# exp_dir = "/data1/ljy/real_robot/arnold_actor/output/real/train_peract_clip_0/ckpts" #revise
# ckpt_path = "2024-11-12 07:09:39_peract_real_rgb_clip_100000.pth" #revise

# with initialize_config_dir(version_base=None, config_dir=os.path.join("/data1/ljy/real_robot/arnold_actor", "configs")):
#     exp_cfg = compose(config_name="3d_diffuser")
# agent = create_agent_3d_diffuser(exp_cfg, device=device)

# clip_model, _ = clip.load("RN50", device="cpu")  # CLIP-ResNet50
# clip_model = clip_model.to(device)
# clip_model.eval()

# ckpt_path = os.path.join(exp_dir, ckpt_path)
# ckpt = torch.load(ckpt_path, map_location="cpu")
# agent.load_state_dict(ckpt["model_state_dict"])
# agent.eval()

app = FastAPI()

# def translation_rotation_to_matrix(params):
#     translation = params[:3]
#     rotation = params[3:]
#     rotation_matrix = R.from_quat(rotation).as_matrix()
#     matrix = np.eye(4)
#     matrix[:3, :3] = rotation_matrix
#     matrix[:3, 3] = translation
#     return matrix


# def depth_to_robot_base(depth, K, camera_parameters):
#     h, w = depth.shape
#     u, v = np.meshgrid(np.arange(w), np.arange(h))
#     z = depth
#     x = (u - K[0, 2]) * z / K[0, 0]
#     y = (v - K[1, 2]) * z / K[1, 1]

#     cam2base = translation_rotation_to_matrix(camera_parameters)
#     points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
#     points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
#     transformed_points = (cam2base @ points_homogeneous.T).T[:, :3]
#     return transformed_points.reshape(h, w, -1)


# def encode_time(t, episode_length=25):
#     return (1. - (t / float(episode_length - 1))) * 2. - 1.


# def move_to_cuda(batch, device: torch.device):
#     if isinstance(batch, torch.Tensor):
#         return batch.to(device, non_blocking=True)
#     elif isinstance(batch, list):
#         return [move_to_cuda(t, device) for t in batch]
#     elif isinstance(batch, tuple):
#         return tuple(move_to_cuda(t, device) for t in batch)
#     elif isinstance(batch, dict):
#         return {n: move_to_cuda(t, device) for n, t in batch.items()}
#     return batch

# def _clip_encode_text(clip_model, text):
#     x = clip_model.token_embedding(text).type(
#         clip_model.dtype
#     )  # [batch_size, n_ctx, d_model]

#     x = x + clip_model.positional_embedding.type(clip_model.dtype)
#     x = x.permute(1, 0, 2)  # NLD -> LND
#     x = clip_model.transformer(x)
#     x = x.permute(1, 0, 2)  # LND -> NLD
#     x = clip_model.ln_final(x).type(clip_model.dtype)

#     emb = x.clone()
#     x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection

#     return x, emb

tasks = ["pickup","reorient","translate","slide","rotate"]
any_state = {
"pickup" : np.linspace(3,30,10),
"translate" : np.linspace(3,30,10),
"reorient" : np.linspace(18,180,10),
"slide" : np.linspace(10,100,10),
"rotate" : np.linspace(18,180,10)
}

# def preprocessing(
#     instr: str,
#     timestep: float,
#     proprio: np.ndarray,
#     image: np.ndarray, 
#     depth: np.ndarray,
#     K: np.ndarray,
#     camera_parameters: np.ndarray,
#     image_size = (128, 128),
# ):
#     # project before resize
#     pcd = depth_to_robot_base(depth, K, camera_parameters)
    
#     # resize
#     if image_size is not None:
#         rgb = cv2.resize(image, dsize=image_size, interpolation=cv2.INTER_NEAREST)
#         pcd = cv2.resize(pcd, dsize=image_size, interpolation=cv2.INTER_NEAREST)
        
#     rgb = rgb.transpose(2, 0, 1)
#     pcd = pcd.transpose(2, 0, 1).astype(np.float32)
    
#     with open("/data1/ljy/real_robot/any.json", "r") as f:
#         any_state = json.load(f)
#     task = None
#     for i in tasks:
#         if i in instr:
#             task = i
#     states = any_state[task]
#     state = None
#     for s in states.keys():
#         if str(s) in instr:
#              state = s
#              (a1,x1), (a2,x2) = any_state[task][str(state)]
#     if state is None:
#         b_state = id[task].keys()
#         for s in b_state:
#             if str(s) in instr:
#                 state = s
#                 a1=a2=0.5
#                 x1=x2=state
#     print(x1, x2)
#     s1, s2 = id[task][str(x1)], id[task][str(x2)]
#     state = [np.array([a1]).astype(np.float32),np.array([s1]).astype(np.float32),np.array([a2]).astype(np.float32),np.array([s2]).astype(np.float32)]
#     instr_ids = clip.tokenize([instr])
#     with torch.no_grad():
#         instr_feats, instr_embs = _clip_encode_text(clip_model, instr_ids.to(device))
#     instr_embs = instr_embs[0].float().detach().cpu()
#     low_dim_state = proprio.astype(np.float32)
#     action = np.zeros(shape=(1,7)).astype(np.float32)
#     action_mask = np.ones(shape=(1,)).astype(np.float32)
#     obs = {
#         "rgb": torch.from_numpy(rgb.astype(np.float32)).unsqueeze(0).unsqueeze(0),
#         "pcd": torch.from_numpy(pcd.astype(np.float32)).unsqueeze(0).unsqueeze(0),
#         "instr_embs": instr_embs.unsqueeze(0),
#         "curr_gripper": torch.from_numpy(low_dim_state).unsqueeze(0).unsqueeze(0),
#         "trajectory": torch.from_numpy(action).unsqueeze(0),
#         "trajectory_mask": torch.from_numpy(action_mask).unsqueeze(0),
#         "value":torch.from_numpy(np.array(state)).unsqueeze(0)
#     }
    
#     return obs
    
    
# def predict_action(obs: dict) -> dict:
#     obs = move_to_cuda(obs, device=device)
#     action = agent.act(
#         obs
#     )
#     # action = {"x": 0.0}
#     return action


@app.post("/act")
async def act(
    image_file: UploadFile = File(...), 
    depth_file: UploadFile = File(...),
    proprio_file: UploadFile = File(...),
    instr: str = Form(...),
    timestep: float = Form(...)
):
    image_bytes = await image_file.read()
    depth_bytes = await depth_file.read()
    proprio_bytes = await proprio_file.read()
    logger.info(f"instruction received: {instr}")

    nparr_image = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr_image, cv2.IMREAD_COLOR)
    logger.info(f"image receivied: {image.shape}({image.dtype})")

    depth_bytes_io = io.BytesIO(depth_bytes)
    depth = np.load(depth_bytes_io)
    logger.info(f"depth receivied: {depth.shape}({depth.dtype})")
    
    proprio_bytes_io = io.BytesIO(proprio_bytes)
    proprio = np.load(proprio_bytes_io)
    logger.info(f"proprio receivied: {proprio.shape}({proprio.dtype})")
    
    # obs = preprocessing(
    #     instr, timestep, proprio[:7], image, depth, K, camera_parameters
    # )
    # print(timestep)

    # action = predict_action(obs)
    action = np.array([0,0,0,0])
    # cv2.imwrite(os.path.join("/data1/tjy/ur5_real/RVT/rvt/data", "images", f"{timestep}.jpg"), image)
    # np.save(os.path.join("/data1/tjy/ur5_real/RVT/rvt/data", "depths", f"{timestep}.npy"), depth)
    # np.save(os.path.join("/data1/tjy/ur5_real/RVT/rvt/data", "actions", f"{timestep}.npy"), np.array(action))
    # np.save(os.path.join("/data1/tjy/ur5_real/RVT/rvt/data", "trajs", f"{timestep}.npy"), np.array(proprio))
    # np.save(os.path.join("/data1/tjy/ur5_real/RVT/rvt/data", "camera.npy"), np.array(camera_parameters))

    logger.info(action)

    return {"action": action}