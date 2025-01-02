# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.logger import logger
import os
from fastapi.background import BackgroundTasks

os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
app = FastAPI()

task_results = {}

def pose_estimation(task_id, obj_name, K, color, depth, mask, est_refine_iter, debug = 0, debug_dir = None):
    try:
        code_dir = os.path.dirname(os.path.realpath(__file__))
        mesh_file = f'{code_dir}/model/{obj_name}/textured.obj'
        debug_dir = f'{code_dir}/debug'
        mesh = trimesh.load(mesh_file)
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
        pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=est_refine_iter)
        mesh_file = f'{code_dir}/model/{obj_name}/textured.obj'
        mesh = trimesh.load(mesh_file)
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
        center_pose = pose@np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
        output = vis[...,::-1]
        cv2.imwrite(f"{code_dir}/output.jpg", output)
        task_results[task_id] = {"status": "completed", "pose": pose.tolist()}
    except Exception as e:
        task_results[task_id] = {"status": "error", "error": str(e)}

def tracker_object():
    pass

@app.post("/act")
async def act(
    background_tasks: BackgroundTasks,
    image_file: UploadFile = File(...), 
    depth_file: UploadFile = File(...),
    mask_file: UploadFile = File(...),
    camera_intrinsics_file: UploadFile = File(...),
    label: str = Form(...),
):
    import uuid
    task_id = str(uuid.uuid4())
    task_results[task_id] = {"status": "processing", "pose": [0]}
    image_bytes = await image_file.read()
    depth_bytes = await depth_file.read()
    mask_bytes = await mask_file.read()
    logger.info(f"instruction received: {label}")

    camera_intrinsics_bytes = await camera_intrinsics_file.read()
    camera_intrinsics_bytes_io = io.BytesIO(camera_intrinsics_bytes)
    K = np.load(camera_intrinsics_bytes_io)
    logger.info(f"instruction received: {K}")

    nparr_image = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr_image, cv2.IMREAD_COLOR)
    logger.info(f"image receivied: {image.shape}({image.dtype})")

    depth_bytes_io = io.BytesIO(depth_bytes)
    depth_image = np.load(depth_bytes_io)
    logger.info(f"depth receivied: {depth_image.shape}({depth_image.dtype})")
    
    mask_bytes_io = io.BytesIO(mask_bytes)
    mask_image = np.load(mask_bytes_io).astype(bool)
    logger.info(f"proprio receivied: {mask_image.shape}({mask_image.dtype})")

    background_tasks.add_task(
        pose_estimation,
        task_id, "can", K, image, depth_image, mask_image, 5
    )

    return {"task_id": task_id, "status": "processing"}

@app.get("/act/task_status/{task_id}")
async def get_task_status(task_id: str):
    if task_id in task_results:
        return task_results[task_id]
    return {"status": "not_found"}

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app="image_server:app", host="0.0.0.0", port=8000, reload=True)
