import logging
import sys
import draccus
import numpy as np
from collections import deque
from termcolor import colored
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from websocket import websocket_policy_server
from websocket import base_policy as _base_policy
from typing_extensions import override
from websocket import websocket_policy_server
sys.path.append("../..")
sys.path.insert(0, "/data2/user/sunhaowen/hw_code/vla/VLA-Adapter/libero") 
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)

from prismatic.vla.constants import NUM_ACTIONS_CHUNK

@dataclass
class GenerateConfig:
    # fmt: off
    port: int = 5001
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "/hard_data/user_dataset/sunhaowen_dataset/vla/vla-adapter"     # Pretrained checkpoint path
    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_minivlm: bool = True                         # If True, uses minivlm
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy
    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on
    save_version: str = "vla-adapter"                # version of 
    use_pro_version: bool = True                     # encourage to use the pro models we released.
    phase: str = "Inference"

def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": obs["state"],
    }

    return observation, img  # Return both processed observation and original image for replay




def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action

def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key

def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)
    model.set_version(cfg.save_version)
    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor

class Policy(_base_policy.BasePolicy):
    def __init__(self, cfg, model, action_head, proprio_projector, noisy_action_projector, processor):
        self.cfg = cfg
        self.model = model
        self.action_head = action_head
        self.proprio_projector = proprio_projector
        self.noisy_action_projector = noisy_action_projector
        self.processor = processor

    @override
    def infer(self, obs):
        outputs = {}
        observation = obs["observation"]
        task_description = obs["task_description"]
        resize_size = get_image_resize_size(self.cfg)
        observation, img = prepare_observation(observation, resize_size)
        actions = get_action(
        self.cfg,
        self.model,
        observation,
        task_description,
        processor=self.processor,
        action_head=self.action_head,
        proprio_projector=self.proprio_projector,
        noisy_action_projector=self.noisy_action_projector,
        use_film=self.cfg.use_film,
        use_minivlm=self.cfg.use_minivlm)
        action_queue = deque(maxlen=self.cfg.num_open_loop_steps)
        action_queue.extend(actions) 
        action = action_queue.popleft()
        action = process_action(action, self.cfg.model_family)
        outputs["actions"] = action
        return outputs
    
@draccus.wrap()
def run(cfg: GenerateConfig):
    cfg.task_suite_name = "real_ur5e_dataset"
    cfg.pretrained_checkpoint = "/hard_data/user_dataset/sunhaowen_dataset/vla/vla-adapter-mine/ur-real-10/"
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    policy = Policy(cfg, model, action_head, proprio_projector, noisy_action_projector, processor)
    print(colored("Starting websocket policy server...", "green"))
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=cfg.port,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    run()
