import os
import torch
import torch.nn as nn
import clip
import cv2
import pickle
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
from reward_model.action_feasibility_model import train_action_feasible
from reward_model.offline_reward_model import train_offline_reward
from utils.image_util import resize_image
from torch.utils.tensorboard import SummaryWriter

class MultiImageLabelDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_eval=False):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        if is_eval:
            self.get_eval_data(data_dir)
        else:
            self.get_data(data_dir)

    def get_data(self, data_dir):
        subfolders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        for folder in subfolders:
            traj_path = os.path.join(data_dir, folder)
            file = sorted(os.listdir(traj_path), key=lambda x: int(x.split(".")[0]))
            for step in range(1, len(file)+1):
                pkl_data = pickle.load(open(os.path.join(traj_path, f"{step}.pkl"), "rb"))
                bird_view_img = pkl_data['bird_view'].astype(np.float32) / 255.0
                front_view_img = pkl_data['front_view'].astype(np.float32) / 255.0
                right_view_img = pkl_data['right_view'].astype(np.float32) / 255.0
                view_label = 1 if pkl_data['result'] == 100 else 0
                step_image = np.concatenate([resize_image(bird_view_img, target_size=(224, 224)), resize_image(front_view_img, target_size=(224, 224)), resize_image(right_view_img, target_size=(224, 224))], axis=2)
                step_image = np.transpose(step_image, (2, 0, 1))
                self.images.append(step_image)
                self.labels.append(view_label)

    def get_eval_data(self, data_dir):
        subfolders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        for folder in subfolders:
            traj_path = os.path.join(data_dir, folder + "/data")
            bird_view_path = os.path.join(data_dir, folder + "/rgb_birdview")
            front_view_path = os.path.join(data_dir, folder + "/rgb_frontview")
            right_view_path = os.path.join(data_dir, folder + "/rgb_rightview")
            files = sorted(os.listdir(traj_path), key=lambda x: int(x.split(".")[0]))
            for step in range(1, len(files)+1):
                bird_view_img = Image.open(os.path.join(bird_view_path, str(step) + ".png")).convert('RGB')
                front_view_img = Image.open(os.path.join(front_view_path, str(step) + ".png")).convert('RGB')
                right_view_img = Image.open(os.path.join(right_view_path, str(step) + ".png")).convert('RGB')
                step_image = np.concatenate([self.transform(bird_view_img), self.transform(front_view_img), self.transform(right_view_img)], axis=0)
                file_path = os.path.join(traj_path, str(step) + ".pkl")
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                    if data['rewards'] == 0:
                        step_label = 0
                    else:
                        step_label = 1
                self.images.append(step_image)
                self.labels.append(step_label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]    
        return image, label
    
class RGBDImageLabelDataset(Dataset):
    def __init__(self, data_dir, test_dir = None, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.get_data(data_dir)
        if test_dir is not None:
            self.get_test_data(test_dir)

    def get_data(self, data_dir):
        subfolders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        for folder in subfolders:
            traj_path = os.path.join(data_dir, folder)
            files = sorted(os.listdir(traj_path), key=lambda x: int(x.split(".")[0]))
            for step in range(1, len(files)+1):
                pkl_data = pickle.load(open(os.path.join(traj_path, f"{step}.pkl"), "rb"))
                scene_rgb_img = pkl_data['sceneview_depth'].astype(np.float32) / 255.0
                scene_depth_img = pkl_data['sceneview_rgb'].astype(np.float32)
                step_image = np.concatenate([scene_rgb_img, scene_depth_img], axis=2)
                step_image = np.transpose(step_image, (2, 0, 1))
                view_label = 1 if pkl_data['result'] == 100 else 0
                self.images.append(step_image)
                self.labels.append(view_label)

    def get_test_data(self, data_dir):
        subfolders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        for folder in subfolders:
            traj_path = os.path.join(data_dir, folder + "/data")
            scene_view_path = os.path.join(data_dir, folder + "/rgb_sceneview")
            depth_view_path = os.path.join(data_dir, folder + "/depth_sceneview")
            files = sorted(os.listdir(traj_path), key=lambda x: int(x.split(".")[0]))
            for step in range(1, len(files)+1):
                scene_view_img = cv2.cvtColor(cv2.imread(os.path.join(scene_view_path, str(step) + ".png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                depth_view_img = np.load(os.path.join(depth_view_path, str(step) + ".npy"))
                file_path = os.path.join(traj_path, str(step) + ".pkl")
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                    if data['rewards'] == 0:
                        step_label = 0
                    else:
                        step_label = 1
                self.images.append(scene_view_img)
                self.labels.append(step_label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]    
        return image, label


def load_multiview_train_dataset(data_dir, batch_size, val_split=0.2):
    dataset = MultiImageLabelDataset(data_dir)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def load_multiview_eval_dataset(data_dir, batch_size, val_split=0.2):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = MultiImageLabelDataset(data_dir, transform=transform)
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return eval_loader


def load_rgbd_train_dataset(data_dir, batch_size, val_split=0.2):
    dataset = RGBDImageLabelDataset(data_dir)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

if __name__ == "__main__":
    data_dir = "/home/haowen/hw_mine/Real_Sim_Real/experiments/Pick up banana/pick up banana-PickBanana-LaNE-test-2025-03-16-20-26-35/online_reward_data"
    base_path = os.path.dirname(os.path.abspath(__file__))
    batch_size = 32
    train_loader, val_loader = load_multiview_train_dataset(data_dir, batch_size)
    offline_log_dir = os.path.join(base_path, "./experiments/offline_reward_model/logs/offline_reward")
    os.makedirs(offline_log_dir, exist_ok=True)
    logger = SummaryWriter(log_dir=offline_log_dir)
    train_offline_reward(train_loader, val_loader, base_path, logger)
    logger.close()
    batch_size = 1
    action_feasible_log_dir = os.path.join(base_path, "./experiments/action_feasible_model/logs/action_feasible")
    action_logger = SummaryWriter(log_dir=action_feasible_log_dir)
    train_loader, val_loader = load_rgbd_train_dataset(data_dir, batch_size)
    train_action_feasible(train_loader, val_loader, base_path, action_logger)
    action_logger.close()
