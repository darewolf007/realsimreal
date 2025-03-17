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

class MultiImageLabelDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.get_data(data_dir)

    def get_data(self, data_dir):
        subfolders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        for folder in subfolders:
            traj_path = os.path.join(data_dir, folder)
            file = sorted(os.listdir(traj_path), key=lambda x: int(x.split(".")[0]))
            for step in range(1, len(file)+1):
                bird_view_img = pickle.load(open(os.path.join(traj_path, str(step) + ".pkl"), "rb"))['bird_view']
                front_view_img = pickle.load(open(os.path.join(traj_path, str(step) + ".pkl"), "rb"))['front_view']
                right_view_img = pickle.load(open(os.path.join(traj_path, str(step) + ".pkl"), "rb"))['right_view']
                view_label = pickle.load(open(os.path.join(traj_path, str(step) + ".pkl"), "rb"))['result']

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
    
def load_dataset(data_dir, batch_size, val_split=0.2):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = MultiImageLabelDataset(data_dir, transform=transform)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


if __name__ == "__main__":
    data_dir = "/home/haowen/hw_mine/Real_Sim_Real/experiments/Pick up banana/pick up banana-PickBanana-LaNE-test-2025-03-16-20-26-35/online_reward_data"
    batch_size = 32
    train_loader, val_loader = load_dataset(data_dir, batch_size)
