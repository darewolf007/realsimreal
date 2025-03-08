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

class RGBDImageLabelDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.depths = []
        self.labels = []
        self.get_test_data(data_dir)

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
                self.depths.append(depth_view_img)
                self.labels.append(step_label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        depth = self.depths[idx]
        label = self.labels[idx]    
        return image, depth, label

def load_dataset(data_dir, batch_size, val_split=0.2):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = RGBDImageLabelDataset(data_dir, transform=transform)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

class DepthToPointCloud(nn.Module):
    def __init__(self, fx, fy, cx, cy):
        super().__init__()
        self.register_buffer('fx', torch.tensor(fx))
        self.register_buffer('fy', torch.tensor(fy))
        self.register_buffer('cx', torch.tensor(cx))
        self.register_buffer('cy', torch.tensor(cy))

    def forward(self, depth):
        B, _, H, W = depth.shape
        
        u = torch.arange(W, device=depth.device).view(1, W).expand(H, W)
        v = torch.arange(H, device=depth.device).view(H, 1).expand(H, W)
        u = u.unsqueeze(0).expand(B, -1, -1)
        v = v.unsqueeze(0).expand(B, -1, -1)
        
        z = depth.squeeze(1)
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        
        return torch.stack([x, y, z], dim=1).float()

class DualBranchFusion(nn.Module):
    def __init__(self, in_channels_rgb, in_channels_pc):
        super().__init__()
        # RGB分支
        self.rgb_conv = nn.Sequential(
            nn.Conv2d(in_channels_rgb, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 下采样
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 点云分支
        self.pc_conv = nn.Sequential(
            nn.Conv2d(in_channels_pc, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 下采样
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 融合全连接层
        self.fc = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1))
        
    def forward(self, rgb, pc):
        # 特征提取
        rgb_feat = self.rgb_conv(rgb)    # (B, 128, H/4, W/4)
        pc_feat = self.pc_conv(pc)        # (B, 128, H/4, W/4)
        
        # 全局平均池化
        rgb_feat = F.adaptive_avg_pool2d(rgb_feat, (1, 1)).squeeze(-1).squeeze(-1)  # (B, 128)
        pc_feat = F.adaptive_avg_pool2d(pc_feat, (1, 1)).squeeze(-1).squeeze(-1)   # (B, 128)
        
        # 特征拼接
        fused = torch.cat([rgb_feat, pc_feat], dim=1)  # (B, 256)
        return self.fc(fused)  # (B, 1)

class RGBDViewReward(nn.Module):
    def __init__(self, fx, fy, cx, cy):
        super().__init__()
        self.depth2pc = DepthToPointCloud(fx, fy, cx, cy)
        self.dual_branch = DualBranchFusion(
            in_channels_rgb=3, 
            in_channels_pc=3
        )
        self.pc_norm = nn.BatchNorm2d(3)
        self.output = nn.Sigmoid()

    def forward(self, x):
        x = x.float()
        rgb = x[:, :3, :, :]
        depth = x[:, 3:, :, :] 
        pc = self.depth2pc(depth)
        pc = self.pc_norm(pc)
        logits = self.dual_branch(rgb, pc)
        return self.output(logits)

def eval(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for image, depth, labels in val_loader:
            image, depth, labels = image.to(device), depth.to(device), labels.to(device)
            obs = torch.cat([image, depth], dim=3).permute(0, 3, 1, 2)
            outputs = model(obs)
            loss = criterion(outputs, labels.float().unsqueeze(1))  # Ensure labels have right shape
            total_loss += loss.item() * image.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == labels.float().unsqueeze(1)).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def train():
    test_data_dir = '/home/haowen/hw_mine/Real_Sim_Real/data/sim_data/depth_test'
    train_data_dir = 'path_to_dataset'
    batch_size = 1
    train_loader, val_loader = load_dataset(test_data_dir, batch_size) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    camera_intrinsics = np.array([
            [978.735788085938, 0.0, 1030.94287109375],
            [0.0, 979.0402221679688, 766.4556274414062],
            [0.0, 0.0, 1.0]])
    fx = camera_intrinsics[0, 0]
    fy = camera_intrinsics[1, 1]
    cx = camera_intrinsics[0, 2]
    cy = camera_intrinsics[1, 2]
    reward_model = RGBDViewReward(fx, fy, cx, cy)
    reward_model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    num_epochs = 30
    step = 0
    best_accuracy = 0.0
    best_model_path = 'best_reward_model.pth'
    
    os.makedirs(os.path.dirname(best_model_path) if os.path.dirname(best_model_path) else '.', exist_ok=True)
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        reward_model.train()
        epoch_loss = 0.0
        
        for image, depth, labels in train_loader:
            image, depth, labels = image.to(device), depth.to(device), labels.to(device)
            obs = torch.cat([image, depth], dim=3).permute(0, 3, 1, 2)
            outputs = reward_model(obs)
            loss = criterion(outputs, labels.float().unsqueeze(1))  # Ensure labels have right shape
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * obs.size(0)
            step += 1
            
            if step % 10 == 0:
                avg_val_loss, accuracy = eval(reward_model, val_loader, criterion, device)
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step}], Train Loss: {loss.item():.4f}, '
                      f'Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}')
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': reward_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'accuracy': best_accuracy,
                    }, best_model_path)
                    print(f"Best model saved with accuracy: {best_accuracy:.4f}")
                
                reward_model.train()
        
        avg_epoch_loss = epoch_loss / len(train_loader.dataset)
        avg_val_loss, accuracy = eval(reward_model, val_loader, criterion, device)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_epoch_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, '
              f'Accuracy: {accuracy:.4f}')
        
        scheduler.step(accuracy)
    
    print(f"Training completed! Best accuracy: {best_accuracy:.4f}")
    print(f"Best model saved to {best_model_path}")
    
    checkpoint = torch.load(best_model_path)
    reward_model.load_state_dict(checkpoint['model_state_dict'])
    final_loss, final_accuracy = eval(reward_model, val_loader, criterion, device)
    print(f"Final evaluation - Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}")
    
    return reward_model, best_accuracy

if __name__ == "__main__":
    reward_model, best_accuracy = train()