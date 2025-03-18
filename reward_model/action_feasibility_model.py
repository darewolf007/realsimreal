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

    def get_train_data(self, data_dir):
        subfolders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        for folder in subfolders:
            traj_path = os.path.join(data_dir, folder + "/data")

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

class FeatureAlign(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_rgb = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv_depth = nn.Conv2d(channels, channels, 3, padding=1)
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, rgb_feat, depth_feat):
        aligned_rgb = self.conv_rgb(rgb_feat)
        aligned_depth = self.conv_depth(depth_feat)
        attention = self.attention(torch.cat([aligned_rgb, aligned_depth], dim=1))
        return aligned_rgb * attention + aligned_depth * (1 - attention)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class RGBDViewReward(nn.Module):
    def __init__(self, base_channels=8):
        super().__init__()
        
        self.rgb_conv1 = DoubleConv(3, base_channels)
        self.rgb_conv2 = DoubleConv(base_channels, base_channels * 2)
        self.rgb_conv3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.rgb_conv4 = DoubleConv(base_channels * 4, base_channels * 8)
        
        self.depth_conv1 = DoubleConv(1, base_channels)
        self.depth_conv2 = DoubleConv(base_channels, base_channels * 2)
        self.depth_conv3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.depth_conv4 = DoubleConv(base_channels * 4, base_channels * 8)
        
        self.align1 = FeatureAlign(base_channels)
        self.align2 = FeatureAlign(base_channels * 2)
        self.align3 = FeatureAlign(base_channels * 4)
        self.align4 = FeatureAlign(base_channels * 8)
        
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec_conv3 = DoubleConv(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec_conv2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec_conv1 = DoubleConv(base_channels * 2, base_channels)
        
        self.pool = nn.MaxPool2d(2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.fc = nn.Linear(base_channels, 1)  

    def forward(self, x):
        rgb = x[:, :3, :, :]
        depth = x[:, 3:, :, :]
        rgb1 = self.rgb_conv1(rgb)
        rgb2 = self.rgb_conv2(self.pool(rgb1))
        rgb3 = self.rgb_conv3(self.pool(rgb2))
        rgb4 = self.rgb_conv4(self.pool(rgb3))

        depth1 = self.depth_conv1(depth)
        depth2 = self.depth_conv2(self.pool(depth1))
        depth3 = self.depth_conv3(self.pool(depth2))
        depth4 = self.depth_conv4(self.pool(depth3))
        
        a1 = self.align1(rgb1, depth1)
        a2 = self.align2(rgb2, depth2)
        a3 = self.align3(rgb3, depth3)
        a4 = self.align4(rgb4, depth4)
        
        d3 = self.up3(a4)
        d3 = torch.cat([d3, a3], dim=1)
        d3 = self.dec_conv3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, a2], dim=1)
        d2 = self.dec_conv2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, a1], dim=1)
        d1 = self.dec_conv1(d1)
        
        pooled = self.global_avg_pool(d1)
        pooled = pooled.view(pooled.size(0), -1)
        logits = self.fc(pooled)

        return logits

    def get_reward(self, obs, task_text=None):
        with torch.no_grad():
            probs = self.forward(obs)
            probs = torch.sigmoid(probs)
            predictions = (probs > 0.5).float()
        return predictions

def eval(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for image, labels in val_loader:
            obs, labels = image.to(device), labels.to(device)
            outputs = model(obs)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            total_loss += loss.item() * image.size(0)
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()
            correct += (predictions == labels.float().unsqueeze(1)).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def train_action_feasible(train_loader, val_loader, base_path, logger=None, task_name="Pick up banana"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    camera_intrinsics = np.array([
            [978.735788085938, 0.0, 1030.94287109375],
            [0.0, 979.0402221679688, 766.4556274414062],
            [0.0, 0.0, 1.0]])
    fx = camera_intrinsics[0, 0]
    fy = camera_intrinsics[1, 1]
    cx = camera_intrinsics[0, 2]
    cy = camera_intrinsics[1, 2]
    reward_model = RGBDViewReward()
    reward_model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    num_epochs = 3000
    step = 0
    best_accuracy = 0.0
    best_model_path = os.path.join(base_path, './experiments/action_feasible_model/' + task_name +'/best_reward_model.pth')
    os.makedirs(os.path.dirname(best_model_path) if os.path.dirname(best_model_path) else '.', exist_ok=True)
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        reward_model.train()
        epoch_loss = 0.0
        
        for image, labels in train_loader:
            obs, labels = image.to(device), labels.to(device)
            outputs = reward_model(obs)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * obs.size(0)
            step += 1
            
            if step % 10 == 0:
                avg_val_loss, accuracy = eval(reward_model, val_loader, criterion, device)
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step}], Train Loss: {loss.item():.4f}, '
                      f'Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}')
                logger.add_scalar("Loss/Train", loss.item(), step)
                logger.add_scalar("Loss/Validation", avg_val_loss, step)
                logger.add_scalar("Accuracy/Validation", accuracy, step)
                logger.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], step)
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
        logger.add_scalar("Loss/Epoch_Train", avg_epoch_loss, epoch)
        logger.add_scalar("Loss/Epoch_Validation", avg_val_loss, epoch)
        logger.add_scalar("Accuracy/Epoch_Validation", accuracy, epoch)
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
