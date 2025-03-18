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
        self.get_test_data(data_dir)

    def get_test_data(self, data_dir):
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

class MultiViewReward(nn.Module):
    def __init__(self, fc_dim=3 * 384, ln_dim=512, image_num=3, device="cuda", use_text = False):
        super(MultiViewReward, self).__init__()
        self.image_num = image_num
        self.device = device
        self.use_text = use_text
        self.base_path = os.path.dirname(os.path.realpath(__file__))
        self.model_path = os.path.join(self.base_path, "../../pre_train/dinov2")
        self.model_pth_path = os.path.join(self.base_path, "../../pre_train/dinov2_vits14.pth")
        
        try:
            self.dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg").to(device)
            print("Loaded DINOv2 from official repository")
        except Exception as e:
            print(f"Error loading from hub: {e}")
            print("Attempting to load from local path")
            self.dino = torch.hub.load(self.model_path, 'dinov2_vits14', source='local', pretrained=False).to(device)
            self.dino.load_state_dict(torch.load(self.model_pth_path))
            
        try:
            self.clip, self.preprocess = clip.load("ViT-B/32", device=device)
            print("CLIP model loaded successfully")
        except Exception as e:
            print(f"Could not load CLIP: {e}")
        
        dino_embedding_dim = 384  # DINOv2 ViT-S/14 embedding dimension
        clip_embedding_dim = 512 if use_text else 0
        total_input_dim = (dino_embedding_dim * image_num) + clip_embedding_dim
        
        self.fc = nn.Linear(total_input_dim, ln_dim)
        self.ln = nn.LayerNorm(ln_dim)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(ln_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
        self.outputs = {}

    def dino_embed(self, obs):
        with torch.no_grad():
            split_obs = torch.split(obs, [3] * self.image_num, dim=1)
            dino_embs = []
            
            for i in range(self.image_num):
                view = split_obs[i]
                if view.shape[2] != 224 or view.shape[3] != 224:
                    view = F.interpolate(view, size=(224, 224), mode='bilinear', align_corners=False)
                dino_emb = self.dino(view)
                dino_embs.append(dino_emb)
            
            dino_embs = torch.cat(dino_embs, dim=1)
        
        return dino_embs

    def forward(self, obs, task_text=None, detach=True):
        dino_embs = self.dino_embed(obs)
        
        if self.use_text and task_text is not None:
            with torch.no_grad():
                text_inputs = clip.tokenize(task_text).to(self.device)
                text_features = self.clip.encode_text(text_inputs)
                batch_size = dino_embs.size(0)
                text_features = text_features.expand(batch_size, -1) 
                combined_embs = torch.cat((dino_embs, text_features), dim=1)
        else:
            combined_embs = dino_embs
            
        if detach:
            combined_embs = combined_embs.detach()
            
        h_fc = self.fc(combined_embs)
        self.outputs["fc"] = h_fc
        
        h_norm = self.ln(h_fc)
        self.outputs["ln"] = h_norm
        
        h_drop = self.dropout(h_norm)
        
        logits = self.classifier(h_drop)
        probs = self.sigmoid(logits)
        self.outputs["probs"] = probs
        
        return probs

    def get_reward(self, obs, task_text=None):
        """Calculate binary reward from observations"""
        self.eval()
        with torch.no_grad():
            probs = self.forward(obs, task_text)
            rewards = (probs > 0.5).float()
        return rewards, probs.squeeze()

def eval(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for obs, labels in val_loader:
            obs, labels = obs.to(device), labels.to(device)
            outputs = model(obs)
            loss = criterion(outputs, labels.float().unsqueeze(1))  # Ensure labels have right shape
            total_loss += loss.item() * obs.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == labels.float().unsqueeze(1)).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def train_offline_reward(train_loader, val_loader, base_path, logger=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reward_model = MultiViewReward(fc_dim=3*384, ln_dim=512, image_num=3, device=device)
    reward_model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    num_epochs = 3000
    step = 0
    best_accuracy = 0.0
    best_model_path = os.path.join(base_path, './experiments/offline_reward_model/best_reward_model.pth')
    os.makedirs(os.path.dirname(best_model_path) if os.path.dirname(best_model_path) else '.', exist_ok=True)
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        reward_model.train()
        epoch_loss = 0.0
        
        for obs, labels in train_loader:
            obs, labels = obs.to(device), labels.to(device)
            outputs = reward_model(obs)
            loss = criterion(outputs, labels.float().unsqueeze(1))  # Ensure labels have right shape
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * obs.size(0)
            step += 1
            
            if step % 10 == 0:
                avg_val_loss, accuracy = eval(reward_model, val_loader, criterion, device)
                logger.add_scalar("Loss/Train", loss.item(), step)
                logger.add_scalar("Loss/Validation", avg_val_loss, step)
                logger.add_scalar("Accuracy/Validation", accuracy, step)
                logger.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], step)
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
        logger.add_scalar("Loss/Epoch_Train", avg_epoch_loss, epoch)
        logger.add_scalar("Loss/Epoch_Validation", avg_val_loss, epoch)
        logger.add_scalar("Accuracy/Epoch_Validation", accuracy, epoch)
        scheduler.step(accuracy)
    
    print(f"Training completed! Best accuracy: {best_accuracy:.4f}")
    print(f"Best model saved to {best_model_path}")
    
    checkpoint = torch.load(best_model_path)
    reward_model.load_state_dict(checkpoint['model_state_dict'])
    final_loss, final_accuracy = eval(reward_model, val_loader, criterion, device)
    print(f"Final evaluation - Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}")