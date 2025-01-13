import os
import torch
import torch.nn as nn
import clip

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image

class ImageLabelDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        for label_dir in os.listdir(data_dir):
            label_path = os.path.join(data_dir, label_dir)
            if os.path.isdir(label_path):
                for img_name in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(int(label_dir))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_dataset(data_dir, batch_size, val_split=0.2):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = ImageLabelDataset(data_dir, transform=transform)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

class MultiViewReward:
    def __init__(self, fc_dim, ln_dim, image_num = 3, device = "cuda"):
        self.image_num = image_num
        self.base_path = os.path.dirname(os.path.realpath(__file__))
        self.model_path = os.path.join(self.base_path, "../../pre_train/dinov2")
        self.model_pth_path = os.path.join(self.base_path, "../../pre_train/dinov2_vits14.pth")
        self.fc = nn.Linear(fc_dim, ln_dim)
        self.ln = nn.LayerNorm(ln_dim)
        self.classifier = nn.Linear(ln_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.outputs = {}
        self.clip, preprocess = clip.load("ViT-B/32", device=device)
        self.dino = torch.hub.load(self.model_path, 'dinov2_vits14', source='local', pretrained=False).to(obs.device)
        self.dino.load_state_dict(torch.load(self.model_pth_path))

    def dino_embed(self, obs):
        with torch.no_grad():
            split_obs = torch.split(obs, [3] * self.image_num, dim=1)
            dino_embs = []
            for i in range(self.image_num):
                dino_emb = self.dino(split_obs[i])
                dino_embs.append(dino_emb)
            dino_embs = torch.cat(dino_embs, dim=1)
        return dino_embs


    def forward(self, obs, task_text, detach=False):
        dino_embs = self.dino_embed(obs)
        if task_text is not None:
            with torch.no_grad():
                text_features = self.clip.encode_text(task_text)
                batch_size = dino_embs.size(0)
                text_features = text_features.expand(batch_size, 512) 
                dino_language_embs = torch.cat((dino_embs, text_features), dim=1)  # 沿着第二维度拼接
        if detach:
            dino_language_embs = dino_language_embs.detach()
        h_fc = self.fc(dino_language_embs)
        self.outputs["fc"] = h_fc
        h_norm = self.ln(h_fc)
        self.outputs["ln"] = h_norm
        logits = self.classifier(h_norm)
        probs = self.sigmoid(logits)
        self.outputs["probs"] = probs
        return probs
    
    def get_reward(self, obs, detach=False):
        pass

class SignalViewReward:
    def __init__(self, reward_model, use_depth = True):
        self.reward_model = reward_model

def eval(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for obs, labels in val_loader:
            obs, labels = obs.to(device), labels.to(device)
            outputs = model(obs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * obs.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def train():
    data_dir = 'path_to_dataset'
    batch_size = 32
    train_loader, val_loader = load_dataset(data_dir, batch_size)
    reward_model = MultiViewReward(fc_dim=256, ln_dim=128, image_num=3)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=0.001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reward_model.to(device)
    num_epochs = 10
    step = 0
    for epoch in range(num_epochs):
        reward_model.train()
        for obs, labels in train_loader:
            obs, labels = obs.to(device), labels.to(device)
            outputs = reward_model(obs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step % 10 == 0:
                avg_loss, accuracy = eval(reward_model, val_loader, criterion, device)
                print(f'Step [{step}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    train()