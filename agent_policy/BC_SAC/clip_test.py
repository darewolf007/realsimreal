import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

text = clip.tokenize(["lift a cube"]).to(device)

with torch.no_grad():
    text_features = model.encode_text(text)
    
