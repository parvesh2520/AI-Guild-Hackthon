"""
Contrastive Fine-Tuning for CLIP (100% Legal)

This script fine-tunes the standard `openai/clip-vit-large-patch14` model
using the provided hackathon dataset. 

It maps each training image's coordinates to its nearest country, and trains
the vision encoder to match the text prompt "A street view photo in [Country]".
This directly replicates how StreetCLIP was made, but uses only the permitted
data, making the subsequent text-matching inference completely legal.
"""

import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
import numpy as np

# ============================================================
# 1. Country Mapping
# ============================================================
COUNTRIES = {
    "United States": (39.8, -98.5), "Canada": (56.1, -106.3),
    "Mexico": (23.6, -102.5), "Brazil": (-14.2, -51.9),
    "Argentina": (-38.4, -63.6), "Colombia": (4.5, -74.2),
    "Peru": (-12.0, -77.0), "Chile": (-35.6, -71.2),
    "Ecuador": (-1.8, -78.2), "Bolivia": (-16.3, -63.6),
    "Uruguay": (-32.5, -55.8), "Paraguay": (-23.4, -58.4),
    "Venezuela": (6.4, -66.6), "Costa Rica": (9.7, -83.8),
    "Panama": (8.5, -80.8), "Guatemala": (15.8, -90.2),
    "Dominican Republic": (18.7, -70.2), "Cuba": (21.5, -77.8),
    "Jamaica": (18.1, -77.3), "Puerto Rico": (18.2, -66.6),
    "Trinidad and Tobago": (10.4, -61.2),
    "United Kingdom": (55.3, -3.4), "France": (46.2, 2.2),
    "Germany": (51.2, 10.4), "Spain": (40.5, -3.7),
    "Italy": (41.9, 12.6), "Portugal": (39.4, -8.2),
    "Netherlands": (52.1, 5.3), "Belgium": (50.5, 4.5),
    "Switzerland": (46.8, 8.2), "Austria": (47.5, 13.3),
    "Poland": (51.9, 19.1), "Czech Republic": (49.8, 15.5),
    "Sweden": (60.1, 18.6), "Norway": (60.5, 8.5),
    "Finland": (61.9, 25.7), "Denmark": (56.3, 9.5),
    "Ireland": (53.4, -6.2), "Greece": (39.1, 21.8),
    "Turkey": (38.9, 35.2), "Romania": (45.9, 24.9),
    "Hungary": (47.2, 19.5), "Ukraine": (48.4, 31.2),
    "Croatia": (45.1, 15.2), "Bulgaria": (42.7, 25.5),
    "Serbia": (44.0, 21.0), "Slovakia": (48.7, 19.7),
    "Slovenia": (46.2, 14.8), "Estonia": (58.6, 25.0),
    "Latvia": (56.9, 24.1), "Lithuania": (55.2, 23.9),
    "Iceland": (64.9, -19.0), "North Macedonia": (41.5, 21.7),
    "Albania": (41.2, 20.2), "Montenegro": (42.7, 19.4),
    "Bosnia and Herzegovina": (43.9, 17.7), "Malta": (35.9, 14.4),
    "Cyprus": (35.1, 33.4), "Luxembourg": (49.8, 6.1),
    "Moldova": (47.0, 28.9), "Belarus": (53.7, 27.6),
    "Russia": (61.5, 105.3), "Kazakhstan": (48.0, 68.0),
    "Uzbekistan": (41.4, 64.6), "Georgia": (42.3, 43.4),
    "Azerbaijan": (40.1, 47.6), "Armenia": (40.1, 44.5),
    "Kyrgyzstan": (41.2, 74.8),
    "Japan": (36.2, 138.2), "South Korea": (35.9, 127.8),
    "China": (35.9, 104.2), "Taiwan": (23.7, 121.0),
    "Mongolia": (46.9, 103.8), "Hong Kong": (22.3, 114.2),
    "Thailand": (15.9, 100.9), "Vietnam": (14.1, 108.3),
    "Indonesia": (-0.8, 113.9), "Philippines": (12.9, 121.8),
    "Malaysia": (4.2, 101.9), "Singapore": (1.3, 103.8),
    "Cambodia": (12.6, 104.9), "Myanmar": (21.9, 95.9),
    "Laos": (19.9, 102.5),
    "India": (20.6, 78.9), "Pakistan": (30.4, 69.3),
    "Bangladesh": (23.7, 90.4), "Sri Lanka": (7.9, 80.8),
    "Nepal": (28.4, 84.1),
    "United Arab Emirates": (23.4, 53.8), "Saudi Arabia": (23.9, 45.1),
    "Israel": (31.0, 34.9), "Jordan": (30.6, 36.2),
    "Lebanon": (33.9, 35.9), "Oman": (21.5, 55.9),
    "Qatar": (25.4, 51.2), "Bahrain": (26.0, 50.6),
    "Kuwait": (29.3, 47.5), "Iran": (32.4, 53.7),
    "Iraq": (33.2, 43.7),
    "South Africa": (-30.6, 22.9), "Nigeria": (9.1, 8.7),
    "Kenya": (-0.02, 37.9), "Egypt": (26.8, 30.8),
    "Morocco": (31.8, -7.1), "Tunisia": (33.9, 9.5),
    "Algeria": (28.0, 1.7), "Ghana": (7.9, -1.0),
    "Senegal": (14.5, -14.5), "Tanzania": (-6.4, 34.9),
    "Ethiopia": (9.1, 40.5), "Uganda": (1.4, 32.3),
    "Rwanda": (-1.9, 29.9), "Mozambique": (-18.7, 35.5),
    "Madagascar": (-18.8, 46.9), "Botswana": (-22.3, 24.7),
    "Namibia": (-22.6, 17.1), "Zimbabwe": (-19.0, 29.2),
    "Cameroon": (7.4, 12.4), "Ivory Coast": (7.5, -5.5),
    "Mali": (17.6, -4.0), "Angola": (-11.2, 17.9),
    "Congo": (-4.0, 21.8), "Libya": (26.3, 17.2),
    "Australia": (-25.3, 133.8), "New Zealand": (-40.9, 174.9),
    "Fiji": (-18.0, 179.0), "Papua New Guinea": (-6.3, 143.9),
}

country_names = list(COUNTRIES.keys())
country_coords = np.array([COUNTRIES[c] for c in country_names])

def get_nearest_country(lat, lon):
    dists = np.sum((country_coords - np.array([lat, lon]))**2, axis=1)
    return country_names[np.argmin(dists)]

# ============================================================
# 2. Dataset
# ============================================================
class ContrastiveDataset(Dataset):
    def __init__(self, csv_path, image_dir):
        self.image_dir = image_dir
        self.samples = []
        
        # We need to map lat/lon to country
        print("[Dataset] Mapping coordinates to countries...")
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id = row['image_id']
                lat = float(row['latitude'])
                lon = float(row['longitude'])
                img_filename = img_id + '.jpg' if not img_id.endswith('.jpg') else img_id
                img_path = os.path.join(image_dir, img_filename)
                
                if os.path.exists(img_path):
                    country = get_nearest_country(lat, lon)
                    text = f"A street view photo in {country}"
                    self.samples.append((img_path, text))
        print(f"[Dataset] Loaded {len(self.samples)} valid samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        return image, text

def collate_fn(batch, processor):
    images, texts = zip(*batch)
    inputs = processor(
        text=list(texts), 
        images=list(images), 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    )
    return inputs

# ============================================================
# 3. Training Loop
# ============================================================
def train():
    print("=" * 60)
    print("Contrastive Fine-Tuning (100% Legal Approach)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model_id = "openai/clip-vit-large-patch14"
    print(f"Loading {model_id}...")
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    
    # Freeze Text Encoder & First 18 Vision blocks to save VRAM
    # ViT-Large has 24 blocks. Unfreeze only the last 2 blocks.
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the vision projection and logit scale
    for param in model.visual_projection.parameters():
        param.requires_grad = True
    model.logit_scale.requires_grad = True
    
    # Unfreeze last 2 vision layers
    for layer in model.vision_model.encoder.layers[-2:]:
        for param in layer.parameters():
            param.requires_grad = True
            
    # Unfreeze vision post-layernorm
    for param in model.vision_model.post_layernorm.parameters():
        param.requires_grad = True
        
    model = model.to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    dataset = ContrastiveDataset(
        csv_path='geolocation-prediction/training_dataset/noised_dataset/ground_truth_coordinates.csv',
        image_dir='geolocation-prediction/training_dataset/noised_dataset/images'
    )
    
    batch_size = 8  # Small batch to prevent OOM
    accumulation_steps = 8  # Effective batch size = 64
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, processor),
        pin_memory=True,
        drop_last=True
    )
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=0.01)
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    
    epochs = 2
    global_step = 0
    
    scaler = torch.cuda.amp.GradScaler()
    
    print("\nStarting Training...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        
        for i, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                
                logits_per_image = outputs.logits_per_image
                logits_per_text = outputs.logits_per_text
                
                # Ground truth is identity matrix (each image matches its own text)
                ground_truth = torch.arange(len(logits_per_image), dtype=torch.long, device=device)
                
                loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
                loss = loss / accumulation_steps
                
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                
            epoch_loss += loss.item() * accumulation_steps
            
            if i % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} [{i}/{len(dataloader)}] Loss: {loss.item() * accumulation_steps:.4f}")
                
        print(f"--- Epoch {epoch+1} Complete. Avg Loss: {epoch_loss/len(dataloader):.4f} ---")
        
        os.makedirs('checkpoints', exist_ok=True)
        save_path = f'checkpoints/contrastive_clip_ep{epoch+1}.pt'
        # Save only trainable params to save space
        trainable_state_dict = {k: v for k, v in model.state_dict().items() if model.get_parameter(k).requires_grad}
        torch.save(trainable_state_dict, save_path)
        print(f"Saved checkpoint: {save_path}")

    print("\nTraining Complete! You are now 100% legal.")

if __name__ == '__main__':
    train()
