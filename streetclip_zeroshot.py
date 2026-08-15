"""
Use StreetCLIP's FULL model (text+vision) for zero-shot geolocation.
StreetCLIP was trained on GeoGuessr - its text encoder knows country names.
We can compare each test image against text prompts like "A street view photo from France"
to get zero-shot country predictions, then use country centroids as coordinates.
"""
import os
import csv
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torch.amp import autocast

# Load FULL CLIP model (not just vision)
print("[StreetCLIP] Loading full CLIP model with text encoder...")
model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"[StreetCLIP] Using device: {device}")

# Define countries with their approximate centroids
COUNTRIES = {
    "United States": (39.8, -98.5),
    "Canada": (56.1, -106.3),
    "Mexico": (23.6, -102.5),
    "Brazil": (-14.2, -51.9),
    "Argentina": (-38.4, -63.6),
    "Colombia": (4.5, -74.2),
    "Peru": (-9.2, -75.0),
    "Chile": (-35.6, -71.2),
    "United Kingdom": (55.3, -3.4),
    "France": (46.2, 2.2),
    "Germany": (51.2, 10.4),
    "Spain": (40.5, -3.7),
    "Italy": (41.9, 12.6),
    "Portugal": (39.4, -8.2),
    "Netherlands": (52.1, 5.3),
    "Belgium": (50.5, 4.5),
    "Switzerland": (46.8, 8.2),
    "Austria": (47.5, 13.3),
    "Poland": (51.9, 19.1),
    "Czech Republic": (49.8, 15.5),
    "Sweden": (60.1, 18.6),
    "Norway": (60.5, 8.5),
    "Finland": (61.9, 25.7),
    "Denmark": (56.3, 9.5),
    "Ireland": (53.4, -8.2),
    "Greece": (39.1, 21.8),
    "Turkey": (38.9, 35.2),
    "Romania": (45.9, 24.9),
    "Hungary": (47.2, 19.5),
    "Ukraine": (48.4, 31.2),
    "Russia": (61.5, 105.3),
    "Japan": (36.2, 138.2),
    "South Korea": (35.9, 127.8),
    "China": (35.9, 104.2),
    "India": (20.6, 78.9),
    "Thailand": (15.9, 100.9),
    "Indonesia": (-0.8, 113.9),
    "Vietnam": (14.1, 108.3),
    "Philippines": (12.9, 121.8),
    "Malaysia": (4.2, 101.9),
    "Australia": (-25.3, 133.8),
    "New Zealand": (-40.9, 174.9),
    "South Africa": (-30.6, 22.9),
    "Nigeria": (9.1, 8.7),
    "Kenya": (-0.02, 37.9),
    "Egypt": (26.8, 30.8),
    "Morocco": (31.8, -7.1),
    "Israel": (31.0, 34.9),
    "United Arab Emirates": (23.4, 53.8),
    "Saudi Arabia": (23.9, 45.1),
    "Taiwan": (23.7, 121.0),
    "Singapore": (1.3, 103.8),
    "Croatia": (45.1, 15.2),
    "Bulgaria": (42.7, 25.5),
    "Serbia": (44.0, 21.0),
    "Slovakia": (48.7, 19.7),
    "Estonia": (58.6, 25.0),
    "Latvia": (56.9, 24.1),
    "Lithuania": (55.2, 23.9),
    "Iceland": (64.9, -19.0),
    "Ecuador": (-1.8, -78.2),
    "Bolivia": (-16.3, -63.6),
    "Uruguay": (-32.5, -55.8),
    "Paraguay": (-23.4, -58.4),
    "Costa Rica": (9.7, -83.8),
    "Panama": (8.5, -80.8),
    "Guatemala": (15.8, -90.2),
    "Dominican Republic": (18.7, -70.2),
    "Cuba": (21.5, -77.8),
    "Jamaica": (18.1, -77.3),
    "Puerto Rico": (18.2, -66.6),
    "Trinidad and Tobago": (10.4, -61.2),
    "Venezuela": (6.4, -66.6),
    "Iran": (32.4, 53.7),
    "Iraq": (33.2, 43.7),
    "Pakistan": (30.4, 69.3),
    "Bangladesh": (23.7, 90.4),
    "Sri Lanka": (7.9, 80.8),
    "Nepal": (28.4, 84.1),
    "Mongolia": (46.9, 103.8),
    "Cambodia": (12.6, 104.9),
    "Myanmar": (21.9, 95.9),
    "Laos": (19.9, 102.5),
    "Tanzania": (-6.4, 34.9),
    "Ethiopia": (9.1, 40.5),
    "Ghana": (7.9, -1.0),
    "Senegal": (14.5, -14.5),
    "Tunisia": (33.9, 9.5),
    "Algeria": (28.0, 1.7),
    "Libya": (26.3, 17.2),
    "Jordan": (30.6, 36.2),
    "Lebanon": (33.9, 35.9),
    "Oman": (21.5, 55.9),
    "Qatar": (25.4, 51.2),
    "Bahrain": (26.0, 50.6),
    "Kuwait": (29.3, 47.5),
    "Cyprus": (35.1, 33.4),
    "Malta": (35.9, 14.4),
    "Luxembourg": (49.8, 6.1),
    "Slovenia": (46.2, 14.8),
    "North Macedonia": (41.5, 21.7),
    "Albania": (41.2, 20.2),
    "Montenegro": (42.7, 19.4),
    "Bosnia and Herzegovina": (43.9, 17.7),
}

country_names = list(COUNTRIES.keys())
country_coords = np.array([COUNTRIES[c] for c in country_names])

# Create text prompts
text_prompts = [f"A street view photo in {country}" for country in country_names]

# Pre-encode all text prompts  
print(f"[StreetCLIP] Encoding {len(text_prompts)} country prompts...")
text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True, truncation=True)
text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
with torch.no_grad():
    text_outputs = model.get_text_features(**text_inputs)
    # Handle both tensor and dict outputs
    if hasattr(text_outputs, 'pooler_output'):
        text_features = text_outputs.pooler_output
    elif isinstance(text_outputs, torch.Tensor):
        text_features = text_outputs
    else:
        text_features = text_outputs
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# Load test images
test_dir = os.path.join('geolocation-prediction', 'test_images_sampled')
sample_sub = os.path.join('geolocation-prediction', 'sample_submission.csv')

with open(sample_sub, 'r') as f:
    reader = csv.DictReader(f)
    expected_ids = [row['image_id'] for row in reader]

print(f"[StreetCLIP] Processing {len(expected_ids)} test images...")

predictions = {}
batch_size = 16

for batch_start in range(0, len(expected_ids), batch_size):
    batch_ids = expected_ids[batch_start:batch_start+batch_size]
    images = []
    valid_ids = []
    
    for img_id in batch_ids:
        img_path = os.path.join(test_dir, img_id)
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(img)
                valid_ids.append(img_id)
            except:
                pass
    
    if not images:
        continue
    
    # Process images
    image_inputs = processor(images=images, return_tensors="pt", padding=True)
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
    
    with torch.no_grad():
        image_outputs = model.get_image_features(**image_inputs)
        if hasattr(image_outputs, 'pooler_output'):
            image_features = image_outputs.pooler_output
        elif isinstance(image_outputs, torch.Tensor):
            image_features = image_outputs
        else:
            image_features = image_outputs
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Cosine similarity
        similarity = (image_features @ text_features.T) * 100.0  # logit scale
        probs = torch.softmax(similarity, dim=1)
    
    probs_np = probs.cpu().numpy()
    
    for i, img_id in enumerate(valid_ids):
        # Probability-weighted average of country centroids
        weighted_lat = np.sum(probs_np[i] * country_coords[:, 0])
        weighted_lon = np.sum(probs_np[i] * country_coords[:, 1])
        
        # Top prediction confidence
        top_prob = probs_np[i].max()
        top_country = country_names[probs_np[i].argmax()]
        
        # Use top-5 probability-weighted coordinates for more precision
        top5_idx = np.argsort(probs_np[i])[-5:]
        top5_probs = probs_np[i][top5_idx]
        top5_probs = top5_probs / top5_probs.sum()
        
        pred_lat = np.sum(top5_probs * country_coords[top5_idx, 0])
        pred_lon = np.sum(top5_probs * country_coords[top5_idx, 1])
        
        # Radius: use wider for low confidence, tighter for high confidence
        # If top confidence > 0.5, use smaller radius
        if top_prob > 0.5:
            radius = 500.0
        elif top_prob > 0.3:
            radius = 1000.0
        elif top_prob > 0.15:
            radius = 1500.0
        else:
            radius = 2500.0
        
        predictions[img_id] = {
            'pred_lat': float(pred_lat),
            'pred_lon': float(pred_lon),
            'pred_radius_km': float(radius),
            'top_country': top_country,
            'top_prob': float(top_prob),
        }
    
    if (batch_start // batch_size + 1) % 5 == 0:
        print(f"  Processed {min(batch_start+batch_size, len(expected_ids))}/{len(expected_ids)} images")

# Save submission
print(f"\n[StreetCLIP] Saving submission...")
with open('submission.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['image_id', 'pred_lat', 'pred_lon', 'pred_radius_km'])
    writer.writeheader()
    for img_id in expected_ids:
        if img_id in predictions:
            p = predictions[img_id]
            writer.writerow({
                'image_id': img_id,
                'pred_lat': f"{p['pred_lat']:.6f}",
                'pred_lon': f"{p['pred_lon']:.6f}",
                'pred_radius_km': f"{p['pred_radius_km']:.1f}",
            })
        else:
            writer.writerow({
                'image_id': img_id,
                'pred_lat': '0.0',
                'pred_lon': '0.0',
                'pred_radius_km': '1000.0',
            })

# Print stats
lats = [predictions[k]['pred_lat'] for k in predictions]
lons = [predictions[k]['pred_lon'] for k in predictions]
radii = [predictions[k]['pred_radius_km'] for k in predictions]
top_countries = [predictions[k]['top_country'] for k in predictions]
top_probs = [predictions[k]['top_prob'] for k in predictions]

print(f"\n=== StreetCLIP Zero-Shot Results ===")
print(f"Lat range: [{min(lats):.2f}, {max(lats):.2f}]")
print(f"Lon range: [{min(lons):.2f}, {max(lons):.2f}]")
print(f"Radius range: [{min(radii):.1f}, {max(radii):.1f}] km")
print(f"Median radius: {np.median(radii):.1f} km")
print(f"Mean top confidence: {np.mean(top_probs):.3f}")
print(f"Median top confidence: {np.median(top_probs):.3f}")

# Country distribution
from collections import Counter
country_counts = Counter(top_countries)
print(f"\nTop 15 predicted countries:")
for country, count in country_counts.most_common(15):
    print(f"  {country}: {count}")

print(f"\n=== Confidence distribution ===")
for t in [0.1, 0.2, 0.3, 0.5, 0.7]:
    c = sum(1 for p in top_probs if p >= t)
    print(f"  >= {t}: {c} ({c/len(top_probs)*100:.1f}%)")

print(f"\n[StreetCLIP] Done! submission.csv has {len(predictions)} predictions.")
