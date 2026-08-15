"""
Analyze submission.csv to understand what our model is actually predicting.
"""
import numpy as np
import csv

# Load submission
with open('submission.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

lats = [float(r['pred_lat']) for r in rows]
lons = [float(r['pred_lon']) for r in rows]
radii = [float(r['pred_radius_km']) for r in rows]

print(f"=== Submission Analysis ({len(rows)} images) ===")
print(f"\nLatitude:  min={min(lats):.2f}  max={max(lats):.2f}  mean={np.mean(lats):.2f}  std={np.std(lats):.2f}")
print(f"Longitude: min={min(lons):.2f}  max={max(lons):.2f}  mean={np.mean(lons):.2f}  std={np.std(lons):.2f}")
print(f"Radius:    min={min(radii):.1f}  max={max(radii):.1f}  mean={np.mean(radii):.1f}  median={np.median(radii):.1f}  std={np.std(radii):.1f}")

# Check how many unique predictions there are
unique_lats = len(set([f"{l:.2f}" for l in lats]))
unique_lons = len(set([f"{l:.2f}" for l in lons]))
print(f"\nUnique lat values (2dp): {unique_lats}")
print(f"Unique lon values (2dp): {unique_lons}")

# Distribution of predictions by region
europe = sum(1 for lat, lon in zip(lats, lons) if 35 <= lat <= 70 and -10 <= lon <= 40)
asia = sum(1 for lat, lon in zip(lats, lons) if 0 <= lat <= 60 and 40 <= lon <= 180)
americas = sum(1 for lat, lon in zip(lats, lons) if -60 <= lat <= 70 and -180 <= lon <= -30)
africa = sum(1 for lat, lon in zip(lats, lons) if -40 <= lat <= 35 and -20 <= lon <= 55)
oceania = sum(1 for lat, lon in zip(lats, lons) if -50 <= lat <= 0 and 100 <= lon <= 180)

print(f"\n--- Regional distribution ---")
print(f"Europe:    {europe}")
print(f"Asia:      {asia}")
print(f"Americas:  {americas}")
print(f"Africa:    {africa}")
print(f"Oceania:   {oceania}")

# Histogram of radii
print(f"\n--- Radius distribution ---")
for threshold in [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000]:
    count = sum(1 for r in radii if r <= threshold)
    print(f"  <= {threshold} km: {count} ({count/len(radii)*100:.1f}%)")

# Show first 10 predictions
print(f"\n--- First 10 predictions ---")
for r in rows[:10]:
    print(f"  {r['image_id']}: ({float(r['pred_lat']):.2f}, {float(r['pred_lon']):.2f}) r={float(r['pred_radius_km']):.0f}km")
