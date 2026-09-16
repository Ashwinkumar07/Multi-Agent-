import os
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_synthetic_scene(scene_id, size=224):
    """
    Generates a paired S2 (12-channel) and S1 (2-channel) image.
    Land cover classes:
    0: Water (dark in S2, very dark in S1)
    1: Forest (dark green/high NIR in S2, moderate in S1)
    2: Urban (grey/bright in S2, very bright in S1)
    3: Agriculture (bright green/very high NIR in S2, moderate/variable in S1)
    4: Barren (sandy/brown in S2, low-moderate in S1)
    """
    # Create spatial masks (segments) in the scene
    # We will start with a dominant background class
    bg_class = np.random.choice([1, 3, 4]) # Forest, Agriculture, or Barren
    
    # Base map of classes
    class_map = np.full((size, size), bg_class, dtype=np.uint8)
    
    # Add a water body (river or lake)
    water_type = np.random.choice(["river", "lake", "none"])
    if water_type == "river":
        # Draw a curvy line for river
        points = []
        x = 0
        y = np.random.randint(size//4, 3*size//4)
        for i in range(5):
            points.append((x, y))
            x += size // 4
            y += np.random.randint(-size//6, size//6)
            y = np.clip(y, 0, size-1)
        points = np.array(points, dtype=np.int32)
        cv2.polylines(class_map, [points], False, 0, thickness=np.random.randint(15, 30))
    elif water_type == "lake":
        # Draw a circle/ellipse for lake
        cv2.circle(class_map, (np.random.randint(size//4, 3*size//4), np.random.randint(size//4, 3*size//4)), 
                   np.random.randint(20, 50), 0, -1)
                   
    # Add an urban area (some blocks/circles)
    if np.random.rand() > 0.3:
        for _ in range(np.random.randint(1, 4)):
            ux = np.random.randint(size//6, 5*size//6)
            uy = np.random.randint(size//6, 5*size//6)
            uw = np.random.randint(25, 60)
            uh = np.random.randint(25, 60)
            cv2.rectangle(class_map, (ux, uy), (ux+uw, uy+uh), 2, -1)
            
    # Add agricultural fields (grid of rectangles)
    if np.random.rand() > 0.3:
        for _ in range(np.random.randint(2, 6)):
            ax = np.random.randint(size//6, 5*size//6)
            ay = np.random.randint(size//6, 5*size//6)
            aw = np.random.randint(20, 45)
            ah = np.random.randint(20, 45)
            # Avoid overwriting water
            mask = (class_map == bg_class)
            field = np.zeros_like(class_map)
            cv2.rectangle(field, (ax, ay), (ax+aw, ay+ah), 3, -1)
            class_map[mask & (field == 3)] = 3
            
    # Add some patches of forest or barren land
    other_class = 1 if bg_class != 1 else 4
    if np.random.rand() > 0.4:
        for _ in range(np.random.randint(1, 4)):
            cx = np.random.randint(size//6, 5*size//6)
            cy = np.random.randint(size//6, 5*size//6)
            cr = np.random.randint(15, 40)
            mask = (class_map == bg_class)
            patch = np.zeros_like(class_map)
            cv2.circle(patch, (cx, cy), cr, other_class, -1)
            class_map[mask & (patch == other_class)] = other_class

    # Determine dominant class for labeling
    unique, counts = np.unique(class_map, return_counts=True)
    dominant_class_id = unique[np.argmax(counts)]
    class_labels = {0: "water", 1: "forest", 2: "urban", 3: "agriculture", 4: "barren"}
    dominant_label = class_labels[dominant_class_id]

    # Generate optical (12 channels)
    # Spectral signatures (approximate reflectance values [0, 10000])
    # B1-B12 mapping
    signatures_s2 = {
        0: [1500, 1200, 1000,  800,  600,  500,  400,  300,  250,  200,  150,  100],  # Water
        1: [ 400,  300,  600,  350, 1500, 3000, 3500, 3800, 4000, 3800, 1800,  800],  # Forest
        2: [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2250, 2300, 2800, 2600],  # Urban
        3: [ 500,  400,  800,  450, 2000, 4000, 4800, 5000, 5200, 5000, 2200, 1000],  # Agriculture
        4: [1200, 1400, 1700, 2000, 2200, 2400, 2600, 2800, 2900, 3000, 3800, 3400]   # Barren
    }
    
    optical = np.zeros((12, size, size), dtype=np.float32)
    for c_id, sig in signatures_s2.items():
        mask = (class_map == c_id)
        if not np.any(mask):
            continue
        for ch in range(12):
            base_val = sig[ch] / 10000.0 # scale to [0, 1]
            # Add spatial texture (low frequency noise)
            noise = np.random.normal(0, 0.03, size=(size, size))
            # Smooth noise to create texture
            noise = cv2.GaussianBlur(noise, (15, 15), 0)
            optical[ch][mask] = np.clip(base_val + noise[mask], 0.0, 1.0)

    # Generate SAR (2 channels: VV, VH in dB)
    # VV / VH signatures in dB
    # Water: low (-22, -28)
    # Forest: moderate (-10, -16)
    # Urban: high (-5, -11)
    # Agriculture: moderate-high (-8, -14)
    # Barren: moderate-low (-15, -22)
    signatures_s1 = {
        0: [-22.0, -28.0],
        1: [-11.0, -17.0],
        2: [-4.0,  -9.0],
        3: [-9.0,  -15.0],
        4: [-15.0, -22.0]
    }
    
    sar = np.zeros((2, size, size), dtype=np.float32)
    for c_id, sig in signatures_s1.items():
        mask = (class_map == c_id)
        if not np.any(mask):
            continue
        for ch in range(2):
            base_val = sig[ch]
            # Add SAR speckle noise (Gaussian noise in dB space)
            noise = np.random.normal(0, 1.8, size=(size, size))
            # Moderate smoothing to simulate spatial resolution
            noise = cv2.GaussianBlur(noise, (5, 5), 0)
            sar[ch][mask] = base_val + noise[mask]

    # Normalize SAR to [0, 1]
    # Standard clipping: VV from [-25, 0], VH from [-30, -5]
    # Let's use a uniform clipping of [-25.0, 0.0] for VV and [-30.0, -5.0] for VH
    sar_norm = np.zeros_like(sar)
    # Channel 0 (VV): Clip to [-25, 0] -> Scale to [0, 1]
    sar_norm[0] = np.clip((sar[0] - (-25.0)) / 25.0, 0.0, 1.0)
    # Channel 1 (VH): Clip to [-30, -5] -> Scale to [0, 1]
    sar_norm[1] = np.clip((sar[1] - (-30.0)) / 25.0, 0.0, 1.0)

    # Let's create an RGB visualization of optical
    # Sentinel-2 RGB: B4 (Red) is ch 3, B3 (Green) is ch 2, B2 (Blue) is ch 1
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(optical[3] * 255, 0, 255).astype(np.uint8) # R
    rgb[..., 1] = np.clip(optical[2] * 255, 0, 255).astype(np.uint8) # G
    rgb[..., 2] = np.clip(optical[1] * 255, 0, 255).astype(np.uint8) # B
    
    # S1 visualization: VV channel scaled to uint8
    sar_viz = np.clip(sar_norm[0] * 255, 0, 255).astype(np.uint8)
    
    return optical, sar_norm, rgb, sar_viz, dominant_label

def main():
    np.random.seed(42)
    os.makedirs("data/raw/optical", exist_ok=True)
    os.makedirs("data/raw/sar", exist_ok=True)
    os.makedirs("data/processed/optical", exist_ok=True)
    os.makedirs("data/processed/sar", exist_ok=True)
    os.makedirs("data/visual", exist_ok=True)

    manifest_path = "data/manifest.csv"
    num_pairs = 500
    
    print(f"Generating {num_pairs} paired synthetic satellite images...")
    
    with open(manifest_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "optical_path", "sar_path", "land_cover_label", "split"])
        
        for i in range(num_pairs):
            opt, sar, rgb, sar_viz, label = generate_synthetic_scene(i)
            
            # Determine train/val/test split
            if i < 400:
                split = "train"
            elif i < 450:
                split = "val"
            else:
                split = "test"
                
            opt_filename = f"opt_{i:04d}.npy"
            sar_filename = f"sar_{i:04d}.npy"
            rgb_filename = f"rgb_{i:04d}.png"
            sar_png_filename = f"sar_{i:04d}.png"
            
            opt_path = os.path.join("data/processed/optical", opt_filename)
            sar_path = os.path.join("data/processed/sar", sar_filename)
            rgb_path = os.path.join("data/visual", rgb_filename)
            sar_png_path = os.path.join("data/visual", sar_png_filename)
            
            # Save raw data (.npy)
            np.save(opt_path, opt)
            np.save(sar_path, sar)
            
            # Save visual images (.png)
            # OpenCV expects BGR
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(rgb_path, bgr)
            cv2.imwrite(sar_png_path, sar_viz)
            
            writer.writerow([i, opt_path, sar_path, label, split])
            
    print(f"Data generation complete. Manifest saved to {manifest_path}")

    # PHASE 1 CHECKPOINT: load 5 random pairs and plot side-by-side
    print("Running Checkpoint 1 verification...")
    fig, axes = plt.subplots(5, 2, figsize=(8, 20))
    for idx, i in enumerate(np.random.choice(num_pairs, 5, replace=False)):
        rgb_path = os.path.join("data/visual", f"rgb_{i:04d}.png")
        sar_path = os.path.join("data/visual", f"sar_{i:04d}.png")
        
        rgb = cv2.imread(rgb_path)
        sar = cv2.imread(sar_path, cv2.IMREAD_GRAYSCALE)
        
        axes[idx, 0].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        axes[idx, 0].set_title(f"Pair {i} - Optical RGB")
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(sar, cmap='gray')
        axes[idx, 1].set_title(f"Pair {i} - SAR VV")
        axes[idx, 1].axis('off')
        
    plt.tight_layout()
    checkpoint_plot_path = "data/alignment_check.png"
    plt.savefig(checkpoint_plot_path)
    print(f"Checkpoint plot saved to {checkpoint_plot_path}")
    print("Checkpoint 1 verification complete.")

if __name__ == "__main__":
    main()
