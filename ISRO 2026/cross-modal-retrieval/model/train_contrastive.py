import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from model.encoder import CrossModalEncoder

class PairedSatelliteDataset(Dataset):
    def __init__(self, manifest_csv, split="train", data_root="."):
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.data_root = data_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        opt_path = os.path.join(self.data_root, row["optical_path"])
        sar_path = os.path.join(self.data_root, row["sar_path"])
        
        # Load numpy arrays
        opt_img = np.load(opt_path)  # (12, 224, 224)
        sar_img = np.load(sar_path)  # (2, 224, 224)
        
        # Convert to torch tensor
        opt_tensor = torch.from_numpy(opt_img).float()
        sar_tensor = torch.from_numpy(sar_img).float()
        
        return opt_tensor, sar_tensor, row["land_cover_label"], row["pair_id"]

def info_nce_loss(opt_embeds, sar_embeds, temperature=0.07):
    # opt_embeds: (B, D), sar_embeds: (B, D)
    # Both are already L2 normalized
    device = opt_embeds.device
    B = opt_embeds.size(0)
    
    # Compute similarity matrix
    logits = torch.matmul(opt_embeds, sar_embeds.t()) / temperature  # (B, B)
    
    # Target indices (matching indices along the diagonal)
    targets = torch.arange(B, device=device)
    
    loss_opt = F.cross_entropy(logits, targets)
    loss_sar = F.cross_entropy(logits.t(), targets)
    
    return (loss_opt + loss_sar) / 2.0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    manifest_csv = "data/manifest.csv"
    
    # Datasets
    train_dataset = PairedSatelliteDataset(manifest_csv, split="train")
    val_dataset = PairedSatelliteDataset(manifest_csv, split="val")
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Initialize encoder
    print("Initializing model...")
    model = CrossModalEncoder(pretrained_path='model/CROMA_base.pt').to(device)
    model.freeze_except_last_block()
    
    # Optimizer (only optimize parameters that require_grad)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)
    
    # Training Loop
    epochs = 5
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for step, (opt_imgs, sar_imgs, _, _) in enumerate(train_loader):
            opt_imgs = opt_imgs.to(device)
            sar_imgs = sar_imgs.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            opt_embeds = model.forward_optical(opt_imgs)
            sar_embeds = model.forward_sar(sar_imgs)
            
            # Loss
            loss = info_nce_loss(opt_embeds, sar_embeds)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}")
        
    # Save tuned model
    tuned_weights_path = "model/tuned_encoder.pt"
    torch.save(model.state_dict(), tuned_weights_path)
    print(f"Tuned model weights saved to {tuned_weights_path}")
    
    # Checkpoint 2: Validation alignment check and PCA plot
    print("Running Checkpoint 2 alignment verification...")
    model.eval()
    
    val_opt_embeds = []
    val_sar_embeds = []
    labels = []
    
    with torch.no_grad():
        for opt_imgs, sar_imgs, lbls, _ in val_loader:
            opt_imgs = opt_imgs.to(device)
            sar_imgs = sar_imgs.to(device)
            
            opt_emb = model.forward_optical(opt_imgs)
            sar_emb = model.forward_sar(sar_imgs)
            
            val_opt_embeds.append(opt_emb.cpu().numpy())
            val_sar_embeds.append(sar_emb.cpu().numpy())
            labels.extend(lbls)
            
    val_opt_embeds = np.concatenate(val_opt_embeds, axis=0)  # (N, D)
    val_sar_embeds = np.concatenate(val_sar_embeds, axis=0)  # (N, D)
    
    # Run PCA to project to 2D
    pca = PCA(n_components=2)
    all_embeds = np.concatenate([val_opt_embeds, val_sar_embeds], axis=0)
    pca.fit(all_embeds)
    
    opt_pca = pca.transform(val_opt_embeds)
    sar_pca = pca.transform(val_sar_embeds)
    
    # Plotting
    plt.figure(figsize=(10, 10))
    label_colors = {"water": "blue", "forest": "green", "urban": "red", "agriculture": "orange", "barren": "brown"}
    
    # Plot matching lines and points
    for i in range(len(val_opt_embeds)):
        c = label_colors.get(labels[i], "grey")
        # Draw line connecting same pair
        plt.plot([opt_pca[i, 0], sar_pca[i, 0]], [opt_pca[i, 1], sar_pca[i, 1]], color=c, alpha=0.3)
        
        # Plot optical point
        plt.scatter(opt_pca[i, 0], opt_pca[i, 1], color=c, marker='o', edgecolors='black', s=60, label="Optical" if i == 0 else "")
        # Plot SAR point
        plt.scatter(sar_pca[i, 0], sar_pca[i, 1], color=c, marker='x', s=60, label="SAR" if i == 0 else "")
        
    plt.title("PCA Projection of Joint Validation Embeddings\n(Circles = Optical, Crosses = SAR, Lines connect same scene)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plot_path = "data/embedding_tsne.png" # Using requested filename
    plt.savefig(plot_path)
    print(f"Alignment check plot saved to {plot_path}")
    print("Checkpoint 2 verification complete.")

if __name__ == "__main__":
    main()
