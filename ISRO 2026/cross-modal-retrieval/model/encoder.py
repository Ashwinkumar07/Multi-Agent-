import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.use_croma import PretrainedCROMA

class CrossModalEncoder(nn.Module):
    def __init__(self, pretrained_path='model/CROMA_base.pt', embed_dim=256, image_resolution=224):
        super().__init__()
        
        # Load optical encoder
        self.croma_opt = PretrainedCROMA(
            pretrained_path=pretrained_path,
            size='base',
            modality='optical',
            image_resolution=image_resolution
        )
        
        # Load SAR encoder
        self.croma_sar = PretrainedCROMA(
            pretrained_path=pretrained_path,
            size='base',
            modality='SAR',
            image_resolution=image_resolution
        )
        
        # Projection heads
        self.optical_proj = nn.Sequential(
            nn.Linear(self.croma_opt.encoder_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.sar_proj = nn.Sequential(
            nn.Linear(self.croma_sar.encoder_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward_optical(self, optical_images):
        """
        Args:
            optical_images: (batch_size, 12, 224, 224)
        Returns:
            Normalized shared embedding: (batch_size, 256)
        """
        outputs = self.croma_opt(optical_images=optical_images)
        opt_gap = outputs['optical_GAP']  # (batch_size, 768)
        proj = self.optical_proj(opt_gap)
        return F.normalize(proj, p=2, dim=-1)
        
    def forward_sar(self, SAR_images):
        """
        Args:
            SAR_images: (batch_size, 2, 224, 224)
        Returns:
            Normalized shared embedding: (batch_size, 256)
        """
        outputs = self.croma_sar(SAR_images=SAR_images)
        sar_gap = outputs['SAR_GAP']  # (batch_size, 768)
        proj = self.sar_proj(sar_gap)
        return F.normalize(proj, p=2, dim=-1)

    def freeze_except_last_block(self):
        # 1. Freeze all parameters in croma_opt and croma_sar
        for param in self.croma_opt.parameters():
            param.requires_grad = False
        for param in self.croma_sar.parameters():
            param.requires_grad = False
            
        # 2. Unfreeze the last transformer block in s1_encoder and s2_encoder
        for param in self.croma_sar.s1_encoder.transformer.layers[-1].parameters():
            param.requires_grad = True
        for param in self.croma_sar.GAP_FFN_s1.parameters():
            param.requires_grad = True
                
        for param in self.croma_opt.s2_encoder.transformer.layers[-1].parameters():
            param.requires_grad = True
        for param in self.croma_opt.GAP_FFN_s2.parameters():
            param.requires_grad = True
                
        # 3. Keep projection heads fully trainable
        for param in self.optical_proj.parameters():
            param.requires_grad = True
        for param in self.sar_proj.parameters():
            param.requires_grad = True
            
        # Log counts
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"Frozen: {frozen:,} parameters. Trainable: {trainable:,} parameters.")
