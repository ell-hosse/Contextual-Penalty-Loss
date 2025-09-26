"""
benchmark_cpl.py
Compare pixel-wise vs class-wise Contextual Penalty Loss (CPL) timing
on synthetic BDD100K-like data (19 classes + ignore=255).
Runs 1 epoch twice (pixel-wise then class-wise) and prints timings.
"""

import time, random
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from cploss import CPLoss as CPLossPixelWise
from cploss import FastCPLoss as CPLossClassWise

# -----------------------------
# Repro & device
# -----------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] Device: {DEVICE}")

# -----------------------------
# BDD100K classes, palette, ignore
# -----------------------------
BDD_CLASSES = [
    "road","sidewalk","building","wall","fence","pole","traffic light","traffic sign",
    "vegetation","terrain","sky","person","rider","car","truck","bus","train","motorcycle","bicycle"
]
NUM_CLASSES = len(BDD_CLASSES)
IGNORE_INDEX = 255

BDD_COLORS = [
    (128,64,128), (244, 35,232), ( 70, 70, 70), (102,102,156), (190,153,153),
    (153,153,153), (250,170, 30), (220,220,  0), (107,142, 35), (152,251,152),
    ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
    (  0, 60,100), (  0, 80,100), (  0,  0,230), (119, 11, 32),
]
COLOR2ID: Dict[Tuple[int,int,int], int] = {c:i for i,c in enumerate(BDD_COLORS)}

# -----------------------------
# Contextual Similarity (S) for BDD100K
# -----------------------------
GROUPS = {
    "ground": ["road","sidewalk","terrain"],
    "construction": ["building","wall","fence"],
    "object": ["pole","traffic light","traffic sign"],
    "nature": ["vegetation","terrain"],
    "sky": ["sky"],
    "human": ["person","rider"],
    "vehicle": ["car","truck","bus","train","motorcycle","bicycle"]
}
RELATED = {
    ("ground","construction"): 0.45,
    ("ground","vehicle"): 0.35,
    ("human","vehicle"): 0.30,
    ("construction","object"): 0.40,
    ("nature","ground"): 0.40,
}
WITHIN = 0.82
LOW    = 0.08

def build_similarity_matrix(classes: List[str]) -> torch.Tensor:
    idx = {c:i for i,c in enumerate(classes)}
    C = len(classes)
    S = torch.full((C,C), LOW, dtype=torch.float32)
    for i in range(C):
        S[i,i] = 1.0
    # within-group
    for members in GROUPS.values():
        ids = [idx[c] for c in members if c in idx]
        for i in ids:
            for j in ids:
                if i != j:
                    S[i,j] = max(float(S[i,j]), WITHIN)
    # related groups
    for (g1,g2), val in RELATED.items():
        ids1 = [idx[c] for c in GROUPS[g1] if c in idx]
        ids2 = [idx[c] for c in GROUPS[g2] if c in idx]
        for i in ids1:
            for j in ids2:
                S[i,j] = max(float(S[i,j]), val)
                S[j,i] = max(float(S[j,i]), val)
    S = 0.5*(S + S.T)
    return S

# -----------------------------
# Synthetic Dataset (4 samples)
# - Random RGB images
# - Random masks with class indices in [0..18], plus some IGNORE_INDEX=255
# - Also keeps an RGB "color mask" (not used for training) to mimic BDD palette
# -----------------------------
class SyntheticBDD(Dataset):
    def __init__(self, n_samples=4, size=512, ignore_ratio=0.05):
        self.size = size
        self.N = n_samples
        H = W = size

        # Random images in [0,1]
        self.images = torch.rand(n_samples, 3, H, W, dtype=torch.float32)

        # Random class indices [0..C-1]
        masks = torch.randint(0, NUM_CLASSES, (n_samples, H, W), dtype=torch.long)

        # Sprinkle ignore pixels (255)
        if ignore_ratio > 0:
            ignore_mask = torch.rand(n_samples, H, W) < ignore_ratio
            masks[ignore_mask] = IGNORE_INDEX
        self.masks = masks

        # Build a color mask (for completeness / sanity)
        palette = torch.tensor(BDD_COLORS, dtype=torch.uint8)  # (19,3)
        rgb = torch.zeros(n_samples, H, W, 3, dtype=torch.uint8)
        valid = masks != IGNORE_INDEX
        # map valid indices to colors
        idx_valid = masks[valid]
        rgb[valid] = palette[idx_valid]
        # ignore stays black (0,0,0)
        self.color_masks = rgb

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        return self.images[i], self.masks[i]

# -----------------------------
# U-Net (compact)
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=NUM_CLASSES, base=32):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)
        self.d2 = DoubleConv(base, base*2)
        self.d3 = DoubleConv(base*2, base*4)
        self.d4 = DoubleConv(base*4, base*8)
        self.b  = DoubleConv(base*8, base*16)

        self.pool = nn.MaxPool2d(2)

        self.u4 = DoubleConv(base*16 + base*8, base*8)
        self.u3 = DoubleConv(base*8  + base*4, base*4)
        self.u2 = DoubleConv(base*4  + base*2, base*2)
        self.u1 = DoubleConv(base*2  + base,   base)

        self.out = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))
        cb = self.b(self.pool(c4))

        u4 = F.interpolate(cb, scale_factor=2, mode="bilinear", align_corners=False)
        u4 = self.u4(torch.cat([u4, c4], dim=1))
        u3 = F.interpolate(u4, scale_factor=2, mode="bilinear", align_corners=False)
        u3 = self.u3(torch.cat([u3, c3], dim=1))
        u2 = F.interpolate(u3, scale_factor=2, mode="bilinear", align_corners=False)
        u2 = self.u2(torch.cat([u2, c2], dim=1))
        u1 = F.interpolate(u2, scale_factor=2, mode="bilinear", align_corners=False)
        u1 = self.u1(torch.cat([u1, c1], dim=1))
        return self.out(u1)

# -----------------------------
# Simple train loop for 1 epoch
# -----------------------------
def train_one_epoch(model, loader, loss_fn, optimizer, amp=True):
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and DEVICE.type == "cuda"))

    start = time.perf_counter()
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(amp and DEVICE.type == "cuda")):
            logits = model(imgs)
            loss = loss_fn(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # make timings fair on GPU
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    return end - start

# -----------------------------
# Main
# -----------------------------
def main():
    # Synthetic data (4 samples of 512x512)
    SIZE = 512
    dataset = SyntheticBDD(n_samples=10, size=SIZE, ignore_ratio=0.05)
    # small batch (2) to create 2 steps/epoch
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=(DEVICE.type=="cuda"))

    # Build similarity & two losses
    S = build_similarity_matrix(BDD_CLASSES).to(DEVICE)
    loss_px = CPLossPixelWise(S, ignore_index=IGNORE_INDEX, reduction='mean', from_logits=True).to(DEVICE)
    loss_cl = CPLossClassWise(S, ignore_index=IGNORE_INDEX, reduction='mean', from_logits=True).to(DEVICE)

    # Model + optimizer (same init for fair comparison)
    model = UNet(in_ch=3, num_classes=NUM_CLASSES, base=32).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # Save initial weights to replay for class-wise run
    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Run 1: Pixel-wise
    t1 = train_one_epoch(model, loader, loss_px, opt, amp=True)
    print(f"[Time] Pixel-wise CPL  (1 epoch, 10 samples): {t1:.3f} s")

    # Reset model & optimizer for second run
    model.load_state_dict({k: v.to(DEVICE) for k, v in init_state.items()})
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # Run 2: Class-wise
    t2 = train_one_epoch(model, loader, loss_cl, opt, amp=True)
    print(f"[Time] Class-wise CPL  (1 epoch, 10 samples): {t2:.3f} s")

    # Speed ratio
    ratio = (t1 / max(t2, 1e-9))
    print(f"[Result] Speedup (pixel/class) = {ratio:.2f}×  ( >1.0 means pixel-wise slower )")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
