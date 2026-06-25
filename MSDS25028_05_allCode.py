# ============================================================
# MSDS25028_05_allCode.py
# Deep Learning Assignment 5 (Bonus) - ALL CODE IN ONE FILE
# Name: Farwa | Roll Number: MSDS25028
# This file contains ALL code from the project combined.
# ============================================================

import os, math, argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# ---- CONFIG ----
class Config:
    IMAGE_SIZE = 64
    SELECTED_CLASSES = 5
    IMAGES_PER_CLASS = 20
    T = 1000
    BETA_START = 1e-4
    BETA_END = 0.02
    BATCH_SIZE = 8
    EPOCHS = 50
    LEARNING_RATE = 2e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_DIR = "saved_models"
    MODEL_PATH = os.path.join(SAVE_DIR, "diffusion_model.pth")
    RESULTS_DIR = "results"

cfg = Config()
os.makedirs(cfg.SAVE_DIR, exist_ok=True)
os.makedirs(cfg.RESULTS_DIR, exist_ok=True)


# ---- DATASET ----
class AnimalDataset(Dataset):
    def __init__(self, root_dir, num_classes=5, images_per_class=20, transform=None):
        self.transform = transform
        self.image_paths = []
        all_classes = sorted([d for d in os.listdir(root_dir)
                               if os.path.isdir(os.path.join(root_dir, d))])
        selected = all_classes[:num_classes]
        print(f"[INFO] Selected classes: {selected}")
        for cls in selected:
            cls_path = os.path.join(root_dir, cls)
            images = [os.path.join(cls_path, f) for f in os.listdir(cls_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            self.image_paths.extend(images[:images_per_class])
        print(f"[INFO] Total images: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def get_transforms():
    return transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


# ---- FORWARD DIFFUSION ----
class ForwardDiffusion:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, T).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def add_noise(self, x_0, t):
        """Closed-form forward diffusion - NOT direct noise addition"""
        epsilon = torch.randn_like(x_0)
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        sqrt_one_minus_ab = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)
        x_t = sqrt_ab * x_0 + sqrt_one_minus_ab * epsilon
        return x_t, epsilon

    def visualize_forward(self, x_0, save_path="results/forward_process.png"):
        steps_to_show = [0, 100, 250, 500, 750, 999]
        fig, axes = plt.subplots(1, len(steps_to_show), figsize=(15, 3))
        fig.suptitle("Forward Diffusion - Noise Addition Steps", fontsize=14)
        x_single = x_0[0:1]
        for idx, step in enumerate(steps_to_show):
            t = torch.tensor([step], dtype=torch.long).to(self.device)
            x_t, _ = self.add_noise(x_single, t)
            img = x_t[0].cpu().detach()
            img = (img * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
            axes[idx].imshow(img)
            axes[idx].set_title(f"t={step}")
            axes[idx].axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[INFO] Forward process saved: {save_path}")


# ---- U-NET MODEL ----
class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, t_dim=None):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding=1),
                                    nn.GroupNorm(8, out_c), nn.SiLU())
        self.conv2 = nn.Sequential(nn.Conv2d(out_c, out_c, 3, padding=1),
                                    nn.GroupNorm(8, out_c), nn.SiLU())
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, out_c)) if t_dim else None
        self.residual = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, t_emb=None):
        res = self.residual(x)
        h = self.conv1(x)
        if self.time_proj and t_emb is not None:
            h = h + self.time_proj(t_emb)[:, :, None, None]
        return self.conv2(h) + res


class UNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, time_emb_dim=256):
        super().__init__()
        bc = base_channels
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(bc),
            nn.Linear(bc, time_emb_dim), nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim))
        # Encoder
        self.enc1 = ConvBlock(in_channels, bc, time_emb_dim)
        self.down1 = nn.Conv2d(bc, bc, 4, 2, 1)
        self.enc2 = ConvBlock(bc, bc*2, time_emb_dim)
        self.down2 = nn.Conv2d(bc*2, bc*2, 4, 2, 1)
        self.enc3 = ConvBlock(bc*2, bc*4, time_emb_dim)
        self.down3 = nn.Conv2d(bc*4, bc*4, 4, 2, 1)
        # Bottleneck
        self.bottleneck1 = ConvBlock(bc*4, bc*8, time_emb_dim)
        self.bottleneck2 = ConvBlock(bc*8, bc*8, time_emb_dim)
        # Decoder
        self.up3 = nn.ConvTranspose2d(bc*8, bc*4, 4, 2, 1)
        self.dec3 = ConvBlock(bc*8, bc*4, time_emb_dim)
        self.up2 = nn.ConvTranspose2d(bc*4, bc*2, 4, 2, 1)
        self.dec2 = ConvBlock(bc*4, bc*2, time_emb_dim)
        self.up1 = nn.ConvTranspose2d(bc*2, bc, 4, 2, 1)
        self.dec1 = ConvBlock(bc*2, bc, time_emb_dim)
        self.output_conv = nn.Conv2d(bc, in_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t)
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.down1(e1), t_emb)
        e3 = self.enc3(self.down2(e2), t_emb)
        b = self.bottleneck2(self.bottleneck1(self.down3(e3), t_emb), t_emb)
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), t_emb)
        return self.output_conv(d1)


# ---- CUSTOM LOSS ----
class DiffusionLoss(nn.Module):
    def __init__(self, loss_type="l2"):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, predicted_noise, actual_noise):
        if self.loss_type == "l2":
            return torch.mean((predicted_noise - actual_noise) ** 2)
        elif self.loss_type == "l1":
            return torch.mean(torch.abs(predicted_noise - actual_noise))
        else:
            return nn.functional.huber_loss(predicted_noise, actual_noise)


# ---- TRAINING ----
def train_diffusion_model(dataset_path, args):
    dataset = AnimalDataset(dataset_path, cfg.SELECTED_CLASSES, cfg.IMAGES_PER_CLASS, get_transforms())
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    model = UNet(3, 64, 256).to(cfg.DEVICE)
    fd = ForwardDiffusion(cfg.T, cfg.BETA_START, cfg.BETA_END, cfg.DEVICE)
    loss_fn = DiffusionLoss("l2")
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    sample_batch = next(iter(dataloader)).to(cfg.DEVICE)
    fd.visualize_forward(sample_batch)
    loss_history = []
    best_loss = float('inf')
    for epoch in range(cfg.EPOCHS):
        model.train()
        epoch_loss = 0
        for x_0 in dataloader:
            x_0 = x_0.to(cfg.DEVICE)
            t = torch.randint(0, cfg.T, (x_0.shape[0],), device=cfg.DEVICE).long()
            x_t, epsilon = fd.add_noise(x_0, t)
            pred = model(x_t, t)
            loss = loss_fn(pred, epsilon)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        scheduler.step()
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{cfg.EPOCHS}] Loss: {avg_loss:.6f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'loss': best_loss,
                        'config': {'T': cfg.T}}, cfg.MODEL_PATH)
        if (epoch+1) % 10 == 0:
            generate_samples(model, fd, f"results/epoch_{epoch+1}_samples.png", 4)
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, 'b-')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Loss Curve')
    plt.grid(True, alpha=0.3); plt.savefig("results/loss_curve.png", dpi=150); plt.close()
    print(f"Training complete. Best loss: {best_loss:.6f}")
    return model, fd


# ---- GENERATION (TEST FUNCTION) ----
@torch.no_grad()
def generate_samples(model, forward_diffusion, save_path="results/generated_samples.png", num_samples=4):
    model.eval()
    x = torch.randn(num_samples, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE).to(cfg.DEVICE)
    betas = forward_diffusion.betas
    alphas = forward_diffusion.alphas
    alpha_bars = forward_diffusion.alpha_bars
    for t_val in reversed(range(cfg.T)):
        t_tensor = torch.full((num_samples,), t_val, dtype=torch.long, device=cfg.DEVICE)
        pred_noise = model(x, t_tensor)
        alpha_t = alphas[t_val]
        alpha_bar_t = alpha_bars[t_val]
        beta_t = betas[t_val]
        x = (1.0/torch.sqrt(alpha_t)) * (x - (beta_t/torch.sqrt(1.0-alpha_bar_t)) * pred_noise)
        if t_val > 0:
            posterior_var = beta_t * (1.0-alpha_bars[t_val-1]) / (1.0-alpha_bar_t)
            x = x + torch.sqrt(posterior_var) * torch.randn_like(x)
        if t_val % 200 == 0:
            print(f"  Sampling {cfg.T - t_val}/{cfg.T}...")
    x = (x.cpu() * 0.5 + 0.5).clamp(0, 1)
    fig, axes = plt.subplots(1, num_samples, figsize=(4*num_samples, 4))
    fig.suptitle("Generated Animal Images", fontsize=14)
    if num_samples == 1: axes = [axes]
    for i in range(num_samples):
        axes[i].imshow(x[i].permute(1,2,0).numpy())
        axes[i].set_title(f"Gen {i+1}")
        axes[i].axis('off')
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"Generated samples saved: {save_path}")
    return x


# ---- MAIN ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--generate_only', action='store_true')
    args = parser.parse_args()
    cfg.EPOCHS = args.epochs
    cfg.BATCH_SIZE = args.batch_size
    cfg.LEARNING_RATE = args.lr
    if args.generate_only:
        model = UNet(3, 64, 256).to(cfg.DEVICE)
        fd = ForwardDiffusion(cfg.T, device=cfg.DEVICE)
        ckpt = torch.load(cfg.MODEL_PATH, map_location=cfg.DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        generate_samples(model, fd, "results/final_generated.png", 8)
    else:
        model, fd = train_diffusion_model(args.dataset, args)
        ckpt = torch.load(cfg.MODEL_PATH, map_location=cfg.DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        generate_samples(model, fd, "results/final_generated.png", 8)

if __name__ == "__main__":
    main()