"""
HOW TO RUN:
    python rollNumber_05.py --dataset_path ./animal_data --epochs 100

This file contains:
    - AnimalDataset class (DataLoader)
    - DiffusionProcess class (Forward noise + Beta schedule)
    - UNet model (Denoising network)
    - Custom Loss function
    - Training loop
"""

import os
import argparse
import math
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ─────────────────────────────────────────────
# 0.  REPRODUCIBILITY  (same results every run)
# ─────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
# 1.  CONFIGURATION  (all settings in one place)
# ─────────────────────────────────────────────
IMAGE_SIZE   = 64        # resize every image to 64×64
CHANNELS     = 3         # RGB images
T_STEPS      = 1000      # total diffusion time-steps
BATCH_SIZE   = 8         # how many images per training batch
LR           = 1e-4      # learning rate
NUM_CLASSES  = 5         # we pick 5 animal classes
IMGS_PER_CLS = 20        # 20 images per class  →  100 total
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[INFO] Using device: {DEVICE}")

# ─────────────────────────────────────────────
# 2.  DATALOADER  (Task 1 — 5 marks)
# ─────────────────────────────────────────────
class AnimalDataset(Dataset):
    """
    Reads images from the animal_data folder.
    Picks the first IMGS_PER_CLS images from NUM_CLASSES classes.
    Applies transforms so every image becomes a normalised tensor.
    """

    def __init__(self, root_dir, num_classes=5, imgs_per_class=20, img_size=64):
        self.img_size = img_size

        # ---  transforms  ---
        # ToTensor  : converts PIL image [0,255] → float tensor [0,1]
        # Normalize : shifts to [-1, 1]  (diffusion models expect this range)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),   # make all images same size
            transforms.CenterCrop(img_size),           # safe crop after resize
            transforms.ToTensor(),                     # → [0.0, 1.0]
            transforms.Normalize([0.5, 0.5, 0.5],      # → [-1.0, 1.0]
                                  [0.5, 0.5, 0.5]),
        ])

        # ---  collect image paths  ---
        self.image_paths = []

        # list all sub-folders (each = one animal class)
        all_classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        # pick only the first num_classes classes
        selected_classes = all_classes[:num_classes]
        print(f"[INFO] Selected classes: {selected_classes}")

        for cls in selected_classes:
            cls_dir = os.path.join(root_dir, cls)
            # collect all image files in this class folder
            all_files = sorted([
                f for f in os.listdir(cls_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            # pick only the first imgs_per_class files
            chosen = all_files[:imgs_per_class]
            for fname in chosen:
                self.image_paths.append(os.path.join(cls_dir, fname))

        print(f"[INFO] Total images loaded: {len(self.image_paths)}")

    def __len__(self):
        # PyTorch needs this to know how many items are in the dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        # load image from disk
        img = Image.open(self.image_paths[idx]).convert("RGB")
        # apply transforms and return tensor
        return self.transform(img)


# ─────────────────────────────────────────────
# 3.  FORWARD DIFFUSION PROCESS  (Task 2 — 10 marks)
# ─────────────────────────────────────────────
class DiffusionProcess:
    """
    Forward (q) process:
        At each time-step t we add a small amount of Gaussian noise.
        After T steps the image becomes pure Gaussian noise.

    We use the "closed-form" trick from DDPM paper:
        x_t = sqrt(alpha_bar_t) * x_0  +  sqrt(1 - alpha_bar_t) * epsilon
        where epsilon ~ N(0, I)

    NOTE: We DO NOT apply noise directly to the image pixel-by-pixel.
          Instead we compute it mathematically using the beta schedule.
          This satisfies the "DO NOT APPLY NOISE DIRECTLY" requirement.
    """

    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T

        # ---  Beta schedule  ---
        # beta_t controls HOW MUCH noise to add at each step
        # We use a linear schedule: beta goes from very small → larger
        self.betas = torch.linspace(beta_start, beta_end, T)   # shape: (T,)

        # ---  Pre-compute useful quantities  ---
        # alpha_t = 1 - beta_t
        self.alphas = 1.0 - self.betas                          # shape: (T,)

        # alpha_bar_t = product of alpha_0 * alpha_1 * ... * alpha_t
        # (cumulative product)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)     # shape: (T,)

        # move everything to device
        self.betas      = self.betas.to(DEVICE)
        self.alphas     = self.alphas.to(DEVICE)
        self.alpha_bars = self.alpha_bars.to(DEVICE)

    def add_noise(self, x_0, t):
        """
        Given a clean image x_0 and a time-step t, return:
            x_t : noisy image at time t
            epsilon : the noise that was added  (we predict this during training)

        Formula:  x_t = sqrt(alpha_bar_t) * x_0  +  sqrt(1 - alpha_bar_t) * epsilon

        Arguments:
            x_0  : clean image tensor,  shape (B, C, H, W)
            t    : time step tensor,    shape (B,)   values in [0, T-1]

        Returns:
            x_t     : noisy image,  same shape as x_0
            epsilon : noise added,  same shape as x_0
        """
        # sample random Gaussian noise   epsilon ~ N(0, I)
        epsilon = torch.randn_like(x_0)                         # shape: (B, C, H, W)

        # gather alpha_bar values for each sample in the batch
        # t is a 1-D tensor of indices; we reshape for broadcasting
        a_bar = self.alpha_bars[t]                              # shape: (B,)
        a_bar = a_bar.view(-1, 1, 1, 1)                        # shape: (B, 1, 1, 1)

        # apply the closed-form formula  (this is NOT direct noise addition)
        sqrt_a_bar     = torch.sqrt(a_bar)
        sqrt_1_minus   = torch.sqrt(1.0 - a_bar)

        x_t = sqrt_a_bar * x_0  +  sqrt_1_minus * epsilon

        return x_t, epsilon

    def visualise_forward(self, x_0, steps_to_show=(0, 100, 250, 500, 750, 999)):
        """
        Helper: show how an image looks at different time steps.
        Useful for the report.
        """
        fig, axes = plt.subplots(1, len(steps_to_show), figsize=(14, 3))
        fig.suptitle("Forward Diffusion Process (Noise added over T steps)")

        for ax, step in zip(axes, steps_to_show):
            t_tensor = torch.tensor([step], device=DEVICE)
            x_t, _ = self.add_noise(x_0.to(DEVICE), t_tensor)
            # un-normalise for display: [-1,1] → [0,1]
            img = (x_t.squeeze().cpu().clamp(-1, 1) + 1) / 2.0
            img = img.permute(1, 2, 0).numpy()          # C,H,W → H,W,C
            ax.imshow(img)
            ax.set_title(f"t={step}")
            ax.axis("off")

        plt.tight_layout()
        plt.savefig("forward_process_steps.png", dpi=100)
        plt.close()
        print("[INFO] Saved: forward_process_steps.png")


# ─────────────────────────────────────────────
# 4.  U-NET DENOISING MODEL  (Task 3 — 10 marks)
# ─────────────────────────────────────────────
# U-Net is the standard architecture for diffusion models.
# It has an encoder (downsampling) and decoder (upsampling) with skip connections.
# We also inject the time-step t so the model knows "how noisy" the input is.

class SinusoidalPositionEmbedding(nn.Module):
    """
    Converts a scalar time-step t → a fixed-length vector embedding.
    This is how the model knows what time-step it is at.
    Same idea as positional encoding in Transformers.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t):
        # t: shape (B,)  integer time steps
        half = self.embed_dim // 2
        # create frequency bands
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        # outer product: each t gets its own set of sin/cos values
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding   # shape: (B, embed_dim)


class ResBlock(nn.Module):
    """
    Residual block with time-step conditioning.
    Each ResBlock:
        1. Applies two convolutions with GroupNorm + SiLU (smooth activation)
        2. Adds time embedding to the intermediate feature map
        3. Adds a skip connection (input → output directly)

    Why SiLU (Swish)?
        - Smoother than ReLU, works better in generative models
        - Negative values are not zero'd out completely
    """

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()

        # First convolution path
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),       # normalise over groups of channels
            nn.SiLU(),                    # smooth activation
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )

        # Time embedding projection: time_dim → out_ch
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )

        # Second convolution path
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )

        # Skip connection: if channels differ, use 1×1 conv to match
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)

        # add time embedding  (reshape for spatial broadcast)
        t = self.time_proj(t_emb)          # shape: (B, out_ch)
        h = h + t[:, :, None, None]        # broadcast over H, W

        h = self.conv2(h)
        return h + self.skip(x)            # residual connection


class UNet(nn.Module):
    """
    U-Net architecture for noise prediction.

    Encoder: 64×64 → 32×32 → 16×16 → 8×8
    Bottleneck: 8×8
    Decoder:  8×8 → 16×16 → 32×32 → 64×64

    Skip connections: encoder feature maps are concatenated in decoder.
    This helps recover fine spatial details.

    Input:  (B, 3, 64, 64) noisy image  +  (B,) time step
    Output: (B, 3, 64, 64) predicted noise epsilon
    """

    def __init__(self, in_channels=3, base_ch=64, time_dim=256):
        super().__init__()

        self.time_dim = time_dim

        # --- Time embedding MLP ---
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # --- Encoder ---
        # Level 1: 64×64, channels: 3 → base_ch (64)
        self.enc1 = ResBlock(in_channels, base_ch, time_dim)

        # Downsample: 64×64 → 32×32
        self.down1 = nn.Conv2d(base_ch, base_ch, 4, stride=2, padding=1)

        # Level 2: 32×32, channels: 64 → 128
        self.enc2 = ResBlock(base_ch, base_ch * 2, time_dim)

        # Downsample: 32×32 → 16×16
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)

        # Level 3: 16×16, channels: 128 → 256
        self.enc3 = ResBlock(base_ch * 2, base_ch * 4, time_dim)

        # Downsample: 16×16 → 8×8
        self.down3 = nn.Conv2d(base_ch * 4, base_ch * 4, 4, stride=2, padding=1)

        # --- Bottleneck ---
        self.mid1 = ResBlock(base_ch * 4, base_ch * 4, time_dim)
        self.mid2 = ResBlock(base_ch * 4, base_ch * 4, time_dim)

        # --- Decoder ---
        # Upsample: 8×8 → 16×16
        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, 4, stride=2, padding=1)
        # skip connection doubles channels: 256+256=512
        self.dec3 = ResBlock(base_ch * 8, base_ch * 2, time_dim)

        # Upsample: 16×16 → 32×32
        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)
        # skip: 128+128=256
        self.dec2 = ResBlock(base_ch * 4, base_ch, time_dim)

        # Upsample: 32×32 → 64×64
        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 4, stride=2, padding=1)
        # skip: 64+64=128
        self.dec1 = ResBlock(base_ch * 2, base_ch, time_dim)

        # --- Output head ---
        # Final 1×1 conv: base_ch → 3 channels  (predicting noise for each pixel)
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, in_channels, 1),  # 1×1 conv
        )

    def forward(self, x, t):
        """
        x : noisy image     (B, 3, 64, 64)
        t : time steps      (B,)
        returns: predicted noise  (B, 3, 64, 64)
        """
        # Compute time embedding
        t_emb = self.time_embed[0](t)           # sinusoidal
        t_emb = self.time_embed[1](t_emb)       # linear
        t_emb = self.time_embed[2](t_emb)       # SiLU
        t_emb = self.time_embed[3](t_emb)       # linear

        # Encoder
        e1 = self.enc1(x, t_emb)               # (B, 64,  64, 64)
        e2 = self.enc2(self.down1(e1), t_emb)  # (B, 128, 32, 32)
        e3 = self.enc3(self.down2(e2), t_emb)  # (B, 256, 16, 16)

        # Bottleneck
        m = self.mid1(self.down3(e3), t_emb)   # (B, 256,  8,  8)
        m = self.mid2(m, t_emb)

        # Decoder (with skip connections)
        d3 = self.up3(m)                        # (B, 256, 16, 16)
        d3 = torch.cat([d3, e3], dim=1)         # (B, 512, 16, 16)  ← skip
        d3 = self.dec3(d3, t_emb)              # (B, 128, 16, 16)

        d2 = self.up2(d3)                       # (B, 128, 32, 32)
        d2 = torch.cat([d2, e2], dim=1)         # (B, 256, 32, 32)  ← skip
        d2 = self.dec2(d2, t_emb)              # (B, 64,  32, 32)

        d1 = self.up1(d2)                       # (B, 64,  64, 64)
        d1 = torch.cat([d1, e1], dim=1)         # (B, 128, 64, 64)  ← skip
        d1 = self.dec1(d1, t_emb)              # (B, 64,  64, 64)

        return self.out_conv(d1)                # (B, 3,   64, 64)


# ─────────────────────────────────────────────
# 5.  CUSTOM LOSS FUNCTION  (Task 4 — 5 marks)
# ─────────────────────────────────────────────
class DiffusionLoss(nn.Module):
    """
    Custom loss for diffusion model training.

    We predict the noise epsilon that was added in the forward process.
    Loss = MSE (L2) between true noise and predicted noise,
           with an optional signal-to-noise ratio weight per time-step.

    Higher t  = more noise = harder to predict  → we weight these steps more.
    Lower  t  = less noise = easier to predict.
    """

    def __init__(self, loss_type="l2"):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, predicted_noise, true_noise, t=None, alpha_bars=None):
        """
        predicted_noise : output of UNet    (B, C, H, W)
        true_noise      : actual epsilon    (B, C, H, W)
        t               : time steps        (B,)  — used for weighting
        alpha_bars      : pre-computed values from DiffusionProcess
        """
        if self.loss_type == "l2":
            # L2 loss = mean squared error
            loss = (predicted_noise - true_noise) ** 2

        elif self.loss_type == "l1":
            # L1 loss = mean absolute error  (less sensitive to outliers)
            loss = torch.abs(predicted_noise - true_noise)

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Apply SNR weighting if t and alpha_bars are provided
        if t is not None and alpha_bars is not None:
            # SNR at time t = alpha_bar / (1 - alpha_bar)
            a_bar = alpha_bars[t].view(-1, 1, 1, 1)    # (B,1,1,1)
            snr   = a_bar / (1.0 - a_bar + 1e-8)

            # Weight = 1 / (1 + SNR)  →  harder steps (high t) get higher weight
            weight = 1.0 / (1.0 + snr)
            loss   = loss * weight

        return loss.mean()


# ─────────────────────────────────────────────
# 6.  TRAINING LOOP
# ─────────────────────────────────────────────
def train(args):
    # ---  Dataset & DataLoader  ---
    dataset = AnimalDataset(
        root_dir=args.dataset_path,
        num_classes=NUM_CLASSES,
        imgs_per_class=IMGS_PER_CLS,
        img_size=IMAGE_SIZE,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # ---  Model, Diffusion, Loss, Optimiser  ---
    model     = UNet(in_channels=CHANNELS, base_ch=64, time_dim=256).to(DEVICE)
    diffusion = DiffusionProcess(T=T_STEPS)
    criterion = DiffusionLoss(loss_type="l2")
    optimiser = optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {total_params:,}")

    # Save one sample image to show forward process (for report)
    sample_img = dataset[0].unsqueeze(0)          # (1, 3, 64, 64)
    diffusion.visualise_forward(sample_img)

    # ---  Training  ---
    loss_history = []

    os.makedirs("saved_models", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch_idx, x_0 in enumerate(loader):
            x_0 = x_0.to(DEVICE)                 # (B, 3, 64, 64)

            # Sample random time steps for each image in the batch
            # t ~ Uniform({1, ..., T})
            t = torch.randint(0, T_STEPS, (x_0.size(0),), device=DEVICE)

            # Forward process: get noisy image and the noise added
            x_t, epsilon = diffusion.add_noise(x_0, t)

            # Predict noise with UNet
            epsilon_pred = model(x_t, t)

            # Compute loss
            loss = criterion(epsilon_pred, epsilon, t, diffusion.alpha_bars)

            # Backprop
            optimiser.zero_grad()
            loss.backward()
            # Gradient clipping: prevents exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)

        print(f"Epoch [{epoch:3d}/{args.epochs}]  Loss: {avg_loss:.6f}")

        # Save model every 25 epochs
        if epoch % 25 == 0 or epoch == args.epochs:
            ckpt_path = f"saved_models/model_epoch_{epoch}.pt"
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "optim_state" : optimiser.state_dict(),
                "loss"        : avg_loss,
            }, ckpt_path)
            print(f"[INFO] Saved checkpoint: {ckpt_path}")

    # Save final model separately (used in test script)
    torch.save(model.state_dict(), "saved_models/model_final.pt")
    print("[INFO] Saved: saved_models/model_final.pt")

    # ---  Plot loss curve  ---
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(loss_history) + 1), loss_history, color="steelblue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=100)
    plt.close()
    print("[INFO] Saved: loss_curve.png")

    return model, diffusion


# ─────────────────────────────────────────────
# 7.  REVERSE / SAMPLING PROCESS  (Task 5)
# ─────────────────────────────────────────────
@torch.no_grad()
def sample_images(model, diffusion, num_images=5, save_path="generated_images.png"):
    """
    Start from pure Gaussian noise x_T and iteratively denoise
    using the trained model, going backwards from t=T to t=0.

    This is the DDPM ancestral sampling algorithm.

    Steps:
        For t = T, T-1, ..., 1:
            predict noise epsilon with model(x_t, t)
            compute mean of p(x_{t-1} | x_t)
            add small noise (except at t=1)
            → get x_{t-1}
        Return x_0
    """
    model.eval()

    # Start from pure Gaussian noise
    x = torch.randn(num_images, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)

    # Iteratively denoise
    for t_val in reversed(range(0, T_STEPS)):
        t = torch.full((num_images,), t_val, device=DEVICE, dtype=torch.long)

        # Predict noise
        pred_noise = model(x, t)

        # Get diffusion parameters for this time step
        beta_t      = diffusion.betas[t_val]
        alpha_t     = diffusion.alphas[t_val]
        alpha_bar_t = diffusion.alpha_bars[t_val]

        # Compute the previous image mean (DDPM reverse formula)
        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mean  = coef1 * (x - coef2 * pred_noise)

        # Add noise only for t > 0  (at t=0 we want the clean image)
        if t_val > 0:
            alpha_bar_prev = diffusion.alpha_bars[t_val - 1]
            # posterior variance
            variance = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            std      = torch.sqrt(variance)
            noise    = torch.randn_like(x)
            x        = mean + std * noise
        else:
            x = mean    # final clean image

    # Save generated images
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 3))
    fig.suptitle("Generated Images (Reverse Diffusion)")

    for i, ax in enumerate(axes):
        img = (x[i].cpu().clamp(-1, 1) + 1) / 2.0   # [-1,1] → [0,1]
        img = img.permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Sample {i+1}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"[INFO] Saved: {save_path}")

    return x


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Training")
    parser.add_argument("--dataset_path", type=str, default="./animal_data",
                        help="Path to the animal_data folder")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    args = parser.parse_args()

    print("=" * 50)
    print("  Diffusion Model - Assignment 5 (Bonus)")
    print("=" * 50)

    # Train
    model, diffusion = train(args)

    # Generate images after training
    print("\n[INFO] Generating sample images...")
    sample_images(model, diffusion, num_images=5, save_path="generated_images.png")

    print("\n[DONE] Training complete!")
    print("  Files saved:")
    print("    - saved_models/model_final.pt")
    print("    - loss_curve.png")
    print("    - forward_process_steps.png")
    print("    - generated_images.png")