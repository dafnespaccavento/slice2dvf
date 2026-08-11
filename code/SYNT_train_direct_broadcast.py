"""

Training script for the DIRECT BROADCAST architecture:
  - No learned 2D encoders
  - Raw DVF slice values broadcast directly into 3D volume
  - Coronal slice repeated along W axis, sagittal along H axis
  - Combined by element-wise average (no learned fusion)
  - UNet cond_channels = 3 (same as DVF channels)

"""

import os
import sys
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm.auto import tqdm


# ── Logger ────────────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log      = open(log_path, 'a', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = "/mimer/NOBACKUP/groups/caim1/dafne/datasets/smooth_synthetic_downsampled_v2"
SAVE_DIR = "/mimer/NOBACKUP/groups/caim1/dafne/checkpoints/newdataset/16_condchannels/direct_broadcast"
VIS_DIR  = "/mimer/NOBACKUP/groups/caim1/dafne/visual/newdataset/16_condchannels/direct_broadcast"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(VIS_DIR,  exist_ok=True)


# ── Dataset (unchanged from train.py) ─────────────────────────────────────────
class SyntheticDVFDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.field_dir = os.path.join(root_dir, split, "b")
        self.slice_dir = os.path.join(root_dir, split, "a")
        self.ids = sorted([
            f.split("_")[1].split(".")[0]
            for f in os.listdir(self.field_dir)
            if f.startswith("field_")
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        i = self.ids[idx]
        dvf    = np.load(os.path.join(self.field_dir, f"field_{i}.npy"))
        slices = np.load(os.path.join(self.slice_dir,  f"slice_{i}.npy"),
                         allow_pickle=True).item()
        cond_coronal  = slices["coronal"]
        cond_sagittal = slices["sagittal"]
        mid_w     = slices["indices"]["mid_w"]
        W         = dvf.shape[2]
        slice_pos = mid_w / (W - 1)
        dvf           = torch.from_numpy(dvf).float()
        cond_coronal  = torch.from_numpy(cond_coronal).float()
        cond_sagittal = torch.from_numpy(cond_sagittal).float()
        return cond_coronal, cond_sagittal, dvf, slice_pos


# ── Diffusion schedule (unchanged) ────────────────────────────────────────────
class DiffusionSchedule:
    def __init__(self, timesteps=1000, device="cpu"):
        self.timesteps = timesteps
        self.device    = device
        self.betas     = torch.linspace(1e-4, 0.02, timesteps).to(device)
        self.alphas    = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_bar           = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1. - self.alpha_bar)

    def q_sample(self, x0, t, noise):
        sqrt_ab           = self.sqrt_alphas_bar[t][:, None, None, None, None]
        sqrt_one_minus_ab = self.sqrt_one_minus_alphas_bar[t][:, None, None, None, None]
        return sqrt_ab * x0 + sqrt_one_minus_ab * noise


# ── Time embedding (unchanged) ────────────────────────────────────────────────
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.exp(
            torch.arange(half_dim, device=t.device) * -(math.log(10000) / (half_dim - 1))
        )
        emb = t[:, None] * emb[None, :] * 2 * math.pi
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)


# ── ResidualBlock3D (unchanged) ───────────────────────────────────────────────
class ResidualBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.ReLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x):
        return F.relu(x + self.block(x))


# ── Direct broadcast SliceToVolume ────────────────────────────────────────────
# No learned parameters. Raw DVF slice values are broadcast into the volume
# and averaged. cond_channels = 3 (same as the number of DVF components).
class SliceToVolume(nn.Module):
    def __init__(self):
        super().__init__()
        # no parameters — purely deterministic broadcast

    def forward(self, coronal, sagittal, D, slice_pos=None):
        """
        coronal  : (B, 3, H, D)   — raw DVF values at mid-coronal plane
        sagittal : (B, 3, W, D)   — raw DVF values at mid-sagittal plane
        Returns  : (B, 3, H, W, D) conditioning volume
        """
        B, C, H, _ = coronal.shape
        W = sagittal.shape[2]

        # broadcast coronal along W: every column gets the same coronal slice
        cor_vol = coronal.unsqueeze(3).expand(B, C, H, W, D)   # (B,3,H,W,D)

        # broadcast sagittal along H: every row gets the same sagittal slice
        sag_vol = sagittal.unsqueeze(2).expand(B, C, H, W, D)  # (B,3,H,W,D)

        # simple average — no learned combination
        return (cor_vol + sag_vol) / 2.0


# ── UNet (cond_channels=3 for direct broadcast) ───────────────────────────────
class UNet3D_Diffusion(nn.Module):
    def __init__(self, cond_channels=3):
        super().__init__()
        self.time_mlp = nn.Sequential(
            TimeEmbedding(64),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.time_proj1    = nn.Linear(64, 64)
        self.time_proj2    = nn.Linear(64, 128)
        self.time_proj_mid = nn.Linear(64, 128)

        self.enc1 = nn.Conv3d(3 + cond_channels, 64, 3, padding=1)
        self.enc2 = nn.Conv3d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool3d(2)

        self.mid     = nn.Conv3d(128, 128, 3, padding=1)
        self.mid_res = ResidualBlock3D(128)

        self.dec1 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.out  = nn.Conv3d(64, 3, 3, padding=1)

    def forward(self, x, cond, t):
        x    = x.permute(0, 1, 4, 2, 3).contiguous()
        cond = cond.permute(0, 1, 4, 2, 3).contiguous()
        t_emb = self.time_mlp(t.float())
        x     = torch.cat([x, cond], dim=1)

        scale1 = self.time_proj1(t_emb)[:, :, None, None, None]
        x1 = F.relu(self.enc1(x) * (1 + scale1))
        x2 = F.relu(self.enc2(self.pool(x1)) + self.time_proj2(t_emb)[:, :, None, None, None])

        x_mid = F.relu(self.mid(x2) + self.time_proj_mid(t_emb)[:, :, None, None, None])
        x_mid = self.mid_res(x_mid)

        x = self.dec1(x_mid) + x1
        x = self.out(x)
        return x.permute(0, 1, 3, 4, 2).contiguous()


# ── Model manager ─────────────────────────────────────────────────────────────
class DiffusionModelManager(nn.Module):
    def __init__(self):
        super().__init__()
        self.slice_to_vol = SliceToVolume()
        self.unet         = UNet3D_Diffusion(cond_channels=3)

    def forward(self, x_t, coronal_2d, sagittal_2d, t, slice_pos=None):
        D       = x_t.shape[-1]
        cond_3d = self.slice_to_vol(coronal_2d, sagittal_2d, D, slice_pos)
        return self.unet(x_t, cond_3d, t)


# ── Loss (unchanged) ──────────────────────────────────────────────────────────
def compute_gradient_loss(field, penalty='l2'):
    dh = torch.abs(field[:, :, 1:, :,  :] - field[:, :, :-1, :,  :])
    dw = torch.abs(field[:, :, :,  1:, :] - field[:, :, :,  :-1, :])
    dd = torch.abs(field[:, :, :,  :, 1:] - field[:, :, :,  :, :-1])
    if penalty == 'l2':
        dh, dw, dd = dh**2, dw**2, dd**2
    return (dh.mean() + dw.mean() + dd.mean()) / 3


def diffusion_loss(model, diffusion, x0, cond_coronal, cond_sagittal, slice_pos,
                   lambda_smooth=1e-4, fixed_t=None):
    B = x0.shape[0]
    if fixed_t is None:
        t = torch.randint(0, diffusion.timesteps, (B,), device=x0.device)
    else:
        t = torch.full((B,), fixed_t, device=x0.device, dtype=torch.long)
    noise = torch.randn_like(x0)
    x_t   = diffusion.q_sample(x0, t, noise)

    noise_pred = model(x_t, cond_coronal, cond_sagittal, t, slice_pos)
    mse_loss   = F.mse_loss(noise_pred, noise)

    s_ab = diffusion.sqrt_alphas_bar[t].view(B, 1, 1, 1, 1)
    s_om = diffusion.sqrt_one_minus_alphas_bar[t].view(B, 1, 1, 1, 1)
    pred_x0   = (x_t - s_om * noise_pred) / s_ab
    grad_loss = compute_gradient_loss(pred_x0)
    total_loss = mse_loss + lambda_smooth * grad_loss

    return total_loss, mse_loss, grad_loss


def get_smooth_lambda(epoch, total_epochs, initial_lambda=1e-4, final_lambda=1e-6):
    if epoch < 200:
        return initial_lambda
    decay_range = total_epochs - 200
    step = (initial_lambda - final_lambda) / decay_range
    return max(initial_lambda - step * (epoch - 200), final_lambda)


# ── Stats loader (unchanged) ──────────────────────────────────────────────────
def load_dvf_stats():
    stats_path = os.path.join(DATA_DIR, "dvf_stats.npy")
    stats = np.load(stats_path, allow_pickle=True).item()
    mean  = float(stats["mean"])
    std   = float(stats["std"])
    print(f"Loaded DVF stats: mean={mean:.6f}, std={std:.6f}")
    return mean, std


# ── Fixed validation timesteps (unchanged) ────────────────────────────────────
VAL_TIMESTEPS = [50, 250, 500, 750, 999]


def val_epoch(model, diffusion, val_loader, device, current_lambda):
    model.eval()
    val_total = val_mse = val_smooth = 0.0
    n_batches = 0

    with torch.no_grad():
        for cond_coronal, cond_sagittal, dvf, slice_pos in val_loader:
            dvf           = dvf.to(device)
            cond_coronal  = cond_coronal.to(device)
            cond_sagittal = cond_sagittal.to(device)
            slice_pos     = slice_pos.to(device)

            batch_total = batch_mse = batch_smooth = 0.0
            for fixed_t in VAL_TIMESTEPS:
                total_loss, mse_loss, smooth_loss = diffusion_loss(
                    model, diffusion, dvf, cond_coronal, cond_sagittal, slice_pos,
                    lambda_smooth=current_lambda, fixed_t=fixed_t
                )
                batch_total  += total_loss.item()
                batch_mse    += mse_loss.item()
                batch_smooth += smooth_loss.item() * current_lambda

            val_total  += batch_total  / len(VAL_TIMESTEPS)
            val_mse    += batch_mse    / len(VAL_TIMESTEPS)
            val_smooth += batch_smooth / len(VAL_TIMESTEPS)
            n_batches  += 1

    val_total  /= n_batches
    val_mse    /= n_batches
    val_smooth /= n_batches
    return val_total, val_mse, val_smooth


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log_path   = os.path.join(SAVE_DIR, "training.log")
    tee        = Tee(log_path)
    sys.stdout = tee
    print(f"{'='*60}")
    print(f"Run started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Architecture: Direct broadcast (no learned encoders, cond_channels=3)")
    print(f"Log file:    {log_path}")
    print(f"{'='*60}")
    print(f"Using device: {device}")

    train_dataset = SyntheticDVFDataset(DATA_DIR, split="train")
    val_size      = int(0.15 * len(train_dataset))
    train_size    = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    test_dataset = SyntheticDVFDataset(DATA_DIR, split="test")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                              num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False,
                              num_workers=8, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=1,  shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    dvf_mean, dvf_std = load_dvf_stats()

    diffusion = DiffusionSchedule(timesteps=1000, device=device)
    model     = DiffusionModelManager().to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=1000, eta_min=1e-6)

    train_loss_history,   val_loss_history   = [], []
    train_mse_history,    val_mse_history    = [], []
    train_smooth_history, val_smooth_history = [], []

    num_epochs    = 1000
    best_val_loss = float("inf")
    start_epoch   = 1

    resume_path = os.path.join(SAVE_DIR, "latest_checkpoint.pth")
    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint.get('scheduler_state_dict',
                                                  scheduler.state_dict()))
        best_val_loss        = checkpoint['best_val_loss']
        train_loss_history   = checkpoint.get('train_loss_history',   [])
        val_loss_history     = checkpoint.get('val_loss_history',     [])
        train_mse_history    = checkpoint.get('train_mse_history',    [])
        val_mse_history      = checkpoint.get('val_mse_history',      [])
        train_smooth_history = checkpoint.get('train_smooth_history', [])
        val_smooth_history   = checkpoint.get('val_smooth_history',   [])
        start_epoch          = checkpoint['epoch'] + 1
        if 'dvf_mean' in checkpoint:
            dvf_mean = checkpoint['dvf_mean']
            dvf_std  = checkpoint['dvf_std']
        print(f"Resumed from epoch {checkpoint['epoch']} | Best val loss: {best_val_loss:.4f}")
    else:
        print("No checkpoint found — starting from scratch")

    for epoch in range(start_epoch, num_epochs + 1):
        current_lambda = get_smooth_lambda(epoch, num_epochs)

        model.train()
        train_total = train_mse = train_smooth = 0.0
        for cond_coronal, cond_sagittal, dvf, slice_pos in train_loader:
            dvf           = dvf.to(device)
            cond_coronal  = cond_coronal.to(device)
            cond_sagittal = cond_sagittal.to(device)
            slice_pos     = slice_pos.to(device)

            total_loss, mse_loss, smooth_loss = diffusion_loss(
                model, diffusion, dvf, cond_coronal, cond_sagittal, slice_pos,
                lambda_smooth=current_lambda
            )
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_total  += total_loss.item()
            train_mse    += mse_loss.item()
            train_smooth += smooth_loss.item() * current_lambda

        n_train       = len(train_loader)
        train_total  /= n_train
        train_mse    /= n_train
        train_smooth /= n_train
        train_loss_history.append(train_total)
        train_mse_history.append(train_mse)
        train_smooth_history.append(train_smooth)

        val_total, val_mse, val_smooth = val_epoch(
            model, diffusion, val_loader, device, current_lambda)
        val_loss_history.append(val_total)
        val_mse_history.append(val_mse)
        val_smooth_history.append(val_smooth)

        scheduler.step()

        is_best = val_total < best_val_loss
        if is_best:
            best_val_loss = val_total
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))

        if epoch % 5 == 0:
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss':        best_val_loss,
                'train_loss_history':   train_loss_history,
                'val_loss_history':     val_loss_history,
                'train_mse_history':    train_mse_history,
                'val_mse_history':      val_mse_history,
                'train_smooth_history': train_smooth_history,
                'val_smooth_history':   val_smooth_history,
                'current_lambda':       current_lambda,
                'dvf_mean':             dvf_mean,
                'dvf_std':              dvf_std,
            }, os.path.join(SAVE_DIR, "latest_checkpoint.pth"))

        if epoch % 50 == 0:
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss':        best_val_loss,
                'train_loss_history':   train_loss_history,
                'val_loss_history':     val_loss_history,
                'train_mse_history':    train_mse_history,
                'val_mse_history':      val_mse_history,
                'train_smooth_history': train_smooth_history,
                'val_smooth_history':   val_smooth_history,
                'current_lambda':       current_lambda,
                'dvf_mean':             dvf_mean,
                'dvf_std':              dvf_std,
            }, os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch:04d}.pth"))

        tag = " *** BEST ***" if is_best else ""
        print(
            f"Epoch {epoch:03d}/{num_epochs} | λ={current_lambda:.2e} | "
            f"LR={scheduler.get_last_lr()[0]:.2e} | "
            f"Train [Total={train_total:.5f} MSE={train_mse:.5f} Smooth={train_smooth:.5f}] | "
            f"Val   [Total={val_total:.5f} MSE={val_mse:.5f} Smooth={val_smooth:.5f}]"
            f"{tag}"
        )

        if epoch % 10 == 0:
            epochs_ax = range(1, len(train_loss_history) + 1)
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            axes[0].plot(epochs_ax, train_loss_history, label="Train")
            axes[0].plot(epochs_ax, val_loss_history,   label="Val")
            axes[0].set_title("Total Loss"); axes[0].legend(); axes[0].grid(True)
            axes[1].plot(epochs_ax, train_mse_history,  label="Train")
            axes[1].plot(epochs_ax, val_mse_history,    label="Val")
            axes[1].set_title("MSE Loss");   axes[1].legend(); axes[1].grid(True)
            axes[2].plot(epochs_ax, train_smooth_history, label="Train")
            axes[2].plot(epochs_ax, val_smooth_history,   label="Val")
            axes[2].set_title("Smoothness Loss (λ-weighted)")
            axes[2].legend(); axes[2].grid(True)
            plt.suptitle(f"Direct broadcast — training curves — epoch {epoch}", fontsize=13)
            plt.tight_layout()
            plt.savefig(os.path.join(VIS_DIR, "loss_curves.png"),
                        dpi=120, bbox_inches='tight')
            plt.close()

    print("Training complete.")
    sys.stdout = tee.terminal
    tee.close()


if __name__ == "__main__":
    main()
