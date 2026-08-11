"""
finetune_real.py
────────────────
Fine-tunes on real COPD DVFs (copd01-08).
Normalisation: global stats from motion voxels of training cases,
               saved by compute_real_stats.py
"""

import os
import sys
import math
import time
import datetime
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler


# ── Paths ─────────────────────────────────────────────────────────────────────
FIELDS_DIR   = "/mimer/NOBACKUP/groups/caim1/dafne/datasets/real/fields"
REAL_STATS   = "/mimer/NOBACKUP/groups/caim1/dafne/datasets/real/real_dvf_stats.npy"
PRETRAIN_DIR = "/mimer/NOBACKUP/groups/caim1/dafne/checkpoints/16_condchannels"
SAVE_DIR     = "/mimer/NOBACKUP/groups/caim1/dafne/checkpoints_real/16_condchannels"
VIS_DIR      = "/mimer/NOBACKUP/groups/caim1/dafne/visual_real/16_condchannels"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(VIS_DIR,  exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
LR               = 1e-5
WEIGHT_DECAY     = 1e-4
EPOCHS           = 150
BATCH_SIZE       = 1
SMOOTH_LAMBDA    = 1e-7
SAMPLES_PER_EPOCH = 40
MOTION_THRESHOLD  = 0.5   # mm

TRAIN_CASES = [0, 1, 2, 3, 4, 5, 6, 7]
TEST_CASES  = [8, 9]
DS_SHAPE    = (128, 128, 64)


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


# ── DVF loading ───────────────────────────────────────────────────────────────
def load_and_preprocess_dvf(case_idx: int, dvf_mean: float, dvf_std: float) -> torch.Tensor:
    path = os.path.join(FIELDS_DIR, f"copd{case_idx + 1:02d}.nii.gz")
    dvf  = nib.load(path).get_fdata().astype(np.float32)
    dvf  = dvf[..., [0, 2, 1]]
    dvf  = dvf.transpose(3, 0, 1, 2)
    dvf  = torch.from_numpy(dvf)

    D = dvf.shape[-1]
    if D < 128:
        pad = 128 - D
        dvf = F.pad(dvf, (pad//2, pad - pad//2))
    elif D > 128:
        c = (D - 128) // 2
        dvf = dvf[..., c:c+128]

    dvf = F.interpolate(dvf.unsqueeze(0), size=DS_SHAPE,
                        mode='trilinear', align_corners=False).squeeze(0)

    # Normalise with global real stats
    dvf_norm = (dvf - dvf_mean) / dvf_std
    return dvf_norm


def compute_motion_mask(dvf_norm: torch.Tensor, dvf_std: float) -> torch.Tensor:
    """Convert threshold to normalised space and mask."""
    norm_threshold = MOTION_THRESHOLD / dvf_std
    mag = dvf_norm.norm(dim=0, keepdim=True)
    return (mag > norm_threshold).float()


# ── Dataset ───────────────────────────────────────────────────────────────────
class COPDDataset(Dataset):
    def __init__(self, case_indices, dvf_mean, dvf_std):
        self.samples = []
        for idx in case_indices:
            dvf = load_and_preprocess_dvf(idx, dvf_mean, dvf_std)
            self.samples.append(dvf)
            dvf_mm = dvf * dvf_std + dvf_mean
            mag    = dvf_mm.norm(dim=0).mean().item()
            print(f"  copd{idx+1:02d}  shape={tuple(dvf.shape)}  mean_mag={mag:.3f} mm  "
                  f"normalised range=[{dvf.min():.2f}, {dvf.max():.2f}]")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


# ── Model ─────────────────────────────────────────────────────────────────────
class DiffusionSchedule:
    def __init__(self, timesteps=1000, device="cpu"):
        self.timesteps = timesteps
        self.betas     = torch.linspace(1e-4, 0.02, timesteps).to(device)
        self.alphas    = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_bar           = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1. - self.alpha_bar)

    def q_sample(self, x0, t, noise):
        a  = self.sqrt_alphas_bar[t][:, None, None, None, None]
        am = self.sqrt_one_minus_alphas_bar[t][:, None, None, None, None]
        return a * x0 + am * noise


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=t.device) *
                        -(math.log(10000) / (half_dim - 1)))
        emb = t[:, None] * emb[None, :] * 2 * math.pi
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)


class SliceToVolume(nn.Module):
    def __init__(self, out_channels=16):
        super().__init__()
        self.coronal_encoder  = nn.Sequential(nn.Conv2d(3,16,3,padding=1), nn.ReLU(),
                                               nn.Conv2d(16,out_channels,3,padding=1))
        self.sagittal_encoder = nn.Sequential(nn.Conv2d(3,16,3,padding=1), nn.ReLU(),
                                               nn.Conv2d(16,out_channels,3,padding=1))
        self.fusion = nn.Conv3d(out_channels, out_channels, 3, padding=1)

    def forward(self, coronal, sagittal, D):
        B=coronal.shape[0]; W=coronal.shape[2]; H=sagittal.shape[2]
        cor = self.coronal_encoder(coronal).permute(0,1,3,2).unsqueeze(3).expand(-1,-1,-1,H,-1)
        sag = self.sagittal_encoder(sagittal).permute(0,1,3,2).unsqueeze(4).expand(-1,-1,-1,-1,W)
        return F.relu(self.fusion(cor + sag))


class ResidualBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels,channels,3,padding=1), nn.GroupNorm(8,channels), nn.ReLU(),
            nn.Conv3d(channels,channels,3,padding=1), nn.GroupNorm(8,channels))

    def forward(self, x):
        return F.relu(x + self.block(x))


class UNet3D_Diffusion(nn.Module):
    def __init__(self, cond_channels=16):
        super().__init__()
        self.time_mlp      = nn.Sequential(TimeEmbedding(64), nn.Linear(64,64), nn.ReLU())
        self.time_proj1    = nn.Linear(64, 32)
        self.time_proj2    = nn.Linear(64, 64)
        self.time_proj_mid = nn.Linear(64, 64)
        self.enc1 = nn.Conv3d(3+cond_channels, 32, 3, padding=1)
        self.enc2 = nn.Conv3d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.mid     = nn.Conv3d(64, 64, 3, padding=1)
        self.mid_res = ResidualBlock3D(64)
        self.dec1 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.out  = nn.Conv3d(32, 3, 3, padding=1)

    def forward(self, x, cond, t):
        x    = x.permute(0,1,4,2,3).contiguous()
        cond = cond.permute(0,1,2,3,4).contiguous()
        t    = t.float() / 1000.0
        t_emb = self.time_mlp(t)
        x     = torch.cat([x, cond], dim=1)
        s1 = self.time_proj1(t_emb)[:,:,None,None,None]
        x1 = F.relu(self.enc1(x) * (1 + s1))
        x2 = F.relu(self.enc2(self.pool(x1)) + self.time_proj2(t_emb)[:,:,None,None,None])
        xm = F.relu(self.mid(x2) + self.time_proj_mid(t_emb)[:,:,None,None,None])
        xm = self.mid_res(xm)
        x  = self.dec1(xm) + x1
        return self.out(x).permute(0,1,3,4,2).contiguous()


class DiffusionModelManager(nn.Module):
    def __init__(self, cond_channels=16):
        super().__init__()
        self.slice_to_vol = SliceToVolume(out_channels=cond_channels)
        self.unet         = UNet3D_Diffusion(cond_channels=cond_channels)

    def forward(self, x_t, coronal_2d, sagittal_2d, t):
        D = x_t.shape[-1]
        return self.unet(x_t, self.slice_to_vol(coronal_2d, sagittal_2d, D), t)


# ── Loss ──────────────────────────────────────────────────────────────────────
def gradient_loss(dvf):
    dh = (dvf[:,:,1:,:,:] - dvf[:,:,:-1,:,:]).pow(2).mean()
    dw = (dvf[:,:,:,1:,:] - dvf[:,:,:,:-1,:]).pow(2).mean()
    dd = (dvf[:,:,:,:,1:] - dvf[:,:,:,:,:-1]).pow(2).mean()
    return (dh + dw + dd) / 3.0


def train_one_epoch(model, diffusion, loader, optimizer, device, dvf_std):
    model.train()
    total_loss = total_recon = total_smooth = 0.0

    for dvf in loader:
        dvf = dvf.to(device)
        B   = dvf.shape[0]

        with torch.no_grad():
            mask = compute_motion_mask(dvf[0], dvf_std).unsqueeze(0).to(device)

        t         = torch.randint(0, diffusion.timesteps, (B,), device=device).long()
        noise     = torch.randn_like(dvf)
        dvf_noisy = diffusion.q_sample(dvf, t, noise)

        mid_h    = dvf.shape[2] // 2
        mid_w    = dvf.shape[3] // 2
        coronal  = dvf[:, :, mid_h, :, :]
        sagittal = dvf[:, :, :, mid_w, :]

        pred_noise  = model(dvf_noisy, coronal, sagittal, t)
        recon_loss  = (F.mse_loss(pred_noise, noise, reduction='none') * mask).sum() \
                      / (mask.sum() * pred_noise.shape[1] + 1e-8)

        a       = diffusion.sqrt_alphas_bar[t].view(B,1,1,1,1)
        am      = diffusion.sqrt_one_minus_alphas_bar[t].view(B,1,1,1,1)
        pred_x0 = (dvf_noisy - am * pred_noise) / a
        smooth  = gradient_loss(pred_x0 * mask)
        loss    = recon_loss + SMOOTH_LAMBDA * smooth

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss   += loss.item()
        total_recon  += recon_loss.item()
        total_smooth += smooth.item()

    n = len(loader)
    return total_loss/n, total_recon/n, total_smooth/n


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log_path   = os.path.join(SAVE_DIR, "finetune_real.log")
    tee        = Tee(log_path)
    sys.stdout = tee

    print("=" * 60)
    print(f"Real Data Fine-tuning — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"LR={LR}  WD={WEIGHT_DECAY}  epochs={EPOCHS}  smooth_lambda={SMOOTH_LAMBDA}")
    print(f"Train: {[f'copd{i+1:02d}' for i in TRAIN_CASES]}")
    print(f"Test:  {[f'copd{i+1:02d}' for i in TEST_CASES]}")
    print("=" * 60)
    print(f"Device: {device}")

    # Load real global stats
    stats    = np.load(REAL_STATS, allow_pickle=True).item()
    dvf_mean = float(stats["mean"])
    dvf_std  = float(stats["std"])
    print(f"Real DVF stats (motion voxels): mean={dvf_mean:.4f} mm, std={dvf_std:.4f} mm")

    print("\n=== Loading training cases ===")
    train_ds     = COPDDataset(TRAIN_CASES, dvf_mean, dvf_std)
    sampler      = RandomSampler(train_ds, replacement=True, num_samples=SAMPLES_PER_EPOCH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    print(f"Train batches per epoch: {len(train_loader)}")

    diffusion = DiffusionSchedule(timesteps=1000, device=device)
    model     = DiffusionModelManager(cond_channels=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    best_train = float("inf")
    best_epoch = 0
    start_epoch = 1
    train_loss_history = []

    resume_path = os.path.join(SAVE_DIR, "latest_checkpoint.pth")
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt.get('scheduler_state_dict', scheduler.state_dict()))
        best_train         = ckpt.get('best_train', float("inf"))
        best_epoch         = ckpt.get('best_epoch', 0)
        train_loss_history = ckpt.get('train_loss_history', [])
        start_epoch        = ckpt['epoch'] + 1
        print(f"Resumed from epoch {ckpt['epoch']} | Best: {best_train:.5f}")
    else:
        pretrain_path = os.path.join(PRETRAIN_DIR, "best_model.pth")
        model.load_state_dict(torch.load(pretrain_path, map_location=device))
        print(f"Loaded pretrained: {pretrain_path}")

    print(f"\n{'Epoch':>6}  {'Train':>10}  {'MSE':>10}  {'Smooth':>10}  {'Time':>8}")
    print("-" * 55)

    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()
        train_loss, recon_loss, smooth_loss = train_one_epoch(
            model, diffusion, train_loader, optimizer, device, dvf_std)
        elapsed = time.time() - t0
        scheduler.step()
        train_loss_history.append(train_loss)

        is_best = train_loss < best_train
        if is_best:
            best_train = train_loss
            best_epoch = epoch
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'train_loss': train_loss, 'dvf_mean': dvf_mean, 'dvf_std': dvf_std},
                       os.path.join(SAVE_DIR, "best_model.pth"))

        tag = " *** BEST ***" if is_best else ""
        print(f"{epoch:>6}  {train_loss:>10.5f}  {recon_loss:>10.5f}  "
              f"{smooth_loss:>10.5f}  {elapsed:>6.1f}s{tag}", flush=True)

        if epoch % 5 == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_train': best_train, 'best_epoch': best_epoch,
                        'train_loss_history': train_loss_history,
                        'dvf_mean': dvf_mean, 'dvf_std': dvf_std},
                       os.path.join(SAVE_DIR, "latest_checkpoint.pth"))

        if epoch % 25 == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'train_loss': train_loss, 'dvf_mean': dvf_mean, 'dvf_std': dvf_std},
                       os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch:04d}.pth"))

        if epoch % 10 == 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(range(1, len(train_loss_history)+1), train_loss_history)
            ax.axvline(best_epoch, color='r', linestyle='--', label=f"Best epoch {best_epoch}")
            ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
            ax.set_title(f"Real Fine-tuning — epoch {epoch}"); ax.legend(); ax.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(VIS_DIR, "loss_curves_real.png"), dpi=120, bbox_inches='tight')
            plt.close()

    print(f"\nDone. Best: {best_train:.5f} at epoch {best_epoch}")
    sys.stdout = tee.terminal
    tee.close()


if __name__ == "__main__":
    main()