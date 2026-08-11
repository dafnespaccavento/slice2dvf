"""

Fine-tunes the cross-attention + distance-channel architecture on real COPD
DVFs (copd01-08).

Changes vs original:
  - Rolling average checkpoint criterion (ROLLING_N=5) instead of single-epoch best
  - Loss curve shows per-epoch + rolling average
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
PRETRAIN_DIR = '/mimer/NOBACKUP/groups/caim1/dafne/checkpoints/newdataset/16_condchannels/concat_slicetovolume/cross_a'
SAVE_DIR     = "/mimer/NOBACKUP/groups/caim1/dafne/checkpoints_real/16_condchannels/cross_a_v2/swap/small"
VIS_DIR      = "/mimer/NOBACKUP/groups/caim1/dafne/visual_real/16_condchannels/cross_a_v2/small"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(VIS_DIR,  exist_ok=True)


# ── Hyperparameters ───────────────────────────────────────────────────────────
LR                  = 1e-5
WEIGHT_DECAY        = 1e-4
EPOCHS              = 150
BATCH_SIZE          = 1
SMOOTH_LAMBDA       = 1e-5
SAMPLES_PER_EPOCH   = 100
SLICE_FEAT_CHANNELS = 16    # UNet receives this + 2 distance channels = 18
ROLLING_N           = 5     # rolling average window for checkpoint criterion

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
def load_and_preprocess_dvf(case_idx, dvf_mean, dvf_std):
    path = os.path.join(FIELDS_DIR, f"copd{case_idx + 1:02d}.nii.gz")
    dvf  = nib.load(path).get_fdata().astype(np.float32)
    dvf  = dvf.transpose(3, 0, 1, 2)
    dvf  = torch.from_numpy(dvf)
    dvf  = F.interpolate(dvf.unsqueeze(0), size=DS_SHAPE,
                         mode='trilinear', align_corners=False).squeeze(0)
    return (dvf - dvf_mean) / dvf_std


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


# ── Diffusion schedule ────────────────────────────────────────────────────────
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


# ── Model building blocks ─────────────────────────────────────────────────────
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


class ResidualBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels), nn.ReLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels))

    def forward(self, x):
        return F.relu(x + self.block(x))


# ── CrossAttention2D ──────────────────────────────────────────────────────────
ATTN_POOL_SIZE = 128

class CrossAttention2D(nn.Module):
    def __init__(self, channels, num_heads=4, pool_size=ATTN_POOL_SIZE):
        super().__init__()
        self.num_heads = num_heads
        self.pool_size = pool_size
        self.head_dim  = channels // num_heads
        assert channels % num_heads == 0
        self.q_cor   = nn.Linear(channels, channels, bias=False)
        self.k_sag   = nn.Linear(channels, channels, bias=False)
        self.v_sag   = nn.Linear(channels, channels, bias=False)
        self.q_sag   = nn.Linear(channels, channels, bias=False)
        self.k_cor   = nn.Linear(channels, channels, bias=False)
        self.v_cor   = nn.Linear(channels, channels, bias=False)
        self.out_cor  = nn.Linear(channels, channels, bias=False)
        self.out_sag  = nn.Linear(channels, channels, bias=False)
        self.norm_cor = nn.LayerNorm(channels)
        self.norm_sag = nn.LayerNorm(channels)

    def _pool_seq(self, feat):
        B, C, S, D = feat.shape
        seq    = feat.permute(0, 2, 3, 1).reshape(B, S * D, C)
        seq_t  = seq.permute(0, 2, 1)
        pooled = F.adaptive_avg_pool1d(seq_t, self.pool_size)
        return pooled.permute(0, 2, 1)

    def _flash_attn(self, q, k, v, B, heads, head_dim):
        def reshape(x): return x.view(B, -1, heads, head_dim).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return out.transpose(1, 2).reshape(B, -1, heads * head_dim)

    def forward(self, cor_feat, sag_feat):
        B, C, H, D = cor_feat.shape
        _, _, W, _ = sag_feat.shape
        cor_pool = self._pool_seq(cor_feat)
        sag_pool = self._pool_seq(sag_feat)
        cor_attn = self.out_cor(self._flash_attn(
            self.q_cor(cor_pool), self.k_sag(sag_pool), self.v_sag(sag_pool),
            B, self.num_heads, self.head_dim))
        sag_attn = self.out_sag(self._flash_attn(
            self.q_sag(sag_pool), self.k_cor(cor_pool), self.v_cor(cor_pool),
            B, self.num_heads, self.head_dim))
        cor_res = F.interpolate(cor_attn.permute(0,2,1), size=H*D,
                                mode='linear', align_corners=False).permute(0,2,1)
        sag_res = F.interpolate(sag_attn.permute(0,2,1), size=W*D,
                                mode='linear', align_corners=False).permute(0,2,1)
        cor_seq = self.norm_cor(cor_feat.permute(0,2,3,1).reshape(B,H*D,C) + cor_res)
        sag_seq = self.norm_sag(sag_feat.permute(0,2,3,1).reshape(B,W*D,C) + sag_res)
        cor_feat = cor_seq.reshape(B, H, D, C).permute(0,3,1,2)
        sag_feat = sag_seq.reshape(B, W, D, C).permute(0,3,1,2)
        return cor_feat, sag_feat


# ── Distance channels ─────────────────────────────────────────────────────────
def make_distance_channels(B, H, W, D, mid_h, mid_w, device):
    h_idx    = torch.arange(H, device=device).float()
    w_idx    = torch.arange(W, device=device).float()
    dist_cor = (h_idx - mid_h).abs() / max(mid_h, H - 1 - mid_h)
    dist_sag = (w_idx - mid_w).abs() / max(mid_w, W - 1 - mid_w)
    dist_cor = dist_cor[:, None, None].expand(H, W, D)
    dist_sag = dist_sag[None, :, None].expand(H, W, D)
    dist_vol = torch.stack([dist_cor, dist_sag], dim=0)
    return dist_vol.unsqueeze(0).expand(B, -1, -1, -1, -1)


# ── SliceToVolume (cross-attention version) ───────────────────────────────────
class SliceToVolume(nn.Module):
    def __init__(self, out_channels=SLICE_FEAT_CHANNELS):
        super().__init__()
        self.coronal_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, out_channels, 3, padding=1))
        self.sagittal_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, out_channels, 3, padding=1))
        self.cross_attn = CrossAttention2D(channels=out_channels, num_heads=4)
        self.pos_mlp = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, out_channels))
        self.fusion = nn.Conv3d(out_channels * 2, out_channels, 3, padding=1)
        self.refine = nn.Sequential(
            ResidualBlock3D(out_channels), ResidualBlock3D(out_channels))

    def forward(self, coronal, sagittal, D, slice_pos, mid_h, mid_w):
        B = coronal.shape[0]
        cor_feat = self.coronal_encoder(coronal)
        sag_feat = self.sagittal_encoder(sagittal)
        H = cor_feat.shape[2]; W = sag_feat.shape[2]
        cor_feat, sag_feat = self.cross_attn(cor_feat, sag_feat)
        pos_emb  = self.pos_mlp(slice_pos.float().unsqueeze(1))
        pos_bias = pos_emb[:, :, None, None]
        cor_feat = cor_feat * (1 + pos_bias)
        cor_vol  = cor_feat.unsqueeze(3).expand(-1, -1, -1, W, -1)
        sag_vol  = sag_feat.unsqueeze(2).expand(-1, -1, H, -1, -1)
        fused    = F.relu(self.fusion(torch.cat([cor_vol, sag_vol], dim=1)))
        fused    = self.refine(fused)
        dist     = make_distance_channels(B, H, W, D, mid_h, mid_w, coronal.device)
        return torch.cat([fused, dist], dim=1)


# ── UNet ──────────────────────────────────────────────────────────────────────
class UNet3D_Diffusion(nn.Module):
    def __init__(self, cond_channels):
        super().__init__()
        self.time_mlp      = nn.Sequential(TimeEmbedding(64), nn.Linear(64,64), nn.ReLU())
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
        x    = x.permute(0,1,4,2,3).contiguous()
        cond = cond.permute(0,1,4,2,3).contiguous()
        t_emb = self.time_mlp(t.float())
        x     = torch.cat([x, cond], dim=1)
        s1 = self.time_proj1(t_emb)[:,:,None,None,None]
        x1 = F.relu(self.enc1(x) * (1 + s1))
        x2 = F.relu(self.enc2(self.pool(x1)) + self.time_proj2(t_emb)[:,:,None,None,None])
        xm = F.relu(self.mid(x2) + self.time_proj_mid(t_emb)[:,:,None,None,None])
        xm = self.mid_res(xm)
        x  = self.dec1(xm) + x1
        return self.out(x).permute(0,1,3,4,2).contiguous()


# ── DiffusionModelManager ─────────────────────────────────────────────────────
class DiffusionModelManager(nn.Module):
    def __init__(self, slice_feat_channels=SLICE_FEAT_CHANNELS):
        super().__init__()
        cond_channels     = slice_feat_channels + 2
        self.slice_to_vol = SliceToVolume(out_channels=slice_feat_channels)
        self.unet         = UNet3D_Diffusion(cond_channels=cond_channels)

    def forward(self, x_t, coronal_2d, sagittal_2d, t, slice_pos=None):
        B, _, H, W, D = x_t.shape
        mid_h = H // 2; mid_w = W // 2
        if slice_pos is None:
            slice_pos = torch.full((B,), 0.5, device=x_t.device)
        cond_3d = self.slice_to_vol(
            coronal_2d, sagittal_2d, D, slice_pos, mid_h, mid_w)
        return self.unet(x_t, cond_3d, t)


# ── Loss ──────────────────────────────────────────────────────────────────────
def gradient_loss(dvf):
    dh = (dvf[:,:,1:,:,:] - dvf[:,:,:-1,:,:]).pow(2).mean()
    dw = (dvf[:,:,:,1:,:] - dvf[:,:,:,:-1,:]).pow(2).mean()
    dd = (dvf[:,:,:,:,1:] - dvf[:,:,:,:,:-1]).pow(2).mean()
    return (dh + dw + dd) / 3.0


def train_one_epoch(model, diffusion, loader, optimizer, device):
    model.train()
    total_loss = total_recon = total_smooth = 0.0

    for dvf in loader:
        dvf = dvf.to(device)
        B   = dvf.shape[0]

        t         = torch.randint(0, diffusion.timesteps, (B,), device=device).long()
        noise     = torch.randn_like(dvf)
        dvf_noisy = diffusion.q_sample(dvf, t, noise)

        mid_h    = dvf.shape[2] // 2
        mid_w    = dvf.shape[3] // 2
        coronal  = dvf[:, :, :, mid_w, :]   # fix W -> coronal plane
        sagittal = dvf[:, :, mid_h, :, :]   # fix H -> sagittal plane
        slice_pos = torch.full((B,), mid_w / (dvf.shape[3] - 1), device=device)

        pred_noise = model(dvf_noisy, coronal, sagittal, t, slice_pos=slice_pos)
        recon_loss = F.mse_loss(pred_noise, noise)

        a       = diffusion.sqrt_alphas_bar[t].view(B,1,1,1,1)
        am      = diffusion.sqrt_one_minus_alphas_bar[t].view(B,1,1,1,1)
        pred_x0 = (dvf_noisy - am * pred_noise) / a
        smooth  = gradient_loss(pred_x0)
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
    print(f"Real Fine-tuning (cross-attention) — "
          f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"LR={LR}  WD={WEIGHT_DECAY}  epochs={EPOCHS}  smooth_lambda={SMOOTH_LAMBDA}")
    print(f"SLICE_FEAT_CHANNELS={SLICE_FEAT_CHANNELS}  "
          f"cond_channels={SLICE_FEAT_CHANNELS+2}")
    print(f"Checkpoint criterion: rolling average over last {ROLLING_N} epochs")
    print(f"DVF loading: no component reordering  |  No motion mask")
    print(f"Train: {[f'copd{i+1:02d}' for i in TRAIN_CASES]}")
    print(f"Test:  {[f'copd{i+1:02d}' for i in TEST_CASES]}")
    print("=" * 60)
    print(f"Device: {device}")

    stats    = np.load(REAL_STATS, allow_pickle=True).item()
    dvf_mean = float(stats["mean"])
    dvf_std  = float(stats["std"])
    print(f"Real DVF stats: mean={dvf_mean:.4f} mm, std={dvf_std:.4f} mm")

    print("\n=== Loading training cases ===")
    train_ds     = COPDDataset(TRAIN_CASES, dvf_mean, dvf_std)
    sampler      = RandomSampler(train_ds, replacement=True, num_samples=SAMPLES_PER_EPOCH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    print(f"Train batches per epoch: {len(train_loader)}")

    diffusion = DiffusionSchedule(timesteps=1000, device=device)
    model     = DiffusionModelManager(slice_feat_channels=SLICE_FEAT_CHANNELS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-7)

    best_avg           = float("inf")
    best_epoch         = 0
    start_epoch        = 1
    train_loss_history = []

    resume_path = os.path.join(SAVE_DIR, "latest_checkpoint.pth")
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt.get('scheduler_state_dict', scheduler.state_dict()))
        best_avg           = ckpt.get('best_avg', float("inf"))
        best_epoch         = ckpt.get('best_epoch', 0)
        train_loss_history = ckpt.get('train_loss_history', [])
        start_epoch        = ckpt['epoch'] + 1
        print(f"Resumed from epoch {ckpt['epoch']} | Best avg: {best_avg:.5f}")
    else:
        pretrain_path = os.path.join(PRETRAIN_DIR, "best_model.pth")
        pretrain_ckpt = torch.load(pretrain_path, map_location=device)
        if isinstance(pretrain_ckpt, dict) and 'model_state_dict' in pretrain_ckpt:
            model.load_state_dict(pretrain_ckpt['model_state_dict'])
        else:
            model.load_state_dict(pretrain_ckpt)
        print(f"Loaded pretrained: {pretrain_path}")

    print(f"\n{'Epoch':>6}  {'Train':>10}  {'MSE':>10}  {'Smooth':>10}  "
          f"{'RollAvg':>10}  {'Time':>8}")
    print("-" * 65)

    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()
        train_loss, recon_loss, smooth_loss = train_one_epoch(
            model, diffusion, train_loader, optimizer, device)
        elapsed = time.time() - t0
        scheduler.step()
        train_loss_history.append(train_loss)

        # ── rolling average checkpoint criterion ──────────────────────────────
        n_hist      = len(train_loss_history)
        window      = train_loss_history[-ROLLING_N:] if n_hist >= ROLLING_N \
                      else train_loss_history
        rolling_avg = sum(window) / len(window)

        is_best = rolling_avg < best_avg
        if is_best:
            best_avg   = rolling_avg
            best_epoch = epoch
            torch.save({'epoch':            epoch,
                        'model_state_dict': model.state_dict(),
                        'train_loss':       train_loss,
                        'rolling_avg':      rolling_avg,
                        'dvf_mean':         dvf_mean,
                        'dvf_std':          dvf_std},
                       os.path.join(SAVE_DIR, "best_model.pth"))

        tag = " *** BEST ***" if is_best else ""
        print(f"{epoch:>6}  {train_loss:>10.5f}  {recon_loss:>10.5f}  "
              f"{smooth_loss:>10.5f}  {rolling_avg:>10.5f}  "
              f"{elapsed:>6.1f}s{tag}", flush=True)

        if epoch % 5 == 0:
            torch.save({'epoch':                epoch,
                        'model_state_dict':     model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_avg':             best_avg,
                        'best_epoch':           best_epoch,
                        'train_loss_history':   train_loss_history,
                        'dvf_mean':             dvf_mean,
                        'dvf_std':              dvf_std},
                       os.path.join(SAVE_DIR, "latest_checkpoint.pth"))

        if epoch % 25 == 0:
            torch.save({'epoch':            epoch,
                        'model_state_dict': model.state_dict(),
                        'train_loss':       train_loss,
                        'rolling_avg':      rolling_avg,
                        'dvf_mean':         dvf_mean,
                        'dvf_std':          dvf_std},
                       os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch:04d}.pth"))

        if epoch % 10 == 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(range(1, len(train_loss_history)+1), train_loss_history,
                    label='Per-epoch loss', alpha=0.6)
            roll = [sum(train_loss_history[max(0,i-ROLLING_N):i]) /
                    len(train_loss_history[max(0,i-ROLLING_N):i])
                    for i in range(1, len(train_loss_history)+1)]
            ax.plot(range(1, len(roll)+1), roll,
                    label=f'Rolling avg (N={ROLLING_N})', linestyle='--')
            ax.axvline(best_epoch, color='r', linestyle=':',
                       label=f"Best epoch {best_epoch}")
            ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
            ax.set_title(f"Real Fine-tuning (cross-attn) — epoch {epoch}")
            ax.legend(); ax.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(VIS_DIR, "loss_curves_real.png"),
                        dpi=120, bbox_inches='tight')
            plt.close()

    print(f"\nDone. Best rolling avg: {best_avg:.5f} at epoch {best_epoch}")
    sys.stdout = tee.terminal
    tee.close()


if __name__ == "__main__":
    main()
