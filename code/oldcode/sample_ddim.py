"""
sample_ddim.py  —  DDIM sampling + evaluation on the test set
──────────────────────────────────────────────────────────────
Loads best_model.pth, runs DDIM deterministic reverse diffusion on the full
test set, computes metrics in original (denormalised) DVF units, saves plots
and results to results/DDIM_Sampling/.

DDIM uses a subsequence of timesteps (num_ddim_steps << 1000) for fast
deterministic sampling. eta=0 → fully deterministic (no noise added).

Usage:
    python sample_ddim.py
"""

import os
import sys
import math
import random
import datetime
import numpy as np
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = "/mimer/NOBACKUP/groups/caim1/dafne/datasets/smooth_synthetic_downsampled"
CKPT_DIR    = "/mimer/NOBACKUP/groups/caim1/dafne/checkpoints/16_condchannels"
RESULTS_DIR = "/mimer/NOBACKUP/groups/caim1/dafne/results/16_condchannels/DDIM_Sampling"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Voxel spacing after downsampling from (256,256,128)@(0.625,0.625,2.5)mm
# to (128,128,64) — each dimension halved → spacing doubles
VOXEL_SPACING   = (1.25, 1.25, 5.0)   # mm  (x=LR, y=AP, z=SI)
NUM_DDIM_STEPS  = 100                  # number of denoising steps (<< 1000)
ETA             = 0.0                  # 0 = fully deterministic DDIM


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


# ── Dataset ───────────────────────────────────────────────────────────────────
class SyntheticDVFDataset(Dataset):
    def __init__(self, root_dir, split="test"):
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
        mid_d         = slices["indices"]["mid_d"]
        dvf           = torch.from_numpy(dvf).float()
        cond_coronal  = torch.from_numpy(cond_coronal).float()
        cond_sagittal = torch.from_numpy(cond_sagittal).float()
        D         = dvf.shape[-1]
        slice_pos = mid_d / (D - 1)
        return cond_coronal, cond_sagittal, dvf, slice_pos


# ── Model (verbatim from train.py) ────────────────────────────────────────────
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


class SliceToVolume(nn.Module):
    def __init__(self, out_channels=32):          # ← 16 → 32
        super().__init__()
        self.coronal_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, 3, padding=1)
        )
        self.sagittal_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, 3, padding=1)
        )
        self.fusion = nn.Conv3d(out_channels, out_channels, 3, padding=1)

    def forward(self, coronal, sagittal, D):
        B = coronal.shape[0]
        W = coronal.shape[2]
        H = sagittal.shape[2]

        cor_feat = self.coronal_encoder(coronal)
        sag_feat = self.sagittal_encoder(sagittal)

        cor_vol = cor_feat.permute(0, 1, 3, 2).unsqueeze(3).expand(-1, -1, -1, H, -1)
        sag_vol = sag_feat.permute(0, 1, 3, 2).unsqueeze(4).expand(-1, -1, -1, -1, W)

        return F.relu(self.fusion(cor_vol + sag_vol))


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


class UNet3D_Diffusion(nn.Module):
    def __init__(self, cond_channels=32):         # ← 16 → 32
        super().__init__()
        self.time_mlp = nn.Sequential(
            TimeEmbedding(64),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.time_proj1    = nn.Linear(64, 64)    # ← 32 → 64
        self.time_proj2    = nn.Linear(64, 128)   # ← 64 → 128
        self.time_proj_mid = nn.Linear(64, 128)   # ← 64 → 128

        self.enc1 = nn.Conv3d(3 + cond_channels, 64,  3, padding=1)   # ← 32 → 64
        self.enc2 = nn.Conv3d(64, 128, 3, padding=1)                   # ← 64 → 128
        self.pool = nn.MaxPool3d(2)

        self.mid     = nn.Conv3d(128, 128, 3, padding=1)               # ← 64 → 128
        self.mid_res = ResidualBlock3D(128)                             # ← 64 → 128

        self.dec1 = nn.ConvTranspose3d(128, 64, 2, stride=2)           # ← 64→32 becomes 128→64
        self.out  = nn.Conv3d(64, 3, 3, padding=1)                     # ← 32 → 64

    def forward(self, x, cond, t):
        x    = x.permute(0, 1, 4, 2, 3).contiguous()
        cond = cond.permute(0, 1, 2, 3, 4).contiguous()
        t    = t.float() / 1000.0
        t_emb = self.time_mlp(t)
        x     = torch.cat([x, cond], dim=1)

        scale1 = self.time_proj1(t_emb)[:, :, None, None, None]
        x1 = F.relu(self.enc1(x) * (1 + scale1))
        x2 = F.relu(self.enc2(self.pool(x1)) + self.time_proj2(t_emb)[:, :, None, None, None])

        x_mid = F.relu(self.mid(x2) + self.time_proj_mid(t_emb)[:, :, None, None, None])
        x_mid = self.mid_res(x_mid)

        x = self.dec1(x_mid) + x1
        x = self.out(x)
        return x.permute(0, 1, 3, 4, 2).contiguous()


class DiffusionModelManager(nn.Module):
    def __init__(self, cond_channels=32):         # ← 16 → 32
        super().__init__()
        self.slice_to_vol = SliceToVolume(out_channels=cond_channels)
        self.unet         = UNet3D_Diffusion(cond_channels=cond_channels)

    def forward(self, x_t, coronal_2d, sagittal_2d, t):
        D       = x_t.shape[-1]
        cond_3d = self.slice_to_vol(coronal_2d, sagittal_2d, D)
        return self.unet(x_t, cond_3d, t)


# ── DDIM Sampling ─────────────────────────────────────────────────────────────
@torch.no_grad()
def sample_ddim(model, diffusion, cond_coronal, cond_sagittal, shape,
                num_steps=NUM_DDIM_STEPS, eta=ETA):
    """
    DDIM reverse diffusion.
    num_steps: number of denoising steps (can be much less than 1000)
    eta=0: fully deterministic; eta=1: equivalent to DDPM
    """
    # Build uniform subsequence of timesteps T, T-step, ..., 0
    total_T   = diffusion.timesteps
    step_size = total_T // num_steps
    timesteps = list(reversed(range(0, total_T, step_size)))  # [999, 989, ..., 9] approx

    x = torch.randn(shape, device=cond_coronal.device)

    for i, t in enumerate(tqdm(timesteps, desc="  DDIM denoising", leave=False)):
        tt      = torch.full((shape[0],), t, device=x.device, dtype=torch.long)
        eps_hat = model(x, cond_coronal, cond_sagittal, tt)

        abar_t    = diffusion.alpha_bar[t].view(1, 1, 1, 1, 1)
        abar_prev = (diffusion.alpha_bar[timesteps[i+1]].view(1, 1, 1, 1, 1)
                     if i + 1 < len(timesteps)
                     else torch.ones_like(abar_t))

        # Predicted x0
        x0_hat = (x - torch.sqrt(1 - abar_t) * eps_hat) / torch.sqrt(abar_t)
        x0_hat = x0_hat.clamp(-3, 3)   # stabilise

        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - abar_prev - eta**2 * (1 - abar_prev) / (1 - abar_t) * (1 - abar_t / abar_prev)) * eps_hat

        # Stochastic term (0 when eta=0)
        if eta > 0 and i + 1 < len(timesteps):
            sigma = eta * torch.sqrt((1 - abar_prev) / (1 - abar_t) * (1 - abar_t / abar_prev))
            noise = sigma * torch.randn_like(x)
        else:
            noise = 0.0

        x = torch.sqrt(abar_prev) * x0_hat + dir_xt + noise

    return x


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_mae_per_component(preds, gts):
    return {
        'mae_dx': (preds[:, 0] - gts[:, 0]).abs().mean().item(),
        'mae_dy': (preds[:, 1] - gts[:, 1]).abs().mean().item(),
        'mae_dz': (preds[:, 2] - gts[:, 2]).abs().mean().item(),
    }


def compute_mde(preds, gts):
    diff = preds - gts
    dist = torch.sqrt((diff ** 2).sum(dim=1))
    return dist.mean().item()


def compute_non_jacobian_fraction(dvf, voxel_spacing=VOXEL_SPACING):
    sx, sy, sz = voxel_spacing

    u = dvf[:, 0] / sx
    v = dvf[:, 1] / sy
    w = dvf[:, 2] / sz

    def grad(f, dim):
        g = torch.zeros_like(f)
        sl_f = [slice(None)] * f.ndim; sl_f[dim] = slice(2, None)
        sl_b = [slice(None)] * f.ndim; sl_b[dim] = slice(None, -2)
        sl_c = [slice(None)] * f.ndim; sl_c[dim] = slice(1, -1)
        g[tuple(sl_c)] = (f[tuple(sl_f)] - f[tuple(sl_b)]) / 2.0
        sl0  = [slice(None)] * f.ndim; sl0[dim]  = 0
        sl1  = [slice(None)] * f.ndim; sl1[dim]  = 1
        slm1 = [slice(None)] * f.ndim; slm1[dim] = -1
        slm2 = [slice(None)] * f.ndim; slm2[dim] = -2
        g[tuple(sl0)]  = f[tuple(sl1)]  - f[tuple(sl0)]
        g[tuple(slm1)] = f[tuple(slm1)] - f[tuple(slm2)]
        return g

    du_dh = grad(u, 1) + 1.0; du_dw = grad(u, 2);       du_dd = grad(u, 3)
    dv_dh = grad(v, 1);       dv_dw = grad(v, 2) + 1.0; dv_dd = grad(v, 3)
    dw_dh = grad(w, 1);       dw_dw = grad(w, 2);       dw_dd = grad(w, 3) + 1.0
    det = (  du_dh * (dv_dw * dw_dd - dv_dd * dw_dw)
           - du_dw * (dv_dh * dw_dd - dv_dd * dw_dh)
           + du_dd * (dv_dh * dw_dw - dv_dw * dw_dh))
    return (det <= 0).float().mean(dim=(1, 2, 3)).mean().item()


def compute_tre(preds, gts, landmarks_h, landmarks_w, landmarks_d):
    pred_lm = preds[:, :, landmarks_h, landmarks_w, landmarks_d]
    gt_lm   = gts[:,   :, landmarks_h, landmarks_w, landmarks_d]
    tre = torch.sqrt(((pred_lm - gt_lm) ** 2).sum(dim=1))
    return tre.mean().item()


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_fig1(si, gt_np, pred_np, mid_d, results_dir):
    gt_sl   = gt_np[0, :, :, mid_d]
    pred_sl = pred_np[0, :, :, mid_d]
    err_sl  = np.abs(gt_sl - pred_sl)
    vmin, vmax = gt_sl.min(), gt_sl.max()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(gt_sl,   cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Sample {si} — DVF dx (GT)")
    axes[1].imshow(pred_sl, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Sample {si} — DVF dx (DDIM)")
    im = axes[2].imshow(err_sl, cmap="hot")
    axes[2].set_title(f"Sample {si} — |Error|  (mean={err_sl.mean():.3f} mm)")
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.axis("off")
    plt.suptitle(
        f"DDIM Sampling — GT vs Reconstructed vs Error\n"
        f"(mid axial slice z={mid_d}, channel dx)",
        fontsize=12
    )
    plt.tight_layout()
    fname = os.path.join(results_dir, f"sample_{si:03d}_fig1_overview.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")


def plot_fig2(si, all_gts, all_preds, mid_d, mid_h, mid_w,
                     results_dir, method_name="DDIM"):
    component_names = ["dx (LR)", "dy (AP)", "dz (SI)"]
    col_titles = ["Ax GT", "Ax Pred", "Cor GT", "Cor Pred", "Sag GT", "Sag Pred"]
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    # column titles
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight='bold')
    for c, cname in enumerate(component_names):
        gt_c   = all_gts[si, c].numpy()
        pred_c = all_preds[si, c].numpy()
        slices = [
            (gt_c[:, :, mid_d],  pred_c[:, :, mid_d]),
            (gt_c[mid_h, :, :],  pred_c[mid_h, :, :]),
            (gt_c[:, mid_w, :],  pred_c[:, mid_w, :]),
        ]
        vmax = max(np.abs(gt_c).max(), np.abs(pred_c).max())
        vmax = vmax if vmax > 0 else 1.0
        for v, (gt_sl, pred_sl) in enumerate(slices):
            col_gt = v * 2
            col_pr = v * 2 + 1
            im_gt = axes[c, col_gt].imshow(
                gt_sl, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto'
            )
            axes[c, col_gt].axis('off')
            im_pr = axes[c, col_pr].imshow(
                pred_sl, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto'
            )
            axes[c, col_pr].axis('off')
        # row label
        axes[c, 0].set_ylabel(cname, fontsize=11)
        # colorbar
        divider = make_axes_locatable(axes[c, -1])
        cax = divider.append_axes("right", size="3%", pad=0.05)
        fig.colorbar(im_pr, cax=cax)
    plt.suptitle(
        f"Sample {si} — DVF Components (GT vs {method_name})\n"
        f"(z={mid_d}, h={mid_h}, w={mid_w})",
        fontsize=13,
        fontweight='bold'
    )
    plt.subplots_adjust(top=0.92, wspace=0.05, hspace=0.1)
    fname = os.path.join(results_dir, f"sample_{si:03d}_compact.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log_path   = os.path.join(RESULTS_DIR, "sampling.log")
    tee        = Tee(log_path)
    sys.stdout = tee

    print("=" * 60)
    print(f"DDIM Sampling — started {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log: {log_path}")
    print(f"DDIM steps: {NUM_DDIM_STEPS}  |  eta: {ETA}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Voxel spacing (downsampled): {VOXEL_SPACING} mm")

    # ── Load DVF stats ────────────────────────────────────────────────────────
    stats    = np.load(os.path.join(DATA_DIR, "dvf_stats.npy"), allow_pickle=True).item()
    dvf_mean = float(stats["mean"])
    dvf_std  = float(stats["std"])
    print(f"DVF stats: mean={dvf_mean:.6f}, std={dvf_std:.6f}")

    # ── Dataset & loader ──────────────────────────────────────────────────────
    test_dataset = SyntheticDVFDataset(DATA_DIR, split="test")
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False,
                              num_workers=4, pin_memory=True)
    print(f"Test samples: {len(test_dataset)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    diffusion = DiffusionSchedule(timesteps=1000, device=device)
    model     = DiffusionModelManager(cond_channels=16).to(device)
    ckpt_path = os.path.join(CKPT_DIR, "best_model.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Loaded: {ckpt_path}")

    # ── Run sampling ──────────────────────────────────────────────────────────
    all_preds_norm = []
    all_gts_norm   = []
    all_preds      = []
    all_gts        = []

    print(f"\nRunning DDIM sampling ({NUM_DDIM_STEPS} steps, eta={ETA}) on {len(test_dataset)} test samples...")
    for idx, (cond_coronal, cond_sagittal, dvf_gt, _) in enumerate(test_loader):
        print(f"  Sample {idx+1:03d}/{len(test_dataset)}", flush=True)
        cond_coronal  = cond_coronal.to(device)
        cond_sagittal = cond_sagittal.to(device)
        dvf_gt        = dvf_gt.to(device)

        B, C, H, W, D = dvf_gt.shape
        dvf_pred_norm = sample_ddim(
            model, diffusion, cond_coronal, cond_sagittal,
            shape=(B, C, H, W, D)
        )

        all_preds_norm.append(dvf_pred_norm.cpu())
        all_gts_norm.append(dvf_gt.cpu())
        all_preds.append((dvf_pred_norm * dvf_std + dvf_mean).cpu())
        all_gts.append((dvf_gt         * dvf_std + dvf_mean).cpu())

    all_preds_norm = torch.cat(all_preds_norm, dim=0)
    all_gts_norm   = torch.cat(all_gts_norm,   dim=0)
    all_preds      = torch.cat(all_preds,      dim=0)
    all_gts        = torch.cat(all_gts,        dim=0)
    print(f"\nSampling complete. Shape: {tuple(all_preds.shape)}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\nComputing metrics...")
    N, _, H, W, D = all_preds.shape
    mae     = compute_mae_per_component(all_preds, all_gts)
    mde     = compute_mde(all_preds, all_gts)
    nj_pred = compute_non_jacobian_fraction(all_preds_norm)
    nj_gt   = compute_non_jacobian_fraction(all_gts_norm)

    lh = torch.linspace(0, H-1, 4).long()
    lw = torch.linspace(0, W-1, 4).long()
    ld = torch.linspace(0, D-1, 4).long()
    grid_h, grid_w, grid_d = torch.meshgrid(lh, lw, ld, indexing='ij')
    tre = compute_tre(all_preds, all_gts,
                      grid_h.flatten(), grid_w.flatten(), grid_d.flatten())

    metrics_str = (
        f"{'='*45}\n"
        f"  Method: DDIM Sampling\n"
        f"  DDIM steps: {NUM_DDIM_STEPS}  |  eta: {ETA}\n"
        f"  Test samples: {N}\n"
        f"  DVF resolution: {H}x{W}x{D}\n"
        f"  Voxel spacing: {VOXEL_SPACING} mm\n"
        f"{'='*45}\n"
        f"  MAE  dx (LR): {mae['mae_dx']:>8.4f} mm\n"
        f"  MAE  dy (AP): {mae['mae_dy']:>8.4f} mm\n"
        f"  MAE  dz (SI): {mae['mae_dz']:>8.4f} mm\n"
        f"{'-'*45}\n"
        f"  Mean 3D Displacement Error: {mde:>8.4f} mm\n"
        f"{'-'*45}\n"
        f"  TRE (grid landmarks):       {tre:>8.4f} mm\n"
        f"{'-'*45}\n"
        f"  Non-Jacobian fraction (normalised space):\n"
        f"    Predicted:  {nj_pred*100:>7.3f} %\n"
        f"    GT:         {nj_gt*100:>7.3f} %\n"
        f"{'='*45}\n"
    )
    print("\n" + metrics_str)

    metrics_path = os.path.join(RESULTS_DIR, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(metrics_str)
    print(f"Metrics saved to: {metrics_path}")

    # ── Visualisation — 3 random samples ─────────────────────────────────────
    random.seed(42)
    vis_indices = random.sample(range(N), 3)
    print(f"\nVisualising samples: {vis_indices}")

    mid_d = D // 2
    mid_h = H // 2
    mid_w = W // 2

    for si in vis_indices:
        print(f"\n  Plotting sample {si}...")
        gt_np   = all_gts[si].numpy()
        pred_np = all_preds[si].numpy()
        plot_fig1(si, gt_np, pred_np, mid_d, RESULTS_DIR)
        plot_fig2(si, all_gts, all_preds, mid_d, mid_h, mid_w, RESULTS_DIR, method_name="DDIM")

    print(f"\n{'='*60}")
    print(f"Done. Results saved to: {RESULTS_DIR}")
    print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    sys.stdout = tee.terminal
    tee.close()


if __name__ == "__main__":
    main()