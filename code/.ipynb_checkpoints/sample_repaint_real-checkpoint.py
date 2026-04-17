"""
sample_repaint_real.py  —  RePaint sampling on real COPD test set (copd09, copd10)
Uses global normalisation stats from real training cases (compute_real_stats.py).
Same RePaint sampler as sample_repaint.py — no temperature scaling.
"""

import os
import sys
import math
import datetime
import time
import numpy as np
import nibabel as nib
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


# ── Paths ─────────────────────────────────────────────────────────────────────
FIELDS_DIR  = "/mimer/NOBACKUP/groups/caim1/dafne/datasets/real/fields"
CT_DIR      = "/mimer/NOBACKUP/groups/caim1/dafne/datasets/real/ct_images"
REAL_STATS  = "/mimer/NOBACKUP/groups/caim1/dafne/datasets/real/real_dvf_stats.npy"
CKPT_DIR    = "/mimer/NOBACKUP/groups/caim1/dafne/checkpoints_real/augmentation"
RESULTS_DIR = "/mimer/NOBACKUP/groups/caim1/dafne/results/RePaint_Real/augmentation/wo_border"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
TEST_CASES       = [8, 9]
VOXEL_SPACING    = (1.25, 1.25, 5.0)
RESAMPLING_STEPS = 5
MOTION_THRESHOLD = 0.5
ORIG_SHAPE       = (512, 512, 128)
DS_SHAPE         = (128, 128, 64)

REFERENCE_TRE = {
    8: {"initial": 14.860, "after_registration": 0.6412},
    9: {"initial": 21.806, "after_registration": 0.8498},
}


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
def load_and_preprocess_dvf(case_idx: int, dvf_mean: float, dvf_std: float):
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
    return (dvf - dvf_mean) / dvf_std


# ── Motion mask ──────────────────────────────────────────────────────────────
def get_motion_mask(dvf_norm: torch.Tensor, dvf_std: float) -> torch.Tensor:
    """
    Returns (H, W, D) boolean mask — True where motion exists.
    Threshold converted from mm to normalised space.
    """
    norm_threshold = MOTION_THRESHOLD / dvf_std
    mag = dvf_norm.norm(dim=0)   # (H, W, D)
    return mag > norm_threshold


# ── Landmark helpers ──────────────────────────────────────────────────────────
def load_landmarks(case_idx):
    case_name = f"copd{case_idx + 1}"
    case_dir  = os.path.join(CT_DIR, case_name)
    inhale_lm = np.loadtxt(os.path.join(case_dir, f"{case_name}_300_iBH_xyz_r1.txt"))
    exhale_lm = np.loadtxt(os.path.join(case_dir, f"{case_name}_300_eBH_xyz_r1.txt"))
    print(f"  Loaded {len(inhale_lm)} landmarks for {case_name}")
    return inhale_lm, exhale_lm


def scale_landmarks_to_ds(lm_orig):
    return lm_orig * (np.array(DS_SHAPE) / np.array(ORIG_SHAPE))


def compute_landmark_tre(dvf_pred_mm, inhale_lm_ds, exhale_lm_ds):
    dvf_np  = dvf_pred_mm.numpy()
    spacing = np.array(VOXEL_SPACING)
    tres = []
    for i in range(len(exhale_lm_ds)):
        ex = exhale_lm_ds[i]; ix = inhale_lm_ds[i]
        h = int(np.clip(round(ex[1]), 0, dvf_np.shape[1]-1))
        w = int(np.clip(round(ex[0]), 0, dvf_np.shape[2]-1))
        d = int(np.clip(round(ex[2]), 0, dvf_np.shape[3]-1))
        pred_mm = ex * spacing + dvf_np[:, h, w, d]
        ix_mm   = ix * spacing
        tres.append(float(np.sqrt(((pred_mm - ix_mm)**2).sum())))
    return float(np.mean(tres))


# ── Model ─────────────────────────────────────────────────────────────────────
class DiffusionSchedule:
    def __init__(self, timesteps=1000, device="cpu"):
        self.timesteps = timesteps
        self.betas     = torch.linspace(1e-4, 0.02, timesteps).to(device)
        self.alphas    = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_bar           = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1. - self.alpha_bar)


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


# ── RePaint sampler (same as sample_repaint.py, no temperature) ───────────────
@torch.no_grad()
def sample_dvf_repaint(model, diffusion, cond_coronal, cond_sagittal,
                       known_coronal, known_sagittal, shape,
                       resampling_steps=RESAMPLING_STEPS):
    B, C, H, W, D = shape
    mid_h = H // 2
    mid_w = W // 2

    mask = torch.zeros(shape, device=cond_coronal.device)
    mask[:, :, mid_h, :, :] = 1.0
    mask[:, :, :, mid_w, :] = 1.0

    x_known_0 = torch.zeros(shape, device=cond_coronal.device)
    x_known_0[:, :, mid_h, :, :] = known_coronal
    x_known_0[:, :, :, mid_w, :] = known_sagittal

    x = torch.randn(shape, device=cond_coronal.device)

    for t in tqdm(reversed(range(diffusion.timesteps)),
                  total=diffusion.timesteps, desc="  RePaint", leave=False):
        for u in range(resampling_steps):
            tt      = torch.full((B,), t, device=cond_coronal.device, dtype=torch.long)
            eps_hat = model(x, cond_coronal, cond_sagittal, tt)

            beta_t    = diffusion.betas[t].view(1,1,1,1,1)
            alpha_t   = diffusion.alphas[t].view(1,1,1,1,1)
            abar_t    = diffusion.alpha_bar[t].view(1,1,1,1,1)
            abar_prev = (diffusion.alpha_bar[t-1].view(1,1,1,1,1)
                         if t > 0 else torch.ones_like(abar_t))

            x0_hat = (x - torch.sqrt(1 - abar_t) * eps_hat) / torch.sqrt(abar_t)
            coef1  = torch.sqrt(abar_prev) * beta_t / (1 - abar_t)
            coef2  = torch.sqrt(alpha_t) * (1 - abar_prev) / (1 - abar_t)
            mean   = coef1 * x0_hat + coef2 * x

            if t > 0:
                var            = beta_t * (1 - abar_prev) / (1 - abar_t)
                x_gen          = mean + torch.sqrt(var) * torch.randn_like(x)
                noise_known    = torch.randn_like(x_known_0)
                x_known_t_prev = (torch.sqrt(abar_prev) * x_known_0
                                  + torch.sqrt(1 - abar_prev) * noise_known)
                x = mask * x_known_t_prev + (1 - mask) * x_gen
                if u < resampling_steps - 1:
                    x = torch.sqrt(1 - beta_t) * x + torch.sqrt(beta_t) * torch.randn_like(x)
            else:
                x = mask * x_known_0 + (1 - mask) * mean
    return x


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_fig1(case_name, gt_np, pred_np, mid_d, results_dir):
    gt_sl   = gt_np[0, :, :, mid_d]
    pred_sl = pred_np[0, :, :, mid_d]
    err_sl  = np.abs(gt_sl - pred_sl)
    vmin, vmax = gt_sl.min(), gt_sl.max()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(gt_sl,   cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{case_name} — GT dx")
    axes[1].imshow(pred_sl, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{case_name} — RePaint dx")
    im = axes[2].imshow(err_sl, cmap="hot")
    axes[2].set_title(f"{case_name} — |Error| (mean={err_sl.mean():.3f} mm)")
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    for ax in axes: ax.axis("off")
    plt.suptitle(f"RePaint Real — GT vs Predicted [{case_name}] (z={mid_d}, dx)", fontsize=12)
    plt.tight_layout()
    fname = os.path.join(results_dir, f"{case_name}_fig1_overview.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  Saved {fname}")


def plot_fig2(case_name, gt_t, pred_t, mid_d, mid_h, mid_w, results_dir):
    component_names = ["dx (LR)", "dy (AP)", "dz (SI)"]
    col_titles = ["Ax GT","Ax Pred","Cor GT","Cor Pred","Sag GT","Sag Pred"]
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    for col, title in enumerate(col_titles):
        axes[0,col].set_title(title, fontsize=10, fontweight='bold')
    for c, cname in enumerate(component_names):
        gt_c   = gt_t[c].numpy()
        pred_c = pred_t[c].numpy()
        slices = [(gt_c[:,:,mid_d], pred_c[:,:,mid_d]),
                  (gt_c[mid_h,:,:], pred_c[mid_h,:,:]),
                  (gt_c[:,mid_w,:], pred_c[:,mid_w,:])]
        vmax = max(np.abs(gt_c).max(), np.abs(pred_c).max())
        vmax = vmax if vmax > 0 else 1.0
        for v, (gs, ps) in enumerate(slices):
            axes[c,v*2].imshow(gs, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
            axes[c,v*2].axis('off')
            im_pr = axes[c,v*2+1].imshow(ps, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
            axes[c,v*2+1].axis('off')
        axes[c,0].set_ylabel(cname, fontsize=11)
        divider = make_axes_locatable(axes[c,-1])
        cax = divider.append_axes("right", size="3%", pad=0.05)
        fig.colorbar(im_pr, cax=cax)
    plt.suptitle(f"{case_name} — GT vs RePaint (z={mid_d}, h={mid_h}, w={mid_w})",
                 fontsize=13, fontweight='bold')
    plt.subplots_adjust(top=0.92, wspace=0.05, hspace=0.1)
    fname = os.path.join(results_dir, f"{case_name}_fig2_components.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  Saved {fname}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log_path   = os.path.join(RESULTS_DIR, "sampling.log")
    tee        = Tee(log_path)
    sys.stdout = tee

    print("=" * 60)
    print(f"RePaint Real Data — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Resampling steps: {RESAMPLING_STEPS}  |  No temperature scaling")
    print(f"Test cases: {[f'copd{i+1:02d}' for i in TEST_CASES]}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    stats    = np.load(REAL_STATS, allow_pickle=True).item()
    dvf_mean = float(stats["mean"])
    dvf_std  = float(stats["std"])
    print(f"Real DVF stats: mean={dvf_mean:.4f} mm, std={dvf_std:.4f} mm")

    diffusion = DiffusionSchedule(timesteps=1000, device=device)
    model     = DiffusionModelManager(cond_channels=16).to(device)
    ckpt_path = os.path.join(CKPT_DIR, "best_model.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint epoch {ckpt.get('epoch','?')}: {ckpt_path}")
    else:
        model.load_state_dict(ckpt)
        print(f"Loaded: {ckpt_path}")
    model.eval()

    all_metrics  = {}
    t_loop_start = time.time()

    for idx, case_idx in enumerate(TEST_CASES):
        case_name = f"copd{case_idx+1:02d}"
        print(f"\n{'─'*60}")
        print(f"Processing {case_name} ({idx+1}/{len(TEST_CASES)})...")

        dvf_gt_norm = load_and_preprocess_dvf(case_idx, dvf_mean, dvf_std)
        dvf_gt_norm = dvf_gt_norm.unsqueeze(0).to(device)
        B, C, H, W, D = dvf_gt_norm.shape
        mid_h = H//2; mid_w = W//2; mid_d = D//2

        cond_coronal  = dvf_gt_norm[:, :, mid_h, :, :]
        cond_sagittal = dvf_gt_norm[:, :, :, mid_w, :]

        # Motion mask — exclude zero border from RePaint anchor
        motion_mask = get_motion_mask(dvf_gt_norm.squeeze(0), dvf_std)  # (H, W, D)

        # Known slices: zero out border voxels so RePaint only anchors real motion
        known_coronal  = cond_coronal.clone()
        known_sagittal = cond_sagittal.clone()
        # Coronal slice at mid_h: mask along (W, D)
        cor_mask = motion_mask[mid_h, :, :]   # (W, D)
        known_coronal[:, :, ~cor_mask]  = 0.0
        # Sagittal slice at mid_w: mask along (H, D)
        sag_mask = motion_mask[:, mid_w, :]   # (H, D)
        known_sagittal[:, :, ~sag_mask] = 0.0

        print(f"  Motion mask: {motion_mask.sum().item():.0f} / {motion_mask.numel()} voxels "
              f"({100*motion_mask.float().mean().item():.1f}%)")
        print(f"  Known coronal non-zero: {cor_mask.sum().item()} / {cor_mask.numel()} voxels")
        print(f"  Known sagittal non-zero: {sag_mask.sum().item()} / {sag_mask.numel()} voxels")

        print(f"  Running RePaint (steps={RESAMPLING_STEPS})...")
        t0 = time.time()
        dvf_pred_norm = sample_dvf_repaint(
            model, diffusion, cond_coronal, cond_sagittal,
            known_coronal, known_sagittal, shape=dvf_gt_norm.shape,
            resampling_steps=RESAMPLING_STEPS)
        print(f"  Done in {time.time()-t0:.1f}s")

        dvf_pred_mm = dvf_pred_norm.squeeze(0).cpu() * dvf_std + dvf_mean
        dvf_gt_mm   = dvf_gt_norm.squeeze(0).cpu()   * dvf_std + dvf_mean

        # Evaluate only on motion voxels — ignore zero border
        mask_cpu = motion_mask.cpu()   # (H, W, D)
        pred_motion = dvf_pred_mm[:, mask_cpu]   # (3, N_motion)
        gt_motion   = dvf_gt_mm[:,   mask_cpu]   # (3, N_motion)

        mae_dx = (pred_motion[0] - gt_motion[0]).abs().mean().item()
        mae_dy = (pred_motion[1] - gt_motion[1]).abs().mean().item()
        mae_dz = (pred_motion[2] - gt_motion[2]).abs().mean().item()
        mde    = torch.sqrt(((pred_motion - gt_motion)**2).sum(dim=0)).mean().item()

        try:
            inhale_lm, exhale_lm = load_landmarks(case_idx)
            inhale_ds = scale_landmarks_to_ds(inhale_lm)
            exhale_ds = scale_landmarks_to_ds(exhale_lm)
            tre = compute_landmark_tre(dvf_pred_mm, inhale_ds, exhale_ds)
            ref = REFERENCE_TRE[case_idx]
            tre_str = (f"  TRE (landmarks):            {tre:>8.4f} mm\n"
                       f"  TRE initial (no reg):       {ref['initial']:>8.4f} mm\n"
                       f"  TRE after GT registration:  {ref['after_registration']:>8.4f} mm\n")
        except FileNotFoundError:
            tre = None
            tre_str = "  TRE: landmark files not found\n"

        metrics_str = (
            f"{'='*45}\n"
            f"  Case: {case_name}\n"
            f"  Method: RePaint (steps={RESAMPLING_STEPS}, no temperature)\n"
        f"  Evaluation: motion voxels only (border excluded)\n"
            f"  Normalisation: mean={dvf_mean:.3f} mm, std={dvf_std:.3f} mm (global real)\n"
            f"  DVF resolution: {H}x{W}x{D}  |  Voxel spacing: {VOXEL_SPACING} mm\n"
            f"{'='*45}\n"
            f"  MAE  dx (LR): {mae_dx:>8.4f} mm\n"
            f"  MAE  dy (AP): {mae_dy:>8.4f} mm\n"
            f"  MAE  dz (SI): {mae_dz:>8.4f} mm\n"
            f"{'-'*45}\n"
            f"  Mean 3D Displacement Error: {mde:>8.4f} mm\n"
            f"{'-'*45}\n"
            f"{tre_str}"
            f"{'='*45}\n"
        )
        print("\n" + metrics_str)
        all_metrics[case_name] = {'mae_dx':mae_dx,'mae_dy':mae_dy,'mae_dz':mae_dz,
                                   'mde':mde,'tre':tre}

        with open(os.path.join(RESULTS_DIR, f"{case_name}_metrics.txt"), "w") as f:
            f.write(metrics_str)

        plot_fig1(case_name, dvf_gt_mm.numpy(), dvf_pred_mm.numpy(), mid_d, RESULTS_DIR)
        plot_fig2(case_name, dvf_gt_mm, dvf_pred_mm, mid_d, mid_h, mid_w, RESULTS_DIR)

    # Summary
    tres = [all_metrics[k]['tre'] for k in all_metrics if all_metrics[k]['tre'] is not None]
    summary = (
        f"  MAE dx: {np.mean([all_metrics[k]['mae_dx'] for k in all_metrics]):.4f} mm\n"
        f"  MAE dy: {np.mean([all_metrics[k]['mae_dy'] for k in all_metrics]):.4f} mm\n"
        f"  MAE dz: {np.mean([all_metrics[k]['mae_dz'] for k in all_metrics]):.4f} mm\n"
        f"  MDE:    {np.mean([all_metrics[k]['mde']    for k in all_metrics]):.4f} mm\n"
        f"  TRE:    {f'{np.mean(tres):.4f} mm' if tres else 'N/A'}\n"
    )
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}\n{summary}")
    with open(os.path.join(RESULTS_DIR, "summary_metrics.txt"), "w") as f:
        f.write("SUMMARY — RePaint Real Data\n\n" + summary)

    print(f"\nDone. Results: {RESULTS_DIR}")
    print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout = tee.terminal
    tee.close()


if __name__ == "__main__":
    main()