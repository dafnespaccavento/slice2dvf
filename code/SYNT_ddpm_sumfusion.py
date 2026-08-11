"""
Evaluates the sum-fusion architecture with standard DDPM sampling
on ALL samples in the synthetic test set (old dataset, no v2).
Saves mean±std metrics to txt and plot for sample 3 as png.
"""

import os, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm.auto import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR      = "/mimer/NOBACKUP/groups/caim1/dafne/datasets/smooth_synthetic_downsampled"
CKPT_PATH     = "/mimer/NOBACKUP/groups/caim1/dafne/checkpoints/16_condchannels/best_model.pth"
RESULTS_DIR   = "/mimer/NOBACKUP/groups/caim1/dafne/results/eval_sum_fusion_ddpm"
SAMPLE_IDX    = 3
PLANE_OFFSET  = 10
FAR_THRESHOLD = 20
VOXEL_SPACING = (1.25, 1.25, 5.0)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── Dataset (original convention — mid_d) ─────────────────────────────────────
class SyntheticDVFDataset(Dataset):
    def __init__(self, root_dir, split="test"):
        self.field_dir = os.path.join(root_dir, split, "b")
        self.slice_dir = os.path.join(root_dir, split, "a")
        self.ids = sorted([
            f.split("_")[1].split(".")[0]
            for f in os.listdir(self.field_dir)
            if f.startswith("field_")
        ])
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx):
        i = self.ids[idx]
        dvf    = np.load(os.path.join(self.field_dir, f"field_{i}.npy"))
        slices = np.load(os.path.join(self.slice_dir,  f"slice_{i}.npy"),
                         allow_pickle=True).item()
        cond_coronal  = slices["coronal"]
        cond_sagittal = slices["sagittal"]
        mid_d     = slices["indices"]["mid_d"]
        D         = dvf.shape[-1]
        slice_pos = mid_d / (D - 1)
        dvf           = torch.from_numpy(dvf).float()
        cond_coronal  = torch.from_numpy(cond_coronal).float()
        cond_sagittal = torch.from_numpy(cond_sagittal).float()
        return cond_coronal, cond_sagittal, dvf, slice_pos

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
        super().__init__(); self.dim = dim
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
    def forward(self, x): return F.relu(x + self.block(x))

class SliceToVolume(nn.Module):
    def __init__(self, out_channels=32):
        super().__init__()
        self.coronal_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, out_channels, 3, padding=1))
        self.sagittal_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, out_channels, 3, padding=1))
        self.fusion = nn.Conv3d(out_channels, out_channels, 3, padding=1)

    def forward(self, coronal, sagittal, D):
        W = coronal.shape[2]
        H = sagittal.shape[2]
        cor_feat = self.coronal_encoder(coronal)
        sag_feat = self.sagittal_encoder(sagittal)
        cor_vol = cor_feat.permute(0,1,3,2).unsqueeze(3).expand(-1,-1,-1,H,-1)
        sag_vol = sag_feat.permute(0,1,3,2).unsqueeze(4).expand(-1,-1,-1,-1,W)
        return F.relu(self.fusion(cor_vol + sag_vol))

class UNet3D_Diffusion(nn.Module):
    def __init__(self, cond_channels=32):
        super().__init__()
        self.time_mlp      = nn.Sequential(TimeEmbedding(64), nn.Linear(64,64), nn.ReLU())
        self.time_proj1    = nn.Linear(64, 64)
        self.time_proj2    = nn.Linear(64, 128)
        self.time_proj_mid = nn.Linear(64, 128)
        self.enc1 = nn.Conv3d(3+cond_channels, 64, 3, padding=1)
        self.enc2 = nn.Conv3d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.mid     = nn.Conv3d(128, 128, 3, padding=1)
        self.mid_res = ResidualBlock3D(128)
        self.dec1 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.out  = nn.Conv3d(64, 3, 3, padding=1)

    def forward(self, x, cond, t):
        x    = x.permute(0,1,4,2,3).contiguous()
        cond = cond.permute(0,1,2,3,4).contiguous()
        t_emb = self.time_mlp(t.float() / 1000.0)
        x     = torch.cat([x, cond], dim=1)
        s1 = self.time_proj1(t_emb)[:,:,None,None,None]
        x1 = F.relu(self.enc1(x) * (1 + s1))
        x2 = F.relu(self.enc2(self.pool(x1)) + self.time_proj2(t_emb)[:,:,None,None,None])
        xm = F.relu(self.mid(x2) + self.time_proj_mid(t_emb)[:,:,None,None,None])
        xm = self.mid_res(xm)
        x  = self.dec1(xm) + x1
        return self.out(x).permute(0,1,3,4,2).contiguous()

class DiffusionModelManager(nn.Module):
    def __init__(self, cond_channels=32):
        super().__init__()
        self.slice_to_vol = SliceToVolume(out_channels=cond_channels)
        self.unet         = UNet3D_Diffusion(cond_channels=cond_channels)
    def forward(self, x_t, coronal_2d, sagittal_2d, t, slice_pos=None):
        D       = x_t.shape[-1]
        cond_3d = self.slice_to_vol(coronal_2d, sagittal_2d, D)
        return self.unet(x_t, cond_3d, t)

# ── DDPM sampler ──────────────────────────────────────────────────────────────
@torch.no_grad()
def sample_dvf_ddpm(model, diffusion, cond_coronal, cond_sagittal, shape):
    B = shape[0]
    x = torch.randn(shape, device=cond_coronal.device)
    for t in tqdm(reversed(range(diffusion.timesteps)),
                  total=diffusion.timesteps, desc="  DDPM", leave=False):
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
            var = beta_t * (1 - abar_prev) / (1 - abar_t)
            x   = mean + torch.sqrt(var) * torch.randn_like(x)
        else:
            x = mean
    return x

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_all_metrics(pred_mm, gt_mm, mid_h, mid_w, far_threshold=FAR_THRESHOLD):
    C, H, W, D = pred_mm.shape
    m = {}
    for ci, name in enumerate(['dx','dy','dz']):
        m[f'mae_{name}_full'] = (pred_mm[ci]-gt_mm[ci]).abs().mean().item()
    m['mde_full'] = torch.sqrt(((pred_mm-gt_mm)**2).sum(dim=0)).mean().item()

    dist_h_vol = (torch.arange(H).float()-mid_h).abs()[:,None,None].expand(H,W,D)
    dist_w_vol = (torch.arange(W).float()-mid_w).abs()[None,:,None].expand(H,W,D)
    far_mask  = (dist_h_vol >= far_threshold) & (dist_w_vol >= far_threshold)
    near_mask = ~far_mask

    err_vol = torch.sqrt(((pred_mm-gt_mm)**2).sum(dim=0))
    m['mde_near'] = err_vol[near_mask].mean().item()
    m['mde_far']  = err_vol[far_mask].mean().item()

    for ci, name in enumerate(['dx','dy','dz']):
        ae = (pred_mm[ci]-gt_mm[ci]).abs()
        m[f'mae_{name}_near'] = ae[near_mask].mean().item()
        m[f'mae_{name}_far']  = ae[far_mask].mean().item()
    m['far_voxel_fraction'] = far_mask.float().mean().item()

    pf = pred_mm.reshape(3,-1); gf = gt_mm.reshape(3,-1)
    cos = (pf*gf).sum(0) / (pf.norm(dim=0).clamp(1e-8) * gf.norm(dim=0).clamp(1e-8))
    m['cosine_sim_full'] = cos.mean().item()
    cos_vol = cos.reshape(H,W,D)
    m['cosine_sim_near'] = cos_vol[near_mask].mean().item()
    m['cosine_sim_far']  = cos_vol[far_mask].mean().item()
    return m

def save_summary(mean_m, std_m, n, out_path):
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"  Sum fusion (DDPM)  —  N={n} test samples")
    lines.append(f"{'='*60}")
    lines.append(f"  --- Full volume ---")
    for name in ['dx','dy','dz']:
        k = f'mae_{name}_full'
        lines.append(f"  MAE {name} (full)   {mean_m[k]:>8.4f} ± {std_m[k]:.4f} mm")
    lines.append(f"  MDE (full)         {mean_m['mde_full']:>8.4f} ± {std_m['mde_full']:.4f} mm")
    lines.append(f"  Cosine sim (full)  {mean_m['cosine_sim_full']:>8.4f} ± {std_m['cosine_sim_full']:.4f}")
    lines.append(f"\n  --- Near known planes (< {FAR_THRESHOLD} vox) ---")
    for name in ['dx','dy','dz']:
        k = f'mae_{name}_near'
        lines.append(f"  MAE {name} (near)   {mean_m[k]:>8.4f} ± {std_m[k]:.4f} mm")
    lines.append(f"  MDE (near)         {mean_m['mde_near']:>8.4f} ± {std_m['mde_near']:.4f} mm")
    lines.append(f"  Cosine sim (near)  {mean_m['cosine_sim_near']:>8.4f} ± {std_m['cosine_sim_near']:.4f}")
    lines.append(f"\n  --- Far from known planes (>= {FAR_THRESHOLD} vox) ---")
    for name in ['dx','dy','dz']:
        k = f'mae_{name}_far'
        lines.append(f"  MAE {name} (far)    {mean_m[k]:>8.4f} ± {std_m[k]:.4f} mm")
    lines.append(f"  MDE (far)          {mean_m['mde_far']:>8.4f} ± {std_m['mde_far']:.4f} mm")
    lines.append(f"  Cosine sim (far)   {mean_m['cosine_sim_far']:>8.4f} ± {std_m['cosine_sim_far']:.4f}")
    lines.append(f"{'='*60}\n")
    text = "\n".join(lines)
    print(text)
    with open(out_path, "w") as f:
        f.write(text)
    print(f"Summary saved to {out_path}")

# ── Visualization ─────────────────────────────────────────────────────────────
def plot_and_save(gt_mm, pred_mm, mid_h, mid_w, mid_d, out_path,
                  plane_offset=PLANE_OFFSET, method_name="Sum fusion (DDPM)"):
    C, H, W, D = gt_mm.shape
    cor_idx = min(mid_h + plane_offset, H - 1)
    sag_idx = min(mid_w + plane_offset, W - 1)
    component_names = ["dx (LR)", "dy (AP)", "dz (SI)"]
    col_titles = [
        f"Axial GT\n(z={mid_d})",      f"Axial Pred\n(z={mid_d})",
        f"Coronal GT\n(h={cor_idx})",  f"Coronal Pred\n(h={cor_idx})",
        f"Sagittal GT\n(w={sag_idx})", f"Sagittal Pred\n(w={sag_idx})",
    ]
    fig, axes = plt.subplots(3, 6, figsize=(20, 9))
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9, fontweight='bold')
    for c, cname in enumerate(component_names):
        gt_c   = gt_mm[c].numpy()
        pred_c = pred_mm[c].numpy()
        gt_slices   = [gt_c[:,:,mid_d],   gt_c[cor_idx,:,:], gt_c[:,sag_idx,:]]
        pred_slices = [pred_c[:,:,mid_d], pred_c[cor_idx,:,:], pred_c[:,sag_idx,:]]
        vmax = np.abs(gt_c).max()
        vmax = vmax if vmax > 0 else 1.0
        for v in range(3):
            axes[c, v*2].imshow(gt_slices[v],   cmap='RdBu_r',
                                vmin=-vmax, vmax=vmax, aspect='auto')
            axes[c, v*2].axis('off')
            im = axes[c, v*2+1].imshow(pred_slices[v], cmap='RdBu_r',
                                       vmin=-vmax, vmax=vmax, aspect='auto')
            axes[c, v*2+1].axis('off')
        axes[c, 0].set_ylabel(cname, fontsize=12, fontweight='bold', labelpad=8)
        divider = make_axes_locatable(axes[c, -1])
        cax = divider.append_axes("right", size="4%", pad=0.08)
        fig.colorbar(im, cax=cax, label="mm")
    plt.suptitle(
        f"Sample {SAMPLE_IDX} — GT vs {method_name}  "
        f"(coronal/sagittal at +{plane_offset} slices from conditioning plane)",
        fontsize=12, fontweight='bold')
    plt.subplots_adjust(top=0.90, wspace=0.04, hspace=0.08, left=0.06, right=0.92)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {out_path}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    stats    = np.load(os.path.join(DATA_DIR, "dvf_stats.npy"), allow_pickle=True).item()
    dvf_mean = float(stats["mean"]); dvf_std = float(stats["std"])
    print(f"DVF stats: mean={dvf_mean:.6f}, std={dvf_std:.6f}")

    dataset = SyntheticDVFDataset(DATA_DIR, split="test")
    print(f"Test samples: {len(dataset)}")

    diffusion = DiffusionSchedule(timesteps=1000, device=device)
    model     = DiffusionModelManager(cond_channels=16).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded: {CKPT_PATH}")

    all_metrics = []
    vis_gt_mm = vis_pred_mm = vis_mid_h = vis_mid_w = vis_mid_d = None

    for idx in range(len(dataset)):
        cond_coronal, cond_sagittal, dvf_gt_norm, slice_pos = dataset[idx]

        cond_coronal  = cond_coronal.unsqueeze(0).to(device)
        cond_sagittal = cond_sagittal.unsqueeze(0).to(device)
        dvf_gt_norm   = dvf_gt_norm.unsqueeze(0).to(device)

        B, C, H, W, D = dvf_gt_norm.shape
        mid_h = H//2; mid_w = W//2; mid_d = D//2

        dvf_pred_norm = sample_dvf_ddpm(
            model, diffusion, cond_coronal, cond_sagittal,
            shape=dvf_gt_norm.shape)

        dvf_pred_mm = dvf_pred_norm.squeeze(0).cpu() * dvf_std + dvf_mean
        dvf_gt_mm   = dvf_gt_norm.squeeze(0).cpu()   * dvf_std + dvf_mean

        m = compute_all_metrics(dvf_pred_mm, dvf_gt_mm, mid_h, mid_w)
        all_metrics.append(m)
        print(f"  [{idx+1:03d}/{len(dataset)}]  "
              f"MDE={m['mde_full']:.3f}  "
              f"MDE_far={m['mde_far']:.3f}  "
              f"cos_far={m['cosine_sim_far']:.3f}", flush=True)

        if idx == SAMPLE_IDX:
            vis_gt_mm   = dvf_gt_mm.clone()
            vis_pred_mm = dvf_pred_mm.clone()
            vis_mid_h, vis_mid_w, vis_mid_d = mid_h, mid_w, mid_d

    # ── Aggregate ─────────────────────────────────────────────────────────────
    keys   = all_metrics[0].keys()
    mean_m = {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}
    std_m  = {k: float(np.std( [m[k] for m in all_metrics])) for k in keys}

    save_summary(mean_m, std_m, n=len(dataset),
                 out_path=os.path.join(RESULTS_DIR, "summary_metrics.txt"))

    # ── Plot sample 3 ─────────────────────────────────────────────────────────
    print(f"\nPlotting sample {SAMPLE_IDX}  (id={dataset.ids[SAMPLE_IDX]})")
    plot_and_save(
        vis_gt_mm, vis_pred_mm, vis_mid_h, vis_mid_w, vis_mid_d,
        out_path=os.path.join(RESULTS_DIR, f"plot_sample{SAMPLE_IDX:03d}.png"),
        plane_offset=PLANE_OFFSET,
        method_name="Sum fusion (DDPM)")

    print("Done.")

if __name__ == "__main__":
    main()
