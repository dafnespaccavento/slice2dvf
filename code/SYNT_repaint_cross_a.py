"""
RePaint sampling with cross-attention + distance channels SliceToVolume.
- Full test set metrics (near/far split, cosine similarity, TRE on grid)
- Plot for sample 03 only: offset slices (PLANE_OFFSET=10), row labels
- All other metrics consistent with the architecture comparison table
"""

import os
import sys
import math
import datetime
import time
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
DATA_DIR    = "/mimer/NOBACKUP/groups/caim1/dafne/datasets/smooth_synthetic_downsampled_v2"
CKPT_DIR    = "/mimer/NOBACKUP/groups/caim1/dafne/checkpoints/newdataset/16_condchannels/concat_slicetovolume/cross_a"
RESULTS_DIR = "/mimer/NOBACKUP/groups/caim1/dafne/results/newdataset/16_condchannels/concat_slicetovolume/cross_a_v2/RePaint_Sampling_ressteps3"
os.makedirs(RESULTS_DIR, exist_ok=True)

VOXEL_SPACING       = (1.25, 1.25, 5.0)
RESAMPLING_STEPS    = 3
TEMPERATURE         = 1
SLICE_FEAT_CHANNELS = 16
SAMPLE_IDX          = 3
PLANE_OFFSET        = 10
FAR_THRESHOLD       = 20


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


# ── Dataset (unchanged) ───────────────────────────────────────────────────────
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
        mid_w     = slices["indices"]["mid_w"]
        W         = dvf.shape[2]
        slice_pos = mid_w / (W - 1)
        dvf           = torch.from_numpy(dvf).float()
        cond_coronal  = torch.from_numpy(cond_coronal).float()
        cond_sagittal = torch.from_numpy(cond_sagittal).float()
        return cond_coronal, cond_sagittal, dvf, slice_pos


# ── Model (unchanged) ─────────────────────────────────────────────────────────
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
        emb = torch.exp(
            torch.arange(half_dim, device=t.device) * -(math.log(10000) / (half_dim - 1))
        )
        emb = t[:, None] * emb[None, :] * 2 * math.pi
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)


class ResidualBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels), nn.ReLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x):
        return F.relu(x + self.block(x))


ATTN_POOL_SIZE = 128


class CrossAttention2D(nn.Module):
    def __init__(self, channels, num_heads=4, pool_size=ATTN_POOL_SIZE):
        super().__init__()
        self.num_heads = num_heads
        self.pool_size = pool_size
        self.head_dim  = channels // num_heads
        assert channels % num_heads == 0
        self.q_cor = nn.Linear(channels, channels, bias=False)
        self.k_sag = nn.Linear(channels, channels, bias=False)
        self.v_sag = nn.Linear(channels, channels, bias=False)
        self.q_sag = nn.Linear(channels, channels, bias=False)
        self.k_cor = nn.Linear(channels, channels, bias=False)
        self.v_cor = nn.Linear(channels, channels, bias=False)
        self.out_cor  = nn.Linear(channels, channels, bias=False)
        self.out_sag  = nn.Linear(channels, channels, bias=False)
        self.norm_cor = nn.LayerNorm(channels)
        self.norm_sag = nn.LayerNorm(channels)

    def _pool_seq(self, feat):
        B, C, S, D = feat.shape
        seq   = feat.permute(0, 2, 3, 1).reshape(B, S * D, C)
        pooled = F.adaptive_avg_pool1d(seq.permute(0, 2, 1), self.pool_size)
        return pooled.permute(0, 2, 1)

    def _flash_attn(self, q, k, v, B, heads, head_dim):
        def reshape(x): return x.view(B, -1, heads, head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(reshape(q), reshape(k), reshape(v), dropout_p=0.0)
        return out.transpose(1, 2).reshape(B, -1, heads * head_dim)

    def forward(self, cor_feat, sag_feat):
        B, C, H, D  = cor_feat.shape
        _,  _, W, _ = sag_feat.shape
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
        return cor_seq.reshape(B,H,D,C).permute(0,3,1,2), \
               sag_seq.reshape(B,W,D,C).permute(0,3,1,2)


def make_distance_channels(B, H, W, D, mid_h, mid_w, device):
    h_idx    = torch.arange(H, device=device).float()
    w_idx    = torch.arange(W, device=device).float()
    dist_cor = (h_idx - mid_h).abs() / max(mid_h, H - 1 - mid_h)
    dist_sag = (w_idx - mid_w).abs() / max(mid_w, W - 1 - mid_w)
    dist_cor = dist_cor[:, None, None].expand(H, W, D)
    dist_sag = dist_sag[None, :, None].expand(H, W, D)
    return torch.stack([dist_cor, dist_sag], dim=0).unsqueeze(0).expand(B,-1,-1,-1,-1)


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
        pos_bias = self.pos_mlp(slice_pos.float().unsqueeze(1))[:,:,None,None]
        cor_feat = cor_feat * (1 + pos_bias)
        cor_vol  = cor_feat.unsqueeze(3).expand(-1,-1,-1,W,-1)
        sag_vol  = sag_feat.unsqueeze(2).expand(-1,-1,H,-1,-1)
        fused    = F.relu(self.fusion(torch.cat([cor_vol, sag_vol], dim=1)))
        fused    = self.refine(fused)
        dist     = make_distance_channels(B, H, W, D, mid_h, mid_w, coronal.device)
        return torch.cat([fused, dist], dim=1)


class UNet3D_Diffusion(nn.Module):
    def __init__(self, cond_channels):
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


class DiffusionModelManager(nn.Module):
    def __init__(self, slice_feat_channels=SLICE_FEAT_CHANNELS):
        super().__init__()
        self.slice_to_vol = SliceToVolume(out_channels=slice_feat_channels)
        self.unet         = UNet3D_Diffusion(cond_channels=slice_feat_channels + 2)

    def forward(self, x_t, coronal_2d, sagittal_2d, t, slice_pos):
        B, _, H, W, D = x_t.shape
        mid_h = H // 2; mid_w = W // 2
        cond_3d = self.slice_to_vol(coronal_2d, sagittal_2d, D, slice_pos, mid_h, mid_w)
        return self.unet(x_t, cond_3d, t)


# ── RePaint sampler (unchanged) ───────────────────────────────────────────────
@torch.no_grad()
def sample_dvf_repaint(model, diffusion, cond_coronal, cond_sagittal,
                       known_coronal, known_sagittal, slice_pos, shape,
                       resampling_steps=RESAMPLING_STEPS, temperature=TEMPERATURE):
    B, C, H, W, D = shape
    mid_h = H // 2; mid_w = W // 2

    mask = torch.zeros(shape, device=cond_coronal.device)
    mask[:, :, mid_h, :, :] = 1.0
    mask[:, :, :, mid_w, :] = 1.0

    x_known_0 = torch.zeros(shape, device=cond_coronal.device)
    x_known_0[:, :, mid_h, :, :] = known_coronal
    x_known_0[:, :, :, mid_w, :] = known_sagittal

    x = torch.randn(shape, device=cond_coronal.device) * temperature

    for t in tqdm(reversed(range(diffusion.timesteps)),
                  total=diffusion.timesteps, desc="  RePaint", leave=False):
        for u in range(resampling_steps):
            tt      = torch.full((B,), t, device=cond_coronal.device, dtype=torch.long)
            eps_hat = model(x, cond_coronal, cond_sagittal, tt, slice_pos)

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
                    x = torch.sqrt(1-beta_t)*x + torch.sqrt(beta_t)*torch.randn_like(x)
            else:
                x = mask * x_known_0 + (1 - mask) * mean
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

    # TRE on 4x4x4 uniform grid (64 landmarks)
    lh = torch.linspace(0, H-1, 4).long()
    lw = torch.linspace(0, W-1, 4).long()
    ld = torch.linspace(0, D-1, 4).long()
    gh, gw, gd = torch.meshgrid(lh, lw, ld, indexing='ij')
    gh = gh.flatten(); gw = gw.flatten(); gd = gd.flatten()
    pred_lm = pred_mm[:, gh, gw, gd]
    gt_lm   = gt_mm[:, gh, gw, gd]
    m['tre'] = torch.sqrt(((pred_lm - gt_lm)**2).sum(dim=0)).mean().item()

    return m


def save_summary(mean_m, std_m, n, out_path):
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"  Cross-attention SliceToVolume — RePaint (U={RESAMPLING_STEPS})  —  N={n}")
    lines.append(f"{'='*60}")
    lines.append(f"  --- Full volume ---")
    for name in ['dx','dy','dz']:
        k = f'mae_{name}_full'
        lines.append(f"  MAE {name} (full)   {mean_m[k]:>8.4f} ± {std_m[k]:.4f} mm")
    lines.append(f"  MDE (full)         {mean_m['mde_full']:>8.4f} ± {std_m['mde_full']:.4f} mm")
    lines.append(f"  Cosine sim (full)  {mean_m['cosine_sim_full']:>8.4f} ± {std_m['cosine_sim_full']:.4f}")
    lines.append(f"  TRE (grid)         {mean_m['tre']:>8.4f} ± {std_m['tre']:.4f} mm")
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


# ── Plot (sample 03, offset slices, row labels) ───────────────────────────────
def plot_and_save(gt_mm, pred_mm, mid_h, mid_w, mid_d, out_path,
                  plane_offset=PLANE_OFFSET,
                  method_name=f"Cross-attn (RePaint U={RESAMPLING_STEPS})"):
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
    log_path   = os.path.join(RESULTS_DIR, "sampling.log")
    tee        = Tee(log_path)
    sys.stdout = tee

    print("=" * 60)
    print(f"RePaint Sampling (cross-attention) — "
          f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Resampling steps: {RESAMPLING_STEPS}  |  Temperature: {TEMPERATURE}")
    print(f"Slice feat channels: {SLICE_FEAT_CHANNELS}  |  "
          f"Total cond channels: {SLICE_FEAT_CHANNELS + 2}")
    print(f"Near/far threshold: {FAR_THRESHOLD} vox  |  Plot offset: {PLANE_OFFSET} vox")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    stats    = np.load(os.path.join(DATA_DIR, "dvf_stats.npy"), allow_pickle=True).item()
    dvf_mean = float(stats["mean"]); dvf_std = float(stats["std"])
    print(f"DVF stats: mean={dvf_mean:.6f}, std={dvf_std:.6f}")

    test_dataset = SyntheticDVFDataset(DATA_DIR, split="test")
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False,
                              num_workers=4, pin_memory=True)
    print(f"Test samples: {len(test_dataset)}")

    diffusion = DiffusionSchedule(timesteps=1000, device=device)
    model     = DiffusionModelManager(slice_feat_channels=SLICE_FEAT_CHANNELS).to(device)
    ckpt_path = os.path.join(CKPT_DIR, "best_model.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Loaded: {ckpt_path}")

    all_metrics = []
    vis_gt_mm = vis_pred_mm = vis_mid_h = vis_mid_w = vis_mid_d = None

    t_start = time.time()
    for idx, (cond_coronal, cond_sagittal, dvf_gt, slice_pos) in enumerate(test_loader):
        dvf_gt        = dvf_gt.to(device)
        cond_coronal  = cond_coronal.to(device)
        cond_sagittal = cond_sagittal.to(device)
        slice_pos     = slice_pos.to(device)

        B, C, H, W, D = dvf_gt.shape
        mid_h = H//2; mid_w = W//2; mid_d = D//2

        known_coronal  = dvf_gt[:, :, mid_h, :, :]
        known_sagittal = dvf_gt[:, :, :, mid_w, :]

        dvf_pred_norm = sample_dvf_repaint(
            model, diffusion, cond_coronal, cond_sagittal,
            known_coronal, known_sagittal,
            slice_pos=slice_pos, shape=dvf_gt.shape,
            resampling_steps=RESAMPLING_STEPS, temperature=TEMPERATURE)

        dvf_pred_mm = dvf_pred_norm.squeeze(0).cpu() * dvf_std + dvf_mean
        dvf_gt_mm   = dvf_gt.squeeze(0).cpu()        * dvf_std + dvf_mean

        m = compute_all_metrics(dvf_pred_mm, dvf_gt_mm, mid_h, mid_w)
        all_metrics.append(m)

        elapsed = time.time() - t_start
        per_s   = elapsed / (idx + 1)
        eta     = per_s * (len(test_dataset) - idx - 1)
        print(f"  [{idx+1:03d}/{len(test_dataset)}]  "
              f"MDE={m['mde_full']:.3f}  "
              f"MDE_far={m['mde_far']:.3f}  "
              f"cos_far={m['cosine_sim_far']:.3f}  "
              f"TRE={m['tre']:.3f}  "
              f"ETA={str(datetime.timedelta(seconds=int(eta)))}",
              flush=True)

        if idx == SAMPLE_IDX:
            vis_gt_mm   = dvf_gt_mm.clone()
            vis_pred_mm = dvf_pred_mm.clone()
            vis_mid_h, vis_mid_w, vis_mid_d = mid_h, mid_w, mid_d

    # ── Aggregate ─────────────────────────────────────────────────────────────
    keys   = all_metrics[0].keys()
    mean_m = {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}
    std_m  = {k: float(np.std( [m[k] for m in all_metrics])) for k in keys}

    save_summary(mean_m, std_m, n=len(test_dataset),
                 out_path=os.path.join(RESULTS_DIR, "summary_metrics.txt"))

    # ── Plot sample 03 ────────────────────────────────────────────────────────
    print(f"\nPlotting sample {SAMPLE_IDX}  (id={test_dataset.ids[SAMPLE_IDX]})")
    plot_and_save(
        vis_gt_mm, vis_pred_mm, vis_mid_h, vis_mid_w, vis_mid_d,
        out_path=os.path.join(RESULTS_DIR, f"plot_sample{SAMPLE_IDX:03d}.png"),
        plane_offset=PLANE_OFFSET)

    print(f"\nDone. Results: {RESULTS_DIR}")
    print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout = tee.terminal
    tee.close()


if __name__ == "__main__":
    main()
