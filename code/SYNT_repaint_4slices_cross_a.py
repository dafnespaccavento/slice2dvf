"""
sample_repaint_4slices_cross_a.py
──────────────────────────────────
RePaint sampling on the synthetic test set using the 4-slice
cross-attention + distance-channel architecture.

Identical to sample_repaint.py except:
  - SliceToVolume uses CrossAttention2D + distance channels (aligned with
    train_4slices_cross_a.py)
  - DiffusionModelManager receives SLICE_FEAT_CHANNELS + 2 cond channels
  - CKPT_DIR / RESULTS_DIR updated to cross_a paths
"""

import os
import sys
import math
import random
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
CKPT_DIR    = "/mimer/NOBACKUP/groups/caim1/dafne/checkpoints/newdataset/16_condchannels/4slices/concat_slicetovolume/cross_a"
RESULTS_DIR = "/mimer/NOBACKUP/groups/caim1/dafne/results/newdataset/16_condchannels/4slices/concat_slicetovolume/cross_a/RePaint_Sampling_ressteps3"
os.makedirs(RESULTS_DIR, exist_ok=True)

VOXEL_SPACING       = (1.25, 1.25, 5.0)
RESAMPLING_STEPS    = 3
TEMPERATURE         = 1
SLICE_FEAT_CHANNELS = 16   # distance channels added on top → UNet gets +2


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

        cor_25 = torch.from_numpy(slices["coronal_25"]).float()
        cor_75 = torch.from_numpy(slices["coronal_75"]).float()
        sag_25 = torch.from_numpy(slices["sagittal_25"]).float()
        sag_75 = torch.from_numpy(slices["sagittal_75"]).float()

        indices    = slices["indices"]
        W          = dvf.shape[2]
        H          = dvf.shape[1]
        pos_cor_25 = torch.tensor(indices["w_25"] / (W - 1), dtype=torch.float32)
        pos_cor_75 = torch.tensor(indices["w_75"] / (W - 1), dtype=torch.float32)
        pos_sag_25 = torch.tensor(indices["h_25"] / (H - 1), dtype=torch.float32)
        pos_sag_75 = torch.tensor(indices["h_75"] / (H - 1), dtype=torch.float32)

        dvf = torch.from_numpy(dvf).float()
        return (cor_25, cor_75, sag_25, sag_75,
                pos_cor_25, pos_cor_75, pos_sag_25, pos_sag_75,
                dvf)


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


# ── Building blocks ───────────────────────────────────────────────────────────
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
        pooled = F.adaptive_avg_pool1d(seq.permute(0, 2, 1), self.pool_size)
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
        cor_feat = cor_seq.reshape(B,H,D,C).permute(0,3,1,2)
        sag_feat = sag_seq.reshape(B,W,D,C).permute(0,3,1,2)
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


# ── SliceToVolume (4-slice + cross-attention + distance channels) ─────────────
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
            ResidualBlock3D(out_channels),
            ResidualBlock3D(out_channels))

    def _encode_and_sum(self, encoder, slice_a, slice_b, pos_a, pos_b):
        feat_a = encoder(slice_a)
        feat_b = encoder(slice_b)
        bias_a = self.pos_mlp(pos_a.unsqueeze(1))[:, :, None, None]
        bias_b = self.pos_mlp(pos_b.unsqueeze(1))[:, :, None, None]
        return feat_a * (1 + bias_a) + feat_b * (1 + bias_b)

    def forward(self, cor_25, cor_75, sag_25, sag_75,
                pos_cor_25, pos_cor_75, pos_sag_25, pos_sag_75):
        B = cor_25.shape[0]
        cor_feat = self._encode_and_sum(
            self.coronal_encoder,  cor_25, cor_75, pos_cor_25, pos_cor_75)
        sag_feat = self._encode_and_sum(
            self.sagittal_encoder, sag_25, sag_75, pos_sag_25, pos_sag_75)

        H = cor_feat.shape[2]
        W = sag_feat.shape[2]
        D = cor_feat.shape[3]

        cor_feat, sag_feat = self.cross_attn(cor_feat, sag_feat)

        cor_vol = cor_feat.unsqueeze(3).expand(-1, -1, -1, W, -1)
        sag_vol = sag_feat.unsqueeze(2).expand(-1, -1, H, -1, -1)
        fused   = F.relu(self.fusion(torch.cat([cor_vol, sag_vol], dim=1)))
        fused   = self.refine(fused)

        mid_h = H // 2
        mid_w = W // 2
        dist  = make_distance_channels(B, H, W, D, mid_h, mid_w, cor_25.device)
        return torch.cat([fused, dist], dim=1)   # (B, C+2, H, W, D)


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
        self.slice_to_vol = SliceToVolume(out_channels=slice_feat_channels)
        self.unet         = UNet3D_Diffusion(cond_channels=slice_feat_channels + 2)

    def forward(self, x_t, cor_25, cor_75, sag_25, sag_75,
                pos_cor_25, pos_cor_75, pos_sag_25, pos_sag_75, t):
        cond_3d = self.slice_to_vol(
            cor_25, cor_75, sag_25, sag_75,
            pos_cor_25, pos_cor_75, pos_sag_25, pos_sag_75)
        return self.unet(x_t, cond_3d, t)


# ── RePaint sampler ───────────────────────────────────────────────────────────
@torch.no_grad()
def sample_dvf_repaint(model, diffusion,
                       cor_25, cor_75, sag_25, sag_75,
                       pos_cor_25, pos_cor_75, pos_sag_25, pos_sag_75,
                       known_cor_25, known_cor_75, known_sag_25, known_sag_75,
                       w_25, w_75, h_25, h_75,
                       shape,
                       resampling_steps=RESAMPLING_STEPS,
                       temperature=TEMPERATURE):
    B, C, H, W, D = shape
    device = cor_25.device

    mask = torch.zeros(shape, device=device)
    mask[:, :, :, w_25, :] = 1.0
    mask[:, :, :, w_75, :] = 1.0
    mask[:, :, h_25, :, :] = 1.0
    mask[:, :, h_75, :, :] = 1.0

    x_known_0 = torch.zeros(shape, device=device)
    x_known_0[:, :, :, w_25, :] = known_cor_25
    x_known_0[:, :, :, w_75, :] = known_cor_75
    x_known_0[:, :, h_25, :, :] = known_sag_25
    x_known_0[:, :, h_75, :, :] = known_sag_75

    x = torch.randn(shape, device=device) * temperature

    for t in tqdm(reversed(range(diffusion.timesteps)),
                  total=diffusion.timesteps, desc="  RePaint", leave=False):
        for u in range(resampling_steps):
            tt      = torch.full((B,), t, device=device, dtype=torch.long)
            eps_hat = model(x, cor_25, cor_75, sag_25, sag_75,
                            pos_cor_25, pos_cor_75, pos_sag_25, pos_sag_75, tt)

            beta_t    = diffusion.betas[t].view(1,1,1,1,1)
            alpha_t   = diffusion.alphas[t].view(1,1,1,1,1)
            abar_t    = diffusion.alpha_bar[t].view(1,1,1,1,1)
            abar_prev = (diffusion.alpha_bar[t-1].view(1,1,1,1,1)
                         if t > 0 else torch.ones_like(abar_t))

            x0_hat = (x - torch.sqrt(1 - abar_t) * eps_hat) / torch.sqrt(abar_t)
            x0_hat = x0_hat.clamp(-3, 3)
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
FAR_THRESHOLD = 20   # voxels from either known plane

def compute_all_metrics(preds, gts, far_threshold=FAR_THRESHOLD):
    """
    preds, gts : (N, 3, H, W, D) in mm
    Returns dict with full/near/far splits for MAE, MDE, cosine similarity.
    Near/far defined relative to the volume mid-planes (H//2, W//2),
    consistent with the 2-slice conditioning convention used at inference.
    """
    N, C, H, W, D = preds.shape
    mid_h = H // 2
    mid_w = W // 2

    m = {}

    # ── per-component MAE (full volume) ──────────────────────────────────────
    for ci, name in enumerate(['dx', 'dy', 'dz']):
        m[f'mae_{name}'] = (preds[:,ci] - gts[:,ci]).abs().mean().item()

    # ── MDE full ──────────────────────────────────────────────────────────────
    err_vol = torch.sqrt(((preds - gts) ** 2).sum(dim=1))   # (N, H, W, D)
    m['mde_full'] = err_vol.mean().item()

    # ── near / far masks ──────────────────────────────────────────────────────
    dist_h = (torch.arange(H).float() - mid_h).abs()   # (H,)
    dist_w = (torch.arange(W).float() - mid_w).abs()   # (W,)
    dist_h_vol = dist_h[:, None, None].expand(H, W, D)
    dist_w_vol = dist_w[None, :, None].expand(H, W, D)
    far_mask  = (dist_h_vol >= far_threshold) & (dist_w_vol >= far_threshold)
    near_mask = ~far_mask

    m['mde_near'] = err_vol[:, near_mask].mean().item()
    m['mde_far']  = err_vol[:, far_mask].mean().item()

    # ── cosine similarity ─────────────────────────────────────────────────────
    pf = preds.reshape(N, 3, -1)   # (N, 3, H*W*D)
    gf = gts.reshape(N, 3, -1)
    cos = (pf * gf).sum(dim=1) / (
        pf.norm(dim=1).clamp(1e-8) * gf.norm(dim=1).clamp(1e-8))  # (N, H*W*D)
    m['cosine_sim_full'] = cos.mean().item()

    cos_vol = cos.reshape(N, H, W, D)
    m['cosine_sim_near'] = cos_vol[:, near_mask].mean().item()
    m['cosine_sim_far']  = cos_vol[:, far_mask].mean().item()

    return m


def compute_non_jacobian_fraction(dvf, voxel_spacing=VOXEL_SPACING):
    sx, sy, sz = voxel_spacing
    u = dvf[:,0]/sx; v = dvf[:,1]/sy; w = dvf[:,2]/sz
    def grad(f, dim):
        g = torch.zeros_like(f)
        sl_f=[slice(None)]*f.ndim; sl_f[dim]=slice(2,None)
        sl_b=[slice(None)]*f.ndim; sl_b[dim]=slice(None,-2)
        sl_c=[slice(None)]*f.ndim; sl_c[dim]=slice(1,-1)
        g[tuple(sl_c)]=(f[tuple(sl_f)]-f[tuple(sl_b)])/2.0
        sl0=[slice(None)]*f.ndim;  sl0[dim]=0
        sl1=[slice(None)]*f.ndim;  sl1[dim]=1
        slm1=[slice(None)]*f.ndim; slm1[dim]=-1
        slm2=[slice(None)]*f.ndim; slm2[dim]=-2
        g[tuple(sl0)]=f[tuple(sl1)]-f[tuple(sl0)]
        g[tuple(slm1)]=f[tuple(slm1)]-f[tuple(slm2)]
        return g
    du_dh=grad(u,1)+1.0; du_dw=grad(u,2);       du_dd=grad(u,3)
    dv_dh=grad(v,1);      dv_dw=grad(v,2)+1.0;  dv_dd=grad(v,3)
    dw_dh=grad(w,1);      dw_dw=grad(w,2);       dw_dd=grad(w,3)+1.0
    det=(du_dh*(dv_dw*dw_dd-dv_dd*dw_dw)
        -du_dw*(dv_dh*dw_dd-dv_dd*dw_dh)
        +du_dd*(dv_dh*dw_dw-dv_dw*dw_dh))
    return (det<=0).float().mean(dim=(1,2,3)).mean().item()

def compute_tre(preds, gts, lh, lw, ld):
    pred_lm = preds[:,:,lh,lw,ld]
    gt_lm   = gts[:,:,lh,lw,ld]
    return torch.sqrt(((pred_lm-gt_lm)**2).sum(dim=1)).mean().item()


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_fig1(si, gt_np, pred_np, mid_d, results_dir):
    gt_sl   = gt_np[0,:,:,mid_d]
    pred_sl = pred_np[0,:,:,mid_d]
    err_sl  = np.abs(gt_sl-pred_sl)
    vmin, vmax = gt_sl.min(), gt_sl.max()
    fig, axes = plt.subplots(1,3,figsize=(13,4))
    axes[0].imshow(gt_sl,   cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Sample {si} — DVF dx (GT)")
    axes[1].imshow(pred_sl, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Sample {si} — DVF dx (RePaint T={TEMPERATURE})")
    im = axes[2].imshow(err_sl, cmap="hot")
    axes[2].set_title(f"Sample {si} — |Error| (mean={err_sl.mean():.3f} mm)")
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    fname = os.path.join(results_dir, f"sample_{si:03d}_fig1_overview.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  Saved {fname}")

def plot_fig2(si, all_gts, all_preds, mid_d, mid_h, mid_w, results_dir):
    component_names = ["dx (LR)", "dy (AP)", "dz (SI)"]
    col_titles = ["Ax GT","Ax Pred","Cor GT","Cor Pred","Sag GT","Sag Pred"]
    fig, axes = plt.subplots(3,6,figsize=(18,9))
    for col, title in enumerate(col_titles):
        axes[0,col].set_title(title, fontsize=10, fontweight='bold')
    for c, cname in enumerate(component_names):
        gt_c   = all_gts[si,c].numpy()
        pred_c = all_preds[si,c].numpy()
        slices = [
            (gt_c[:,:,mid_d],  pred_c[:,:,mid_d]),
            (gt_c[mid_h,:,:],  pred_c[mid_h,:,:]),
            (gt_c[:,mid_w,:],  pred_c[:,mid_w,:]),
        ]
        vmax = max(np.abs(gt_c).max(), np.abs(pred_c).max()) or 1.0
        for v,(gt_sl,pred_sl) in enumerate(slices):
            axes[c,v*2].imshow(gt_sl,   cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
            axes[c,v*2].axis('off')
            im = axes[c,v*2+1].imshow(pred_sl, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
            axes[c,v*2+1].axis('off')
        axes[c,0].set_ylabel(cname, fontsize=11)
        divider = make_axes_locatable(axes[c,-1])
        cax = divider.append_axes("right", size="3%", pad=0.05)
        fig.colorbar(im, cax=cax)
    plt.suptitle(f"Sample {si} — DVF Components (GT vs RePaint cross-attn, T={TEMPERATURE})\n"
                 f"(z={mid_d}, h={mid_h}, w={mid_w})", fontsize=13, fontweight='bold')
    plt.subplots_adjust(top=0.92, wspace=0.05, hspace=0.1)
    fname = os.path.join(results_dir, f"sample_{si:03d}_compact.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  Saved {fname}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log_path   = os.path.join(RESULTS_DIR, "sampling.log")
    tee        = Tee(log_path)
    sys.stdout = tee

    print("=" * 60)
    print(f"RePaint 4-slice cross-attn — "
          f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Resampling steps: {RESAMPLING_STEPS}  |  Temperature: {TEMPERATURE}")
    print(f"SLICE_FEAT_CHANNELS={SLICE_FEAT_CHANNELS}  "
          f"cond_channels={SLICE_FEAT_CHANNELS+2}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    stats    = np.load(os.path.join(DATA_DIR, "dvf_stats.npy"), allow_pickle=True).item()
    dvf_mean = float(stats["mean"])
    dvf_std  = float(stats["std"])
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

    all_preds_norm, all_gts_norm = [], []
    all_preds,      all_gts      = [], []

    t_loop_start = time.time()
    for idx, batch in enumerate(test_loader):
        (cor_25, cor_75, sag_25, sag_75,
         pos_cor_25, pos_cor_75, pos_sag_25, pos_sag_75,
         dvf_gt) = [b.to(device) for b in batch]

        B, _, H, W, D = dvf_gt.shape
        w_25 = int(round(pos_cor_25.item() * (W - 1)))
        w_75 = int(round(pos_cor_75.item() * (W - 1)))
        h_25 = int(round(pos_sag_25.item() * (H - 1)))
        h_75 = int(round(pos_sag_75.item() * (H - 1)))

        known_cor_25 = dvf_gt[:, :, :, w_25, :]
        known_cor_75 = dvf_gt[:, :, :, w_75, :]
        known_sag_25 = dvf_gt[:, :, h_25, :, :]
        known_sag_75 = dvf_gt[:, :, h_75, :, :]

        dvf_pred_norm = sample_dvf_repaint(
            model, diffusion,
            cor_25, cor_75, sag_25, sag_75,
            pos_cor_25, pos_cor_75, pos_sag_25, pos_sag_75,
            known_cor_25, known_cor_75, known_sag_25, known_sag_75,
            w_25, w_75, h_25, h_75,
            shape=dvf_gt.shape,
            resampling_steps=RESAMPLING_STEPS,
            temperature=TEMPERATURE)

        all_preds_norm.append(dvf_pred_norm.cpu())
        all_gts_norm.append(dvf_gt.cpu())
        all_preds.append((dvf_pred_norm * dvf_std + dvf_mean).cpu())
        all_gts.append((dvf_gt         * dvf_std + dvf_mean).cpu())

        elapsed = time.time() - t_loop_start
        per_s   = elapsed / (idx + 1)
        remain  = per_s * (len(test_dataset) - idx - 1)
        print(f"  Sample {idx+1:03d}/{len(test_dataset)} | "
              f"elapsed: {str(datetime.timedelta(seconds=int(elapsed)))} | "
              f"per sample: {per_s:.0f}s | "
              f"ETA: {str(datetime.timedelta(seconds=int(remain)))}", flush=True)

    all_preds_norm = torch.cat(all_preds_norm, dim=0)
    all_gts_norm   = torch.cat(all_gts_norm,   dim=0)
    all_preds      = torch.cat(all_preds,      dim=0)
    all_gts        = torch.cat(all_gts,        dim=0)
    print(f"\nSampling complete. Shape: {tuple(all_preds.shape)}")

    N, _, H, W, D = all_preds.shape
    m       = compute_all_metrics(all_preds, all_gts, far_threshold=FAR_THRESHOLD)
    nj_pred = compute_non_jacobian_fraction(all_preds_norm)
    nj_gt   = compute_non_jacobian_fraction(all_gts_norm)
    lh = torch.linspace(0,H-1,4).long()
    lw = torch.linspace(0,W-1,4).long()
    ld = torch.linspace(0,D-1,4).long()
    gh,gw,gd = torch.meshgrid(lh,lw,ld,indexing='ij')
    tre = compute_tre(all_preds, all_gts, gh.flatten(), gw.flatten(), gd.flatten())

    metrics_str = (
        f"{'='*50}\n"
        f"  Method: RePaint 4-slice cross-attn\n"
        f"  Resampling steps: {RESAMPLING_STEPS}  |  Temperature: {TEMPERATURE}\n"
        f"  Test samples: {N}  |  DVF: {H}x{W}x{D}\n"
        f"  Near/far threshold: {FAR_THRESHOLD} voxels from mid-plane\n"
        f"{'='*50}\n"
        f"  --- Full volume ---\n"
        f"  MAE dx (LR):          {m['mae_dx']:>8.4f} mm\n"
        f"  MAE dy (AP):          {m['mae_dy']:>8.4f} mm\n"
        f"  MAE dz (SI):          {m['mae_dz']:>8.4f} mm\n"
        f"  MDE (full):           {m['mde_full']:>8.4f} mm\n"
        f"  Cosine sim (full):    {m['cosine_sim_full']:>8.4f}\n"
        f"  --- Near known planes (< {FAR_THRESHOLD} vox) ---\n"
        f"  MDE (near):           {m['mde_near']:>8.4f} mm\n"
        f"  Cosine sim (near):    {m['cosine_sim_near']:>8.4f}\n"
        f"  --- Far from known planes (>= {FAR_THRESHOLD} vox) ---\n"
        f"  MDE (far):            {m['mde_far']:>8.4f} mm\n"
        f"  Cosine sim (far):     {m['cosine_sim_far']:>8.4f}\n"
        f"  --- Other ---\n"
        f"  TRE (grid landmarks): {tre:>8.4f} mm\n"
        f"  Non-Jacobian (pred):  {nj_pred*100:>7.3f} %\n"
        f"  Non-Jacobian (GT):    {nj_gt*100:>7.3f} %\n"
        f"{'='*50}\n"
    )
    print("\n" + metrics_str)
    with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:
        f.write(metrics_str)

    random.seed(42)
    vis_indices = random.sample(range(N), 3)
    mid_d = D // 2; mid_h = H // 2; mid_w = W // 2
    for si in vis_indices:
        gt_np   = all_gts[si].numpy()
        pred_np = all_preds[si].numpy()
        plot_fig1(si, gt_np, pred_np, mid_d, RESULTS_DIR)
        plot_fig2(si, all_gts, all_preds, mid_d, mid_h, mid_w, RESULTS_DIR)

    print(f"\nDone. Results: {RESULTS_DIR}")
    print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout = tee.terminal
    tee.close()


if __name__ == "__main__":
    main()