"""
 RePaint sampling on real COPD test set
Architecture: cross-attention + distance channels
Everything identical to sample_ddpm_real_cross_a except:
  - Sampler: RePaint (U=3) instead of DDPM
  - x0_hat clamped to [-3, 3] inside the RePaint loop
  - known_coronal / known_sagittal extracted and passed as hard constraints
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
CKPT_DIR    = "/mimer/NOBACKUP/groups/caim1/dafne/checkpoints_real/16_condchannels/cross_a_v2/swap/small"
RESULTS_DIR = "/mimer/NOBACKUP/groups/caim1/dafne/results/RePaint_Real/16_condchannels/cross_a_v2/restep3/small"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
TEST_CASES          = [8, 9]
DS_SHAPE            = (128, 128, 64)
SLICE_FEAT_CHANNELS = 16
PLANE_OFFSET        = 10
FAR_THRESHOLD       = 20
RESAMPLING_STEPS    = 3
TEMPERATURE         = 1

ORIG_SPACING = {
    8: np.array([0.664, 0.664, 2.5]),
    9: np.array([0.742, 0.742, 2.5]),
}
ORIG_Z_SLICES = {
    8: 116,
    9: 135,
}
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


# ── DVF loading (direct interpolation, no pad/crop) ───────────────────────────
def load_and_preprocess_dvf(case_idx, dvf_mean, dvf_std):
    path = os.path.join(FIELDS_DIR, f"copd{case_idx + 1:02d}.nii.gz")
    dvf  = nib.load(path).get_fdata().astype(np.float32)
    dvf  = dvf.transpose(3, 0, 1, 2)
    dvf  = torch.from_numpy(dvf)
    dvf  = F.interpolate(dvf.unsqueeze(0), size=DS_SHAPE,
                         mode='trilinear', align_corners=False).squeeze(0)
    return (dvf - dvf_mean) / dvf_std


# ── Landmark loading ──────────────────────────────────────────────────────────
def load_landmarks(case_idx):
    case_name = f"copd{case_idx + 1}"
    case_dir  = os.path.join(CT_DIR, case_name)
    pts_fix   = np.loadtxt(os.path.join(case_dir, f"{case_name}_300_iBH_xyz_r1.txt"))
    pts_mov   = np.loadtxt(os.path.join(case_dir, f"{case_name}_300_eBH_xyz_r1.txt"))
    print(f"  Loaded {len(pts_fix)} landmarks for {case_name}")
    return pts_fix, pts_mov


# ── TRE ───────────────────────────────────────────────────────────────────────
def compute_tre(pts_fix, pts_mov, disp_field, spc):
    n_lms = pts_fix.shape[0]
    D, H, W, _ = disp_field.shape
    disp_pix  = disp_field / spc[np.newaxis, np.newaxis, np.newaxis, :]
    moved_pts = np.zeros_like(pts_fix)
    for li in range(n_lms):
        px = pts_fix[li] - 1.0
        x0 = int(np.floor(px[0])); x1 = x0 + 1
        y0 = int(np.floor(px[1])); y1 = y0 + 1
        z0 = int(np.floor(px[2])); z1 = z0 + 1
        xd = px[0]-x0; yd = px[1]-y0; zd = px[2]-z0
        x0=max(0,min(D-1,x0)); x1=max(0,min(D-1,x1))
        y0=max(0,min(H-1,y0)); y1=max(0,min(H-1,y1))
        z0=max(0,min(W-1,z0)); z1=max(0,min(W-1,z1))
        for d in range(3):
            c000=disp_pix[x0,y0,z0,d]; c001=disp_pix[x0,y0,z1,d]
            c010=disp_pix[x0,y1,z0,d]; c011=disp_pix[x0,y1,z1,d]
            c100=disp_pix[x1,y0,z0,d]; c101=disp_pix[x1,y0,z1,d]
            c110=disp_pix[x1,y1,z0,d]; c111=disp_pix[x1,y1,z1,d]
            c00=c000*(1-xd)+c100*xd; c01=c001*(1-xd)+c101*xd
            c10=c010*(1-xd)+c110*xd; c11=c011*(1-xd)+c111*xd
            c0=c00*(1-yd)+c10*yd;    c1=c01*(1-yd)+c11*yd
            moved_pts[li,d] = pts_fix[li,d] + c0*(1-zd)+c1*zd
    moved_pts_r = np.round(moved_pts)
    errs = np.sqrt(np.sum(((moved_pts_r - pts_mov)*spc)**2, axis=1))
    return float(np.mean(errs)), float(np.std(errs))


def compute_landmark_tre(dvf_pred_mm, pts_fix, pts_mov, case_idx):
    spc    = ORIG_SPACING[case_idx]
    orig_z = ORIG_Z_SLICES[case_idx]
    dvf_up = F.interpolate(dvf_pred_mm.unsqueeze(0),
                           size=(512, 512, orig_z),
                           mode='trilinear', align_corners=False).squeeze(0)
    dvf_np = dvf_up.cpu().numpy().transpose(1, 2, 3, 0)
    return compute_tre(pts_fix, pts_mov, dvf_np, spc)


# ── Model (unchanged) ─────────────────────────────────────────────────────────
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


class ResidualBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels,channels,3,padding=1), nn.GroupNorm(8,channels), nn.ReLU(),
            nn.Conv3d(channels,channels,3,padding=1), nn.GroupNorm(8,channels))

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
        seq    = feat.permute(0,2,3,1).reshape(B, S*D, C)
        pooled = F.adaptive_avg_pool1d(seq.permute(0,2,1), self.pool_size)
        return pooled.permute(0,2,1)

    def _flash_attn(self, q, k, v, B, heads, head_dim):
        def reshape(x): return x.view(B,-1,heads,head_dim).transpose(1,2)
        out = F.scaled_dot_product_attention(reshape(q), reshape(k), reshape(v), dropout_p=0.0)
        return out.transpose(1,2).reshape(B,-1,heads*head_dim)

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
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(),
            nn.Conv2d(16,out_channels,3,padding=1))
        self.sagittal_encoder = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(),
            nn.Conv2d(16,out_channels,3,padding=1))
        self.cross_attn = CrossAttention2D(channels=out_channels, num_heads=4)
        self.pos_mlp = nn.Sequential(
            nn.Linear(1,16), nn.ReLU(), nn.Linear(16,out_channels))
        self.fusion = nn.Conv3d(out_channels*2, out_channels, 3, padding=1)
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

    def forward(self, x_t, coronal_2d, sagittal_2d, t, slice_pos=None):
        B, _, H, W, D = x_t.shape
        mid_h = H // 2; mid_w = W // 2
        if slice_pos is None:
            slice_pos = torch.full((B,), 0.5, device=x_t.device)
        cond_3d = self.slice_to_vol(coronal_2d, sagittal_2d, D, slice_pos, mid_h, mid_w)
        return self.unet(x_t, cond_3d, t)


# ── Checkpoint loading ────────────────────────────────────────────────────────
def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        if 'ema_state_dict' in ckpt:
            model.load_state_dict(ckpt['ema_state_dict'])
            print(f"  Loaded EMA weights from epoch {ckpt.get('epoch','?')}: {ckpt_path}")
        elif 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"  Loaded model weights from epoch {ckpt.get('epoch','?')}: {ckpt_path}")
        else:
            model.load_state_dict(ckpt)
            print(f"  Loaded plain state dict: {ckpt_path}")
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(ckpt)}")


# ── RePaint sampler (from synthetic, + clamp) ─────────────────────────────────
@torch.no_grad()
def sample_dvf_repaint(model, diffusion, cond_coronal, cond_sagittal,
                       known_coronal, known_sagittal, shape, slice_pos,
                       resampling_steps=RESAMPLING_STEPS,
                       temperature=TEMPERATURE):
    B, C, H, W, D = shape
    mid_h = H // 2; mid_w = W // 2

    mask = torch.zeros(shape, device=cond_coronal.device)
    mask[:, :, :, mid_w, :] = 1.0   # coronal plane lives at mid_w
    mask[:, :, mid_h, :, :] = 1.0   # sagittal plane lives at mid_h

    x_known_0 = torch.zeros(shape, device=cond_coronal.device)
    x_known_0[:, :, :, mid_w, :] = known_coronal    # (B,3,H,D) → fixes W
    x_known_0[:, :, mid_h, :, :] = known_sagittal   # (B,3,W,D) → fixes H

    x = torch.randn(shape, device=cond_coronal.device) * temperature

    for t in tqdm(reversed(range(diffusion.timesteps)),
                  total=diffusion.timesteps, desc="  RePaint", leave=False):
        for u in range(resampling_steps):
            tt      = torch.full((B,), t, device=cond_coronal.device, dtype=torch.long)
            eps_hat = model(x, cond_coronal, cond_sagittal, tt, slice_pos=slice_pos)

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

    pf = pred_mm.reshape(3,-1); gf = gt_mm.reshape(3,-1)
    cos = (pf*gf).sum(0) / (pf.norm(dim=0).clamp(1e-8) * gf.norm(dim=0).clamp(1e-8))
    m['cosine_sim_full'] = cos.mean().item()
    cos_vol = cos.reshape(H,W,D)
    m['cosine_sim_near'] = cos_vol[near_mask].mean().item()
    m['cosine_sim_far']  = cos_vol[far_mask].mean().item()

    return m


def print_and_save_metrics(m, case_name, tre, tre_std, ref, dvf_mean,
                           dvf_std_val, H, W, D, spc, out_path):
    lines = []
    lines.append(f"{'='*50}")
    lines.append(f"  Case: {case_name}")
    lines.append(f"  Method: RePaint U={RESAMPLING_STEPS} (cross-attn SliceToVolume)")
    lines.append(f"  Normalisation: mean={dvf_mean:.3f} mm, std={dvf_std_val:.3f} mm")
    lines.append(f"  DVF resolution: {H}x{W}x{D}  |  Orig spacing: {spc} mm")
    lines.append(f"{'='*50}")
    lines.append(f"  --- Full volume ---")
    for name in ['dx','dy','dz']:
        lines.append(f"  MAE {name} (full)   {m[f'mae_{name}_full']:>8.4f} mm")
    lines.append(f"  MDE (full)         {m['mde_full']:>8.4f} mm")
    lines.append(f"  Cosine sim (full)  {m['cosine_sim_full']:>8.4f}")
    lines.append(f"\n  --- Near known planes (< {FAR_THRESHOLD} vox) ---")
    for name in ['dx','dy','dz']:
        lines.append(f"  MAE {name} (near)   {m[f'mae_{name}_near']:>8.4f} mm")
    lines.append(f"  MDE (near)         {m['mde_near']:>8.4f} mm")
    lines.append(f"  Cosine sim (near)  {m['cosine_sim_near']:>8.4f}")
    lines.append(f"\n  --- Far from known planes (>= {FAR_THRESHOLD} vox) ---")
    for name in ['dx','dy','dz']:
        lines.append(f"  MAE {name} (far)    {m[f'mae_{name}_far']:>8.4f} mm")
    lines.append(f"  MDE (far)          {m['mde_far']:>8.4f} mm")
    lines.append(f"  Cosine sim (far)   {m['cosine_sim_far']:>8.4f}")
    lines.append(f"\n  --- TRE (DIR-Lab landmarks) ---")
    if tre is not None:
        lines.append(f"  TRE (landmarks):            {tre:>8.4f} ± {tre_std:.4f} mm")
        lines.append(f"  TRE initial (no reg):       {ref['initial']:>8.4f} mm")
        lines.append(f"  TRE after GT registration:  {ref['after_registration']:>8.4f} mm")
    else:
        lines.append(f"  TRE: landmark files not found")
    lines.append(f"{'='*50}")
    text = "\n".join(lines)
    print("\n" + text)
    with open(out_path, "w") as f:
        f.write(text)
    print(f"  Metrics saved to {out_path}")


# ── Plot (offset slices at +10, row labels) ───────────────────────────────────
def plot_case(case_name, gt_mm, pred_mm, mid_h, mid_w, mid_d, results_dir,
              plane_offset=PLANE_OFFSET,
              method_name=f"RePaint U={RESAMPLING_STEPS} (cross-attn)"):
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
        gt_slices   = [gt_c[:,:,mid_d],   gt_c[:,sag_idx,:], gt_c[cor_idx,:,:]]
        pred_slices = [pred_c[:,:,mid_d], pred_c[:,sag_idx,:], pred_c[cor_idx,:,:]]
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
        f"{case_name} — GT vs {method_name}  "
        f"(coronal/sagittal at +{plane_offset} slices from conditioning plane)",
        fontsize=12, fontweight='bold')
    plt.subplots_adjust(top=0.90, wspace=0.04, hspace=0.08, left=0.06, right=0.92)
    fname = os.path.join(results_dir, f"{case_name}_plot.png")
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fname}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log_path   = os.path.join(RESULTS_DIR, "sampling.log")
    tee        = Tee(log_path)
    sys.stdout = tee

    print("=" * 60)
    print(f"RePaint Real (cross-attention v2) — "
          f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Resampling steps: {RESAMPLING_STEPS}  |  Temperature: {TEMPERATURE}")
    print(f"Fixes: coronal=fix W, sagittal=fix H | direct DVF interp | x0 clamp")
    print(f"Test cases: {[f'copd{i+1:02d}' for i in TEST_CASES]}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    stats    = np.load(REAL_STATS, allow_pickle=True).item()
    dvf_mean = float(stats["mean"])
    dvf_std  = float(stats["std"])
    print(f"Real DVF stats: mean={dvf_mean:.4f} mm, std={dvf_std:.4f} mm")

    diffusion = DiffusionSchedule(timesteps=1000, device=device)
    model     = DiffusionModelManager(slice_feat_channels=SLICE_FEAT_CHANNELS).to(device)
    load_checkpoint(model, os.path.join(CKPT_DIR, "best_model.pth"), device)
    model.eval()

    all_metrics = {}

    for idx, case_idx in enumerate(TEST_CASES):
        case_name = f"copd{case_idx+1:02d}"
        spc = ORIG_SPACING[case_idx]
        print(f"\n{'─'*60}")
        print(f"Processing {case_name} ({idx+1}/{len(TEST_CASES)})...")

        dvf_gt_norm = load_and_preprocess_dvf(case_idx, dvf_mean, dvf_std)
        dvf_gt_norm = dvf_gt_norm.unsqueeze(0).to(device)
        B, C, H, W, D = dvf_gt_norm.shape
        mid_h = H//2; mid_w = W//2; mid_d = D//2

        # coronal fixes W, sagittal fixes H — consistent with fine-tuning
        cond_coronal  = dvf_gt_norm[:, :, :, mid_w, :]   # (B, 3, H, D)
        cond_sagittal = dvf_gt_norm[:, :, mid_h, :, :]   # (B, 3, W, D)
        known_coronal  = cond_coronal.clone()
        known_sagittal = cond_sagittal.clone()

        slice_pos = torch.full((B,), mid_w / (W - 1), device=device)
        print(f"  slice_pos={slice_pos[0].item():.4f}  mid_h={mid_h}  mid_w={mid_w}")
        print(f"  Running RePaint (U={RESAMPLING_STEPS})...")

        t0 = time.time()
        dvf_pred_norm = sample_dvf_repaint(
            model, diffusion, cond_coronal, cond_sagittal,
            known_coronal, known_sagittal,
            shape=dvf_gt_norm.shape, slice_pos=slice_pos,
            resampling_steps=RESAMPLING_STEPS, temperature=TEMPERATURE)
        print(f"  Done in {time.time()-t0:.1f}s")

        dvf_pred_mm = dvf_pred_norm.squeeze(0).cpu() * dvf_std + dvf_mean
        dvf_gt_mm   = dvf_gt_norm.squeeze(0).cpu()   * dvf_std + dvf_mean

        m = compute_all_metrics(dvf_pred_mm, dvf_gt_mm, mid_h, mid_w)

        try:
            pts_fix, pts_mov = load_landmarks(case_idx)
            tre, tre_std = compute_landmark_tre(dvf_pred_mm, pts_fix, pts_mov, case_idx)
        except FileNotFoundError:
            tre, tre_std = None, None

        ref = REFERENCE_TRE[case_idx]
        print_and_save_metrics(
            m, case_name, tre, tre_std, ref, dvf_mean, dvf_std,
            H, W, D, spc,
            out_path=os.path.join(RESULTS_DIR, f"{case_name}_metrics.txt"))

        all_metrics[case_name] = {**m, 'tre': tre}

        plot_case(case_name, dvf_gt_mm, dvf_pred_mm, mid_h, mid_w, mid_d,
                  RESULTS_DIR, plane_offset=PLANE_OFFSET)

    # ── Summary ───────────────────────────────────────────────────────────────
    tres = [all_metrics[k]['tre'] for k in all_metrics
            if all_metrics[k]['tre'] is not None]
    summary_lines = [f"SUMMARY — RePaint U={RESAMPLING_STEPS} Real (cross-attn v2)\n"]
    for key in ['mae_dx_full','mae_dy_full','mae_dz_full',
                'mde_full','mde_near','mde_far',
                'cosine_sim_full','cosine_sim_near','cosine_sim_far']:
        vals = [all_metrics[k][key] for k in all_metrics]
        summary_lines.append(f"  {key:<25} {np.mean(vals):.4f}")
    if tres:
        summary_lines.append(f"  {'TRE':<25} {np.mean(tres):.4f} mm")
    summary = "\n".join(summary_lines)
    print(f"\n{'='*60}\n{summary}\n{'='*60}")
    with open(os.path.join(RESULTS_DIR, "summary_metrics.txt"), "w") as f:
        f.write(summary)

    print(f"\nDone. Results: {RESULTS_DIR}")
    print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout = tee.terminal
    tee.close()


if __name__ == "__main__":
    main()