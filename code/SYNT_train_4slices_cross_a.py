"""
4-slice variant of the cross-attention + distance-channel architecture.

Conditioning input: four 2D slices
  - two coronal slices (fixing W) at 25% and 75% of W
  - two sagittal slices (fixing H) at 25% and 75% of H

Changes vs original 4-slice code (aligned with train.py cross_a):
  1. CrossAttention2D between the fused coronal and sagittal feature maps
     before broadcasting into 3D, so the four slices can exchange
     information across planes before fusion.
  2. Distance-to-slice channels: two extra scalar maps (normalised distance
     from the mid-coronal and mid-sagittal planes) appended to the
     conditioning volume. UNet cond_channels = SLICE_FEAT_CHANNELS + 2.
  3. Two ResidualBlock3D refinement blocks instead of one.
  4. SLICE_FEAT_CHANNELS constant added for consistency.
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
SAVE_DIR = '/mimer/NOBACKUP/groups/caim1/dafne/checkpoints/newdataset/16_condchannels/4slices/concat_slicetovolume/cross_a'
VIS_DIR  = '/mimer/NOBACKUP/groups/caim1/dafne/visual/newdataset/16_condchannels/4slices/concat_slicetovolume/cross_a'
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(VIS_DIR,  exist_ok=True)

SLICE_FEAT_CHANNELS = 16   # feature channels before distance channels
                            # UNet receives SLICE_FEAT_CHANNELS + 2 total


# ── Dataset ───────────────────────────────────────────────────────────────────
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

        cor_25 = torch.from_numpy(slices["coronal_25"]).float()   # (3, H, D)
        cor_75 = torch.from_numpy(slices["coronal_75"]).float()   # (3, H, D)
        sag_25 = torch.from_numpy(slices["sagittal_25"]).float()  # (3, W, D)
        sag_75 = torch.from_numpy(slices["sagittal_75"]).float()  # (3, W, D)

        indices = slices["indices"]
        W = dvf.shape[2]
        H = dvf.shape[1]

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
            nn.GroupNorm(8, channels),
            nn.ReLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x):
        return F.relu(x + self.block(x))


# ── CrossAttention2D (from train.py cross_a) ──────────────────────────────────
ATTN_POOL_SIZE = 128

class CrossAttention2D(nn.Module):
    """
    Memory-efficient bidirectional cross-attention between two 2D feature maps.
    cor_feat : (B, C, H, D)
    sag_feat : (B, C, W, D)
    Each map is pooled to ATTN_POOL_SIZE tokens before attention.
    """
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

        cor_res = F.interpolate(cor_attn.permute(0, 2, 1), size=H * D,
                                mode='linear', align_corners=False).permute(0, 2, 1)
        sag_res = F.interpolate(sag_attn.permute(0, 2, 1), size=W * D,
                                mode='linear', align_corners=False).permute(0, 2, 1)

        cor_seq = self.norm_cor(cor_feat.permute(0, 2, 3, 1).reshape(B, H * D, C) + cor_res)
        sag_seq = self.norm_sag(sag_feat.permute(0, 2, 3, 1).reshape(B, W * D, C) + sag_res)

        cor_feat = cor_seq.reshape(B, H, D, C).permute(0, 3, 1, 2)
        sag_feat = sag_seq.reshape(B, W, D, C).permute(0, 3, 1, 2)
        return cor_feat, sag_feat


# ── Distance-to-slice channels (from train.py cross_a) ───────────────────────
def make_distance_channels(B, H, W, D, mid_h, mid_w, device):
    """
    Returns (B, 2, H, W, D):
      channel 0 — normalised distance to the coronal plane  (varies along H)
      channel 1 — normalised distance to the sagittal plane (varies along W)
    """
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
    """
    Encodes four 2D slices into a 3D conditioning volume.

    Steps
    -----
    1. Encode each slice with shared 2D CNN encoders (coronal/sagittal).
    2. Sum the two coronal feature maps and the two sagittal feature maps
       into one representative coronal and one sagittal feature map.
    3. Apply CrossAttention2D between the fused coronal and sagittal maps.
    4. Inject per-slice position bias before broadcast.
    5. Broadcast into 3D, concatenate, fuse with Conv3d + two ResidualBlock3D.
    6. Append two normalised distance maps as extra channels.
    Returns (B, SLICE_FEAT_CHANNELS + 2, H, W, D).
    """
    def __init__(self, out_channels=SLICE_FEAT_CHANNELS):
        super().__init__()
        # shared encoders — same weights for 25% and 75% slices
        self.coronal_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, out_channels, 3, padding=1))
        self.sagittal_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, out_channels, 3, padding=1))

        # cross-attention on the fused coronal/sagittal feature maps
        self.cross_attn = CrossAttention2D(channels=out_channels, num_heads=4)

        self.pos_mlp = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, out_channels))

        # fusion: 2 volumes (coronal + sagittal) → out_channels
        self.fusion = nn.Conv3d(out_channels * 2, out_channels, 3, padding=1)

        # two residual blocks for more fusion capacity
        self.refine = nn.Sequential(
            ResidualBlock3D(out_channels),
            ResidualBlock3D(out_channels))

    def _encode_and_sum(self, encoder, slice_a, slice_b, pos_a, pos_b):
        """
        Encode two slices with the same encoder, apply position bias to each,
        and sum them into a single feature map.
        """
        feat_a = encoder(slice_a)   # (B, C, X, D)
        feat_b = encoder(slice_b)
        bias_a = self.pos_mlp(pos_a.unsqueeze(1))[:, :, None, None]
        bias_b = self.pos_mlp(pos_b.unsqueeze(1))[:, :, None, None]
        return feat_a * (1 + bias_a) + feat_b * (1 + bias_b)

    def forward(self, cor_25, cor_75, sag_25, sag_75,
                pos_cor_25, pos_cor_75, pos_sag_25, pos_sag_75):
        B = cor_25.shape[0]

        # 1 + 2. Encode and sum per plane
        cor_feat = self._encode_and_sum(
            self.coronal_encoder,  cor_25, cor_75, pos_cor_25, pos_cor_75)
        sag_feat = self._encode_and_sum(
            self.sagittal_encoder, sag_25, sag_75, pos_sag_25, pos_sag_75)

        H = cor_feat.shape[2]
        W = sag_feat.shape[2]
        D = cor_feat.shape[3]

        # 3. Cross-attention between fused coronal and sagittal maps
        cor_feat, sag_feat = self.cross_attn(cor_feat, sag_feat)

        # 4. Broadcast into 3D volume
        cor_vol = cor_feat.unsqueeze(3).expand(-1, -1, -1, W, -1)  # (B, C, H, W, D)
        sag_vol = sag_feat.unsqueeze(2).expand(-1, -1, H, -1, -1)  # (B, C, H, W, D)

        # 5. Fuse + refine
        fused = F.relu(self.fusion(torch.cat([cor_vol, sag_vol], dim=1)))
        fused = self.refine(fused)

        # 6. Distance channels — use volume mid-planes as reference
        mid_h = H // 2
        mid_w = W // 2
        dist  = make_distance_channels(B, H, W, D, mid_h, mid_w, cor_25.device)

        return torch.cat([fused, dist], dim=1)   # (B, C+2, H, W, D)


# ── UNet ──────────────────────────────────────────────────────────────────────
class UNet3D_Diffusion(nn.Module):
    def __init__(self, cond_channels):
        super().__init__()
        self.time_mlp      = nn.Sequential(TimeEmbedding(64), nn.Linear(64, 64), nn.ReLU())
        self.time_proj1    = nn.Linear(64, 64)
        self.time_proj2    = nn.Linear(64, 128)
        self.time_proj_mid = nn.Linear(64, 128)
        self.enc1 = nn.Conv3d(3 + cond_channels, 64,  3, padding=1)
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
        s1 = self.time_proj1(t_emb)[:, :, None, None, None]
        x1 = F.relu(self.enc1(x) * (1 + s1))
        x2 = F.relu(self.enc2(self.pool(x1)) + self.time_proj2(t_emb)[:, :, None, None, None])
        xm = F.relu(self.mid(x2) + self.time_proj_mid(t_emb)[:, :, None, None, None])
        xm = self.mid_res(xm)
        x  = self.dec1(xm) + x1
        return self.out(x).permute(0, 1, 3, 4, 2).contiguous()


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


# ── Loss ──────────────────────────────────────────────────────────────────────
def compute_gradient_loss(field, penalty='l2'):
    dh = torch.abs(field[:, :, 1:, :,  :] - field[:, :, :-1, :,  :])
    dw = torch.abs(field[:, :, :,  1:, :] - field[:, :, :,  :-1, :])
    dd = torch.abs(field[:, :, :,  :, 1:] - field[:, :, :,  :, :-1])
    if penalty == 'l2':
        dh, dw, dd = dh**2, dw**2, dd**2
    return (dh.mean() + dw.mean() + dd.mean()) / 3


def diffusion_loss(model, diffusion, x0,
                   cor_25, cor_75, sag_25, sag_75,
                   pos_cor_25, pos_cor_75, pos_sag_25, pos_sag_75,
                   lambda_smooth=1e-4, fixed_t=None):
    B = x0.shape[0]
    t = (torch.randint(0, diffusion.timesteps, (B,), device=x0.device)
         if fixed_t is None
         else torch.full((B,), fixed_t, device=x0.device, dtype=torch.long))

    noise = torch.randn_like(x0)
    x_t   = diffusion.q_sample(x0, t, noise)

    noise_pred = model(x_t, cor_25, cor_75, sag_25, sag_75,
                       pos_cor_25, pos_cor_75, pos_sag_25, pos_sag_75, t)
    mse_loss   = F.mse_loss(noise_pred, noise)

    s_ab      = diffusion.sqrt_alphas_bar[t].view(B, 1, 1, 1, 1)
    s_om      = diffusion.sqrt_one_minus_alphas_bar[t].view(B, 1, 1, 1, 1)
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


# ── Stats loader ──────────────────────────────────────────────────────────────
def load_dvf_stats():
    stats_path = os.path.join(DATA_DIR, "dvf_stats.npy")
    stats = np.load(stats_path, allow_pickle=True).item()
    mean  = float(stats["mean"])
    std   = float(stats["std"])
    print(f"Loaded DVF stats: mean={mean:.6f}, std={std:.6f}")
    return mean, std


# ── Fixed validation timesteps ────────────────────────────────────────────────
VAL_TIMESTEPS = [50, 250, 500, 750, 999]


def val_epoch(model, diffusion, val_loader, device, current_lambda):
    model.eval()
    val_total = val_mse = val_smooth = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            (cor_25, cor_75, sag_25, sag_75,
             pos_cor_25, pos_cor_75, pos_sag_25, pos_sag_75,
             dvf) = [b.to(device) for b in batch]

            batch_total = batch_mse = batch_smooth = 0.0
            for fixed_t in VAL_TIMESTEPS:
                total_loss, mse_loss, smooth_loss = diffusion_loss(
                    model, diffusion, dvf,
                    cor_25, cor_75, sag_25, sag_75,
                    pos_cor_25, pos_cor_75, pos_sag_25, pos_sag_75,
                    lambda_smooth=current_lambda, fixed_t=fixed_t)
                batch_total  += total_loss.item()
                batch_mse    += mse_loss.item()
                batch_smooth += smooth_loss.item() * current_lambda

            val_total  += batch_total  / len(VAL_TIMESTEPS)
            val_mse    += batch_mse    / len(VAL_TIMESTEPS)
            val_smooth += batch_smooth / len(VAL_TIMESTEPS)
            n_batches  += 1

    return val_total/n_batches, val_mse/n_batches, val_smooth/n_batches


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log_path   = os.path.join(SAVE_DIR, "training.log")
    tee        = Tee(log_path)
    sys.stdout = tee
    print(f"{'='*60}")
    print(f"Run started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file:    {log_path}")
    print(f"Slice feat channels: {SLICE_FEAT_CHANNELS}  |  "
          f"Total cond channels: {SLICE_FEAT_CHANNELS + 2}")
    print(f"{'='*60}")
    print(f"Using device: {device}")

    train_dataset = SyntheticDVFDataset(DATA_DIR, split="train")
    val_size      = int(0.15 * len(train_dataset))
    train_size    = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42))
    test_dataset = SyntheticDVFDataset(DATA_DIR, split="test")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                              num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False,
                              num_workers=8, pin_memory=True)

    print(f"Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    dvf_mean, dvf_std = load_dvf_stats()

    diffusion = DiffusionSchedule(timesteps=1000, device=device)
    model     = DiffusionModelManager(slice_feat_channels=SLICE_FEAT_CHANNELS).to(device)
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
        for batch in train_loader:
            (cor_25, cor_75, sag_25, sag_75,
             pos_cor_25, pos_cor_75, pos_sag_25, pos_sag_75,
             dvf) = [b.to(device) for b in batch]

            total_loss, mse_loss, smooth_loss = diffusion_loss(
                model, diffusion, dvf,
                cor_25, cor_75, sag_25, sag_75,
                pos_cor_25, pos_cor_75, pos_sag_25, pos_sag_75,
                lambda_smooth=current_lambda)
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
            f"{tag}")

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
            plt.suptitle(f"Training curves — epoch {epoch}", fontsize=13)
            plt.tight_layout()
            plt.savefig(os.path.join(VIS_DIR, "loss_curves.png"),
                        dpi=120, bbox_inches='tight')
            plt.close()

    print("Training complete.")
    sys.stdout = tee.terminal
    tee.close()


if __name__ == "__main__":
    main()
