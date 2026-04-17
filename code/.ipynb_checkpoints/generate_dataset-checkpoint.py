"""
generate_dataset.py
====================
Standalone script to generate 500 train + 100 test synthetic DVF samples.
Run on Alvis with: python generate_dataset.py
No Jupyter needed.
"""

import os
import glob
import numpy as np
from scipy.ndimage import gaussian_filter, zoom, distance_transform_edt
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
TARGET_H, TARGET_W, TARGET_D = 256, 256, 128
VOXEL_SPACING = np.array([0.625, 0.625, 2.5])
SAVE_DIR  = "/mimer/NOBACKUP/groups/caim1/dafne/datasets/synthetic"
NUM_TRAIN = 500
NUM_TEST  = 100

# ── Best params (from your Optuna run, verified in notebook) ──────────────────
BEST_PARAMS = {
    "dx_amp_lo":          3.484759969156242,
    "dx_amp_hi":          5.715480198336111,
    "dx_final_amp_lo":    10.158304214477539,
    "dx_final_amp_hi":    12.083813667297363,
    "dx_scale_xy":        0.08464117081239977,
    "dx_scale_z":         0.18389863384342217,
    "dx_sigma_xy":        4.440869682680044,
    "dx_sigma_z":         3.549173758546373,
    "dx_mask_blend":      0.5595341165401964,
    "dy_global_lo":       1.4846445322036743,
    "dy_global_hi":       2.5099542140960693,
    "dy_upper_lo":        2.112242685124402,
    "dy_upper_hi":        5.608946692563109,
    "dy_tex_amp_lo":      2.215081962070789,
    "dy_tex_amp_hi":      3.508053261542818,
    "dy_final_amp_lo":    26.00830078125,
    "dy_final_amp_hi":    31.184856414794922,
    "dy_tex_scale_xy":    0.04245019165986802,
    "dy_tex_scale_z":     0.11878493509233237,
    "dy_sigma_xy":        9.445708297660333,
    "dy_sigma_z":         3.536359506225657,
    "dy_clip_lo":        -1.4180407612586232,
    "dz_max_lo":          14.787049293518066,
    "dz_max_hi":          31.578336715698242,
    "dz_scale_xy":        0.0993956694420997,
    "dz_scale_z":         0.14657860966706757,
    "dz_sigma_xy":        3.1819976222354405,
    "dz_sigma_z":         1.1817841682383317,
    "dz_diag_lo":         4.065254765317933,
    "dz_diag_hi":         7.491865148044537,
    "dz_patch_amp_lo":    5.379836813918008,
    "dz_patch_amp_hi":    10.424273663219354,
    "dz_clip_neg_mult":   1.0101849062939523,
    "dz_clip_pos_mult":   0.5910514479089909,
    "final_smooth_dy":    1.9094213842997223,
    "final_smooth_dz":    3.4962417458890016,
    "boundary_ramp_width": 17.321939479672448,
}


# ── Lung mask ─────────────────────────────────────────────────────────────────
def make_lung_mask(H, W, D, margin=0.08):
    ys = np.linspace(0, 1, H)
    xs = np.linspace(0, 1, W)
    zs = np.linspace(0, 1, D)
    gy, gx, gz = np.meshgrid(ys, xs, zs, indexing='ij')
    cy, cx, cz = 0.5, 0.5, 0.45
    ry, rx, rz = 0.38, 0.40, 0.42
    dist = ((gy-cy)/ry)**2 + ((gx-cx)/rx)**2 + ((gz-cz)/rz)**2
    mask = np.clip(1.0 - (dist - 1.0 + margin) / margin, 0.0, 1.0)
    return mask.astype(np.float32)


# ── Base generator (TunableDVFGenerator) ─────────────────────────────────────
class TunableDVFGenerator:
    def __init__(self, shape, voxel_spacing, params):
        self.H, self.W, self.D = shape
        self.voxel_spacing     = np.array(voxel_spacing)
        self.p                 = params

        ys = np.linspace(0, 1, self.H)
        xs = np.linspace(0, 1, self.W)
        zs = np.linspace(0, 1, self.D)
        self.grid_y, self.grid_x, self.grid_z = np.meshgrid(
            ys, xs, zs, indexing='ij')
        self.diaphragm_grad = self.grid_z.astype(np.float32)
        self.lung_mask      = make_lung_mask(self.H, self.W, self.D)
        self.boundary_weight = self._make_boundary_weight()

    def _make_boundary_weight(self):
        lung_binary  = self.lung_mask > 0.5
        dist_outside = distance_transform_edt(~lung_binary).astype(np.float32)
        ramp_width   = max(2.0, self.p.get("boundary_ramp_width", 15.0))
        weight       = np.exp(-dist_outside / ramp_width)
        weight[lung_binary] = 1.0
        return weight.astype(np.float32)

    def _irregular_patch(self, scale_xy, scale_z, amplitude,
                         sigma_xy, sigma_z, n_layers=2):
        field = np.zeros((self.H, self.W, self.D), dtype=np.float32)
        for _ in range(n_layers):
            sh = max(2, int(self.H * scale_xy * np.random.uniform(0.7, 1.3)))
            sw = max(2, int(self.W * scale_xy * np.random.uniform(0.7, 1.3)))
            sd = max(2, int(self.D * scale_z  * np.random.uniform(0.7, 1.3)))
            seed  = np.random.randn(sh, sw, sd).astype(np.float32)
            layer = zoom(seed, (self.H/sh, self.W/sw, self.D/sd), order=3)
            layer = gaussian_filter(layer, sigma=(sigma_xy, sigma_xy, sigma_z))
            layer /= (np.abs(layer).max() + 1e-8)
            field += layer
        field /= (np.abs(field).max() + 1e-8)
        return (field * amplitude).astype(np.float32)

    def generate(self):
        p = self.p

        dy_global    = np.random.uniform(p["dy_global_lo"], p["dy_global_hi"])
        upper_weight = (1.0 - self.grid_y) * (1.0 - self.grid_x)
        dy_upper     = upper_weight * np.random.uniform(
            p["dy_upper_lo"], p["dy_upper_hi"])
        dy_texture   = self._irregular_patch(
            scale_xy=p["dy_tex_scale_xy"], scale_z=p["dy_tex_scale_z"],
            amplitude=np.random.uniform(p["dy_tex_amp_lo"], p["dy_tex_amp_hi"]),
            sigma_xy=p["dy_sigma_xy"] * 3.0,
            sigma_z =p["dy_sigma_z"]  * 3.0,
            n_layers=2)
        dy = dy_global + dy_upper + dy_texture
        dy_texture_component = dy - dy_global
        dy_tex_target_std = np.random.uniform(
            p["dy_final_amp_lo"], p["dy_final_amp_hi"])
        dy_texture_component = (dy_texture_component
                                / (dy_texture_component.std() + 1e-8)
                                * dy_tex_target_std)
        dy = dy_global + dy_texture_component
        dy = np.clip(dy, p["dy_clip_lo"], None)

        dz_max  = np.random.uniform(p["dz_max_lo"], p["dz_max_hi"])
        dz_base = -dz_max * self.diaphragm_grad
        a = np.random.uniform(0.3, 0.7)
        b = np.random.uniform(0.3, 0.7)
        diag = a * self.grid_y + b * self.grid_x
        diag = (diag - diag.mean()) / (diag.std() + 1e-8)
        dz_diag  = diag * np.random.uniform(p["dz_diag_lo"], p["dz_diag_hi"])
        dz_patch = self._irregular_patch(
            scale_xy=p["dz_scale_xy"], scale_z=p["dz_scale_z"],
            amplitude=np.random.uniform(
                p["dz_patch_amp_lo"], p["dz_patch_amp_hi"]),
            sigma_xy=p["dz_sigma_xy"], sigma_z=p["dz_sigma_z"], n_layers=3)
        dz = dz_base + dz_diag + dz_patch
        dz = np.clip(dz,
                     -dz_max * p["dz_clip_neg_mult"],
                      dz_max * p["dz_clip_pos_mult"])

        dx_main = self._irregular_patch(
            scale_xy=p["dx_scale_xy"], scale_z=p["dx_scale_z"],
            amplitude=np.random.uniform(p["dx_amp_lo"], p["dx_amp_hi"]),
            sigma_xy=p["dx_sigma_xy"], sigma_z=p["dx_sigma_z"], n_layers=3)
        dx = dx_main * (p["dx_mask_blend"] +
                        (1.0 - p["dx_mask_blend"]) * self.lung_mask)
        dx_target_std = np.random.uniform(
            p["dx_final_amp_lo"], p["dx_final_amp_hi"])
        dx = dx / (dx.std() + 1e-8) * dx_target_std

        flow = np.stack([dx, dy, dz], axis=-1)
        for c, sig in enumerate([p.get("final_smooth_dx", 1.05),
                                  p["final_smooth_dy"],
                                  p["final_smooth_dz"]]):
            flow[..., c] = gaussian_filter(flow[..., c], sigma=sig)
        for c in range(3):
            flow[..., c] *= self.boundary_weight
        flow *= self.voxel_spacing[np.newaxis, np.newaxis, np.newaxis, :]
        return flow.astype(np.float32)


# ── Fixed generator (FixedDXDVFGenerator) ────────────────────────────────────
class FixedDXDVFGenerator(TunableDVFGenerator):
    """
    dx generated FROM dz for spatial correlation + mean correction.
    dy and dz: identical to TunableDVFGenerator.
    """

    def generate(self):
        p = self.p

        # ── dy (AP) ───────────────────────────────────────────────────────────
        dy_global    = np.random.uniform(p["dy_global_lo"], p["dy_global_hi"])
        upper_weight = (1.0 - self.grid_y) * (1.0 - self.grid_x)
        dy_upper     = upper_weight * np.random.uniform(
            p["dy_upper_lo"], p["dy_upper_hi"])
        dy_texture   = self._irregular_patch(
            scale_xy=p["dy_tex_scale_xy"], scale_z=p["dy_tex_scale_z"],
            amplitude=np.random.uniform(p["dy_tex_amp_lo"], p["dy_tex_amp_hi"]),
            sigma_xy=p["dy_sigma_xy"] * 3.0,
            sigma_z =p["dy_sigma_z"]  * 3.0,
            n_layers=2)
        dy = dy_global + dy_upper + dy_texture
        dy_texture_component = dy - dy_global
        dy_tex_target_std = np.random.uniform(
            p["dy_final_amp_lo"], p["dy_final_amp_hi"])
        dy_texture_component = (dy_texture_component
                                / (dy_texture_component.std() + 1e-8)
                                * dy_tex_target_std)
        dy = dy_global + dy_texture_component
        dy = np.clip(dy, p["dy_clip_lo"], None)

        # ── dz (SI) ───────────────────────────────────────────────────────────
        dz_max  = np.random.uniform(p["dz_max_lo"], p["dz_max_hi"])
        dz_base = -dz_max * self.diaphragm_grad
        a = np.random.uniform(0.3, 0.7)
        b = np.random.uniform(0.3, 0.7)
        diag = a * self.grid_y + b * self.grid_x
        diag = (diag - diag.mean()) / (diag.std() + 1e-8)
        dz_diag  = diag * np.random.uniform(p["dz_diag_lo"], p["dz_diag_hi"])
        dz_patch = self._irregular_patch(
            scale_xy=p["dz_scale_xy"], scale_z=p["dz_scale_z"],
            amplitude=np.random.uniform(
                p["dz_patch_amp_lo"], p["dz_patch_amp_hi"]),
            sigma_xy=p["dz_sigma_xy"], sigma_z=p["dz_sigma_z"], n_layers=3)
        dz = dz_base + dz_diag + dz_patch
        dz = np.clip(dz,
                     -dz_max * p["dz_clip_neg_mult"],
                      dz_max * p["dz_clip_pos_mult"])

        # ── dx (LR) — derived from dz for spatial correlation ────────────────
        dz_norm = np.abs(dz) / (np.abs(dz).max() + 1e-8)
        dz_norm = gaussian_filter(dz_norm, sigma=3.0)
        sign    = np.random.choice([-1, 1])
        dx_corr = sign * dz_norm
        if dx_corr.std() > 1e-8:
            dx_corr = dx_corr / dx_corr.std()

        dx_noise = self._irregular_patch(
            scale_xy=p["dx_scale_xy"], scale_z=p["dx_scale_z"],
            amplitude=1.0,
            sigma_xy=p["dx_sigma_xy"] * 3.0,
            sigma_z =p["dx_sigma_z"]  * 3.0,
            n_layers=2)
        if dx_noise.std() > 1e-8:
            dx_noise = dx_noise / dx_noise.std()

        gamma = 0.70   # 70% correlated with dz, 30% independent
        dx = gamma * dx_corr + (1.0 - gamma) * dx_noise
        dx = dx * (p["dx_mask_blend"] +
                   (1.0 - p["dx_mask_blend"]) * self.lung_mask)

        dx_target_std = np.random.uniform(
            p["dx_final_amp_lo"], p["dx_final_amp_hi"])
        dx = dx / (dx.std() + 1e-8) * dx_target_std

        # Shift mean toward real dx mean (-1.32mm)
        dx_mean_target = np.random.uniform(-2.0, 0.0)
        dx = dx - dx.mean() + dx_mean_target

        # ── Final smoothing ───────────────────────────────────────────────────
        flow = np.stack([dx, dy, dz], axis=-1)
        for c, sig in enumerate([4.0,
                                  p["final_smooth_dy"],
                                  p["final_smooth_dz"]]):
            flow[..., c] = gaussian_filter(flow[..., c], sigma=sig)

        for c in range(3):
            flow[..., c] *= self.boundary_weight
        flow *= self.voxel_spacing[np.newaxis, np.newaxis, np.newaxis, :]
        return flow.astype(np.float32)


# ── Dataset generation ────────────────────────────────────────────────────────
def generate_dataset(generator, num_samples, save_dir, seed_offset=0):
    os.makedirs(f"{save_dir}/a", exist_ok=True)
    os.makedirs(f"{save_dir}/b", exist_ok=True)

    H, W, D = generator.H, generator.W, generator.D
    mid_h, mid_w, mid_d = H // 2, W // 2, D // 2

    existing = sorted(glob.glob(f"{save_dir}/b/field_*.npy"))
    start_i  = len(existing)

    if start_i >= num_samples:
        print(f"  Already complete ({num_samples} samples).")
        return

    if start_i > 0:
        print(f"  Resuming from sample {start_i} / {num_samples}")
    else:
        print(f"  Starting fresh — {num_samples} samples → {save_dir}")

    for i in tqdm(range(start_i, num_samples),
                  initial=start_i, total=num_samples):
        np.random.seed(seed_offset + i)
        flow_np = generator.generate()          # (H, W, D, 3)
        dvf_t   = flow_np.transpose(3, 0, 1, 2)  # (3, H, W, D)

        input_2d = {
            "coronal":  dvf_t[:, mid_h, :,  :],   # (3, W, D)
            "sagittal": dvf_t[:, :,  mid_w,  :],  # (3, H, D)
            "axial":    dvf_t[:, :,  :,  mid_d],  # (3, H, W)
            "indices":  {"mid_h": mid_h,
                         "mid_w": mid_w,
                         "mid_d": mid_d}
        }
        np.save(f"{save_dir}/a/slice_{i:05d}.npy", input_2d)
        np.save(f"{save_dir}/b/field_{i:05d}.npy", dvf_t)

    print(f"  Done.")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Initializing generator...")
    gen = FixedDXDVFGenerator(
        shape=(TARGET_H, TARGET_W, TARGET_D),
        voxel_spacing=VOXEL_SPACING,
        params=BEST_PARAMS
    )

    print(f"Generating training set ({NUM_TRAIN} samples)...")
    generate_dataset(gen, NUM_TRAIN,
                     f"{SAVE_DIR}/train", seed_offset=0)

    print(f"Generating test set ({NUM_TEST} samples)...")
    generate_dataset(gen, NUM_TEST,
                     f"{SAVE_DIR}/test", seed_offset=NUM_TRAIN)

    # Sanity check
    s = np.load(f"{SAVE_DIR}/train/b/field_00000.npy")
    sl = np.load(f"{SAVE_DIR}/train/a/slice_00000.npy",
                 allow_pickle=True).item()
    print(f"\nSanity check:")
    print(f"  DVF shape   : {s.shape}  (expected (3, {TARGET_H}, {TARGET_W}, {TARGET_D}))")
    print(f"  Slice keys  : {list(sl.keys())}")
    print(f"  dx stats    : mean={s[0].mean():+.2f}  std={s[0].std():.2f}")
    print(f"  dy stats    : mean={s[1].mean():+.2f}  std={s[1].std():.2f}")
    print(f"  dz stats    : mean={s[2].mean():+.2f}  std={s[2].std():.2f}")
    print(f"\nDataset saved to: {SAVE_DIR}")
