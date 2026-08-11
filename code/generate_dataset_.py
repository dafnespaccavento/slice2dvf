

import os
import glob
import numpy as np
from scipy.ndimage import gaussian_filter, zoom, distance_transform_edt
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
TARGET_H, TARGET_W, TARGET_D = 256, 256, 128
VOXEL_SPACING = np.array([0.625, 0.625, 2.5])
SAVE_DIR  = "/mimer/NOBACKUP/groups/caim1/dafne/datasets/smooth_synthetic"
NUM_TRAIN = 500
NUM_TEST  = 100

# ── best_params_lohi ──────────────────────────────────────────────────────────
BEST_PARAMS = {
    "dx_amp_mid":         4.600120083746177,
    "dx_amp_range":       2.2307202291798696,
    "dx_final_mid":       5.785035594274386,
    "dx_final_range":     1.0016255130143858,
    "dx_scale_xy":        0.08464117081239977,
    "dx_scale_z":         0.18389863384342217,
    "dx_sigma_xy":        4.440869682680044,
    "dx_sigma_z":         3.549173758546373,
    "dx_mask_blend":      0.5595341165401964,
    "dy_global_mid":      13.92300086342907,
    "dy_global_range":    7.147343174155115,
    "dy_upper_mid":       3.8605946888437552,
    "dy_upper_range":     3.4967040074387064,
    "dy_tex_amp_mid":     2.8615676118068034,
    "dy_tex_amp_range":   1.2929712994720288,
    "dy_final_mid":       15.45641821509185,
    "dy_final_range":     2.7979233812976645,
    "dy_tex_scale_xy":    0.04245019165986802,
    "dy_tex_scale_z":     0.11878493509233237,
    "dy_sigma_xy":        9.445708297660333,
    "dy_sigma_z":         3.536359506225657,
    "dy_clip_lo":        -1.4180407612586232,
    "dz_max_mid":         16.550798679741778,
    "dz_max_range":       11.987788587157493,
    "dz_diag_mid":        5.778559956681235,
    "dz_diag_range":      3.426610382726605,
    "dz_patch_mid":       7.902055238568681,
    "dz_patch_range":     5.044436849301348,
    "dz_scale_xy":        0.0993956694420997,
    "dz_scale_z":         0.14657860966706757,
    "dz_sigma_xy":        3.1819976222354405,
    "dz_sigma_z":         1.1817841682383317,
    "dz_clip_neg_mult":   1.0101849062939523,
    "dz_clip_pos_mult":   0.5910514479089909,
    "final_smooth_dx":    1.0527499802410345,
    "final_smooth_dy":    1.9094213842997223,
    "final_smooth_dz":    3.4962417458890016,
    "boundary_ramp_width": 17.321939479672448,
    "dx_amp_lo":          3.484759969156242,
    "dx_amp_hi":          5.715480198336111,
    "dx_final_amp_lo":    10.158304214477539,
    "dx_final_amp_hi":    12.083813667297363,
    "dy_global_lo":       1.4846445322036743,
    "dy_global_hi":       2.5099542140960693,
    "dy_upper_lo":        2.112242685124402,
    "dy_upper_hi":        5.608946692563109,
    "dy_tex_amp_lo":      2.215081962070789,
    "dy_tex_amp_hi":      3.508053261542818,
    "dy_final_amp_lo":    15.00830078125,
    "dy_final_amp_hi":    21.184856414794922,
    "dz_max_lo":          14.787049293518066,
    "dz_max_hi":          31.578336715698242,
    "dz_diag_lo":         4.065254765317933,
    "dz_diag_hi":         7.491865148044537,
    "dz_patch_amp_lo":    5.379836813918008,
    "dz_patch_amp_hi":    10.424273663219354,
    "dx_coupling_strength": 0.6,
}


# ── make_lung_mask ────────────────────────────────────────────────────────────
def make_lung_mask(H, W, D):
    ys = np.linspace(0, 1, H)
    xs = np.linspace(0, 1, W)
    zs = np.linspace(0, 1, D)
    gy, gx, gz = np.meshgrid(ys, xs, zs, indexing='ij')

    left  = np.exp(-((gy - 0.50)**2 / 0.18**2 + (gx - 0.28)**2 / 0.15**2))
    right = np.exp(-((gy - 0.50)**2 / 0.18**2 + (gx - 0.72)**2 / 0.15**2))
    mask_xy = np.clip(left + right, 0, 1)
    mask_z  = np.exp(-((gz - 0.45)**2 / 0.30**2))

    return (mask_xy * mask_z).astype(np.float32)


# ── TunableDVFGenerator ───────────────────────────────────────────────────────
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
        self.diaphragm_grad  = self.grid_z.astype(np.float32)
        self.lung_mask       = make_lung_mask(self.H, self.W, self.D)
        self.boundary_weight = self._make_boundary_weight()

    def _make_boundary_weight(self):
        lung_binary  = self.lung_mask > 0.5
        dist_outside = distance_transform_edt(~lung_binary).astype(np.float32)
        ramp_width   = max(2.0, self.H * 15.0 / 64.0)
        weight       = np.exp(-dist_outside / ramp_width)
        weight[lung_binary] = 1.0
        return weight.astype(np.float32)

    def _irregular_patch(self, scale_xy, scale_z, amplitude,
                     sigma_xy, sigma_z, n_layers=2):
        field = np.zeros((self.H, self.W, self.D), dtype=np.float32)
        for _ in range(n_layers):
            seed   = np.random.randn(self.H, self.W, self.D).astype(np.float32)  # full res, no zoom
            sig_xy = self.H * scale_xy
            sig_z  = self.D * scale_z
            layer  = gaussian_filter(seed, sigma=(sig_xy, sig_xy, sig_z))
            layer /= (layer.std() + 1e-8)                                         # std norm, not abs max
            field += layer
        field /= (field.std() + 1e-8)                                             # std norm, not abs max
        return (field * amplitude).astype(np.float32)

    def generate(self):
        p = self.p

        dx_main = self._irregular_patch(
            scale_xy=p["dx_scale_xy"], scale_z=p["dx_scale_z"],
            amplitude=np.random.uniform(p["dx_amp_lo"], p["dx_amp_hi"]),
            sigma_xy=p["dx_sigma_xy"], sigma_z=p["dx_sigma_z"], n_layers=3)
        dx = dx_main * (p["dx_mask_blend"] + (1.0 - p["dx_mask_blend"]) * self.lung_mask)
        dx_target_std = np.random.uniform(p["dx_final_amp_lo"], p["dx_final_amp_hi"])
        dx = dx / (dx.std() + 1e-8) * dx_target_std

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
        dy_tex_target_std = np.random.uniform(p["dy_final_amp_lo"], p["dy_final_amp_hi"])
        dy_texture_component = dy_texture_component / (dy_texture_component.std() + 1e-8) \
                               * dy_tex_target_std
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
            amplitude=np.random.uniform(p["dz_patch_amp_lo"], p["dz_patch_amp_hi"]),
            sigma_xy=p["dz_sigma_xy"], sigma_z=p["dz_sigma_z"], n_layers=3)
        dz = dz_base + dz_diag + dz_patch
        dz = np.clip(dz, -dz_max * p["dz_clip_neg_mult"],
                          dz_max * p["dz_clip_pos_mult"])

        flow = np.stack([dx, dy, dz], axis=-1)
        for c, sig in enumerate([p["final_smooth_dx"],
                                  p["final_smooth_dy"],
                                  p["final_smooth_dz"]]):
            flow[..., c] = gaussian_filter(flow[..., c], sigma=sig)

        for c in range(3):
            flow[..., c] *= self.boundary_weight
        flow *= self.voxel_spacing[np.newaxis, np.newaxis, np.newaxis, :]
        return flow.astype(np.float32)


# ── FixedDXDVFGenerator ───────────────────────────────────────────────────────
class FixedDXDVFGenerator(TunableDVFGenerator):

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

        # ── dx (LR) ───────────────────────────────────────────────────────────
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

        gamma = 0.70
        dx = gamma * dx_corr + (1.0 - gamma) * dx_noise
        dx = dx * (p["dx_mask_blend"] +
                   (1.0 - p["dx_mask_blend"]) * self.lung_mask)
        dx_target_std = np.random.uniform(
            p["dx_final_amp_lo"], p["dx_final_amp_hi"])
        dx = dx / (dx.std() + 1e-8) * dx_target_std
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
        flow_np = generator.generate()
        dvf_t   = flow_np.transpose(3, 0, 1, 2)
        input_2d = {
            "coronal":  dvf_t[:, mid_h, :,  :],
            "sagittal": dvf_t[:, :,  mid_w,  :],
            "axial":    dvf_t[:, :,  :,  mid_d],
            "indices":  {"mid_h": mid_h, "mid_w": mid_w, "mid_d": mid_d}
        }
        np.save(f"{save_dir}/a/slice_{i:05d}.npy", input_2d)
        np.save(f"{save_dir}/b/field_{i:05d}.npy", dvf_t)

    print(f"  Done.")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Initializing generator...")
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

    s  = np.load(f"{SAVE_DIR}/train/b/field_00000.npy")
    print(f"\nSanity check:")
    print(f"  DVF shape : {s.shape}")
    print(f"  dx: mean={s[0].mean():+.2f}  std={s[0].std():.2f}")
    print(f"  dy: mean={s[1].mean():+.2f}  std={s[1].std():.2f}")
    print(f"  dz: mean={s[2].mean():+.2f}  std={s[2].std():.2f}")
    print(f"\nDone. Dataset saved to: {SAVE_DIR}")
