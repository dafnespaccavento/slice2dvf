"""
analyze_distributions.py
─────────────────────────
Compares the DVF distributions of:
  1. Synthetic training data (already downsampled + normalised)
  2. Real COPD data (raw, before any normalisation)

Prints stats and saves histograms to VIS_DIR.

Usage:
    python analyze_distributions.py
"""

import os
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Paths ─────────────────────────────────────────────────────────────────────
SYNTH_DIR  = "/mimer/NOBACKUP/groups/caim1/dafne/datasets/smooth_synthetic_downsampled"
FIELDS_DIR = "/mimer/NOBACKUP/groups/caim1/dafne/datasets/real/fields"
STATS_PATH = "/mimer/NOBACKUP/groups/caim1/dafne/datasets/smooth_synthetic_downsampled/dvf_stats.npy"
VIS_DIR    = "/mimer/NOBACKUP/groups/caim1/dafne/visual_analysis"
os.makedirs(VIS_DIR, exist_ok=True)

ALL_CASES  = list(range(10))   # copd01-10
DS_SHAPE   = (128, 128, 64)


# ── Load synthetic stats ───────────────────────────────────────────────────────
stats    = np.load(STATS_PATH, allow_pickle=True).item()
SYN_MEAN = float(stats["mean"])
SYN_STD  = float(stats["std"])
print(f"Synthetic normalisation stats: mean={SYN_MEAN:.4f}, std={SYN_STD:.4f}")


# ── Load a sample of synthetic DVFs (raw, before normalisation) ───────────────
print("\n=== Synthetic DVFs (downsampled, before z-score) ===")
synth_field_dir = os.path.join(SYNTH_DIR, "train", "b")
synth_ids = sorted([
    f.split("_")[1].split(".")[0]
    for f in os.listdir(synth_field_dir)
    if f.startswith("field_")
])[:50]   # first 50 samples

synth_vals = []
for i in synth_ids:
    dvf = np.load(os.path.join(synth_field_dir, f"field_{i}.npy"))
    # These are already normalised — denormalise to get raw mm values
    dvf_mm = dvf * SYN_STD + SYN_MEAN
    synth_vals.append(dvf_mm.flatten())

synth_vals = np.concatenate(synth_vals)
print(f"  N voxels sampled: {len(synth_vals):,}")
print(f"  Mean:   {synth_vals.mean():.4f} mm")
print(f"  Std:    {synth_vals.std():.4f} mm")
print(f"  Min:    {synth_vals.min():.4f} mm")
print(f"  Max:    {synth_vals.max():.4f} mm")
print(f"  p1:     {np.percentile(synth_vals, 1):.4f} mm")
print(f"  p99:    {np.percentile(synth_vals, 99):.4f} mm")

# Per-component
synth_per_comp = []
for i in synth_ids:
    dvf = np.load(os.path.join(synth_field_dir, f"field_{i}.npy"))
    dvf_mm = dvf * SYN_STD + SYN_MEAN   # (3, H, W, D)
    synth_per_comp.append(dvf_mm)

synth_per_comp = np.stack(synth_per_comp, axis=0)   # (N, 3, H, W, D)
for c, cname in enumerate(["dx (LR)", "dy (AP)", "dz (SI)"]):
    vals = synth_per_comp[:, c].flatten()
    print(f"  {cname}: mean={vals.mean():.3f}  std={vals.std():.3f}  "
          f"range=[{vals.min():.2f}, {vals.max():.2f}]")


# ── Load real DVFs (raw mm, before any normalisation) ─────────────────────────
print("\n=== Real COPD DVFs (raw mm, before normalisation) ===")

def load_real_dvf_raw(case_idx):
    path = os.path.join(FIELDS_DIR, f"copd{case_idx + 1:02d}.nii.gz")
    dvf  = nib.load(path).get_fdata().astype(np.float32)   # (512, 512, N, 3)
    dvf  = dvf[..., [0, 2, 1]]                              # axis fix
    dvf  = dvf.transpose(3, 0, 1, 2)                        # (3, 512, 512, N)
    dvf  = torch.from_numpy(dvf)

    D = dvf.shape[-1]
    target_D = 128
    if D < target_D:
        pad_total  = target_D - D
        pad_before = pad_total // 2
        pad_after  = pad_total - pad_before
        dvf = F.pad(dvf, (pad_before, pad_after))
    elif D > target_D:
        crop_total  = D - target_D
        crop_before = crop_total // 2
        dvf = dvf[..., crop_before:crop_before + target_D]

    dvf = dvf.unsqueeze(0)
    dvf = F.interpolate(dvf, size=DS_SHAPE, mode='trilinear', align_corners=False)
    dvf = dvf.squeeze(0).numpy()   # (3, 128, 128, 64) in mm
    return dvf

real_dvfs = {}
real_vals_all = []
real_per_comp = []

for idx in ALL_CASES:
    dvf_mm = load_real_dvf_raw(idx)
    real_dvfs[idx] = dvf_mm
    real_vals_all.append(dvf_mm.flatten())
    real_per_comp.append(dvf_mm)

    mag  = np.sqrt((dvf_mm ** 2).sum(axis=0)).mean()
    norm = (dvf_mm - SYN_MEAN) / SYN_STD
    print(f"  copd{idx+1:02d}:  mean_mag={mag:.3f} mm  "
          f"raw range=[{dvf_mm.min():.2f}, {dvf_mm.max():.2f}]  "
          f"normalised range=[{norm.min():.2f}, {norm.max():.2f}]")

real_vals_all  = np.concatenate(real_vals_all)
real_per_comp  = np.stack(real_per_comp, axis=0)   # (10, 3, H, W, D)

print(f"\n  All cases combined:")
print(f"  Mean:   {real_vals_all.mean():.4f} mm")
print(f"  Std:    {real_vals_all.std():.4f} mm")
print(f"  Min:    {real_vals_all.min():.4f} mm")
print(f"  Max:    {real_vals_all.max():.4f} mm")
print(f"  p1:     {np.percentile(real_vals_all, 1):.4f} mm")
print(f"  p99:    {np.percentile(real_vals_all, 99):.4f} mm")

for c, cname in enumerate(["dx (LR)", "dy (AP)", "dz (SI)"]):
    vals = real_per_comp[:, c].flatten()
    print(f"  {cname}: mean={vals.mean():.3f}  std={vals.std():.3f}  "
          f"range=[{vals.min():.2f}, {vals.max():.2f}]")

# What would the real data stats be if we normalised independently?
real_mean = real_vals_all.mean()
real_std  = real_vals_all.std()
print(f"\n  Real data independent stats: mean={real_mean:.4f}, std={real_std:.4f}")
print(f"  Synthetic stats:             mean={SYN_MEAN:.4f},  std={SYN_STD:.4f}")
print(f"  Difference in mean: {abs(real_mean - SYN_MEAN):.4f} mm")
print(f"  Ratio of stds:      {real_std / SYN_STD:.4f}")

# After applying synthetic normalisation to real data
real_norm = (real_vals_all - SYN_MEAN) / SYN_STD
print(f"\n  Real data after synthetic z-score:")
print(f"  Mean:  {real_norm.mean():.4f}  (should be ~0 if distributions match)")
print(f"  Std:   {real_norm.std():.4f}   (should be ~1 if distributions match)")
print(f"  Range: [{real_norm.min():.2f}, {real_norm.max():.2f}]")


# ── Plots ─────────────────────────────────────────────────────────────────────
print("\nGenerating plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

component_names = ["dx (LR)", "dy (AP)", "dz (SI)"]
colors_synth = ['steelblue', 'steelblue', 'steelblue']
colors_real  = ['tomato',    'tomato',    'tomato']

for c, cname in enumerate(component_names):
    ax = axes[c]

    synth_c = synth_per_comp[:, c].flatten()
    real_c  = real_per_comp[:, c].flatten()

    # Subsample for speed
    rng = np.random.default_rng(42)
    synth_c = rng.choice(synth_c, size=min(500_000, len(synth_c)), replace=False)
    real_c  = rng.choice(real_c,  size=min(500_000, len(real_c)),  replace=False)

    bins = np.linspace(
        min(synth_c.min(), real_c.min()),
        max(synth_c.max(), real_c.max()),
        100
    )
    ax.hist(synth_c, bins=bins, alpha=0.6, color='steelblue',
            density=True, label=f"Synthetic (n=50)")
    ax.hist(real_c,  bins=bins, alpha=0.6, color='tomato',
            density=True, label=f"Real COPD (n=10)")
    ax.set_title(f"{cname} distribution (mm)", fontsize=11)
    ax.set_xlabel("DVF value (mm)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle("Synthetic vs Real DVF distributions (raw mm, downsampled to 128x128x64)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
fname = os.path.join(VIS_DIR, "distribution_comparison_mm.png")
plt.savefig(fname, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved {fname}")

# ── Normalised space comparison ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for c, cname in enumerate(component_names):
    ax = axes[c]

    synth_c = synth_per_comp[:, c].flatten()
    synth_c_norm = (synth_c - SYN_MEAN) / SYN_STD

    real_c  = real_per_comp[:, c].flatten()
    real_c_norm  = (real_c  - SYN_MEAN) / SYN_STD
    real_c_norm2 = (real_c  - real_mean) / real_std   # with own stats

    rng = np.random.default_rng(42)
    synth_c_norm = rng.choice(synth_c_norm, size=min(500_000, len(synth_c_norm)), replace=False)
    real_c_norm  = rng.choice(real_c_norm,  size=min(500_000, len(real_c_norm)),  replace=False)
    real_c_norm2 = rng.choice(real_c_norm2, size=min(500_000, len(real_c_norm2)), replace=False)

    bins = np.linspace(-6, 6, 100)
    ax.hist(synth_c_norm, bins=bins, alpha=0.6, color='steelblue',
            density=True, label="Synthetic (synth stats)")
    ax.hist(real_c_norm,  bins=bins, alpha=0.5, color='tomato',
            density=True, label="Real (synth stats)")
    ax.hist(real_c_norm2, bins=bins, alpha=0.4, color='green',
            density=True, label="Real (own stats)")
    ax.set_title(f"{cname} — normalised", fontsize=11)
    ax.set_xlabel("Normalised DVF value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle("Normalised DVF distributions — synthetic stats vs real-own stats",
             fontsize=13, fontweight='bold')
plt.tight_layout()
fname = os.path.join(VIS_DIR, "distribution_comparison_normalised.png")
plt.savefig(fname, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved {fname}")

print(f"\nDone. All plots saved to {VIS_DIR}")
