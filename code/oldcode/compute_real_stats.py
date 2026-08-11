"""
compute_real_stats.py
─────────────────────
Computes global mean and std from real COPD training cases (copd01-08),
using only motion voxels (mag > 0.5mm) to exclude the zero border.

Saves stats to datasets/real/real_dvf_stats.npy

Run once on login node: python compute_real_stats.py
"""

import os
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

FIELDS_DIR      = "/mimer/NOBACKUP/groups/caim1/dafne/datasets/real/fields"
STATS_OUT_PATH  = "/mimer/NOBACKUP/groups/caim1/dafne/datasets/real/real_dvf_stats.npy"
TRAIN_CASES     = [0, 1, 2, 3, 4, 5, 6, 7]   # copd01-08
DS_SHAPE        = (128, 128, 64)
MOTION_THRESHOLD = 0.5   # mm

all_vals = []

for case_idx in TRAIN_CASES:
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

    # Only motion voxels
    mask = dvf.norm(dim=0) > MOTION_THRESHOLD
    motion_vals = dvf[:, mask].numpy()
    all_vals.append(motion_vals.flatten())

    print(f"  copd{case_idx+1:02d}: {mask.sum().item()} motion voxels  "
          f"range=[{dvf.min():.2f}, {dvf.max():.2f}]  "
          f"motion range=[{motion_vals.min():.2f}, {motion_vals.max():.2f}]")

all_vals = np.concatenate(all_vals)
real_mean = float(all_vals.mean())
real_std  = float(all_vals.std())

print(f"\nGlobal stats from motion voxels of copd01-08:")
print(f"  mean = {real_mean:.6f} mm")
print(f"  std  = {real_std:.6f} mm")
print(f"  range = [{all_vals.min():.3f}, {all_vals.max():.3f}]")

np.save(STATS_OUT_PATH, {"mean": real_mean, "std": real_std})
print(f"\nSaved to: {STATS_OUT_PATH}")
