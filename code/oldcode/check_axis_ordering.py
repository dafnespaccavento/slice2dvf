"""
check_axis_ordering.py
───────────────────────
Tries all 6 permutations of the 3 DVF component axes and prints
stats for each. The correct permutation should give:
  - dz (SI) largest magnitude, mostly negative (diaphragm moves down at exhale)
  - dx (LR) smallest magnitude
  - dy (AP) moderate

Also checks both sign conventions (±).

Run on login node: python check_axis_ordering.py
"""

import os
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from itertools import permutations


FIELDS_DIR = "/mimer/NOBACKUP/groups/caim1/dafne/datasets/real/fields"
DS_SHAPE   = (128, 128, 64)

# Load copd01 as representative case
path = os.path.join(FIELDS_DIR, "copd01.nii.gz")
print(f"Loading {path}...")
raw = nib.load(path).get_fdata().astype(np.float32)
print(f"Raw shape: {raw.shape}   (expected: H, W, D, 3)")
print(f"Raw header voxel sizes: {nib.load(path).header.get_zooms()}")
print()

# raw is (512, 512, N, 3) — last dim is the 3 DVF components
# Component 0,1,2 in the file — we don't know what they mean yet
raw_t = raw.transpose(3, 0, 1, 2)   # (3, 512, 512, N) — no axis reorder yet

# Downsample to DS_SHAPE for speed
dvf_t = torch.from_numpy(raw_t)
D = dvf_t.shape[-1]
target_D = 128
if D < target_D:
    pad_total  = target_D - D
    pad_before = pad_total // 2
    dvf_t = F.pad(dvf_t, (pad_before, target_D - D - pad_before))
elif D > target_D:
    crop_before = (D - target_D) // 2
    dvf_t = dvf_t[..., crop_before:crop_before + target_D]

dvf_t = dvf_t.unsqueeze(0)
dvf_t = F.interpolate(dvf_t, size=DS_SHAPE, mode='trilinear', align_corners=False)
dvf_t = dvf_t.squeeze(0).numpy()   # (3, 128, 128, 64)

print("Component stats BEFORE any axis reordering:")
labels = ["comp0", "comp1", "comp2"]
for c in range(3):
    v = dvf_t[c]
    # Ignore zero border (motion mask)
    nonzero = v[np.abs(v) > 0.5]
    if len(nonzero) == 0:
        nonzero = v.flatten()
    print(f"  {labels[c]}: mean={v.mean():+7.3f}  std={v.std():6.3f}  "
          f"nonzero_mean={nonzero.mean():+7.3f}  range=[{v.min():.2f}, {v.max():.2f}]")

print()
print("=" * 70)
print("Trying all 6 permutations of [comp0, comp1, comp2] → [LR, AP, SI]:")
print("Expected: |SI| > |AP| > |LR|, SI mean < 0 (exhale→inhale = down)")
print("=" * 70)

perms = list(permutations([0, 1, 2]))
results = []

for perm in perms:
    reordered = dvf_t[list(perm)]   # (3, H, W, D)
    stats = []
    for c in range(3):
        v = reordered[c]
        nonzero = v[np.abs(v) > 0.5]
        if len(nonzero) == 0:
            nonzero = v.flatten()
        stats.append({
            'mean':   float(nonzero.mean()),
            'std':    float(v.std()),
            'absmax': float(np.abs(v).max()),
        })

    lr_std = stats[0]['std']
    ap_std = stats[1]['std']
    si_std = stats[2]['std']
    si_neg = stats[2]['mean'] < 0

    # Score: SI should be largest, LR smallest, SI mean negative
    size_ok   = (si_std > ap_std > lr_std)
    sign_ok   = si_neg

    results.append({
        'perm':     perm,
        'stats':    stats,
        'size_ok':  size_ok,
        'sign_ok':  sign_ok,
        'score':    int(size_ok) + int(sign_ok),
    })

# Sort by score descending
results.sort(key=lambda x: -x['score'])

for r in results:
    perm = r['perm']
    s    = r['stats']
    tag  = ""
    if r['score'] == 2:
        tag = "  ✓ LIKELY CORRECT"
    elif r['score'] == 1:
        tag = "  ~ partial match"

    print(f"\n  Permutation {perm}  →  [LR=comp{perm[0]}, AP=comp{perm[1]}, SI=comp{perm[2]}]{tag}")
    print(f"    LR: mean={s[0]['mean']:+7.3f}  std={s[0]['std']:.3f}")
    print(f"    AP: mean={s[1]['mean']:+7.3f}  std={s[1]['std']:.3f}")
    print(f"    SI: mean={s[2]['mean']:+7.3f}  std={s[2]['std']:.3f}  "
          f"(neg={'✓' if s[2]['mean']<0 else '✗'}  largest={'✓' if r['size_ok'] else '✗'})")

print()
print("=" * 70)
print("Current code uses: dvf[..., [0, 2, 1]]  →  perm=(0, 2, 1)")
print("Check if that matches the ✓ LIKELY CORRECT entry above.")
