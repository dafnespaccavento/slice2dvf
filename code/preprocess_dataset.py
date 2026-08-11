"""
preprocess_v2.py
─────────────
Downsamples and z-score normalises the dataset and
saves it to disk so train.py never has to do interpolation at runtime.

Input  layout: DATA_DIR/{train,test}/{a,b}/
Output layout: OUT_DIR/{train,test}/{a,b}/   (same structure, smaller files)

Changes vs previous version:
- coronal/sagittal labels corrected (coronal = slice along W, sagittal = slice along H)
- added coronal_25, coronal_75, sagittal_25, sagittal_75 slices
- indices now reflect downsampled resolution instead of original
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = "/mimer/NOBACKUP/groups/caim1/dafne/datasets/smooth_synthetic"
OUT_DIR  = "/mimer/NOBACKUP/groups/caim1/dafne/datasets/smooth_synthetic_downsampled_v2"

# ── Target resolutions (must match train.py) ──────────────────────────────────
DVF_SIZE   = (128, 128, 64)   # trilinear  → (H, W, D)
SLICE_SIZE = (128, 64)        # bilinear   → used for all 2D slices


# ── Helpers ───────────────────────────────────────────────────────────────────
def downsample_dvf(dvf_np):
    """(3, H, W, D) numpy → (3, 128, 128, 64) numpy, trilinear."""
    t = torch.from_numpy(dvf_np).float().unsqueeze(0)   # (1, 3, H, W, D)
    t = F.interpolate(t, size=DVF_SIZE, mode="trilinear", align_corners=False)
    return t.squeeze(0).numpy()                          # (3, 128, 128, 64)


def downsample_slice(slice_np):
    """(3, X, Y) numpy → (3, 128, 64) numpy, bilinear."""
    t = torch.from_numpy(slice_np).float().unsqueeze(0)  # (1, 3, X, Y)
    t = F.interpolate(t, size=SLICE_SIZE, mode="bilinear", align_corners=False)
    return t.squeeze(0).numpy()                           # (3, 128, 64)


def get_ids(field_dir):
    return sorted([
        f.split("_")[1].split(".")[0]
        for f in os.listdir(field_dir)
        if f.startswith("field_")
    ])


def extract_slices(dvf_down):
    """
    Extract coronal and sagittal slices at 25%, 50%, 75% from a downsampled DVF.
    dvf_down: (3, H, W, D) numpy array at DVF_SIZE resolution.

    Conventions (matching real data):
      coronal  = slice along W axis (anterior-posterior) → shape (3, H, D)
      sagittal = slice along H axis (left-right)         → shape (3, W, D)
    """
    _, H, W, D = dvf_down.shape

    mid_h = H // 2
    mid_w = W // 2
    h_25  = int(0.25 * (H - 1))
    h_75  = int(0.75 * (H - 1))
    w_25  = int(0.25 * (W - 1))
    w_75  = int(0.75 * (W - 1))

    return {
        # coronal: fix W, vary H and D
        "coronal":    dvf_down[:, :, mid_w, :],  # (3, H, D)
        "coronal_25": dvf_down[:, :, w_25,  :],  # (3, H, D)
        "coronal_75": dvf_down[:, :, w_75,  :],  # (3, H, D)
        # sagittal: fix H, vary W and D
        "sagittal":    dvf_down[:, mid_h, :, :],  # (3, W, D)
        "sagittal_25": dvf_down[:, h_25,  :, :],  # (3, W, D)
        "sagittal_75": dvf_down[:, h_75,  :, :],  # (3, W, D)
        # indices at downsampled resolution
        "indices": {
            "mid_h": mid_h, "mid_w": mid_w,
            "h_25":  h_25,  "h_75":  h_75,
            "w_25":  w_25,  "w_75":  w_75,
        }
    }


# ── Pass 1: downsample and save everything, collect DVF values for stats ──────
def process_split(split, collect_stats=False):
    """
    Downsample one split and save to OUT_DIR.
    Slices are re-extracted from the downsampled DVF directly (not copied
    from the raw slice files) so labels and indices are always consistent.
    """
    in_field_dir  = os.path.join(DATA_DIR, split, "b")
    out_field_dir = os.path.join(OUT_DIR,  split, "b")
    out_slice_dir = os.path.join(OUT_DIR,  split, "a")
    os.makedirs(out_field_dir, exist_ok=True)
    os.makedirs(out_slice_dir, exist_ok=True)

    ids = get_ids(in_field_dir)
    print(f"\n[{split}] {len(ids)} samples → downsampling...")

    all_dvfs = [] if collect_stats else None

    for i in tqdm(ids, desc=split):
        # ── DVF ───────────────────────────────────────────────────────────────
        dvf      = np.load(os.path.join(in_field_dir, f"field_{i}.npy"))
        dvf_down = downsample_dvf(dvf)
        np.save(os.path.join(out_field_dir, f"field_{i}.npy"), dvf_down)

        if collect_stats:
            all_dvfs.append(dvf_down)

        # ── Slices — extracted from downsampled DVF with correct labels ───────
        slices = extract_slices(dvf_down)

        # downsample each 2D slice independently (they come from dvf_down
        # which is already at DVF_SIZE, so this just ensures consistent size)
        new_slices = {
            "coronal":     downsample_slice(slices["coronal"]),
            "coronal_25":  downsample_slice(slices["coronal_25"]),
            "coronal_75":  downsample_slice(slices["coronal_75"]),
            "sagittal":    downsample_slice(slices["sagittal"]),
            "sagittal_25": downsample_slice(slices["sagittal_25"]),
            "sagittal_75": downsample_slice(slices["sagittal_75"]),
            "indices":     slices["indices"],
        }
        np.save(os.path.join(out_slice_dir, f"slice_{i}.npy"), new_slices)

    return all_dvfs


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("DVF Dataset Preprocessor")
    print(f"  Input:  {DATA_DIR}")
    print(f"  Output: {OUT_DIR}")
    print(f"  DVF target:   {DVF_SIZE}   (trilinear)")
    print(f"  Slice target: {SLICE_SIZE}  (bilinear)")
    print("=" * 60)

    # ── Process train split and collect DVF arrays for stats ──────────────────
    all_dvfs = process_split("train", collect_stats=True)

    # ── Compute global mean/std on downsampled train DVFs ─────────────────────
    print("\nComputing global mean/std over downsampled train DVFs...")
    stacked  = np.stack(all_dvfs, axis=0)   # (N, 3, 128, 128, 64)
    dvf_mean = float(stacked.mean())
    dvf_std  = float(stacked.std())
    print(f"  mean = {dvf_mean:.6f}")
    print(f"  std  = {dvf_std:.6f}")

    # ── Z-score normalise and re-save train DVFs ──────────────────────────────
    print("\nNormalising and re-saving train DVFs...")
    in_field_dir  = os.path.join(DATA_DIR, "train", "b")
    out_field_dir = os.path.join(OUT_DIR,  "train", "b")
    ids = get_ids(in_field_dir)
    for i, dvf_down in tqdm(zip(ids, all_dvfs), total=len(ids), desc="normalise"):
        dvf_norm = (dvf_down - dvf_mean) / dvf_std
        np.save(os.path.join(out_field_dir, f"field_{i}.npy"), dvf_norm)

    # ── Save stats ────────────────────────────────────────────────────────────
    stats_path = os.path.join(OUT_DIR, "dvf_stats.npy")
    np.save(stats_path, {"mean": dvf_mean, "std": dvf_std})
    print(f"\nStats saved to: {stats_path}")

    # ── Process test split (normalise with train stats) ────────────────────────
    test_dvfs = process_split("test", collect_stats=True)
    print("\nNormalising and re-saving test DVFs...")
    in_field_dir  = os.path.join(DATA_DIR, "test", "b")
    out_field_dir = os.path.join(OUT_DIR,  "test", "b")
    ids = get_ids(in_field_dir)
    for i, dvf_down in tqdm(zip(ids, test_dvfs), total=len(ids), desc="normalise test"):
        dvf_norm = (dvf_down - dvf_mean) / dvf_std
        np.save(os.path.join(out_field_dir, f"field_{i}.npy"), dvf_norm)

    print("\n" + "=" * 60)
    print("Preprocessing complete.")
    print(f"  dvf_mean = {dvf_mean:.6f}")
    print(f"  dvf_std  = {dvf_std:.6f}")
    print(f"  Output:  {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()