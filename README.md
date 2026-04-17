# Master's Thesis — DVF Reconstruction with Diffusion Models

**Title:** 3D Deformation Vector Field Reconstruction from Sparse 2D 
Observations using Conditional Diffusion Models

**Author:** Dafne Spaccavento 
**Programme:** Image Analysis and Machine Learning  
**Institution:** Uppsala University, Department of Information Technology 

---

## Overview

This thesis investigates whether a conditional denoising diffusion 
probabilistic model (DDPM) can reconstruct a physically plausible 
3D deformation vector field (DVF) from only two orthogonal 2D 
mid-plane slices (coronal and sagittal), motivated by the clinical 
need for real-time 3D motion estimation during MR-Linac radiotherapy 
treatments.

---

## Repository Structure
slice2dvf/
├── code/                    
├── visual/  # Figures and images
├── results/
└── README.md

---

## Data

| Dataset | Location | Description |
|---|---|---|
| Real DVFs | `/mimer/.../real/fields/` | 10 COPD cases, DIR-Lab 4DCT |
| Synthetic (full res) | `/mimer/.../smooth_synthetic/` | 500 train / 100 test at 256×256×128 |
| Synthetic (downsampled) | `/mimer/.../smooth_synthetic_downsampled/` | z-score normalised at 128×128×64 |
| Visualisations | `/mimer/.../visual/` | Saved figures |

---

## Key Scripts

| Script | Description |
|---|---|
| `optimize_dvf_generator.ipynb` | Bayesian optimisation of synthetic DVF generator parameters using Optuna |
| `generate_dataset_v2.py` | Generates 500 train + 100 test synthetic DVFs using optimised parameters |
| `preprocess.py` | Downsamples and z-score normalises the dataset to 128×128×64 |
| `train.py` | Trains the conditional DDPM on the synthetic dataset |
| `sample_ddpm.py` | DDPM stochastic sampling at inference time |
| `sample_ddim.py` | DDIM deterministic sampling at inference time |
| `sample_repaint.py` | RePaint inpainting-based sampling enforcing known slice constraints |
| `finetune_real.py` | Fine-tunes the pre-trained model on the real DIR-Lab 4DCT cases |

---

## Dependencies

- Python 3.10+
- `numpy`, `nibabel`, `scipy`, `matplotlib`
- `optuna` (Bayesian optimisation)
- `torch` (preprocessing and training)
- `tqdm`

---

## Notes

- The real DVFs are in millimetres and stored as NIfTI files (`.nii.gz`)
- The synthetic DVFs at full resolution are in mm; the downsampled 
  versions are z-score normalised and **not** in mm
- Voxel spacing of real data: 0.586--0.742 mm in-plane, 2.5 mm slice spacing
- Synthetic data generated at uniform 0.625×0.625×2.5 mm spacing
