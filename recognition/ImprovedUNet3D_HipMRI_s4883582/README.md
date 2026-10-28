# 3D Improved U-Net for Prostate Segmentation on HipMRI

**Student:** s4883582 · **Task:** 3D Improved U-Net [Hard Difficulty] on downsampled HipMRI 3D data ·
**Baseline:** 2D U-Net in [`../UNet2D_HipMRI_s4883582`](../UNet2D_HipMRI_s4883582)

## Contents
1. [Problem](#1-problem)

External-beam radiotherapy for different cancers is a millimetre precise therapy. For prostate cancer,
it needs the prostate contoured on MRI whilst avoiding other vital areas like the bladder and rectum.
Over-contouring will irradiate healthy organs and tissue potentially causing cystitis or proctitis.
Under-contouring leaves untreated tumour behind potentially leading to the cancer not regressing completely.
As contouring by hand is slow and tedious, automatic drafting with slight corrections can save time
as long as errors produced are predictable.

This prototype segments six classes in each image (background, body, bone, bladder, rectum, prostate)
in 3D pelvic MRI volumes from the HipMRI study. As the prostate only occupies approximately 0.1-0.3%
of each volume, it is simultaneously the most difficult and most critical class to target. With 
this, the following can be answered:

**Does true 3D volumetric context fix the boundary errors at the prostate base and apex that a 2D 
slice-by-slice model makes, or does it only improve the easy central slices while costing more GPU 
memory and time?**

2. [Algorithm](#2-algorithm)
3. [Feasibility Review](#3-feasibility-review)
4. [Data, Pre-processing and Splits](#4-data-pre-processing-and-splits)
5. [Usage](#5-usage)
6. [Dependencies and Reproducibility](#6-dependencies-and-reproducibility)
7. [Results](#7-results)
8. [Open Research Dilemma: 2D vs 3D Context](#8-open-research-dilemma-2d-vs-3d-context)
9. [Recommendation](#9-recommendation)
10. [Limitations and Next Steps](#10-limitations-and-next-steps)
11. [Artificial Intelligence Usage Disclosure](#11-artificial-intelligence-usage-disclosure)
12. [References](#12-references)

## 1. Problem

## 3. Feasibility Review

## 4. Data, Pre-processing and Splits

## 5. Usage

## 6. Dependencies and Reproducibility

## 7. Results

## 8. Open Research Dilemma: 2D vs 3D Context

## 9. Recommendation

## 10. Limitations and Next Steps

## 11. Artificial Intelligence Usage Disclosure
- **Tool used:** Claude (Anthropic), through Claude Code.
- **Tasks it assisted with:**
  - Planning the 3D work from the spec and my 2D baseline.
  - Initial scaffolding of the 3D `dataset.py`, `modules.py`, `train.py`, `predict.py` and `utils.py`, adapted from my
    own 2D code.
  - Explaining the Slurm setup.
  - Scaffolding the initial README design.
- **How I verified it:**
  - Ran every script on synthetic Nifti volumes before using real data:
    - checked output shapes;
    - checked Dice loss = 0 and Dice = 1 on a perfect prediction;
    - checked the masks stay aligned with the images after flips and rotations;
    - checked that the 144-slice volume is clipped correctly.
  - Checked the architecture against Isensee et al. [2] and the loader against spec Appendix B.
  - On real data:
    - confirmed the patient split has no leakage;
    - confirmed `predict.py` reproduces `train.py`'s test scores exactly;

## 12. References
1. O. Ronneberger, P. Fischer, T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," MICCAI, 2015.
2. F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, K. H. Maier-Hein, "Brain tumor segmentation and radiomics survival prediction: Contribution to the BRATS 2017 challenge," BrainLes (MICCAI), 2018.
3. Ö. Çiçek, A. Abdulkadir, S. S. Lienkamp, T. Brox, O. Ronneberger, "3D U-Net: Learning dense volumetric segmentation from sparse annotation," MICCAI, 2016.
