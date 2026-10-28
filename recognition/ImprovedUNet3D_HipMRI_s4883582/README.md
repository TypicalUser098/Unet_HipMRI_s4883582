# 3D Improved U-Net for Prostate Segmentation on HipMRI

**Student:** s4883582 · **Task:** 3D Improved U-Net [Hard Difficulty] on downsampled HipMRI 3D data ·
**Baseline:** 2D U-Net in [`../UNet2D_HipMRI_s4883582`](../UNet2D_HipMRI_s4883582)

## Contents
1. [Problem](#1-problem)

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

## 2. Algorithm

The model utilised is the 3D Improved U-Net of Isensee et al. [2], implemented from the paper in
PyTorch ([`modules.py`](modules.py)). Like a regular 2D U-Net [1, 3], an encoder repeatedly halves the
input resolution to build up context and a decoder upsamples the resultant image back into full
resolution, with skip connections carrying fine details across. Insensee's changes to the original
model makes it better in theory for small, imbalanced medical datasets.

Firstly, Residual context modules are used to downsample with a stride-2 3x3x3 convolution. Then it
applies a pre-activation residual block (being norm → LeakyReLU → conv → dropout 0.3 → norm → LeakyReLU →
  conv, added to its input). Instance normalisation and LeakyReLU (0.01) with batches of 1-2 volumes
were also used. The decoder upsamples by repeating voxels (avoiding checkerboard artefacts), concatenates
the skip connection, then recombines features with a 3x3x3 and 1x1x1 convolution. Segmentation layers
at 1/4, 1/2, and full resolution are upscaled and summed into the output so gradients reach lower decoder
layers directly.

The loss is cross-entropy plus soft multi-class Dice (the same as the 2D baseline). Dice weighs every class
equally regardless of size, so the tiny prostate is not drowned out by background; cross-entropy keeps
voxel-wise gradients stable early in training. The model has **9.46M parameters**, a third of the 2D
U-Net's 31.0M.

## 3. Feasibility Review

The feasibility review set out the users (imaging specialists planning external-beam radiotherapy), the
dilemma (spatial context vs. clinical boundary utility) and four acceptance criteria. Each criterion is
checked below against the final full-resolution 3D run, with a link to where the evidence is in this README.

| ID | Dimension | Target (from review) | Result | Status | Evidence |
|----|-----------|----------------------|--------|--------|----------|
| AC1 | Task performance | Prostate Dice > 0.7 on the HipMRI test set | Test prostate Dice **0.8556** (IoU 0.7476) | Met | [Results](#7-results) |
| AC2 | Clinical boundary utility | Track base/apex slice accuracy; avoid leakage into critical OARs | End-slice Dice **0.70** vs central **0.885**; base ends worse (0.63) than apex ends (0.77); prostate predicted on 1-11 slices beyond the true extent in 25/30 volumes | Tracked, boundary gap remains | [Dilemma](#8-open-research-dilemma-2d-vs-3d-context) |
| AC3 | Resource footprint | Peak GPU memory <= 16 GB | Training **7.83 GB**, inference **3.36 GB** | Met | [Results](#7-results) |
| AC4 | Runtime / throughput | Inference fast enough for drafting | **79.4 ms** per volume | Met | [Results](#7-results) |

**[Updated] Changes from the original review:**
- **Hardware:** trained on Rangpur rather than a single Colab T4/L4. The 7.83 GB peak would still fit on a
  16 GB T4.
- **Cost estimate:** the review estimated ~16 GB of VRAM and 16-25 hours for 100 epochs of 3D training. The
  actual run used 7.83 GB and ~1.9 hours for 123 epochs (~57 s per epoch), with batch size 1 and mixed
  precision.
- **Loss:** cross-entropy + Dice was used rather than Dice/Focal. The prostate Dice never collapsed to zero,
  so Focal loss was not needed.
- **Same test data:** the review planned to evaluate 2D and 3D on the same test patients. The final 3D split
  is independent of the 2D one, so the comparison is on different test patients (see
  [Limitations](#10-limitations-and-next-steps)).
- **Risks:** the unexpected image risk did occur: one volume has 144 slices instead of 128. It is clipped
  to 128 in `load_nifti_volume` rather than rejected, and any 4D volume is reduced to 3D the same way
  (see [Data](#4-data-pre-processing-and-splits)). The class imbalance risk was handled by the Dice term in
  the loss.

## 4. Data, Pre-processing and Splits
**Data.** HipMRI Study 3D volumes on Rangpur: images in `semantic_MRs`, labels in `semantic_labels_only`
(211 volumes, 38 patients, up to 8 weekly scans per patient). Files are paired by name
(`B006_Week0_LFOV.nii.gz` ↔ `B006_Week0_SEMANTIC.nii.gz`) rather than by sorted order.

**Pre-processing** ([`dataset.py`](dataset.py)):
1. Load with nibabel and drop any trailing 4th dimension (adapted from the spec's Appendix B loader).
2. Clip to the first 128 slices, so the one 144-slice volume has the same depth and spacing as the rest
   (as Appendix B does).
3. Z-score normalise each image volume (raw intensities range from 0 to ~600 and vary between scans).
4. Resize to `VOLUME_SIZE`: trilinear interpolation for images, **nearest-neighbour for masks** so class
   labels are never blended. The final model uses the native 256×256×128; the first experiment used
   128×128×64.
5. Keep masks as integer class indices; the loss one-hot encodes them internally.

All volumes are loaded and resized once at start-up, so the compressed files are not decoded every epoch.

**Augmentation** (training split only, applied identically to image and mask):
- Random left-right flip. Only left-right: the pelvis is roughly symmetric that way, while flipping the
  other axes would put the bladder below the prostate, which never occurs in a real scan.
- Random in-plane rotation (±10°) and isotropic scaling (0.9-1.1×), which also covers the 1.41-1.88 mm
  spread in voxel size between scans.
- Random brightness/contrast change and Gaussian noise (image only).

**Splits.** 
Patients are shuffled with a fixed seed (42) and split 80/10/10. As every weekly scan of a patient 
lands in the same split, splitting volumes would leak test data into the train data jeopardising
future results.`check_patient_leakage` confirms that the splits are disjoint.

| Split | Patients | Volumes |
|-------|----------|---------|
| Train | 30 | 163 |
| Validation | 4 | 18 |
| Test | 4 (J026, M030, R039, S022) | 30 |

The best checkpoint is chosen on validation prostate Dice, and the test split is used only once, 
at the end.

   
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
