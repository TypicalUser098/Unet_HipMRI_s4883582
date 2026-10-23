"""
Data loading, patient-level splitting and augmentation for the HipMRI Study 3D volumes.

Each volume is a 3D Nifti file (semantic_MRs) with a matching label volume (semantic_labels_only).
Images are z-score normalised per volume and both are downsampled to a fixed size. Masks are kept as
integer class labels (CrossEntropyLoss takes class indices directly).

There is no predefined 3D split, so patients are split here. The validation and test patients are
taken from the 2D keras_slices_data split, so the 3D model is evaluated on exactly the same patients
as the 2D baseline.
"""
import math
import os
import re

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from torch.utils.data import Dataset


# ==========================================
# Data Paths
# ==========================================
# Root of the HipMRI dataset, set via HIPMRI_ROOT in a .env file next to this script (not committed)
# or as an environment variable (e.g. /home/groups/comp3710/HipMRI_Study_open on Rangpur).
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
HIPMRI_ROOT = os.environ["HIPMRI_ROOT"]
MR_PATH = os.path.join(HIPMRI_ROOT, "semantic_MRs")
SEG_PATH = os.path.join(HIPMRI_ROOT, "semantic_labels_only")

# The 2D split, used only to read which patients are in validation and test
SLICES_PATH = os.path.join(HIPMRI_ROOT, "keras_slices_data")
SLICES_VALIDATE_PATH = os.path.join(SLICES_PATH, "keras_slices_validate")
SLICES_TEST_PATH = os.path.join(SLICES_PATH, "keras_slices_test")

# 0 = Background, 1 = Body, 2 = Bone, 3 = Bladder, 4 = Rectum, 5 = Prostate
CLASS_NAMES = ["Background", "Body", "Bone", "Bladder", "Rectum", "Prostate"]
NUM_CLASSES = len(CLASS_NAMES)
PROSTATE_CLASS = 5

# Volumes are 256x256x128 (one is 256x256x144) with ~1.6 mm near-isotropic voxels. Half resolution
# keeps the prostate (0.1-0.3% of voxels) at ~1-2.5k voxels and every axis divisible by 16.
VOLUME_DEPTH = 128
VOLUME_SIZE = (128, 128, 64)


def load_nifti_volume(path):
    """Load a single 3D Nifti volume as a float32 NumPy array (adapted from spec Appendix B)."""
    volume = nib.load(path).get_fdata(caching="unchanged")
    if volume.ndim == 4:
        # Some HipMRI volumes carry an extra trailing dimension
        volume = volume[:, :, :, 0]
    # Clip extra slices so every volume has the same depth and voxel spacing before resizing
    return volume[:, :, :VOLUME_DEPTH].astype(np.float32)


def pair_key(file_name):
    """Shared key between an image and its mask, e.g. 'B006_Week0_LFOV.nii.gz' -> 'B006_Week0'."""
    return file_name.rsplit("_", 1)[0]


def patient_id(file_name):
    """Patient ID is the first number in the file name ('B006_...' -> '006', 'case_006_...' -> '006')."""
    match = re.search(r"\d+", file_name)
    return match.group() if match else file_name


def is_nifti(file_name):
    return file_name.endswith(".nii") or file_name.endswith(".nii.gz")


# ==========================================
# Patient-level Split
# ==========================================
def slice_folder_patients(slice_dir):
    """Patient IDs present in one of the 2D keras_slices folders."""
    return {patient_id(f) for f in os.listdir(slice_dir) if is_nifti(f)}


def split_volumes(mr_dir=MR_PATH):
    """Split volume file names into train/validate/test by patient, mirroring the 2D split.

    Every weekly scan of a patient goes to the same split. Patients in the 2D validation or test
    folders go to the same 3D split; all other patients are used for training.
    """
    val_patients = slice_folder_patients(SLICES_VALIDATE_PATH)
    test_patients = slice_folder_patients(SLICES_TEST_PATH)

    splits = {"train": [], "validate": [], "test": []}
    for f in sorted(os.listdir(mr_dir)):
        if not is_nifti(f):
            continue
        pid = patient_id(f)
        if pid in test_patients:
            splits["test"].append(f)
        elif pid in val_patients:
            splits["validate"].append(f)
        else:
            splits["train"].append(f)
    return splits


# ==========================================
# 3D Augmentation
# ==========================================
def random_flip(image, mask):
    """Mirror along each spatial axis with probability 0.5 (as in Isensee et al.)."""
    for axis in (1, 2, 3):
        if torch.rand(1).item() < 0.5:
            image = image.flip(axis)
            mask = mask.flip(axis - 1)
    return image, mask


def random_rotate_scale(image, mask, max_angle=10.0, scale_range=(0.9, 1.1)):
    """Small random in-plane rotation and isotropic scaling, applied identically to image and mask."""
    angle = math.radians((torch.rand(1).item() * 2 - 1) * max_angle)
    scale = scale_range[0] + torch.rand(1).item() * (scale_range[1] - scale_range[0])
    cos, sin = math.cos(angle) / scale, math.sin(angle) / scale

    # affine_grid uses (x, y, z) = (W, H, D); the in-plane axes are D and H, so rotate in the (y, z) plane
    theta = torch.tensor([[[1 / scale, 0, 0, 0],
                           [0, cos, -sin, 0],
                           [0, sin, cos, 0]]], dtype=torch.float32)
    grid = F.affine_grid(theta, (1,) + tuple(image.shape), align_corners=False)

    # Border padding for the image (z-scored background is not 0), nearest for the mask
    image = F.grid_sample(image[None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0]
    mask = F.grid_sample(mask[None, None].float(), grid, mode="nearest", padding_mode="zeros",
                         align_corners=False)[0, 0].long()
    return image, mask


def random_intensity(image, scale=0.1, shift=0.1, noise=0.05):
    """Random brightness/contrast change and Gaussian noise (image only)."""
    image = image * (1 + (torch.rand(1).item() * 2 - 1) * scale)
    image = image + (torch.rand(1).item() * 2 - 1) * shift
    return image + torch.randn_like(image) * noise


def augment(image, mask):
    image, mask = random_flip(image, mask)
    image, mask = random_rotate_scale(image, mask)
    image = random_intensity(image)
    return image, mask


# ==========================================
# Dataset
# ==========================================
class HipMRIVolumeDataset(Dataset):
    """Returns (image, mask) pairs of 3D HipMRI volumes: image (1, D, H, W) float, mask (D, H, W) long.

    Volumes are preloaded and downsampled once in __init__, since decompressing the full-size .nii.gz
    files every epoch would dominate training time (~4 MB per downsampled volume).
    """

    def __init__(self, file_names, mr_dir=MR_PATH, seg_dir=SEG_PATH, volume_size=VOLUME_SIZE,
                 augment=False):
        self.file_names = list(file_names)
        self.augment = augment

        # Pair images and masks by name rather than relying on sorted order lining up
        masks = {pair_key(f): f for f in os.listdir(seg_dir) if is_nifti(f)}

        self.images, self.masks = [], []
        for f in self.file_names:
            key = pair_key(f)
            if key not in masks:
                raise FileNotFoundError(f"No mask found for {f} in {seg_dir}")

            # 1. Load Image and z-score normalise per volume
            image = load_nifti_volume(os.path.join(mr_dir, f))
            image = (image - image.mean()) / (image.std() + 1e-8)
            image = torch.from_numpy(image)[None, None]  # (1, 1, D, H, W) for interpolate

            # 2. Load Mask as integer class labels
            mask = np.rint(load_nifti_volume(os.path.join(seg_dir, masks[key]))).astype(np.int64)
            mask = torch.from_numpy(mask)[None, None].float()

            # Trilinear for the image, NEAREST for the mask so class labels are never blended
            image = F.interpolate(image, size=volume_size, mode="trilinear", align_corners=False)[0]
            mask = F.interpolate(mask, size=volume_size, mode="nearest")[0, 0].to(torch.uint8)

            self.images.append(image)
            self.masks.append(mask)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        image, mask = self.images[idx], self.masks[idx].long()
        if self.augment:
            image, mask = augment(image, mask)
        return image, mask

    def name(self, idx):
        return self.file_names[idx]

    def patient_ids(self):
        return {patient_id(f) for f in self.file_names}


def check_patient_leakage(**splits):
    """Print any patient IDs shared between splits. Returns True if the splits are patient-disjoint."""
    ids = {name: ds.patient_ids() for name, ds in splits.items()}
    names = list(ids)
    clean = True
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = ids[names[i]] & ids[names[j]]
            if overlap:
                clean = False
                print(f"WARNING: patients shared between {names[i]} and {names[j]}: {sorted(overlap)}")
    for name in names:
        print(f"{name}: {len(splits[name])} volumes from {len(ids[name])} patients")
    return clean
