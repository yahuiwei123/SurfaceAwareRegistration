from __future__ import annotations

from dataclasses import dataclass

import nibabel as nib
import numpy as np
import torch


RAS_AXCODES = ("R", "A", "S")


@dataclass(frozen=True)
class OrientationContext:
    original: nib.spatialimages.SpatialImage
    working: nib.spatialimages.SpatialImage
    original_to_working: np.ndarray
    working_to_original: np.ndarray
    working_to_original_vox: np.ndarray

    @classmethod
    def create(cls, image, enabled=True):
        original_orientation = nib.orientations.io_orientation(image.affine)
        target_orientation = (
            nib.orientations.axcodes2ornt(RAS_AXCODES)
            if enabled
            else original_orientation
        )
        original_to_working = nib.orientations.ornt_transform(
            original_orientation,
            target_orientation,
        )
        working = image.as_reoriented(original_to_working)
        working_orientation = nib.orientations.io_orientation(working.affine)
        working_to_original = nib.orientations.ornt_transform(
            working_orientation,
            original_orientation,
        )
        working_to_original_vox = nib.orientations.inv_ornt_aff(
            original_to_working,
            image.shape[:3],
        )
        return cls(
            original=image,
            working=working,
            original_to_working=original_to_working,
            working_to_original=working_to_original,
            working_to_original_vox=working_to_original_vox,
        )

    @property
    def original_axcodes(self):
        return nib.aff2axcodes(self.original.affine)

    @property
    def working_axcodes(self):
        return nib.aff2axcodes(self.working.affine)

    def reorient_image(self, image, name="image"):
        if image.shape[:3] != self.original.shape[:3]:
            raise ValueError(
                f"{name} shape {image.shape[:3]} does not match volume shape "
                f"{self.original.shape[:3]}"
            )
        if not np.allclose(image.affine, self.original.affine, atol=1e-4):
            raise ValueError(f"{name} affine does not match its paired volume")
        return image.as_reoriented(self.original_to_working)

    def restore_array(self, data):
        restored = nib.orientations.apply_orientation(
            np.asarray(data),
            self.working_to_original,
        )
        if restored.shape[:3] != self.original.shape[:3]:
            raise ValueError(
                f"Restored shape {restored.shape[:3]} does not match original shape "
                f"{self.original.shape[:3]}"
            )
        return restored


def _normalized_grid_to_vox_numpy(grid, shape):
    depth, height, width = shape
    voxels = np.empty_like(grid, dtype=np.float32)
    voxels[..., 0] = ((grid[..., 2] + 1.0) * depth - 1.0) / 2.0
    voxels[..., 1] = ((grid[..., 1] + 1.0) * height - 1.0) / 2.0
    voxels[..., 2] = ((grid[..., 0] + 1.0) * width - 1.0) / 2.0
    return voxels


def _vox_to_normalized_grid_numpy(voxels, shape):
    depth, height, width = shape
    grid = np.empty_like(voxels, dtype=np.float32)
    grid[..., 0] = (2.0 * voxels[..., 2] + 1.0) / width - 1.0
    grid[..., 1] = (2.0 * voxels[..., 1] + 1.0) / height - 1.0
    grid[..., 2] = (2.0 * voxels[..., 0] + 1.0) / depth - 1.0
    return grid


def sampling_grid_to_original(
    grid,
    reference_orientation,
    moving_orientation,
):
    """Convert a canonical sampling grid to the original reference/moving grids."""
    grid_numpy = grid.detach().cpu().numpy()
    converted_batches = []
    for batch_grid in grid_numpy:
        spatially_restored = reference_orientation.restore_array(batch_grid)
        moving_working_voxels = _normalized_grid_to_vox_numpy(
            spatially_restored,
            moving_orientation.working.shape[:3],
        )
        homogeneous = np.concatenate(
            [
                moving_working_voxels,
                np.ones((*moving_working_voxels.shape[:-1], 1), dtype=np.float32),
            ],
            axis=-1,
        )
        moving_original_voxels = np.einsum(
            "ij,...j->...i",
            moving_orientation.working_to_original_vox,
            homogeneous,
        )[..., :3]
        converted_batches.append(
            _vox_to_normalized_grid_numpy(
                moving_original_voxels,
                moving_orientation.original.shape[:3],
            )
        )
    converted = np.stack(converted_batches, axis=0)
    return torch.from_numpy(converted).to(device=grid.device, dtype=grid.dtype)
