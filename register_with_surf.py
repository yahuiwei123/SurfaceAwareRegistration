import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import nibabel as nib
import copy

from typing import List, Union, Tuple, Callable, Optional

import argparse
from tqdm import tqdm
from monai.losses.ssim_loss import SSIMLoss
from monai.losses.hausdorff_loss import HausdorffDTLoss
from monai.losses import (
    DiceLoss,
    BendingEnergyLoss,
    LocalNormalizedCrossCorrelationLoss,
    GlobalMutualInformationLoss,
    DiffusionLoss,
)

from fireants.neuio.gifti import load_surf_gii, save_surf_gii, load_label_gii
from fireants.neuio.nifti import load_vol_nii, save_vol_nii
from fireants.utils.mesh import MeshDeformLoss, deform_mesh, affine_mesh, vox2phy
from fireants.utils.grid import (
    displacements_to_warps,
    v2img_3d,
    img2v_3d,
    compute_inverse_displacement,
)
from fireants.utils.schedule import build_scheduler
from fireants.utils.loss import MultiLossWrapper, JacobianDeterminantLoss
from fireants.utils.weights import MGDASolver

from fireants.io import Image, BatchedImages
from fireants.registration.affine import get_affine_transform
from fireants.registration.greedy import GreedyRegistration
from fireants.interpolator import fireants_interpolator
from fireants.utils.imageutils import jacobian
from fireants.utils.warp_export import (
    grid_to_ants_displacement,
    grid_to_fsl_relative_displacement,
    save_ants_displacement,
    save_fsl_displacement,
)

ALIGN_CORNERS = False


# ==========================
#   crop / pad helpers
# ==========================
def crop_image_to_mask(img, mask_data, margin=9):
    """Crop image to non-zero region of mask, with margin.

    Return:
        cropped_img: cropped Nifti image
        crop_params: dict recording original shape, affine and crop offset
    """
    orig_shape = img.shape[:3]
    orig_affine = img.affine.copy()
    data = np.asarray(img.dataobj)

    nonzero = np.nonzero(mask_data > 0)
    if len(nonzero[0]) == 0:
        return img, None

    i_min = max(0, int(nonzero[0].min()) - margin)
    i_max = min(int(nonzero[0].max()) + margin, orig_shape[0] - 1)

    j_min = max(0, int(nonzero[1].min()) - margin)
    j_max = min(int(nonzero[1].max()) + margin, orig_shape[1] - 1)

    k_min = max(0, int(nonzero[2].min()) - margin)
    k_max = min(int(nonzero[2].max()) + margin, orig_shape[2] - 1)

    cropped_data = data[
        i_min:i_max + 1,
        j_min:j_max + 1,
        k_min:k_max + 1,
    ]

    new_affine = orig_affine.copy()
    trans = new_affine[:3, :3]
    new_affine[:3, 3] = (
        new_affine[:3, 3]
        + trans @ np.array([i_min, j_min, k_min], dtype=np.float64)
    )

    cropped_img = nib.Nifti1Image(cropped_data.astype(data.dtype), new_affine)

    cropped_img.header.set_qform(
        new_affine,
        code=int(img.header["qform_code"]),
    )
    cropped_img.header.set_sform(
        new_affine,
        code=int(img.header["sform_code"]),
    )

    crop_params = {
        "i_min": i_min,
        "i_max": i_max,
        "j_min": j_min,
        "j_max": j_max,
        "k_min": k_min,
        "k_max": k_max,
        "orig_shape": orig_shape,
        "orig_affine": orig_affine,
    }

    print(
        f"  crop: {orig_shape} -> {cropped_data.shape}  "
        f"offset=({i_min},{j_min},{k_min})"
    )

    return cropped_img, crop_params


def crop_bbox_from_params(crop_params, full_shape):
    """Return crop bbox in D/H/W order: i, j, k."""
    if crop_params is None:
        return (
            0,
            full_shape[0],
            0,
            full_shape[1],
            0,
            full_shape[2],
        )

    return (
        crop_params["i_min"],
        crop_params["i_max"] + 1,
        crop_params["j_min"],
        crop_params["j_max"] + 1,
        crop_params["k_min"],
        crop_params["k_max"] + 1,
    )


def crop_offset_xyz(crop_params, device, dtype):
    """Return crop offset in grid_sample order: x, y, z = k, j, i."""
    if crop_params is None:
        return torch.zeros(3, device=device, dtype=dtype)

    return torch.tensor(
        [
            crop_params["k_min"],  # x / W
            crop_params["j_min"],  # y / H
            crop_params["i_min"],  # z / D
        ],
        device=device,
        dtype=dtype,
    )


# ==========================
#   coordinate conversion
# ==========================
def norm_to_vox(grid, shape, align_corners=False):
    """Convert normalized grid coordinates to voxel coordinates.

    Args:
        grid: [..., 3], order is x, y, z
        shape: (D, H, W)

    Return:
        voxel coordinates in x, y, z order
    """
    D, H, W = shape
    out = torch.empty_like(grid)

    if align_corners:
        out[..., 0] = (grid[..., 0] + 1.0) * (W - 1) / 2.0
        out[..., 1] = (grid[..., 1] + 1.0) * (H - 1) / 2.0
        out[..., 2] = (grid[..., 2] + 1.0) * (D - 1) / 2.0
    else:
        out[..., 0] = ((grid[..., 0] + 1.0) * W - 1.0) / 2.0
        out[..., 1] = ((grid[..., 1] + 1.0) * H - 1.0) / 2.0
        out[..., 2] = ((grid[..., 2] + 1.0) * D - 1.0) / 2.0

    return out


def vox_to_norm(vox, shape, align_corners=False):
    """Convert voxel coordinates to normalized grid coordinates.

    Args:
        vox: [..., 3], order is x, y, z
        shape: (D, H, W)

    Return:
        normalized coordinates in x, y, z order
    """
    D, H, W = shape
    out = torch.empty_like(vox)

    if align_corners:
        out[..., 0] = 2.0 * vox[..., 0] / (W - 1) - 1.0
        out[..., 1] = 2.0 * vox[..., 1] / (H - 1) - 1.0
        out[..., 2] = 2.0 * vox[..., 2] / (D - 1) - 1.0
    else:
        out[..., 0] = (2.0 * vox[..., 0] + 1.0) / W - 1.0
        out[..., 1] = (2.0 * vox[..., 1] + 1.0) / H - 1.0
        out[..., 2] = (2.0 * vox[..., 2] + 1.0) / D - 1.0

    return out


def make_identity_grid(batch_size, spatial_shape, device, dtype, align_corners=False):
    """Create identity sampling grid for a 3D image.

    spatial_shape: (D, H, W)
    return: [B, D, H, W, 3]
    """
    theta = torch.eye(4, device=device, dtype=dtype)[None, :-1, :].repeat(
        batch_size, 1, 1
    )
    size = torch.Size((batch_size, 1, *spatial_shape))
    return F.affine_grid(theta, size, align_corners=align_corners)


def apply_homogeneous_transform_to_grid(grid, mat):
    """Apply [B, 4, 4] homogeneous matrix to normalized grid.

    Args:
        grid: [B, D, H, W, 3]
        mat:  [B, 4, 4]

    Return:
        transformed grid: [B, D, H, W, 3]
    """
    B = grid.shape[0]
    ones = torch.ones(
        *grid.shape[:-1],
        1,
        device=grid.device,
        dtype=grid.dtype,
    )
    homo = torch.cat([grid, ones], dim=-1)

    out = torch.bmm(
        mat,
        homo.view(B, -1, 4).transpose(1, 2),
    ).transpose(1, 2)[..., :3]

    return out.view(*grid.shape)


# ==========================
#   full-size warp builders
# ==========================
def make_full_domain_as_crop_norm_grid(
    full_shape,
    crop_shape,
    crop_params,
    batch_size,
    device,
    dtype,
    align_corners=False,
):
    """
    For every voxel in full image domain, compute its coordinate
    in the cropped image's normalized coordinate system.

    Return:
        grid_crop_norm: [B, D_full, H_full, W_full, 3]
        order: x, y, z
    """
    full_identity_grid = make_identity_grid(
        batch_size=batch_size,
        spatial_shape=full_shape,
        device=device,
        dtype=dtype,
        align_corners=align_corners,
    )

    # full normalized coordinate -> full voxel coordinate
    full_vox = norm_to_vox(
        full_identity_grid,
        full_shape,
        align_corners=align_corners,
    )

    # full voxel coordinate -> crop voxel coordinate
    offset_xyz = crop_offset_xyz(
        crop_params,
        device=device,
        dtype=dtype,
    )

    crop_vox = full_vox - offset_xyz.view(1, 1, 1, 1, 3)

    # crop voxel coordinate -> crop normalized coordinate
    crop_norm = vox_to_norm(
        crop_vox,
        crop_shape,
        align_corners=align_corners,
    )

    return crop_norm


def sample_crop_disp_on_full_domain(
    disp_crop,
    full_domain_as_crop_norm_grid,
    align_corners=False,
):
    """
    Sample cropped displacement field on full image domain.

    disp_crop:
        [B, D_crop, H_crop, W_crop, 3]

    full_domain_as_crop_norm_grid:
        [B, D_full, H_full, W_full, 3]

    Important:
        padding_mode='border' means crop-outside region uses the nearest
        boundary displacement, instead of becoming identity.
    """
    disp_ch = disp_crop.permute(0, 4, 1, 2, 3)

    disp_full_ch = F.grid_sample(
        disp_ch,
        full_domain_as_crop_norm_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=align_corners,
    )

    disp_full = disp_full_ch.permute(0, 2, 3, 4, 1)

    return disp_full


def build_full_forward_warp_grid(
    fwd_disp_crop,
    init_affine,
    fixed_crop_shape,
    moving_crop_shape,
    fixed_full_shape,
    moving_full_shape,
    fixed_crop_params,
    moving_crop_params,
    align_corners=False,
):
    """
    Build full-size forward warp grid from cropped registration.

    Meaning:
        output domain: fixed full image
        sampling domain: moving full image

    This version applies the crop-estimated affine to the whole full domain,
    and extends nonlinear displacement outside crop by border padding.
    """
    device = fwd_disp_crop.device
    dtype = fwd_disp_crop.dtype
    B = fwd_disp_crop.shape[0]

    # 1. full fixed domain -> fixed crop normalized coordinate
    fixed_full_as_crop_norm = make_full_domain_as_crop_norm_grid(
        full_shape=fixed_full_shape,
        crop_shape=fixed_crop_shape,
        crop_params=fixed_crop_params,
        batch_size=B,
        device=device,
        dtype=dtype,
        align_corners=align_corners,
    )

    # 2. apply cropped affine on the whole full domain
    # fixed crop normalized -> moving crop normalized
    moving_crop_affine_norm = apply_homogeneous_transform_to_grid(
        fixed_full_as_crop_norm,
        init_affine,
    )

    # 3. extend cropped nonlinear displacement to full fixed domain
    # displacement is still in moving crop normalized units
    fwd_disp_full_crop_norm = sample_crop_disp_on_full_domain(
        fwd_disp_crop,
        fixed_full_as_crop_norm,
        align_corners=align_corners,
    )

    # 4. affine + nonlinear residual in moving crop normalized space
    moving_crop_norm = moving_crop_affine_norm + fwd_disp_full_crop_norm

    # 5. moving crop normalized -> moving crop voxel
    moving_crop_vox = norm_to_vox(
        moving_crop_norm,
        moving_crop_shape,
        align_corners=align_corners,
    )

    # 6. moving crop voxel -> moving full voxel
    moving_offset_xyz = crop_offset_xyz(
        moving_crop_params,
        device=device,
        dtype=dtype,
    )

    moving_full_vox = moving_crop_vox + moving_offset_xyz.view(1, 1, 1, 1, 3)

    # 7. moving full voxel -> moving full normalized grid
    fwd_full_grid = vox_to_norm(
        moving_full_vox,
        moving_full_shape,
        align_corners=align_corners,
    )

    return fwd_full_grid


def build_full_reverse_warp_grid(
    rev_disp_crop,
    inv_init_affine,
    moving_crop_shape,
    fixed_crop_shape,
    moving_full_shape,
    fixed_full_shape,
    moving_crop_params,
    fixed_crop_params,
    align_corners=False,
):
    """
    Build full-size reverse warp grid from cropped registration.

    Meaning:
        output domain: moving full image
        sampling domain: fixed full image

    This follows the original composition:
        reverse nonlinear displacement first,
        then inverse affine.
    """
    device = rev_disp_crop.device
    dtype = rev_disp_crop.dtype
    B = rev_disp_crop.shape[0]

    # 1. full moving domain -> moving crop normalized coordinate
    moving_full_as_crop_norm = make_full_domain_as_crop_norm_grid(
        full_shape=moving_full_shape,
        crop_shape=moving_crop_shape,
        crop_params=moving_crop_params,
        batch_size=B,
        device=device,
        dtype=dtype,
        align_corners=align_corners,
    )

    # 2. extend reverse cropped nonlinear displacement to full moving domain
    rev_disp_full_crop_norm = sample_crop_disp_on_full_domain(
        rev_disp_crop,
        moving_full_as_crop_norm,
        align_corners=align_corners,
    )

    # 3. original reverse logic:
    # moving crop normalized + reverse displacement
    moving_crop_deformed_norm = moving_full_as_crop_norm + rev_disp_full_crop_norm

    # 4. apply inverse affine:
    # moving crop normalized -> fixed crop normalized
    fixed_crop_norm = apply_homogeneous_transform_to_grid(
        moving_crop_deformed_norm,
        inv_init_affine,
    )

    # 5. fixed crop normalized -> fixed crop voxel
    fixed_crop_vox = norm_to_vox(
        fixed_crop_norm,
        fixed_crop_shape,
        align_corners=align_corners,
    )

    # 6. fixed crop voxel -> fixed full voxel
    fixed_offset_xyz = crop_offset_xyz(
        fixed_crop_params,
        device=device,
        dtype=dtype,
    )

    fixed_full_vox = fixed_crop_vox + fixed_offset_xyz.view(1, 1, 1, 1, 3)

    # 7. fixed full voxel -> fixed full normalized grid
    rev_full_grid = vox_to_norm(
        fixed_full_vox,
        fixed_full_shape,
        align_corners=align_corners,
    )

    return rev_full_grid


# ==========================
#   main
# ==========================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###################
    # Read volume data
    ###################
    print("Loading images...")

    src_vol_orig = nib.load(args.src_vol)
    trg_vol_orig = nib.load(args.trg_vol)

    mask_margin = getattr(args, "crop_margin", 9)

    if args.src_mask:
        src_mask_data = nib.load(args.src_mask).get_fdata()
        src_vol, src_crop = crop_image_to_mask(
            src_vol_orig,
            src_mask_data,
            margin=mask_margin,
        )
    else:
        src_mask_data = None
        src_vol, src_crop = src_vol_orig, None

    if args.trg_mask:
        trg_mask_data = nib.load(args.trg_mask).get_fdata()
        trg_vol, trg_crop = crop_image_to_mask(
            trg_vol_orig,
            trg_mask_data,
            margin=mask_margin,
        )
    else:
        trg_mask_data = None
        trg_vol, trg_crop = trg_vol_orig, None

    src_affine = src_vol.affine
    trg_affine = trg_vol.affine

    ###################
    # Normalize cropped volumes
    ###################
    src_vol_data = src_vol.get_fdata()
    trg_vol_data = trg_vol.get_fdata()

    src_vol_mask = src_vol_data > 0
    trg_vol_mask = trg_vol_data > 0

    src_mean = src_vol_data[src_vol_mask].mean()
    src_std = max(src_vol_data[src_vol_mask].std(), 1e-6)
    src_vol_data = (src_vol_data - src_mean) / src_std

    trg_mean = trg_vol_data[trg_vol_mask].mean()
    trg_std = max(trg_vol_data[trg_vol_mask].std(), 1e-6)
    trg_vol_data = (trg_vol_data - trg_mean) / trg_std

    moving_image = torch.from_numpy(src_vol_data).float()[None][None].to(device)
    fixed_image = torch.from_numpy(trg_vol_data).float()[None][None].to(device)

    ###################
    # Read label data
    ###################
    if args.src_lbl:
        src_lbl_img = nib.load(args.src_lbl)
        trg_lbl_img = nib.load(args.trg_lbl)

        if src_crop is not None:
            src_lbl_img, _ = crop_image_to_mask(
                src_lbl_img,
                src_mask_data > 0,
                margin=mask_margin,
            )

        if trg_crop is not None:
            trg_lbl_img, _ = crop_image_to_mask(
                trg_lbl_img,
                trg_mask_data > 0,
                margin=mask_margin,
            )

        moving_label = torch.from_numpy(src_lbl_img.get_fdata()).long()[None][None].to(device)
        fixed_label = torch.from_numpy(trg_lbl_img.get_fdata()).long()[None][None].to(device)
    else:
        moving_label = None
        fixed_label = None

    ####################
    # Read surface data
    ####################
    if args.src_surf:
        src_surf = load_surf_gii(args.src_surf, src_affine)
        trg_surf = load_surf_gii(args.trg_surf, trg_affine)

        fixed_surf = trg_surf.verts_packed()[None].to(device)
        moving_surf = src_surf.verts_packed()[None].to(device)
    else:
        src_surf = None
        trg_surf = None
        fixed_surf = None
        moving_surf = None

    ####################
    # Read surface ROI
    ####################
    if args.src_cort:
        fixed_roi = load_label_gii(args.trg_cort)[None].to(device)
        moving_roi = load_label_gii(args.src_cort)[None].to(device)
    else:
        fixed_roi = None
        moving_roi = None

    ###################
    # Register
    ###################
    ncc_img_loss_fn = LocalNormalizedCrossCorrelationLoss(kernel_size=5)
    mesh_loss_fn = MeshDeformLoss(loss_mode="mse")
    reg_loss_fns = DiffusionLoss(normalize=True, reduction="mean")

    init_affine, _ = get_affine_transform(
        fixed_image,
        moving_image,
        fixed_surf,
        moving_surf,
    )

    row = torch.zeros(
        init_affine.shape[0],
        1,
        4,
        device=init_affine.device,
        dtype=init_affine.dtype,
    )
    row[:, 0, -1] = 1.0

    init_affine = torch.cat([init_affine.detach(), row], dim=1)
    inv_init_affine = torch.linalg.inv(init_affine)

    reg = GreedyRegistration(
        scales=args.scales,
        iterations=args.iterations,
        fixed_images=fixed_image,
        moving_images=moving_image,
        img_loss_fn=ncc_img_loss_fn,
        optimizer="Adam",
        optimizer_lr=args.learning_rate,
        init_affine=init_affine,
        fixed_surfs=fixed_surf,
        moving_surfs=moving_surf,
        fixed_roi=fixed_roi,
        moving_roi=moving_roi,
        surf_loss_func=mesh_loss_fn,
        displacement_reg=reg_loss_fns,
    )

    reg.optimize()

    # cropped displacement fields
    # shape: [B, D, H, W, 3]
    fwd_disp_crop = reg.fwd_warp.warp.data
    rev_disp_crop = reg.rev_warp.warp.data

    ###################
    # Build warp grids
    ###################
    has_crop = (trg_crop is not None) or (src_crop is not None)

    if has_crop:
        print("Building full-size warp grid from cropped warp grid...")

        fixed_crop_shape = trg_vol.shape[:3]
        moving_crop_shape = src_vol.shape[:3]

        fixed_full_shape = trg_vol_orig.shape[:3]
        moving_full_shape = src_vol_orig.shape[:3]

        fwd_warp_grid = build_full_forward_warp_grid(
            fwd_disp_crop=fwd_disp_crop,
            init_affine=init_affine,
            fixed_crop_shape=fixed_crop_shape,
            moving_crop_shape=moving_crop_shape,
            fixed_full_shape=fixed_full_shape,
            moving_full_shape=moving_full_shape,
            fixed_crop_params=trg_crop,
            moving_crop_params=src_crop,
            align_corners=ALIGN_CORNERS,
        )

        rev_warp_grid = build_full_reverse_warp_grid(
            rev_disp_crop=rev_disp_crop,
            inv_init_affine=inv_init_affine,
            moving_crop_shape=moving_crop_shape,
            fixed_crop_shape=fixed_crop_shape,
            moving_full_shape=moving_full_shape,
            fixed_full_shape=fixed_full_shape,
            moving_crop_params=src_crop,
            fixed_crop_params=trg_crop,
            align_corners=ALIGN_CORNERS,
        )

        # reload and normalize full-size images
        _src_data = np.asarray(src_vol_orig.dataobj, dtype=np.float32)
        _trg_data = np.asarray(trg_vol_orig.dataobj, dtype=np.float32)

        _src_mask = _src_data > 0
        _trg_mask = _trg_data > 0

        _src_mean = _src_data[_src_mask].mean()
        _src_std = max(_src_data[_src_mask].std(), 1e-6)

        _trg_mean = _trg_data[_trg_mask].mean()
        _trg_std = max(_trg_data[_trg_mask].std(), 1e-6)

        _src_data_norm = (_src_data - _src_mean) / _src_std
        _trg_data_norm = (_trg_data - _trg_mean) / _trg_std

        moving_image = torch.from_numpy(_src_data_norm).float()[None][None].to(device)
        fixed_image = torch.from_numpy(_trg_data_norm).float()[None][None].to(device)

        # full-size label
        if args.src_lbl:
            moving_label = torch.from_numpy(
                nib.load(args.src_lbl).get_fdata()
            ).long()[None][None].to(device)

        # surface loaded with cropped affine is in cropped voxel space.
        # for full-size deformation, shift it back to original voxel space.
        if args.src_surf and src_crop is not None:
            shift_ijk = torch.tensor(
                [
                    src_crop["i_min"],
                    src_crop["j_min"],
                    src_crop["k_min"],
                ],
                device=moving_surf.device,
                dtype=moving_surf.dtype,
            )
            moving_surf_full = moving_surf + shift_ijk.view(1, 1, 3)
        else:
            moving_surf_full = moving_surf if args.src_surf else None

        _out_aff = trg_vol_orig.affine
        _out_ref = trg_vol_orig
        _out_mov = src_vol_orig

        _inv_ref = src_vol_orig
        _inv_mov = trg_vol_orig

    else:
        print("No crop. Building warp grid directly on original image size...")

        fwd_warp_grid = F.affine_grid(
            init_affine[:, :-1, :],
            fixed_image.size(),
            align_corners=ALIGN_CORNERS,
        ) + fwd_disp_crop

        rev_warp_grid = make_identity_grid(
            batch_size=rev_disp_crop.shape[0],
            spatial_shape=moving_image.shape[2:],
            device=moving_image.device,
            dtype=rev_disp_crop.dtype,
            align_corners=ALIGN_CORNERS,
        ) + rev_disp_crop

        rev_warp_grid = apply_homogeneous_transform_to_grid(
            rev_warp_grid,
            inv_init_affine,
        )

        _src_mean = src_mean
        _src_std = src_std
        _trg_mean = trg_mean
        _trg_std = trg_std

        moving_surf_full = moving_surf if args.src_surf else None

        _out_aff = trg_affine
        _out_ref = trg_vol
        _out_mov = src_vol

        _inv_ref = src_vol
        _inv_mov = trg_vol

    ###################
    # Warp full-size moving image
    ###################
    moving_image_warped = F.grid_sample(
        moving_image,
        fwd_warp_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=ALIGN_CORNERS,
    ) * _src_std + _src_mean

    if args.out_vol:
        save_vol_nii(
            moving_image_warped[0, 0, ...].detach().cpu().numpy(),
            _out_aff,
            args.out_vol,
        )

    ###################
    # Export forward warp
    ###################
    if args.out_warp:
        if args.warp_format == "ants":
            disp = grid_to_ants_displacement(
                fwd_warp_grid,
                _out_ref,
                _out_mov,
            )
            save_ants_displacement(disp, _out_ref, args.out_warp)
        else:
            disp = grid_to_fsl_relative_displacement(
                fwd_warp_grid,
                _out_ref,
                _out_mov,
            )
            save_fsl_displacement(disp, _out_ref, args.out_warp)

    ###################
    # Export inverse warp
    ###################
    if args.out_inv_warp:
        if args.warp_format == "ants":
            disp = grid_to_ants_displacement(
                rev_warp_grid,
                _inv_ref,
                _inv_mov,
            )
            save_ants_displacement(disp, _inv_ref, args.out_inv_warp)
        else:
            disp = grid_to_fsl_relative_displacement(
                rev_warp_grid,
                _inv_ref,
                _inv_mov,
            )
            save_fsl_displacement(disp, _inv_ref, args.out_inv_warp)

    ###################
    # Warp label
    ###################
    if args.src_lbl and args.out_lbl:
        moving_label = moving_label.to(device).squeeze(1).long()
        moving_label = torch.where(moving_label == -1, 0, moving_label)

        num_classes = int(moving_label.max().item()) + 1

        moving_label_oh = F.one_hot(
            moving_label,
            num_classes=num_classes,
        ).permute(0, 4, 1, 2, 3).float()

        moved_label = F.grid_sample(
            moving_label_oh,
            fwd_warp_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=ALIGN_CORNERS,
        ).argmax(dim=1)[0, ...].detach().cpu().numpy().astype(np.uint8)

        save_vol_nii(
            moved_label,
            _out_aff,
            args.out_lbl,
        )

    ###################
    # Deform surface
    ###################
    if args.src_surf and args.out_surf:
        deformed_surf = deform_mesh(
            moving_surf_full,
            rev_warp_grid.permute(0, 4, 1, 2, 3),
            fixed_image.shape[2:],
        )

        deformed_surf = vox2phy(
            deformed_surf,
            torch.tensor(_out_aff, device=deformed_surf.device)[None],
        )[0].detach().cpu().numpy()

        save_surf_gii(
            deformed_surf,
            src_surf.faces_packed(),
            args.out_surf,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image Registration with crop-to-full warp reconstruction"
    )

    # Volume data arguments
    parser.add_argument(
        "--src_vol",
        type=str,
        required=True,
        help="Source volume file name",
    )
    parser.add_argument(
        "--trg_vol",
        type=str,
        required=True,
        help="Target volume file name",
    )

    # Optional brain masks for internal crop-then-full-warp
    parser.add_argument(
        "--src_mask",
        type=str,
        default=None,
        help="Source brain mask",
    )
    parser.add_argument(
        "--trg_mask",
        type=str,
        default=None,
        help="Target brain mask",
    )
    parser.add_argument(
        "--crop_margin",
        type=int,
        default=7,
        help="Voxel margin around mask when cropping",
    )

    # Label data arguments
    parser.add_argument(
        "--src_lbl",
        type=str,
        default=None,
        help="Source label file name",
    )
    parser.add_argument(
        "--trg_lbl",
        type=str,
        default=None,
        help="Target label file name",
    )

    # Surface data arguments
    parser.add_argument(
        "--src_surf",
        type=str,
        default=None,
        help="Source surface file name",
    )
    parser.add_argument(
        "--trg_surf",
        type=str,
        default=None,
        help="Target surface file name",
    )

    # Surface cortical region arguments
    parser.add_argument(
        "--src_cort",
        type=str,
        default=None,
        help="Source surface ROI file name",
    )
    parser.add_argument(
        "--trg_cort",
        type=str,
        default=None,
        help="Target surface ROI file name",
    )

    # Output data arguments
    parser.add_argument(
        "--out_vol",
        type=str,
        default=None,
        help="Output warped volume file name",
    )
    parser.add_argument(
        "--out_lbl",
        type=str,
        default=None,
        help="Output warped label file name",
    )
    parser.add_argument(
        "--out_surf",
        type=str,
        default=None,
        help="Output warped surface file name",
    )
    parser.add_argument(
        "--out_warp",
        type=str,
        default=None,
        help="Forward warp file name",
    )
    parser.add_argument(
        "--out_inv_warp",
        type=str,
        default=None,
        help="Inverse warp file name",
    )
    parser.add_argument(
        "--warp_format",
        choices=["fsl", "ants"],
        default="fsl",
        help="Warp output format",
    )

    # Registration configuration
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[4, 2, 1],
        help="Downsample scales",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=[800, 600, 400],
        help="Number of iterations per scale",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.4,
        help="Learning rate",
    )
    parser.add_argument(
        "--convergence_eps",
        type=float,
        nargs="+",
        default=[1e-12, 1e-12, 1e-12],
        help="Convergence epsilon",
    )

    args = parser.parse_args()
    main(args)