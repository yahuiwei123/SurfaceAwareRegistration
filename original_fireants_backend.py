#!/usr/bin/env python3
"""Isolated command-line backend for the unmodified sibling FireANTs tree.

This file is executed in a separate Python process because both this project and
the upstream project expose a top-level package named ``fireants``.  Keeping the
upstream imports in this process prevents modules from the two implementations
from being mixed accidentally.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def _bootstrap_upstream(root: str) -> Path:
    root_path = Path(root).expanduser().resolve()
    package_path = root_path / "fireants" / "__init__.py"
    if not package_path.is_file():
        raise FileNotFoundError(
            f"Original FireANTs package not found at {package_path}"
        )

    # The directory containing this script also has a package called fireants.
    # Put upstream first before importing either implementation.
    root_string = str(root_path)
    sys.path = [entry for entry in sys.path if Path(entry or os.curdir).resolve() != root_path]
    sys.path.insert(0, root_string)
    return root_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original_fireants_root", required=True)
    parser.add_argument("--mode", choices=("affine", "full"), required=True)
    parser.add_argument("--src_vol", required=True)
    parser.add_argument("--trg_vol", required=True)
    parser.add_argument("--src_mask")
    parser.add_argument("--trg_mask")
    parser.add_argument("--src_lbl")

    parser.add_argument("--affine_scales", type=float, nargs="+", default=[4, 2, 1])
    parser.add_argument(
        "--affine_iterations", type=int, nargs="+", default=[200, 100, 50]
    )
    parser.add_argument("--affine_learning_rate", type=float, default=3e-3)
    parser.add_argument("--affine_loss", choices=("cc", "mse", "mi"), default="cc")
    parser.add_argument(
        "--affine_space", choices=("voxel", "physical"), default="voxel"
    )
    parser.add_argument("--cc_kernel_size", type=int, default=5)

    parser.add_argument("--scales", type=float, nargs="+", default=[4, 2, 1])
    parser.add_argument("--iterations", type=int, nargs="+", default=[800, 600, 400])
    parser.add_argument("--learning_rate", type=float, default=0.5)
    parser.add_argument("--smooth_warp_sigma", type=float, default=0.5)
    parser.add_argument("--smooth_grad_sigma", type=float, default=1.0)

    parser.add_argument("--matrix_json", required=True)
    parser.add_argument("--out_affine_vol")
    parser.add_argument("--out_affine_lbl")
    parser.add_argument("--affine_only", action="store_true")
    parser.add_argument("--out_vol")
    parser.add_argument("--out_lbl")
    parser.add_argument("--out_warp")
    parser.add_argument("--out_inv_warp")
    parser.add_argument("--warp_format", choices=("fsl", "ants"), default="fsl")
    parser.add_argument("--device", default=None)
    parser.add_argument("--disable_ffo", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if len(args.affine_scales) != len(args.affine_iterations):
        raise ValueError("--affine_scales and --affine_iterations must have equal length")
    if len(args.scales) != len(args.iterations):
        raise ValueError("--scales and --iterations must have equal length")
    if args.src_mask and not args.trg_mask:
        raise ValueError("--src_mask and --trg_mask must be supplied together")
    if args.trg_mask and not args.src_mask:
        raise ValueError("--src_mask and --trg_mask must be supplied together")
    if args.out_affine_lbl and not args.src_lbl:
        raise ValueError("--out_affine_lbl requires --src_lbl")
    if args.out_lbl and not args.src_lbl:
        raise ValueError("--out_lbl requires --src_lbl")


def _write_matrix(path: str, affine_lps) -> None:
    payload = {
        "coordinate_system": "LPS",
        "mapping": "fixed_physical_to_moving_physical",
        "matrix": affine_lps.detach().cpu().double().numpy()[0].tolist(),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _apply_binary_masks(fixed_image, moving_image, fixed_mask_path, moving_mask_path, Image):
    if fixed_mask_path is None:
        return
    fixed_mask = Image.load_file(fixed_mask_path, device=fixed_image.device)
    moving_mask = Image.load_file(moving_mask_path, device=moving_image.device)
    if fixed_mask.array.shape != fixed_image.array.shape:
        raise ValueError(
            f"Fixed mask shape {fixed_mask.array.shape} != image shape {fixed_image.array.shape}"
        )
    if moving_mask.array.shape != moving_image.array.shape:
        raise ValueError(
            f"Moving mask shape {moving_mask.array.shape} != image shape {moving_image.array.shape}"
        )
    fixed_image.array.mul_((fixed_mask.array > 0).to(fixed_image.array.dtype))
    moving_image.array.mul_((moving_mask.array > 0).to(moving_image.array.dtype))


def _save_sampled_tensor(tensor, reference_batch, filename, FakeBatchedImages) -> None:
    output = Path(filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    FakeBatchedImages(tensor, reference_batch).write_image(str(output))


def _warp_label(reg, fixed_batch, moving_batch, label_path, output_path, torch, sitk, fi):
    label_itk = sitk.ReadImage(label_path)
    label_array = sitk.GetArrayFromImage(label_itk).astype("float32")
    label_tensor = torch.from_numpy(label_array)[None, None].to(fixed_batch.device)
    coords = reg.get_warp_parameters(fixed_batch, moving_batch)
    warped = fi(
        label_tensor,
        **coords,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0]
    output_itk = sitk.GetImageFromArray(warped.detach().cpu().numpy())
    output_itk.CopyInformation(fixed_batch.images[0].itk_image)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(output_itk, str(output))


def _save_fsl_grid(reg, fixed_batch, moving_batch, output_path, inverse, torch, sitk):
    if inverse:
        coords = reg.get_inverse_warped_coordinates(fixed_batch, moving_batch)
        reference_batch, sample_batch = moving_batch, fixed_batch
    else:
        coords = reg.get_warped_coordinates(fixed_batch, moving_batch)
        reference_batch, sample_batch = fixed_batch, moving_batch

    # coords is [B, z, y, x, xyz] and upstream consistently uses
    # align_corners=True. FSL relative fields use scaled voxel coordinates.
    z_size, y_size, x_size = sample_batch.shape[2:]
    sample_vox = torch.empty_like(coords)
    sample_vox[..., 0] = (coords[..., 0] + 1.0) * (x_size - 1) / 2.0
    sample_vox[..., 1] = (coords[..., 1] + 1.0) * (y_size - 1) / 2.0
    sample_vox[..., 2] = (coords[..., 2] + 1.0) * (z_size - 1) / 2.0

    rz, ry, rx = reference_batch.shape[2:]
    zz, yy, xx = torch.meshgrid(
        torch.arange(rz, device=coords.device, dtype=coords.dtype),
        torch.arange(ry, device=coords.device, dtype=coords.dtype),
        torch.arange(rx, device=coords.device, dtype=coords.dtype),
        indexing="ij",
    )
    reference_vox = torch.stack((xx, yy, zz), dim=-1)[None]
    sample_spacing = torch.tensor(
        sample_batch.images[0].itk_image.GetSpacing(),
        device=coords.device,
        dtype=coords.dtype,
    )
    reference_spacing = torch.tensor(
        reference_batch.images[0].itk_image.GetSpacing(),
        device=coords.device,
        dtype=coords.dtype,
    )
    displacement = sample_vox * sample_spacing - reference_vox * reference_spacing
    warp_itk = sitk.GetImageFromArray(
        displacement[0].detach().cpu().numpy().astype("float32"), isVector=True
    )
    warp_itk.CopyInformation(reference_batch.images[0].itk_image)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(warp_itk, str(output))


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    _bootstrap_upstream(args.original_fireants_root)

    if args.disable_ffo:
        os.environ["USE_FFO"] = "False"

    import SimpleITK as sitk
    import torch

    from fireants.interpolator import fireants_interpolator as fi
    from fireants.io.image import BatchedImages, FakeBatchedImages, Image
    from fireants.registration.affine import AffineRegistration
    from fireants.registration.greedy import GreedyRegistration

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Original FireANTs backend: {Path(args.original_fireants_root).resolve()}")
    print(f"Device: {device}")

    fixed_image = Image.load_file(args.trg_vol, device=device)
    moving_image = Image.load_file(args.src_vol, device=device)
    _apply_binary_masks(
        fixed_image,
        moving_image,
        args.trg_mask,
        args.src_mask,
        Image,
    )
    fixed_batch = BatchedImages([fixed_image])
    moving_batch = BatchedImages([moving_image])

    # AffineRegistration normally composes the NIfTI header transforms around
    # every sampling operation. In voxel mode both normalized array domains
    # are instead treated as one abstract coordinate system, so identity
    # initialization always has overlap even for disjoint physical domains.
    fixed_torch2lps = fixed_batch.get_torch2phy().detach().clone()
    fixed_lps2torch = fixed_batch.get_phy2torch().detach().clone()
    moving_torch2lps = moving_batch.get_torch2phy().detach().clone()
    moving_lps2torch = moving_batch.get_phy2torch().detach().clone()
    if args.affine_space == "voxel":
        identity = torch.eye(4, device=fixed_batch.device)[None]
        fixed_batch.torch2phy = identity.clone()
        fixed_batch.phy2torch = identity.clone()
        moving_batch.torch2phy = identity.clone()
        moving_batch.phy2torch = identity.clone()
        print("Affine coordinate space: normalized voxel arrays")
    else:
        print("Affine coordinate space: physical LPS")

    affine_reg = AffineRegistration(
        scales=args.affine_scales,
        iterations=args.affine_iterations,
        fixed_images=fixed_batch,
        moving_images=moving_batch,
        loss_type=args.affine_loss,
        optimizer="Adam",
        optimizer_lr=args.affine_learning_rate,
        cc_kernel_size=args.cc_kernel_size,
        progress_bar=True,
    )
    affine_reg.optimize()
    optimized_affine = affine_reg.get_affine_matrix().detach()
    if args.affine_space == "voxel":
        affine_norm = optimized_affine
        affine_lps = moving_torch2lps @ affine_norm @ fixed_lps2torch
        print("Normalized voxel affine (fixed norm -> moving norm):")
        print(affine_norm[0].detach().cpu().numpy())
    else:
        affine_lps = optimized_affine
    print("Physical affine for downstream use (fixed LPS -> moving LPS):")
    print(affine_lps[0].detach().cpu().numpy())

    if args.out_affine_vol:
        affine_moved = affine_reg.evaluate(fixed_batch, moving_batch)
        _save_sampled_tensor(
            affine_moved, fixed_batch, args.out_affine_vol, FakeBatchedImages
        )
    if args.out_affine_lbl:
        _warp_label(
            affine_reg,
            fixed_batch,
            moving_batch,
            args.src_lbl,
            args.out_affine_lbl,
            torch,
            sitk,
            fi,
        )

    # Affine-only sampling above must use the temporary normalized metadata.
    # Restore the true LPS metadata before downstream registration and export.
    fixed_batch.torch2phy = fixed_torch2lps
    fixed_batch.phy2torch = fixed_lps2torch
    moving_batch.torch2phy = moving_torch2lps
    moving_batch.phy2torch = moving_lps2torch
    _write_matrix(args.matrix_json, affine_lps)

    if args.mode == "affine" or args.affine_only:
        return

    reg = GreedyRegistration(
        scales=args.scales,
        iterations=args.iterations,
        fixed_images=fixed_batch,
        moving_images=moving_batch,
        loss_type="cc",
        cc_kernel_size=args.cc_kernel_size,
        deformation_type="compositive",
        optimizer="Adam",
        optimizer_lr=args.learning_rate,
        smooth_warp_sigma=args.smooth_warp_sigma,
        smooth_grad_sigma=args.smooth_grad_sigma,
        init_affine=affine_lps,
        progress_bar=True,
    )
    reg.optimize()

    if args.out_vol:
        moved = reg.evaluate(fixed_batch, moving_batch)
        _save_sampled_tensor(moved, fixed_batch, args.out_vol, FakeBatchedImages)
    if args.out_lbl:
        _warp_label(
            reg,
            fixed_batch,
            moving_batch,
            args.src_lbl,
            args.out_lbl,
            torch,
            sitk,
            fi,
        )

    if args.out_warp:
        Path(args.out_warp).parent.mkdir(parents=True, exist_ok=True)
        if args.warp_format == "ants":
            reg.save_as_ants_transforms(args.out_warp)
        else:
            _save_fsl_grid(
                reg, fixed_batch, moving_batch, args.out_warp, False, torch, sitk
            )
    if args.out_inv_warp:
        Path(args.out_inv_warp).parent.mkdir(parents=True, exist_ok=True)
        if args.warp_format == "ants":
            reg.save_as_ants_transforms(args.out_inv_warp, save_inverse=True)
        else:
            _save_fsl_grid(
                reg, fixed_batch, moving_batch, args.out_inv_warp, True, torch, sitk
            )


if __name__ == "__main__":
    main()
