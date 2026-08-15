"""Bridge between the surface fork and the sibling original FireANTs tree."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
DEFAULT_ORIGINAL_FIREANTS_ROOT = HERE.parent / "FireANTs"
BACKEND_SCRIPT = HERE / "original_fireants_backend.py"


def _append_option(command, name, value):
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        command.append(name)
        command.extend(str(item) for item in value)
    else:
        command.extend((name, str(value)))


def run_original_fireants(
    *,
    mode,
    src_vol,
    trg_vol,
    original_fireants_root=DEFAULT_ORIGINAL_FIREANTS_ROOT,
    src_mask=None,
    trg_mask=None,
    src_lbl=None,
    affine_scales=(4, 2, 1),
    affine_iterations=(200, 100, 50),
    affine_learning_rate=3e-3,
    affine_loss="cc",
    affine_space="voxel",
    cc_kernel_size=5,
    scales=(4, 2, 1),
    iterations=(800, 600, 400),
    learning_rate=0.5,
    smooth_warp_sigma=0.5,
    smooth_grad_sigma=1.0,
    out_affine_vol=None,
    out_affine_lbl=None,
    affine_only=False,
    out_vol=None,
    out_lbl=None,
    out_warp=None,
    out_inv_warp=None,
    warp_format="fsl",
    device=None,
    disable_ffo=False,
):
    """Run upstream in an isolated process and return its physical LPS affine."""
    root = Path(original_fireants_root).expanduser().resolve()
    if not (root / "fireants" / "__init__.py").is_file():
        raise FileNotFoundError(f"Original FireANTs tree not found: {root}")

    with tempfile.TemporaryDirectory(prefix="original_fireants_") as tmpdir:
        matrix_json = Path(tmpdir) / "affine.json"
        command = [
            sys.executable,
            str(BACKEND_SCRIPT),
            "--original_fireants_root",
            str(root),
            "--mode",
            mode,
            "--src_vol",
            str(src_vol),
            "--trg_vol",
            str(trg_vol),
            "--matrix_json",
            str(matrix_json),
        ]
        options = {
            "--src_mask": src_mask,
            "--trg_mask": trg_mask,
            "--src_lbl": src_lbl,
            "--affine_scales": list(affine_scales),
            "--affine_iterations": list(affine_iterations),
            "--affine_learning_rate": affine_learning_rate,
            "--affine_loss": affine_loss,
            "--affine_space": affine_space,
            "--cc_kernel_size": cc_kernel_size,
            "--scales": list(scales),
            "--iterations": list(iterations),
            "--learning_rate": learning_rate,
            "--smooth_warp_sigma": smooth_warp_sigma,
            "--smooth_grad_sigma": smooth_grad_sigma,
            "--out_affine_vol": out_affine_vol,
            "--out_affine_lbl": out_affine_lbl,
            "--out_vol": out_vol,
            "--out_lbl": out_lbl,
            "--out_warp": out_warp,
            "--out_inv_warp": out_inv_warp,
            "--warp_format": warp_format,
            "--device": device,
        }
        for option, value in options.items():
            _append_option(command, option, value)
        if affine_only:
            command.append("--affine_only")
        if disable_ffo:
            command.append("--disable_ffo")

        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(root)
            if not existing_pythonpath
            else str(root) + os.pathsep + existing_pythonpath
        )
        print("Running isolated original FireANTs backend:")
        print("  " + " ".join(command))
        subprocess.run(command, check=True, cwd=str(root), env=env)

        payload = json.loads(matrix_json.read_text(encoding="utf-8"))
        if payload.get("coordinate_system") != "LPS":
            raise ValueError(f"Unexpected affine coordinate system: {payload}")
        return np.asarray(payload["matrix"], dtype=np.float64)


def _norm_to_nib_voxel_matrix(shape, align_corners):
    """Map grid_sample xyz normalized coordinates to nibabel ijk voxels."""
    i_size, j_size, k_size = (float(value) for value in shape[:3])
    if align_corners:
        i_scale, j_scale, k_scale = (
            (i_size - 1.0) / 2.0,
            (j_size - 1.0) / 2.0,
            (k_size - 1.0) / 2.0,
        )
        i_offset, j_offset, k_offset = i_scale, j_scale, k_scale
    else:
        i_scale, j_scale, k_scale = i_size / 2.0, j_size / 2.0, k_size / 2.0
        i_offset, j_offset, k_offset = (
            (i_size - 1.0) / 2.0,
            (j_size - 1.0) / 2.0,
            (k_size - 1.0) / 2.0,
        )
    return np.asarray(
        [
            [0.0, 0.0, i_scale, i_offset],
            [0.0, j_scale, 0.0, j_offset],
            [k_scale, 0.0, 0.0, k_offset],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def physical_lps_affine_to_sampling_matrix(
    affine_lps,
    fixed_nib,
    moving_nib,
    *,
    align_corners=False,
    device=None,
    dtype=torch.float32,
):
    """Convert fixed-LPS→moving-LPS affine to a grid_sample matrix.

    The returned matrix maps fixed normalized grid coordinates to moving
    normalized grid coordinates, including nibabel axis order and both NIfTI
    header affines.
    """
    lps_ras_flip = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine_ras = lps_ras_flip @ np.asarray(affine_lps) @ lps_ras_flip

    fixed_norm_to_vox = _norm_to_nib_voxel_matrix(
        fixed_nib.shape[:3], align_corners
    )
    moving_norm_to_vox = _norm_to_nib_voxel_matrix(
        moving_nib.shape[:3], align_corners
    )
    sampling = (
        np.linalg.inv(moving_norm_to_vox)
        @ np.linalg.inv(np.asarray(moving_nib.affine, dtype=np.float64))
        @ affine_ras
        @ np.asarray(fixed_nib.affine, dtype=np.float64)
        @ fixed_norm_to_vox
    )
    sampling[3] = np.asarray([0.0, 0.0, 0.0, 1.0])
    return torch.as_tensor(sampling, device=device, dtype=dtype).unsqueeze(0)
