import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F


def _voxel_grid(shape, device, dtype):
    coords = torch.stack(torch.meshgrid(
        torch.arange(shape[0], device=device, dtype=dtype),
        torch.arange(shape[1], device=device, dtype=dtype),
        torch.arange(shape[2], device=device, dtype=dtype),
        indexing="ij",
    ), dim=-1)
    return coords.unsqueeze(0)


def _normalized_grid_to_vox(grid, shape):
    size = torch.tensor(shape, device=grid.device, dtype=grid.dtype)
    vox = torch.empty_like(grid)
    vox[..., 0] = (grid[..., 2] + 1.0) * size[0] * 0.5 - 0.5
    vox[..., 1] = (grid[..., 1] + 1.0) * size[1] * 0.5 - 0.5
    vox[..., 2] = (grid[..., 0] + 1.0) * size[2] * 0.5 - 0.5
    return vox


def _vox_to_world(vox, affine):
    affine = torch.as_tensor(affine, device=vox.device, dtype=vox.dtype)
    vox_h = torch.cat([vox, torch.ones(*vox.shape[:-1], 1, device=vox.device, dtype=vox.dtype)], dim=-1)
    return torch.einsum("ij,b...j->b...i", affine, vox_h)[..., :3]


def _pixdim(img):
    return np.asarray(img.header["pixdim"][1:4], dtype=np.float32)


def _analyze_scaled(vox, pixdim):
    pixdim = torch.as_tensor(pixdim, device=vox.device, dtype=vox.dtype)
    return vox * pixdim


def _save_vector_nifti(data, reference_img, out_file):
    data = np.asarray(data, dtype=np.float32)
    out = nib.Nifti1Image(data, reference_img.affine)
    out.set_qform(reference_img.affine, code=int(reference_img.header['qform_code']))
    out.set_sform(reference_img.affine, code=int(reference_img.header['sform_code']))
    out.header.set_data_dtype(np.float32)
    nib.save(out, out_file)


def torch2phy_from_nib(img, device=None, dtype=torch.float32):
    shape = img.shape[:3]
    grid = F.affine_grid(
        torch.eye(3, 4, device=device, dtype=dtype).unsqueeze(0),
        [1, 1, *shape],
        align_corners=False,
    )
    vox = _normalized_grid_to_vox(grid, shape)
    phy = _vox_to_world(vox, img.affine)
    basis = []
    origin = _vox_to_world(torch.zeros(1, 1, 1, 1, 3, device=device, dtype=dtype), img.affine)[0, 0, 0, 0]
    for axis in range(3):
        unit = torch.zeros(1, 1, 1, 1, 3, device=device, dtype=dtype)
        unit[..., axis] = 1.0
        basis.append((_vox_to_world(unit, img.affine)[0, 0, 0, 0] - origin))
    mat = torch.eye(4, device=device, dtype=dtype)
    mat[:3, :3] = torch.stack(basis, dim=1)
    mat[:3, 3] = origin
    return mat.unsqueeze(0)


def grid_to_ants_displacement(grid, reference_img, moving_img):
    moving_vox = _normalized_grid_to_vox(grid, moving_img.shape[:3])
    moving_world = _vox_to_world(moving_vox, moving_img.affine)
    reference_vox = _voxel_grid(reference_img.shape[:3], grid.device, grid.dtype)
    reference_world = _vox_to_world(reference_vox, reference_img.affine)
    return (moving_world - reference_world)[0].detach().cpu().numpy().astype(np.float32)


def grid_to_fsl_relative_displacement(grid, reference_img, moving_img):
    moving_vox = _normalized_grid_to_vox(grid, moving_img.shape[:3])
    reference_vox = _voxel_grid(reference_img.shape[:3], grid.device, grid.dtype)
    moving_scaled = _analyze_scaled(moving_vox, _pixdim(moving_img))
    reference_scaled = _analyze_scaled(reference_vox, _pixdim(reference_img))
    return (moving_scaled - reference_scaled)[0].detach().cpu().numpy().astype(np.float32)


def save_ants_displacement(disp, reference_img, out_file):
    import SimpleITK as sitk

    ref_file = reference_img.get_filename()
    if ref_file is None:
        _save_vector_nifti(disp, reference_img, out_file)
        return

    warp = sitk.GetImageFromArray(np.asarray(disp, dtype=np.float32), isVector=True)
    warp.CopyInformation(sitk.ReadImage(ref_file))
    sitk.WriteImage(warp, out_file)


def save_fsl_displacement(disp, reference_img, out_file):
    _save_vector_nifti(disp, reference_img, out_file)
