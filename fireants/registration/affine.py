from fireants.solver.linear import AffineWarpField
import torch
import torch.nn.functional as F
from monai.losses import (
    GlobalMutualInformationLoss,
    LocalNormalizedCrossCorrelationLoss,
)
from tqdm import tqdm


def _normalize_schedule(affine_iters, affine_scales):
    if isinstance(affine_iters, int):
        iterations = [affine_iters]
    else:
        iterations = [int(value) for value in affine_iters]

    if affine_scales is None:
        scales = [1.0] * len(iterations)
    elif isinstance(affine_scales, (int, float)):
        scales = [float(affine_scales)]
    else:
        scales = [float(value) for value in affine_scales]

    if len(scales) != len(iterations):
        raise ValueError("affine_scales and affine_iters must have equal length")
    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError("affine_scales must contain positive values")
    if any(iters < 0 for iters in iterations):
        raise ValueError("affine_iters must contain non-negative values")
    return scales, iterations


def _make_image_loss(loss_type, cc_kernel_size):
    if loss_type == "cc":
        return LocalNormalizedCrossCorrelationLoss(kernel_size=cc_kernel_size)
    if loss_type == "mi":
        return GlobalMutualInformationLoss()
    if loss_type == "mse":
        return F.mse_loss
    raise ValueError(f"Unsupported affine loss: {loss_type}")


def _downsample(image, scale):
    if scale == 1:
        return image
    size = [max(2, int(round(length / scale))) for length in image.shape[2:]]
    return F.interpolate(image, size=size, mode="trilinear", align_corners=False)


def get_affine_transform(
    fixed_image,
    moving_image,
    fixed_surf=None,
    moving_surf=None,
    affine_iters=400,
    affine_scales=None,
    learning_rate=1e-3,
    loss_type="cc",
    cc_kernel_size=11,
    max_shear=0.25,
):
    """Estimate fixed-normalized to moving-normalized sampling coordinates."""
    device = fixed_image.device
    scales, iterations = _normalize_schedule(affine_iters, affine_scales)

    affine_model = AffineWarpField(
        batch_size=fixed_image.shape[0],
        n_dims=3,
        device=device,
        max_shear=max_shear,
    )
    affine_optimizer = torch.optim.AdamW(
        affine_model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    affine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        affine_optimizer,
        T_max=max(sum(iterations), 1),
        eta_min=min(1e-5, learning_rate),
    )
    img_loss_fn = _make_image_loss(loss_type, cc_kernel_size)
    surf_loss_fn = None
    if fixed_surf is not None and moving_surf is not None:
        from fireants.utils.mesh import MeshDeformLoss
        surf_loss_fn = MeshDeformLoss(loss_mode="mse")
    full_fixed_shape = fixed_image.shape[2:]
    full_moving_shape = moving_image.shape[2:]

    for scale, iters in zip(scales, iterations):
        fixed_down = _downsample(fixed_image.detach(), scale)
        moving_down = _downsample(moving_image.detach(), scale)

        if fixed_surf is not None and moving_surf is not None:
            fixed_ratio = torch.as_tensor(
                [p / q for p, q in zip(fixed_down.shape[2:], full_fixed_shape)],
                device=device,
                dtype=fixed_surf.dtype,
            )
            moving_ratio = torch.as_tensor(
                [p / q for p, q in zip(moving_down.shape[2:], full_moving_shape)],
                device=device,
                dtype=moving_surf.dtype,
            )
            fixed_surf_down = fixed_surf.to(device) * fixed_ratio
            moving_surf_down = moving_surf.to(device) * moving_ratio
        else:
            fixed_surf_down = moving_surf_down = None

        best_state = {
            name: value.detach().clone()
            for name, value in affine_model.state_dict().items()
        }
        best_loss = float("inf")
        best_iteration = 0
        last_loss = float("nan")

        pbar = tqdm(range(iters), desc=f"Voxel affine (scale={scale:g})")
        for iteration in pbar:
            moved_image, affine_grid = affine_model(
                moving_down,
                fixed_down.shape,
            )
            img_loss = img_loss_fn(moved_image, fixed_down)
            surf_loss = torch.zeros((), device=device, dtype=img_loss.dtype)
            if moving_surf_down is not None:
                _, surf_loss = surf_loss_fn(
                    fixed_surf_down,
                    moving_surf_down,
                    None,
                    None,
                    affine_grid.permute(0, 4, 1, 2, 3),
                    moving_down.shape[2:],
                )

            loss_affine = 1e-3 * img_loss + surf_loss
            last_loss = loss_affine.detach().item()
            if torch.isfinite(loss_affine).item() and last_loss < best_loss:
                best_loss = last_loss
                best_iteration = iteration
                best_state = {
                    name: value.detach().clone()
                    for name, value in affine_model.state_dict().items()
                }
            pbar.set_postfix(
                total=f"{last_loss:.3e}",
                vol=f"{img_loss.item():.3e}",
                surf=f"{surf_loss.item():.3e}",
                lr=f"{affine_scheduler.get_last_lr()[0]:.1e}",
            )
            affine_optimizer.zero_grad()
            loss_affine.backward()
            affine_optimizer.step()
            affine_scheduler.step()

        affine_model.load_state_dict(best_state)
        # Adam moments correspond to the discarded last state. Reset them at
        # a resolution boundary while preserving the scheduler's current LR.
        affine_optimizer.state.clear()
        shear = (
            affine_model.max_shear * torch.tanh(affine_model.raw_shear)
        ).detach()
        determinant = torch.linalg.det(affine_model.matrix[:, :, :3]).mean().item()
        print(
            f"AFFINE_SCALE_REPORT scale={scale:g} best_iteration={best_iteration} "
            f"best_loss={best_loss:.6e} last_loss={last_loss:.6e} "
            f"max_abs_shear={shear.abs().max().item():.6f} "
            f"mean_det={determinant:.6f}",
            flush=True,
        )

    moved_image, _ = affine_model(moving_image.detach(), fixed_image.shape)
    return affine_model.matrix, moved_image
