"""Surface-aware volume registration using the bundled unified FireANTs code."""

from __future__ import annotations

import argparse

import register_with_surf_surfacefork_impl as _impl


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src_vol", required=True)
    parser.add_argument("--trg_vol", required=True)
    parser.add_argument("--src_mask", default=None)
    parser.add_argument("--trg_mask", default=None)
    parser.add_argument("--crop_margin", type=int, default=7)
    parser.add_argument(
        "--no_reorient",
        action="store_true",
        help="Disable automatic internal reorientation to RAS",
    )
    parser.add_argument("--src_lbl", default=None)
    parser.add_argument("--trg_lbl", default=None)
    parser.add_argument("--src_surf", required=True)
    parser.add_argument("--trg_surf", required=True)
    parser.add_argument("--src_cort", default=None)
    parser.add_argument("--trg_cort", default=None)

    parser.add_argument("--out_vol", default=None)
    parser.add_argument("--out_lbl", default=None)
    parser.add_argument("--out_surf", default=None)
    parser.add_argument("--out_affine_surf", default=None)
    parser.add_argument("--out_affine_vol", default=None)
    parser.add_argument("--out_affine_lbl", default=None)
    parser.add_argument("--affine_only", action="store_true")
    parser.add_argument("--out_warp", default=None)
    parser.add_argument("--out_inv_warp", default=None)
    parser.add_argument("--warp_format", choices=("fsl", "ants"), default="fsl")

    parser.add_argument("--scales", type=int, nargs="+", default=[4, 2, 1])
    parser.add_argument("--iterations", type=int, nargs="+", default=[800, 600, 400])
    parser.add_argument("--learning_rate", type=float, default=0.4)
    parser.add_argument("--image_weight", type=float, default=6e-5)
    parser.add_argument("--surface_weight", type=float, default=1.0)
    parser.add_argument("--displacement_weight", type=float, default=0.5)
    parser.add_argument("--consistency_weight", type=float, default=1.0)
    parser.add_argument(
        "--convergence_eps",
        type=float,
        nargs="+",
        default=[1e-12, 1e-12, 1e-12],
    )

    parser.add_argument("--affine_scales", type=float, nargs="+", default=[4, 2, 1])
    parser.add_argument(
        "--affine_iterations", type=int, nargs="+", default=[400, 300, 200]
    )
    parser.add_argument("--affine_learning_rate", type=float, default=3e-3)
    parser.add_argument("--affine_loss", choices=("cc", "mse", "mi"), default="cc")
    parser.add_argument("--affine_cc_kernel_size", type=int, default=5)
    parser.add_argument(
        "--affine_max_shear",
        type=float,
        default=0.25,
        help="Maximum absolute coefficient for each of the three affine shear terms",
    )
    parser.set_defaults(nonlinear_only=False)
    return parser


def _validate(args):
    if len(args.scales) != len(args.iterations):
        raise ValueError("--scales and --iterations must have the same length")
    if len(args.affine_scales) != len(args.affine_iterations):
        if len(args.affine_iterations) == 1:
            args.affine_scales = [1]
        else:
            raise ValueError(
                "--affine_scales and --affine_iterations must have the same length"
            )
    if bool(args.src_mask) != bool(args.trg_mask):
        raise ValueError("--src_mask and --trg_mask must be supplied together")
    if bool(args.src_cort) != bool(args.trg_cort):
        raise ValueError("--src_cort and --trg_cort must be supplied together")
    if args.out_affine_lbl and not args.src_lbl:
        raise ValueError("--out_affine_lbl requires --src_lbl")
    if args.out_lbl and not args.src_lbl:
        raise ValueError("--out_lbl requires --src_lbl")
    for name in (
        "image_weight",
        "surface_weight",
        "displacement_weight",
        "consistency_weight",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name} must be non-negative")
    if not 0 <= args.affine_max_shear <= 1:
        raise ValueError(
            "--affine_max_shear must be between 0 and 1"
        )


def main(args):
    _validate(args)
    _impl.main(args)


if __name__ == "__main__":
    main(build_parser().parse_args())
