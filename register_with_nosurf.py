"""Volume-only registration using the bundled unified FireANTs implementation."""

from __future__ import annotations

import argparse

import register_with_surf_surfacefork_impl as _impl


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src_vol", required=True, help="Source/moving volume")
    parser.add_argument("--trg_vol", required=True, help="Target/fixed volume")
    parser.add_argument("--src_mask", default=None, help="Source brain mask")
    parser.add_argument("--trg_mask", default=None, help="Target brain mask")
    parser.add_argument("--crop_margin", type=int, default=7)
    parser.add_argument(
        "--no_reorient",
        action="store_true",
        help="Disable automatic internal reorientation to RAS",
    )
    parser.add_argument("--src_lbl", default=None, help="Source label map")
    parser.add_argument("--trg_lbl", default=None, help="Target label map")

    parser.add_argument("--out_vol", default=None, help="Nonlinearly warped volume")
    parser.add_argument("--out_lbl", default=None, help="Nonlinearly warped label map")
    parser.add_argument("--out_affine_vol", default=None, help="Affine-only volume")
    parser.add_argument("--out_affine_lbl", default=None, help="Affine-only label map")
    parser.add_argument("--affine_only", action="store_true")
    parser.add_argument(
        "--nonlinear_only",
        action="store_true",
        help="Skip affine estimation and optimize only a residual nonlinear warp",
    )
    parser.add_argument("--out_warp", default=None)
    parser.add_argument("--out_inv_warp", default=None)
    parser.add_argument("--warp_format", choices=("fsl", "ants"), default="fsl")

    parser.add_argument("--scales", type=int, nargs="+", default=[4, 2, 1])
    parser.add_argument("--iterations", type=int, nargs="+", default=[800, 600, 400])
    parser.add_argument("--learning_rate", type=float, default=3e-2)
    parser.add_argument("--image_weight", type=float, default=1e0)
    parser.add_argument("--displacement_weight", type=float, default=2e4)
    parser.add_argument("--consistency_weight", type=float, default=1e4)
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
    parser.set_defaults(
        src_surf=None,
        trg_surf=None,
        src_cort=None,
        trg_cort=None,
        out_surf=None,
        out_affine_surf=None,
        surface_weight=0.0,
    )
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
    if args.affine_only and args.nonlinear_only:
        raise ValueError("--affine_only and --nonlinear_only are mutually exclusive")
    if args.nonlinear_only and (args.out_affine_vol or args.out_affine_lbl):
        raise ValueError("Affine outputs are unavailable with --nonlinear_only")
    if args.out_affine_lbl and not args.src_lbl:
        raise ValueError("--out_affine_lbl requires --src_lbl")
    if args.out_lbl and not args.src_lbl:
        raise ValueError("--out_lbl requires --src_lbl")
    for name in ("image_weight", "displacement_weight", "consistency_weight"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name} must be non-negative")


def main(args):
    _validate(args)
    _impl.main(args)


if __name__ == "__main__":
    main(build_parser().parse_args())
