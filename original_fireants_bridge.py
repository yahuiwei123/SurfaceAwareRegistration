"""Public bridge that normalizes paths before invoking the isolated backend."""

from pathlib import Path

from original_fireants_bridge_impl import *  # noqa: F401,F403
from original_fireants_bridge_impl import run_original_fireants as _run_impl


def run_original_fireants(**kwargs):
    """Resolve file arguments before the backend changes its working directory."""
    launch_cwd = Path.cwd()

    def resolve(value):
        if value is None:
            return None
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else (launch_cwd / path).resolve())

    for name in (
        "src_vol",
        "trg_vol",
        "src_mask",
        "trg_mask",
        "src_lbl",
        "out_affine_vol",
        "out_affine_lbl",
        "out_vol",
        "out_lbl",
        "out_warp",
        "out_inv_warp",
    ):
        if name in kwargs:
            kwargs[name] = resolve(kwargs[name])
    return _run_impl(**kwargs)
