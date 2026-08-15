#!/usr/bin/env python3
"""Benchmark voxel-domain affine and surface-aware registration on PRIME-DE.

One deterministic MacaSurfer ``Resample/Original`` example is selected per
site.  Inputs under PRIME-DE are read-only; all generated masks, registrations,
metrics, logs, and figures are written below ``--output-dir``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage


HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = Path("/home/weiyahui/projects/monkey/PRIME-DE")
TARGET_VOL = HERE / "resources" / "mebrain_04mm_LIA.nii.gz"
TARGET_RIBBON = HERE / "resources" / "mebrain_04mm_ribbon_aparc_LIA.nii.gz"
TARGET_SURF = HERE / "resources" / "mebrain_combined.surf.gii"
TARGET_CORT = HERE / "resources" / "mebrain_combined.label.gii"
MISSING_SITES = {"site-caltech", "site-neurospin", "site-oxford-PM"}
UTRECHT_SUBJECT = "sub-032241MacacaMulatta"
STAGES = (
    "original",
    "affine_nosurf",
    "nonlinear_nosurf",
    "affine_surf",
    "nonlinear_surf",
)
STAGE_LABELS = (
    "Original (voxel view)",
    "Affine, no surface",
    "Nonlinear, no surface",
    "Affine, with surface",
    "Nonlinear, with surface",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=HERE / "benchmark_prime_original")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--sites", nargs="*", help="Site names; default is every site")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--affine-scales", nargs="+", type=float, default=[4, 2, 1])
    parser.add_argument("--affine-iterations", nargs="+", type=int, default=[400, 300, 200])
    parser.add_argument("--nonlinear-scales", nargs="+", type=int, default=[4, 2, 1])
    parser.add_argument("--nonlinear-iterations", nargs="+", type=int, default=[800, 600, 400])
    parser.add_argument("--nosurf-learning-rate", type=float, default=0.5)
    parser.add_argument("--surf-learning-rate", type=float, default=0.4)
    parser.add_argument("--crop-margin", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()

def foreground_largest_component_ratio(volume: Path) -> float:
    """Fraction of positive voxels in the largest 6-connected component."""
    data = np.asarray(nib.load(str(volume)).dataobj)
    foreground = data > 0
    if not foreground.any():
        return 0.0
    components, count = ndimage.label(foreground)
    sizes = np.bincount(components.ravel())[1 : count + 1]
    return float(sizes.max() / foreground.sum())


def first_complete_example(site_dir: Path) -> dict[str, str] | None:
    volumes = sorted(
        site_dir.glob(
            "output/**/Resample/Original/Volume/"
            "*_space-orig_desc-brain_T1w.nii.gz"
        )
    )
    if site_dir.name == "site-utrecht":
        volumes = [p for p in volumes if UTRECHT_SUBJECT in p.parts]
    for volume in volumes:
        original = volume.parents[1]
        stem = volume.name.removesuffix("_space-orig_desc-brain_T1w.nii.gz")
        surface = original / "fsaverage_LR32k" / f"{stem}_combined.surf.gii"
        cort = original / "fsaverage_LR32k" / f"{stem}_combined.label.gii"
        ribbon = original / "Volume" / f"{stem}_desc-ribbon_dseg.nii.gz"
        if surface.is_file() and cort.is_file() and ribbon.is_file():
            foreground_ratio = foreground_largest_component_ratio(volume)
            if foreground_ratio < 0.99 and len(volumes) > 1:
                continue
            return {
                "site": site_dir.name,
                "example": stem,
                "src_vol": str(volume),
                "src_ribbon": str(ribbon),
                "src_surf": str(surface),
                "src_cort": str(cort),
                "foreground_lcc_ratio": f"{foreground_ratio:.8f}",
            }
    return None


def build_manifest(dataset_root: Path, selected_sites: list[str] | None) -> list[dict[str, str]]:
    requested = set(selected_sites) if selected_sites else None
    manifest: list[dict[str, str]] = []
    for site_dir in sorted(dataset_root.glob("site-*")):
        if requested is not None and site_dir.name not in requested:
            continue
        item = first_complete_example(site_dir)
        if item is None:
            manifest.append(
                {
                    "site": site_dir.name,
                    "status": "missing_complete_original",
                    "example": "",
                    "src_vol": "",
                    "src_ribbon": "",
                    "src_surf": "",
                    "src_cort": "",
                }
            )
        else:
            item["status"] = "ready"
            manifest.append(item)
    if requested:
        discovered = {row["site"] for row in manifest}
        unknown = requested - discovered
        if unknown:
            raise ValueError(f"Unknown or absent site(s): {sorted(unknown)}")
    return manifest


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fieldnames:
        return
    names = fieldnames or list(rows[0])

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_binary_like(source: Path, output: Path) -> None:
    image = nib.load(str(source))
    mask = (np.asarray(image.dataobj) > 0).astype(np.uint8)
    out = nib.Nifti1Image(mask, image.affine, image.header)
    out.set_data_dtype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(output))

def save_evaluation_label(volume_path: Path, ribbon_path: Path, output: Path) -> None:
    volume = nib.load(str(volume_path))
    ribbon = nib.load(str(ribbon_path))
    if volume.shape[:3] != ribbon.shape[:3] or not np.allclose(
        volume.affine, ribbon.affine, atol=1e-5
    ):
        raise ValueError("Volume and ribbon must share a grid")
    label = (np.asarray(volume.dataobj) > 0).astype(np.uint8)
    label[np.asarray(ribbon.dataobj) > 0] = 2
    out = nib.Nifti1Image(label, volume.affine, volume.header)
    out.set_data_dtype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(output))


def inspect_row(row: dict[str, str]) -> dict[str, str | int | float]:
    result: dict[str, str | int | float] = dict(row)
    if row["status"] != "ready":
        return result
    volume = nib.load(row["src_vol"])
    ribbon = nib.load(row["src_ribbon"])
    surface = nib.load(row["src_surf"])
    cort = nib.load(row["src_cort"])
    vertices = np.asarray(surface.agg_data("pointset"))
    labels = np.asarray(cort.agg_data()).reshape(-1)
    result.update(
        {
            "volume_shape": "x".join(str(v) for v in volume.shape[:3]),
            "voxel_sizes_mm": ",".join(f"{v:.4g}" for v in volume.header.get_zooms()[:3]),
            "physical_origin_ras": ",".join(f"{v:.4g}" for v in volume.affine[:3, 3]),
            "ribbon_same_grid": int(
                ribbon.shape[:3] == volume.shape[:3]
                and np.allclose(ribbon.affine, volume.affine, atol=1e-5)
            ),
            "surface_vertices": len(vertices),
            "surface_labels": len(labels),
            "surface_label_match": int(len(vertices) == len(labels)),
        }
    )
    return result


def run_logged(command: list[str], log_path: Path, env: dict[str, str]) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + shlex.join(command) + "\n\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=HERE,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.monotonic() - started
    if completed.returncode:
        raise RuntimeError(
            f"Command failed ({completed.returncode}); inspect {log_path}"
        )
    return elapsed


def add_multi(command: list[str], option: str, values: list[int | float]) -> None:
    command.append(option)
    command.extend(str(v) for v in values)


def registration_outputs(site_dir: Path) -> dict[str, Path]:
    return {
        "affine_nosurf_vol": site_dir / "affine_nosurf_T1w.nii.gz",
        "affine_nosurf_eval": site_dir / "affine_nosurf_eval.nii.gz",
        "nonlinear_nosurf_vol": site_dir / "nonlinear_nosurf_T1w.nii.gz",
        "nonlinear_nosurf_eval": site_dir / "nonlinear_nosurf_eval.nii.gz",
        "affine_surf_vol": site_dir / "affine_surf_T1w.nii.gz",
        "affine_surf_eval": site_dir / "affine_surf_eval.nii.gz",
        "affine_surf_gii": site_dir / "affine_surf.surf.gii",
        "nonlinear_surf_vol": site_dir / "nonlinear_surf_T1w.nii.gz",
        "nonlinear_surf_eval": site_dir / "nonlinear_surf_eval.nii.gz",
        "nonlinear_surf_gii": site_dir / "nonlinear_surf.surf.gii",
    }


def run_site(args: argparse.Namespace, row: dict[str, str], target_mask: Path, target_eval: Path) -> dict[str, float]:
    site_dir = args.output_dir / "sites" / row["site"]
    site_dir.mkdir(parents=True, exist_ok=True)
    source_mask = site_dir / "source_brainmask.nii.gz"
    if args.overwrite or not source_mask.exists():
        save_binary_like(Path(row["src_vol"]), source_mask)
    outputs = registration_outputs(site_dir)
    common_affine = ["--affine_space", "voxel", "--affine_loss", "cc"]
    source_eval = site_dir / "source_evaluation_label.nii.gz"
    if args.overwrite or not source_eval.exists():
        save_evaluation_label(Path(row["src_vol"]), Path(row["src_ribbon"]), source_eval)
    add_multi(common_affine, "--affine_scales", args.affine_scales)
    add_multi(common_affine, "--affine_iterations", args.affine_iterations)
    nonlinear = []
    add_multi(nonlinear, "--scales", args.nonlinear_scales)
    add_multi(nonlinear, "--iterations", args.nonlinear_iterations)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    timings: dict[str, float] = {}

    nosurf_required = [
        outputs["affine_nosurf_vol"],
        outputs["affine_nosurf_eval"],
        outputs["nonlinear_nosurf_vol"],
        outputs["nonlinear_nosurf_eval"],
    ]
    if args.overwrite or not all(path.exists() for path in nosurf_required):
        command = [
            args.python,
            str(HERE / "register_with_nosurf.py"),
            "--src_vol", row["src_vol"],
            "--trg_vol", str(TARGET_VOL),
            "--src_mask", str(source_mask),
            "--trg_mask", str(target_mask),
            "--src_lbl", str(source_eval),
            "--out_affine_vol", str(outputs["affine_nosurf_vol"]),
            "--out_affine_lbl", str(outputs["affine_nosurf_eval"]),
            "--out_vol", str(outputs["nonlinear_nosurf_vol"]),
            "--out_lbl", str(outputs["nonlinear_nosurf_eval"]),
            "--learning_rate", str(args.nosurf_learning_rate),
            "--device", args.device,
            *common_affine,
            *nonlinear,
        ]
        timings["nosurf_seconds"] = run_logged(command, site_dir / "nosurf.log", env)

    surf_required = [
        outputs["affine_surf_vol"],
        outputs["affine_surf_eval"],
        outputs["affine_surf_gii"],
        outputs["nonlinear_surf_vol"],
        outputs["nonlinear_surf_eval"],
        outputs["nonlinear_surf_gii"],
    ]
    if args.overwrite or not all(path.exists() for path in surf_required):
        command = [
            args.python,
            str(HERE / "register_with_surf.py"),
            "--src_vol", row["src_vol"],
            "--trg_vol", str(TARGET_VOL),
            "--src_mask", str(source_mask),
            "--trg_mask", str(target_mask),
            "--crop_margin", str(args.crop_margin),
            "--src_lbl", str(source_eval),
            "--trg_lbl", str(target_eval),
            "--src_surf", row["src_surf"],
            "--trg_surf", str(TARGET_SURF),
            "--src_cort", row["src_cort"],
            "--trg_cort", str(TARGET_CORT),
            "--out_affine_vol", str(outputs["affine_surf_vol"]),
            "--out_affine_lbl", str(outputs["affine_surf_eval"]),
            "--out_affine_surf", str(outputs["affine_surf_gii"]),
            "--out_vol", str(outputs["nonlinear_surf_vol"]),
            "--out_lbl", str(outputs["nonlinear_surf_eval"]),
            "--out_surf", str(outputs["nonlinear_surf_gii"]),
            "--learning_rate", str(args.surf_learning_rate),
            *common_affine,
            *nonlinear,
        ]
        timings["surf_seconds"] = run_logged(command, site_dir / "surf.log", env)
    return timings


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    denom = int(a.sum()) + int(b.sum())
    return float(2 * np.logical_and(a, b).sum() / denom) if denom else math.nan


def surface_distances(a: np.ndarray, b: np.ndarray, sampling: tuple[float, ...]) -> tuple[float, float]:
    a = a.astype(bool)
    b = b.astype(bool)
    if not a.any() or not b.any():
        return math.nan, math.nan
    a_border = np.logical_xor(a, ndimage.binary_erosion(a))
    b_border = np.logical_xor(b, ndimage.binary_erosion(b))
    dt_a = ndimage.distance_transform_edt(~a_border, sampling=sampling)
    dt_b = ndimage.distance_transform_edt(~b_border, sampling=sampling)
    distances = np.concatenate((dt_b[a_border], dt_a[b_border]))
    return float(np.mean(distances)), float(np.percentile(distances, 95))


def masked_ncc(source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    valid = mask.astype(bool) & np.isfinite(source) & np.isfinite(target)
    if valid.sum() < 2:
        return math.nan
    x = source[valid].astype(np.float64)
    y = target[valid].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denom) if denom else math.nan


def load_on_target(path: Path) -> np.ndarray:
    image = nib.load(str(path))
    target = nib.load(str(TARGET_VOL))
    if image.shape[:3] != target.shape[:3] or not np.allclose(image.affine, target.affine, atol=1e-4):
        raise ValueError(f"Output grid mismatch: {path}")
    return np.asarray(image.dataobj)

def resample_normalized_identity(data: np.ndarray, target_shape: tuple[int, ...], order: int) -> np.ndarray:
    source_shape = np.asarray(data.shape[:3], dtype=np.float64)
    output_shape = np.asarray(target_shape[:3], dtype=np.float64)
    scale = source_shape / output_shape
    return ndimage.affine_transform(
        data, matrix=np.diag(scale), offset=0.5 * scale - 0.5,
        output_shape=tuple(int(v) for v in output_shape), order=order,
        mode="constant", cval=0, prefilter=False,
    )


def normalized_identity_surface(row: dict[str, str], target_img) -> np.ndarray:
    source_img = nib.load(row["src_vol"])
    vertices = np.asarray(nib.load(row["src_surf"]).agg_data("pointset"))
    source_ijk = nib.affines.apply_affine(np.linalg.inv(source_img.affine), vertices)
    source_xyz = source_ijk[:, ::-1]
    source_whd = np.asarray(source_img.shape[:3], dtype=np.float64)[::-1]
    target_whd = np.asarray(target_img.shape[:3], dtype=np.float64)[::-1]
    normalized = (2.0 * source_xyz + 1.0) / source_whd - 1.0
    target_xyz = ((normalized + 1.0) * target_whd - 1.0) / 2.0
    return nib.affines.apply_affine(target_img.affine, target_xyz[:, ::-1])


def make_metric_row(row, stage, volume, evaluation, target, target_mask, target_ribbon, spacing):
    mask = evaluation > 0
    warped_ribbon = evaluation == 2
    assd, hd95 = surface_distances(mask, target_mask, spacing)
    return {
        "site": row["site"], "example": row["example"], "space": "Original",
        "stage": stage, "brain_dice": dice(mask, target_mask),
        "brain_assd_mm": assd, "brain_hd95_mm": hd95,
        "masked_ncc": masked_ncc(volume, target, np.logical_and(mask, target_mask)),
        "surface_rmse_mm": math.nan, "surface_median_mm": math.nan,
        "surface_p95_mm": math.nan, "ribbon_dice": dice(warped_ribbon, target_ribbon),

    }

def evaluate_site(row: dict[str, str], output_dir: Path) -> list[dict[str, str | float]]:
    site_dir = output_dir / "sites" / row["site"]
    outputs = registration_outputs(site_dir)
    target_img = nib.load(str(TARGET_VOL))
    target = np.asarray(target_img.dataobj)
    target_mask = target > 0
    target_ribbon = np.asarray(nib.load(str(TARGET_RIBBON)).dataobj) > 0
    spacing = tuple(float(v) for v in target_img.header.get_zooms()[:3])
    stage_files = {
        "affine_nosurf": (outputs["affine_nosurf_vol"], outputs["affine_nosurf_eval"]),
        "nonlinear_nosurf": (outputs["nonlinear_nosurf_vol"], outputs["nonlinear_nosurf_eval"]),
        "affine_surf": (outputs["affine_surf_vol"], outputs["affine_surf_eval"]),
        "nonlinear_surf": (outputs["nonlinear_surf_vol"], outputs["nonlinear_surf_eval"]),
    }
    source = np.asarray(nib.load(row["src_vol"]).dataobj)
    source_evaluation = np.asarray(
        nib.load(site_dir / "source_evaluation_label.nii.gz").dataobj
    )
    identity_volume = resample_normalized_identity(source, target.shape, order=1)
    identity_evaluation = resample_normalized_identity(
        source_evaluation, target.shape, order=0
    )
    metrics: list[dict[str, str | float]] = [
        make_metric_row(
            row, "initial_voxel_identity", identity_volume, identity_evaluation,
            target, target_mask, target_ribbon, spacing,
        )
    ]
    for stage, (volume_path, eval_path) in stage_files.items():
        volume = load_on_target(volume_path)
        evaluation = load_on_target(eval_path)
        metrics.append(
            make_metric_row(
                row, stage, volume, evaluation, target, target_mask,
                target_ribbon, spacing,
            )
        )
    target_vertices = np.asarray(nib.load(str(TARGET_SURF)).agg_data("pointset"))
    identity_vertices = normalized_identity_surface(row, target_img)
    identity_distances = np.linalg.norm(identity_vertices - target_vertices, axis=1)
    metrics[0]["surface_rmse_mm"] = float(np.sqrt(np.mean(identity_distances**2)))
    metrics[0]["surface_median_mm"] = float(np.median(identity_distances))
    metrics[0]["surface_p95_mm"] = float(np.percentile(identity_distances, 95))
    for surface_stage, surface_path in (
        ("affine_surf", outputs["affine_surf_gii"]),
        ("nonlinear_surf", outputs["nonlinear_surf_gii"]),
    ):
        warped_vertices = np.asarray(nib.load(str(surface_path)).agg_data("pointset"))
        if target_vertices.shape != warped_vertices.shape:
            continue
        distances = np.linalg.norm(warped_vertices - target_vertices, axis=1)
        for metric in metrics:
            if metric["stage"] != surface_stage:
                continue
            metric["surface_rmse_mm"] = float(np.sqrt(np.mean(distances**2)))
            metric["surface_median_mm"] = float(np.median(distances))
            metric["surface_p95_mm"] = float(np.percentile(distances, 95))
    make_site_figure(row, site_dir, stage_files, target, target_mask, target_ribbon)
    return metrics


def normalized_slice(data: np.ndarray, axis: int, index: int) -> np.ndarray:
    plane = np.take(data, index, axis=axis).astype(np.float32)
    values = plane[np.isfinite(plane) & (plane != 0)]
    if values.size:
        low, high = np.percentile(values, [1, 99])
        plane = np.clip((plane - low) / max(high - low, 1e-6), 0, 1)
    else:
        plane[:] = 0
    return np.rot90(plane)


def make_site_figure(
    row: dict[str, str],
    site_dir: Path,
    stage_files: dict[str, tuple[Path, Path]],
    target: np.ndarray,
    target_mask: np.ndarray,
    target_ribbon: np.ndarray,
) -> None:
    source = np.asarray(nib.load(row["src_vol"]).dataobj)
    source_ribbon = np.asarray(nib.load(row["src_ribbon"]).dataobj) > 0
    volumes = [source] + [load_on_target(stage_files[s][0]) for s in STAGES[1:]]
    evaluations = [None] + [load_on_target(stage_files[s][1]) for s in STAGES[1:]]
    masks = [source > 0] + [evaluation > 0 for evaluation in evaluations[1:]]
    ribbons = [source_ribbon] + [evaluation == 2 for evaluation in evaluations[1:]]
    center = ndimage.center_of_mass(target_mask)
    indices = [int(round(v)) for v in center]
    fig, axes = plt.subplots(3, 5, figsize=(19, 10), constrained_layout=True)
    for row_axis, axis in enumerate((0, 1, 2)):
        target_index = indices[axis]
        for col, (volume, mask, ribbon, label) in enumerate(
            zip(volumes, masks, ribbons, STAGE_LABELS)
        ):
            index = volume.shape[axis] // 2 if col == 0 else target_index
            src_plane = normalized_slice(volume, axis, index)
            src_mask_plane = np.rot90(np.take(mask, index, axis=axis))
            src_ribbon_plane = np.rot90(np.take(ribbon, index, axis=axis))
            if col == 0:
                rgb = np.repeat(src_plane[..., None], 3, axis=-1)
                axes[row_axis, col].contour(
                    src_mask_plane, levels=[0.5], colors=["#00e5ff"], linewidths=0.7
                )
                axes[row_axis, col].contour(
                    src_ribbon_plane, levels=[0.5], colors=["#2fff57"], linewidths=0.45
                )
            else:
                trg_plane = normalized_slice(target, axis, target_index)
                rgb = np.stack((trg_plane, src_plane, np.zeros_like(trg_plane)), axis=-1)
                target_mask_plane = np.rot90(
                    np.take(target_mask, target_index, axis=axis)
                )
                target_ribbon_plane = np.rot90(
                    np.take(target_ribbon, target_index, axis=axis)
                )
                axes[row_axis, col].contour(
                    target_mask_plane, levels=[0.5], colors=["#ff2bd6"], linewidths=0.65
                )
                axes[row_axis, col].contour(
                    src_mask_plane, levels=[0.5], colors=["#00e5ff"], linewidths=0.65
                )
                axes[row_axis, col].contour(
                    target_ribbon_plane, levels=[0.5], colors=["#ffd43b"], linewidths=0.4
                )
                axes[row_axis, col].contour(
                    src_ribbon_plane, levels=[0.5], colors=["#2fff57"], linewidths=0.4
                )
            axes[row_axis, col].imshow(rgb, origin="lower")
            axes[row_axis, col].axis("off")
            if row_axis == 0:
                axes[row_axis, col].set_title(label, fontsize=10)
    title = "{} · {} · Original input\n".format(row["site"], row["example"])
    fig.suptitle(
        title + "T1 target/source=magenta/cyan; ribbon target/source=yellow/green",
        fontsize=13,
    )
    fig.savefig(site_dir / "comparison.png", dpi=150)
    plt.close(fig)




def make_summary(metrics: list[dict], output_dir: Path) -> None:
    if not metrics:
        return
    stages = ["initial_voxel_identity", *STAGES[1:]]
    sites = sorted({str(row["site"]) for row in metrics})
    values = np.full((len(sites), len(stages)), np.nan)
    lookup = {(str(row["site"]), str(row["stage"])): row for row in metrics}
    for i, site in enumerate(sites):
        for j, stage in enumerate(stages):
            values[i, j] = float(lookup[(site, stage)]["brain_dice"])
    fig, ax = plt.subplots(figsize=(9, max(7, len(sites) * 0.35)), constrained_layout=True)
    image = ax.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(stages)), [s.replace("_", "\n") for s in stages])
    ax.set_yticks(range(len(sites)), sites)
    for i in range(len(sites)):
        for j in range(len(stages)):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if values[i, j] < 0.65 else "black")
    ax.set_title("PRIME-DE Original-space brain-mask Dice")
    fig.colorbar(image, ax=ax, label="Dice")
    fig.savefig(output_dir / "summary_brain_dice.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.dataset_root.resolve(), args.sites)
    inspected = [inspect_row(row) for row in manifest]
    write_csv(args.output_dir / "manifest.csv", inspected)
    print(json.dumps(inspected, indent=2, ensure_ascii=False))
    if args.list_only:
        return
    target_mask = args.output_dir / "target_brainmask.nii.gz"
    if args.overwrite or not target_mask.exists():
        save_binary_like(TARGET_VOL, target_mask)
    target_eval = args.output_dir / "target_evaluation_label.nii.gz"
    if args.overwrite or not target_eval.exists():
        save_evaluation_label(TARGET_VOL, TARGET_RIBBON, target_eval)
    if args.prepare_only:
        return
    ready = [row for row in manifest if row["status"] == "ready"]
    timing_rows: list[dict[str, str | float]] = []
    if not args.evaluate_only:
        for index, row in enumerate(ready, 1):
            print(f"[{index}/{len(ready)}] {row['site']} {row['example']}", flush=True)
            try:
                timings = run_site(args, row, target_mask, target_eval)
                timing_rows.append({"site": row["site"], **timings, "status": "complete"})
            except Exception as error:
                timing_rows.append({"site": row["site"], "status": f"failed: {error}"})
                write_csv(args.output_dir / "timings.csv", timing_rows)
                continue
        write_csv(args.output_dir / "timings.csv", timing_rows)

    all_metrics: list[dict] = []
    for row in ready:
        all_metrics.extend(evaluate_site(row, args.output_dir))
    write_csv(args.output_dir / "metrics.csv", all_metrics)
    make_summary(all_metrics, args.output_dir)


if __name__ == "__main__":
    main()
