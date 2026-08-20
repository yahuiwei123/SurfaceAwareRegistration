#!/usr/bin/env python3
"""Validate volume-only registration on one deterministic T1 per PRIME-DE site.

Only processed brain-extracted T1 images are selected. Sites without a usable
anatomical image are retained in the manifest as unavailable. Registration
outputs, logs, metrics, Jacobian maps, and figures are written below the output
directory; PRIME-DE itself is never modified.
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
DATASET = Path("/home/weiyahui/projects/monkey/PRIME-DE")
TARGET = HERE / "resources" / "mebrain_04mm_LIA.nii.gz"
UTRECHT_SUBJECT = "sub-032241MacacaMulatta"
DEFAULT_AFFINE_ITERATIONS = [400, 300, 200]
DEFAULT_NONLINEAR_ITERATIONS = [800, 600, 400]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET)
    parser.add_argument("--output-dir", type=Path, default=HERE / "validation_prime_nosurf")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--sites", nargs="*")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--affine-iterations", nargs="+", type=int, default=DEFAULT_AFFINE_ITERATIONS)
    parser.add_argument("--iterations", nargs="+", type=int, default=DEFAULT_NONLINEAR_ITERATIONS)
    parser.add_argument("--scales", nargs="+", type=int, default=[4, 2, 1])
    parser.add_argument("--affine-scales", nargs="+", type=float, default=[4, 2, 1])
    parser.add_argument("--affine-max-shear", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--smooth-warp-sigma", type=float, default=0.5)
    parser.add_argument("--smooth-grad-sigma", type=float, default=1.0)
    parser.add_argument("--displacement-weight", type=float, default=1e4)
    parser.add_argument("--consistency-weight", type=float, default=1e4)
    parser.add_argument("--gradient-report-interval", type=int, default=50)
    parser.add_argument("--crop-margin", type=int, default=7)
    parser.add_argument("--cuda-visible-devices", default="0")
    args = parser.parse_args()
    if len(args.scales) != len(args.iterations):
        parser.error("--scales and --iterations must have equal length")
    if len(args.affine_scales) != len(args.affine_iterations):
        parser.error("--affine-scales and --affine-iterations must have equal length")
    if not 0 <= args.affine_max_shear <= 1:
        parser.error("--affine-max-shear must be between 0 and 1")
    return args


def foreground_lcc_ratio(path: Path) -> float:
    foreground = np.asarray(nib.load(str(path)).dataobj) > 0
    if not foreground.any():
        return 0.0
    components, count = ndimage.label(foreground)
    sizes = np.bincount(components.ravel())[1:count + 1]
    return float(sizes.max() / foreground.sum())


def candidates(site: Path) -> list[Path]:
    patterns = (
        "output/*/*/Enhance/T1w/*_res-04mm_desc-brain_T1w.nii.gz",
        "output/*/Enhance/T1w/*_res-04mm_desc-brain_T1w.nii.gz",
        "output/*/*/Resample/Original/Volume/*_space-orig_desc-brain_T1w.nii.gz",
        "output/*/Resample/Original/Volume/*_space-orig_desc-brain_T1w.nii.gz",
    )
    paths = []
    for pattern in patterns:
        paths = sorted(site.glob(pattern))
        if paths:
            break
    if site.name == "site-utrecht":
        rhesus = [path for path in paths if UTRECHT_SUBJECT in path.parts]
        if rhesus:
            paths = rhesus
    return paths


def build_manifest(root: Path, selected: list[str] | None):
    requested = set(selected or [])
    rows = []
    site_dirs = [p for p in sorted(root.glob("site-*")) if p.is_dir() and p.suffix != ".zip"]
    for site in site_dirs:
        if requested and site.name not in requested:
            continue
        paths = candidates(site)
        chosen = None
        ratio = math.nan
        for path in paths:
            ratio = foreground_lcc_ratio(path)
            if ratio >= 0.99 or len(paths) == 1:
                chosen = path
                break
        if chosen is None:
            rows.append({"site": site.name, "status": "unavailable_no_processed_brain_T1", "example": "", "src_vol": "", "foreground_lcc_ratio": ""})
        else:
            rows.append({"site": site.name, "status": "ready", "example": chosen.name.removesuffix(".nii.gz"), "src_vol": str(chosen), "foreground_lcc_ratio": f"{ratio:.8f}"})
    return rows


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_mask(source: Path, output: Path):
    image = nib.load(str(source))
    mask = (np.asarray(image.dataobj) > 0).astype(np.uint8)
    result = nib.Nifti1Image(mask, image.affine, image.header)
    result.set_data_dtype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(result, str(output))


def outputs(site_dir: Path):
    return {
        "source_mask": site_dir / "source_mask.nii.gz",
        "affine": site_dir / "affine_T1w.nii.gz",
        "affine_mask": site_dir / "affine_mask.nii.gz",
        "nonlinear": site_dir / "nonlinear_T1w.nii.gz",
        "nonlinear_mask": site_dir / "nonlinear_mask.nii.gz",
        "warp": site_dir / "forward_warp_ants.nii.gz",
        "log": site_dir / "registration.log",
        "comparison": site_dir / "comparison.png",
        "deformation": site_dir / "deformation.png",
        "jacobian": site_dir / "jacobian_determinant.nii.gz",
        "gradient_csv": site_dir / "gradient_contributions.csv",
        "gradient_plot": site_dir / "gradient_contributions.png",
    }


def append_values(command, option, values):
    command.append(option)
    command.extend(str(value) for value in values)


def run_registration(args, row):
    site_dir = args.output_dir / "sites" / row["site"]
    site_dir.mkdir(parents=True, exist_ok=True)
    out = outputs(site_dir)
    if args.overwrite or not out["source_mask"].exists():
        save_mask(Path(row["src_vol"]), out["source_mask"])
    required = [
        out[k] for k in
        ("affine", "affine_mask", "nonlinear", "nonlinear_mask", "warp", "gradient_csv")
    ]
    if not args.overwrite and all(path.exists() for path in required):
        return 0.0
    command = [
        args.python, str(HERE / "register_with_nosurf.py"),
        "--src_vol", row["src_vol"], "--trg_vol", str(TARGET),
        "--src_mask", str(out["source_mask"]), "--trg_mask", str(args.output_dir / "target_mask.nii.gz"),
        "--src_lbl", str(out["source_mask"]),
        "--out_affine_vol", str(out["affine"]), "--out_affine_lbl", str(out["affine_mask"]),
        "--out_vol", str(out["nonlinear"]), "--out_lbl", str(out["nonlinear_mask"]),
        "--out_warp", str(out["warp"]), "--warp_format", "ants",
        "--crop_margin", str(args.crop_margin), "--learning_rate", str(args.learning_rate),
        "--smooth_warp_sigma", str(args.smooth_warp_sigma),
        "--smooth_grad_sigma", str(args.smooth_grad_sigma),
        "--affine_max_shear", str(args.affine_max_shear),
        "--displacement_weight", str(args.displacement_weight),
        "--consistency_weight", str(args.consistency_weight),
        "--gradient_report_interval", str(args.gradient_report_interval),
        "--gradient_report_csv", str(out["gradient_csv"]),
    ]
    append_values(command, "--scales", args.scales)
    append_values(command, "--iterations", args.iterations)
    append_values(command, "--affine_scales", args.affine_scales)
    append_values(command, "--affine_iterations", args.affine_iterations)
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": args.cuda_visible_devices, "USE_FFO": "False", "MPLBACKEND": "Agg"})
    started = time.monotonic()
    with out["log"].open("w", encoding="utf-8") as log:
        log.write("$ " + shlex.join(command) + "\n\n")
        log.flush()
        result = subprocess.run(command, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    elapsed = time.monotonic() - started
    if result.returncode:
        raise RuntimeError(f"registration exited {result.returncode}; inspect {out['log']}")
    return elapsed


def dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    denom = int(a.sum() + b.sum())
    return float(2 * np.logical_and(a, b).sum() / denom) if denom else math.nan


def masked_ncc(a, b, mask):
    valid = mask & np.isfinite(a) & np.isfinite(b)
    x, y = a[valid].astype(np.float64), b[valid].astype(np.float64)
    if x.size < 2:
        return math.nan
    x -= x.mean(); y -= y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denom) if denom else math.nan


def jacobian_from_ants(warp_path: Path, reference):
    displacement_lps = np.asarray(nib.load(str(warp_path)).dataobj, dtype=np.float64)
    displacement_lps = np.squeeze(displacement_lps)
    if displacement_lps.shape != (*reference.shape[:3], 3):
        raise ValueError(f"Unexpected ANTs warp shape: {displacement_lps.shape}")
    displacement_ras = displacement_lps * np.asarray([-1.0, -1.0, 1.0])
    ijk = np.stack(np.meshgrid(*(np.arange(n) for n in reference.shape[:3]), indexing="ij"), axis=-1)
    world = nib.affines.apply_affine(reference.affine, ijk)
    phi = world + displacement_ras
    jac_index = np.empty((*reference.shape[:3], 3, 3), dtype=np.float64)
    for component in range(3):
        for axis, derivative in enumerate(np.gradient(phi[..., component], edge_order=1)):
            jac_index[..., component, axis] = derivative
    return np.linalg.det(jac_index) / np.linalg.det(reference.affine[:3, :3]), np.linalg.norm(displacement_ras, axis=-1), phi


def normalize_slice(data, axis, index):
    plane = np.take(data, index, axis=axis).astype(np.float32)
    values = plane[np.isfinite(plane) & (plane != 0)]
    if values.size:
        low, high = np.percentile(values, [1, 99])
        plane = np.clip((plane - low) / max(high - low, 1e-6), 0, 1)
    else:
        plane[:] = 0
    return np.rot90(plane)


def resample_identity(data, shape, order):
    scale = np.asarray(data.shape[:3], float) / np.asarray(shape[:3], float)
    return ndimage.affine_transform(data, np.diag(scale), 0.5 * scale - 0.5, output_shape=shape[:3], order=order, mode="constant", cval=0, prefilter=False)


def make_figures(row, out, target, target_mask, affine, nonlinear, jac, magnitude, phi):
    source = np.asarray(nib.load(row["src_vol"]).dataobj)
    initial = resample_identity(source, target.shape, 1)
    center = [int(round(v)) for v in ndimage.center_of_mass(target_mask)]
    volumes = [initial, target, affine, nonlinear]
    labels = ["Initial voxel identity", "Target", "Affine", "Nonlinear + smoothing"]
    fig, axes = plt.subplots(3, 4, figsize=(14, 10), constrained_layout=True)
    for r, axis in enumerate((0, 1, 2)):
        for c, (volume, label) in enumerate(zip(volumes, labels)):
            src = normalize_slice(volume, axis, center[axis])
            if c in (0, 1):
                rgb = np.repeat(src[..., None], 3, axis=-1)
            else:
                trg = normalize_slice(target, axis, center[axis])
                rgb = np.stack((trg, src, np.zeros_like(trg)), axis=-1)
            axes[r, c].imshow(rgb, origin="lower")
            axes[r, c].axis("off")
            if r == 0: axes[r, c].set_title(label)
    fig.suptitle(f"{row['site']} · target=red, registered source=green")
    fig.savefig(out["comparison"], dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for c, axis in enumerate((0, 1, 2)):
        idx = center[axis]
        jplane = np.rot90(np.take(jac, idx, axis=axis))
        mplane = np.rot90(np.take(magnitude, idx, axis=axis))
        axes[0, c].imshow(jplane, cmap="RdBu_r", vmin=0, vmax=2, origin="lower")
        axes[0, c].contour(jplane <= 0, levels=[0.5], colors="black", linewidths=0.5)
        axes[0, c].set_title(("Sagittal", "Coronal", "Axial")[c] + " det(J)")
        axes[1, c].imshow(mplane, cmap="magma", origin="lower")
        components = [component for component in range(3) if component != axis]
        for component, color in zip(components, ("cyan", "lime")):
            plane = np.rot90(np.take(phi[..., component], idx, axis=axis))
            lo, hi = np.nanpercentile(plane, [5, 95])
            if hi > lo:
                axes[1, c].contour(plane, levels=np.linspace(lo, hi, 12), colors=color, linewidths=0.35, alpha=0.8)
        axes[1, c].set_title("Displacement magnitude + deformed grid")
        for r in (0, 1): axes[r, c].axis("off")
    fig.suptitle(f"{row['site']} · deformation diagnostics")
    fig.savefig(out["deformation"], dpi=150); plt.close(fig)

def make_gradient_figure(csv_path: Path, output_path: Path, site: str):
    if not csv_path.exists():
        return {}

    numeric_fields = (
        "global_iteration",
        "scale",
        "weighted_combined_grad_rms",
        "weighted_norm_share",
        "projection_fraction_on_total",
        "cosine_with_total_gradient",
    )
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed = {"component": row["component"]}
            parsed.update({field: float(row[field]) for field in numeric_fields})
            rows.append(parsed)
    if not rows:
        return {}

    preferred_order = ("image", "diffusion", "consistency", "surface")
    components = [
        component for component in preferred_order
        if any(
            row["component"] == component
            and row["weighted_combined_grad_rms"] > 0
            for row in rows
        )
    ]
    components.extend(
        sorted({row["component"] for row in rows} - set(preferred_order))
    )
    colors = {
        "image": "tab:blue",
        "diffusion": "tab:orange",
        "consistency": "tab:green",
        "surface": "tab:red",
    }
    panels = (
        ("weighted_combined_grad_rms", "Weighted gradient RMS", True),
        ("weighted_norm_share", "Gradient-norm share", False),
        ("projection_fraction_on_total", "Projection on total gradient", False),
        ("cosine_with_total_gradient", "Cosine with total gradient", False),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes = axes.ravel()
    for component in components:
        subset = [row for row in rows if row["component"] == component]
        x = np.asarray([row["global_iteration"] for row in subset])
        for axis, (field, _, use_log) in zip(axes, panels):
            y = np.asarray([row[field] for row in subset])
            if use_log:
                y = np.maximum(y, np.finfo(float).tiny)
            axis.plot(
                x,
                y,
                marker="o",
                markersize=2.5,
                linewidth=1.2,
                label=component,
                color=colors.get(component),
            )

    scale_starts = {}
    for row in rows:
        scale_starts.setdefault(int(row["scale"]), row["global_iteration"])
    for axis, (field, title, use_log) in zip(axes, panels):
        if use_log:
            axis.set_yscale("log")
        if field == "projection_fraction_on_total":
            axis.axhline(0, color="black", linewidth=0.7)
            axis.axhline(1, color="0.5", linewidth=0.7, linestyle=":")
        if field == "cosine_with_total_gradient":
            axis.axhline(0, color="black", linewidth=0.7)
        for scale, start in scale_starts.items():
            axis.axvline(start, color="0.75", linewidth=0.7, linestyle="--")
            axis.text(
                start,
                0.98,
                f"s={scale}",
                transform=axis.get_xaxis_transform(),
                va="top",
                ha="left",
                fontsize=8,
            )
        axis.set_title(title)
        axis.set_xlabel("Global nonlinear iteration")
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=max(1, len(components)), fontsize=8)
    fig.suptitle(f"{site} · per-loss deformation-gradient contributions")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    component_stats = {}
    for component in components:
        subset = [row for row in rows if row["component"] == component]
        component_stats[component] = {
            "norm_share": float(np.mean([row["weighted_norm_share"] for row in subset])),
            "projection": float(
                np.mean([row["projection_fraction_on_total"] for row in subset])
            ),
        }
    if not component_stats:
        return {}
    dominant_norm = max(component_stats, key=lambda key: component_stats[key]["norm_share"])
    dominant_projection = max(
        component_stats,
        key=lambda key: component_stats[key]["projection"],
    )
    summary = {
        "gradient_dominant_by_norm": dominant_norm,
        "gradient_dominant_by_projection": dominant_projection,
    }
    for component, stats in component_stats.items():
        summary[f"gradient_{component}_mean_norm_share"] = stats["norm_share"]
        summary[f"gradient_{component}_mean_projection"] = stats["projection"]
    return summary

def evaluate(row, output_dir):
    out = outputs(output_dir / "sites" / row["site"])
    reference = nib.load(str(TARGET)); target = np.asarray(reference.dataobj)
    target_mask = target > 0
    affine = np.asarray(nib.load(str(out["affine"])).dataobj)
    nonlinear = np.asarray(nib.load(str(out["nonlinear"])).dataobj)
    affine_mask = np.asarray(nib.load(str(out["affine_mask"])).dataobj) > 0
    nonlinear_mask = np.asarray(nib.load(str(out["nonlinear_mask"])).dataobj) > 0
    jac, magnitude, phi = jacobian_from_ants(out["warp"], reference)
    valid = ndimage.binary_erosion(target_mask, iterations=2)
    values = jac[valid]
    ijk = np.stack(np.meshgrid(*(np.arange(n) for n in reference.shape[:3]), indexing="ij"), axis=-1)
    world = nib.affines.apply_affine(reference.affine, ijk)
    fit = valid & np.all((ijk % 4) == 0, axis=-1)
    design = np.c_[world[fit], np.ones(fit.sum())]
    affine_fit = np.linalg.lstsq(design, phi[fit], rcond=None)[0]
    fitted_phi = np.einsum("...j,jk->...k", world, affine_fit[:3]) + affine_fit[3]
    residual_magnitude = np.linalg.norm(phi - fitted_phi, axis=-1)
    jac_img = nib.Nifti1Image(jac.astype(np.float32), reference.affine, reference.header)
    jac_img.set_data_dtype(np.float32); nib.save(jac_img, str(out["jacobian"]))
    make_figures(row, out, target, target_mask, affine, nonlinear, jac, magnitude, phi)
    gradient_summary = make_gradient_figure(
        out["gradient_csv"],
        out["gradient_plot"],
        row["site"],
    )
    return {
        "site": row["site"], "example": row["example"],
        **gradient_summary,
        "affine_mask_dice": dice(affine_mask, target_mask),
        "nonlinear_mask_dice": dice(nonlinear_mask, target_mask),
        "affine_ncc": masked_ncc(affine, target, affine_mask & target_mask),
        "nonlinear_ncc": masked_ncc(nonlinear, target, nonlinear_mask & target_mask),
        "folding_percent": float(100 * np.mean(values <= 0)),
        "jacobian_below_0_2_percent": float(100 * np.mean(values < 0.2)),
        "jacobian_min": float(np.min(values)), "jacobian_p01": float(np.percentile(values, 1)),
        "jacobian_median": float(np.median(values)), "jacobian_p99": float(np.percentile(values, 99)),
        "residual_displacement_p95_mm": float(np.percentile(residual_magnitude[valid], 95)),
        "displacement_p95_mm": float(np.percentile(magnitude[valid], 95)),
    }


def make_summary(metrics, output_dir):
    if not metrics: return
    labels = ["Mask Dice", "NCC", "Folding %", "det(J)<0.2 %"]
    values = np.asarray([[m["nonlinear_mask_dice"], m["nonlinear_ncc"], m["folding_percent"], m["jacobian_below_0_2_percent"]] for m in metrics])
    fig, axes = plt.subplots(1, 4, figsize=(14, max(7, len(metrics) * 0.32)), constrained_layout=True)
    for c, (axis, label) in enumerate(zip(axes, labels)):
        column = values[:, c:c + 1]
        image = axis.imshow(column, aspect="auto", cmap="viridis" if c < 2 else "magma")
        axis.set_xticks([0], [label]); axis.set_yticks(range(len(metrics)), [m["site"] for m in metrics] if c == 0 else [])
        for r, value in enumerate(column[:, 0]): axis.text(0, r, f"{value:.3g}", ha="center", va="center", fontsize=7, color="white")
        fig.colorbar(image, ax=axis, fraction=0.08)
    fig.suptitle("PRIME-DE volume-only registration validation")
    fig.savefig(output_dir / "summary_metrics.png", dpi=180); plt.close(fig)


def main():
    args = parse_args(); args.output_dir = args.output_dir.resolve(); args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.dataset_root.resolve(), args.sites)
    write_csv(args.output_dir / "manifest.csv", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if args.list_only: return
    target_mask = args.output_dir / "target_mask.nii.gz"
    if args.overwrite or not target_mask.exists(): save_mask(TARGET, target_mask)
    ready = [row for row in manifest if row["status"] == "ready"]
    timings = []
    metrics = []
    if not args.evaluate_only:
        for index, row in enumerate(ready, 1):
            print(f"[{index}/{len(ready)}] {row['site']} {row['example']}", flush=True)
            try:
                elapsed = run_registration(args, row)
                timings.append({"site": row["site"], "seconds": elapsed, "status": "complete"})
                metric = evaluate(row, args.output_dir)
                metrics.append(metric)
                write_csv(args.output_dir / "metrics.csv", metrics)
                if metric["folding_percent"] > 0:
                    print(f"WARNING: {row['site']} has {metric['folding_percent']:.6g}% non-positive Jacobians", flush=True)
            except Exception as error:
                timings.append({"site": row["site"], "seconds": "", "status": f"failed: {error}"})
            write_csv(args.output_dir / "timings.csv", timings)
    else:
        for row in ready:
            try: metrics.append(evaluate(row, args.output_dir))
            except Exception as error: print(f"Evaluation skipped for {row['site']}: {error}", file=sys.stderr)
    write_csv(args.output_dir / "metrics.csv", metrics)
    make_summary(metrics, args.output_dir)


if __name__ == "__main__":
    main()
