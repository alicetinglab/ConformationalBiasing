#!/usr/bin/env python3
"""
combined_oligomer_analysis_kratky_addition.py

Modified version of combined_oligomer_analysis_analysis_92725.py that:
1. Adds Kratky plots (q^2*I(q) vs q) while keeping Iexp vs Ifit plots
2. Adds the date (folder name) to the top of each page

This script:
1. Matches .fit files from oligomer_outputs_v2/pdb_set_X with plot_data.txt and UV_data.txt
   from Plot_Data_Shifted based on sample names
2. For each matched sample, creates a multi-page PDF containing:
   - Fit goodness plots (I(q) vs q, residuals, Iexp vs Ifit, Kratky plots) for each .fit file
   - SEC-style plots (Rg, I0, IqL, UV) zoomed around Rg > 5
3. Outputs a single comprehensive PDF per sample

Usage:
    python combined_oligomer_analysis_kratky_addition.py [--pdb_dir path] [--plot_dir path] [--outdir path]
"""

import argparse
import os
import re
import glob
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Set
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

# axisartist triple-axis layout when available
try:
    from mpl_toolkits.axes_grid1 import host_subplot
    import mpl_toolkits.axisartist as AA

    HAS_AA = True
except Exception:
    HAS_AA = False

# Color scheme from batch_secstyle_rg_zoom_to_pdf.py
SEC_COL = [
    "#ff0000",
    "#0000FF",
    "#00ff5d",
    "#fff000",
    "#e8b218",
    "#00ff5d",
]  # Rg, I0, IqL, UV1(CH8), UV2(CH9)


@dataclass
class FitData:
    q: np.ndarray
    Iexp: np.ndarray
    sigma: np.ndarray
    Ifit: np.ndarray
    chi2: Optional[float]
    path: str


HEADER_RE = re.compile(r"Chi\^?2\s*=\s*([0-9.+-Ee]+)")


def extract_sample_name_from_fit(filename: str) -> str:
    """
    Extract sample name from .fit filename.
    Format: YYYYMMDD_Ave_SAMPLENAME_0_RANGE_trimmed_0.4.fit
    Returns: SAMPLENAME
    """
    # Remove extension
    base = filename.replace(".fit", "")

    # Split by underscores and find the pattern
    parts = base.split("_")

    # Look for the pattern: date_Ave_sample_0_range_trimmed_0.4
    if (len(parts) >= 6 and parts[1] == "Ave") or (
        len(parts) >= 8 and parts[1] == "edge" and parts[4] == "Ave"
    ):
        # Sample name is everything between 'Ave' and the first '0'
        ave_idx = parts.index("Ave")
        zero_idx = None
        for i, part in enumerate(parts[ave_idx + 1 :], ave_idx + 1):
            if part == "0":
                zero_idx = i
                break

        if zero_idx:
            sample_parts = parts[ave_idx + 1 : zero_idx]
            return "_".join(sample_parts)

    return None


def extract_sample_name_from_plot(filename: str) -> str:
    """
    Extract sample name from plot_data.txt filename.
    Format: YYYYMMDD_SAMPLENAME_plot_data.txt
    Returns: SAMPLENAME
    """
    # Remove extension and suffix
    base = filename.replace("_plot_data.txt", "")

    # Split by underscores, sample name is everything after the date
    parts = base.split("_")
    if len(parts) >= 2:
        # Skip the date (first part) and join the rest
        return "_".join(parts[1:])

    return None


def extract_date_from_plot(filename: str) -> str:
    """
    Extract date from plot_data.txt filename.
    Format: YYYYMMDD_SAMPLENAME_plot_data.txt
    Returns: YYYYMMDD
    """
    # Remove extension and suffix
    base = filename.replace("_plot_data.txt", "")

    # Split by underscores, date is the first part
    parts = base.split("_")
    if len(parts) >= 1:
        return parts[0]

    return None


def find_matching_files(
    pdb_dir: Path, plot_dir: Path
) -> Dict[str, Dict[str, List[Path]]]:
    """
    Find matching files between a single pdb_set directory and Plot_Data_Shifted directory.

    Returns:
        Dict mapping sample_name -> {
            'fit_files': [list of .fit files],
            'plot_data': Path to plot_data.txt,
            'uv_data': Path to UV_data.txt
        }
    """
    matches = {}

    # Find all .fit files in the pdb directory
    fit_files = list(pdb_dir.glob("*.fit"))
    print(f"Found {len(fit_files)} .fit files in {pdb_dir.name}")

    # Group .fit files by sample name
    fit_by_sample = {}
    for fit_file in fit_files:
        sample_name = extract_sample_name_from_fit(fit_file.name)
        if sample_name:
            if sample_name not in fit_by_sample:
                fit_by_sample[sample_name] = []
            fit_by_sample[sample_name].append(fit_file)

    print(f"Found {len(fit_by_sample)} unique samples in .fit files")

    # Find matching plot_data and UV_data files
    for sample_name, fit_files_list in fit_by_sample.items():
        plot_matches = list(plot_dir.glob(f"*{sample_name}_plot_data.txt"))
        uv_matches = list(plot_dir.glob(f"*{sample_name}_UV_data.txt"))

        if plot_matches and uv_matches:
            # Extract date from plot_data filename
            date = extract_date_from_plot(plot_matches[0].name)
            matches[sample_name] = {
                "fit_files": sorted(fit_files_list),
                "plot_data": plot_matches[0],
                "uv_data": uv_matches[0],
                "date": date,
            }
            print(f"Matched sample: {sample_name} ({len(fit_files_list)} fit files)")
        else:
            print(f"No plot/UV data found for sample: {sample_name}")

    return matches


def parse_fit(path: Path) -> FitData:
    """Parse .fit file and extract data."""
    with open(path, "r", errors="ignore") as f:
        lines = f.read().strip().splitlines()

    chi2 = None
    start_idx = 0
    if lines:
        m = HEADER_RE.search(lines[0])
        if m:
            try:
                chi2 = float(m.group(1))
            except ValueError:
                chi2 = None
            start_idx = 1

    data_lines = []
    for i, line in enumerate(lines[start_idx:], start=start_idx):
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            continue
        parts = line.split()
        parts = parts[:4]
        if len(parts) < 4:
            continue
        try:
            row = [float(x.replace("D", "E")) for x in parts]
            data_lines.append(row)
        except Exception:
            break

    if not data_lines:
        raise ValueError(f"No numeric data found in {path}")

    arr = np.array(data_lines, dtype=float)
    q, Iexp, sigma, Ifit = arr.T

    return FitData(q=q, Iexp=Iexp, sigma=sigma, Ifit=Ifit, chi2=chi2, path=str(path))


def compute_metrics(fd: FitData) -> dict:
    """Compute fit quality metrics."""
    resid = fd.Iexp - fd.Ifit
    with np.errstate(divide="ignore", invalid="ignore"):
        wresid = resid / fd.sigma
    r_factor = np.sum(np.abs(resid)) / np.sum(np.abs(fd.Iexp))
    w_r_factor = np.mean(np.abs(wresid[np.isfinite(wresid)]))
    return {
        "file": os.path.basename(fd.path),
        "n_points": int(fd.q.size),
        "chi2_header": fd.chi2,
        "R_factor": float(r_factor),
        "wR_factor": float(w_r_factor),
    }


def clean_fit_title(filename: str) -> str:
    """Clean up fit file title by removing extraneous suffixes."""
    # Remove .fit extension
    base = filename.replace(".fit", "")

    # Remove common suffixes
    suffixes_to_remove = ["_trimmed_0.4", "_trimmed", "_0.4"]
    for suffix in suffixes_to_remove:
        if base.endswith(suffix):
            base = base[: -len(suffix)]

    # Extract the range part (e.g., "285-289") for a cleaner title
    parts = base.split("_")
    if len(parts) >= 2 and parts[-1].count("-") == 1:
        # Last part looks like a range
        range_part = parts[-1]
        # Keep everything except the last part (range)
        clean_parts = parts[:-1]
        return "_".join(clean_parts) + f" ({range_part})"

    return base


def plot_fit_goodness(fd: FitData, metrics: dict, ax1, ax2, ax3, ax4):
    """Plot fit goodness on the provided axes with 4 panels including Kratky plot."""
    q, Iexp, sigma, Ifit = fd.q, fd.Iexp, fd.sigma, fd.Ifit
    resid = Iexp - Ifit
    with np.errstate(divide="ignore", invalid="ignore"):
        wresid = resid / sigma

    # Panel 1: I(q) vs q
    ax1.errorbar(
        q,
        Iexp,
        yerr=sigma,
        fmt="o",
        ms=3,
        lw=0.8,
        alpha=0.8,
        color="#1f77b4",
        label="Experimental",
        capsize=2,
    )
    ax1.plot(q, Ifit, "-", lw=2, color="#ff7f0e", label="Fit")
    ax1.set_yscale("log")
    ax1.set_xlabel("q (Å⁻¹)", fontsize=12)
    ax1.set_ylabel("I(q)", fontsize=12)
    ax1.tick_params(axis="both", which="major", labelsize=10)

    # Clean title
    clean_title = clean_fit_title(os.path.basename(fd.path))
    chi = metrics.get("chi2_header", None)
    subtitle = f"χ² = {chi:.3f}" if chi is not None else "χ² = n/a"
    ax1.set_title(f"{clean_title}\n{subtitle}", fontsize=10, pad=8)
    ax1.legend(loc="best", fontsize=9, frameon=True, fancybox=True, shadow=True)

    # Panel 2: Residuals
    ax2.axhline(0, lw=1, color="black", alpha=0.7)
    ax2.axhline(3, lw=0.8, linestyle="--", color="red", alpha=0.7)
    ax2.axhline(-3, lw=0.8, linestyle="--", color="red", alpha=0.7)
    ax2.plot(q, wresid, "o", ms=2.5, color="#2ca02c", alpha=0.7)
    ax2.set_ylabel("(Iexp - Ifit) / σ", fontsize=12)
    ax2.set_xlabel("q (Å⁻¹)", fontsize=12)
    ax2.tick_params(axis="both", which="major", labelsize=10)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Iexp vs Ifit
    ax3.plot(Ifit, Iexp, "o", ms=3, color="#d62728", alpha=0.7)
    lo = max(np.min(Ifit), np.min(Iexp))
    hi = min(np.max(Ifit), np.max(Iexp))
    ax3.plot([lo, hi], [lo, hi], "-", lw=2, color="black", alpha=0.8)
    ax3.set_xlabel("Ifit", fontsize=12)
    ax3.set_ylabel("Iexp", fontsize=12)
    ax3.tick_params(axis="both", which="major", labelsize=10)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Kratky plot (q^2 * I(q) vs q)
    kratky_exp = q**2 * Iexp
    kratky_fit = q**2 * Ifit

    ax4.plot(q, kratky_exp, "o", ms=3, color="#1f77b4", alpha=0.7, label="Experimental")
    ax4.plot(q, kratky_fit, "-", lw=2, color="#ff7f0e", label="Fit")
    ax4.set_xlabel("q (Å⁻¹)", fontsize=12)
    ax4.set_ylabel("q² · I(q)", fontsize=12)
    ax4.tick_params(axis="both", which="major", labelsize=10)
    ax4.set_title("Kratky Plot", fontsize=10, pad=8)
    ax4.legend(loc="best", fontsize=9, frameon=True, fancybox=True, shadow=True)
    ax4.grid(True, alpha=0.3)


def load_table(path: Path) -> np.ndarray:
    """Load data table from file."""
    return np.loadtxt(path, comments="#")


def zoom_xlim_rg(im, rg, threshold=5.0, left_pad=20, right_pad=10):
    """Calculate x-axis limits zoomed around Rg > threshold."""
    mask = rg > threshold
    if not np.any(mask):
        return float(np.min(im)), float(np.max(im))
    i_min = int(np.argmax(mask))
    i_max = int(len(mask) - 1 - np.argmax(mask[::-1]))
    x_min = float(im[i_min] - left_pad)
    x_max = float(im[i_max] + right_pad)
    x_min = max(x_min, float(np.min(im)))
    x_max = min(x_max, float(np.max(im)))
    if x_min >= x_max:
        return float(np.min(im)), float(np.max(im))
    return x_min, x_max


def plot_sec_style(plot_data_path: Path, uv_data_path: Path, ax1, ax2, ax3):
    """Plot SEC-style data on the provided axes."""
    pd = load_table(plot_data_path)
    uv = load_table(uv_data_path)
    if pd.shape[1] < 4:
        raise ValueError(f"Unexpected columns in plot_data: {pd.shape}")
    if uv.shape[1] < 2:
        raise ValueError(f"Unexpected columns in UV_data: {uv.shape}")

    im = pd[:, 0]
    rg = pd[:, 1]
    i0 = pd[:, 2]
    iql = pd[:, 3]

    uv_im = uv[:, 0]
    uv1 = uv[:, 1]  # UV1 (A280)
    uv2 = uv[:, 2] if uv.shape[1] > 2 else None  # UV2 (A260)

    # Titles/labels (publication quality)
    ax1.set_xlabel("Image Number", fontsize=14)
    ax1.tick_params(axis="both", which="major", labelsize=11)
    ax1.grid(True, alpha=0.3)

    # Data with improved styling
    ax1.plot(
        im,
        rg,
        "o",
        markersize=5,
        markeredgewidth=1.5,
        markeredgecolor=SEC_COL[0],
        markerfacecolor="none",
        label="R$_g$",
        zorder=4,
        alpha=0.8,
    )
    # Always use standard Matplotlib label setting to avoid axisartist API on normal Axes
    ax1.set_ylabel("R$_g$ (Å)", color=SEC_COL[0], fontsize=14)

    ax2.plot(
        im,
        i0,
        "o",
        markersize=4,
        color=SEC_COL[1],
        label="I$_{0}$",
        zorder=3,
        alpha=0.8,
    )
    ax2.plot(
        im,
        iql,
        "o",
        markersize=5,
        markeredgewidth=1.5,
        markeredgecolor=SEC_COL[2],
        markerfacecolor="none",
        label="I$_{q_L}$",
        zorder=4.5,
        alpha=0.8,
    )
    ax2.tick_params(axis="both", which="major", labelsize=11)
    ax2.set_ylabel("I$_{0}$, I$_{q_L}$", color=SEC_COL[1], fontsize=14)

    # Plot UV data (only UV280, UV260 removed)
    ax3.plot(
        uv_im,
        uv1,
        "-",
        color=SEC_COL[3],
        linewidth=2.5,
        label="UV$_{280}$",
        zorder=1,
        alpha=0.9,
    )
    ax3.set_ylim(bottom=0)
    ax3.tick_params(axis="both", which="major", labelsize=11)
    ax3.set_ylabel("UV (mAU)", color=SEC_COL[3], fontsize=14)
    ax3.spines["right"].set_position(("outward", 60))
    ax3.spines["right"].set_visible(True)

    ax2.grid(True, alpha=0.3)

    # Legend with improved styling
    lines = []
    for ax in (ax1, ax2, ax3):
        for ln in ax.get_lines():
            if ln.get_label() != "_nolegend_":
                lines.append(ln)
    labels = [ln.get_label() for ln in lines]
    ax2.legend(
        lines,
        labels,
        numpoints=1,
        loc="upper right",
        fontsize=11,
        frameon=True,
        fancybox=True,
        shadow=True,
        labelspacing=0.3,
    )

    # Zoom by Rg>5
    x_min, x_max = zoom_xlim_rg(im, rg, threshold=5.0, left_pad=20, right_pad=10)
    ax1.set_xlim(x_min, x_max)


def plot_uv_combined(uv_data_path: Path, plot_data_path: Path, ax):
    """Plot UV260, UV280 on left axis and UV260/280 ratio on right axis."""
    uv = load_table(uv_data_path)
    pd = load_table(plot_data_path)
    if uv.shape[1] < 3:
        raise ValueError(
            f"UV data needs at least 3 columns for ratio calculation: {uv.shape}"
        )
    if pd.shape[1] < 4:
        raise ValueError(
            f"Plot data needs at least 4 columns for zoom calculation: {pd.shape}"
        )

    uv_im = uv[:, 0]
    uv1 = uv[:, 1]  # UV1 (A280)
    uv2 = uv[:, 2]  # UV2 (A260)

    uv1[uv1 < 0] = 0
    uv2[uv2 < 0] = 0

    # Get Rg data for zoom calculation
    im = pd[:, 0]
    rg = pd[:, 1]

    x_min, x_max = zoom_xlim_rg(im, rg, threshold=5.0, left_pad=20, right_pad=10)
    uv1_corr = uv1 - np.min(uv1[int(x_min) : int(x_max)])

    # Calculate UV260/280 ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        uv_ratio = uv2 / uv1
        # Replace inf and nan with 0 for plotting
        uv_ratio = np.where(np.isfinite(uv_ratio), uv_ratio, 0)

        uv_ratio_corr = uv2 / uv1_corr
        uv_ratio_corr = np.where(np.isfinite(uv_ratio_corr), uv_ratio_corr, 0)

    # Calculate zoom window based on Rg > 5 (same as SEC plot)

    # Create dual y-axes
    ax1 = ax  # UV260 and UV280 (left axis)
    ax2 = ax.twinx()  # UV260/280 ratio (right axis)

    # Plot UV280 and UV260 on left axis
    ax1.plot(
        uv_im,
        uv1,
        "-",
        color=SEC_COL[3],
        linewidth=2.5,
        label="UV$_{280}$",
        zorder=1,
        alpha=0.9,
    )
    ax1.plot(
        uv_im,
        uv2,
        "-",
        color=SEC_COL[4],
        linewidth=2.5,
        label="UV$_{260}$",
        zorder=2,
        alpha=0.9,
    )

    ax1.set_ylim(bottom=0)
    ax1.set_xlim(x_min, x_max)
    # Hide left y-axis for UV plot
    ax1.set_ylabel("")
    ax1.tick_params(axis="y", labelleft=False, left=False)
    ax1.spines["left"].set_visible(False)

    # Calculate UV ratio scaling to make values higher and more visible
    ratio_mean = np.mean(uv_ratio[uv_ratio > 0])  # Mean of non-zero values
    ratio_max = np.max(uv_ratio)
    ratio_min = np.min(uv_ratio[uv_ratio > 0])  # Min of non-zero values

    # Scale to make UV ratio more visible (multiply by larger factor)
    scale_factor = 10.0  # Increased from 4 to 10 for better visibility
    scaled_ratio = uv_ratio * scale_factor
    scaled_ratio_corr = uv_ratio_corr * scale_factor

    # Plot UV260/280 ratio on right axis
    ax2.plot(
        uv_im,
        scaled_ratio,
        "-",
        color="purple",
        linewidth=2.5,
        label="UV$_{260/280}$",
        zorder=3,
        alpha=0.9,
    )

    ax2.set_ylabel("UV$_{260/280}$ Ratio", color="purple", fontsize=14)
    # Modify y-tick labels to be the true value
    ax2.tick_params(axis="y", labelcolor="purple", labelsize=11)

    ax2.set_ylim(0, 50)
    yticks = ax2.get_yticks()
    ax2.set_yticks(yticks)  # ensure fixed positions
    ax2.set_yticklabels([f"{y/scale_factor:.1f}" for y in yticks])

    # No far-right axis for UV plot - UV (mAU) axis is now shared with SEC plot

    # Set x-axis properties
    ax1.set_xlabel("Image Number", fontsize=14)
    ax1.tick_params(axis="x", labelsize=11)
    ax1.grid(True, alpha=0.3)

    # Legend for all plots
    lines = []
    for ax_plot in (ax1, ax2):
        for ln in ax_plot.get_lines():
            if ln.get_label() != "_nolegend_":
                lines.append(ln)
    labels = [ln.get_label() for ln in lines]
    ax1.legend(
        lines,
        labels,
        numpoints=1,
        loc="upper right",
        fontsize=11,
        frameon=True,
        fancybox=True,
        shadow=True,
        labelspacing=0.3,
    )


def create_sample_pdf(sample_name: str, sample_data: Dict, output_dir: Path):
    """Create a single-page comprehensive PDF for a sample with all plots."""
    # Include date in filename if available
    date = sample_data.get("date", "")
    if date:
        pdf_path = output_dir / f"{date}_{sample_name}_analysis.pdf"
    else:
        pdf_path = output_dir / f"{sample_name}_analysis.pdf"

    # Calculate layout based on number of fit files
    n_fits = len(sample_data["fit_files"])

    # Create a large figure that can accommodate all plots
    # Layout: SEC plot and UV combined plot at top, then fit plots in a grid (now 4 columns for each fit)
    fig_height = 4 + (n_fits * 2.8)  # Height for SEC + UV + fits
    fig_width = 20  # Increased width to accommodate 4 panels per fit
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Set up the layout with better spacing - SEC and UV plots at top
    # Now using 4 columns for fit plots instead of 3
    gs = fig.add_gridspec(
        n_fits + 1,
        4,
        height_ratios=[2] + [1] * n_fits,
        hspace=0.6,
        wspace=0.5,
        left=0.06,
        right=0.92,
        top=0.93,
        bottom=0.08,
    )

    try:
        # Add date as title at the top
        date = sample_data.get("date", "")
        if date:
            fig.suptitle(f"Date: {date}", fontsize=18, fontweight="bold", y=0.98)

        # SEC plot at the top left (spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 100))
        ax3.spines["right"].set_visible(True)

        ax1.set_title(f"{sample_name} - SEC-SAXS Analysis", fontsize=16, pad=20)
        plot_sec_style(sample_data["plot_data"], sample_data["uv_data"], ax1, ax2, ax3)

        # UV combined plot at the top right (spans 2 columns)
        ax_uv = fig.add_subplot(gs[0, 2:])
        ax_uv.set_title(f"{sample_name} - UV Analysis", fontsize=16, pad=20)
        plot_uv_combined(sample_data["uv_data"], sample_data["plot_data"], ax_uv)

        # Fit goodness plots below top plots (now 4 panels per fit)
        for i, fit_file in enumerate(sample_data["fit_files"]):
            try:
                fd = parse_fit(fit_file)
                metrics = compute_metrics(fd)

                # Create subplots for this fit (4 panels side by side)
                # Note: i+1 because we now have SEC and UV plots at row 0
                ax1_fit = fig.add_subplot(gs[i + 1, 0])
                ax2_fit = fig.add_subplot(gs[i + 1, 1])
                ax3_fit = fig.add_subplot(gs[i + 1, 2])
                ax4_fit = fig.add_subplot(gs[i + 1, 3])

                plot_fit_goodness(fd, metrics, ax1_fit, ax2_fit, ax3_fit, ax4_fit)

            except Exception as e:
                print(f"Error processing fit file {fit_file} for {sample_name}: {e}")
                continue

        # Save the complete figure
        pdf = PdfPages(pdf_path)
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.1)
        pdf.close()
        plt.close(fig)

        print(f"Created single-page PDF for {sample_name}: {pdf_path}")

    except Exception as e:
        print(f"Error creating PDF for {sample_name}: {e}")
        plt.close("all")


def main():
    parser = argparse.ArgumentParser(
        description="Combined oligomer analysis with Kratky plots added to existing plots"
    )
    parser.add_argument(
        "--pdb_dir",
        default="./oligomer_outputs",
        help="Directory containing pdb_set_X folders",
    )
    parser.add_argument(
        "--plot_dir",
        default="./Raw SAXS data/Plot_Data_Shifted",
        help="Directory containing plot_data.txt and UV_data.txt files",
    )
    parser.add_argument(
        "--outdir",
        default="Analysis_11-19-25_Reports",
        help="Output directory for PDFs",
    )
    args = parser.parse_args()

    pdb_base_dir = Path(args.pdb_dir).expanduser().resolve()
    plot_dir = Path(args.plot_dir).expanduser().resolve()
    base_output_dir = Path(args.outdir).expanduser().resolve()

    if not pdb_base_dir.exists():
        raise SystemExit(f"PDB base directory not found: {pdb_base_dir}")
    if not plot_dir.exists():
        raise SystemExit(f"Plot directory not found: {plot_dir}")

    # Find all pdb_set directories (filtered to only pdb_set_19)
    pdb_dirs = [
        d for d in pdb_base_dir.iterdir() if d.is_dir() and d.name == "pdb_set_19"
    ]
    pdb_dirs.sort(
        key=lambda x: (
            int(x.name.split("_")[2]) if x.name.split("_")[2].isdigit() else 999
        )
    )

    print(f"Found {len(pdb_dirs)} PDB set directories: {[d.name for d in pdb_dirs]}")

    for pdb_dir in pdb_dirs:
        pdb_set_name = pdb_dir.name

        # Create separate output directory for each PDB set
        output_dir = base_output_dir / pdb_set_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Processing {pdb_set_name}")
        print(f"PDB directory: {pdb_dir}")
        print(f"Plot directory: {plot_dir}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")

        # Find matching files for this PDB set
        matches = find_matching_files(pdb_dir, plot_dir)

        if not matches:
            print(f"No matching files found for {pdb_set_name}.")
            continue

        print(f"\nProcessing {len(matches)} samples for {pdb_set_name}...")

        # Create PDF for each sample
        for sample_name, sample_data in tqdm(matches.items()):
            print(f"Processing sample: {sample_name}")
            create_sample_pdf(sample_name, sample_data, output_dir)

        print(f"\nCompleted {pdb_set_name}! PDFs saved to: {output_dir}")

    print(f"\nAll processing complete! Check the following directories:")
    for pdb_dir in pdb_dirs:
        output_dir = base_output_dir / pdb_dir.name
        if output_dir.exists():
            print(f"  {output_dir}")


if __name__ == "__main__":
    main()
