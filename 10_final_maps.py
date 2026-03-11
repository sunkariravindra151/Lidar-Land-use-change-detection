"""
10_final_maps.py
----------------
Generates publication-ready figures (300 DPI) for the paper.

Figure 1: LULC 2012 vs 2016 side-by-side comparison
Figure 2: Change detection map (gain/loss)
Figure 3: Damage severity map
Figure 4: Binary change map with statistics

All figures saved as PNG (300 DPI) and PDF.
"""

import os
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

BASE_DIR = r"C:\Users\Sunkari Ravindra\OneDrive\Desktop\My Documents\QGIS project"
OUT_DIR  = os.path.join(BASE_DIR, "outputs")

NODATA = -9999
DPI    = 300

# ------------------------------------------------
# Colour schemes
# ------------------------------------------------

LULC_COLORS = ["#d4b483", "#a8d08d", "#4daf4a", "#1b7837", "#808080"]
LULC_LABELS = ["Bare ground", "Grass / low veg", "Medium veg", "Dense veg", "Built / rock"]
lulc_cmap   = ListedColormap(LULC_COLORS)
lulc_norm   = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], lulc_cmap.N)

CHANGE_COLORS = ["#d73027", "#d0d0d0", "#1a9641"]
CHANGE_LABELS = ["Vegetation loss", "No change", "Vegetation gain"]
change_cmap   = ListedColormap(CHANGE_COLORS)
change_norm   = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], change_cmap.N)

SEV_COLORS = ["#d0d0d0", "#1a9641", "#fdae61", "#d73027"]
SEV_LABELS = ["No change", "Recovery", "Medium damage", "High damage"]
sev_cmap   = ListedColormap(SEV_COLORS)
sev_norm   = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], sev_cmap.N)

# ------------------------------------------------
# Load all maps
# ------------------------------------------------

def load_masked(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
    data[data == NODATA]   = np.nan
    data[data == -32768]   = np.nan
    return data

print("Loading maps...")
lulc_2012 = load_masked(os.path.join(OUT_DIR, "lulc_2012.tif"))
lulc_2016 = load_masked(os.path.join(OUT_DIR, "lulc_2016.tif"))
gain_loss  = load_masked(os.path.join(OUT_DIR, "gain_loss_map.tif"))
severity   = load_masked(os.path.join(OUT_DIR, "damage_severity.tif"))
binary     = load_masked(os.path.join(OUT_DIR, "binary_change_map.tif"))


def add_legend(ax, cmap, norm, labels, title=""):
    patches = [mpatches.Patch(color=cmap(norm(i)), label=labels[i]) for i in range(len(labels))]
    ax.legend(handles=patches, loc="lower left", fontsize=7,
              framealpha=0.9, title=title, title_fontsize=8)


def add_scalebar(ax, pixel_size_m=1, length_m=1000):
    """Add a simple scale bar."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    bar_pixels = length_m / pixel_size_m
    x0 = xlim[0] + (xlim[1] - xlim[0]) * 0.05
    y0 = ylim[0] + (ylim[1] - ylim[0]) * 0.05
    ax.plot([x0, x0 + bar_pixels], [y0, y0], "k-", linewidth=3)
    ax.text(x0 + bar_pixels / 2, y0 + (ylim[1]-ylim[0])*0.02,
            f"{length_m}m", ha="center", fontsize=7, fontweight="bold")


# ================================================
# Figure 1: LULC 2012 vs 2016
# ================================================
print("Generating Figure 1: LULC comparison...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Land Use / Land Cover — Before and After 2016 Kaikoura Earthquake",
             fontsize=14, fontweight="bold", y=1.01)

for ax, data, year in zip(axes, [lulc_2012, lulc_2016], ["2012 (Pre-earthquake)", "2016 (Post-earthquake)"]):
    im = ax.imshow(data, cmap=lulc_cmap, norm=lulc_norm, interpolation="nearest")
    ax.set_title(year, fontsize=12, fontweight="bold")
    ax.set_xlabel("Easting (pixels)", fontsize=9)
    ax.set_ylabel("Northing (pixels)", fontsize=9)
    add_legend(ax, lulc_cmap, lulc_norm, LULC_LABELS, title="LULC Class")
    ax.tick_params(labelsize=7)

plt.tight_layout()
for ext in ["png", "pdf"]:
    path = os.path.join(OUT_DIR, f"figure1_lulc_comparison.{ext}")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"  Saved: figure1_lulc_comparison.{ext}")
plt.close()

# ================================================
# Figure 2: Gain / Loss Change Map
# ================================================
print("Generating Figure 2: Change map...")

fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(gain_loss, cmap=change_cmap, norm=change_norm, interpolation="nearest")
ax.set_title("Vegetation Change Map — 2016 Kaikoura Earthquake\n"
             "(Derived from LiDAR + Sentinel-2 NDVI, 2012 vs 2016)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Easting (pixels)", fontsize=10)
ax.set_ylabel("Northing (pixels)", fontsize=10)
add_legend(ax, change_cmap, change_norm, CHANGE_LABELS, title="Change Type")

# Stats annotation
valid_px = np.sum(~np.isnan(gain_loss))
loss_pct = np.sum(gain_loss == -1) / valid_px * 100
gain_pct = np.sum(gain_loss ==  1) / valid_px * 100
stats_txt = (f"Vegetation loss: {loss_pct:.1f}%\n"
             f"Vegetation gain: {gain_pct:.1f}%\n"
             f"Study area: Kaikoura, NZ")
ax.text(0.98, 0.98, stats_txt, transform=ax.transAxes,
        fontsize=9, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

plt.tight_layout()
for ext in ["png", "pdf"]:
    path = os.path.join(OUT_DIR, f"figure2_change_map.{ext}")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"  Saved: figure2_change_map.{ext}")
plt.close()

# ================================================
# Figure 3: Damage Severity Map
# ================================================
print("Generating Figure 3: Damage severity...")

fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(severity, cmap=sev_cmap, norm=sev_norm, interpolation="nearest")
ax.set_title("Earthquake Damage Severity Map — Kaikoura 2016\n"
             "(Based on LULC class transition magnitude)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Easting (pixels)", fontsize=10)
ax.set_ylabel("Northing (pixels)", fontsize=10)
add_legend(ax, sev_cmap, sev_norm, SEV_LABELS, title="Damage Severity")

plt.tight_layout()
for ext in ["png", "pdf"]:
    path = os.path.join(OUT_DIR, f"figure3_damage_severity.{ext}")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"  Saved: figure3_damage_severity.{ext}")
plt.close()

# ================================================
# Figure 4: Binary Change Map
# ================================================
print("Generating Figure 4: Binary change map...")

bin_cmap = ListedColormap(["#f0f0f0", "#e31a1c"])
bin_norm = BoundaryNorm([-0.5, 0.5, 1.5], bin_cmap.N)

fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(binary, cmap=bin_cmap, norm=bin_norm, interpolation="nearest")
ax.set_title("Binary Change Detection Map — Kaikoura 2016\n"
             "(Red = any land-cover class change detected)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Easting (pixels)", fontsize=10)
ax.set_ylabel("Northing (pixels)", fontsize=10)

bin_patches = [
    mpatches.Patch(color="#f0f0f0", label="Unchanged (79.7%)"),
    mpatches.Patch(color="#e31a1c", label="Changed (20.3%)"),
]
ax.legend(handles=bin_patches, loc="lower left", fontsize=9, framealpha=0.9)

plt.tight_layout()
for ext in ["png", "pdf"]:
    path = os.path.join(OUT_DIR, f"figure4_binary_change.{ext}")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"  Saved: figure4_binary_change.{ext}")
plt.close()

print("\n" + "=" * 65)
print("  All final maps generated (300 DPI PNG + PDF).")
print("  Outputs:")
print("    figure1_lulc_comparison.png / .pdf")
print("    figure2_change_map.png / .pdf")
print("    figure3_damage_severity.png / .pdf")
print("    figure4_binary_change.png / .pdf")
print("=" * 65)