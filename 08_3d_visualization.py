"""
08_3d_visualization.py
----------------------
3D visualization comparing 2012 vs 2016 for the Kaikoura study area.

Produces 4 figures:
  3d_compare_2012_2016.png   — side-by-side 3D surfaces coloured by LULC
  3d_elevation_change.png    — single 3D surface coloured by elevation change
  3d_gain_loss.png           — 3D surface draped with vegetation gain/loss
  2d_profile_comparison.png  — elevation + CHM cross-section 2012 vs 2016
"""

import os
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

BASE_DIR = r"C:\Users\Sunkari Ravindra\OneDrive\Desktop\My Documents\QGIS project"
OUT_DIR  = os.path.join(BASE_DIR, "outputs")

NODATA = -9999
STEP   = 25   # downsample step for 3D rendering

# LULC colour scheme
LULC_COLORS = ["#d4b483", "#a8d08d", "#4daf4a", "#1b7837", "#808080"]
LULC_LABELS = ["Bare ground", "Grass/low veg", "Medium veg", "Dense veg", "Built/rock"]
lulc_cmap   = ListedColormap(LULC_COLORS)
lulc_norm   = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], lulc_cmap.N)

CHANGE_COLORS = ["#d73027", "#d0d0d0", "#1a9641"]
change_cmap   = ListedColormap(CHANGE_COLORS)
change_norm   = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], change_cmap.N)

# ------------------------------------------------
# Load and downsample rasters
# ------------------------------------------------

def load_ds(path, step=STEP, nodata_val=NODATA):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
    data[data == nodata_val] = np.nan
    data[data == -32768]     = np.nan
    return data[::step, ::step]

print("Loading rasters...")
t1_dtm    = load_ds(os.path.join(OUT_DIR, "t1_dtm.tif"))
t2_dtm    = load_ds(os.path.join(OUT_DIR, "t2_dtm.tif"))
t1_chm    = load_ds(os.path.join(OUT_DIR, "t1_chm.tif"))
t2_chm    = load_ds(os.path.join(OUT_DIR, "t2_chm.tif"))
lulc_2012 = load_ds(os.path.join(OUT_DIR, "lulc_2012.tif"))
lulc_2016 = load_ds(os.path.join(OUT_DIR, "lulc_2016.tif"))
gain_loss  = load_ds(os.path.join(OUT_DIR, "gain_loss_map.tif"))

rows, cols = t1_dtm.shape
X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

print(f"Downsampled grid: {rows} x {cols}")

# ================================================
# Figure 1: Side-by-side 3D LULC — 2012 vs 2016
# ================================================
print("\nFigure 1: Side-by-side 3D LULC comparison...")

fig = plt.figure(figsize=(20, 9))
fig.suptitle("3D LULC Comparison — 2012 vs 2016 (Kaikoura Earthquake)",
             fontsize=14, fontweight="bold", y=1.01)

for idx, (dtm, lulc, year) in enumerate([
    (t1_dtm, lulc_2012, "2012 — Pre-earthquake"),
    (t2_dtm, lulc_2016, "2016 — Post-earthquake"),
]):
    ax = fig.add_subplot(1, 2, idx + 1, projection="3d")

    # Build face colours from LULC map
    face_col = lulc_cmap(lulc_norm(lulc))
    face_col[np.isnan(lulc)] = [0.85, 0.85, 0.85, 0.2]

    ax.plot_surface(X, Y, dtm,
                    facecolors=face_col,
                    linewidth=0,
                    antialiased=True,
                    shade=True,
                    alpha=0.95)

    ax.set_title(year, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Easting", fontsize=8, labelpad=5)
    ax.set_ylabel("Northing", fontsize=8, labelpad=5)
    ax.set_zlabel("Elevation (m)", fontsize=8, labelpad=5)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=30, azim=225)
    ax.set_box_aspect([cols, rows, 25])

# Shared legend
patches = [mpatches.Patch(color=LULC_COLORS[i], label=LULC_LABELS[i]) for i in range(5)]
fig.legend(handles=patches, loc="lower center", ncol=5,
           fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
out = os.path.join(OUT_DIR, "3d_compare_2012_2016.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 3d_compare_2012_2016.png")

# ================================================
# Figure 2: 3D Elevation Change (t2 - t1 DTM)
# ================================================
print("Figure 2: 3D elevation change surface...")

elev_change = t2_dtm - t1_dtm

fig = plt.figure(figsize=(14, 9))
ax  = fig.add_subplot(111, projection="3d")

# Colour surface by elevation change
vmax = np.nanpercentile(np.abs(elev_change), 95)
norm_ec = plt.Normalize(vmin=-vmax, vmax=vmax)
face_ec = plt.cm.RdYlGn(norm_ec(elev_change))
face_ec[np.isnan(elev_change)] = [0.85, 0.85, 0.85, 0.1]

ax.plot_surface(X, Y, t1_dtm,
                facecolors=face_ec,
                linewidth=0,
                antialiased=True,
                shade=True,
                alpha=0.95)

ax.set_title("3D Elevation Change — 2012 to 2016 (Kaikoura)\n"
             "Green = elevation gain  |  Red = elevation loss",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Easting", fontsize=9)
ax.set_ylabel("Northing", fontsize=9)
ax.set_zlabel("Elevation (m)", fontsize=9)
ax.view_init(elev=30, azim=225)
ax.set_box_aspect([cols, rows, 25])

# Colorbar
sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm_ec)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
cbar.set_label("Elevation change (m)", fontsize=9)

plt.tight_layout()
out = os.path.join(OUT_DIR, "3d_elevation_change.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 3d_elevation_change.png")

# ================================================
# Figure 3: 3D Surface draped with gain/loss map
# ================================================
print("Figure 3: 3D gain/loss draped on DTM...")

fig = plt.figure(figsize=(14, 9))
ax  = fig.add_subplot(111, projection="3d")

face_gl = change_cmap(change_norm(gain_loss))
face_gl[np.isnan(gain_loss)] = [0.85, 0.85, 0.85, 0.2]

ax.plot_surface(X, Y, t1_dtm,
                facecolors=face_gl,
                linewidth=0,
                antialiased=True,
                shade=True,
                alpha=0.95)

ax.set_title("Vegetation Change Draped on DTM — Kaikoura 2016\n"
             "Red = vegetation loss  |  Grey = no change  |  Green = vegetation gain",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Easting", fontsize=9)
ax.set_ylabel("Northing", fontsize=9)
ax.set_zlabel("Elevation (m)", fontsize=9)
ax.view_init(elev=35, azim=200)
ax.set_box_aspect([cols, rows, 25])

patches = [
    mpatches.Patch(color="#d73027", label="Vegetation loss"),
    mpatches.Patch(color="#d0d0d0", label="No change"),
    mpatches.Patch(color="#1a9641", label="Vegetation gain"),
]
ax.legend(handles=patches, loc="upper left", fontsize=9)

plt.tight_layout()
out = os.path.join(OUT_DIR, "3d_gain_loss.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 3d_gain_loss.png")

# ================================================
# Figure 4: 2D Cross-section profile — 2012 vs 2016
# Elevation + canopy height along middle row
# ================================================
print("Figure 4: Cross-section profile comparison...")

mid_row = rows // 2

# Full resolution cross-section (reload without downsampling)
def load_row(path, row_idx, step=STEP, nodata_val=NODATA):
    with rasterio.open(path) as src:
        h = src.height
        actual_row = min(row_idx * step, h - 1)
        data = src.read(1, window=rasterio.windows.Window(0, actual_row, src.width, 1))
    data = data[0].astype(np.float32)
    data[data == nodata_val] = np.nan
    data[data == -32768]     = np.nan
    return data

print("  Loading cross-section rows...")
dtm_row_2012 = load_row(os.path.join(OUT_DIR, "t1_dtm.tif"), mid_row)
dtm_row_2016 = load_row(os.path.join(OUT_DIR, "t2_dtm.tif"), mid_row)
chm_row_2012 = load_row(os.path.join(OUT_DIR, "t1_chm.tif"), mid_row)
chm_row_2016 = load_row(os.path.join(OUT_DIR, "t2_chm.tif"), mid_row)

x_prof = np.arange(len(dtm_row_2012))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
fig.suptitle("Elevation & Canopy Height Cross-Section — 2012 vs 2016 (Kaikoura)\n"
             f"Profile along row {mid_row * STEP} (middle of study area)",
             fontsize=13, fontweight="bold")

# Top panel: DTM elevation
ax1.plot(x_prof, dtm_row_2012, color="#2166ac", linewidth=1.2, label="DTM 2012")
ax1.plot(x_prof, dtm_row_2016, color="#d73027", linewidth=1.2, label="DTM 2016", alpha=0.8)
ax1.fill_between(x_prof, dtm_row_2012, dtm_row_2016,
                 where=(dtm_row_2016 < dtm_row_2012),
                 color="#d73027", alpha=0.25, label="Elevation loss")
ax1.fill_between(x_prof, dtm_row_2012, dtm_row_2016,
                 where=(dtm_row_2016 >= dtm_row_2012),
                 color="#1a9641", alpha=0.2, label="Elevation gain")
ax1.set_ylabel("Elevation (m)", fontsize=10)
ax1.legend(fontsize=9, loc="upper right")
ax1.grid(True, alpha=0.3)
ax1.set_title("DTM Elevation Profile", fontsize=10)

# Bottom panel: CHM canopy height
ax2.plot(x_prof, chm_row_2012, color="#2166ac", linewidth=1.2, label="CHM 2012")
ax2.plot(x_prof, chm_row_2016, color="#d73027", linewidth=1.2, label="CHM 2016", alpha=0.8)
ax2.fill_between(x_prof, chm_row_2012, chm_row_2016,
                 where=(chm_row_2016 < chm_row_2012),
                 color="#d73027", alpha=0.25, label="Canopy loss")
ax2.fill_between(x_prof, chm_row_2012, chm_row_2016,
                 where=(chm_row_2016 >= chm_row_2012),
                 color="#1a9641", alpha=0.2, label="Canopy gain")
ax2.set_ylabel("Canopy height (m)", fontsize=10)
ax2.set_xlabel("Distance along profile (pixels)", fontsize=10)
ax2.legend(fontsize=9, loc="upper right")
ax2.grid(True, alpha=0.3)
ax2.set_title("CHM Canopy Height Profile", fontsize=10)

plt.tight_layout()
out = os.path.join(OUT_DIR, "2d_profile_comparison.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 2d_profile_comparison.png")

print("\n" + "=" * 60)
print("  3D Visualization complete. Outputs:")
print("    3d_compare_2012_2016.png   — LULC side-by-side")
print("    3d_elevation_change.png    — elevation diff surface")
print("    3d_gain_loss.png           — gain/loss draped on DTM")
print("    2d_profile_comparison.png  — cross-section profiles")
print("=" * 60)