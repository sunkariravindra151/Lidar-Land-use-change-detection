# ================================================================
#  export_tifs_for_dashboard.py
#  Kaikōura Earthquake — LULC Analysis Dashboard Image Export
# ================================================================
#  Reads every confirmed .tif from your project folder,
#  renders each one as a styled PNG, and saves to:
#       <project>\tiff_outputs\dashboard_images\
#
#  HOW TO RUN (pdal_env Anaconda prompt):
#      cd "C:\Users\Sunkari Ravindra\OneDrive\Desktop\My Documents\QGIS project"
#      python export_tifs_for_dashboard.py
#
#  REQUIREMENTS: rasterio  numpy  matplotlib
#      pip install rasterio numpy matplotlib
# ================================================================

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

try:
    import rasterio
except ImportError:
    print("ERROR: rasterio not installed.  Run:  pip install rasterio")
    sys.exit(1)

# ── Paths ────────────────────────────────────────────────────
PROJECT_DIR = r"C:\Users\Sunkari Ravindra\OneDrive\Desktop\My Documents\QGIS project"
TIF_DIR     = os.path.join(PROJECT_DIR, "outputs")
IMG_DIR  = os.path.join(PROJECT_DIR, "tiff_outputs", "dashboard_images")
os.makedirs(IMG_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────
BG         = "#0a0e14"
TICK_C     = "#6b7fa3"
SPINE_C    = "#1e2d42"
DPI        = 130
FIG_SIZE   = (8, 6)

# ── Discrete colourmaps ───────────────────────────────────────
LULC_HEX    = ["#d4b483", "#a8d08d", "#4daf4a", "#1b7837", "#808080"]
LULC_LABELS = ["Bare ground", "Grass / low veg", "Medium veg", "Dense veg", "Built / rock"]
LULC_CMAP   = ListedColormap(LULC_HEX)
LULC_NORM   = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], LULC_CMAP.N)

SEV_HEX     = ["#d0d0d0", "#1a9641", "#fdae61", "#d73027"]
SEV_LABELS  = ["No change", "Recovery", "Medium damage", "High damage"]
SEV_CMAP    = ListedColormap(SEV_HEX)
SEV_NORM    = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], SEV_CMAP.N)

GL_HEX      = ["#d73027", "#555555", "#1a9641"]
GL_LABELS   = ["Vegetation loss", "No change", "Vegetation gain"]
GL_CMAP     = ListedColormap(GL_HEX)
GL_NORM     = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], GL_CMAP.N)

BIN_HEX     = ["#2a2a2a", "#e31a1c"]
BIN_LABELS  = ["Unchanged", "Changed"]
BIN_CMAP    = ListedColormap(BIN_HEX)
BIN_NORM    = BoundaryNorm([-0.5, 0.5, 1.5], BIN_CMAP.N)

# ── Counters ──────────────────────────────────────────────────
saved   = 0
skipped = 0

# ── Core helpers ──────────────────────────────────────────────
def tif_path(fname):
    return os.path.join(TIF_DIR, fname)

def load(fname):
    """Load band-1, mask nodata, return float32 array."""
    p = tif_path(fname)
    if not os.path.exists(p):
        return None
    with rasterio.open(p) as src:
        data = src.read(1).astype(np.float32)
        nd   = src.nodata
    if nd is not None:
        data[data == nd] = np.nan
    for v in [-9999, -32768]:
        data[data == v] = np.nan
    with np.errstate(invalid="ignore"):
        data[np.abs(data) > 1e6] = np.nan
    return data

def base_fig():
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    return fig, ax

def style(ax, title):
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(colors=TICK_C, labelsize=6)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_C)
    ax.set_xlabel("Column (px)", color=TICK_C, fontsize=7)
    ax.set_ylabel("Row (px)",    color=TICK_C, fontsize=7)

def add_cb(fig, ax, im, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(label, color=TICK_C, fontsize=8)
    cb.ax.tick_params(colors=TICK_C, labelsize=7)

def add_legend(ax, hex_list, labels):
    patches = [mpatches.Patch(color=h, label=l)
               for h, l in zip(hex_list, labels)]
    ax.legend(handles=patches, loc="lower right",
              fontsize=7, framealpha=0.3,
              labelcolor="white", facecolor=BG,
              edgecolor=SPINE_C)

def save(fig, out_name):
    global saved
    out = os.path.join(IMG_DIR, out_name)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"   ✓  {out_name}")
    saved += 1

def skip(fname):
    global skipped
    print(f"   --  SKIPPED (not found): {fname}")
    skipped += 1

# ── Render types ──────────────────────────────────────────────

def r_elev(fname, out_name, title, cbar_label="Elevation (m)", cmap="Blues_r"):
    data = load(fname)
    if data is None:
        return skip(fname)
    valid = data[~np.isnan(data)]
    if valid.size == 0:
        return skip(fname)
    vmin, vmax = np.percentile(valid, 2), np.percentile(valid, 98)
    fig, ax = base_fig()
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    style(ax, title)
    add_cb(fig, ax, im, cbar_label)
    plt.tight_layout(pad=1.0)
    save(fig, out_name)

def r_elev_diff(fname1, fname2, out_name, title):
    d1 = load(fname1)
    d2 = load(fname2)
    if d1 is None or d2 is None:
        return skip(f"{fname1} or {fname2}")
    diff  = d2 - d1
    valid = diff[~np.isnan(diff)]
    if valid.size == 0:
        return skip(fname1)
    vmax = max(float(np.percentile(np.abs(valid), 95)), 0.01)
    fig, ax = base_fig()
    im = ax.imshow(diff, cmap="RdYlGn", vmin=-vmax, vmax=vmax, interpolation="nearest")
    style(ax, title)
    add_cb(fig, ax, im, "Elevation change (m)")
    plt.tight_layout(pad=1.0)
    save(fig, out_name)

def r_slope(fname, out_name, title):
    data = load(fname)
    if data is None:
        return skip(fname)
    valid = data[~np.isnan(data)]
    if valid.size == 0:
        return skip(fname)
    vmax = np.percentile(valid, 98)
    fig, ax = base_fig()
    im = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=vmax, interpolation="nearest")
    style(ax, title)
    add_cb(fig, ax, im, "Slope (degrees)")
    plt.tight_layout(pad=1.0)
    save(fig, out_name)

def r_chm(fname, out_name, title):
    data = load(fname)
    if data is None:
        return skip(fname)
    valid = data[~np.isnan(data)]
    if valid.size == 0:
        return skip(fname)
    vmin = np.percentile(valid, 2)
    vmax = np.percentile(valid, 98)
    fig, ax = base_fig()
    im = ax.imshow(data, cmap="Greens", vmin=vmin, vmax=vmax, interpolation="nearest")
    style(ax, title)
    add_cb(fig, ax, im, "Canopy height (m)")
    plt.tight_layout(pad=1.0)
    save(fig, out_name)

def r_ndvi(fname, out_name, title):
    data = load(fname)
    if data is None:
        return skip(fname)
    fig, ax = base_fig()
    im = ax.imshow(data, cmap="RdYlGn", vmin=-0.5, vmax=0.8, interpolation="nearest")
    style(ax, title)
    add_cb(fig, ax, im, "NDVI value")
    plt.tight_layout(pad=1.0)
    save(fig, out_name)

def r_ndbi(fname, out_name, title):
    data = load(fname)
    if data is None:
        return skip(fname)
    fig, ax = base_fig()
    im = ax.imshow(data, cmap="RdBu_r", vmin=-0.5, vmax=0.5, interpolation="nearest")
    style(ax, title)
    add_cb(fig, ax, im, "NDBI value")
    plt.tight_layout(pad=1.0)
    save(fig, out_name)

def r_lulc(fname, out_name, title):
    data = load(fname)
    if data is None:
        return skip(fname)
    fig, ax = base_fig()
    ax.imshow(data, cmap=LULC_CMAP, norm=LULC_NORM, interpolation="nearest")
    style(ax, title)
    add_legend(ax, LULC_HEX, LULC_LABELS)
    plt.tight_layout(pad=1.0)
    save(fig, out_name)

def r_binary(fname, out_name, title):
    data = load(fname)
    if data is None:
        return skip(fname)
    fig, ax = base_fig()
    ax.imshow(data, cmap=BIN_CMAP, norm=BIN_NORM, interpolation="nearest")
    style(ax, title)
    add_legend(ax, BIN_HEX, BIN_LABELS)
    plt.tight_layout(pad=1.0)
    save(fig, out_name)

def r_gainloss(fname, out_name, title):
    data = load(fname)
    if data is None:
        return skip(fname)
    fig, ax = base_fig()
    ax.imshow(data, cmap=GL_CMAP, norm=GL_NORM, interpolation="nearest")
    style(ax, title)
    add_legend(ax, GL_HEX, GL_LABELS)
    plt.tight_layout(pad=1.0)
    save(fig, out_name)

def r_directional(fname, out_name, title):
    data = load(fname)
    if data is None:
        return skip(fname)
    fig, ax = base_fig()
    im = ax.imshow(data, cmap="RdYlGn", vmin=-4, vmax=4, interpolation="nearest")
    style(ax, title)
    add_cb(fig, ax, im, "Class change  (-4 loss … +4 gain)")
    plt.tight_layout(pad=1.0)
    save(fig, out_name)

def r_severity(fname, out_name, title):
    data = load(fname)
    if data is None:
        return skip(fname)
    fig, ax = base_fig()
    ax.imshow(data, cmap=SEV_CMAP, norm=SEV_NORM, interpolation="nearest")
    style(ax, title)
    add_legend(ax, SEV_HEX, SEV_LABELS)
    plt.tight_layout(pad=1.0)
    save(fig, out_name)

def r_continuous(fname, out_name, title, cmap="plasma", cbar_label="Value"):
    """Generic continuous raster (feature stacks, predicted change, etc.)."""
    data = load(fname)
    if data is None:
        return skip(fname)
    valid = data[~np.isnan(data)]
    if valid.size == 0:
        return skip(fname)
    vmin, vmax = np.percentile(valid, 2), np.percentile(valid, 98)
    fig, ax = base_fig()
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    style(ax, title)
    add_cb(fig, ax, im, cbar_label)
    plt.tight_layout(pad=1.0)
    save(fig, out_name)

# ================================================================
#  EXPORT — all confirmed TIF files  (4737×6977 grid)
# ================================================================

print()
print("=" * 62)
print("  Kaikōura Dashboard — TIF to PNG Export")
print(f"  TIFs   : {TIF_DIR}")
print(f"  Output : {IMG_DIR}")
print("=" * 62)

# ── 1. LiDAR — DTM ──────────────────────────────────────────
print("\n[1/10]  DTM  (Digital Terrain Model)")
r_elev("t1_dtm.tif",           "t1_dtm.png",           "DTM 2012 — Pre-earthquake (m)")
r_elev("t2_dtm.tif",           "t2_dtm.png",           "DTM 2016 — Post-earthquake (m)")
# named alternatives confirmed in project root
r_elev("dtm_2012_common.tif",  "dtm_2012_common.png",  "DTM 2012 Common Grid (m)")
r_elev("dtm_2016_common.tif",  "dtm_2016_common.png",  "DTM 2016 Common Grid (m)")
r_elev("dtm_2016_aligned.tif", "dtm_2016_aligned.png", "DTM 2016 Aligned (m)")

# ── 2. Elevation Change ──────────────────────────────────────
print("\n[2/10]  Elevation Change")
# preferred: use t1_dtm / t2_dtm (4737×6977)
r_elev_diff("t1_dtm.tif", "t2_dtm.tif",
            "elevation_change_t1t2.png",
            "Elevation Change 2012→2016  (t1/t2 DTM, m)")
# also render the pre-computed elevation_change.tif if present
r_elev("elevation_change.tif", "elevation_change.png",
       "Elevation Change 2012→2016 (m)",
       cbar_label="Change (m)", cmap="RdYlGn")

# ── 3. LiDAR — DSM ──────────────────────────────────────────
print("\n[3/10]  DSM  (Digital Surface Model)")
r_elev("t1_dsm.tif", "t1_dsm.png", "DSM 2012 — Pre-earthquake (m)")
r_elev("t2_dsm.tif", "t2_dsm.png", "DSM 2016 — Post-earthquake (m)")

# ── 4. LiDAR — Slope ────────────────────────────────────────
print("\n[4/10]  Slope")
r_slope("t1_slope.tif", "t1_slope.png", "Slope 2012 (degrees)")
r_slope("t2_slope.tif", "t2_slope.png", "Slope 2016 (degrees)")

# ── 5. LiDAR — CHM ──────────────────────────────────────────
print("\n[5/10]  CHM  (Canopy Height Model)")
r_chm("t1_chm.tif", "t1_chm.png", "Canopy Height Model 2012 (m)")
r_chm("t2_chm.tif", "t2_chm.png", "Canopy Height Model 2016 (m)")

# ── 6. Spectral — NDVI ──────────────────────────────────────
print("\n[6/10]  NDVI")
r_ndvi("NDVI_2012_fixed.tif",   "ndvi_2012.png",         "NDVI 2012 — Offset-corrected")
r_ndvi("NDVI_2016_fixed.tif",   "ndvi_2016.png",         "NDVI 2016 — Offset-corrected")
r_ndvi("NDVI_2012_aligned.tif", "ndvi_2012_aligned.png", "NDVI 2012 — Aligned")
r_ndvi("NDVI_2016_aligned.tif", "ndvi_2016_aligned.png", "NDVI 2016 — Aligned")

# ── 7. Spectral — NDBI ──────────────────────────────────────
print("\n[7/10]  NDBI")
r_ndbi("NDBI_2012_fixed.tif",   "ndbi_2012.png",         "NDBI 2012 — Offset-corrected")
r_ndbi("NDBI_2016_fixed.tif",   "ndbi_2016.png",         "NDBI 2016 — Offset-corrected")
r_ndbi("NDBI_2012_aligned.tif", "ndbi_2012_aligned.png", "NDBI 2012 — Aligned")
r_ndbi("NDBI_2016_aligned.tif", "ndbi_2016_aligned.png", "NDBI 2016 — Aligned")

# ── 8. LULC Classification ───────────────────────────────────
print("\n[8/10]  LULC  Classification Maps")
r_lulc("lulc_2012.tif",           "lulc_2012.png",  "LULC 2012 — Observed")
r_lulc("lulc_2016.tif",           "lulc_2016.png",  "LULC 2016 — Observed")
r_lulc("predicted_lulc_2020.tif", "lulc_2020.png",  "LULC 2020 — CA-Markov Predicted")
r_lulc("predicted_lulc_2024.tif", "lulc_2024.png",  "LULC 2024 — CA-Markov Predicted")

# ── 9. Change Detection ──────────────────────────────────────
print("\n[9/10]  Change Detection Maps")

# Binary change maps
r_binary("binary_change_map.tif",       "binary_change.png",       "Binary Change 2012→2016")
r_binary("binary_change_2016_2020.tif", "binary_change_2016_2020.png", "Binary Change 2016→2020 (Predicted)")
r_binary("binary_change_2020_2024.tif", "binary_change_2020_2024.png", "Binary Change 2020→2024 (Predicted)")

# Gain / Loss maps
r_gainloss("gain_loss_map.tif",       "gain_loss.png",            "Gain / Loss 2012→2016")
r_gainloss("gain_loss_2016_2020.tif", "gain_loss_2016_2020.png",  "Gain / Loss 2016→2020 (Predicted)")
r_gainloss("gain_loss_2020_2024.tif", "gain_loss_2020_2024.png",  "Gain / Loss 2020→2024 (Predicted)")

# Directional change maps
r_directional("directional_change_map.tif",       "directional_change.png",
              "Directional Change 2012→2016")
r_directional("directional_change_2016_2020.tif", "directional_change_2016_2020.png",
              "Directional Change 2016→2020 (Predicted)")
r_directional("directional_change_2020_2024.tif", "directional_change_2020_2024.png",
              "Directional Change 2020→2024 (Predicted)")

# Predicted change summary maps
r_gainloss("predicted_change_2016_2020.tif", "predicted_change_2016_2020.png",
           "Predicted Change 2016→2020")
r_gainloss("predicted_change_2016_2024.tif", "predicted_change_2016_2024.png",
           "Predicted Change 2016→2024")

# ── 10. Damage Severity ──────────────────────────────────────
print("\n[10/10]  Damage Severity Maps")
r_severity("damage_severity.tif",            "damage_severity.png",
           "Damage Severity 2012→2016")
r_severity("damage_severity_2016_2020.tif",  "damage_severity_2016_2020.png",
           "Damage Severity 2016→2020 (Predicted)")
r_severity("damage_severity_2020_2024.tif",  "damage_severity_2020_2024.png",
           "Damage Severity 2020→2024 (Predicted)")

# ── Done ─────────────────────────────────────────────────────
print()
print("=" * 62)
print(f"  Saved   : {saved}  images")
print(f"  Skipped : {skipped}  (file not found)")
print(f"  Folder  : {IMG_DIR}")
print()
print("  Next step:")
print("  Open  tiff_outputs\\kaikoura_dashboard.html  in Chrome/Edge")
print("=" * 62)
print()