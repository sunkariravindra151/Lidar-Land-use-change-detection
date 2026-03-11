# ================================================================
#  add_missing_pngs.py
#  Generates missing dashboard images AND feature stack TIFs:
#
#    0. Build t1_feature_stack.tif + t2_feature_stack.tif
#         (stacks DTM · Slope · CHM · NDVI · NDBI into 5-band rasters,
#          aligning/resampling all layers to a common grid)
#    1. Feature stack band visualizations  (t1 + t2, 5 bands each)
#    2. Feature fusion composite PNGs      (t1 + t2 overview)
#    3. Feature stack RGB composite        (t1 + t2)
#    4. Confusion matrix PNGs              (all 3 periods)
#    5. LAZ / point-cloud renders          (2012 + 2016)
#
#  RUN FIRST, then run build_dashboard.py
#
#  cd "C:\Users\Sunkari Ravindra\...\QGIS project"
#  python add_missing_pngs.py
# ================================================================

import os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject, calculate_default_transform
except ImportError:
    print("ERROR: pip install rasterio"); sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────
PROJECT  = r"C:\Users\Sunkari Ravindra\OneDrive\Desktop\My Documents\QGIS project"
TIF_DIR  = os.path.join(PROJECT, "outputs")
IMG_DIR  = os.path.join(PROJECT, "tiff_outputs", "dashboard_images")
os.makedirs(IMG_DIR, exist_ok=True)

BG, TC, SC = "#0a0e14", "#6b7fa3", "#1e2d42"
DPI = 130

# ── helpers ───────────────────────────────────────────────────
def tif(f):  return os.path.join(TIF_DIR, f)
def img_p(f): return os.path.join(IMG_DIR, f)

def load_band(fname, band=1):
    p = tif(fname)
    if not os.path.exists(p): return None
    with rasterio.open(p) as src:
        data = src.read(band).astype(np.float32)
        nd   = src.nodata
    if nd is not None: data[data == nd] = np.nan
    for v in [-9999, -32768]: data[data == v] = np.nan
    with np.errstate(invalid="ignore"): data[np.abs(data) > 1e6] = np.nan
    return data

def pct2(arr, lo=2, hi=98):
    v = arr[~np.isnan(arr)]
    if v.size == 0: return 0, 1
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))

def save(fig, name):
    out = img_p(name)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"   ✓  {name}")

def load_rgb_tif(fname):
    """Load a QGIS-rendered RGB/RGBA TIF → uint8 HxWx3 numpy array, or None."""
    p = tif(fname)
    if not os.path.exists(p):
        return None
    with rasterio.open(p) as src:
        r = src.read(1); g = src.read(2); b = src.read(3)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)

def save_qgis_rgb(src_tif, out_png, title, subtitle="QGIS export"):
    """Read a QGIS-rendered RGB TIF and save it straight to dashboard_images as PNG."""
    rgb = load_rgb_tif(src_tif)
    if rgb is None:
        print(f"   --  SKIPPED ({src_tif} not found)")
        return
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(f"{title}\n{subtitle}", color="white", fontsize=11,
                 fontweight="bold", pad=8, fontfamily="monospace")
    ax.tick_params(colors=TC, labelsize=6)
    for sp in ax.spines.values(): sp.set_edgecolor(SC)
    plt.tight_layout(pad=1.0)
    save(fig, out_png)

# ================================================================
#  STEP 0b — Export QGIS-rendered TIFs → dashboard PNG names
#  These replace matplotlib-generated PNGs for CHM, NDVI, NDBI
#  so the dashboard shows your exact QGIS colour styling.
# ================================================================
print("\n[0b]  Exporting QGIS-rendered images → dashboard PNGs")
save_qgis_rgb("t1_chm_turbo.tiff", "t1_chm.png",
              "CHM 2012 — Canopy Height Model", "QGIS · turbo · 0–20 m")
save_qgis_rgb("t2_chm_turbo.tiff", "t2_chm.png",
              "CHM 2016 — Canopy Height Model", "QGIS · turbo · 0–20 m")
save_qgis_rgb("NDVI_2012_qgis.tif", "ndvi_2012.png",
              "NDVI 2012 — Vegetation Index", "QGIS · RdYlGn · −0.5 → 0.8")
save_qgis_rgb("NDVI_2016_qgis.tif", "ndvi_2016.png",
              "NDVI 2016 — Vegetation Index", "QGIS · RdYlGn · −0.5 → 0.8")
save_qgis_rgb("NDBI_2012_qgis.tif", "ndbi_2012.png",
              "NDBI 2012 — Built-up Index", "QGIS · RdBu · −0.5 → 0.5")
save_qgis_rgb("NDBI_2016_qgis.tif", "ndbi_2016.png",
              "NDBI 2016 — Built-up Index", "QGIS · RdBu · −0.5 → 0.5")

# ================================================================
#  STEP 0 — Build t1_feature_stack.tif  &  t2_feature_stack.tif
#
#  Each TIF is a 5-band GeoTIFF:
#    Band 1 — DTM   (LiDAR Digital Terrain Model)
#    Band 2 — Slope (derived from DTM)
#    Band 3 — CHM   (Canopy Height Model)
#    Band 4 — NDVI  (Sentinel-2 vegetation index)
#    Band 5 — NDBI  (Sentinel-2 built-up index)
#
#  Strategy:
#    • Use the DTM as the reference grid (best spatial resolution, 1 m)
#    • Reproject/resample every other layer onto the DTM grid using
#      bilinear interpolation
#    • Write a Float32 GeoTIFF with nodata = -9999
# ================================================================
print("\n" + "="*60)
print("  STEP 0 — Building feature stack TIFs")
print("="*60)

BAND_SOURCES_T1 = [
    ("t1_dtm.tif",           1, "DTM"),
    ("t1_slope.tif",         1, "Slope"),
    ("t1_chm_turbo.tiff",  1, "CHM"),
    ("NDVI_2012_qgis.tif",   1, "NDVI"),
    ("NDBI_2012_qgis.tif",   1, "NDBI"),
]

BAND_SOURCES_T2 = [
    ("t2_dtm.tif",           1, "DTM"),
    ("t2_slope.tif",         1, "Slope"),
    ("t2_chm_turbo.tiff",  1, "CHM"),
    ("NDVI_2016_qgis.tif",   1, "NDVI"),
    ("NDBI_2016_qgis.tif",   1, "NDBI"),
]

NODATA_VAL = np.float32(-9999)


def build_feature_stack(band_sources, out_fname, label):
    """
    Reproject all layers onto the first layer's grid and write a 5-band TIF.

    Parameters
    ----------
    band_sources : list of (tif_filename, band_index, band_name)
    out_fname    : output filename inside TIF_DIR
    label        : human label for progress printing
    """
    out_path = tif(out_fname)
    if os.path.exists(out_path):
        print(f"   ℹ  {out_fname} already exists — skipping. Delete to regenerate.")
        return True

    # ── Check reference layer exists (Band 1 = DTM) ──
    ref_file = tif(band_sources[0][0])
    if not os.path.exists(ref_file):
        print(f"   ✗  Reference TIF not found: {band_sources[0][0]}")
        print(f"      Cannot build {out_fname} without the DTM layer.")
        return False

    # ── Read reference grid metadata ──
    with rasterio.open(ref_file) as ref:
        ref_crs       = ref.crs
        ref_transform = ref.transform
        ref_width     = ref.width
        ref_height    = ref.height
        ref_profile   = ref.profile.copy()

    ref_profile.update(
        count   = len(band_sources),
        dtype   = "float32",
        nodata  = float(NODATA_VAL),
        compress= "lzw",
        tiled   = True,
        blockxsize=256,
        blockysize=256,
        driver  = "GTiff",
    )

    stacked = []   # list of 2-D arrays in reference grid space

    for src_fname, src_band, bname in band_sources:
        src_path = tif(src_fname)

        # ── Missing source → fill with nodata ──
        if not os.path.exists(src_path):
            print(f"   ⚠  {src_fname} not found → filling band '{bname}' with nodata")
            arr_out = np.full((ref_height, ref_width), NODATA_VAL, dtype=np.float32)
            stacked.append(arr_out)
            continue

        with rasterio.open(src_path) as src:
            src_crs       = src.crs
            src_transform = src.transform
            src_nodata    = src.nodata if src.nodata is not None else NODATA_VAL
            src_data      = src.read(src_band).astype(np.float32)

        # Mask known sentinel nodata values
        src_data[src_data == -9999]  = np.nan
        src_data[src_data == -32768] = np.nan
        with np.errstate(invalid="ignore"):
            src_data[np.abs(src_data) > 1e6] = np.nan
        # Replace nan → nodata for rasterio
        src_nodata_f = float(NODATA_VAL)
        src_data_nd  = np.where(np.isnan(src_data), src_nodata_f, src_data)

        # ── Check if reprojection is needed ──
        same_grid = (
            src_crs    == ref_crs       and
            src_data.shape[0] == ref_height and
            src_data.shape[1] == ref_width  and
            np.allclose(np.array(src_transform)[:6],
                        np.array(ref_transform)[:6], atol=1e-6)
        )

        if same_grid:
            arr_out = src_data_nd
        else:
            print(f"      Reprojecting {src_fname} (band {src_band}) onto reference grid …")
            arr_out = np.full((ref_height, ref_width), src_nodata_f, dtype=np.float32)
            reproject(
                source      = src_data_nd,
                destination = arr_out,
                src_transform  = src_transform,
                src_crs        = src_crs,
                src_nodata     = src_nodata_f,
                dst_transform  = ref_transform,
                dst_crs        = ref_crs,
                dst_nodata     = src_nodata_f,
                resampling     = Resampling.bilinear,
            )

        stacked.append(arr_out.astype(np.float32))
        print(f"   ✓  Band '{bname}' from {src_fname}")

    # ── Write output TIF ──
    with rasterio.open(out_path, "w", **ref_profile) as dst:
        for i, arr in enumerate(stacked, start=1):
            dst.write(arr, i)
        # Write band descriptions
        for i, (_, _, bname) in enumerate(band_sources, start=1):
            dst.update_tags(i, name=bname)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\n   ✅  Written: {out_fname}  ({size_mb:.1f} MB, {len(stacked)} bands)")
    return True


ok1 = build_feature_stack(BAND_SOURCES_T1, "t1_feature_stack.tif", "T1 2012")
ok2 = build_feature_stack(BAND_SOURCES_T2, "t2_feature_stack.tif", "T2 2016")


# ── Helper: load a band from the stack TIFs ──
def load(fname, band=1):
    """Load one band from a TIF in TIF_DIR, returning a float32 array with NaN nodata."""
    p = tif(fname)
    if not os.path.exists(p): return None
    with rasterio.open(p) as src:
        data = src.read(band).astype(np.float32)
        nd   = src.nodata
    if nd is not None: data[data == nd] = np.nan
    for v in [-9999, -32768]: data[data == v] = np.nan
    with np.errstate(invalid="ignore"): data[np.abs(data) > 1e6] = np.nan
    return data


# ================================================================
#  STEP 1 — Feature stack band maps  (t1 + t2, bands 1–5)
# ================================================================
print("\n[1/5]  Feature stack — per-band PNG maps")

BAND_CFG = [
    (1, "DTM",   "Blues_r",  "Elevation (m)"),
    (2, "Slope", "YlOrRd",   "Slope (°)"),
    (3, "CHM",   "turbo",    "Canopy height (m)"),
    (4, "NDVI",  "RdYlGn",   "NDVI value"),
    (5, "NDBI",  "RdBu_r",   "NDBI value"),
]

# Shared fixed colour limits — ensures BOTH years always use identical scales
FIXED_RANGES = {
    "NDVI": (-0.5, 0.8),
    "NDBI": (-0.5, 0.5),
    "CHM":  (0.0, 20.0),   # metres — same scale 2012 and 2016
}

def shared_range(bname, *arrays):
    """Return (lo, hi) — fixed if in FIXED_RANGES, else pooled 2–98 pct."""
    if bname in FIXED_RANGES:
        return FIXED_RANGES[bname]
    combined = []
    for arr in arrays:
        if arr is not None:
            v = arr[~np.isnan(arr)]
            if v.size:
                combined.append(v)
    if combined:
        all_vals = np.concatenate(combined)
        return float(np.percentile(all_vals, 2)), float(np.percentile(all_vals, 98))
    return 0.0, 1.0

for prefix, year_label, stack_fname in [
    ("t1", "2012", "t1_feature_stack.tif"),
    ("t2", "2016", "t2_feature_stack.tif"),
]:
    if not os.path.exists(tif(stack_fname)):
        print(f"   --  SKIPPED (not found): {stack_fname}")
        continue

    # ── A. Individual band maps ──
    # For CHM (band 3): use QGIS turbo render directly as RGB
    chm_qgis_src = "t1_chm_turbo.tiff" if prefix == "t1" else "t2_chm_turbo.tiff"

    for band_idx, bname, cmap, clabel in BAND_CFG:

        # ── Band 3 CHM: save QGIS RGB render directly ──
        if band_idx == 3:
            rgb_chm = load_rgb_tif(chm_qgis_src)
            for out_name in [
                f"feat_{prefix}_band{band_idx}_{bname.lower()}.png",
                f"{prefix}_feature_stack_band{band_idx}.png",
            ]:
                if rgb_chm is not None:
                    fig, ax = plt.subplots(figsize=(7, 5))
                    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
                    ax.imshow(rgb_chm, interpolation="nearest")
                    ax.set_title(
                        f"Feature Stack {year_label} — Band {band_idx}: {bname}\n"
                        f"QGIS · turbo · 0–20 m",
                        color="white", fontsize=11, fontweight="bold", pad=7,
                    )
                    ax.tick_params(colors=TC, labelsize=6)
                    for sp in ax.spines.values(): sp.set_edgecolor(SC)
                    plt.tight_layout(pad=1.0)
                    save(fig, out_name)
                else:
                    print(f"   --  SKIPPED {out_name} ({chm_qgis_src} not found)")
            continue

        arr = load(stack_fname, band=band_idx)
        if arr is None:
            print(f"   --  {bname} band missing in {stack_fname}")
            continue
        lo, hi = shared_range(bname, arr)

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
        im = ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(
            f"Feature Stack {year_label} — Band {band_idx}: {bname}",
            color="white", fontsize=11, fontweight="bold", pad=7,
        )
        ax.tick_params(colors=TC, labelsize=6)
        for sp in ax.spines.values(): sp.set_edgecolor(SC)
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label(clabel, color=TC, fontsize=8)
        cb.ax.tick_params(colors=TC, labelsize=7)
        plt.tight_layout(pad=1.0)
        save(fig, f"feat_{prefix}_band{band_idx}_{bname.lower()}.png")

        # Duplicate as "t1/t2_feature_stack_bandN.png"
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        fig2.patch.set_facecolor(BG); ax2.set_facecolor(BG)
        arr2 = load(stack_fname, band=band_idx)
        im2  = ax2.imshow(arr2, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        ax2.set_title(
            f"{stack_fname.split('.')[0].upper()} — Band {band_idx}: {bname} ({year_label})",
            color="white", fontsize=11, fontweight="bold", pad=7,
        )
        ax2.tick_params(colors=TC, labelsize=6)
        for sp in ax2.spines.values(): sp.set_edgecolor(SC)
        cb2 = fig2.colorbar(im2, ax=ax2, fraction=0.03, pad=0.02)
        cb2.set_label(clabel, color=TC, fontsize=8)
        cb2.ax.tick_params(colors=TC, labelsize=7)
        plt.tight_layout(pad=1.0)
        save(fig2, f"{prefix}_feature_stack_band{band_idx}.png")

    # ── B. 5-band overview mosaic — uses QGIS renders for CHM/NDVI/NDBI ──
    # Map: which QGIS RGB source to use per band (None = use float stack)
    QGIS_RGB_SOURCES = {
        3: ("t1_chm_turbo.tiff"  if prefix=="t1" else "t2_chm_turbo.tiff",  "CHM · turbo · 0–20m"),
        4: ("NDVI_2012_qgis.tif" if prefix=="t1" else "NDVI_2016_qgis.tif", "NDVI · RdYlGn"),
        5: ("NDBI_2012_qgis.tif" if prefix=="t1" else "NDBI_2016_qgis.tif", "NDBI · RdBu"),
    }
    fig = plt.figure(figsize=(22, 5), facecolor=BG)
    fig.suptitle(
        f"Feature Fusion Stack — {year_label}  (5 bands: DTM · Slope · CHM · NDVI · NDBI)",
        color="white", fontsize=13, fontweight="bold", y=1.01,
    )
    gs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.08)
    for i, (band_idx, bname, cmap, clabel) in enumerate(BAND_CFG):
        ax  = fig.add_subplot(gs[0, i])
        ax.set_facecolor(BG)
        ax.set_title(f"B{band_idx}: {bname}", color="white", fontsize=10, fontweight="bold")
        ax.tick_params(colors=TC, labelsize=5)
        for sp in ax.spines.values(): sp.set_edgecolor(SC)

        if band_idx in QGIS_RGB_SOURCES:
            # Use QGIS-rendered RGB image directly
            qsrc, qlbl = QGIS_RGB_SOURCES[band_idx]
            rgb = load_rgb_tif(qsrc)
            if rgb is not None:
                ax.imshow(rgb, interpolation="nearest")
                ax.text(0.5, -0.08, qlbl, transform=ax.transAxes,
                        ha="center", fontsize=6, color=TC, fontfamily="monospace")
            else:
                ax.text(0.5, 0.5, "Missing", color=TC, ha="center",
                        va="center", transform=ax.transAxes, fontsize=9)
        else:
            # Float stack band — use matplotlib colormap
            arr = load(stack_fname, band=band_idx)
            lo, hi = shared_range(bname, arr) if arr is not None else (0, 1)
            if arr is not None:
                im = ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
                cb = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.03, orientation="horizontal")
                cb.ax.tick_params(colors=TC, labelsize=6)

    fig.patch.set_facecolor(BG)
    plt.tight_layout(pad=0.8)
    save(fig, f"feature_fusion_{prefix}_{year_label}.png")


# ================================================================
#  STEP 2 — Feature fusion comparison  (t1 vs t2 side-by-side)
#  Loads directly from original source TIFs (not the stacks) so
#  each band uses the true native data range before stacking.
#  Both panels share a SINGLE colorscale derived from the union
#  of all valid pixels across both epochs.
# ================================================================
print("\n[2/5]  Feature fusion comparison poster (band-by-band)")

# Map band index → (t1 source file, t2 source file)
# QGIS-exported TIFs (CHM/NDVI/NDBI) are RGB-rendered → load as image
# Scientific TIFs (DTM/Slope) are single-band float → load as array
ORIG_SOURCES = {
    1: ("t1_dtm.tif",            "t2_dtm.tif",           "float"),
    2: ("t1_slope.tif",          "t2_slope.tif",         "float"),
    3: ("t1_chm_turbo.tiff",   "t2_chm_turbo.tiff",  "rgb"),
    4: ("NDVI_2012_qgis.tif",    "NDVI_2016_qgis.tif",   "rgb"),
    5: ("NDBI_2012_qgis.tif",    "NDBI_2016_qgis.tif",   "rgb"),
}

for band_idx, bname, cmap, clabel in BAND_CFG:
    src1, src2, mode = ORIG_SOURCES[band_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"Feature Fusion Comparison — Band {band_idx}: {bname}  ({clabel})",
        color="white", fontsize=12, fontweight="bold",
    )

    if mode == "rgb":
        # ── QGIS rendered export — display as-is, no colormap needed ──
        imgs = [load_rgb_tif(src1), load_rgb_tif(src2)]
        for ax, img, yr in zip(axes, imgs, ["2012", "2016"]):
            ax.set_facecolor(BG)
            if img is None:
                ax.text(0.5, 0.5, "Not available", color=TC, ha="center",
                        va="center", transform=ax.transAxes, fontsize=11)
            else:
                ax.imshow(img, interpolation="nearest")
            ax.set_title(yr, color="white", fontsize=11, fontweight="bold")
            ax.tick_params(colors=TC, labelsize=6)
            for sp in ax.spines.values():
                sp.set_edgecolor(SC)
        # No colorbar for RGB renders — QGIS already baked the colours in
        fig.text(0.5, 0.01, f"Colour scale: QGIS export  ({clabel})",
                 ha="center", color=TC, fontsize=8, fontfamily="monospace")

    else:
        # ── Single-band float — scientific colormap with shared scale ──
        a1 = load_band(src1)
        a2 = load_band(src2)
        if a1 is None and a2 is None:
            print(f"   --  SKIPPED (both missing: {bname})")
            plt.close(fig)
            continue
        lo, hi = shared_range(bname, a1, a2)
        print(f"      Band {band_idx} {bname}: shared range [{lo:.3f}, {hi:.3f}]")

        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize as MNorm
        sm = ScalarMappable(cmap=cmap, norm=MNorm(vmin=lo, vmax=hi))
        sm.set_array([])

        for ax, arr, yr in zip(axes, [a1, a2], ["2012", "2016"]):
            ax.set_facecolor(BG)
            if arr is None:
                ax.text(0.5, 0.5, "Not available", color=TC, ha="center",
                        va="center", transform=ax.transAxes, fontsize=11)
                ax.set_title(yr, color="white", fontsize=11, fontweight="bold")
                continue
            ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_title(yr, color="white", fontsize=11, fontweight="bold")
            ax.tick_params(colors=TC, labelsize=6)
            for sp in ax.spines.values():
                sp.set_edgecolor(SC)

        cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
        cbar.set_label(clabel, color=TC, fontsize=9)
        cbar.ax.tick_params(colors=TC, labelsize=7)

    plt.tight_layout(pad=1.2)
    save(fig, f"fusion_compare_band{band_idx}_{bname.lower()}.png")


# ================================================================
#  STEP 3 — Feature stack RGB composite (false-colour overview)
#           Uses NDVI(B4), DTM(B1), Slope(B2) as R, G, B channels
# ================================================================
print("\n[3/5]  Feature stack false-colour RGB composites")

def make_rgb_composite(prefix, year_label, out_name):
    """Build a 5-panel composite using real QGIS-rendered images for all bands."""
    if prefix == "t1":
        sources = [
            ("t1_dtm.tif",           None,  "DTM",   "Blues_r",  "Elevation (m)"),
            ("t1_slope.tif",         None,  "Slope", "YlOrRd",   "Slope (°)"),
            ("t1_chm_turbo.tiff",  "rgb", "CHM",   None,       "turbo · 0–20 m"),
            ("NDVI_2012_qgis.tif",   "rgb", "NDVI",  None,       "RdYlGn · −0.5→0.8"),
            ("NDBI_2012_qgis.tif",   "rgb", "NDBI",  None,       "RdBu · −0.5→0.5"),
        ]
    else:
        sources = [
            ("t2_dtm.tif",           None,  "DTM",   "Blues_r",  "Elevation (m)"),
            ("t2_slope.tif",         None,  "Slope", "YlOrRd",   "Slope (°)"),
            ("t2_chm_turbo.tiff",  "rgb", "CHM",   None,       "turbo · 0–20 m"),
            ("NDVI_2016_qgis.tif",   "rgb", "NDVI",  None,       "RdYlGn · −0.5→0.8"),
            ("NDBI_2016_qgis.tif",   "rgb", "NDBI",  None,       "RdBu · −0.5→0.5"),
        ]

    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"Feature Stack — All 5 Bands · {year_label}  "
        f"(DTM · Slope · CHM [turbo] · NDVI [RdYlGn] · NDBI [RdBu])",
        color="white", fontsize=12, fontweight="bold", y=1.02,
    )

    for ax, (src_fname, mode, bname, cmap, clabel) in zip(axes, sources):
        ax.set_facecolor(BG)
        ax.set_title(f"{bname}", color="white", fontsize=11, fontweight="bold", pad=6)
        ax.tick_params(colors=TC, labelsize=5)
        for sp in ax.spines.values(): sp.set_edgecolor(SC)
        ax.text(0.5, -0.06, clabel, transform=ax.transAxes,
                ha="center", fontsize=7, color=TC, fontfamily="monospace")

        if mode == "rgb":
            rgb = load_rgb_tif(src_fname)
            if rgb is not None:
                ax.imshow(rgb, interpolation="nearest")
            else:
                ax.text(0.5, 0.5, f"Missing\n{src_fname}", color=TC, ha="center",
                        va="center", transform=ax.transAxes, fontsize=8, wrap=True)
        else:
            arr = load_band(src_fname)
            if arr is not None:
                lo, hi = shared_range(bname, arr)
                im = ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
                cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, orientation="horizontal")
                cb.ax.tick_params(colors=TC, labelsize=6)
            else:
                ax.text(0.5, 0.5, f"Missing\n{src_fname}", color=TC, ha="center",
                        va="center", transform=ax.transAxes, fontsize=8, wrap=True)

    plt.tight_layout(pad=1.0)
    save(fig, out_name)


make_rgb_composite("t1", "2012", "t1_feature_stack_rgb.png")
make_rgb_composite("t2", "2016", "t2_feature_stack_rgb.png")


# ================================================================
#  STEP 4 — Confusion matrices — all 3 periods
# ================================================================
print("\n[4/5]  Confusion matrices — all periods")

CL_NAMES  = ["Bare", "Grass", "Med Veg", "Dense Veg", "Built"]
CL_COLORS = ["#d4b483","#a8d08d","#4daf4a","#1b7837","#808080"]


def compute_confusion(from_fname, to_fname):
    a = load_band(from_fname)
    b = load_band(to_fname)
    if a is None or b is None: return None
    mask = ~(np.isnan(a) | np.isnan(b))
    f = a[mask].astype(int)
    t = b[mask].astype(int)
    cm = np.zeros((5, 5), dtype=np.int64)
    for i in range(5):
        for j in range(5):
            cm[i, j] = int(np.sum((f == i) & (t == j)))
    return cm


def plot_confusion(cm, title, outname):
    if cm is None:
        print(f"   --  SKIPPED (TIFs missing): {outname}")
        return
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        cm_pct = np.where(row_sums > 0, cm / row_sums * 100, 0)
    oa    = cm.diagonal().sum() / cm.sum() * 100
    pe    = sum(cm[i].sum() * cm[:, i].sum() for i in range(5)) / (cm.sum() ** 2)
    kappa = (cm.diagonal().sum() / cm.sum() - pe) / (1 - pe) if pe < 1 else 0

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor(BG)
    fig.suptitle(title, color="white", fontsize=13, fontweight="bold", y=1.01)

    # Raw counts
    ax = axes[0]
    ax.set_facecolor(BG)
    im = ax.imshow(np.log1p(cm), cmap="Blues", aspect="auto")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    fontsize=8, color="white" if cm[i,j] > cm.max() * 0.4 else TC,
                    fontfamily="monospace")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(CL_NAMES, color=TC, fontsize=8, rotation=20)
    ax.set_yticklabels(CL_NAMES, color=TC, fontsize=8)
    ax.set_xlabel("Predicted →", color=TC, fontsize=9)
    ax.set_ylabel("Actual (from) →", color=TC, fontsize=9)
    ax.set_title("Raw pixel counts", color="white", fontsize=10, fontweight="bold")
    for sp in ax.spines.values(): sp.set_edgecolor(SC)
    ax.tick_params(colors=TC)

    # Normalised %
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    im2 = ax2.imshow(cm_pct, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    for i in range(5):
        for j in range(5):
            ax2.text(j, i, f"{cm_pct[i,j]:.1f}%", ha="center", va="center",
                     fontsize=8, color="black" if cm_pct[i,j] > 50 else TC,
                     fontfamily="monospace")
    ax2.set_xticks(range(5)); ax2.set_yticks(range(5))
    ax2.set_xticklabels(CL_NAMES, color=TC, fontsize=8, rotation=20)
    ax2.set_yticklabels(CL_NAMES, color=TC, fontsize=8)
    ax2.set_xlabel("Predicted →", color=TC, fontsize=9)
    ax2.set_ylabel("Actual (from) →", color=TC, fontsize=9)
    ax2.set_title("Row-normalised (%)", color="white", fontsize=10, fontweight="bold")
    for sp in ax2.spines.values(): sp.set_edgecolor(SC)
    ax2.tick_params(colors=TC)
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.04, pad=0.03)
    cb2.ax.tick_params(colors=TC, labelsize=7)

    fig.text(
        0.5, -0.04,
        f"Overall Accuracy: {oa:.2f}%   |   Kappa: {kappa:.4f}   |   Total pixels: {cm.sum():,}",
        ha="center", color="white", fontsize=10, fontfamily="monospace",
        bbox=dict(facecolor=BG, edgecolor=TC, boxstyle="round,pad=0.4"),
    )
    plt.tight_layout(pad=1.4)
    save(fig, outname)


cm1 = compute_confusion("lulc_2012.tif", "lulc_2016.tif")
plot_confusion(cm1, "Confusion Matrix — 2012 → 2016  (Observed)",  "confusion_2012_2016.png")

cm2 = compute_confusion("lulc_2016.tif", "predicted_lulc_2020.tif")
plot_confusion(cm2, "Confusion Matrix — 2016 → 2020  (Predicted)", "confusion_2016_2020.png")

cm3 = compute_confusion("predicted_lulc_2020.tif", "predicted_lulc_2024.tif")
plot_confusion(cm3, "Confusion Matrix — 2020 → 2024  (Predicted)", "confusion_2020_2024.png")


# ================================================================
#  STEP 5 — LAZ / Point-cloud renders (synthetic from DTM + CHM)
# ================================================================
print("\n[5/5]  LiDAR point-cloud renders (synthetic from DTM + CHM)")


def render_laz(dtm_fname, chm_fname, year, prefix):
    dtm = load_band(dtm_fname)
    chm = load_band(chm_fname)
    if dtm is None:
        print(f"   --  SKIPPED ({dtm_fname} not found)")
        return

    step = 12
    rows = np.arange(0, dtm.shape[0], step)
    cols = np.arange(0, dtm.shape[1], step)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    zz   = dtm[rr, cc]; valid = ~np.isnan(zz)
    x    = cc[valid].ravel().astype(float)
    y    = rr[valid].ravel().astype(float)
    z    = zz[valid].ravel()

    if chm is not None:
        hh = chm[rr, cc][valid].ravel()
        hh = np.where(np.isnan(hh), 0, hh)
        hh = np.clip(hh, 0, 20)
        color_vals = hh / 20.0
        cmap_name  = "turbo"; cbar_label = "Canopy height (m)"
    else:
        lo, hi = np.percentile(z, 2), np.percentile(z, 98)
        color_vals = np.clip((z - lo) / max(hi - lo, 1), 0, 1)
        cmap_name  = "Blues_r"; cbar_label = "Elevation (m)"

    sizes = np.clip(0.2 + (z - z.min()) / max(float(z.max() - z.min()), 1) * 0.6, 0.1, 0.9)

    # ── A. Plan view ──
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BG); ax.set_facecolor("#040810")
    sc_plot = ax.scatter(x, y, c=color_vals, cmap=cmap_name,
                         s=sizes, linewidths=0, alpha=0.8, rasterized=True)
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_title(f"LiDAR Point Cloud — {year}  (plan view, {len(x):,} pts sampled)",
                 color="white", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Column →", color=TC, fontsize=8)
    ax.set_ylabel("Row →",    color=TC, fontsize=8)
    ax.tick_params(colors=TC, labelsize=6)
    for sp in ax.spines.values(): sp.set_edgecolor(SC)
    cb = fig.colorbar(sc_plot, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(cbar_label, color=TC, fontsize=8)
    cb.ax.tick_params(colors=TC, labelsize=7)
    plt.tight_layout(pad=1.0)
    save(fig, f"laz_planview_{prefix}_{year}.png")

    # ── B. 3-D perspective ──
    step3 = 24
    rows3 = np.arange(0, dtm.shape[0], step3)
    cols3 = np.arange(0, dtm.shape[1], step3)
    rr3, cc3 = np.meshgrid(rows3, cols3, indexing="ij")
    zz3 = dtm[rr3, cc3]; v3 = ~np.isnan(zz3)
    x3  = cc3[v3].ravel().astype(float)
    y3  = rr3[v3].ravel().astype(float)
    z3  = zz3[v3].ravel()
    if chm is not None:
        h3 = chm[rr3, cc3][v3].ravel()
        h3 = np.where(np.isnan(h3), 0, h3)
        cv3 = np.clip(h3, 0, 20) / 20.0
    else:
        lo3, hi3 = np.percentile(z3, 2), np.percentile(z3, 98)
        cv3 = np.clip((z3 - lo3) / max(hi3 - lo3, 1), 0, 1)

    fig3 = plt.figure(figsize=(11, 7))
    fig3.patch.set_facecolor(BG)
    ax3  = fig3.add_subplot(111, projection="3d")
    ax3.set_facecolor(BG)
    sc3  = ax3.scatter(x3, y3, z3, c=cv3, cmap=cmap_name,
                       s=0.3, linewidths=0, alpha=0.7, rasterized=True)
    ax3.set_xlabel("X (col)", color=TC, fontsize=7, labelpad=4)
    ax3.set_ylabel("Y (row)", color=TC, fontsize=7, labelpad=4)
    ax3.set_zlabel("Z (m)",   color=TC, fontsize=7, labelpad=4)
    ax3.tick_params(colors=TC, labelsize=5)
    ax3.xaxis.pane.fill = False; ax3.yaxis.pane.fill = False; ax3.zaxis.pane.fill = False
    ax3.xaxis.pane.set_edgecolor(SC); ax3.yaxis.pane.set_edgecolor(SC); ax3.zaxis.pane.set_edgecolor(SC)
    ax3.view_init(elev=28, azim=225)
    ax3.set_title(
        f"LiDAR 3-D Perspective — {year}  (coloured by {'canopy height' if chm is not None else 'elevation'})",
        color="white", fontsize=11, fontweight="bold", pad=10,
    )
    cb3 = fig3.colorbar(sc3, ax=ax3, fraction=0.025, pad=0.04, shrink=0.6)
    cb3.set_label(cbar_label, color=TC, fontsize=8)
    cb3.ax.tick_params(colors=TC, labelsize=7)
    plt.tight_layout(pad=1.0)
    save(fig3, f"laz_3d_{prefix}_{year}.png")

    # ── C. Cross-section profile ──
    mid_row     = dtm.shape[0] // 2
    profile     = dtm[mid_row, :]
    chm_profile = chm[mid_row, :] if chm is not None else None

    fig4, ax4 = plt.subplots(figsize=(12, 4))
    fig4.patch.set_facecolor(BG); ax4.set_facecolor("#040810")
    px = np.arange(len(profile))
    ax4.fill_between(px, 0, np.where(np.isnan(profile), 0, profile),
                     color="#4a90d9", alpha=0.7, label="Ground (DTM)")
    if chm_profile is not None:
        top = np.where(
            np.isnan(profile), np.nan,
            np.where(np.isnan(chm_profile), profile,
                     profile + np.clip(chm_profile, 0, 30))
        )
        ax4.fill_between(px, np.where(np.isnan(profile), 0, profile), top,
                         color="#1a9641", alpha=0.6, label="Canopy (CHM)")
    ax4.set_title(f"LiDAR Cross-Section Profile — {year}  (mid-row transect)",
                  color="white", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Column pixel →", color=TC, fontsize=8)
    ax4.set_ylabel("Elevation (m)",  color=TC, fontsize=8)
    ax4.tick_params(colors=TC, labelsize=7)
    for sp in ax4.spines.values(): sp.set_edgecolor(SC)
    ax4.legend(fontsize=8, labelcolor="white", facecolor=BG, edgecolor=SC, framealpha=0.6)
    ax4.grid(axis="y", color=SC, linewidth=0.4, alpha=0.5)
    plt.tight_layout(pad=1.0)
    save(fig4, f"laz_profile_{prefix}_{year}.png")


render_laz("t1_dtm.tif", "t1_chm_turbo.tiff", "2012", "t1")
render_laz("t2_dtm.tif", "t2_chm_turbo.tiff", "2016", "t2")

# ── D. Side-by-side comparison panel ──
print("   Building LAZ comparison panel …")
dtm1 = load_band("t1_dtm.tif"); dtm2 = load_band("t2_dtm.tif")
chm1 = load_band("t1_chm_turbo.tiff"); chm2 = load_band("2016_chm_turbo.tiff")

if dtm1 is not None and dtm2 is not None:
    step_c = 14
    fig5, axes5 = plt.subplots(1, 2, figsize=(18, 7))
    fig5.patch.set_facecolor(BG)
    fig5.suptitle("LiDAR Point Cloud Comparison — 2012 vs 2016  (coloured by canopy height · turbo)",
                  color="white", fontsize=13, fontweight="bold")

    for ax5, dtm, chm, yr in zip(axes5, [dtm1, dtm2], [chm1, chm2], ["2012", "2016"]):
        ax5.set_facecolor("#040810")
        rows5 = np.arange(0, dtm.shape[0], step_c)
        cols5 = np.arange(0, dtm.shape[1], step_c)
        rr5, cc5 = np.meshgrid(rows5, cols5, indexing="ij")
        zz5  = dtm[rr5, cc5]; v5 = ~np.isnan(zz5)
        x5   = cc5[v5].ravel().astype(float)
        y5   = rr5[v5].ravel().astype(float)
        if chm is not None:
            h5  = chm[rr5, cc5][v5].ravel()
            h5  = np.where(np.isnan(h5), 0, np.clip(h5, 0, 20))
            cv5 = h5 / 20.0; cm5 = "turbo"
        else:
            z5  = zz5[v5].ravel()
            cv5 = np.clip((z5 - np.nanpercentile(z5, 2)) /
                          max(np.nanpercentile(z5, 98) - np.nanpercentile(z5, 2), 1), 0, 1)
            cm5 = "Blues_r"
        sc5 = ax5.scatter(x5, y5, c=cv5, cmap=cm5, s=0.35,
                          linewidths=0, alpha=0.85, rasterized=True)
        ax5.set_aspect("equal"); ax5.invert_yaxis()
        ax5.set_title(f"{yr}  ({len(x5):,} pts)", color="white", fontsize=11, fontweight="bold")
        ax5.tick_params(colors=TC, labelsize=5)
        for sp in ax5.spines.values(): sp.set_edgecolor(SC)
        cb5 = fig5.colorbar(sc5, ax=ax5, fraction=0.025, pad=0.02)
        cb5.set_label("Canopy height (m)  [0–20m]", color=TC, fontsize=7)
        cb5.ax.tick_params(colors=TC, labelsize=6)

    plt.tight_layout(pad=1.2)
    save(fig5, "laz_compare_2012_2016.png")

# ================================================================
print("\n" + "="*60)
print("  ✅  All steps complete!")
print(f"  Output folder : {IMG_DIR}")
print(f"  TIF folder    : {TIF_DIR}")
print()
print("  Files generated:")
print("    t1_feature_stack.tif  (5-band: DTM/Slope/CHM/NDVI/NDBI)")
print("    t2_feature_stack.tif  (5-band: DTM/Slope/CHM/NDVI/NDBI)")
print("    + all dashboard PNG images")
print()
print("  Next step:  python build_dashboard.py")
print("="*60 + "\n")