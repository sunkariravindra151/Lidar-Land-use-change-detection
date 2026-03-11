"""
step1_fix_gee_offset.py
-----------------------
Fixes the GEE false easting offset on NDVI/NDBI files and
reprojects them to match the DTM reference grid.

Run this FIRST before step2_build_feature_stacks.py
"""

import os
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling

BASE_DIR = r"C:\Users\Sunkari Ravindra\OneDrive\Desktop\My Documents\QGIS project"
OUT_DIR  = os.path.join(BASE_DIR, "outputs")

# ------------------------------------------------
# Reference grid from DTM
# ------------------------------------------------
ref_path = os.path.join(OUT_DIR, "t2_dtm.tif")
with rasterio.open(ref_path) as ref:
    ref_transform = ref.transform
    ref_shape     = (ref.height, ref.width)
    ref_crs       = ref.crs
    ref_bounds    = ref.bounds

print("Reference DTM:")
print(f"  CRS    : {ref_crs}")
print(f"  Shape  : {ref_shape}")
print(f"  Bounds : {ref_bounds}\n")

# ------------------------------------------------
# Compute GEE easting/northing offset
# ------------------------------------------------
ndvi_sample = os.path.join(OUT_DIR, "NDVI_2016_aligned.tif")
with rasterio.open(ndvi_sample) as src:
    ndvi_bounds = src.bounds

easting_offset  = ref_bounds.left   - ndvi_bounds.left
northing_offset = ref_bounds.bottom - ndvi_bounds.bottom

print(f"GEE offset → easting: {easting_offset:+.2f} m  northing: {northing_offset:+.2f} m\n")

# ------------------------------------------------
# Fix and reproject a single NDVI/NDBI file
# ------------------------------------------------

def fix_gee_and_reproject(src_path, out_path, easting_off, northing_off):
    with rasterio.open(src_path) as src:
        data       = src.read(1).astype(np.float32)
        old_tf     = src.transform
        src_nodata = src.nodata

    # Replace source nodata with NaN before reprojecting
    if src_nodata is not None:
        data[data == src_nodata] = np.nan

    corrected_tf = Affine(
        old_tf.a, old_tf.b, old_tf.c + easting_off,
        old_tf.d, old_tf.e, old_tf.f + northing_off,
    )

    ref_height, ref_width = ref_shape
    dest = np.full((ref_height, ref_width), fill_value=np.nan, dtype=np.float32)

    reproject(
        source=data,
        destination=dest,
        src_transform=corrected_tf,
        src_crs=ref_crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    # Convert NaN → -9999 for output
    dest_out = np.where(np.isnan(dest), -9999, dest)
    valid_pct = np.sum(dest_out != -9999) / dest_out.size * 100

    out_meta = {
        "driver"   : "GTiff",
        "crs"      : ref_crs,
        "transform": ref_transform,
        "width"    : ref_width,
        "height"   : ref_height,
        "dtype"    : "float32",
        "nodata"   : -9999,
        "count"    : 1,
    }

    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(dest_out, 1)

    print(f"  {os.path.basename(src_path):<30} valid: {valid_pct:.1f}%  → {os.path.basename(out_path)}")


# ------------------------------------------------
# Fix all four NDVI/NDBI files
# ------------------------------------------------

print("Fixing GEE offset on NDVI/NDBI files...")

pairs = [
    ("NDVI_2012_aligned.tif", "NDVI_2012_fixed.tif"),
    ("NDBI_2012_aligned.tif", "NDBI_2012_fixed.tif"),
    ("NDVI_2016_aligned.tif", "NDVI_2016_fixed.tif"),
    ("NDBI_2016_aligned.tif", "NDBI_2016_fixed.tif"),
]

for src_name, out_name in pairs:
    fix_gee_and_reproject(
        src_path    =os.path.join(OUT_DIR, src_name),
        out_path    =os.path.join(OUT_DIR, out_name),
        easting_off =easting_offset,
        northing_off=northing_offset,
    )

print("\nStep 1 complete. Now run step2_build_feature_stacks.py")