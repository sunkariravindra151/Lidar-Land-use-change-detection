"""
fix_gee_and_fuse_v2.py
----------------------
Fixes two issues from v1:
  1. DTM/Slope nodata (-32768) was leaking in as valid data
  2. NDVI/NDBI only partially overlapping (~41%) — we report this
     but it is expected if the satellite scene doesn't fully cover
     the LiDAR extent. Pixels outside NDVI coverage are set to -9999.
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
# Compute GEE easting/northing offset from t1 files
# (use t1 since we're fixing both t1 and t2)
# ------------------------------------------------
ndvi_sample = os.path.join(OUT_DIR, "NDVI_2016_aligned.tif")
with rasterio.open(ndvi_sample) as src:
    ndvi_bounds = src.bounds

easting_offset  = ref_bounds.left   - ndvi_bounds.left
northing_offset = ref_bounds.bottom - ndvi_bounds.bottom

print(f"GEE offset  → easting: {easting_offset:+.2f} m  northing: {northing_offset:+.2f} m\n")

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

    print(f"  {os.path.basename(src_path):<30} valid: {valid_pct:.1f}%")
    return dest_out


# ------------------------------------------------
# Read a LiDAR raster and clean its nodata (-32768)
# ------------------------------------------------

def read_lidar(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nd   = src.nodata  # typically -32768

    if nd is not None:
        data[data == nd] = -9999

    valid_pct = np.sum(data != -9999) / data.size * 100
    print(f"  {os.path.basename(path):<30} valid: {valid_pct:.1f}%")
    return data


# ------------------------------------------------
# Fix NDVI/NDBI files
# ------------------------------------------------

print("Fixing GEE offset on NDVI/NDBI...")
ndvi_ndbi_pairs = [
    ("NDVI_2012_aligned.tif", "NDVI_2012_fixed.tif"),
    ("NDBI_2012_aligned.tif", "NDBI_2012_fixed.tif"),
    ("NDVI_2016_aligned.tif", "NDVI_2016_fixed.tif"),
    ("NDBI_2016_aligned.tif", "NDBI_2016_fixed.tif"),
]

for src_name, out_name in ndvi_ndbi_pairs:
    fix_gee_and_reproject(
        src_path    =os.path.join(OUT_DIR, src_name),
        out_path    =os.path.join(OUT_DIR, out_name),
        easting_off =easting_offset,
        northing_off=northing_offset,
    )

# ------------------------------------------------
# Build feature stacks
# ------------------------------------------------

def build_stack(lidar_files, spectral_files, output_path):
    BAND_NAMES = ["DTM", "Slope", "CHM", "NDVI", "NDBI"]
    arrays = []

    print(f"\nReading LiDAR bands:")
    for f in lidar_files:
        arrays.append(read_lidar(f))

    print(f"Reading spectral bands (fixed):")
    for f in spectral_files:
        with rasterio.open(f) as src:
            arr = src.read(1).astype(np.float32)
        valid = np.sum(arr != -9999) / arr.size * 100
        print(f"  {os.path.basename(f):<30} valid: {valid:.1f}%")
        arrays.append(arr)

    stack = np.stack(arrays)

    # Use first LiDAR file meta as base
    with rasterio.open(lidar_files[0]) as ref:
        meta = ref.meta.copy()

    meta.update({
        "count" : len(arrays),
        "dtype" : "float32",
        "nodata": -9999,
    })

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(stack)

    print(f"\nSaved: {output_path}")
    print(f"Shape: {stack.shape}  (bands × height × width)")
    for i, (arr, name) in enumerate(zip(arrays, BAND_NAMES)):
        mn   = arr[arr != -9999].min() if np.any(arr != -9999) else float("nan")
        mx   = arr[arr != -9999].max() if np.any(arr != -9999) else float("nan")
        mean = arr[arr != -9999].mean() if np.any(arr != -9999) else float("nan")
        valid = np.sum(arr != -9999) / arr.size * 100
        print(f"  Band {i+1} ({name:<5}): {valid:5.1f}% valid  |  min={mn:.3f}  max={mx:.3f}  mean={mean:.3f}")


print("\n--- Building t1 feature stack (2012) ---")
build_stack(
    lidar_files=[
        os.path.join(OUT_DIR, "t1_dtm.tif"),
        os.path.join(OUT_DIR, "t1_slope.tif"),
        os.path.join(OUT_DIR, "t1_chm.tif"),
    ],
    spectral_files=[
        os.path.join(OUT_DIR, "NDVI_2012_fixed.tif"),
        os.path.join(OUT_DIR, "NDBI_2012_fixed.tif"),
    ],
    output_path=os.path.join(OUT_DIR, "t1_feature_stack.tif"),
)

print("\n--- Building t2 feature stack (2016) ---")
build_stack(
    lidar_files=[
        os.path.join(OUT_DIR, "t2_dtm.tif"),
        os.path.join(OUT_DIR, "t2_slope.tif"),
        os.path.join(OUT_DIR, "t2_chm.tif"),
    ],
    spectral_files=[
        os.path.join(OUT_DIR, "NDVI_2016_fixed.tif"),
        os.path.join(OUT_DIR, "NDBI_2016_fixed.tif"),
    ],
    output_path=os.path.join(OUT_DIR, "t2_feature_stack.tif"),
)

print("\nFeature fusion complete.")