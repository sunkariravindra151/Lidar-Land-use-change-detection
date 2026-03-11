"""
step2_build_feature_stacks.py
------------------------------
Builds t1 (2012) and t2 (2016) 5-band feature stacks by combining:
  Band 1 → DTM
  Band 2 → Slope
  Band 3 → CHM
  Band 4 → NDVI  (GEE-fixed from step1)
  Band 5 → NDBI  (GEE-fixed from step1)

Run AFTER step1_fix_gee_offset.py
"""

import os
import numpy as np
import rasterio

BASE_DIR = r"C:\Users\Sunkari Ravindra\OneDrive\Desktop\My Documents\QGIS project"
OUT_DIR  = os.path.join(BASE_DIR, "outputs")

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
# Build and save a 5-band feature stack
# ------------------------------------------------

def build_stack(lidar_files, spectral_files, output_path):
    BAND_NAMES = ["DTM", "Slope", "CHM", "NDVI", "NDBI"]
    arrays = []

    print("Reading LiDAR bands:")
    for f in lidar_files:
        arrays.append(read_lidar(f))

    print("Reading spectral bands (fixed):")
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

    print(f"\nSaved : {output_path}")
    print(f"Shape : {stack.shape}  (bands × height × width)")
    for i, (arr, name) in enumerate(zip(arrays, BAND_NAMES)):
        valid = np.sum(arr != -9999) / arr.size * 100
        mn    = arr[arr != -9999].min()  if np.any(arr != -9999) else float("nan")
        mx    = arr[arr != -9999].max()  if np.any(arr != -9999) else float("nan")
        mean  = arr[arr != -9999].mean() if np.any(arr != -9999) else float("nan")
        print(f"  Band {i+1} ({name:<5}): {valid:5.1f}% valid  |  min={mn:.3f}  max={mx:.3f}  mean={mean:.3f}")


# ------------------------------------------------
# Build t1 feature stack (2012)
# ------------------------------------------------

print("--- Building t1 feature stack (2012) ---")
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

# ------------------------------------------------
# Build t2 feature stack (2016)
# ------------------------------------------------

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