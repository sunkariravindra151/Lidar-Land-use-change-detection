import os
import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling

BASE_DIR = r"C:\Users\Sunkari Ravindra\OneDrive\Desktop\My Documents\QGIS project"

DTM1 = os.path.join(BASE_DIR, "data", "dtm_2012.tif")
DTM2 = os.path.join(BASE_DIR, "data", "dtm_2016.tif")

OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

DTM2_ALIGNED = os.path.join(OUT_DIR, "dtm_2016_aligned.tif")
DTM1_CROP = os.path.join(OUT_DIR, "dtm_2012_common.tif")
DTM2_CROP = os.path.join(OUT_DIR, "dtm_2016_common.tif")

print("Loading DTM 2012 as reference...")

with rasterio.open(DTM1) as ref:
    ref_data = ref.read(1)
    ref_meta = ref.meta.copy()
    ref_transform = ref.transform
    ref_crs = ref.crs
    nodata = ref.nodata

print("Aligning DTM 2016 to 2012 grid...")

with rasterio.open(DTM2) as src:

    aligned = np.zeros_like(ref_data, dtype=np.float32)

    reproject(
        source=rasterio.band(src, 1),
        destination=aligned,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.bilinear
    )

ref_meta.update(dtype=rasterio.float32)

with rasterio.open(DTM2_ALIGNED, "w", **ref_meta) as dst:
    dst.write(aligned, 1)

print("Finding common valid area...")

mask1 = ref_data != nodata
mask2 = aligned != nodata

common_mask = mask1 & mask2

dtm1_common = np.where(common_mask, ref_data, nodata)
dtm2_common = np.where(common_mask, aligned, nodata)

with rasterio.open(DTM1_CROP, "w", **ref_meta) as dst:
    dst.write(dtm1_common.astype(np.float32), 1)

with rasterio.open(DTM2_CROP, "w", **ref_meta) as dst:
    dst.write(dtm2_common.astype(np.float32), 1)

print("Done.")
print("Saved:")
print(DTM1_CROP)
print(DTM2_CROP)