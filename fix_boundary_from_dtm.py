import os
import rasterio
import numpy as np

BASE_DIR = r"C:\Users\Sunkari Ravindra\OneDrive\Desktop\My Documents\QGIS project"

DTM1 = os.path.join(BASE_DIR, "data", "dtm_2012.tif")
DTM2 = os.path.join(BASE_DIR, "data", "dtm_2016.tif")

OUT = os.path.join(BASE_DIR, "outputs", "common_mask.tif")

os.makedirs(os.path.join(BASE_DIR,"outputs"), exist_ok=True)

print("Reading DTMs...")

with rasterio.open(DTM1) as src1:
    dtm1 = src1.read(1)
    meta = src1.meta
    nodata1 = src1.nodata

with rasterio.open(DTM2) as src2:
    dtm2 = src2.read(1)
    nodata2 = src2.nodata

print("Creating overlap mask...")

mask1 = dtm1 != nodata1
mask2 = dtm2 != nodata2

common_mask = mask1 & mask2

meta.update(dtype=rasterio.uint8)

with rasterio.open(OUT, "w", **meta) as dst:
    dst.write(common_mask.astype(np.uint8), 1)

print("Saved:", OUT)