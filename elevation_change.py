import os
import rasterio
import numpy as np

BASE_DIR = r"C:\Users\Sunkari Ravindra\OneDrive\Desktop\My Documents\QGIS project"

DTM_2012 = os.path.join(BASE_DIR, "data", "dtm_2012.tif")
DTM_2016 = os.path.join(BASE_DIR, "outputs", "dtm_2016_aligned.tif")

OUT = os.path.join(BASE_DIR, "outputs", "elevation_change.tif")

with rasterio.open(DTM_2012) as src1:
    dtm1 = src1.read(1)
    meta = src1.meta

with rasterio.open(DTM_2016) as src2:
    dtm2 = src2.read(1)

change = dtm2 - dtm1

with rasterio.open(OUT, "w", **meta) as dst:
    dst.write(change.astype("float32"), 1)

print("Saved:", OUT)