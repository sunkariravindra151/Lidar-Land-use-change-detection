import os
import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling

BASE_DIR = r"C:\Users\Sunkari Ravindra\OneDrive\Desktop\My Documents\QGIS project"

DTM_2012 = os.path.join(BASE_DIR, "data", "dtm_2012.tif")
DTM_2016 = os.path.join(BASE_DIR, "data", "dtm_2016.tif")

OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

ALIGNED_2016 = os.path.join(OUT_DIR, "dtm_2016_aligned.tif")

print("Loading 2012 raster as reference...")

with rasterio.open(DTM_2012) as ref:
    ref_data = ref.read(1)
    ref_meta = ref.meta.copy()
    ref_transform = ref.transform
    ref_crs = ref.crs

print("Resampling 2016 to match 2012 grid...")

with rasterio.open(DTM_2016) as src:

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

with rasterio.open(ALIGNED_2016, "w", **ref_meta) as dst:
    dst.write(aligned, 1)

print("Alignment complete.")
print("Saved:", ALIGNED_2016)