import os
import rasterio
import numpy as np

BASE_DIR = r"C:\Users\Sunkari Ravindra\OneDrive\Desktop\My Documents\QGIS project"

DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")

PRE = os.path.join(DATA_DIR, "Kaikoura_Pre_exact.tif")
POST = os.path.join(DATA_DIR, "Kaikoura_POST_exact.tif")

os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------
# NDVI
# ----------------------------------

def compute_ndvi(image, output):

    with rasterio.open(image) as src:

        red = src.read(3).astype(float)   # SR_B4
        nir = src.read(4).astype(float)   # SR_B5

        meta = src.meta.copy()

    ndvi = (nir - red) / (nir + red + 1e-10)

    meta.update(dtype=rasterio.float32, count=1)

    with rasterio.open(output, "w", **meta) as dst:
        dst.write(ndvi.astype(np.float32), 1)

    print("Saved:", output)


# ----------------------------------
# NDBI
# ----------------------------------

def compute_ndbi(image, output):

    with rasterio.open(image) as src:

        nir = src.read(4).astype(float)   # SR_B5
        swir = src.read(5).astype(float)  # SR_B6

        meta = src.meta.copy()

    ndbi = (swir - nir) / (swir + nir + 1e-10)

    meta.update(dtype=rasterio.float32, count=1)

    with rasterio.open(output, "w", **meta) as dst:
        dst.write(ndbi.astype(np.float32), 1)

    print("Saved:", output)


# ----------------------------------
# RUN
# ----------------------------------

compute_ndvi(PRE, os.path.join(OUT_DIR, "t1_ndvi.tif"))
compute_ndbi(PRE, os.path.join(OUT_DIR, "t1_ndbi.tif"))

compute_ndvi(POST, os.path.join(OUT_DIR, "t2_ndvi.tif"))
compute_ndbi(POST, os.path.join(OUT_DIR, "t2_ndbi.tif"))

print("\nSpectral feature extraction complete.")