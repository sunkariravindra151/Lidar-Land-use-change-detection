import os
import rasterio

BASE_DIR = r"C:\Users\Sunkari Ravindra\OneDrive\Desktop\My Documents\QGIS project"

OUT_DIR = os.path.join(BASE_DIR, "outputs")

FILES = [
    "t1_ndvi.tif",
    "t1_ndbi.tif",
    "t2_ndvi.tif",
    "t2_ndbi.tif"
]

for f in FILES:

    path = os.path.join(OUT_DIR, f)

    with rasterio.open(path, "r+") as src:
        src.crs = "EPSG:32760"

    print("CRS fixed:", f)

print("\nAll spectral rasters now have CRS.")