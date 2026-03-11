import os
import rasterio

BASE_DIR = r"C:\Users\Sunkari Ravindra\OneDrive\Desktop\My Documents\QGIS project"
OUT_DIR = os.path.join(BASE_DIR, "outputs")

files = [
    "t1_dtm.tif",
    "t1_dsm.tif",
    "t1_chm.tif",
    "t1_slope.tif",
    "t2_dtm.tif",
    "t2_dsm.tif",
    "t2_chm.tif",
    "t2_slope.tif"
]

for f in files:

    path = os.path.join(OUT_DIR, f)

    with rasterio.open(path, "r+") as src:
        src.crs = "EPSG:32760"

    print("CRS fixed:", f)

print("\nAll LiDAR rasters now use EPSG:32760")