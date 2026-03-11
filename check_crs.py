import rasterio
import os

OUT_DIR = r"C:\Users\Sunkari Ravindra\OneDrive\Desktop\My Documents\QGIS project\outputs"

files = {
    "t2_dtm":  "t2_dtm.tif",
    "t2_slope": "t2_slope.tif",
    "t2_chm":  "t2_chm.tif",
    "NDVI_2016": "NDVI_2016_aligned.tif",
    "NDBI_2016": "NDBI_2016_aligned.tif",
}

for name, fname in files.items():
    path = os.path.join(OUT_DIR, fname)
    with rasterio.open(path) as src:
        print(f"\n{name}")
        print(f"  CRS     : {src.crs}")
        print(f"  Size    : {src.width} x {src.height}")
        print(f"  Bounds  : {src.bounds}")
        print(f"  NoData  : {src.nodata}")