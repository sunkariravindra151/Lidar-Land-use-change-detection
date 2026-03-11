import os
import rasterio

BASE_DIR = r"C:\Users\Sunkari Ravindra\OneDrive\Desktop\My Documents\QGIS project"
OUT_DIR = os.path.join(BASE_DIR, "outputs")

print("\nRaster size check\n")

for file in os.listdir(OUT_DIR):
    if file.endswith(".tif"):
        path = os.path.join(OUT_DIR, file)

        with rasterio.open(path) as src:
            print(f"{file:30s} {src.height} x {src.width}")