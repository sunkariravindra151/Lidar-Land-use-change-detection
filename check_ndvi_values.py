import rasterio
import numpy as np

path = r"C:\Users\Sunkari Ravindra\OneDrive\Desktop\My Documents\QGIS project\outputs\t1_ndvi.tif"

with rasterio.open(path) as src:
    data = src.read(1)

print("min:", np.nanmin(data))
print("max:", np.nanmax(data))