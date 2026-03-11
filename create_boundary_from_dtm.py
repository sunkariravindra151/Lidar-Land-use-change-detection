import os
import rasterio
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape

BASE_DIR = r"C:\Users\Sunkari Ravindra\OneDrive\Desktop\My Documents\QGIS project"

COMMON_DTM = os.path.join(BASE_DIR, "outputs", "dtm_2012_common.tif")
BOUNDARY = os.path.join(BASE_DIR, "outputs", "common_boundary.geojson")

with rasterio.open(COMMON_DTM) as src:
    data = src.read(1)
    mask = data != src.nodata
    transform = src.transform
    crs = src.crs

results = (
    {"geometry": shape(geom), "properties": {"value": val}}
    for geom, val in shapes(mask.astype("uint8"), mask=mask, transform=transform)
)

geoms = [r["geometry"] for r in results if r["properties"]["value"] == 1]

gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
boundary = gdf.dissolve()

boundary.to_file(BOUNDARY, driver="GeoJSON")

print("Boundary created:", BOUNDARY)