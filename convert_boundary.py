import geopandas as gpd
import os

BASE_DIR = r"C:\Users\Sunkari Ravindra\OneDrive\Desktop\My Documents\QGIS project"

geojson = os.path.join(BASE_DIR, "outputs", "common_boundary.geojson")
shp = os.path.join(BASE_DIR, "outputs", "common_boundary.shp")

print("Reading GeoJSON...")
gdf = gpd.read_file(geojson)

print("Saving Shapefile...")
gdf.to_file(shp)

print("Done:", shp)