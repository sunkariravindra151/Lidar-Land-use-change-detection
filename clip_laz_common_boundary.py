import os
from whitebox import WhiteboxTools

BASE_DIR = r"C:\Users\Sunkari Ravindra\OneDrive\Desktop\My Documents\QGIS project"

LAZ1 = os.path.join(BASE_DIR, "data", "points (2).laz")
LAZ2 = os.path.join(BASE_DIR, "data", "points (3).laz")

BOUNDARY = os.path.join(BASE_DIR, "outputs", "common_boundary.shp")
OUT1 = os.path.join(BASE_DIR, "outputs", "points2012_clipped.laz")
OUT2 = os.path.join(BASE_DIR, "outputs", "points2016_clipped.laz")

wbt = WhiteboxTools()
wbt.set_working_dir(os.path.join(BASE_DIR, "outputs"))

print("Clipping 2016 LiDAR...")
wbt.clip_lidar_to_polygon(
    i=LAZ2,
    polygons=BOUNDARY,
    output=OUT2
)

print("Done.")