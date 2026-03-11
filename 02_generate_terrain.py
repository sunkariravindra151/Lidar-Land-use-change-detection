import os
import rasterio
import numpy as np
from whitebox import WhiteboxTools

# ---------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------
BASE_DIR = r"C:\Users\Sunkari Ravindra\OneDrive\Desktop\My Documents\QGIS project"

OUTDIR = os.path.join(BASE_DIR, "outputs")

LAZ_T1 = os.path.join(OUTDIR, "points2012_clipped.laz")
LAZ_T2 = os.path.join(OUTDIR, "points2016_clipped.laz")

RESOLUTION = 1.0

os.makedirs(OUTDIR, exist_ok=True)

wbt = WhiteboxTools()
wbt.set_working_dir(OUTDIR)
wbt.set_verbose_mode(True)

# ---------------------------------------------------
# FUNCTION
# ---------------------------------------------------
def generate_terrain(laz_file, prefix):

    print(f"\nProcessing {prefix}")

    dtm = os.path.join(OUTDIR, f"{prefix}_dtm.tif")
    dsm = os.path.join(OUTDIR, f"{prefix}_dsm.tif")
    chm = os.path.join(OUTDIR, f"{prefix}_chm.tif")
    slope = os.path.join(OUTDIR, f"{prefix}_slope.tif")

    # --- DTM ---
    print("Generating DTM...")
    wbt.lidar_tin_gridding(
        i=laz_file,
        output=dtm,
        parameter="elevation",
        returns="ground",
        resolution=RESOLUTION
    )

    # --- DSM ---
    print("Generating DSM...")
    wbt.lidar_tin_gridding(
        i=laz_file,
        output=dsm,
        parameter="elevation",
        returns="first",
        resolution=RESOLUTION
    )

    # --- CHM ---
    print("Generating CHM...")
    with rasterio.open(dsm) as ds, rasterio.open(dtm) as dt:

        dsm_data = ds.read(1).astype(float)
        dtm_data = dt.read(1).astype(float)

        meta = ds.meta.copy()

        chm_data = dsm_data - dtm_data

    with rasterio.open(chm, "w", **meta) as dst:
        dst.write(chm_data.astype(np.float32), 1)

    # --- SLOPE ---
    # --- SLOPE ---
    print("Calculating slope...")
    wbt.slope(
        dem=dtm,
        output=slope
    )

    print(f"Finished {prefix}")

# ---------------------------------------------------
# RUN FOR BOTH YEARS
# ---------------------------------------------------

generate_terrain(LAZ_T1, "t1")
generate_terrain(LAZ_T2, "t2")

print("\nTerrain generation complete.")