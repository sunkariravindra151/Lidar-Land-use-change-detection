"""
05_knn_classification.py
------------------------
Unsupervised LULC classification using KMeans clustering.

Reshapes each feature stack into (pixels x features) table,
normalises bands, clusters into 5 land-cover classes, then
saves back as a classified raster.

Classes (auto-assigned by NDVI rank):
  0 → Bare ground
  1 → Grass / low vegetation
  2 → Medium vegetation
  3 → Dense vegetation
  4 → Built / rock

Outputs:
  lulc_2012.tif
  lulc_2016.tif
"""

import os
import numpy as np
import rasterio
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE_DIR = r"C:\Users\Sunkari Ravindra\OneDrive\Desktop\My Documents\QGIS project"
OUT_DIR  = os.path.join(BASE_DIR, "outputs")

N_CLASSES   = 5
NODATA_VAL  = -9999
RANDOM_SEED = 42

BAND_NAMES  = ["DTM", "Slope", "CHM", "NDVI", "NDBI"]
CLASS_NAMES = {
    0: "Bare ground",
    1: "Grass / low veg",
    2: "Medium vegetation",
    3: "Dense vegetation",
    4: "Built / rock",
}

# ------------------------------------------------
# Remap raw cluster IDs to semantic classes
# sorted by NDVI mean ascending
# ------------------------------------------------

def map_clusters_to_classes(labels, X_valid_raw, n_classes):
    ndvi_idx = 3

    cluster_ndvi = {}
    for c in range(n_classes):
        mask_c = labels == c
        cluster_ndvi[c] = X_valid_raw[mask_c, ndvi_idx].mean() if mask_c.sum() > 0 else -999

    sorted_clusters = sorted(cluster_ndvi.keys(), key=lambda c: cluster_ndvi[c])
    remap = {old: new for new, old in enumerate(sorted_clusters)}

    print("\n  Cluster → Class mapping (sorted by NDVI mean):")
    print(f"  {'Cluster':>8} {'-> Class':>8} {'NDVI mean':>10} {'CHM mean':>10}")
    for old, new in remap.items():
        mask_c = labels == old
        ndvi_m = X_valid_raw[mask_c, 3].mean() if mask_c.sum() > 0 else float("nan")
        chm_m  = X_valid_raw[mask_c, 2].mean() if mask_c.sum() > 0 else float("nan")
        print(f"  {old:>8} {'-> ' + str(new):>8} {ndvi_m:>10.3f} {chm_m:>10.3f}")

    return np.array([remap[l] for l in labels], dtype=np.int16)


# ------------------------------------------------
# Step 1: Reshape raster to pixel table
# Step 2: Normalise, KMeans, remap, save
# ------------------------------------------------

def classify_stack(stack_path, out_path, n_classes=N_CLASSES):
    print(f"\nLoading: {os.path.basename(stack_path)}")

    with rasterio.open(stack_path) as src:
        stack = src.read().astype(np.float32)   # (bands, H, W)
        meta  = src.meta.copy()

    bands, h, w = stack.shape

    # Step 1: Reshape to pixel table (pixels x features)
    X = stack.reshape(bands, -1).T              # (H*W, 5)

    # Mask any pixel where at least one band is nodata
    nodata_mask  = np.any(X == NODATA_VAL, axis=1)
    X_valid_raw  = X[~nodata_mask]

    print(f"  Stack shape  : {stack.shape}  -> pixel table: {X.shape}")
    print(f"  Valid pixels : {X_valid_raw.shape[0]:,}  ({X_valid_raw.shape[0]/X.shape[0]*100:.1f}%)")
    print(f"  Nodata pixels: {nodata_mask.sum():,}")

    # Step 2: Normalise — DTM is 0-138m, NDVI is -0.4 to 0.45 — must scale equally
    scaler       = StandardScaler()
    X_valid_norm = scaler.fit_transform(X_valid_raw)

    # Step 3: KMeans clustering
    print(f"\n  Running KMeans (k={n_classes})...")
    kmeans = KMeans(
        n_clusters=n_classes,
        random_state=RANDOM_SEED,
        n_init=10,
        max_iter=300,
    )
    raw_labels = kmeans.fit_predict(X_valid_norm)
    print(f"  KMeans inertia: {kmeans.inertia_:.2f}")

    # Step 4: Remap cluster IDs to semantic class labels
    semantic_labels = map_clusters_to_classes(raw_labels, X_valid_raw, n_classes)

    # Step 5: Write labels back to full 2D grid
    result_flat               = np.full(X.shape[0], NODATA_VAL, dtype=np.int16)
    result_flat[~nodata_mask] = semantic_labels
    result_2d                 = result_flat.reshape(h, w)

    # Class distribution summary
    print("\n  Class distribution:")
    total_valid = X_valid_raw.shape[0]
    for cls, name in CLASS_NAMES.items():
        count = int(np.sum(result_2d == cls))
        pct   = count / total_valid * 100
        print(f"    Class {cls} ({name:<20}): {count:>8,} pixels  ({pct:5.1f}%)")

    # Save classified raster
    meta.update({"count": 1, "dtype": "int16", "nodata": NODATA_VAL})

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(result_2d, 1)

    print(f"\n  Saved: {out_path}")


# ------------------------------------------------
# Run for both time points
# ------------------------------------------------

print("=" * 55)
print("  LULC Classification - KMeans (5 classes)")
print("=" * 55)

print("\n[1/2] Classifying 2012 (t1)...")
classify_stack(
    stack_path=os.path.join(OUT_DIR, "t1_feature_stack.tif"),
    out_path  =os.path.join(OUT_DIR, "lulc_2012.tif"),
)

print("\n[2/2] Classifying 2016 (t2)...")
classify_stack(
    stack_path=os.path.join(OUT_DIR, "t2_feature_stack.tif"),
    out_path  =os.path.join(OUT_DIR, "lulc_2016.tif"),
)

print("\n" + "=" * 55)
print("  Done. Outputs: lulc_2012.tif | lulc_2016.tif")
print("  Next: run 06_change_detection.py")
print("=" * 55)