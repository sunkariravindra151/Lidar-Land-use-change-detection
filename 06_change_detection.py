"""
06_change_detection.py
----------------------
Computes land-cover change between 2012 and 2016 LULC maps.

Outputs:
  binary_change_map.tif      — 1 = changed, 0 = unchanged, -9999 = nodata
  directional_change_map.tif — raw difference (lulc_2016 - lulc_2012)
  gain_loss_map.tif          — +1 = vegetation gain, -1 = vegetation loss,
                               0 = no change, -9999 = nodata

Class reference:
  0 = Bare ground
  1 = Grass / low veg
  2 = Medium vegetation
  3 = Dense vegetation
  4 = Built / rock
"""

import os
import numpy as np
import rasterio

BASE_DIR = r"C:\Users\Sunkari Ravindra\OneDrive\Desktop\My Documents\QGIS project"
OUT_DIR  = os.path.join(BASE_DIR, "outputs")

NODATA = -9999

CLASS_NAMES = {
    0: "Bare ground",
    1: "Grass / low veg",
    2: "Medium vegetation",
    3: "Dense vegetation",
    4: "Built / rock",
}

# ------------------------------------------------
# Load both LULC maps
# ------------------------------------------------

lulc_2012_path = os.path.join(OUT_DIR, "lulc_2012.tif")
lulc_2016_path = os.path.join(OUT_DIR, "lulc_2016.tif")

with rasterio.open(lulc_2012_path) as src:
    lulc_2012 = src.read(1).astype(np.int16)
    meta      = src.meta.copy()

with rasterio.open(lulc_2016_path) as src:
    lulc_2016 = src.read(1).astype(np.int16)

print("Loaded LULC maps:")
print(f"  2012 shape: {lulc_2012.shape}")
print(f"  2016 shape: {lulc_2016.shape}")

# Valid pixel mask — both years must have data
valid = (lulc_2012 != NODATA) & (lulc_2016 != NODATA)
total_valid = valid.sum()
print(f"  Valid pixels (both years): {total_valid:,}\n")

# ------------------------------------------------
# 1. Binary change map
#    1 = class changed between 2012 and 2016
#    0 = class unchanged
# ------------------------------------------------

binary = np.full_like(lulc_2012, NODATA, dtype=np.int16)
binary[valid] = (lulc_2016[valid] != lulc_2012[valid]).astype(np.int16)

changed   = int(np.sum(binary == 1))
unchanged = int(np.sum(binary == 0))
print("Binary Change Map:")
print(f"  Changed  : {changed:>10,} pixels  ({changed/total_valid*100:.1f}%)")
print(f"  Unchanged: {unchanged:>10,} pixels  ({unchanged/total_valid*100:.1f}%)")

# ------------------------------------------------
# 2. Directional change map
#    Raw difference: lulc_2016 - lulc_2012
#    Positive = class number increased (e.g. bare→dense)
#    Negative = class number decreased (e.g. dense→bare)
# ------------------------------------------------

directional = np.full_like(lulc_2012, NODATA, dtype=np.int16)
directional[valid] = lulc_2016[valid] - lulc_2012[valid]

print("\nDirectional Change Map (lulc_2016 - lulc_2012):")
for diff in range(-4, 5):
    count = int(np.sum(directional[valid] == diff))
    if count > 0:
        pct = count / total_valid * 100
        direction = "no change" if diff == 0 else ("increase" if diff > 0 else "decrease")
        print(f"  diff={diff:+d}  ({direction:>10}): {count:>8,} pixels  ({pct:.1f}%)")

# ------------------------------------------------
# 3. Gain / Loss map  (vegetation focused)
#    Classes 1-3 = vegetation
#    Class 0,4   = non-vegetation
#    +1 = gained vegetation (non-veg → veg)
#    -1 = lost vegetation   (veg → non-veg)
#     0 = no vegetation change
# ------------------------------------------------

VEG_CLASSES = {1, 2, 3}

was_veg  = np.isin(lulc_2012, list(VEG_CLASSES))
is_veg   = np.isin(lulc_2016, list(VEG_CLASSES))

gain_loss = np.full_like(lulc_2012, NODATA, dtype=np.int16)
gain_loss[valid & ~was_veg & is_veg]  =  1   # gained vegetation
gain_loss[valid & was_veg  & ~is_veg] = -1   # lost vegetation
gain_loss[valid & ~(~was_veg & is_veg) & ~(was_veg & ~is_veg)] = 0  # no change

veg_gain = int(np.sum(gain_loss ==  1))
veg_loss = int(np.sum(gain_loss == -1))
no_chg   = int(np.sum(gain_loss ==  0))

print("\nGain / Loss Map (vegetation focused):")
print(f"  Vegetation gain (+1): {veg_gain:>8,} pixels  ({veg_gain/total_valid*100:.1f}%)")
print(f"  Vegetation loss (-1): {veg_loss:>8,} pixels  ({veg_loss/total_valid*100:.1f}%)")
print(f"  No change       ( 0): {no_chg:>8,} pixels  ({no_chg/total_valid*100:.1f}%)")

# ------------------------------------------------
# Transition matrix (what changed to what)
# ------------------------------------------------

print("\nTransition Matrix (rows=2012, cols=2016, values=pixel count):")
col_label = "2012 / 2016"
header = f"{col_label:>18}" + "".join(f"{CLASS_NAMES[c]:>20}" for c in range(5))
print(header)

for r in range(5):
    row = f"{CLASS_NAMES[r]:>18}"
    for c in range(5):
        count = int(np.sum(valid & (lulc_2012 == r) & (lulc_2016 == c)))
        row += f"{count:>20,}"
    print(row)

# ------------------------------------------------
# Save all three outputs
# ------------------------------------------------

meta.update({"count": 1, "dtype": "int16", "nodata": NODATA})

outputs = {
    "binary_change_map.tif"     : binary,
    "directional_change_map.tif": directional,
    "gain_loss_map.tif"         : gain_loss,
}

print("\nSaving outputs...")
for fname, arr in outputs.items():
    out_path = os.path.join(OUT_DIR, fname)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(arr, 1)
    print(f"  Saved: {out_path}")

print("\n" + "=" * 55)
print("  Change detection complete.")
print("  Outputs:")
print("    binary_change_map.tif")
print("    directional_change_map.tif")
print("    gain_loss_map.tif")
print("=" * 55)