"""
07_damage_assessment.py
-----------------------
Computes earthquake damage statistics for the 2016 Kaikoura event.
Uses change detection outputs to quantify:
  - Area statistics per LULC class (hectares)
  - Vegetation loss / gain areas
  - Damage severity zones
  - CSV table for paper
"""

import os
import csv
import numpy as np
import rasterio

BASE_DIR = r"C:\Users\Sunkari Ravindra\OneDrive\Desktop\My Documents\QGIS project"
OUT_DIR  = os.path.join(BASE_DIR, "outputs")

NODATA      = -9999
PIXEL_AREA  = 1.0          # 1m x 1m pixels -> 1 m² each
M2_TO_HA    = 1 / 10000    # convert m² to hectares

CLASS_NAMES = {
    0: "Bare ground",
    1: "Grass / low veg",
    2: "Medium vegetation",
    3: "Dense vegetation",
    4: "Built / rock",
}

# ------------------------------------------------
# Load all maps
# ------------------------------------------------

def load(path):
    with rasterio.open(path) as src:
        return src.read(1)

lulc_2012   = load(os.path.join(OUT_DIR, "lulc_2012.tif"))
lulc_2016   = load(os.path.join(OUT_DIR, "lulc_2016.tif"))
gain_loss   = load(os.path.join(OUT_DIR, "gain_loss_map.tif"))
binary      = load(os.path.join(OUT_DIR, "binary_change_map.tif"))

valid = (lulc_2012 != NODATA) & (lulc_2016 != NODATA)
total_valid_ha = valid.sum() * PIXEL_AREA * M2_TO_HA

print("=" * 65)
print("  2016 Kaikoura Earthquake — Damage Assessment")
print("=" * 65)
print(f"\n  Study area (valid pixels): {total_valid_ha:,.1f} ha\n")

# ------------------------------------------------
# 1. LULC Area Statistics per class (2012 vs 2016)
# ------------------------------------------------

print("-" * 65)
print(f"  {'Class':<22} {'2012 (ha)':>12} {'2016 (ha)':>12} {'Change (ha)':>12}")
print("-" * 65)

class_stats = []
for cls, name in CLASS_NAMES.items():
    ha_2012 = int(np.sum(valid & (lulc_2012 == cls))) * PIXEL_AREA * M2_TO_HA
    ha_2016 = int(np.sum(valid & (lulc_2016 == cls))) * PIXEL_AREA * M2_TO_HA
    delta   = ha_2016 - ha_2012
    sign    = "+" if delta >= 0 else ""
    print(f"  {name:<22} {ha_2012:>12,.1f} {ha_2016:>12,.1f} {sign}{delta:>11,.1f}")
    class_stats.append([name, round(ha_2012, 2), round(ha_2016, 2), round(delta, 2)])

print("-" * 65)

# ------------------------------------------------
# 2. Vegetation Loss / Gain Summary
# ------------------------------------------------

veg_loss_px   = int(np.sum(gain_loss == -1))
veg_gain_px   = int(np.sum(gain_loss ==  1))
no_change_px  = int(np.sum(gain_loss ==  0))
changed_px    = int(np.sum(binary    ==  1))

veg_loss_ha   = veg_loss_px  * PIXEL_AREA * M2_TO_HA
veg_gain_ha   = veg_gain_px  * PIXEL_AREA * M2_TO_HA
changed_ha    = changed_px   * PIXEL_AREA * M2_TO_HA
net_veg_ha    = veg_gain_ha  - veg_loss_ha

print(f"\n  Vegetation Change Summary:")
print(f"  {'Vegetation loss':<28}: {veg_loss_ha:>8,.1f} ha  ({veg_loss_px:,} pixels)")
print(f"  {'Vegetation gain':<28}: {veg_gain_ha:>8,.1f} ha  ({veg_gain_px:,} pixels)")
print(f"  {'Net vegetation change':<28}: {'+' if net_veg_ha>=0 else ''}{net_veg_ha:>7,.1f} ha")
print(f"  {'Total changed area':<28}: {changed_ha:>8,.1f} ha  ({changed_px/valid.sum()*100:.1f}% of study area)")

# ------------------------------------------------
# 3. Damage Severity Zones
# Based on directional change magnitude
# ------------------------------------------------

dir_map = load(os.path.join(OUT_DIR, "directional_change_map.tif"))

# Severity classification
#   High damage   : class dropped 3-4 levels (e.g. dense veg -> bare)
#   Medium damage : class dropped 1-2 levels
#   Low damage    : class dropped < 1 (unchanged or minor)
#   Recovery      : class increased (positive change)

high_damage   = valid & (dir_map <= -3)
medium_damage = valid & (dir_map >= -2) & (dir_map <= -1)
recovery      = valid & (dir_map >= 1)
no_chg        = valid & (dir_map == 0)

severity_map = np.full_like(lulc_2012, NODATA, dtype=np.int16)
severity_map[no_chg]        = 0   # No change
severity_map[recovery]      = 1   # Recovery / gain
severity_map[medium_damage] = 2   # Medium damage
severity_map[high_damage]   = 3   # High damage

print(f"\n  Damage Severity Zones:")
severity_labels = {
    0: "No change",
    1: "Recovery / vegetation gain",
    2: "Medium damage (1-2 class drop)",
    3: "High damage  (3-4 class drop)",
}
severity_stats = []
for sev, label in severity_labels.items():
    count = int(np.sum(severity_map == sev))
    ha    = count * PIXEL_AREA * M2_TO_HA
    pct   = count / valid.sum() * 100
    print(f"  {label:<34}: {ha:>8,.1f} ha  ({pct:.1f}%)")
    severity_stats.append([label, count, round(ha, 2), round(pct, 2)])

# Save severity map
with rasterio.open(os.path.join(OUT_DIR, "lulc_2012.tif")) as ref:
    meta = ref.meta.copy()
meta.update({"count": 1, "dtype": "int16", "nodata": NODATA})
with rasterio.open(os.path.join(OUT_DIR, "damage_severity.tif"), "w", **meta) as dst:
    dst.write(severity_map, 1)
print(f"\n  Saved: damage_severity.tif")

# ------------------------------------------------
# 4. Key Transition Statistics (for paper)
# ------------------------------------------------

print(f"\n  Key Transitions (Kaikoura earthquake impact):")
transitions = [
    (3, 0, "Dense veg -> Bare ground   (landslide scarring)"),
    (3, 1, "Dense veg -> Grass         (canopy removal)"),
    (1, 0, "Grass     -> Bare ground   (slope failure)"),
    (4, 3, "Built/rock -> Dense veg    (revegetation)"),
    (0, 3, "Bare ground -> Dense veg   (recovery)"),
]
transition_stats = []
for from_cls, to_cls, label in transitions:
    count = int(np.sum(valid & (lulc_2012 == from_cls) & (lulc_2016 == to_cls)))
    ha    = count * PIXEL_AREA * M2_TO_HA
    print(f"  {label:<45}: {ha:>7,.1f} ha  ({count:,} px)")
    transition_stats.append([label, count, round(ha, 2)])

# ------------------------------------------------
# 5. Save CSV tables for paper
# ------------------------------------------------

csv_path = os.path.join(OUT_DIR, "damage_statistics.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow(["2016 Kaikoura Earthquake - Damage Assessment Statistics"])
    writer.writerow([])

    writer.writerow(["LULC Area Statistics"])
    writer.writerow(["Class", "Area 2012 (ha)", "Area 2016 (ha)", "Change (ha)"])
    writer.writerows(class_stats)
    writer.writerow([])

    writer.writerow(["Vegetation Change Summary"])
    writer.writerow(["Metric", "Value (ha)"])
    writer.writerow(["Vegetation loss", round(veg_loss_ha, 2)])
    writer.writerow(["Vegetation gain", round(veg_gain_ha, 2)])
    writer.writerow(["Net vegetation change", round(net_veg_ha, 2)])
    writer.writerow(["Total changed area", round(changed_ha, 2)])
    writer.writerow([])

    writer.writerow(["Damage Severity Zones"])
    writer.writerow(["Severity", "Pixels", "Area (ha)", "Percent (%)"])
    writer.writerows(severity_stats)
    writer.writerow([])

    writer.writerow(["Key Transitions"])
    writer.writerow(["Transition", "Pixels", "Area (ha)"])
    writer.writerows(transition_stats)

print(f"  Saved: damage_statistics.csv")

print("\n" + "=" * 65)
print("  Damage assessment complete.")
print("  Outputs: damage_severity.tif | damage_statistics.csv")
print("=" * 65)