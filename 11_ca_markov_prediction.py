"""
11_ca_markov_prediction.py
--------------------------
Future land cover prediction using Cellular Automata + Markov Chain (CA-Markov).

Method:
  1. Markov Chain  — computes transition probability matrix from 2012→2016
  2. Cellular Automata — applies spatial neighbourhood rules to simulate
     realistic spatial patterns (not just random pixel flipping)

Predicts:
  predicted_lulc_2020.tif   — 4 years after 2016 (same interval as 2012→2016)
  predicted_lulc_2024.tif   — 8 years after 2016

Classes:
  0 = Bare ground
  1 = Grass / low veg
  2 = Medium vegetation
  3 = Dense vegetation
  4 = Built / rock
"""

import os
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import uniform_filter

BASE_DIR = r"C:\Users\Sunkari Ravindra\OneDrive\Desktop\My Documents\QGIS project"
OUT_DIR  = os.path.join(BASE_DIR, "outputs")

NODATA     = -9999
N_CLASSES  = 5
YEARS_STEP = 4      # 2012→2016 = 4 years, so each step = 4 years
TARGET_YEARS = [2020, 2024]

CLASS_NAMES = {
    0: "Bare ground",
    1: "Grass / low veg",
    2: "Medium vegetation",
    3: "Dense vegetation",
    4: "Built / rock",
}

LULC_COLORS = ["#d4b483", "#a8d08d", "#4daf4a", "#1b7837", "#808080"]
lulc_cmap   = ListedColormap(LULC_COLORS)
lulc_norm   = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], lulc_cmap.N)

# ------------------------------------------------
# Load LULC maps
# ------------------------------------------------

print("=" * 65)
print("  CA-Markov Future Land Cover Prediction")
print("  Study: 2016 Kaikoura Earthquake (New Zealand)")
print("=" * 65)

with rasterio.open(os.path.join(OUT_DIR, "lulc_2012.tif")) as src:
    lulc_2012 = src.read(1).astype(np.int16)
    meta      = src.meta.copy()

with rasterio.open(os.path.join(OUT_DIR, "lulc_2016.tif")) as src:
    lulc_2016 = src.read(1).astype(np.int16)

valid = (lulc_2012 != NODATA) & (lulc_2016 != NODATA)
print(f"\n  Valid pixels: {valid.sum():,}")

# ================================================
# STEP 1: Markov Chain — Transition Probability Matrix
# ================================================

print("\n--- Step 1: Computing Markov Transition Matrix ---")

# Count transitions from 2012 → 2016
transition_counts = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
for from_cls in range(N_CLASSES):
    for to_cls in range(N_CLASSES):
        transition_counts[from_cls, to_cls] = int(
            np.sum(valid & (lulc_2012 == from_cls) & (lulc_2016 == to_cls))
        )

# Convert counts to probabilities (row-normalise)
row_sums = transition_counts.sum(axis=1, keepdims=True)
transition_prob = np.where(
    row_sums > 0,
    transition_counts / row_sums,
    0.0
)

print("\n  Transition Probability Matrix (rows=from, cols=to):")
col_label = "From / To"
header = f"  {col_label:<22}" + "".join(f"{CLASS_NAMES[c][:10]:>13}" for c in range(N_CLASSES))
print(header)
print("  " + "-" * 87)
for i in range(N_CLASSES):
    row = f"  {CLASS_NAMES[i]:<22}"
    for j in range(N_CLASSES):
        row += f"{transition_prob[i,j]:>13.4f}"
    print(row)

# Cumulative probability matrix (for Monte Carlo sampling)
cumulative_prob = np.cumsum(transition_prob, axis=1)

# ================================================
# STEP 2: CA Suitability Maps
# Spatial neighbourhood filter — pixels near a class
# are more likely to transition to that class
# ================================================

print("\n--- Step 2: Computing CA Suitability Maps ---")

def compute_suitability(lulc_map, n_classes, filter_size=5):
    """
    For each class, compute the proportion of that class
    in a local neighbourhood window. This is the CA spatial
    suitability component — pixels near dense veg are more
    likely to become dense veg.
    """
    h, w = lulc_map.shape
    suitability = np.zeros((n_classes, h, w), dtype=np.float32)

    for cls in range(n_classes):
        class_mask = (lulc_map == cls).astype(np.float32)
        # Smooth with uniform filter = proportion of class in neighbourhood
        suitability[cls] = uniform_filter(class_mask, size=filter_size)

    # Normalise so suitability sums to 1 per pixel
    total = suitability.sum(axis=0, keepdims=True)
    total = np.where(total == 0, 1, total)
    suitability = suitability / total

    return suitability

print("  Computing spatial suitability from 2016 map (filter=5px)...")
suitability_2016 = compute_suitability(lulc_2016, N_CLASSES, filter_size=5)
print("  Suitability maps computed.")

# ================================================
# STEP 3: CA-Markov Iteration
# Combine Markov probabilities with CA suitability
# ================================================

def ca_markov_step(current_lulc, trans_prob, suitability, valid_mask, random_seed=42):
    """
    One CA-Markov time step:
    1. For each pixel, get Markov transition probabilities
    2. Weight by CA spatial suitability
    3. Sample the next class
    """
    rng = np.random.default_rng(random_seed)
    h, w = current_lulc.shape
    next_lulc = current_lulc.copy()

    # Get transition probabilities for each valid pixel
    valid_idx = np.where(valid_mask.ravel())[0]

    current_flat = current_lulc.ravel()
    suit_flat    = suitability.reshape(N_CLASSES, -1)  # (5, H*W)

    for idx in valid_idx:
        from_cls = current_flat[idx]
        if from_cls < 0 or from_cls >= N_CLASSES:
            continue

        # Markov probability for this pixel's current class
        markov_p = trans_prob[from_cls].copy()     # (5,)

        # CA suitability for this pixel
        ca_suit  = suit_flat[:, idx]               # (5,)

        # Combined probability = Markov × CA suitability
        combined = markov_p * ca_suit
        total    = combined.sum()

        if total > 0:
            combined = combined / total
        else:
            combined = markov_p

        # Sample next class
        next_cls = rng.choice(N_CLASSES, p=combined)
        next_lulc.ravel()[idx] = next_cls

    return next_lulc


# ================================================
# STEP 4: Predict 2020 and 2024
# ================================================

print("\n--- Step 3: Running CA-Markov Prediction ---")

results     = {}
current_map = lulc_2016.copy()
base_year   = 2016

for step_num, target_year in enumerate([2020, 2024]):
    print(f"\n  Predicting {target_year} (step {step_num+1})...")

    # Recompute suitability from current map each step
    suit = compute_suitability(current_map, N_CLASSES, filter_size=5)

    predicted = ca_markov_step(
        current_lulc=current_map,
        trans_prob=transition_prob,
        suitability=suit,
        valid_mask=valid,
        random_seed=42 + step_num,
    )

    # Class distribution
    print(f"  {target_year} class distribution:")
    for cls, name in CLASS_NAMES.items():
        count = int(np.sum((predicted != NODATA) & (predicted == cls)))
        ha    = count * 1.0 / 10000
        pct   = count / valid.sum() * 100
        print(f"    Class {cls} ({name:<20}): {count:>8,} px  {ha:>7.1f} ha  ({pct:.1f}%)")

    # Save raster
    out_path = os.path.join(OUT_DIR, f"predicted_lulc_{target_year}.tif")
    out_meta = meta.copy()
    out_meta.update({"count": 1, "dtype": "int16", "nodata": NODATA})

    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(predicted, 1)
    print(f"  Saved: predicted_lulc_{target_year}.tif")

    results[target_year] = predicted
    current_map = predicted   # chain: 2016 → 2020 → 2024


# ================================================
# STEP 5: Change maps vs 2016
# ================================================

print("\n--- Step 4: Computing change maps vs 2016 ---")

for year, pred in results.items():
    changed = np.full_like(lulc_2016, NODATA, dtype=np.int16)
    changed[valid] = (pred[valid] != lulc_2016[valid]).astype(np.int16)

    out_path = os.path.join(OUT_DIR, f"predicted_change_2016_{year}.tif")
    out_meta = meta.copy()
    out_meta.update({"count": 1, "dtype": "int16", "nodata": NODATA})
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(changed, 1)

    chg_pct = changed[valid].sum() / valid.sum() * 100
    print(f"  2016 → {year}: {chg_pct:.1f}% predicted change  →  saved predicted_change_2016_{year}.tif")

# ================================================
# STEP 6: Publication figure — 4-panel map
# 2012 | 2016 | 2020 | 2024
# ================================================

print("\n--- Step 5: Generating prediction figure ---")

maps  = [lulc_2012, lulc_2016, results[2020], results[2024]]
years = ["2012 (Observed)", "2016 (Observed)", "2020 (Predicted)", "2024 (Predicted)"]

fig, axes = plt.subplots(1, 4, figsize=(24, 7))
fig.suptitle(
    "Land Cover: Observed (2012, 2016) and CA-Markov Predictions (2020, 2024)\n"
    "Kaikoura, New Zealand — Post-earthquake recovery trajectory",
    fontsize=13, fontweight="bold", y=1.02
)

for ax, data, year in zip(axes, maps, years):
    plot_data = data.astype(np.float32)
    plot_data[plot_data == NODATA] = np.nan

    ax.imshow(plot_data, cmap=lulc_cmap, norm=lulc_norm, interpolation="nearest")
    ax.set_title(year, fontsize=11, fontweight="bold",
                 color="black" if "Observed" in year else "#d73027")
    ax.set_xlabel("Easting (px)", fontsize=8)
    ax.set_ylabel("Northing (px)", fontsize=8)
    ax.tick_params(labelsize=7)

# Add "Predicted" border to 2020 and 2024
for ax in axes[2:]:
    for spine in ax.spines.values():
        spine.set_edgecolor("#d73027")
        spine.set_linewidth(2.5)

# Shared legend
patches = [mpatches.Patch(color=LULC_COLORS[i], label=CLASS_NAMES[i]) for i in range(5)]
fig.legend(handles=patches, loc="lower center", ncol=5,
           fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout()

for ext in ["png", "pdf"]:
    out = os.path.join(OUT_DIR, f"figure5_lulc_prediction_2020_2024.{ext}")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: figure5_lulc_prediction_2020_2024.{ext}")
plt.close()

# ================================================
# STEP 7: Area trend chart (2012 → 2016 → 2020 → 2024)
# ================================================

print("\n--- Step 6: Generating area trend chart ---")

all_maps  = [lulc_2012, lulc_2016, results[2020], results[2024]]
all_years = [2012, 2016, 2020, 2024]

fig, ax = plt.subplots(figsize=(10, 6))

for cls in range(N_CLASSES):
    areas = []
    for m in all_maps:
        ha = int(np.sum((m != NODATA) & (m == cls))) / 10000
        areas.append(ha)

    style = "--" if 2020 in all_years[2:] else "-"
    ax.plot(all_years[:2], areas[:2], "o-",
            color=LULC_COLORS[cls], linewidth=2.5,
            label=CLASS_NAMES[cls])
    ax.plot(all_years[1:], areas[1:], "o--",
            color=LULC_COLORS[cls], linewidth=2, alpha=0.7)

# Mark the split between observed and predicted
ax.axvline(x=2016, color="gray", linestyle=":", linewidth=1.5)
ax.text(2016.1, ax.get_ylim()[1] * 0.95, "Prediction →",
        fontsize=9, color="gray", style="italic")

ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Area (ha)", fontsize=11)
ax.set_title("LULC Area Trends — Observed and Predicted\n"
             "Kaikoura, New Zealand (Solid = observed, Dashed = predicted)",
             fontsize=12, fontweight="bold")
ax.set_xticks(all_years)
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
for ext in ["png", "pdf"]:
    out = os.path.join(OUT_DIR, f"figure6_area_trend_chart.{ext}")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: figure6_area_trend_chart.{ext}")
plt.close()

print("\n" + "=" * 65)
print("  CA-Markov Prediction Complete.")
print("  Raster outputs:")
print("    predicted_lulc_2020.tif")
print("    predicted_lulc_2024.tif")
print("    predicted_change_2016_2020.tif")
print("    predicted_change_2016_2024.tif")
print("  Figures:")
print("    figure5_lulc_prediction_2020_2024.png / .pdf")
print("    figure6_area_trend_chart.png / .pdf")
print("=" * 65)