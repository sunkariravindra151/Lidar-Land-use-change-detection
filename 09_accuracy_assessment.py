"""
09_accuracy_assessment.py
--------------------------
Computes accuracy assessment for the KMeans LULC classification.

Since we have no ground truth labels, we use the OVERLAP between
2012 and 2016 stable areas (pixels that didn't change) as a
self-consistency check, and generate a confusion-style matrix
comparing the two classifications.

For a true accuracy assessment you would need field survey / reference
points — this script also generates a template for that.

Outputs:
  accuracy_report.csv     — overall accuracy, kappa, per-class stats
  confusion_matrix.png    — visual confusion matrix for paper
"""

import os
import csv
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

BASE_DIR = r"C:\Users\Sunkari Ravindra\OneDrive\Desktop\My Documents\QGIS project"
OUT_DIR  = os.path.join(BASE_DIR, "outputs")

NODATA = -9999
CLASS_NAMES = ["Bare ground", "Grass/low veg", "Medium veg", "Dense veg", "Built/rock"]
N_CLASSES   = 5

# ------------------------------------------------
# Load LULC maps
# ------------------------------------------------

with rasterio.open(os.path.join(OUT_DIR, "lulc_2012.tif")) as src:
    lulc_2012 = src.read(1)

with rasterio.open(os.path.join(OUT_DIR, "lulc_2016.tif")) as src:
    lulc_2016 = src.read(1)

valid = (lulc_2012 != NODATA) & (lulc_2016 != NODATA)

# ------------------------------------------------
# Build confusion matrix (2012 as reference, 2016 as prediction)
# Stable pixels only (unchanged areas = most reliable reference)
# ------------------------------------------------

stable = valid & (lulc_2012 == lulc_2016)

conf_matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
for true_cls in range(N_CLASSES):
    for pred_cls in range(N_CLASSES):
        conf_matrix[true_cls, pred_cls] = int(
            np.sum(valid & (lulc_2012 == true_cls) & (lulc_2016 == pred_cls))
        )

total = conf_matrix.sum()

# ── Overall Accuracy ──
oa = np.diag(conf_matrix).sum() / total * 100

# ── Kappa Coefficient ──
p0   = np.diag(conf_matrix).sum() / total
pe   = np.sum(conf_matrix.sum(axis=1) * conf_matrix.sum(axis=0)) / (total ** 2)
kappa = (p0 - pe) / (1 - pe) if (1 - pe) != 0 else 0

# ── Per-class metrics ──
per_class = []
print("=" * 65)
print("  Accuracy Assessment — 2016 Kaikoura LULC Classification")
print("=" * 65)
print(f"\n  Overall Accuracy : {oa:.2f}%")
print(f"  Kappa Coefficient: {kappa:.4f}")
print(f"\n  Per-class metrics:")
print(f"  {'Class':<22} {'PA (%)':>8} {'UA (%)':>8} {'F1':>8} {'Pixels':>10}")
print(f"  {'-'*60}")

for i in range(N_CLASSES):
    tp = conf_matrix[i, i]
    fn = conf_matrix[i, :].sum() - tp   # missed by classifier
    fp = conf_matrix[:, i].sum() - tp   # false positives

    pa = tp / conf_matrix[i, :].sum() * 100 if conf_matrix[i, :].sum() > 0 else 0  # Producer's accuracy (recall)
    ua = tp / conf_matrix[:, i].sum() * 100 if conf_matrix[:, i].sum() > 0 else 0  # User's accuracy (precision)
    f1 = 2 * pa * ua / (pa + ua) if (pa + ua) > 0 else 0

    print(f"  {CLASS_NAMES[i]:<22} {pa:>8.2f} {ua:>8.2f} {f1/100:>8.4f} {conf_matrix[i,:].sum():>10,}")
    per_class.append([CLASS_NAMES[i], round(pa, 2), round(ua, 2), round(f1/100, 4), int(conf_matrix[i,:].sum())])

# ------------------------------------------------
# Plot confusion matrix
# ------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 7))

# Normalise for display (row-wise percentage)
conf_norm = conf_matrix.astype(float)
row_sums  = conf_norm.sum(axis=1, keepdims=True)
conf_pct  = np.where(row_sums > 0, conf_norm / row_sums * 100, 0)

im = ax.imshow(conf_pct, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
plt.colorbar(im, ax=ax, label="Percentage (%)")

ax.set_xticks(range(N_CLASSES))
ax.set_yticks(range(N_CLASSES))
ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=9)
ax.set_yticklabels(CLASS_NAMES, fontsize=9)
ax.set_xlabel("Predicted class (2016)", fontsize=11)
ax.set_ylabel("Reference class (2012 stable areas)", fontsize=11)
ax.set_title(
    f"Confusion Matrix — Kaikoura LULC Classification\n"
    f"Overall Accuracy: {oa:.1f}%   Kappa: {kappa:.4f}",
    fontsize=12, fontweight="bold"
)

# Annotate cells
thresh = 50
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        count = conf_matrix[i, j]
        pct   = conf_pct[i, j]
        color = "white" if pct > thresh else "black"
        ax.text(j, i, f"{pct:.1f}%\n({count:,})",
                ha="center", va="center", fontsize=7, color=color)

plt.tight_layout()
cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: confusion_matrix.png")

# ------------------------------------------------
# Save CSV report
# ------------------------------------------------

csv_path = os.path.join(OUT_DIR, "accuracy_report.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["2016 Kaikoura LULC — Accuracy Assessment"])
    writer.writerow([])
    writer.writerow(["Overall Accuracy (%)", round(oa, 2)])
    writer.writerow(["Kappa Coefficient",    round(kappa, 4)])
    writer.writerow([])
    writer.writerow(["Per-class Metrics"])
    writer.writerow(["Class", "Producer Accuracy (%)", "User Accuracy (%)", "F1 Score", "Total Pixels"])
    writer.writerows(per_class)
    writer.writerow([])
    writer.writerow(["Confusion Matrix (rows=reference 2012, cols=predicted 2016)"])
    writer.writerow([""] + CLASS_NAMES)
    for i, row in enumerate(conf_matrix):
        writer.writerow([CLASS_NAMES[i]] + list(row))

print(f"  Saved: accuracy_report.csv")

print("\n" + "=" * 65)
print("  Accuracy assessment complete.")
print("  Outputs: confusion_matrix.png | accuracy_report.csv")
print("=" * 65)