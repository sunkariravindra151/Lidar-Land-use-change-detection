"""
12_predicted_analysis.py
------------------------
Runs the full analysis pipeline on predicted LULC maps:
  - Change detection (2016→2020, 2020→2024)
  - Gain/loss maps
  - Damage severity
  - Area statistics CSV
  - Confusion-style matrices
  - Publication figures

Mirrors everything done for 2012→2016 but applied to predictions.
"""

import os
import csv
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

BASE_DIR = r"C:\Users\Sunkari Ravindra\OneDrive\Desktop\My Documents\QGIS project"
OUT_DIR  = os.path.join(BASE_DIR, "outputs")

NODATA     = -9999
N_CLASSES  = 5
DPI        = 300

CLASS_NAMES = {
    0: "Bare ground",
    1: "Grass / low veg",
    2: "Medium vegetation",
    3: "Dense vegetation",
    4: "Built / rock",
}
VEG_CLASSES  = {1, 2, 3}

LULC_COLORS  = ["#d4b483", "#a8d08d", "#4daf4a", "#1b7837", "#808080"]
lulc_cmap    = ListedColormap(LULC_COLORS)
lulc_norm    = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], lulc_cmap.N)

CHANGE_COLORS = ["#d73027", "#d0d0d0", "#1a9641"]
change_cmap   = ListedColormap(CHANGE_COLORS)
change_norm   = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], change_cmap.N)

SEV_COLORS = ["#d0d0d0", "#1a9641", "#fdae61", "#d73027"]
SEV_LABELS = ["No change", "Recovery", "Medium damage", "High damage"]
sev_cmap   = ListedColormap(SEV_COLORS)
sev_norm   = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], sev_cmap.N)

# ------------------------------------------------
# Load all LULC maps
# ------------------------------------------------

def load(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.int16), src.meta.copy()

def load_masked(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
    data[data == NODATA]  = np.nan
    data[data == -32768]  = np.nan
    return data

print("=" * 65)
print("  Post-Prediction Analysis: 2016→2020 and 2020→2024")
print("=" * 65)

lulc_2012, meta = load(os.path.join(OUT_DIR, "lulc_2012.tif"))
lulc_2016, _    = load(os.path.join(OUT_DIR, "lulc_2016.tif"))
lulc_2020, _    = load(os.path.join(OUT_DIR, "predicted_lulc_2020.tif"))
lulc_2024, _    = load(os.path.join(OUT_DIR, "predicted_lulc_2024.tif"))

# ================================================
# CORE ANALYSIS FUNCTION
# Runs full change detection + stats for any pair
# ================================================

def analyse_pair(lulc_t1, lulc_t2, year_t1, year_t2, meta, all_stats):
    label    = f"{year_t1}_{year_t2}"
    tag      = f"{year_t1}→{year_t2}"
    observed = year_t1 <= 2016 and year_t2 <= 2016

    print(f"\n{'='*65}")
    print(f"  Analysis: {tag}  ({'Observed' if observed else 'Predicted'})")
    print(f"{'='*65}")

    valid = (lulc_t1 != NODATA) & (lulc_t2 != NODATA)
    total_valid = valid.sum()
    total_ha    = total_valid / 10000

    # ── 1. Binary change map ──
    binary = np.full_like(lulc_t1, NODATA, dtype=np.int16)
    binary[valid] = (lulc_t2[valid] != lulc_t1[valid]).astype(np.int16)

    changed   = int(np.sum(binary == 1))
    unchanged = int(np.sum(binary == 0))
    print(f"\n  Binary Change:")
    print(f"    Changed  : {changed:>10,} px  ({changed/total_valid*100:.1f}%)")
    print(f"    Unchanged: {unchanged:>10,} px  ({unchanged/total_valid*100:.1f}%)")

    # ── 2. Directional change map ──
    directional = np.full_like(lulc_t1, NODATA, dtype=np.int16)
    directional[valid] = lulc_t2[valid] - lulc_t1[valid]

    # ── 3. Gain / Loss map ──
    was_veg  = np.isin(lulc_t1, list(VEG_CLASSES))
    is_veg   = np.isin(lulc_t2, list(VEG_CLASSES))

    gain_loss = np.full_like(lulc_t1, NODATA, dtype=np.int16)
    gain_loss[valid & ~was_veg & is_veg]  =  1
    gain_loss[valid & was_veg  & ~is_veg] = -1
    gain_loss[valid & ~(~was_veg & is_veg) & ~(was_veg & ~is_veg)] = 0

    veg_gain = int(np.sum(gain_loss ==  1))
    veg_loss = int(np.sum(gain_loss == -1))
    net_ha   = (veg_gain - veg_loss) / 10000

    print(f"\n  Vegetation Change:")
    print(f"    Gain: {veg_gain:>8,} px  ({veg_gain/total_valid*100:.1f}%)  {veg_gain/10000:.1f} ha")
    print(f"    Loss: {veg_loss:>8,} px  ({veg_loss/total_valid*100:.1f}%)  {veg_loss/10000:.1f} ha")
    print(f"    Net : {'+' if net_ha>=0 else ''}{net_ha:.1f} ha")

    # ── 4. Damage severity ──
    severity = np.full_like(lulc_t1, NODATA, dtype=np.int16)
    severity[valid & (directional == 0)]                       = 0
    severity[valid & (directional >= 1)]                       = 1
    severity[valid & (directional >= -2) & (directional <= -1)]= 2
    severity[valid & (directional <= -3)]                      = 3

    print(f"\n  Damage Severity:")
    sev_stats = []
    sev_labels_map = {0: "No change", 1: "Recovery", 2: "Medium damage", 3: "High damage"}
    for s, slabel in sev_labels_map.items():
        count = int(np.sum(severity == s))
        ha    = count / 10000
        pct   = count / total_valid * 100
        print(f"    {slabel:<25}: {count:>8,} px  {ha:>7.1f} ha  ({pct:.1f}%)")
        sev_stats.append([slabel, count, round(ha, 2), round(pct, 2)])

    # ── 5. LULC area statistics ──
    print(f"\n  LULC Area Statistics ({tag}):")
    print(f"  {'Class':<22} {str(year_t1)+' (ha)':>12} {str(year_t2)+' (ha)':>12} {'Change (ha)':>12}")
    print(f"  {'-'*60}")
    class_stats = []
    for cls, cname in CLASS_NAMES.items():
        ha_t1 = int(np.sum(valid & (lulc_t1 == cls))) / 10000
        ha_t2 = int(np.sum(valid & (lulc_t2 == cls))) / 10000
        delta = ha_t2 - ha_t1
        sign  = "+" if delta >= 0 else ""
        print(f"  {cname:<22} {ha_t1:>12.1f} {ha_t2:>12.1f} {sign}{delta:>11.1f}")
        class_stats.append([cname, round(ha_t1,2), round(ha_t2,2), round(delta,2)])

    # ── 6. Transition matrix ──
    conf = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for r in range(N_CLASSES):
        for c in range(N_CLASSES):
            conf[r, c] = int(np.sum(valid & (lulc_t1 == r) & (lulc_t2 == c)))

    # ── 7. Save rasters ──
    out_meta = meta.copy()
    out_meta.update({"count": 1, "dtype": "int16", "nodata": NODATA})

    rasters = {
        f"binary_change_{label}.tif"     : binary,
        f"directional_change_{label}.tif": directional,
        f"gain_loss_{label}.tif"         : gain_loss,
        f"damage_severity_{label}.tif"   : severity,
    }
    for fname, arr in rasters.items():
        with rasterio.open(os.path.join(OUT_DIR, fname), "w", **out_meta) as dst:
            dst.write(arr, 1)
        print(f"\n  Saved: {fname}")

    # ── 8. Save CSV ──
    csv_path = os.path.join(OUT_DIR, f"statistics_{label}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"Change Analysis: {tag}"])
        writer.writerow([])
        writer.writerow(["Binary Change"])
        writer.writerow(["Changed (px)", changed, f"{changed/total_valid*100:.1f}%"])
        writer.writerow(["Unchanged (px)", unchanged, f"{unchanged/total_valid*100:.1f}%"])
        writer.writerow([])
        writer.writerow(["Vegetation Change"])
        writer.writerow(["Gain (px)", veg_gain, f"{veg_gain/10000:.1f} ha"])
        writer.writerow(["Loss (px)", veg_loss, f"{veg_loss/10000:.1f} ha"])
        writer.writerow(["Net (ha)", round(net_ha, 2)])
        writer.writerow([])
        writer.writerow(["LULC Area Statistics"])
        writer.writerow(["Class", f"{year_t1} (ha)", f"{year_t2} (ha)", "Change (ha)"])
        writer.writerows(class_stats)
        writer.writerow([])
        writer.writerow(["Damage Severity"])
        writer.writerow(["Severity", "Pixels", "Area (ha)", "Percent (%)"])
        writer.writerows(sev_stats)
        writer.writerow([])
        writer.writerow(["Transition Matrix"])
        writer.writerow([f"{year_t1} to {year_t2}"] + [CLASS_NAMES[c] for c in range(N_CLASSES)])
        for i in range(N_CLASSES):
            writer.writerow([CLASS_NAMES[i]] + list(conf[i]))
    print(f"  Saved: statistics_{label}.csv")

    # Store for combined figure
    all_stats[tag] = {
        "binary": binary, "gain_loss": gain_loss,
        "severity": severity, "conf": conf,
        "veg_gain": veg_gain, "veg_loss": veg_loss,
        "changed": changed, "total": total_valid,
        "class_stats": class_stats,
    }

    return conf

# ================================================
# RUN ANALYSIS FOR BOTH PAIRS
# ================================================

all_stats = {}
conf_2016_2020 = analyse_pair(lulc_2016, lulc_2020, 2016, 2020, meta, all_stats)
conf_2020_2024 = analyse_pair(lulc_2020, lulc_2024, 2020, 2024, meta, all_stats)

# ================================================
# CONFUSION MATRIX FIGURES
# ================================================

def plot_confusion(conf, title, out_path):
    fig, ax = plt.subplots(figsize=(9, 7))

    row_sums = conf.sum(axis=1, keepdims=True)
    conf_pct = np.where(row_sums > 0, conf / row_sums * 100, 0)

    im = ax.imshow(conf_pct, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="Percentage (%)")

    labels = [CLASS_NAMES[i] for i in range(N_CLASSES)]
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted / Future class", fontsize=11)
    ax.set_ylabel("Reference / Current class", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")

    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            color = "white" if conf_pct[i, j] > 50 else "black"
            ax.text(j, i, f"{conf_pct[i,j]:.1f}%\n({conf[i,j]:,})",
                    ha="center", va="center", fontsize=7, color=color)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(out_path)}")

print("\n--- Generating confusion matrices ---")
plot_confusion(conf_2016_2020,
               "Transition Matrix 2016 -> 2020 (Predicted)\nKaikoura, New Zealand",
               os.path.join(OUT_DIR, "confusion_matrix_2016_2020.png"))

plot_confusion(conf_2020_2024,
               "Transition Matrix 2020 -> 2024 (Predicted)\nKaikoura, New Zealand",
               os.path.join(OUT_DIR, "confusion_matrix_2020_2024.png"))

# ================================================
# COMBINED CHANGE MAP FIGURE (3 periods side by side)
# ================================================

print("\n--- Generating combined change map figure ---")

gl_2012_2016 = load_masked(os.path.join(OUT_DIR, "gain_loss_map.tif"))
gl_2016_2020 = all_stats["2016→2020"]["gain_loss"].astype(np.float32)
gl_2020_2024 = all_stats["2020→2024"]["gain_loss"].astype(np.float32)
gl_2016_2020[gl_2016_2020 == NODATA] = np.nan
gl_2020_2024[gl_2020_2024 == NODATA] = np.nan

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle("Vegetation Change Maps — Observed and Predicted\nKaikoura Earthquake Recovery Trajectory",
             fontsize=13, fontweight="bold", y=1.02)

periods = [
    (gl_2012_2016, "2012 → 2016 (Observed)",  "black"),
    (gl_2016_2020, "2016 → 2020 (Predicted)", "#d73027"),
    (gl_2020_2024, "2020 → 2024 (Predicted)", "#d73027"),
]

for ax, (data, title, tcolor) in zip(axes, periods):
    ax.imshow(data, cmap=change_cmap, norm=change_norm, interpolation="nearest")
    ax.set_title(title, fontsize=11, fontweight="bold", color=tcolor)
    ax.set_xlabel("Easting (px)", fontsize=8)
    ax.set_ylabel("Northing (px)", fontsize=8)
    ax.tick_params(labelsize=7)

patches = [
    mpatches.Patch(color="#d73027", label="Vegetation loss"),
    mpatches.Patch(color="#d0d0d0", label="No change"),
    mpatches.Patch(color="#1a9641", label="Vegetation gain"),
]
fig.legend(handles=patches, loc="lower center", ncol=3,
           fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout()
for ext in ["png", "pdf"]:
    out = os.path.join(OUT_DIR, f"figure7_change_all_periods.{ext}")
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"  Saved: figure7_change_all_periods.{ext}")
plt.close()

# ================================================
# DAMAGE SEVERITY COMPARISON FIGURE
# ================================================

print("\n--- Generating damage severity comparison figure ---")

sev_2012_2016 = load_masked(os.path.join(OUT_DIR, "damage_severity.tif"))
sev_2016_2020 = all_stats["2016→2020"]["severity"].astype(np.float32)
sev_2020_2024 = all_stats["2020→2024"]["severity"].astype(np.float32)
sev_2016_2020[sev_2016_2020 == NODATA] = np.nan
sev_2020_2024[sev_2020_2024 == NODATA] = np.nan

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle("Damage Severity Maps — All Periods\nKaikoura Earthquake Recovery Trajectory",
             fontsize=13, fontweight="bold", y=1.02)

for ax, (data, title, tcolor) in zip(axes, [
    (sev_2012_2016, "2012 -> 2016 (Observed)",  "black"),
    (sev_2016_2020, "2016 -> 2020 (Predicted)", "#d73027"),
    (sev_2020_2024, "2020 -> 2024 (Predicted)", "#d73027"),
]):
    ax.imshow(data, cmap=sev_cmap, norm=sev_norm, interpolation="nearest")
    ax.set_title(title, fontsize=11, fontweight="bold", color=tcolor)
    ax.set_xlabel("Easting (px)", fontsize=8)
    ax.tick_params(labelsize=7)

patches = [mpatches.Patch(color=SEV_COLORS[i], label=SEV_LABELS[i]) for i in range(4)]
fig.legend(handles=patches, loc="lower center", ncol=4,
           fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout()
for ext in ["png", "pdf"]:
    out = os.path.join(OUT_DIR, f"figure8_severity_all_periods.{ext}")
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"  Saved: figure8_severity_all_periods.{ext}")
plt.close()

# ================================================
# SUMMARY TABLE — all three periods combined
# ================================================

print("\n--- Generating combined summary CSV ---")

summary_csv = os.path.join(OUT_DIR, "summary_all_periods.csv")
with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Kaikoura Earthquake — Full Period Summary"])
    writer.writerow([])
    writer.writerow(["Period", "Changed (%)", "Veg Gain (ha)", "Veg Loss (ha)", "Net Veg (ha)"])

    # 2012→2016 observed
    with rasterio.open(os.path.join(OUT_DIR, "binary_change_map.tif")) as src:
        b = src.read(1)
    with rasterio.open(os.path.join(OUT_DIR, "gain_loss_map.tif")) as src:
        gl = src.read(1)
    valid_obs = (b != NODATA)
    writer.writerow([
        "2012->2016 (Observed)",
        round(np.sum(b[valid_obs]==1)/valid_obs.sum()*100, 1),
        round(np.sum(gl==1)/10000, 1),
        round(np.sum(gl==-1)/10000, 1),
        round((np.sum(gl==1)-np.sum(gl==-1))/10000, 1),
    ])

    for tag, key in [("2016->2020 (Predicted)", "2016→2020"),
                     ("2020->2024 (Predicted)", "2020→2024")]:
        s = all_stats[key]
        writer.writerow([
            tag,
            round(s["changed"]/s["total"]*100, 1),
            round(s["veg_gain"]/10000, 1),
            round(s["veg_loss"]/10000, 1),
            round((s["veg_gain"]-s["veg_loss"])/10000, 1),
        ])

print(f"  Saved: summary_all_periods.csv")

print("\n" + "=" * 65)
print("  Post-prediction analysis complete.")
print("  Rasters:")
print("    binary/directional/gain_loss/severity for 2016_2020 & 2020_2024")
print("  CSVs:")
print("    statistics_2016_2020.csv")
print("    statistics_2020_2024.csv")
print("    summary_all_periods.csv")
print("  Figures:")
print("    confusion_matrix_2016_2020.png")
print("    confusion_matrix_2020_2024.png")
print("    figure7_change_all_periods.png / .pdf")
print("    figure8_severity_all_periods.png / .pdf")
print("=" * 65)