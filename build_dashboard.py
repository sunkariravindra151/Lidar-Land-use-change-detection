# ================================================================
#  build_dashboard.py  —  Kaikōura LULC Offline Dashboard Builder
#  Embeds ALL images as base64 → single shareable HTML file
#
#  RUN ORDER:
#    1.  python export_tifs_for_dashboard.py
#    2.  python add_missing_pngs.py
#    3.  python build_dashboard.py
# ================================================================
import os, sys, json, base64, warnings
import numpy as np
warnings.filterwarnings("ignore")
try:
    import rasterio
except ImportError:
    print("ERROR: pip install rasterio"); sys.exit(1)

PROJECT  = r"C:\Users\Sunkari Ravindra\OneDrive\Desktop\My Documents\QGIS project"
TIF_DIR  = os.path.join(PROJECT, "outputs")
IMG_DIR  = os.path.join(PROJECT, "tiff_outputs", "dashboard_images")
OUT_DIR  = os.path.join(PROJECT, "tiff_outputs")
HTML_OUT = os.path.join(OUT_DIR, "kaikoura_dashboard_OFFLINE.html")
os.makedirs(OUT_DIR, exist_ok=True)

PIXEL_HA = 1.0 / 10000.0
def tif(f):  return os.path.join(TIF_DIR, f)
def img(f):  return os.path.join(IMG_DIR, f)

def b64img(fname):
    p = img(fname)
    if not os.path.exists(p): return ""
    with open(p,"rb") as fh:
        return "data:image/png;base64,"+base64.b64encode(fh.read()).decode()

def load(fname, band=1):
    p = tif(fname)
    if not os.path.exists(p): return None
    with rasterio.open(p) as src:
        data = src.read(band).astype(np.float32)
        nd   = src.nodata
    if nd is not None: data[data==nd] = np.nan
    for v in [-9999,-32768]: data[data==v] = np.nan
    with np.errstate(invalid="ignore"): data[np.abs(data)>1e6] = np.nan
    return data

px_ha = lambda n: round(n*PIXEL_HA,1)
pct   = lambda n,t: round(100.0*n/t,2) if t else 0.0
fmt   = lambda v,d=2: round(float(v),d) if v is not None and not (isinstance(v,float) and np.isnan(v)) else None

print("\n"+"="*56+"\n  Extracting real statistics …\n"+"="*56)

def elev_stats(fname,label):
    arr=load(fname)
    if arr is None: return {}
    v=arr[~np.isnan(arr)]
    return {"label":label,"min":fmt(v.min()),"max":fmt(v.max()),"mean":fmt(v.mean()),"std":fmt(v.std()),"valid_ha":px_ha(v.size)}

dtm_2012=elev_stats("t1_dtm.tif","DTM 2012"); dtm_2016=elev_stats("t2_dtm.tif","DTM 2016")
d12,d16=load("t1_dtm.tif"),load("t2_dtm.tif")
if d12 is not None and d16 is not None:
    diff=d16-d12; dv=diff[~np.isnan(diff)]
    elev_change={"min":fmt(dv.min()),"max":fmt(dv.max()),"mean":fmt(dv.mean()),
                 "raised_ha":px_ha(int(np.sum(dv>0.1))),"lowered_ha":px_ha(int(np.sum(dv<-0.1)))}
else: elev_change={}

def slope_stats(fname,label):
    arr=load(fname)
    if arr is None: return {}
    v=arr[~np.isnan(arr)]
    return {"label":label,"min":fmt(v.min()),"max":fmt(v.max()),"mean":fmt(v.mean()),"std":fmt(v.std()),
            "flat_ha":px_ha(int(np.sum(v<5))),"gentle_ha":px_ha(int(np.sum((v>=5)&(v<15)))),
            "moderate_ha":px_ha(int(np.sum((v>=15)&(v<30)))),"steep_ha":px_ha(int(np.sum(v>=30)))}
slope_2012=slope_stats("t1_slope.tif","Slope 2012"); slope_2016=slope_stats("t2_slope.tif","Slope 2016")

def chm_stats(fname,label):
    arr=load(fname)
    if arr is None: return {}
    v=arr[~np.isnan(arr)]
    return {"label":label,"min":fmt(v.min()),"max":fmt(v.max()),"mean":fmt(v.mean()),
            "canopy_ha":px_ha(int(np.sum(v>2.0))),"no_canopy_ha":px_ha(int(np.sum(v<=2.0)))}
chm_2012=chm_stats("t1_chm.tif","CHM 2012"); chm_2016=chm_stats("t2_chm.tif","CHM 2016")

def ndvi_stats(fname,label):
    arr=load(fname)
    if arr is None: return {}
    v=arr[~np.isnan(arr)]
    return {"label":label,"min":fmt(v.min()),"max":fmt(v.max()),"mean":fmt(v.mean()),
            "high_veg_ha":px_ha(int(np.sum(v>0.4))),"med_veg_ha":px_ha(int(np.sum((v>0.2)&(v<=0.4)))),
            "low_veg_ha":px_ha(int(np.sum((v>0.0)&(v<=0.2)))),"non_veg_ha":px_ha(int(np.sum(v<=0.0)))}
ndvi_2012=ndvi_stats("NDVI_2012_qgis.tif","NDVI 2012"); ndvi_2016=ndvi_stats("NDVI_2016_qgis.tif","NDVI 2016")

def ndbi_stats(fname,label):
    arr=load(fname)
    if arr is None: return {}
    v=arr[~np.isnan(arr)]
    return {"label":label,"min":fmt(v.min()),"max":fmt(v.max()),"mean":fmt(v.mean()),
            "high_built_ha":px_ha(int(np.sum(v>0.1)))}
ndbi_2012=ndbi_stats("NDBI_2012_qgis.tif","NDBI 2012"); ndbi_2016=ndbi_stats("NDBI_2016_qgis.tif","NDBI 2016")

CL_NAMES=["Bare ground","Grass/low veg","Medium veg","Dense veg","Built/rock"]
CL_COLORS=["#d4b483","#a8d08d","#4daf4a","#1b7837","#808080"]
def lulc_stats(fname,label):
    arr=load(fname)
    if arr is None: return {}
    valid=arr[~np.isnan(arr)]; total=valid.size
    classes=[{"id":i,"name":CL_NAMES[i],"color":CL_COLORS[i],"pixels":int(np.sum(valid==i)),
               "ha":px_ha(int(np.sum(valid==i))),"pct":pct(int(np.sum(valid==i)),total)} for i in range(5)]
    return {"label":label,"total_px":total,"total_ha":px_ha(total),"classes":classes}
lulc_2012=lulc_stats("lulc_2012.tif","LULC 2012"); lulc_2016=lulc_stats("lulc_2016.tif","LULC 2016")
lulc_2020=lulc_stats("predicted_lulc_2020.tif","LULC 2020"); lulc_2024=lulc_stats("predicted_lulc_2024.tif","LULC 2024")
print(f"   Study area: {lulc_2012.get('total_ha')} ha")

def binary_stats(fname,label):
    arr=load(fname)
    if arr is None: return {}
    valid=arr[~np.isnan(arr)]; total=valid.size; changed=int(np.sum(valid==1))
    return {"label":label,"total_ha":px_ha(total),"changed_ha":px_ha(changed),"changed_pct":pct(changed,total),
            "unchanged_ha":px_ha(int(np.sum(valid==0))),"unchanged_pct":pct(int(np.sum(valid==0)),total)}

def gl_stats(fname,label):
    arr=load(fname)
    if arr is None: return {}
    valid=arr[~np.isnan(arr)]; total=valid.size
    gain=int(np.sum(valid==1)); loss=int(np.sum(valid==-1))
    return {"label":label,"gain_ha":px_ha(gain),"gain_pct":pct(gain,total),
            "loss_ha":px_ha(loss),"loss_pct":pct(loss,total),"net_ha":round(px_ha(gain)-px_ha(loss),1)}

def dir_stats(fname,label):
    arr=load(fname)
    if arr is None: return {}
    valid=arr[~np.isnan(arr)]; total=valid.size
    return {"label":label,"bins":{str(v):{"ha":px_ha(int(np.sum(valid==v))),"pct":pct(int(np.sum(valid==v)),total)} for v in range(-4,5)}}

bin_obs=binary_stats("binary_change_map.tif","Bin 2012-16")
bin_1620=binary_stats("binary_change_2016_2020.tif","Bin 2016-20")
bin_2024=binary_stats("binary_change_2020_2024.tif","Bin 2020-24")
gl_obs=gl_stats("gain_loss_map.tif","GL 2012-16")
gl_1620=gl_stats("gain_loss_2016_2020.tif","GL 2016-20")
gl_2024=gl_stats("gain_loss_2020_2024.tif","GL 2020-24")
dir_obs=dir_stats("directional_change_map.tif","Dir 2012-16")
dir_1620=dir_stats("directional_change_2016_2020.tif","Dir 2016-20")
dir_2024=dir_stats("directional_change_2020_2024.tif","Dir 2020-24")
print(f"   Changed 2012-16: {bin_obs.get('changed_ha')} ha")

SEV_NAMES=["No change","Recovery","Medium damage","High damage"]
SEV_COLORS=["#d0d0d0","#1a9641","#fdae61","#d73027"]
def sev_stats(fname,label):
    arr=load(fname)
    if arr is None: return {}
    valid=arr[~np.isnan(arr)]; total=valid.size
    return {"label":label,"total_ha":px_ha(total),
            "classes":[{"id":i,"name":SEV_NAMES[i],"color":SEV_COLORS[i],"ha":px_ha(int(np.sum(valid==i))),"pct":pct(int(np.sum(valid==i)),total)} for i in range(4)]}
sev_obs=sev_stats("damage_severity.tif","Sev 2012-16")
sev_1620=sev_stats("damage_severity_2016_2020.tif","Sev 2016-20")
sev_2024=sev_stats("damage_severity_2020_2024.tif","Sev 2020-24")

a12=load("lulc_2012.tif"); a16=load("lulc_2016.tif")
tm=[]; total_valid=0
if a12 is not None and a16 is not None:
    mask=~(np.isnan(a12)|np.isnan(a16)); f12=a12[mask].astype(int); f16=a16[mask].astype(int); total_valid=int(mask.sum())
    tm=[[int(np.sum((f12==i)&(f16==j))) for j in range(5)] for i in range(5)]

def feat_band_stats(fname):
    p=tif(fname)
    if not os.path.exists(p): return {}
    BNAMES=["DTM","Slope","CHM","NDVI","NDBI"]; result={}
    with rasterio.open(p) as src:
        for b in range(1,min(src.count+1,6)):
            arr=src.read(b).astype(np.float32); nd=src.nodata
            if nd is not None: arr[arr==nd]=np.nan
            arr[arr==-9999]=np.nan; arr[arr==-32768]=np.nan
            v=arr[~np.isnan(arr)]; pv=round(v.size/arr.size*100,1) if arr.size else 0
            nm=BNAMES[b-1] if b<=5 else f"B{b}"
            result[nm]={"min":fmt(v.min()) if v.size else None,"max":fmt(v.max()) if v.size else None,
                        "mean":fmt(v.mean()) if v.size else None,"valid_pct":pv}
    return result
feat_t1=feat_band_stats("t1_feature_stack.tif"); feat_t2=feat_band_stats("t2_feature_stack.tif")
print(f"   Feature stack bands: {list(feat_t1.keys())}")

# ── NEW: load full feature stack TIFs as multi-band images for display ──
def feat_stack_all_bands(fname):
    """Return per-band stats for all bands in the feature stack TIF."""
    p = tif(fname)
    if not os.path.exists(p): return {}
    BNAMES = ["DTM","Slope","CHM","NDVI","NDBI"]
    result = {}
    with rasterio.open(p) as src:
        for b in range(1, src.count + 1):
            arr = src.read(b).astype(np.float32)
            nd  = src.nodata
            if nd is not None: arr[arr == nd] = np.nan
            arr[arr == -9999] = np.nan; arr[arr == -32768] = np.nan
            v   = arr[~np.isnan(arr)]
            pv  = round(v.size / arr.size * 100, 1) if arr.size else 0
            nm  = BNAMES[b-1] if b <= len(BNAMES) else f"B{b}"
            result[nm] = {
                "band": b,
                "min":  fmt(v.min())  if v.size else None,
                "max":  fmt(v.max())  if v.size else None,
                "mean": fmt(v.mean()) if v.size else None,
                "std":  fmt(v.std())  if v.size else None,
                "valid_pct": pv
            }
    return result

feat_t1_full = feat_stack_all_bands("t1_feature_stack.tif")
feat_t2_full = feat_stack_all_bands("t2_feature_stack.tif")

summary={
    "total_ha":lulc_2012.get("total_ha",0),
    "changed_ha":bin_obs.get("changed_ha",0),"changed_pct":bin_obs.get("changed_pct",0),
    "veg_gain_ha":gl_obs.get("gain_ha",0),"veg_loss_ha":gl_obs.get("loss_ha",0),
    "net_veg_ha":gl_obs.get("net_ha",0),
    "high_dmg_ha":sev_obs["classes"][3]["ha"] if sev_obs.get("classes") else 0,
    "med_dmg_ha":sev_obs["classes"][2]["ha"] if sev_obs.get("classes") else 0,
    "total_valid_px":total_valid,
}
STATS={
    "dtm_2012":dtm_2012,"dtm_2016":dtm_2016,"elev_change":elev_change,
    "slope_2012":slope_2012,"slope_2016":slope_2016,"chm_2012":chm_2012,"chm_2016":chm_2016,
    "ndvi_2012":ndvi_2012,"ndvi_2016":ndvi_2016,"ndbi_2012":ndbi_2012,"ndbi_2016":ndbi_2016,
    "lulc_2012":lulc_2012,"lulc_2016":lulc_2016,"lulc_2020":lulc_2020,"lulc_2024":lulc_2024,
    "bin_obs":bin_obs,"bin_1620":bin_1620,"bin_2024":bin_2024,
    "gl_obs":gl_obs,"gl_1620":gl_1620,"gl_2024":gl_2024,
    "dir_obs":dir_obs,"dir_1620":dir_1620,"dir_2024":dir_2024,
    "sev_obs":sev_obs,"sev_1620":sev_1620,"sev_2024":sev_2024,
    "transition_matrix":tm,
    "feat_t1":feat_t1,"feat_t2":feat_t2,
    "feat_t1_full":feat_t1_full,"feat_t2_full":feat_t2_full,
    "summary":summary
}
STATS_JSON=json.dumps(STATS)

print("\n  Embedding images as base64 …")
ALL_IMGS=[
    "t1_dtm.png","t2_dtm.png","elevation_change_t1t2.png","elevation_change.png",
    "t1_dsm.png","t2_dsm.png","t1_slope.png","t2_slope.png","t1_chm.png","t2_chm.png",
    "ndvi_2012.png","ndvi_2016.png",
    "ndbi_2012.png","ndbi_2016.png",
    "lulc_2012.png","lulc_2016.png","lulc_2020.png","lulc_2024.png",
    "binary_change.png","binary_change_2016_2020.png","binary_change_2020_2024.png",
    "gain_loss.png","gain_loss_2016_2020.png","gain_loss_2020_2024.png",
    "directional_change.png","directional_change_2016_2020.png","directional_change_2020_2024.png",
    "predicted_change_2016_2020.png","predicted_change_2016_2024.png",
    "damage_severity.png","damage_severity_2016_2020.png","damage_severity_2020_2024.png",
    "feature_fusion_t1_2012.png","feature_fusion_t2_2016.png",
    "fusion_compare_band1_dtm.png","fusion_compare_band2_slope.png",
    "fusion_compare_band3_chm.png","fusion_compare_band4_ndvi.png","fusion_compare_band5_ndbi.png",
    "feat_t1_band1_dtm.png","feat_t1_band2_slope.png","feat_t1_band3_chm.png","feat_t1_band4_ndvi.png","feat_t1_band5_ndbi.png",
    "feat_t2_band1_dtm.png","feat_t2_band2_slope.png","feat_t2_band3_chm.png","feat_t2_band4_ndvi.png","feat_t2_band5_ndbi.png",
    # NEW: full feature stack TIF renders
    "t1_feature_stack_rgb.png","t2_feature_stack_rgb.png",
    "t1_feature_stack_band1.png","t1_feature_stack_band2.png","t1_feature_stack_band3.png","t1_feature_stack_band4.png","t1_feature_stack_band5.png",
    "t2_feature_stack_band1.png","t2_feature_stack_band2.png","t2_feature_stack_band3.png","t2_feature_stack_band4.png","t2_feature_stack_band5.png",
    "laz_compare_2012_2016.png","laz_planview_t1_2012.png","laz_planview_t2_2016.png",
    "laz_3d_t1_2012.png","laz_3d_t2_2016.png","laz_profile_t1_2012.png","laz_profile_t2_2016.png",
    "confusion_2012_2016.png","confusion_2016_2020.png","confusion_2020_2024.png",
]
B64={}; found=0
for fn in ALL_IMGS:
    s=b64img(fn); B64[fn]=s
    if s: found+=1
print(f"  Embedded {found}/{len(ALL_IMGS)} images")

def I(fname,alt=""):
    s=B64.get(fname,"")
    if not s: return f'<div class="miss"><span>📂</span><small>{fname}</small></div>'
    return f'<img src="{s}" alt="{alt}" loading="lazy" style="width:100%;display:block">'

def IC(title,badge,bcls,fname,foot="",sk=""):
    return f'<div class="ic" onclick=\'openM("{fname}","{title}","{sk}")\'><div class="ic-hdr"><h3>{title}</h3><span class="badge {bcls}">{badge}</span></div><div class="ic-img">{I(fname,title)}</div>{"<div class=ic-foot>"+foot+"</div>" if foot else ""}</div>'

def LD(pairs): return "".join(f'<div class="li"><div class="ld" style="background:{c}"></div>{l}</div>' for c,l in pairs)
LLEG=LD(zip(CL_COLORS,CL_NAMES)); SLEG=LD(zip(SEV_COLORS,SEV_NAMES))
BLEG=LD([("#2a2a2a","Unchanged"),("#e31a1c","Changed")]); GLEG=LD([("#d73027","Loss"),("#555","Stable"),("#1a9641","Gain")]); DLEG=LD([("#d73027","Decrease"),("#1a9641","Increase")])

# ════════════════════════════════════════════════════════════════
#  PROFESSIONAL REDESIGN — Refined Scientific Dark Theme
#  Aesthetic: High-precision cartographic instrument panel
#  Fonts: DM Serif Display (headings) + IBM Plex Mono (data)
# ════════════════════════════════════════════════════════════════
CSS="""
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=IBM+Plex+Mono:wght@300;400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
  --bg:     #080c10;
  --bg1:    #0d1219;
  --bg2:    #111922;
  --bg3:    #16212e;
  --bdr:    #1d2d3f;
  --bdr2:   #243347;
  --acc:    #4fc3f7;
  --acc2:   #0288d1;
  --grn:    #43d9ad;
  --red:    #f06292;
  --org:    #ffb74d;
  --pur:    #ce93d8;
  --yel:    #fff176;
  --txt:    #cdd8e8;
  --txt2:   #8fa3bc;
  --txt3:   #4d6680;
  --serif:  'DM Serif Display', Georgia, serif;
  --mono:   'IBM Plex Mono', 'Courier New', monospace;
  --sans:   'IBM Plex Sans', system-ui, sans-serif;
}

*, *::before, *::after { margin:0; padding:0; box-sizing:border-box; }

html { scroll-behavior: smooth; }

body {
  background: var(--bg);
  color: var(--txt);
  font-family: var(--sans);
  overflow-x: hidden;
  min-height: 100vh;
}

/* ── Topographic grid background ── */
body::before {
  content: '';
  position: fixed; inset: 0;
  background-image:
    linear-gradient(rgba(79,195,247,.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(79,195,247,.025) 1px, transparent 1px),
    linear-gradient(rgba(79,195,247,.012) 1px, transparent 1px),
    linear-gradient(90deg, rgba(79,195,247,.012) 1px, transparent 1px);
  background-size: 80px 80px, 80px 80px, 16px 16px, 16px 16px;
  pointer-events: none; z-index: 0;
}

/* ── Header ── */
header {
  position: relative; z-index: 20;
  background: linear-gradient(180deg, rgba(4,10,17,.98) 0%, rgba(8,12,16,.96) 100%);
  border-bottom: 1px solid var(--bdr2);
  padding: 0 40px;
}

.hdr-top {
  display: flex; align-items: flex-start;
  justify-content: space-between; gap: 20px;
  padding: 24px 0 16px;
  border-bottom: 1px solid var(--bdr);
  flex-wrap: wrap;
}

.hdr-title { flex: 1; min-width: 260px; }

.hdr-eyebrow {
  font-family: var(--mono);
  font-size: 9px; font-weight: 600;
  letter-spacing: 2.5px;
  text-transform: uppercase;
  color: var(--acc2);
  margin-bottom: 6px;
  display: flex; align-items: center; gap: 8px;
}
.hdr-eyebrow::before {
  content: '';
  display: inline-block;
  width: 18px; height: 1px;
  background: var(--acc2);
}

.hdr-title h1 {
  font-family: var(--serif);
  font-size: clamp(18px, 2.8vw, 32px);
  font-weight: 400;
  line-height: 1.15;
  color: #e8f4fd;
  letter-spacing: -.3px;
}
.hdr-title h1 em {
  font-style: italic;
  color: var(--acc);
}
.hdr-title h1 strong {
  font-style: normal;
  color: #fff;
  font-weight: 400;
}

.hdr-meta {
  font-family: var(--mono);
  font-size: 9px; color: var(--txt3);
  margin-top: 7px; letter-spacing: .5px;
}

.hdr-right {
  display: flex; flex-direction: column;
  align-items: flex-end; gap: 8px;
}

.pills { display: flex; gap: 5px; flex-wrap: wrap; justify-content: flex-end; }

.pill {
  padding: 3px 10px;
  border-radius: 2px;
  font-size: 9px; font-family: var(--mono); font-weight: 600;
  letter-spacing: .8px; border: 1px solid;
}
.pb { color: var(--acc);  border-color: rgba(79,195,247,.35);  background: rgba(79,195,247,.07); }
.pr { color: var(--red);  border-color: rgba(240,98,146,.35);  background: rgba(240,98,146,.07); }
.pg { color: var(--grn);  border-color: rgba(67,217,173,.35);  background: rgba(67,217,173,.07); }
.po { color: var(--org);  border-color: rgba(255,183,77,.35);  background: rgba(255,183,77,.07); }

.hdr-kpis {
  display: flex; gap: 0;
  border: 1px solid var(--bdr2);
  border-radius: 3px; overflow: hidden;
}
.kpi {
  padding: 8px 16px; text-align: center;
  border-right: 1px solid var(--bdr);
  background: rgba(13,18,25,.6);
}
.kpi:last-child { border-right: none; }
.kpi .kv {
  font-family: var(--mono); font-size: 16px; font-weight: 700;
  color: var(--acc); line-height: 1;
  display: flex; align-items: baseline; gap: 2px;
}
.kpi .kv span { font-size: 9px; color: var(--acc2); font-weight: 400; }
.kpi .kl { font-size: 8px; color: var(--txt3); margin-top: 3px; font-family: var(--mono); text-transform: uppercase; letter-spacing: .7px; }

/* ── Navigation tabs ── */
.tabs {
  display: flex; gap: 0;
  padding: 12px 0 0;
  overflow-x: auto; scrollbar-width: none;
}
.tabs::-webkit-scrollbar { display: none; }

.tab {
  padding: 7px 16px;
  background: transparent;
  border: none;
  border-bottom: 2px solid transparent;
  color: var(--txt3);
  font-family: var(--mono); font-size: 9px; font-weight: 600;
  letter-spacing: 1px; text-transform: uppercase;
  cursor: pointer; white-space: nowrap;
  transition: all .15s; position: relative;
}
.tab:hover { color: var(--txt); border-bottom-color: var(--bdr2); }
.tab.active { color: var(--acc); border-bottom-color: var(--acc); }

/* ── Main content ── */
main {
  position: relative; z-index: 10;
  padding: 0 40px 64px;
  border-top: 1px solid var(--bdr);
}

.panel { display: none; }
.panel.active { display: block; animation: fadeUp .22s ease; }
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* ── Section header ── */
.sh {
  display: flex; align-items: center; gap: 14px;
  padding: 26px 0 14px;
}
.sh-icon {
  width: 32px; height: 32px;
  background: var(--bg3); border: 1px solid var(--bdr2);
  border-radius: 4px; display: flex; align-items: center;
  justify-content: center; font-size: 13px; flex-shrink: 0;
}
.sh-text h2 {
  font-family: var(--serif); font-size: 18px; font-weight: 400;
  color: #dce9f5; letter-spacing: -.2px;
}
.sh-text small {
  font-size: 9px; font-family: var(--mono);
  color: var(--txt3); letter-spacing: .5px;
}
.sh .ln { flex: 1; height: 1px; background: linear-gradient(90deg, var(--bdr2) 0%, transparent 100%); }

/* ── Sub-section header ── */
.ssh {
  font-family: var(--mono); font-size: 9px; font-weight: 600;
  letter-spacing: 1.5px; text-transform: uppercase;
  color: var(--txt3); padding: 14px 0 8px;
  border-bottom: 1px solid var(--bdr);
  margin-bottom: 10px;
  display: flex; align-items: center; gap: 8px;
}
.ssh::before {
  content: '';
  width: 3px; height: 12px;
  background: var(--acc); border-radius: 1px;
  flex-shrink: 0;
}

/* ── Stat cards ── */
.sc-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 8px; margin-bottom: 18px;
}

.sc {
  background: var(--bg1);
  border: 1px solid var(--bdr);
  border-radius: 4px;
  padding: 14px 14px 12px;
  position: relative; overflow: hidden;
  transition: border-color .15s, transform .15s;
}
.sc::after {
  content: '';
  position: absolute; bottom: 0; left: 0; right: 0; height: 2px;
  background: var(--ca, var(--acc)); opacity: .6;
}
.sc:hover { border-color: var(--ca, var(--acc)); transform: translateY(-2px); }
.sc .lbl {
  font-size: 8px; font-family: var(--mono);
  color: var(--txt3); letter-spacing: 1.2px;
  text-transform: uppercase; margin-bottom: 6px;
}
.sc .val {
  font-size: 22px; font-weight: 700;
  font-family: var(--mono);
  color: var(--ca, var(--acc)); line-height: 1;
}
.sc .sub {
  font-size: 8px; color: var(--txt3);
  margin-top: 4px; font-family: var(--mono);
}

/* ── Image grid ── */
.ig { display: grid; gap: 10px; margin-bottom: 14px; }
.ig1  { grid-template-columns: 1fr; }
.ig2  { grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }
.ig3  { grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }
.ig4  { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
.ig5  { grid-template-columns: repeat(5, 1fr); }

/* ── Image card ── */
.ic {
  background: var(--bg1);
  border: 1px solid var(--bdr);
  border-radius: 4px; overflow: hidden;
  cursor: pointer;
  transition: border-color .15s, transform .15s, box-shadow .15s;
}
.ic:hover {
  border-color: var(--acc);
  transform: translateY(-2px);
  box-shadow: 0 8px 28px rgba(79,195,247,.1);
}
.ic-hdr {
  padding: 8px 12px;
  border-bottom: 1px solid var(--bdr);
  display: flex; align-items: center;
  justify-content: space-between; gap: 6px;
  background: var(--bg2);
}
.ic-hdr h3 {
  font-size: 9px; font-weight: 500;
  font-family: var(--mono); color: var(--txt2);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.badge {
  font-size: 7px; font-family: var(--mono);
  padding: 2px 7px; border-radius: 2px; font-weight: 700;
  letter-spacing: .5px; flex-shrink: 0;
}
.bobs  { background: rgba(79,195,247,.12); color: var(--acc); }
.bpred { background: rgba(240,98,146,.12); color: var(--red); }
.bchg  { background: rgba(67,217,173,.12); color: var(--grn); }
.braw  { background: rgba(255,183,77,.12);  color: var(--org); }
.blaz  { background: rgba(206,147,216,.12); color: var(--pur); }
.bfeat { background: rgba(255,241,118,.10); color: var(--yel); }

.ic-img {
  overflow: hidden; background: #040810;
  min-height: 120px; position: relative;
}
.ic-img img {
  width: 100%; display: block;
  transition: transform .3s ease;
}
.ic:hover .ic-img img { transform: scale(1.04); }

.miss {
  display: flex; align-items: center;
  justify-content: center; min-height: 110px;
  color: var(--txt3); font-size: 8px; font-family: var(--mono);
  flex-direction: column; gap: 5px; padding: 12px; text-align: center;
}
.miss span { font-size: 22px; }

.ic-foot {
  padding: 6px 10px; display: flex; gap: 6px;
  flex-wrap: wrap; border-top: 1px solid var(--bdr);
  background: var(--bg2);
}

/* ── Legend items ── */
.li { display: flex; align-items: center; gap: 4px; font-size: 7px; color: var(--txt3); font-family: var(--mono); }
.ld { width: 8px; height: 8px; border-radius: 1px; flex-shrink: 0; }

/* ── Tables ── */
.tw {
  background: var(--bg1);
  border: 1px solid var(--bdr);
  border-radius: 4px; overflow: hidden;
  margin-bottom: 14px; overflow-x: auto;
}
.tw-hdr {
  padding: 10px 16px;
  border-bottom: 1px solid var(--bdr);
  background: var(--bg2);
  display: flex; align-items: center; justify-content: space-between;
}
.tw-hdr h3 {
  font-size: 9px; font-weight: 600; font-family: var(--mono);
  color: var(--acc); text-transform: uppercase; letter-spacing: 1px;
}
.tw-hdr small { font-size: 8px; color: var(--txt3); font-family: var(--mono); }

table { width: 100%; border-collapse: collapse; min-width: 460px; }
thead th {
  padding: 7px 12px; text-align: left;
  font-family: var(--mono); font-size: 8px;
  letter-spacing: 1px; text-transform: uppercase;
  color: var(--txt3); background: var(--bg2);
  border-bottom: 1px solid var(--bdr2);
}
tbody tr { border-bottom: 1px solid var(--bdr); transition: background .1s; }
tbody tr:last-child { border-bottom: none; }
tbody tr:hover { background: rgba(79,195,247,.03); }
tbody td { padding: 7px 12px; font-family: var(--mono); font-size: 9px; }

.tp { color: var(--grn); }
.tn { color: var(--red); }
.tm { color: var(--txt3); }
.to { color: var(--org); }

/* ── Chip ── */
.chip {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 2px 7px; border-radius: 2px;
  font-size: 8px; background: rgba(255,255,255,.04);
  border: 1px solid var(--bdr); font-family: var(--mono);
}

/* ── Transition matrix ── */
.mw {
  overflow-x: auto;
  background: var(--bg1); border: 1px solid var(--bdr);
  border-radius: 4px; margin-bottom: 14px;
}
.mw-hdr { padding: 10px 16px; border-bottom: 1px solid var(--bdr); background: var(--bg2); }
.mw-hdr h3 {
  font-size: 9px; font-weight: 600; font-family: var(--mono);
  color: var(--acc); text-transform: uppercase; letter-spacing: 1px;
}
.mt { min-width: 520px; }
.mt th, .mt td {
  padding: 6px 10px; text-align: center;
  font-family: var(--mono); font-size: 8px;
  border: 1px solid var(--bdr);
}
.mt th { background: var(--bg2); color: var(--txt3); }
.mt .rh { text-align: left; background: var(--bg2); color: var(--txt2); }
.mhi { background: rgba(79,195,247,.18);  color: var(--acc);  font-weight: 700; }
.mmd { background: rgba(79,195,247,.07); }
.mlo { background: rgba(240,98,146,.07); color: var(--red); }

/* ── Bar charts ── */
.bc {
  background: var(--bg1); border: 1px solid var(--bdr);
  border-radius: 4px; padding: 14px 16px;
  margin-bottom: 14px;
}
.bc h3 {
  font-size: 9px; font-weight: 600; font-family: var(--mono);
  color: var(--acc); text-transform: uppercase; letter-spacing: 1px;
  margin-bottom: 12px;
}
.br { display: flex; align-items: center; gap: 8px; margin-bottom: 7px; }
.bl { width: 110px; font-size: 8px; color: var(--txt3); text-align: right; flex-shrink: 0; font-family: var(--mono); }
.bt { flex: 1; height: 18px; background: var(--bg3); border-radius: 2px; overflow: hidden; }
.bf {
  height: 100%; border-radius: 2px;
  display: flex; align-items: center; padding-left: 5px;
  min-width: 3px; transition: width .8s cubic-bezier(.4,0,.2,1);
}
.bf span { font-size: 8px; font-family: var(--mono); color: rgba(0,0,0,.85); font-weight: 600; white-space: nowrap; }

/* ── Feature stack section ── */
.fs-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 10px; margin-bottom: 12px;
}
.fs-card {
  background: var(--bg1); border: 1px solid var(--bdr);
  border-radius: 4px; overflow: hidden;
}
.fs-card-hdr {
  padding: 7px 12px; background: var(--bg2);
  border-bottom: 1px solid var(--bdr);
  display: flex; align-items: center; justify-content: space-between;
}
.fs-card-hdr span {
  font-family: var(--mono); font-size: 8px; color: var(--txt2); font-weight: 600;
}
.fs-stats {
  display: flex; gap: 0;
  border-top: 1px solid var(--bdr); padding: 6px 12px; flex-wrap: wrap; gap: 10px;
}
.fs-stat { }
.fs-stat .fv { font-family: var(--mono); font-size: 11px; font-weight: 700; color: var(--acc); }
.fs-stat .fl { font-family: var(--mono); font-size: 7px; color: var(--txt3); text-transform: uppercase; letter-spacing: .7px; }

/* ── Accuracy section ── */
.acc-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 14px;
}
@media(max-width:800px){ .acc-grid { grid-template-columns: 1fr; } }

/* ── Tabs sub (period switcher) ── */
.ptabs { display: flex; gap: 4px; margin-bottom: 12px; }
.ptab {
  padding: 4px 12px; background: var(--bg2);
  border: 1px solid var(--bdr); border-radius: 3px;
  font-family: var(--mono); font-size: 8px; font-weight: 600;
  color: var(--txt3); cursor: pointer; letter-spacing: .5px;
  transition: all .12s;
}
.ptab:hover { color: var(--txt); border-color: var(--bdr2); }
.ptab.active { background: rgba(79,195,247,.1); color: var(--acc); border-color: var(--acc); }

/* ── Modal ── */
.mo {
  display: none; position: fixed; inset: 0;
  background: rgba(0,0,0,.92); z-index: 2000;
  align-items: center; justify-content: center;
  backdrop-filter: blur(6px);
}
.mo.open { display: flex; }
.md {
  background: var(--bg1); border: 1px solid var(--bdr2);
  border-radius: 6px; max-width: 94vw; max-height: 94vh;
  overflow: auto; padding: 18px; position: relative; min-width: 260px;
}
.mc {
  position: absolute; top: 12px; right: 12px;
  background: var(--bg2); border: 1px solid var(--bdr);
  color: var(--txt2); width: 26px; height: 26px; border-radius: 3px;
  cursor: pointer; font-size: 11px; display: flex;
  align-items: center; justify-content: center; transition: background .12s;
  font-family: var(--mono);
}
.mc:hover { background: var(--red); color: #fff; }
.md img { display: block; max-width: 88vw; max-height: 78vh; border-radius: 4px; margin-top: 10px; object-fit: contain; width: 100%; }
.md .sr { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--bdr); }
.md .ms .mv { font-size: 16px; font-weight: 700; font-family: var(--mono); color: var(--acc); }
.md .ms .ml { font-size: 7px; color: var(--txt3); font-family: var(--mono); margin-top: 2px; text-transform: uppercase; letter-spacing: .7px; }

/* ── Divider ── */
.divider {
  height: 1px; background: var(--bdr);
  margin: 20px 0;
}

/* ── Accuracy combined layout ── */
.confusion-imgs {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 10px; margin-bottom: 14px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--bdr2); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--acc2); }

/* ── Footer ── */
footer {
  position: relative; z-index: 10;
  text-align: center; padding: 14px;
  border-top: 1px solid var(--bdr);
  font-size: 8px; font-family: var(--mono);
  color: var(--txt3); letter-spacing: 1px;
}
footer span { color: var(--acc2); }

@media(max-width: 600px) {
  header, main { padding-left: 14px; padding-right: 14px; }
  .tabs { padding-left: 14px; }
  .ig2,.ig3,.ig4,.ig5 { grid-template-columns: 1fr; }
  .acc-grid { grid-template-columns: 1fr; }
  .hdr-kpis { display: none; }
}
"""

JS=r"""
const S = __STATS__;
const CL  = ["Bare ground","Grass/low veg","Medium veg","Dense veg","Built/rock"];
const CC  = ["#d4b483","#a8d08d","#4daf4a","#1b7837","#808080"];
const SN  = ["No change","Recovery","Medium damage","High damage"];
const SC2 = ["#d0d0d0","#1a9641","#fdae61","#d73027"];
const BN  = ["DTM","Slope","CHM","NDVI","NDBI"];

// ── Tab switching ──
function sw(id, btn) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
  document.getElementById('p-' + id).classList.add('active');
  btn.classList.add('active');
}

// ── Modal ──
function openM(fname, title, sk) {
  const img = document.querySelector(`.ic img[alt="${title}"]`);
  document.getElementById('m-img').src = img ? img.src : '';
  document.getElementById('m-title').textContent = title;
  const sr = document.getElementById('m-stats'); sr.innerHTML = '';
  const d = sk && S[sk];
  if (d) {
    const add = (v, l) => {
      const e = document.createElement('div'); e.className = 'ms';
      e.innerHTML = `<div class="mv">${v}</div><div class="ml">${l}</div>`;
      sr.appendChild(e);
    };
    if (d.mean != null)    add(d.mean + 'm', 'Mean');
    if (d.min != null)     add(d.min  + 'm', 'Min');
    if (d.max != null)     add(d.max  + 'm', 'Max');
    if (d.total_ha)        add(d.total_ha + ' ha', 'Area');
    if (d.changed_ha)      add(d.changed_ha + ' ha', 'Changed');
    if (d.gain_ha)         add(d.gain_ha  + ' ha', 'Veg Gain');
    if (d.loss_ha)         add(d.loss_ha  + ' ha', 'Veg Loss');
    if (d.net_ha != null)  add((d.net_ha > 0 ? '+' : '') + d.net_ha + ' ha', 'Net Veg');
  }
  document.getElementById('modal').classList.add('open');
}
function cmo(e) { if (e.target === document.getElementById('modal')) cmd(); }
function cmd()  { document.getElementById('modal').classList.remove('open'); }
document.addEventListener('keydown', e => { if (e.key === 'Escape') cmd(); });

// ── Helpers ──
const sc  = (ca, lbl, val, sub) =>
  `<div class="sc" style="--ca:${ca}"><div class="lbl">${lbl}</div><div class="val">${val}</div><div class="sub">${sub}</div></div>`;
const sH  = (id, h) => { const e = document.getElementById(id); if (e) e.innerHTML = h; };
const sT  = (id, t) => { const e = document.getElementById(id); if (e) e.textContent = t; };
const ds  = (a, b) => {
  if (a == null || b == null || a === '—' || b === '—') return '—';
  const d = Math.round((+b - +a) * 10) / 10;
  return d > 0 ? `<span class="tp">+${d}</span>` : `<span class="tn">${d}</span>`;
};

// ── Transition matrix builder ──
function bldMatrix(id) {
  const tm = S.transition_matrix;
  if (!tm || !tm.length) return;
  let h = '<thead><tr><th>From / To</th>' + CL.map(c => `<th>${c}</th>`).join('') + '<th>Total</th></tr></thead><tbody>';
  tm.forEach((row, i) => {
    const tot = row.reduce((a, b) => a + b, 0);
    h += `<tr><td class="rh">${CL[i]}</td>`;
    row.forEach((v, j) => {
      const p = tot > 0 ? (v / tot * 100) : 0;
      const cls = i === j ? 'mhi' : p > 8 ? 'mmd' : p < 1 && v > 0 ? 'mlo' : '';
      h += `<td class="${cls}">${v.toLocaleString()}<br><span style="font-size:6px;color:var(--txt3)">${p.toFixed(1)}%</span></td>`;
    });
    h += `<td class="tm">${tot.toLocaleString()}</td></tr>`;
  });
  sH(id, h + '</tbody>');
}

// ── Overview ──
function bldOverview() {
  const sm = S.summary, gl = S.gl_obs, sv = S.sev_obs;
  // top KPI pills
  const pillHa  = document.getElementById('pill-ha');
  const pillChg = document.getElementById('pill-chg');
  const kpiHa   = document.getElementById('kpi-ha');
  const kpiChg  = document.getElementById('kpi-chg');
  const kpiGain = document.getElementById('kpi-gain');
  const kpiLoss = document.getElementById('kpi-loss');
  if (pillHa)  pillHa.textContent  = sm.total_ha + ' HA';
  if (pillChg) pillChg.textContent = sm.changed_pct + '% CHG';
  if (kpiHa)   kpiHa.textContent   = sm.total_ha;
  if (kpiChg)  kpiChg.textContent  = sm.changed_pct + '%';
  if (kpiGain) kpiGain.textContent = sm.veg_gain_ha;
  if (kpiLoss) kpiLoss.textContent = sm.veg_loss_ha;

  sH('ov-stats',
    sc('var(--acc)',  'Study Area',  sm.total_ha  + ' ha', 'total extent') +
    sc('var(--red)',  'Changed',     sm.changed_ha + ' ha', sm.changed_pct + '% of area') +
    sc('var(--grn)',  'Veg Gain',    sm.veg_gain_ha + ' ha', '2012 → 2016') +
    sc('var(--red)',  'Veg Loss',    sm.veg_loss_ha + ' ha', '2012 → 2016') +
    sc('var(--grn)',  'Net Veg',     (sm.net_veg_ha > 0 ? '+' : '') + sm.net_veg_ha + ' ha', 'net change') +
    sc('var(--org)',  'Med Damage',  sm.med_dmg_ha  + ' ha', 'class 2') +
    sc('var(--red)',  'High Damage', sm.high_dmg_ha + ' ha', 'class 3–4') +
    sc('var(--grn)',  'Recovery',    sv && sv.classes ? sv.classes[1].ha + ' ha' : '—', 'revegetation'));

  sT('trans-total', 'Valid pixels: ' + S.summary.total_valid_px.toLocaleString());
  const tm = S.transition_matrix, tv = sm.total_valid_px;
  const TR = [
    [3,0,'Dense veg → Bare',   'Landslide scarring'],
    [3,1,'Dense veg → Grass',  'Canopy removal'],
    [1,0,'Grass → Bare',       'Slope failure'],
    [4,3,'Built → Dense veg',  'Revegetation'],
    [0,3,'Bare → Dense veg',   'Veg recovery'],
    [0,1,'Bare → Grass',       'Colonisation'],
    [2,3,'Medium → Dense',     'Succession'],
  ];
  sH('trans-table', TR.map(([fi, ti, lbl, interp]) => {
    const px = tm.length ? tm[fi][ti] : 0;
    const ha = (px * .0001).toFixed(1);
    const p  = (px / tv * 100).toFixed(2);
    const gain = ti > fi;
    return `<tr>
      <td><span class="chip"><span class="ld" style="background:${CC[fi]}"></span>${lbl}</span></td>
      <td class="${gain ? 'tp' : 'tn'}">${px.toLocaleString()}</td>
      <td class="${gain ? 'tp' : 'tn'}">${ha}</td>
      <td class="tm">${p}%</td>
      <td class="tm">${interp}</td>
    </tr>`;
  }).join(''));

  const lulcs  = [S.lulc_2012, S.lulc_2016, S.lulc_2020, S.lulc_2024];
  const yrs    = ['2012', '2016', '2020*', '2024*'];
  const mxH    = Math.max(...CL.map((_, i) => Math.max(...lulcs.filter(Boolean).map(l => l.classes ? l.classes[i].ha : 0))));
  let bh = '';
  CL.forEach((nm, ci) => {
    bh += `<div style="margin-bottom:12px">
      <div style="font-size:8px;color:var(--txt3);margin-bottom:4px;font-family:var(--mono);text-transform:uppercase;letter-spacing:.8px">${nm}</div>`;
    lulcs.forEach((l, yi) => {
      if (!l || !l.classes) return;
      const ha = l.classes[ci].ha;
      const w  = Math.max((ha / mxH * 100), .4).toFixed(1);
      bh += `<div class="br">
        <div class="bl">${yrs[yi]}</div>
        <div class="bt"><div class="bf" style="width:${w}%;background:${CC[ci]}cc"><span>${ha} ha</span></div></div>
      </div>`;
    });
    bh += '</div>';
  });
  sH('lulc-bars', bh);
}

// ── LiDAR ──
function bldLidar() {
  const d12=S.dtm_2012, d16=S.dtm_2016, ec=S.elev_change, s12=S.slope_2012, s16=S.slope_2016, c12=S.chm_2012, c16=S.chm_2016;
  sH('lidar-stats',
    sc('var(--acc)', 'DTM 2012 Mean', (d12.mean||'—')+'m', '') +
    sc('var(--acc)', 'DTM 2016 Mean', (d16.mean||'—')+'m', '') +
    sc('var(--grn)', 'Raised',   (ec.raised_ha||'—')+' ha', '>0.1m') +
    sc('var(--red)', 'Lowered',  (ec.lowered_ha||'—')+' ha', '<−0.1m') +
    sc('var(--org)', 'Slope 12', (s12.mean||'—')+'°', '') +
    sc('var(--org)', 'Slope 16', (s16.mean||'—')+'°', '') +
    sc('var(--grn)', 'Canopy 12',(c12.canopy_ha||'—')+' ha', 'CHM>2m') +
    sc('var(--grn)', 'Canopy 16',(c16.canopy_ha||'—')+' ha', 'CHM>2m'));

  const layers = [['DTM 2012',d12],['DTM 2016',d16],['Slope 2012',s12],['Slope 2016',s16],['CHM 2012',c12],['CHM 2016',c16]];
  sH('lidar-table', layers.map(([nm,d]) =>
    d && d.min != null
      ? `<tr><td>${nm}</td><td class="tm">${d.min}</td><td class="tm">${d.max}</td><td class="tp">${d.mean}</td><td class="tm">${d.std||'—'}</td><td class="tm">${d.valid_ha||'—'}</td></tr>`
      : `<tr><td>${nm}</td><td colspan="5" class="tm">—</td></tr>`
  ).join(''));

  const cats = [['Flat','0–5°','flat_ha'],['Gentle','5–15°','gentle_ha'],['Moderate','15–30°','moderate_ha'],['Steep','≥30°','steep_ha']];
  sH('slope-table', cats.map(([nm,rng,k]) =>
    `<tr><td>${nm}</td><td class="tm">${rng}</td><td>${s12[k]??'—'}</td><td>${s16[k]??'—'}</td><td>${ds(s12[k],s16[k])}</td></tr>`
  ).join(''));
}

// ── LAZ ──
function bldLAZ() {
  const c12=S.chm_2012, c16=S.chm_2016;
  sH('laz-stats',
    sc('var(--acc)', 'Max Canopy 12', (c12.max||'—')+'m', '') +
    sc('var(--acc)', 'Max Canopy 16', (c16.max||'—')+'m', '') +
    sc('var(--grn)', 'Canopy 12', (c12.canopy_ha||'—')+' ha', 'CHM>2m') +
    sc('var(--grn)', 'Canopy 16', (c16.canopy_ha||'—')+' ha', 'CHM>2m') +
    sc('var(--org)', 'Mean Ht 12', (c12.mean||'—')+'m', '') +
    sc('var(--org)', 'Mean Ht 16', (c16.mean||'—')+'m', ''));
  sH('chm-table',
    `<tr><td>CHM 2012</td><td class="tm">${c12.max||'—'}</td><td class="tm">${c12.mean||'—'}</td><td class="tp">${c12.canopy_ha||'—'}</td><td class="tm">${c12.no_canopy_ha||'—'}</td></tr>` +
    `<tr><td>CHM 2016</td><td class="tm">${c16.max||'—'}</td><td class="tm">${c16.mean||'—'}</td><td class="tp">${c16.canopy_ha||'—'}</td><td class="tm">${c16.no_canopy_ha||'—'}</td></tr>`);
}

// ── Feature Fusion ── (enhanced with t1/t2 stack data)
function bldFusion() {
  const f1=S.feat_t1, f2=S.feat_t2;
  const f1f=S.feat_t1_full||{}, f2f=S.feat_t2_full||{};

  sH('feat-stats',
    BN.slice(0,3).map(nm => {
      const b=f1[nm]; return b ? sc('var(--acc)', nm+' 2012', b.mean??'—', 'valid '+b.valid_pct+'%') : '';
    }).join('') +
    BN.slice(0,3).map(nm => {
      const b=f2[nm]; return b ? sc('var(--grn)', nm+' 2016', b.mean??'—', 'valid '+b.valid_pct+'%') : '';
    }).join(''));

  // Feature stack TIF per-band cards
  let stackCards = '';
  BN.forEach((nm, i) => {
    const b1 = f1f[nm]||{}, b2 = f2f[nm]||{};
    stackCards += `
      <div class="fs-card">
        <div class="fs-card-hdr">
          <span>B${i+1} · ${nm}</span>
          <span class="badge bfeat">STACK</span>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0;border-top:1px solid var(--bdr)">
          <div style="padding:8px 10px;border-right:1px solid var(--bdr)">
            <div style="font-size:7px;color:var(--txt3);font-family:var(--mono);margin-bottom:4px;letter-spacing:.8px">T1 · 2012</div>
            <div class="fs-stats" style="padding:0;flex-direction:column;gap:2px">
              <div class="fs-stat"><span class="fv">${b1.mean??'—'}</span><span class="fl"> mean</span></div>
              <div style="font-size:7px;color:var(--txt3);font-family:var(--mono)">${b1.min??'—'} → ${b1.max??'—'}</div>
              <div style="font-size:7px;color:var(--acc2);font-family:var(--mono)">${b1.valid_pct??'—'}% valid</div>
            </div>
          </div>
          <div style="padding:8px 10px">
            <div style="font-size:7px;color:var(--txt3);font-family:var(--mono);margin-bottom:4px;letter-spacing:.8px">T2 · 2016</div>
            <div class="fs-stats" style="padding:0;flex-direction:column;gap:2px">
              <div class="fs-stat"><span class="fv" style="color:var(--grn)">${b2.mean??'—'}</span><span class="fl"> mean</span></div>
              <div style="font-size:7px;color:var(--txt3);font-family:var(--mono)">${b2.min??'—'} → ${b2.max??'—'}</div>
              <div style="font-size:7px;color:var(--acc2);font-family:var(--mono)">${b2.valid_pct??'—'}% valid</div>
            </div>
          </div>
        </div>
      </div>`;
  });
  sH('feat-stack-cards', stackCards);

  sH('feat-table', BN.map((nm,i) => {
    const b1=f1[nm]||{}, b2=f2[nm]||{};
    return `<tr>
      <td class="tm">B${i+1}</td><td>${nm}</td>
      <td class="tm">${b1.min??'—'}</td><td class="tm">${b1.max??'—'}</td>
      <td class="tp">${b1.mean??'—'}</td><td class="tm">${b1.valid_pct??'—'}%</td>
      <td class="tm">${b2.min??'—'}</td><td class="tm">${b2.max??'—'}</td>
      <td class="tp">${b2.mean??'—'}</td><td class="tm">${b2.valid_pct??'—'}%</td>
    </tr>`;
  }).join(''));
}

// ── Spectral ──
function bldSpec() {
  const n12=S.ndvi_2012, n16=S.ndvi_2016, b12=S.ndbi_2012, b16=S.ndbi_2016;
  sH('spec-stats',
    sc('var(--grn)', 'NDVI 2012', n12.mean||'—', 'mean') +
    sc('var(--grn)', 'NDVI 2016', n16.mean||'—', 'mean') +
    sc('var(--red)',  'NDBI 2012', b12.mean||'—', 'mean') +
    sc('var(--red)',  'NDBI 2016', b16.mean||'—', 'mean') +
    sc('var(--grn)', 'Hi-Veg 2012', (n12.high_veg_ha||'—')+' ha', 'NDVI>0.4') +
    sc('var(--grn)', 'Hi-Veg 2016', (n16.high_veg_ha||'—')+' ha', 'NDVI>0.4') +
    sc('var(--org)',  'Built 2012',  (b12.high_built_ha||'—')+' ha', 'NDBI>0.1') +
    sc('var(--org)',  'Built 2016',  (b16.high_built_ha||'—')+' ha', 'NDBI>0.1'));

  sH('ndvi-table', [
    ['High veg',  '> 0.4',   'high_veg_ha'],
    ['Med veg',   '0.2–0.4', 'med_veg_ha'],
    ['Low veg',   '0–0.2',   'low_veg_ha'],
    ['Non-veg',   '≤ 0',     'non_veg_ha'],
  ].map(([nm,rng,k]) =>
    `<tr><td>${nm}</td><td class="tm">${rng}</td><td>${n12[k]??'—'}</td><td>${n16[k]??'—'}</td><td>${ds(n12[k],n16[k])}</td></tr>`
  ).join(''));

  const nd = (v12, v16, pos_interp, neg_interp) => {
    if (v12==null || v16==null) return '—';
    const d = (+v16 - +v12).toFixed(3);
    return `<span class="${+d>=0?'tp':'tn'}">${+d>=0?'+':''}${d}</span>`;
  };
  sH('spec-table', [
    ['NDVI', n12.mean, n16.mean, 'Vegetation health', 'Vegetation loss'],
    ['NDBI', b12.mean, b16.mean, 'Built-up increase', 'Built-up decrease'],
  ].map(([nm,v12,v16,pi,ni]) => {
    const diff = v12!=null&&v16!=null ? (+v16 - +v12) : null;
    const interp = diff==null?'—': diff>=0 ? pi : ni;
    return `<tr>
      <td><strong>${nm}</strong></td>
      <td class="tm">${v12??'—'}</td>
      <td class="tm">${v16??'—'}</td>
      <td>${diff!=null?`<span class="${diff>=0?'tp':'tn'}">${diff>=0?'+':''}${diff.toFixed(3)}</span>`:'—'}</td>
      <td class="tm">${interp}</td>
    </tr>`;
  }).join(''));
}

// ── LULC ──
function bldLULC() {
  sT('lulc-total-note', 'Study area: ' + (S.lulc_2012.total_ha||'—') + ' ha');
  const lulcs = [S.lulc_2012, S.lulc_2016, S.lulc_2020, S.lulc_2024];
  sH('lulc-full-table', CL.map((nm, i) => {
    const [l12,l16,l20,l24] = lulcs.map(l => l && l.classes ? l.classes[i] : null);
    const tr = l12 && l16
      ? (l16.ha >= l12.ha
          ? `<span class="tp">↑ +${Math.round((l16.ha-l12.ha)*10)/10}</span>`
          : `<span class="tn">↓ ${Math.round((l16.ha-l12.ha)*10)/10}</span>`)
      : '—';
    return `<tr>
      <td><span class="chip"><span class="ld" style="background:${CC[i]}"></span>${nm}</span></td>
      <td class="tm">${l12?l12.pixels.toLocaleString():'—'}</td>
      <td>${l12?l12.ha:'—'}</td><td>${l16?l16.ha:'—'}</td>
      <td class="tm">${l20?l20.ha+'*':'—'}</td><td class="tm">${l24?l24.ha+'*':'—'}</td>
      <td>${tr}</td>
    </tr>`;
  }).join(''));
}

// ── Change ──
function bldChange() {
  const b0=S.bin_obs, g0=S.gl_obs;
  sH('change-stats',
    sc('var(--red)', 'Changed 12→16', (b0.changed_ha||'—')+' ha', (b0.changed_pct||'—')+'%') +
    sc('var(--grn)', 'Gain 12→16',    (g0.gain_ha||'—')+' ha', '') +
    sc('var(--red)', 'Loss 12→16',    (g0.loss_ha||'—')+' ha', '') +
    sc('var(--grn)', 'Net 12→16',     (g0.net_ha!=null?(g0.net_ha>0?'+':'')+g0.net_ha:'—')+' ha', '') +
    sc('var(--acc)', 'Changed 16→20', (S.bin_1620.changed_ha||'—')+' ha', '') +
    sc('var(--acc)', 'Changed 20→24', (S.bin_2024.changed_ha||'—')+' ha', ''));

  const pairs = [
    ['2012→2016','OBSERVED', S.bin_obs, S.gl_obs],
    ['2016→2020','PREDICTED',S.bin_1620,S.gl_1620],
    ['2020→2024','PREDICTED',S.bin_2024,S.gl_2024],
  ];
  sH('change-table', pairs.map(([per,type,bn,gl]) =>
    `<tr>
      <td>${per}</td>
      <td class="${type==='OBSERVED'?'tp':'to'}">${type}</td>
      <td class="tn">${bn.changed_ha||'—'}</td>
      <td class="tn">${bn.changed_pct||'—'}%</td>
      <td class="tp">${gl.gain_ha||'—'}</td>
      <td class="tn">${gl.loss_ha||'—'}</td>
      <td class="${gl.net_ha>=0?'tp':'tn'}">${gl.net_ha!=null?(gl.net_ha>0?'+':'')+gl.net_ha:'—'}</td>
    </tr>`
  ).join(''));

  const dir = S.dir_obs;
  const dcols = {'-4':'#67000d','-3':'#a50f15','-2':'#d73027','-1':'#fc8d59','0':'#555','1':'#74c476','2':'#41ab5d','3':'#238b45','4':'#005a32'};
  let dh = '';
  for (let v=-4; v<=4; v++) {
    const k = String(v), bin = dir && dir.bins ? dir.bins[k] : {ha:0,pct:0};
    const w = Math.max(bin.pct||0, .3);
    dh += `<div class="br">
      <div class="bl">${v>0?'+':''}${v} class diff</div>
      <div class="bt"><div class="bf" style="width:${w}%;background:${dcols[k]};min-width:3px">
        <span>${bin.ha||0} ha · ${bin.pct||0}%</span>
      </div></div>
    </div>`;
  }
  sH('dir-bars', dh);
  bldMatrix('tm-table');
}

// ── Damage ──
function bldDmg() {
  const obs = S.sev_obs;
  sH('damage-stats', SN.map((nm,i) => {
    const c = obs && obs.classes ? obs.classes[i] : {ha:'—',pct:'—'};
    return sc(SC2[i], nm, (c.ha||'—')+' ha', (c.pct||'—')+'%');
  }).join(''));
  sH('damage-table', SN.map((nm,si) => {
    const vals = [S.sev_obs,S.sev_1620,S.sev_2024].map(s => s&&s.classes ? s.classes[si] : {ha:'—',pct:'—'});
    return `<tr>
      <td><span class="chip"><span class="ld" style="background:${SC2[si]}"></span>${nm}</span></td>
      ${vals.map(v=>`<td>${v.ha}</td><td class="tm">${v.pct}%</td>`).join('')}
    </tr>`;
  }).join(''));
  const mxS = obs && obs.classes ? Math.max(...obs.classes.map(c=>c.ha)) : 1;
  sH('sev-bars', obs && obs.classes
    ? obs.classes.map(c =>
        `<div class="br"><div class="bl">${c.name}</div><div class="bt">
          <div class="bf" style="width:${Math.max(c.ha/mxS*100,.4).toFixed(1)}%;background:${c.color}">
            <span>${c.ha} ha · ${c.pct}%</span>
          </div></div></div>`
      ).join('')
    : '');
}

// ── Prediction ──
function bldPred() {
  const lulcs = [S.lulc_2012,S.lulc_2016,S.lulc_2020,S.lulc_2024];
  sH('pred-table', CL.map((nm,i) => {
    const d = lulcs.map(l => l && l.classes ? l.classes[i] : null);
    return `<tr>
      <td><span class="chip"><span class="ld" style="background:${CC[i]}"></span>${nm}</span></td>
      ${d.map((c,yi) => c
        ? `<td ${yi>=2?'class="tm"':''}>${c.ha}</td><td class="tm">${c.pct}%</td>`
        : '<td>—</td><td>—</td>'
      ).join('')}
    </tr>`;
  }).join(''));
}

// ── Accuracy + Confusion combined ──
function bldAcc() {
  const tm = S.transition_matrix;
  let td=0, ta=0, rt=[], ct=[], dg=[];
  if (tm && tm.length) {
    for (let i=0; i<5; i++) {
      rt.push(tm[i].reduce((a,b)=>a+b,0));
      ct.push(tm.map(r=>r[i]).reduce((a,b)=>a+b,0));
      dg.push(tm[i][i]);
      td += tm[i][i]; ta += rt[i];
    }
  }
  const oa    = ta > 0 ? (td/ta*100).toFixed(2) : '—';
  let pe = 0;
  if (ta > 0) rt.forEach((_,i) => { pe += rt[i]*ct[i]; });
  pe = ta > 0 ? pe/(ta*ta) : 0;
  const kappa = ta > 0 ? ((td/ta-pe)/(1-pe)).toFixed(4) : '—';

  sH('acc-stats',
    sc('var(--grn)', 'Overall Accuracy', oa+'%', 'OA') +
    sc('var(--acc)', 'Kappa Coefficient', kappa, 'κ') +
    sc('var(--org)', 'Total Pixels', ta.toLocaleString(), 'valid') +
    sc('var(--acc)', 'Classes', '5', 'KMeans'));

  bldMatrix('cm-table');

  sH('acc-table', CL.map((nm,i) => {
    const pa = rt[i]>0 ? (dg[i]/rt[i]*100).toFixed(1) : '—';
    const ua = ct[i]>0 ? (dg[i]/ct[i]*100).toFixed(1) : '—';
    const f1 = (pa!=='—' && ua!=='—' && (+pa+ +ua)>0)
      ? (2*(+pa/100)*(+ua/100)/((+pa/100)+(+ua/100))).toFixed(4) : '—';
    return `<tr>
      <td><span class="chip"><span class="ld" style="background:${CC[i]}"></span>${nm}</span></td>
      <td class="tm">${rt[i]?.toLocaleString()||'—'}</td>
      <td class="tm">${dg[i]?.toLocaleString()||'—'}</td>
      <td class="${+pa>75?'tp':+pa>60?'to':'tn'}">${pa}%</td>
      <td class="${+ua>75?'tp':+ua>60?'to':'tn'}">${ua}%</td>
      <td class="${f1!=='—'&&+f1>0.75?'tp':f1!=='—'&&+f1>0.6?'to':'tn'}">${f1}</td>
    </tr>`;
  }).join(''));
}

window.addEventListener('load', () => {
  bldOverview(); bldLidar(); bldLAZ(); bldFusion();
  bldSpec(); bldLULC(); bldChange(); bldDmg(); bldPred(); bldAcc();
});
""".replace("__STATS__", STATS_JSON)

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kaikōura Earthquake — LULC Analysis Dashboard</title>
<style>{CSS}</style>
</head>
<body>

<!-- ══ HEADER ══ -->
<header>
  <div class="hdr-top">
    <div class="hdr-title">
      <div class="hdr-eyebrow">Remote Sensing Analysis · New Zealand</div>
      <h1>Kaikōura Earthquake — <em>Land Use &amp; Cover</em> <strong>Change Analysis</strong></h1>
      <p class="hdr-meta">LiDAR + Sentinel-2 · EPSG:32760 · 2012–2024 · KMeans k=5 · CA-Markov Prediction · Offline</p>
    </div>
    <div class="hdr-right">
      <div class="pills">
        <span class="pill pb">EPSG:32760</span>
        <span class="pill pr">M7.8 · 14 NOV 2016</span>
        <span class="pill pg" id="pill-ha">LOADING…</span>
        <span class="pill po" id="pill-chg">LOADING…</span>
        <span class="pill pb">OFFLINE</span>
      </div>
      <div class="hdr-kpis">
        <div class="kpi"><div class="kv" id="kpi-ha">—<span>ha</span></div><div class="kl">Study Area</div></div>
        <div class="kpi"><div class="kv" id="kpi-chg" style="color:var(--red)">—</div><div class="kl">Changed</div></div>
        <div class="kpi"><div class="kv" id="kpi-gain" style="color:var(--grn)">—<span>ha</span></div><div class="kl">Veg Gain</div></div>
        <div class="kpi"><div class="kv" id="kpi-loss" style="color:var(--red)">—<span>ha</span></div><div class="kl">Veg Loss</div></div>
      </div>
    </div>
  </div>
  <div class="tabs">
    <button class="tab active"  onclick="sw('overview',this)">Overview</button>
    <button class="tab"         onclick="sw('lidar',this)">LiDAR</button>
    <button class="tab"         onclick="sw('laz',this)">Point Cloud</button>
    <button class="tab"         onclick="sw('spectral',this)">NDVI / NDBI</button>
    <button class="tab"         onclick="sw('fusion',this)">Feature Fusion</button>
    <button class="tab"         onclick="sw('lulc',this)">LULC Maps</button>
    <button class="tab"         onclick="sw('change',this)">Change Detection</button>
    <button class="tab"         onclick="sw('damage',this)">Damage</button>
    <button class="tab"         onclick="sw('prediction',this)">Prediction</button>
    <button class="tab"         onclick="sw('accuracy',this)">Accuracy &amp; Confusion</button>
  </div>
</header>

<main>

<!-- ══ OVERVIEW ══ -->
<div class="panel active" id="p-overview">
  <div class="sh">
    <div class="sh-icon">🗺</div>
    <div class="sh-text"><h2>Study Summary</h2><small>Real values from TIF data · All periods</small></div>
    <div class="ln"></div>
  </div>
  <div class="sc-row" id="ov-stats"></div>
  <div class="ig ig2">
    {IC("LULC 2012 — Pre-earthquake","OBSERVED","bobs","lulc_2012.png",LLEG,"lulc_2012")}
    {IC("LULC 2016 — Post-earthquake","OBSERVED","bobs","lulc_2016.png",LLEG,"lulc_2016")}
  </div>
  <div class="bc" id="lulc-bar-chart">
    <h3>LULC Area — All Periods (ha)</h3>
    <div id="lulc-bars"></div>
  </div>
  <div class="tw">
    <div class="tw-hdr"><h3>Key Pixel Transitions 2012→2016</h3><small id="trans-total"></small></div>
    <table>
      <thead><tr><th>Transition</th><th>Pixels</th><th>Area (ha)</th><th>%</th><th>Interpretation</th></tr></thead>
      <tbody id="trans-table"></tbody>
    </table>
  </div>
</div>

<!-- ══ LIDAR ══ -->
<div class="panel" id="p-lidar">
  <div class="sh">
    <div class="sh-icon">📡</div>
    <div class="sh-text"><h2>LiDAR Data</h2><small>DTM · DSM · Slope · CHM · 1m resolution</small></div>
    <div class="ln"></div>
  </div>
  <div class="sc-row" id="lidar-stats"></div>
  <div class="ig ig3">
    {IC("DTM 2012","LiDAR","braw","t1_dtm.png","","dtm_2012")}
    {IC("DTM 2016","LiDAR","braw","t2_dtm.png","","dtm_2016")}
    {IC("Elevation Change 2012→2016","CHANGE","bchg","elevation_change_t1t2.png",LD([("#d73027","Lowered"),("#1a9641","Raised")]),"elev_change")}
    {IC("DSM 2012","LiDAR","braw","t1_dsm.png")}
    {IC("DSM 2016","LiDAR","braw","t2_dsm.png")}
    {IC("Slope 2012","LiDAR","braw","t1_slope.png",LD([("#ffffb2","Flat"),("#d7301f","Steep")]),"slope_2012")}
    {IC("Slope 2016","LiDAR","braw","t2_slope.png",LD([("#ffffb2","Flat"),("#d7301f","Steep")]),"slope_2016")}
    {IC("CHM 2012 — Canopy Height","LiDAR","braw","t1_chm.png",LD([("#30123b","0m"),("#1ac7c2","5m"),("#efb419","10m"),("#7a0403","≥20m")]),"chm_2012")}
    {IC("CHM 2016 — Canopy Height","LiDAR","braw","t2_chm.png",LD([("#30123b","0m"),("#1ac7c2","5m"),("#efb419","10m"),("#7a0403","≥20m")]),"chm_2016")}
  </div>
  <div class="tw">
    <div class="tw-hdr"><h3>LiDAR Statistics Summary</h3></div>
    <table>
      <thead><tr><th>Layer</th><th>Min (m)</th><th>Max (m)</th><th>Mean (m)</th><th>Std Dev</th><th>Valid Area (ha)</th></tr></thead>
      <tbody id="lidar-table"></tbody>
    </table>
  </div>
  <div class="tw">
    <div class="tw-hdr"><h3>Slope Classification by Area</h3></div>
    <table>
      <thead><tr><th>Category</th><th>Range</th><th>2012 (ha)</th><th>2016 (ha)</th><th>Change (ha)</th></tr></thead>
      <tbody id="slope-table"></tbody>
    </table>
  </div>
</div>

<!-- ══ LAZ POINT CLOUD ══ -->
<div class="panel" id="p-laz">
  <div class="sh">
    <div class="sh-icon">☁</div>
    <div class="sh-text"><h2>LiDAR Point Cloud</h2><small>Plan view · 3D perspective · Cross-section profiles</small></div>
    <div class="ln"></div>
  </div>
  <div class="sc-row" id="laz-stats"></div>
  <div class="ig ig1">
    {IC("Point Cloud Comparison — 2012 vs 2016","LiDAR","blaz","laz_compare_2012_2016.png",LD([("#006400","Dense canopy"),("#90ee90","Low canopy"),("#d4b483","Ground")]))}
  </div>
  <div class="ig ig2">
    {IC("Plan View 2012","LiDAR","blaz","laz_planview_t1_2012.png")}
    {IC("Plan View 2016","LiDAR","blaz","laz_planview_t2_2016.png")}
    {IC("3D Perspective 2012","LiDAR","blaz","laz_3d_t1_2012.png")}
    {IC("3D Perspective 2016","LiDAR","blaz","laz_3d_t2_2016.png")}
    {IC("Cross-Section Profile 2012","LiDAR","blaz","laz_profile_t1_2012.png")}
    {IC("Cross-Section Profile 2016","LiDAR","blaz","laz_profile_t2_2016.png")}
  </div>
  <div class="tw">
    <div class="tw-hdr"><h3>CHM — Canopy Height Statistics</h3></div>
    <table>
      <thead><tr><th>Layer</th><th>Max Height (m)</th><th>Mean Height (m)</th><th>Canopy Area &gt;2m (ha)</th><th>No-Canopy (ha)</th></tr></thead>
      <tbody id="chm-table"></tbody>
    </table>
  </div>
</div>

<!-- ══ SPECTRAL ══ -->
<div class="panel" id="p-spectral">
  <div class="sh">
    <div class="sh-icon">🌿</div>
    <div class="sh-text"><h2>NDVI / NDBI — Spectral Indices</h2><small>Sentinel-2 · QGIS-rendered · RdYlGn / RdBu · Fixed scale</small></div>
    <div class="ln"></div>
  </div>
  <div class="sc-row" id="spec-stats"></div>

  <div class="ssh">NDVI — Normalised Difference Vegetation Index</div>
  <div class="ig ig2">
    {IC("NDVI 2012 — Pre-earthquake","SENTINEL-2","braw","ndvi_2012.png",LD([("#d73027","Low veg / bare"),("#ffffbf","Mixed"),("#1a9641","High veg")]))}
    {IC("NDVI 2016 — Post-earthquake","SENTINEL-2","braw","ndvi_2016.png",LD([("#d73027","Low veg / bare"),("#ffffbf","Mixed"),("#1a9641","High veg")]))}
  </div>

  <div class="ssh">NDBI — Normalised Difference Built-up Index</div>
  <div class="ig ig2">
    {IC("NDBI 2012 — Pre-earthquake","SENTINEL-2","braw","ndbi_2012.png",LD([("#2166ac","Low built-up"),("#f7f7f7","Neutral"),("#d73027","High built-up")]))}
    {IC("NDBI 2016 — Post-earthquake","SENTINEL-2","braw","ndbi_2016.png",LD([("#2166ac","Low built-up"),("#f7f7f7","Neutral"),("#d73027","High built-up")]))}
  </div>

  <div class="tw">
    <div class="tw-hdr"><h3>NDVI Vegetation Cover Statistics</h3></div>
    <table>
      <thead><tr><th>Category</th><th>NDVI Range</th><th>2012 (ha)</th><th>2016 (ha)</th><th>Change (ha)</th></tr></thead>
      <tbody id="ndvi-table"></tbody>
    </table>
  </div>
  <div class="tw">
    <div class="tw-hdr"><h3>NDVI / NDBI Mean Values</h3></div>
    <table>
      <thead><tr><th>Index</th><th>2012 Mean</th><th>2016 Mean</th><th>Change</th><th>Interpretation</th></tr></thead>
      <tbody id="spec-table"></tbody>
    </table>
  </div>
</div>

<!-- ══ FEATURE FUSION ══ -->
<div class="panel" id="p-fusion">
  <div class="sh">
    <div class="sh-icon">🔬</div>
    <div class="sh-text"><h2>Feature Fusion</h2><small>5-band stack: DTM · Slope · CHM (turbo) · NDVI · NDBI — 2012 vs 2016</small></div>
    <div class="ln"></div>
  </div>

  <!-- ① Per-band stats cards -->
  <div class="ssh">Band Statistics — T1 (2012) vs T2 (2016)</div>
  <div class="sc-row" id="feat-stats"></div>
  <div id="feat-stack-cards" class="fs-grid"></div>

  <!-- ② False-colour RGB composites -->
  <div class="ssh">False-Colour RGB Composites (R=NDVI · G=DTM · B=Slope)</div>
  <div class="ig ig2">
    {IC("Feature Stack RGB — 2012","5-BAND","bfeat","t1_feature_stack_rgb.png")}
    {IC("Feature Stack RGB — 2016","5-BAND","bfeat","t2_feature_stack_rgb.png")}
  </div>

  <!-- ③ Band-by-band comparison — 2012 vs 2016 -->
  <div class="ssh">Band-by-Band Comparison — 2012 vs 2016 (shared colour scale)</div>
  <div class="ig ig3">
    {IC("B1 · DTM — 2012 vs 2016","COMPARE","bfeat","fusion_compare_band1_dtm.png",LD([("#f7fbff","Low elev"),("#2171b5","High elev")]))}
    {IC("B2 · Slope — 2012 vs 2016","COMPARE","bfeat","fusion_compare_band2_slope.png",LD([("#ffffb2","Flat"),("#d7301f","Steep")]))}
    {IC("B3 · CHM — 2012 vs 2016","COMPARE","bfeat","fusion_compare_band3_chm.png",LD([("#30123b","0m"),("#1ac7c2","5m"),("#efb419","10m"),("#7a0403","≥20m")]))}
    {IC("B4 · NDVI — 2012 vs 2016","COMPARE","bfeat","fusion_compare_band4_ndvi.png",LD([("#d73027","Low veg"),("#ffffbf","Mixed"),("#1a9641","High veg")]))}
    {IC("B5 · NDBI — 2012 vs 2016","COMPARE","bfeat","fusion_compare_band5_ndbi.png",LD([("#2166ac","Low built"),("#f7f7f7","Neutral"),("#d73027","High built")]))}
  </div>

  <!-- ④ Statistics table -->
  <div class="tw">
    <div class="tw-hdr"><h3>Feature Stack Band Statistics — T1 (2012) vs T2 (2016)</h3></div>
    <table>
      <thead><tr>
        <th>Band</th><th>Layer</th>
        <th>T1 Min</th><th>T1 Max</th><th>T1 Mean</th><th>T1 Valid%</th>
        <th>T2 Min</th><th>T2 Max</th><th>T2 Mean</th><th>T2 Valid%</th>
      </tr></thead>
      <tbody id="feat-table"></tbody>
    </table>
  </div>
</div>

<!-- ══ LULC ══ -->
<div class="panel" id="p-lulc">
  <div class="sh">
    <div class="sh-icon">🗂</div>
    <div class="sh-text"><h2>LULC Classification Maps</h2><small>KMeans k=5 · NDVI-ranked classes</small></div>
    <div class="ln"></div>
  </div>
  <div class="ig ig2">
    {IC("LULC 2012","OBSERVED","bobs","lulc_2012.png",LLEG,"lulc_2012")}
    {IC("LULC 2016","OBSERVED","bobs","lulc_2016.png",LLEG,"lulc_2016")}
  </div>
  <div class="tw">
    <div class="tw-hdr"><h3>LULC Area Statistics — All Periods</h3><small id="lulc-total-note"></small></div>
    <table>
      <thead><tr><th>Class</th><th>2012 px</th><th>2012 ha</th><th>2016 ha</th><th>2020 ha</th><th>2024 ha</th><th>Trend 12→16</th></tr></thead>
      <tbody id="lulc-full-table"></tbody>
    </table>
  </div>
</div>

<!-- ══ CHANGE DETECTION ══ -->
<div class="panel" id="p-change">
  <div class="sh">
    <div class="sh-icon">🔄</div>
    <div class="sh-text"><h2>Change Detection</h2><small>Binary · Gain/Loss · Directional · All periods</small></div>
    <div class="ln"></div>
  </div>
  <div class="sc-row" id="change-stats"></div>
  <div class="ig ig3">
    {IC("Binary Change 2012→2016","OBSERVED","bchg","binary_change.png",BLEG,"bin_obs")}
    {IC("Gain/Loss 2012→2016","OBSERVED","bchg","gain_loss.png",GLEG,"gl_obs")}
    {IC("Directional Change 2012→2016","OBSERVED","bchg","directional_change.png",DLEG,"dir_obs")}
    {IC("Binary Change 2016→2020","PREDICTED","bpred","binary_change_2016_2020.png",BLEG,"bin_1620")}
    {IC("Gain/Loss 2016→2020","PREDICTED","bpred","gain_loss_2016_2020.png",GLEG,"gl_1620")}
    {IC("Directional Change 2016→2020","PREDICTED","bpred","directional_change_2016_2020.png",DLEG,"dir_1620")}
    {IC("Binary Change 2020→2024","PREDICTED","bpred","binary_change_2020_2024.png",BLEG,"bin_2024")}
    {IC("Gain/Loss 2020→2024","PREDICTED","bpred","gain_loss_2020_2024.png",GLEG,"gl_2024")}
    {IC("Directional Change 2020→2024","PREDICTED","bpred","directional_change_2020_2024.png",DLEG,"dir_2024")}
  </div>
  <div class="tw">
    <div class="tw-hdr"><h3>Change Statistics — All Periods</h3></div>
    <table>
      <thead><tr><th>Period</th><th>Type</th><th>Changed (ha)</th><th>Changed %</th><th>Veg Gain</th><th>Veg Loss</th><th>Net (ha)</th></tr></thead>
      <tbody id="change-table"></tbody>
    </table>
  </div>
  <div class="bc"><h3>Directional Change Distribution — 2012→2016</h3><div id="dir-bars"></div></div>
  <div class="mw">
    <div class="mw-hdr"><h3>Pixel Transition Matrix — 2012 → 2016</h3></div>
    <table class="mt" id="tm-table"></table>
  </div>
</div>

<!-- ══ DAMAGE ══ -->
<div class="panel" id="p-damage">
  <div class="sh">
    <div class="sh-icon">⚠</div>
    <div class="sh-text"><h2>Damage Assessment</h2><small>Severity classification — all 3 periods</small></div>
    <div class="ln"></div>
  </div>
  <div class="sc-row" id="damage-stats"></div>
  <div class="ig ig3">
    {IC("Damage Severity 2012→2016","OBSERVED","bobs","damage_severity.png",SLEG,"sev_obs")}
    {IC("Damage Severity 2016→2020","PREDICTED","bpred","damage_severity_2016_2020.png",SLEG,"sev_1620")}
    {IC("Damage Severity 2020→2024","PREDICTED","bpred","damage_severity_2020_2024.png",SLEG,"sev_2024")}
  </div>
  <div class="tw">
    <div class="tw-hdr"><h3>Damage Severity — All Periods</h3></div>
    <table>
      <thead><tr><th>Category</th><th>2012→16 ha</th><th>2012→16 %</th><th>2016→20 ha</th><th>2016→20 %</th><th>2020→24 ha</th><th>2020→24 %</th></tr></thead>
      <tbody id="damage-table"></tbody>
    </table>
  </div>
  <div class="bc"><h3>Damage Severity Breakdown — 2012→2016</h3><div id="sev-bars"></div></div>
</div>

<!-- ══ PREDICTION ══ -->
<div class="panel" id="p-prediction">
  <div class="sh">
    <div class="sh-icon">🔮</div>
    <div class="sh-text"><h2>CA-Markov Prediction 2020 &amp; 2024</h2><small>Cellular Automata + Markov chain modelling</small></div>
    <div class="ln"></div>
  </div>
  <div class="ig ig4">
    {IC("LULC 2012","OBSERVED","bobs","lulc_2012.png",LLEG,"lulc_2012")}
    {IC("LULC 2016","OBSERVED","bobs","lulc_2016.png",LLEG,"lulc_2016")}
    {IC("LULC 2020","PREDICTED","bpred","lulc_2020.png",LLEG,"lulc_2020")}
    {IC("LULC 2024","PREDICTED","bpred","lulc_2024.png",LLEG,"lulc_2024")}
  </div>
  <div class="ig ig2">
    {IC("Predicted Change 2016→2020","PREDICTED","bpred","predicted_change_2016_2020.png",GLEG)}
    {IC("Predicted Change 2016→2024","PREDICTED","bpred","predicted_change_2016_2024.png",GLEG)}
  </div>
  <div class="tw">
    <div class="tw-hdr"><h3>LULC Area Trends — All Periods</h3></div>
    <table>
      <thead><tr><th>Class</th><th>2012 ha</th><th>2012 %</th><th>2016 ha</th><th>2016 %</th><th>2020 ha</th><th>2020 %</th><th>2024 ha</th><th>2024 %</th></tr></thead>
      <tbody id="pred-table"></tbody>
    </table>
  </div>
</div>

<!-- ══ ACCURACY + CONFUSION (MERGED) ══ -->
<div class="panel" id="p-accuracy">
  <div class="sh">
    <div class="sh-icon">✓</div>
    <div class="sh-text"><h2>Accuracy Assessment &amp; Confusion Matrices</h2><small>Overall accuracy · Kappa · Per-class metrics · All periods</small></div>
    <div class="ln"></div>
  </div>

  <!-- KPI cards -->
  <div class="sc-row" id="acc-stats"></div>

  <!-- Confusion matrix images — all 3 periods -->
  <div class="ssh">Confusion Matrix Images — All Periods</div>
  <div class="confusion-imgs">
    {IC("Confusion Matrix — 2012 → 2016 (Observed)","OBSERVED","bobs","confusion_2012_2016.png")}
    {IC("Confusion Matrix — 2016 → 2020 (Predicted)","PREDICTED","bpred","confusion_2016_2020.png")}
    {IC("Confusion Matrix — 2020 → 2024 (Predicted)","PREDICTED","bpred","confusion_2020_2024.png")}
  </div>

  <!-- Interactive matrix — single copy -->
  <div class="ssh">Pixel Transition Matrix — 2012 → 2016</div>
  <div class="mw">
    <div class="mw-hdr"><h3>Transition Matrix · Raw Pixel Counts + Row %</h3></div>
    <table class="mt" id="cm-table"></table>
  </div>

  <!-- Per-class accuracy table -->
  <div class="ssh">Per-Class Accuracy Metrics</div>
  <div class="tw">
    <div class="tw-hdr"><h3>Producer / User Accuracy &amp; F1 Scores</h3></div>
    <table>
      <thead><tr>
        <th>Class</th>
        <th>Row Total (px)</th>
        <th>Diagonal (px)</th>
        <th>Producer Acc %</th>
        <th>User Acc %</th>
        <th>F1 Score</th>
      </tr></thead>
      <tbody id="acc-table"></tbody>
    </table>
  </div>
</div>

</main>

<!-- ══ MODAL ══ -->
<div class="mo" id="modal" onclick="cmo(event)">
  <div class="md">
    <button class="mc" onclick="cmd()">✕</button>
    <h3 id="m-title" style="font-family:var(--mono);font-size:10px;font-weight:600;color:var(--acc);padding-right:32px;text-transform:uppercase;letter-spacing:.8px"></h3>
    <div class="sr" id="m-stats"></div>
    <img id="m-img" src="" alt="">
  </div>
</div>

<footer>
  <span>KAIKŌURA EARTHQUAKE LULC ANALYSIS</span> · LiDAR + SENTINEL-2 · EPSG:32760 · 2012–2024 ·
  KMeans k=5 · CA-Markov · Real TIF Values · <span>OFFLINE SINGLE-FILE</span>
</footer>

<script>{JS}</script>
</body>
</html>"""

with open(HTML_OUT, "w", encoding="utf-8") as f:
    f.write(HTML)
size_mb = os.path.getsize(HTML_OUT) / 1024 / 1024
print(f"\n  Written : {HTML_OUT}")
print(f"  Size    : {size_mb:.1f} MB  (self-contained, all images embedded)")
print("\n" + "="*56)
print("  Share the ONE file: kaikoura_dashboard_OFFLINE.html")
print("  Works offline — no server, no internet needed")
print("="*56 + "\n")