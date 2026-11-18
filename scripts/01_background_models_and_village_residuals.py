"""Gunung Salak single-scene analysis: clean DEM, background-LST models, spatial block CV,
block bootstrap CIs per village, rank stability, Moran's I, illumination terms."""
import json, sys
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.features import geometry_mask
from rasterio.transform import Affine
import geopandas as gpd
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

import os
UP = os.environ.get("GS_INPUT_DIR", "data/raw") + "/"
OUT = os.environ.get("GS_OUT_DIR", "outputs") + "/"
os.makedirs(OUT, exist_ok=True)
SC = "LC08_L2SP_122065_20250826_20250903_02_T1_"
rng = np.random.default_rng(42)

# ---------------- AOI and clip window (rounded window => no sub-pixel misregistration) -------------
gdf = gpd.read_file(UP + "Wilayah_Geothermal-Gunung_Salak.shp")
minx, miny, maxx, maxy = gdf.total_bounds
buf = 0.01
ub = transform_bounds("EPSG:4326", "EPSG:32648", minx - buf, miny - buf, maxx + buf, maxy + buf)

def clip(path):
    with rasterio.open(path) as s:
        w = from_bounds(*ub, transform=s.transform).round_offsets().round_lengths()
        return s.read(1, window=w), s.window_transform(w), s.crs

b4, tr, crs = clip(UP + SC + "SR_B4.TIF")
b5, _, _ = clip(UP + SC + "SR_B5.TIF")
st, _, _ = clip(UP + SC + "ST_B10.TIF")
qa, _, _ = clip(UP + SC + "QA_PIXEL.TIF")
shape = b4.shape
print("clip shape", shape, "res", tr.a)

bit = lambda v, n: ((v >> n) & 1).astype(bool)
cloudy = bit(qa, 0) | bit(qa, 1) | bit(qa, 2) | bit(qa, 3) | bit(qa, 4)

red = b4.astype(np.float32) * 0.0000275 - 0.2
nir = b5.astype(np.float32) * 0.0000275 - 0.2
ndvi = np.where((b4 > 0) & (b5 > 0), (nir - red) / (nir + red + 1e-9), np.nan).astype(np.float32)
lst = np.where(st > 0, st.astype(np.float32) * 0.00341802 + 149.0 - 273.15, np.nan).astype(np.float32)

# ---------------- AOI mask on Landsat grid ----------------
g_utm = gdf.to_crs(crs)
aoi = geometry_mask(list(g_utm.geometry), out_shape=shape, transform=tr, invert=True)
vid = np.full(shape, -1, dtype=np.int16)
for i, geom in enumerate(g_utm.geometry):
    m = geometry_mask([geom], out_shape=shape, transform=tr, invert=True)
    vid[m & (vid < 0)] = i
print("AOI pixels", aoi.sum(), " cloudy inside AOI", (cloudy & aoi).sum(),
      f"({(cloudy & aoi).sum() / aoi.sum() * 100:.2f}%)")

# ---------------- Clean DEM: zeros are nodata (outside polygons) -> NaN, then area-average to 30 m -------
with rasterio.open(UP + "DEMNAS_Gunung_Salak.tif") as s:
    d = s.read(1).astype(np.float32)
    d[d == 0] = np.nan
    dem = np.full(shape, np.nan, dtype=np.float32)
    reproject(d, dem, src_transform=s.transform, src_crs=s.crs, dst_transform=tr, dst_crs=crs,
              resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)

# slope / aspect (30 m grid; rows run north->south)
dzdx = np.gradient(dem, tr.a, axis=1)
dzdy_n = -np.gradient(dem, abs(tr.e), axis=0)
slope = np.arctan(np.hypot(dzdx, dzdy_n))
aspect = np.mod(np.arctan2(-dzdx, -dzdy_n), 2 * np.pi)  # downslope direction, clockwise from north

# ---------------- Sun position at overpass (approximate; scene MTL was not supplied) --------------------
def sun_pos(lat, lon, doy, hour_utc):
    g = 2 * np.pi / 365 * (doy - 1 + (hour_utc - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
           + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = hour_utc * 60 + eqt + 4 * lon
    ha = np.radians(tst / 4 - 180)
    la = np.radians(lat)
    cz = np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)
    zen = np.arccos(cz)
    az = np.arccos(np.clip((np.sin(dec) - cz * np.sin(la)) / (np.sin(zen) * np.cos(la)), -1, 1))
    if ha > 0:
        az = 2 * np.pi - az
    return zen, az

lat_c, lon_c = (miny + maxy) / 2, (minx + maxx) / 2
HOUR = 2 + 53 / 60  # ~10:00 local solar time for Landsat 8 at ~106.7E
zen, saz = sun_pos(lat_c, lon_c, 238, HOUR)
print(f"sun elevation {90 - np.degrees(zen):.1f} deg, azimuth {np.degrees(saz):.1f} deg (approx.)")
cos_inc = np.cos(zen) * np.cos(slope) + np.sin(zen) * np.sin(slope) * np.cos(saz - aspect)

# ---------------- Valid mask ----------------
valid = aoi & ~cloudy & np.isfinite(lst) & np.isfinite(ndvi) & np.isfinite(dem) & np.isfinite(slope)
n_valid = int(valid.sum())
print("valid pixels", n_valid, f"({n_valid / aoi.sum() * 100:.1f}% of AOI)")

E = dem[valid].astype(np.float64); L = lst[valid].astype(np.float64)
N = ndvi[valid].astype(np.float64); S = np.degrees(slope[valid]); CI = cos_inc[valid]
AS = aspect[valid]; V = vid[valid]
rr, cc = np.where(valid)
block_px = 100  # 3 km
BID = (rr // block_px) * 1000 + (cc // block_px)
print("elevation inside AOI (valid): min %.0f max %.0f mean %.0f" % (E.min(), E.max(), E.mean()))
print("LST (deg C): min %.2f max %.2f mean %.2f sd %.2f" % (L.min(), L.max(), L.mean(), L.std()))
print("NDVI mean %.3f" % N.mean())
print("number of 3-km blocks", len(np.unique(BID)))

stats = dict(n_valid=n_valid, aoi_pixels=int(aoi.sum()), cloud_pct_aoi=float((cloudy & aoi).sum() / aoi.sum() * 100),
             elev_min=float(E.min()), elev_max=float(E.max()), elev_mean=float(E.mean()),
             lst_min=float(L.min()), lst_max=float(L.max()), lst_mean=float(L.mean()), lst_sd=float(L.std()),
             ndvi_mean=float(N.mean()), sun_elev=float(90 - np.degrees(zen)), sun_az=float(np.degrees(saz)),
             n_blocks=int(len(np.unique(BID))))

# ---------------- Design matrices ----------------
def feats(name):
    if name == "M0": return np.c_[E]
    if name == "M1": return np.c_[E, E ** 2]
    if name == "M2": return np.c_[E, S, CI]
    if name == "M3": return np.c_[E, S, CI, N]
    if name == "M4": return np.c_[E, S, CI, N, np.sin(AS), np.cos(AS)]
labels = {"M0": "Elevation (linear)", "M1": "Elevation (quadratic)", "M2": "Elevation + slope + illumination",
          "M3": "M2 + NDVI", "M4": "RF: elevation, slope, illumination, NDVI, aspect"}

gkf = GroupKFold(n_splits=5)
splits_sp = list(gkf.split(E, L, BID))
splits_rd = list(KFold(5, shuffle=True, random_state=1).split(E))

def fit_predict(name, tr_idx, te_idx):
    X = feats(name)
    if name == "M4":
        sub = rng.choice(tr_idx, size=min(40000, len(tr_idx)), replace=False)
        m = RandomForestRegressor(n_estimators=150, min_samples_leaf=30, n_jobs=-1, random_state=0)
        m.fit(X[sub], L[sub])
    else:
        m = LinearRegression().fit(X[tr_idx], L[tr_idx])
    return m.predict(X[te_idx])

cv_rows, resid = [], {}
for name in ["M0", "M1", "M2", "M3", "M4"]:
    oof_sp = np.zeros_like(L); oof_rd = np.zeros_like(L)
    for tri, tei in splits_sp: oof_sp[tei] = fit_predict(name, tri, tei)
    for tri, tei in splits_rd: oof_rd[tei] = fit_predict(name, tri, tei)
    # in-sample fit (linear models) for anomaly definition; RF uses out-of-block predictions
    if name == "M4":
        pred_anom = oof_sp
    else:
        pred_anom = LinearRegression().fit(feats(name), L).predict(feats(name))
    resid[name] = L - pred_anom
    cv_rows.append(dict(model=name, label=labels[name],
                        r2_spatial=r2_score(L, oof_sp), rmse_spatial=mean_squared_error(L, oof_sp) ** 0.5,
                        r2_random=r2_score(L, oof_rd), rmse_random=mean_squared_error(L, oof_rd) ** 0.5,
                        r2_insample=r2_score(L, pred_anom) if name != "M4" else np.nan))
    print(name, {k: round(v, 3) for k, v in cv_rows[-1].items() if isinstance(v, float)})
cv = pd.DataFrame(cv_rows); cv.to_csv(OUT + "model_cv.csv", index=False)

# M0 summary
r_elev, _ = pearsonr(E, L)
c1 = np.polyfit(E, L, 1); c2 = np.polyfit(E, L, 2)
stats.update(r_elev_lst=float(r_elev), lapse_per_km=float(c1[0] * 1000), intercept=float(c1[1]),
             r2_lin=float(r2_score(L, np.polyval(c1, E))), r2_quad=float(r2_score(L, np.polyval(c2, E))),
             anom_sd_M0=float(resid["M0"].std()))
print("r(elev,LST)=%.3f lapse=%.3f C/km R2lin=%.3f R2quad=%.3f" % (r_elev, c1[0] * 1000, stats["r2_lin"], stats["r2_quad"]))
print("anomaly SD per model:", {k: round(float(v.std()), 3) for k, v in resid.items()})

# ---------------- Illumination relationships of M0 anomaly -----------------
rho_ci, _ = spearmanr(resid["M0"], CI)
rho_sl, _ = spearmanr(resid["M0"], S)
rho_nd, _ = spearmanr(resid["M0"], N)
r2_ill = r2_score(resid["M0"], LinearRegression().fit(np.c_[S, CI], resid["M0"]).predict(np.c_[S, CI]))
stats.update(rho_anomM0_cosinc=float(rho_ci), rho_anomM0_slope=float(rho_sl), rho_anomM0_ndvi=float(rho_nd),
             r2_anomM0_on_illum=float(r2_ill))
print("Spearman anomaly M0 vs cos_inc %.3f, slope %.3f, NDVI %.3f; R2 anomaly~(slope,cos_inc)=%.3f" % (rho_ci, rho_sl, rho_nd, r2_ill))
# fit M3 coefficients
lr3 = LinearRegression().fit(feats("M3"), L)
stats["M3_coef"] = dict(zip(["elev", "slope_deg", "cos_inc", "ndvi"], map(float, lr3.coef_)))
print("M3 coef", stats["M3_coef"])

# ---------------- Moran's I (queen contiguity on 30-m grid) -----------------
def moran(arr_valid_values):
    Z = np.full(shape, np.nan); Z[valid] = arr_valid_values
    z = Z - np.nanmean(Z)
    num = 0.0; W = 0.0
    for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        a = z[max(0, dr):shape[0] + min(0, dr), max(0, dc):shape[1] + min(0, dc)]
        b = z[max(0, -dr):shape[0] + min(0, -dr), max(0, -dc):shape[1] + min(0, -dc)]
        ok = np.isfinite(a) & np.isfinite(b)
        num += np.sum(a[ok] * b[ok]); W += ok.sum()
    n = np.isfinite(z).sum(); den = np.nansum(z ** 2)
    return (n / W) * num / den
stats["moran_M0"] = float(moran(resid["M0"])); stats["moran_M3"] = float(moran(resid["M3"]))
print("Moran's I (queen, 30 m): M0 %.3f, M3 %.3f" % (stats["moran_M0"], stats["moran_M3"]))

# ---------------- Village aggregation + block bootstrap CI -----------------
names = g_utm["NAME_4"].tolist()
B = 2000
def village_table(model):
    rows = []
    z = resid[model]
    for i, nm in enumerate(names):
        m = V == i
        if m.sum() < 30: continue
        zb = z[m]; bb = BID[m]
        ub_, inv = np.unique(bb, return_inverse=True)
        sums = np.bincount(inv, weights=zb); cnt = np.bincount(inv).astype(float)
        boots = np.empty(B)
        for k in range(B):
            pick = rng.integers(0, len(ub_), len(ub_))
            boots[k] = sums[pick].sum() / cnt[pick].sum()
        rows.append(dict(village=nm, n_pixels=int(m.sum()), n_blocks=int(len(ub_)), mean_elev=float(E[m].mean()),
                         mean_lst=float(L[m].mean()), mean_ndvi=float(N[m].mean()), mean_anom=float(zb.mean()),
                         ci_lo=float(np.percentile(boots, 2.5)), ci_hi=float(np.percentile(boots, 97.5)),
                         p90_anom=float(np.percentile(zb, 90))))
    return pd.DataFrame(rows).sort_values("mean_anom", ascending=False).reset_index(drop=True)

vt = {m: village_table(m) for m in ["M0", "M1", "M2", "M3", "M4"]}
for m, t in vt.items(): t.to_csv(OUT + f"village_{m}.csv", index=False)
print(vt["M0"].round(2).to_string())
print(vt["M3"].round(2).to_string())

# rank stability vs M0
base = vt["M0"].set_index("village")["mean_anom"]
rank_rows = []
for m in ["M1", "M2", "M3", "M4"]:
    cur = vt[m].set_index("village")["mean_anom"].reindex(base.index)
    rho, _ = spearmanr(base, cur)
    top5_0 = set(base.sort_values(ascending=False).index[:5]); top5_m = set(cur.sort_values(ascending=False).index[:5])
    rank_rows.append(dict(model=m, spearman_vs_M0=float(rho), top5_overlap=len(top5_0 & top5_m),
                          top3=", ".join(cur.sort_values(ascending=False).index[:3])))
rk = pd.DataFrame(rank_rows); rk.to_csv(OUT + "rank_stability.csv", index=False); print(rk.to_string())

# ---------------- Hotspot (top 5% of M0 anomaly) vs illumination -----------------
thr = np.percentile(resid["M0"], 95)
hs = resid["M0"] >= thr
stats.update(hot_thr=float(thr), hot_median_cosinc=float(np.median(CI[hs])), all_median_cosinc=float(np.median(CI)),
             hot_median_slope=float(np.median(S[hs])), all_median_slope=float(np.median(S)),
             hot_median_ndvi=float(np.median(N[hs])), all_median_ndvi=float(np.median(N)),
             hot_in_M3_top5=float(np.mean(resid["M3"][hs] >= np.percentile(resid["M3"], 95))))
print("hotspot thr %.2f; median cos_inc hot/all %.3f/%.3f; slope %.1f/%.1f; NDVI %.3f/%.3f; share also top5%% in M3: %.2f" % (
    thr, stats["hot_median_cosinc"], stats["all_median_cosinc"], stats["hot_median_slope"], stats["all_median_slope"],
    stats["hot_median_ndvi"], stats["all_median_ndvi"], stats["hot_in_M3_top5"]))

# ---------------- Save arrays for plotting ----------------
def grid(vals):
    Z = np.full(shape, np.nan, dtype=np.float32); Z[valid] = vals; return Z
np.savez_compressed(OUT + "grids.npz", dem=dem, ndvi=np.where(aoi, ndvi, np.nan), lst=np.where(aoi & ~cloudy, lst, np.nan),
                    slope=np.degrees(slope), cos_inc=cos_inc, vid=vid, valid=valid, aoi=aoi,
                    a0=grid(resid["M0"]), a3=grid(resid["M3"]), a4=grid(resid["M4"]))
json.dump(dict(transform=list(tr)[:6], crs=crs.to_string(), shape=list(shape)), open(OUT + "meta.json", "w"))
json.dump(stats, open(OUT + "stats.json", "w"), indent=2)
pd.DataFrame(dict(elev=E, lst=L, ndvi=N, slope=S, cos_inc=CI, village=V, block=BID, a0=resid["M0"], a3=resid["M3"])).sample(
    30000, random_state=1).to_csv(OUT + "sample_pixels.csv", index=False)
print("saved")
