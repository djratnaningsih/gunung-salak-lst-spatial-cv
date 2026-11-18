import json, numpy as np, pandas as pd, geopandas as gpd, rasterio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
import os
O=os.environ.get("GS_OUT_DIR", "outputs")+"/"; F=O+"fig/"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.edgecolor":"#c3c2b7","axes.linewidth":0.6,
 "text.color":"#0b0b0b","axes.labelcolor":"#52514e","xtick.color":"#52514e","ytick.color":"#52514e","figure.facecolor":"#fcfcfb","axes.facecolor":"#fcfcfb","savefig.facecolor":"#fcfcfb"})
BL="#2a78d6"; OR="#eb6834"
seq_b=LinearSegmentedColormap.from_list("sb",["#cde2fb","#86b6ef","#3987e5","#1c5cab","#0d366b"])
seq_o=LinearSegmentedColormap.from_list("so",["#fde6da","#f6b393","#eb6834","#b8431a","#6e2509"])
seq_g=LinearSegmentedColormap.from_list("sg",["#e3f1de","#9fd29a","#3a9f3a","#006300","#003a00"])
div=LinearSegmentedColormap.from_list("dv",["#184f95","#3987e5","#b7d3f6","#f0efec","#f4b3a8","#e34948","#8f1f1f"])
g=np.load(O+"grids.npz"); meta=json.load(open(O+"meta.json")); tr=meta["transform"]
H,W=meta["shape"]; ext=[tr[2],tr[2]+W*tr[0],tr[5]+H*tr[4],tr[5]]
gd=gpd.read_file(os.environ.get("GS_INPUT_DIR", "data/raw")+"/Wilayah_Geothermal-Gunung_Salak.shp").to_crs(meta["crs"])
aoi=g["aoi"]
def mp(ax,A,cmap,vmin,vmax,title,norm=None,bnd=True):
    A=np.where(aoi,A,np.nan)
    im=ax.imshow(A,extent=ext,cmap=cmap,vmin=None if norm else vmin,vmax=None if norm else vmax,norm=norm,interpolation="nearest")
    if bnd: gd.boundary.plot(ax=ax,color="#0b0b0b",lw=0.3,alpha=.6)
    ax.set_xlim(ext[0],ext[1]); ax.set_ylim(ext[2],ext[3]); ax.set_aspect("equal"); ax.set_title(title,fontsize=8.5,loc="left")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    for s in ax.spines.values(): s.set_visible(False)
    return im
# Fig 2: inputs
fig,axs=plt.subplots(1,3,figsize=(7.2,2.9))
for ax,A,cm,lab,t in [(axs[0],g["dem"],seq_b,"Elevation (m)","(a) Elevation"),(axs[1],g["lst"],seq_o,"LST (°C)","(b) LST, 26 Aug 2025"),(axs[2],g["ndvi"],seq_g,"NDVI","(c) NDVI")]:
    im=mp(ax,A,cm,None,None,t,bnd=False) if False else mp(ax,A,cm,np.nanpercentile(A,1),np.nanpercentile(A,99),t,bnd=False)
    cb=fig.colorbar(im,ax=ax,orientation="horizontal",fraction=0.05,pad=0.04,shrink=.85); cb.set_label(lab,fontsize=7.5); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7,length=2)
plt.tight_layout(rect=(0,0,1,0.97)); plt.savefig(F+"fig_inputs.png",dpi=400); plt.close()
# Fig 3: scatter
s=pd.read_csv(O+"sample_pixels.csv"); c1=np.polyfit(s.elev,s.lst,1)
fig,ax=plt.subplots(figsize=(3.4,2.9))
hb=ax.hexbin(s.elev,s.lst,gridsize=45,cmap=seq_b,mincnt=1,linewidths=0.1,bins="log")
xs=np.array([280,2100]); ax.plot(xs,np.polyval(c1,xs),color="#0b0b0b",lw=1.2)
ax.set_xlabel("Elevation (m a.s.l.)"); ax.set_ylabel("LST (°C)")
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
ax.grid(color="#e1e0d9",lw=.4); ax.set_axisbelow(True)
cb=fig.colorbar(hb,ax=ax,pad=.02); cb.set_label("Pixel count (log)",fontsize=7.5); cb.outline.set_visible(False)
plt.tight_layout(); plt.savefig(F+"fig_scatter.png",dpi=400); plt.close()
# Fig 4: anomaly maps
fig,axs=plt.subplots(1,2,figsize=(7.2,3.3)); nm=TwoSlopeNorm(vmin=-4,vcenter=0,vmax=4)
for ax,k,t in [(axs[0],"a0","(a) M0: elevation-only background"),(axs[1],"a3","(b) M3: + slope, illumination, NDVI")]:
    im=mp(ax,g[k],div,None,None,t,norm=nm)
cb=fig.colorbar(im,ax=axs,orientation="horizontal",fraction=0.04,pad=0.03,shrink=.5); cb.set_label("Residual LST (°C); positive = warmer than expected",fontsize=7.5); cb.outline.set_visible(False)
plt.savefig(F+"fig_anomaly.png",dpi=400,bbox_inches="tight"); plt.close()
# Fig 5: forest
v0=pd.read_csv(O+"village_M0.csv"); v3=pd.read_csv(O+"village_M3.csv").set_index("village")
order=v0.village.tolist()[::-1]; y=np.arange(len(order))
fig,ax=plt.subplots(figsize=(5.2,3.8))
for off,df,col,lab in [(0.17,v0.set_index("village"),BL,"M0 (elevation only)"),(-0.17,v3,OR,"M3 (+ slope, illumination, NDVI)")]:
    d=df.loc[order]
    ok=(d.n_blocks>=4).values
    ax.hlines((y+off)[ok],d.ci_lo[ok],d.ci_hi[ok],color=col,lw=1.1)
    ax.plot(d.mean_anom[ok],(y+off)[ok],"o",ms=4.5,color=col,mec="#fcfcfb",mew=.8)
    ax.plot(d.mean_anom[~ok],(y+off)[~ok],"o",ms=4.5,mfc="none",color=col,mew=1)
ax.axvline(0,color="#898781",lw=.7); ax.set_yticks(y); ax.set_yticklabels(order,fontsize=7.5)
ax.set_xlabel("Mean residual LST (°C)\nBars: 95% block-bootstrap interval; open marker: <4 blocks, no interval",fontsize=7.5)
ax.grid(axis="x",color="#e1e0d9",lw=.4); ax.set_axisbelow(True)
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
ax.legend(handles=[Line2D([],[],color=BL,marker="o",lw=1.1,label="M0 (elevation only)"),Line2D([],[],color=OR,marker="o",lw=1.1,label="M3 (+ slope, illumination, NDVI)")],frameon=False,loc="lower center",bbox_to_anchor=(0.45,1.0),ncol=2,fontsize=7)
plt.tight_layout(); plt.savefig(F+"fig_forest.png",dpi=400); plt.close()
# Fig 6: CV
cv=pd.read_csv(O+"model_cv.csv"); x=np.arange(5); w=.36
fig,ax=plt.subplots(figsize=(4.4,2.7))
ax.bar(x-w/2-.01,cv.r2_random,w,color=BL,label="Random 5-fold"); ax.bar(x+w/2+.01,cv.r2_spatial,w,color=OR,label="Spatial block 5-fold (3 km)")
ax.set_xticks(x); ax.set_xticklabels(cv.model); ax.set_ylim(.75,.95); ax.set_ylabel("Out-of-fold R²")
ax.grid(axis="y",color="#e1e0d9",lw=.4); ax.set_axisbelow(True)
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
ax.legend(frameon=False,fontsize=7.5,loc="upper left")
plt.tight_layout(); plt.savefig(F+"fig_cv.png",dpi=400); plt.close()
print("ok")
