import numpy as np, pandas as pd, json
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import r2_score
import os
O=os.environ.get("GS_OUT_DIR", "outputs")+"/"
g=np.load(O+"grids.npz"); valid=g["valid"]
dem=g["dem"]; dzdx=np.gradient(dem,30,axis=1); dzdy=-np.gradient(dem,30,axis=0)
asp=np.mod(np.arctan2(-dzdx,-dzdy),2*np.pi)
rr,cc=np.where(valid)
E=dem[valid];S=g["slope"][valid];CI=g["cos_inc"][valid];N=g["ndvi"][valid];AS=asp[valid];L=g["lst"][valid]
X0=np.c_[E]; X3=np.c_[E,S,CI,N]; X4=np.c_[E,S,CI,N,np.sin(AS),np.cos(AS)]; X6=np.c_[X4,cc*30.0,rr*30.0]
models={"Linear, elevation (M0)":("lin",X0),"Linear, +slope/illum./NDVI (M3)":("lin",X3),
 "Random Forest (M4)":("rf",X4),"Gradient boosting":("gb",X4),"Random Forest + x,y coordinates":("rf",X6)}
rng=np.random.default_rng(0)
def fit(kind,X,tr):
    if kind=="lin": return LinearRegression().fit(X[tr],L[tr])
    sub=rng.choice(tr,min(40000,len(tr)),replace=False)
    m=RandomForestRegressor(150,min_samples_leaf=30,n_jobs=-1,random_state=0) if kind=="rf" else HistGradientBoostingRegressor(max_iter=200,learning_rate=0.1,min_samples_leaf=50,random_state=0)
    return m.fit(X[sub],L[sub])
schemes={"Random 5-fold":None,"Block 1 km":33,"Block 3 km":100,"Block 6 km":200}
rows=[]
for sn,bs in schemes.items():
    if bs is None: sp=list(KFold(5,shuffle=True,random_state=1).split(E))
    else:
        B=(rr//bs)*1000+(cc//bs); sp=list(GroupKFold(5).split(E,L,B)); nb=len(np.unique(B))
    for mn,(k,X) in models.items():
        oof=np.zeros_like(L)
        for tr,te in sp: oof[te]=fit(k,X,tr).predict(X[te])
        rows.append(dict(scheme=sn,model=mn,r2=r2_score(L,oof),rmse=float(np.sqrt(np.mean((L-oof)**2)))))
        print(sn,mn,round(rows[-1]["r2"],3),round(rows[-1]["rmse"],2),flush=True)
pd.DataFrame(rows).to_csv(O+"ml_scheme_comparison.csv",index=False)
# permutation importance of RF M4 on block-held-out folds (3 km)
B=(rr//100)*1000+(cc//100); sp=list(GroupKFold(5).split(E,L,B))
names=["Elevation","Slope","Illumination cos","NDVI","sin(aspect)","cos(aspect)"]
imp=np.zeros((5,6))
for i,(tr,te) in enumerate(sp):
    m=fit("rf",X4,tr); te=rng.choice(te,min(15000,len(te)),replace=False); base=r2_score(L[te],m.predict(X4[te]))
    for j in range(6):
        d=[]
        for _ in range(3):
            Xp=X4[te].copy(); Xp[:,j]=rng.permutation(Xp[:,j]); d.append(base-r2_score(L[te],m.predict(Xp)))
        imp[i,j]=np.mean(d)
pd.DataFrame(dict(feature=names,mean=imp.mean(0),sd=imp.std(0))).to_csv(O+"rf_perm_importance.csv",index=False)
print(pd.DataFrame(dict(feature=names,mean=imp.mean(0).round(3),sd=imp.std(0).round(3))))
