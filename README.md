# gunung-salak-lst-spatial-cv

Code, derived data and result tables for *"Spatially honest evaluation of machine-learning background models for single-scene
land surface temperature screening in a volcanic geothermal area: a Landsat 8 case study at Gunung Salak, Indonesia"*
(Ratnaningsih, Purwoningsih, Leviany, Sugiarti, Prasetyo; submitted to the *Journal of Applied Remote Sensing*).

The scripts fit six background models of Landsat 8 land surface temperature (LST) over fifteen villages at Gunung Salak
(linear, quadratic, linear with slope/illumination/NDVI, Random Forest, gradient boosting, Random Forest with coordinates),
evaluate them under random and spatial block cross-validation (1, 3 and 6 km), compute permutation importance, village-level
residuals with block-bootstrap intervals, rank stability and Moran's I.

* **License:** MIT (`LICENSE`). **Language:** English; Python ≥ 3.10.

## Layout

| Path | Content |
|---|---|
| `scripts/01_background_models_and_village_residuals.py` | Clips the scene, cleans the DEM, fits models M0–M4, runs random and block CV, village residuals, bootstrap, rank stability, Moran's I |
| `scripts/02_ml_scheme_comparison.py` | Five-model comparison across random, 1, 3 and 6 km CV and RF permutation importance (reads outputs of script 01) |
| `scripts/03_make_figures.py` | Figures of the article (reads outputs of script 01/02) |
| `data/derived/` | Derived 30 m rasters: LST (°C), NDVI, aligned elevation (m), residuals of M0 and M3 (UTM 48S) |
| `results/` | Result tables: model CV, ML scheme comparison, permutation importance, village residuals M0/M3 |

## Reproducing the analysis

The raw inputs are public but are not redistributed here: the Landsat 8 Collection 2 Level-2 scene
`LC08_L2SP_122065_20250826_02_T1` (USGS/Earth Explorer), the DEMNAS tile (Indonesian Geospatial Information Agency) and the
village polygons (GADM level 4). Place them in a directory and run:

```bash
pip install -r requirements.txt
export GS_INPUT_DIR=/path/to/raw   GS_OUT_DIR=outputs
python scripts/01_background_models_and_village_residuals.py
python scripts/02_ml_scheme_comparison.py
python scripts/03_make_figures.py
```

Expected input file names are set at the top of script 01 (scene prefix, `DEMNAS_Gunung_Salak.tif`,
`Wilayah_Geothermal-Gunung_Salak.shp` for the village polygons). Random seeds are fixed in the scripts. The tables in
`results/` are the ones reported in the article; the derived rasters let you inspect or reuse the residual maps without the raw data.

## Not included

* The Ciseeng benchmark (random-split Random Forest versus linear regression, 4,129 pixel-months) was produced in an earlier
  phase; its pixel table and script are not part of this repository (see Sec. 3.5 of the article). It is available from the
  corresponding author on reasonable request.
* The script paths were generalised to environment variables for release; the scripts were not re-run end to end after that change
  because the raw inputs are not stored in this repository.

## Interpretation

LST residuals here are a screening layer. They are not evidence of geothermal potential: no georeferenced manifestations, fault
data or heat-flow measurements were used.

## Citation

See `CITATION.cff`.
