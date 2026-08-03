# Evaluation

This document describes how to reproduce the CONUS evaluation results for RiverMamba.

## Prerequisites

1. Download the pretrained model and all required datasets following the instructions in `README.md`.
2. Ensure the following directories are populated under `scripts/`:
   - `RiverMamba_pretrained_models/` (pretrained model weights and config)
   - `GloFAS_Reanalysis_Global/` (GloFAS reanalysis daily discharge, organized as `YYYY/YYYYMMDD.nc`)
   - `GloFAS_Static/` (static fields: `masks/mask_valid.nc`, `masks/mask_AIFAS_points.nc`, `NeuralFAS_HydroRIVERS_static.nc`, `threshold/flood_threshold_glofas_v4_rl_*.nc`)
   - `ERA5-Land_Reanalysis_Global/`
   - `ECMWF_HRES_Global/`
   - `CPC_Global/`

## Step 1: Run inference

Generate full-map predictions for the test year (2024). Each init date produces a NetCDF file with 7-day lead-time predictions over all 6.2M valid global pixels.

```bash
python3 inference_full_map.py --config_file ./scripts/RiverMamba_pretrained_models/RiverMamba_full_map_reanalysis.txt --gpu_id 0 --root_glofas_reanalysis ./scripts/GloFAS_Reanalysis_Global --root_static ./scripts/GloFAS_Static --root_era5_land_reanalysis ./scripts/ERA5-Land_Reanalysis_Global --root_hres_forecast ./scripts/ECMWF_HRES_Global --root_cpc ./scripts/CPC_Global --root_obs ./scripts/GloFAS_Reanalysis_Global
```

Output: 294 per-date files in `./inference_full_map/` (e.g., `20240105.nc` with shape `(7, 6221926)`).

## Step 2: Evaluate on CONUS

The evaluation script computes aggregated continuous metrics (RMSE, MAE, NSE, KGE) and categorical flood metrics (F1, POD, FAR, CSI) at 9 return periods (1.5 to 500 years) for both the model and a persistence baseline (carry-forward of observed discharge at init date).

CONUS is defined as the intersection of the GloFAS valid-pixel mask with swinflood's river discharge mask (~414,795 grid cells at 0.05 deg resolution).

### All CONUS points (default)

```bash
python3 evaluate_f1_conus.py --config_file ./scripts/RiverMamba_pretrained_models/RiverMamba_full_map_reanalysis.txt --gpu_id 0 --root_glofas_reanalysis ./scripts/GloFAS_Reanalysis_Global --root_static ./scripts/GloFAS_Static --root_era5_land_reanalysis ./scripts/ERA5-Land_Reanalysis_Global --root_hres_forecast ./scripts/ECMWF_HRES_Global --root_cpc ./scripts/CPC_Global
```

### Subset evaluation

Use `--split` to break down metrics by subset:

**By upstream area** (small <500 km², medium 500-5000 km², large >5000 km²):
```bash
python3 evaluate_f1_conus.py --split uparea --config_file ./scripts/RiverMamba_pretrained_models/RiverMamba_full_map_reanalysis.txt --gpu_id 0 --root_glofas_reanalysis ./scripts/GloFAS_Reanalysis_Global --root_static ./scripts/GloFAS_Static --root_era5_land_reanalysis ./scripts/ERA5-Land_Reanalysis_Global --root_hres_forecast ./scripts/ECMWF_HRES_Global --root_cpc ./scripts/CPC_Global
```

**By AIFAS diagnostic river points vs non-AIFAS:**
```bash
python3 evaluate_f1_conus.py --split aifas --config_file ./scripts/RiverMamba_pretrained_models/RiverMamba_full_map_reanalysis.txt --gpu_id 0 --root_glofas_reanalysis ./scripts/GloFAS_Reanalysis_Global --root_static ./scripts/GloFAS_Static --root_era5_land_reanalysis ./scripts/ERA5-Land_Reanalysis_Global --root_hres_forecast ./scripts/ECMWF_HRES_Global --root_cpc ./scripts/CPC_Global
```

**By flood activity** (points that exceeded RP=2 threshold vs not):
```bash
python3 evaluate_f1_conus.py --split flood --config_file ./scripts/RiverMamba_pretrained_models/RiverMamba_full_map_reanalysis.txt --gpu_id 0 --root_glofas_reanalysis ./scripts/GloFAS_Reanalysis_Global --root_static ./scripts/GloFAS_Static --root_era5_land_reanalysis ./scripts/ERA5-Land_Reanalysis_Global --root_hres_forecast ./scripts/ECMWF_HRES_Global --root_cpc ./scripts/CPC_Global
```

Subset masks are cached in `./inference_full_map/cached_masks/` after the first run.

### Per-cell metric NetCDFs

Use `--write_nc` to produce per-cell NSE and F1 arrays broken down by all 5 subsets (aifas, flood-active, uparea-small/medium/large). These match the schema used by swinflood's `evaluate_splits.py` for cross-model comparison.

```bash
python3 evaluate_f1_conus.py --write_nc --config_file ./scripts/RiverMamba_pretrained_models/RiverMamba_full_map_reanalysis.txt --gpu_id 0 --root_glofas_reanalysis ./scripts/GloFAS_Reanalysis_Global --root_static ./scripts/GloFAS_Static --root_era5_land_reanalysis ./scripts/ERA5-Land_Reanalysis_Global --root_hres_forecast ./scripts/ECMWF_HRES_Global --root_cpc ./scripts/CPC_Global
```

Output:
- `./inference_full_map/rivermamba_subset_metrics.nc`
- `./inference_full_map/persistence_subset_metrics.nc`

Each file contains:
- `nse_<subset>(lead_time, cell_<subset>)` — per-cell NSE over the test year
- `f1_<subset>(lead_time, return_period, cell_<subset>)` — per-cell F1 at each return period (NaN where no events occurred)

`--write_nc` can be combined with `--split` to also print aggregated metrics for a specific split.
