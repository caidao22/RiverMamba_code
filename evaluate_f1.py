import numpy as np
import xarray as xr
import os
import glob
from utils.utils import evaluator, get_logger, log_string, fix_seed
from dataset.RiverMamba_dataset import RiverMamba_Dataset
import config as config_file
import argparse

output_path = r'./inference_full_map'
years_test = ['2024']

def evaluate():

    config = config_file.read_arguments(train=False, print=False, save=False)
    config.name = 'evaluate_f1'
    config.years_test = years_test

    logger = get_logger(config)
    fix_seed(config.seed)

    log_string(logger, "loading dataset for thresholds ...")

    val_dataset = RiverMamba_Dataset(
        root_glofas_reanalysis=config.root_glofas_reanalysis,
        root_static=config.root_static,
        root_era5_land_reanalysis=config.root_era5_land_reanalysis,
        root_hres_forecast=config.root_hres_forecast,
        root_cpc=config.root_cpc,
        root_obs=config.root_obs,
        nan_fill=config.nan_fill,
        delta_t=config.delta_t,
        delta_t_f=config.delta_t_f,
        is_hres_forecast=True,
        is_shuffle=False,
        is_sample_aifas=False,
        is_sample=False,
        n_points=6221926,
        variables_glofas=config.variables_glofas,
        variables_era5_land=config.variables_era5_land,
        variables_static=config.variables_static,
        variables_hres_forecast=config.variables_hres_forecast,
        variables_cpc=config.variables_cpc,
        variables_glofas_log1p=config.variables_glofas_log1p,
        variables_era5_land_log1p=config.variables_era5_land_log1p,
        variables_static_log1p=config.variables_static_log1p,
        variables_hres_forecast_log1p=config.variables_hres_forecast_log1p,
        variables_cpc_log1p=config.variables_cpc_log1p,
        is_add_xyz=config.is_add_xyz,
        curves=config.curves,
        is_shuffle_curves=False,
        is_norm=config.is_norm,
        years=years_test,
        lat_min=None,
        lat_max=None,
        lon_min=None,
        lon_max=None,
        static_dataset=config.static_dataset,
        is_sample_curves=False,
        is_obs=False,
        is_val=True
    )

    thresholds = val_dataset.get_flood_thresholds()
    thresholds = thresholds.T

    eval_metric = evaluator(logger, mode='test', lead_time=config.delta_t_f)

    pred_files = sorted(glob.glob(os.path.join(output_path, '*.nc')))
    log_string(logger, "found %d prediction files" % len(pred_files))

    for f_idx, pred_file in enumerate(pred_files):
        file_name = os.path.splitext(os.path.basename(pred_file))[0]
        year = file_name[:4]

        pred_ds = xr.open_dataset(pred_file)
        pred = pred_ds['dis24'].values

        target_files = []
        for t in range(config.delta_t_f):
            date = np.datetime64(f'{year}-{file_name[4:6]}-{file_name[6:8]}') + np.timedelta64(t + 1, 'D')
            date_str = str(date).replace('-', '')
            target_path = os.path.join(config.root_glofas_reanalysis, date_str[:4], date_str + '.nc')
            if os.path.isfile(target_path):
                target_files.append(target_path)
            else:
                break

        if len(target_files) != config.delta_t_f:
            continue

        target = []
        for tf in target_files:
            ds = xr.open_dataset(tf)
            target.append(ds['dis24'].values)
        target = np.stack(target, axis=0)

        data_lead_time = np.arange(1, config.delta_t_f + 1)

        for t in range(config.delta_t_f):
            pred_t = pred[t:t+1, :, np.newaxis]
            target_t = target[t:t+1, :, np.newaxis]

            eval_metric(pred_t, target_t, thresholds[np.newaxis, :, :], data_lead_time[t:t+1])

        if (f_idx + 1) % 100 == 0:
            log_string(logger, "processed %d / %d files" % (f_idx + 1, len(pred_files)))

    eval_metric.get_results()


if __name__ == '__main__':
    evaluate()
