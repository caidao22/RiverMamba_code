# Usage:
# python3 evaluate_f1_conus.py --config_file ./scripts/RiverMamba_pretrained_models/RiverMamba_full_map_reanalysis.txt --gpu_id 0 --root_glofas_reanalysis ./scripts/GloFAS_Reanalysis_Global --root_static ./scripts/GloFAS_Static --root_era5_land_reanalysis ./scripts/ERA5-Land_Reanalysis_Global --root_hres_forecast ./scripts/ECMWF_HRES_Global --root_cpc ./scripts/CPC_Global
#
# Optional: --split {none,uparea,aifas,flood}
#   none   - all CONUS points (default)
#   uparea - split by upstream area: small (<500 km²), medium (500-5000 km²), large (>5000 km²)
#   aifas  - split by AIFAS diagnostic river points vs non-AIFAS
#   flood  - split by whether a point exceeded RP=2 flood threshold during the test period
#
# Optional: --write_nc
#   Write per-cell NSE and F1 NetCDFs (rivermamba_subset_metrics.nc, persistence_subset_metrics.nc)
#   matching the schema from swinflood's evaluate_splits.py for cross-model comparison plots.

import numpy as np
import xarray as xr
import os
import re
import sys
import glob
import csv
from utils.utils import get_logger, log_string, fix_seed
import config as config_file

output_path = r'./inference_full_map'
years_test = ['2024']

SWINFLOOD_MASKS = '/vast/users/hongzhang/Projects/swinflood/preprocessed_data/masks.npz'
SWINFLOOD_LAT = '/vast/users/hongzhang/Projects/swinflood/preprocessed_data/lat_24hr_2005-2024.npy'
SWINFLOOD_LON = '/vast/users/hongzhang/Projects/swinflood/preprocessed_data/lon_24hr_2005-2024.npy'
DISCHARGE_VAR = 'river_discharge_in_the_last_24_hours'

return_periods = [1.5, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]

UPAREA_SMALL = 5e8    # 500 km² in m²
UPAREA_LARGE = 5e9    # 5000 km² in m²

SUBSET_DISPLAY = {
    "aifas": "aifas",
    "flood": "flood-active",
    "uparea_small": "uparea-small",
    "uparea_medium": "uparea-medium",
    "uparea_large": "uparea-large",
}

MIXED_LEAD_SENTINEL = -1


def _safe_dim(name):
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


def get_conus_mask(root_static):
    mask_path = os.path.join(root_static, "masks/mask_valid.nc")
    if not os.path.isfile(mask_path):
        mask_path = os.path.join('./scripts', root_static, "masks/mask_valid.nc")
    ds = xr.open_dataset(mask_path)
    lat_rm = ds['latitude'].values
    lon_rm = ds['longitude'].values
    mask_valid_2d = ds['mask_valid'].values
    ds.close()

    lat_sw = np.load(SWINFLOOD_LAT)
    lon_sw = np.load(SWINFLOOD_LON)
    sw_discharge = np.load(SWINFLOOD_MASKS)[DISCHARGE_VAR]

    lat_idx = np.array([np.argmin(np.abs(lat_rm - l)) for l in lat_sw])
    lon_idx = np.array([np.argmin(np.abs(lon_rm - l)) for l in lon_sw])

    discharge_on_rm = np.zeros_like(mask_valid_2d, dtype=bool)
    for i, li in enumerate(lat_idx):
        for j, lj in enumerate(lon_idx):
            if sw_discharge[i, j]:
                discharge_on_rm[li, lj] = True

    valid_indices = np.where(mask_valid_2d.flatten() == 1)[0]
    conus_mask = discharge_on_rm.flatten()[valid_indices]
    return conus_mask


MASK_CACHE_DIR = os.path.join(output_path, 'cached_masks')


def _cache_path(name):
    return os.path.join(MASK_CACHE_DIR, name + '.npz')


def _save_masks(name, masks_dict, logger):
    os.makedirs(MASK_CACHE_DIR, exist_ok=True)
    path = _cache_path(name)
    np.savez_compressed(path, **masks_dict)
    log_string(logger, "saved cached mask: %s" % path)


def _load_masks(name, logger):
    path = _cache_path(name)
    if not os.path.isfile(path):
        return None
    data = dict(np.load(path))
    log_string(logger, "loaded cached mask: %s" % path)
    return data


def build_subsets(split_mode, config, conus_mask, mask_valid, thresholds_conus,
                  lead_time, n_rp, pred_files, logger):
    """Build named subsets. Returns dict of {name: {'mask': bool array over conus, 'thresholds': (n_rp, n_sub)}}."""
    n_conus = int(conus_mask.sum())

    if split_mode == 'none':
        return {'all': {
            'mask': np.ones(n_conus, dtype=bool),
            'thresholds': thresholds_conus,
        }}

    cached = _load_masks(split_mode, logger)

    if split_mode == 'uparea':
        if cached is None:
            static_path = os.path.join(config.root_static, "NeuralFAS_HydroRIVERS_static.nc")
            ds = xr.open_dataset(static_path)
            uparea = ds['uparea'].values.flatten()[mask_valid == 1]
            ds.close()
            uparea_conus = uparea[conus_mask]

            small = uparea_conus < UPAREA_SMALL
            large = uparea_conus >= UPAREA_LARGE
            medium = ~small & ~large
            _save_masks('uparea', dict(small=small, medium=medium, large=large), logger)
        else:
            small, medium, large = cached['small'], cached['medium'], cached['large']

        log_string(logger, "uparea split: small=%d, medium=%d, large=%d" %
                   (small.sum(), medium.sum(), large.sum()))

        subsets = {}
        for name, sub_mask in [('small', small), ('medium', medium), ('large', large)]:
            subsets[name] = {
                'mask': sub_mask,
                'thresholds': thresholds_conus[:, sub_mask],
            }
        return subsets

    if split_mode == 'aifas':
        if cached is None:
            aifas_path = os.path.join(config.root_static, "masks/mask_AIFAS_points.nc")
            ds = xr.open_dataset(aifas_path)
            aifas_all = ds['mask_points'].values.flatten()[mask_valid == 1]
            ds.close()
            aifas_conus = (aifas_all[conus_mask] == 1)
            _save_masks('aifas', dict(aifas=aifas_conus), logger)
        else:
            aifas_conus = cached['aifas']

        log_string(logger, "aifas split: aifas=%d, non_aifas=%d" %
                   (aifas_conus.sum(), (~aifas_conus).sum()))

        return {
            'aifas': {'mask': aifas_conus, 'thresholds': thresholds_conus[:, aifas_conus]},
            'non_aifas': {'mask': ~aifas_conus, 'thresholds': thresholds_conus[:, ~aifas_conus]},
        }

    if split_mode == 'flood':
        if cached is None:
            log_string(logger, "flood split: pre-scanning targets for RP=2 exceedance ...")
            rp2_idx = return_periods.index(2.0)
            rp2_thr = thresholds_conus[rp2_idx]
            ever_flooded = np.zeros(n_conus, dtype=bool)

            for pred_file in pred_files:
                file_name = os.path.splitext(os.path.basename(pred_file))[0]
                year = file_name[:4]
                init_date = f'{year}-{file_name[4:6]}-{file_name[6:8]}'

                for t in range(lead_time):
                    date = np.datetime64(init_date) + np.timedelta64(t + 1, 'D')
                    date_str = str(date).replace('-', '')
                    target_path = os.path.join(config.root_glofas_reanalysis,
                                               date_str[:4], date_str + '.nc')
                    if not os.path.isfile(target_path):
                        break
                    ds = xr.open_dataset(target_path)
                    obs = ds['dis24'].values[conus_mask].astype(np.float64)
                    ds.close()
                    ever_flooded |= (obs >= rp2_thr)
                    if ever_flooded.all():
                        break
                if ever_flooded.all():
                    break

            _save_masks('flood', dict(ever_flooded=ever_flooded), logger)
        else:
            ever_flooded = cached['ever_flooded']

        log_string(logger, "flood split: flood_active=%d, non_flood=%d" %
                   (ever_flooded.sum(), (~ever_flooded).sum()))

        return {
            'flood_active': {'mask': ever_flooded, 'thresholds': thresholds_conus[:, ever_flooded]},
            'non_flood': {'mask': ~ever_flooded, 'thresholds': thresholds_conus[:, ~ever_flooded]},
        }

    raise ValueError(f"unknown split mode: {split_mode}")


def build_all_subsets(config, conus_mask, mask_valid, thresholds_conus,
                      lead_time, n_rp, pred_files, logger):
    """Build all 5 subsets needed for --write_nc output."""
    subsets = {}

    aifas_sub = build_subsets('aifas', config, conus_mask, mask_valid,
                              thresholds_conus, lead_time, n_rp, pred_files, logger)
    subsets['aifas'] = aifas_sub['aifas']['mask']

    flood_sub = build_subsets('flood', config, conus_mask, mask_valid,
                              thresholds_conus, lead_time, n_rp, pred_files, logger)
    subsets['flood'] = flood_sub['flood_active']['mask']

    uparea_sub = build_subsets('uparea', config, conus_mask, mask_valid,
                               thresholds_conus, lead_time, n_rp, pred_files, logger)
    subsets['uparea_small'] = uparea_sub['small']['mask']
    subsets['uparea_medium'] = uparea_sub['medium']['mask']
    subsets['uparea_large'] = uparea_sub['large']['mask']

    return subsets


def write_source_nc(out_path, source_name, rps, lead_coords, nse_by_subset, f1_by_subset):
    data_vars = {}
    coords = {
        "lead_time": np.asarray(lead_coords, dtype=np.int64),
        "return_period": np.asarray(rps, dtype=np.float32),
    }
    for csv_name, display_name in SUBSET_DISPLAY.items():
        dim = f"cell_{_safe_dim(display_name)}"
        n_cells = nse_by_subset[csv_name].shape[1]
        coords[dim] = np.arange(n_cells, dtype=np.int64)
        data_vars[f"nse_{_safe_dim(display_name)}"] = (
            ("lead_time", dim), nse_by_subset[csv_name].astype(np.float32),
            {"long_name": f"per-cell NSE over the test year ({display_name})"},
        )
        data_vars[f"f1_{_safe_dim(display_name)}"] = (
            ("lead_time", "return_period", dim), f1_by_subset[csv_name].astype(np.float32),
            {"long_name": f"per-cell F1 at GloFAS return-period thresholds ({display_name}); "
                          f"NaN where TP+FP+FN = 0"},
        )
    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs={
        "source_name": source_name,
        "subset_display_order": ",".join(SUBSET_DISPLAY[k] for k in SUBSET_DISPLAY),
        "mixed_lead_sentinel": MIXED_LEAD_SENTINEL,
        "created_by": "evaluate_f1_conus.py",
    })
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ds.to_netcdf(out_path)
    print(f"wrote {out_path}  ({len(lead_coords)} lead(s), {len(rps)} RP(s), "
          f"{len(SUBSET_DISPLAY)} subsets)")


def make_acc(lead_time, n_rp):
    return {
        'sum_p': np.zeros(lead_time, dtype=np.float64),
        'sum_t': np.zeros(lead_time, dtype=np.float64),
        'sum_pp': np.zeros(lead_time, dtype=np.float64),
        'sum_tt': np.zeros(lead_time, dtype=np.float64),
        'sum_pt': np.zeros(lead_time, dtype=np.float64),
        'sum_abs': np.zeros(lead_time, dtype=np.float64),
        'sum_sq': np.zeros(lead_time, dtype=np.float64),
        'n_cells': np.zeros(lead_time, dtype=np.int64),
        'tp': np.zeros((lead_time, n_rp), dtype=np.int64),
        'fp': np.zeros((lead_time, n_rp), dtype=np.int64),
        'fn': np.zeros((lead_time, n_rp), dtype=np.int64),
    }


def accumulate(a, t, p, o, thr):
    diff = p - o
    a['sum_p'][t] += p.sum()
    a['sum_t'][t] += o.sum()
    a['sum_pp'][t] += (p * p).sum()
    a['sum_tt'][t] += (o * o).sum()
    a['sum_pt'][t] += (p * o).sum()
    a['sum_abs'][t] += np.abs(diff).sum()
    a['sum_sq'][t] += (diff * diff).sum()
    a['n_cells'][t] += len(p)
    pred_ev = p[np.newaxis, :] >= thr
    tgt_ev = o[np.newaxis, :] >= thr
    a['tp'][t] += (pred_ev & tgt_ev).sum(axis=1)
    a['fp'][t] += (pred_ev & ~tgt_ev).sum(axis=1)
    a['fn'][t] += (~pred_ev & tgt_ev).sum(axis=1)


def compute_metrics(a):
    n = a['n_cells'].astype(np.float64)
    n = np.where(n > 0, n, 1)

    rmse = np.sqrt(a['sum_sq'] / n)
    mae = a['sum_abs'] / n
    nse = 1.0 - a['sum_sq'] / np.maximum(a['sum_tt'] - a['sum_t'] * a['sum_t'] / n, 1e-12)

    mean_p = a['sum_p'] / n
    mean_t = a['sum_t'] / n
    std_p = np.sqrt(np.maximum(a['sum_pp'] - a['sum_p'] * a['sum_p'] / n, 0) / n)
    std_t = np.sqrt(np.maximum(a['sum_tt'] - a['sum_t'] * a['sum_t'] / n, 0) / n)
    cov = (a['sum_pt'] - a['sum_p'] * a['sum_t'] / n) / n
    r = cov / np.maximum(std_p * std_t, 1e-12)
    alpha = std_p / np.maximum(std_t, 1e-12)
    beta = mean_p / np.maximum(mean_t, 1e-12)
    kge = 1.0 - np.sqrt(np.maximum((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2, 0))

    tp_f = a['tp'].astype(np.float64)
    fp_f = a['fp'].astype(np.float64)
    fn_f = a['fn'].astype(np.float64)
    precision = tp_f / np.maximum(tp_f + fp_f, 1)
    recall = tp_f / np.maximum(tp_f + fn_f, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    csi = tp_f / np.maximum(tp_f + fp_f + fn_f, 1)
    far = fp_f / np.maximum(tp_f + fp_f, 1)

    return dict(rmse=rmse, mae=mae, nse=nse, kge=kge,
                f1=f1, recall=recall, far=far, csi=csi,
                tp=a['tp'], fp=a['fp'], fn=a['fn'])


def format_results(all_metrics, subset_names, sources, lead_time, n_points_per_subset):
    msg = ""
    for sub_name in subset_names:
        n_pts = n_points_per_subset[sub_name]
        msg += "\n=== CONUS Evaluation [%s] (valid pixels: %d) ===\n" % (sub_name, n_pts)

        msg += "\nContinuous metrics on river discharge (physical units):\n"
        msg += "%12s %5s %12s %12s %10s %10s\n" % ("source", "lead", "RMSE", "MAE", "NSE", "KGE")
        for s in sources:
            m = all_metrics[(sub_name, s)]
            for t in range(lead_time):
                msg += "%12s %5d %12.4f %12.4f %10.4f %10.4f\n" % (
                    s, t + 1, m['rmse'][t], m['mae'][t], m['nse'][t], m['kge'][t])

        msg += "\nCategorical metrics by return period:\n"
        for t in range(lead_time):
            msg += "\n  lead_time = %d day(s)\n" % (t + 1)
            msg += "  %12s %8s %8s %8s %8s %8s %10s %10s %10s\n" % (
                "source", "RP(yr)", "F1", "POD", "FAR", "CSI", "TP", "FP", "FN")
            for s in sources:
                m = all_metrics[(sub_name, s)]
                for k, rp in enumerate(return_periods):
                    tp_k = m['tp'][t, k]
                    fp_k = m['fp'][t, k]
                    fn_k = m['fn'][t, k]
                    if tp_k + fn_k == 0:
                        msg += "  %12s %8.1f %8s %8s %8s %8s %10d %10d %10d\n" % (
                            s, rp, "N/A", "N/A", "N/A" if tp_k + fp_k == 0 else "%.4f" % m['far'][t, k],
                            "N/A", tp_k, fp_k, fn_k)
                    else:
                        msg += "  %12s %8.1f %8.4f %8.4f %8.4f %8.4f %10d %10d %10d\n" % (
                            s, rp, m['f1'][t, k], m['recall'][t, k], m['far'][t, k], m['csi'][t, k],
                            tp_k, fp_k, fn_k)
    return msg


def parse_split_arg():
    split_mode = 'none'
    valid = {'none', 'uparea', 'aifas', 'flood'}
    if '--split' in sys.argv:
        idx = sys.argv.index('--split')
        if idx + 1 < len(sys.argv) and sys.argv[idx + 1] in valid:
            split_mode = sys.argv[idx + 1]
            sys.argv = sys.argv[:idx] + sys.argv[idx + 2:]
        else:
            print("--split must be one of: %s" % ', '.join(sorted(valid)))
            sys.exit(1)
    write_nc = False
    if '--write_nc' in sys.argv:
        idx = sys.argv.index('--write_nc')
        sys.argv = sys.argv[:idx] + sys.argv[idx + 1:]
        write_nc = True
    return split_mode, write_nc


def evaluate():
    split_mode, write_nc = parse_split_arg()

    config = config_file.read_arguments(train=False, print=False, save=False)
    config.name = 'evaluate_f1_conus'
    config.years_test = years_test

    logger = get_logger(config)
    fix_seed(config.seed)

    lead_time = config.delta_t_f
    n_rp = len(return_periods)

    log_string(logger, "building CONUS mask ...")
    cached_conus = _load_masks('conus', logger)
    if cached_conus is not None:
        conus_mask = cached_conus['conus_mask']
    else:
        conus_mask = get_conus_mask(config.root_static)
        _save_masks('conus', dict(conus_mask=conus_mask), logger)
    n_conus = int(conus_mask.sum())
    log_string(logger, "CONUS points: %d / %d" % (n_conus, len(conus_mask)))

    log_string(logger, "loading flood thresholds ...")
    thr_dir = os.path.join(config.root_static, "threshold")
    mask_path = os.path.join(config.root_static, "masks/mask_valid.nc")
    if not os.path.isfile(mask_path):
        mask_path = os.path.join('./scripts', config.root_static, "masks/mask_valid.nc")
    mask_valid = xr.open_dataset(mask_path)['mask_valid'].values.flatten()

    thr_list = []
    for rp in return_periods:
        ds = xr.open_dataset(os.path.join(thr_dir, f"flood_threshold_glofas_v4_rl_{rp}.nc"))
        arr = ds[f"rl_{rp}"].values
        thr_list.append(arr.flatten()[mask_valid == 1])
        ds.close()
    thresholds = np.stack(thr_list, axis=0)  # (n_rp, n_valid_pixels)
    thresholds_conus = thresholds[:, conus_mask]  # (n_rp, n_conus)
    log_string(logger, "thresholds loaded: %s" % str(thresholds_conus.shape))

    pred_files = sorted(glob.glob(os.path.join(output_path, '*.nc')))
    log_string(logger, "found %d prediction files" % len(pred_files))

    log_string(logger, "split mode: %s" % split_mode)
    subsets = build_subsets(split_mode, config, conus_mask, mask_valid,
                           thresholds_conus, lead_time, n_rp, pred_files, logger)
    subset_names = list(subsets.keys())

    # --- build per-cell subset masks for --write_nc ---
    nc_subsets = None
    if write_nc:
        log_string(logger, "building all subset masks for --write_nc ...")
        nc_subsets = build_all_subsets(config, conus_mask, mask_valid,
                                      thresholds_conus, lead_time, n_rp,
                                      pred_files, logger)
        for k, v in nc_subsets.items():
            log_string(logger, "  %s: %d cells" % (k, int(v.sum())))

    sources = ['model', 'persistence']
    acc = {}
    for sub_name in subset_names:
        for s in sources:
            acc[(sub_name, s)] = make_acc(lead_time, n_rp)

    # --- per-cell accumulators for --write_nc (indexed over n_conus) ---
    pc = None
    if write_nc:
        pc = {}
        for s in sources:
            pc[s] = {
                'sum_g':  np.zeros((lead_time, n_conus), dtype=np.float64),
                'sum_gg': np.zeros((lead_time, n_conus), dtype=np.float64),
                'sum_d2': np.zeros((lead_time, n_conus), dtype=np.float64),
                'n_valid': np.zeros((lead_time, n_conus), dtype=np.int64),
                'tp': np.zeros((lead_time, n_rp, n_conus), dtype=np.int64),
                'fp': np.zeros((lead_time, n_rp, n_conus), dtype=np.int64),
                'fn': np.zeros((lead_time, n_rp, n_conus), dtype=np.int64),
            }

    daily_rows = []

    for f_idx, pred_file in enumerate(pred_files):
        file_name = os.path.splitext(os.path.basename(pred_file))[0]
        year = file_name[:4]
        init_date = f'{year}-{file_name[4:6]}-{file_name[6:8]}'

        pred_ds = xr.open_dataset(pred_file)
        pred = pred_ds['dis24'].values  # (lead_time, n_points)
        pred = pred[:, conus_mask]  # (lead_time, n_conus)

        # load persistence: discharge at time t (per paper definition)
        pers_date = np.datetime64(init_date)
        pers_str = str(pers_date).replace('-', '')
        pers_path = os.path.join(config.root_glofas_reanalysis, pers_str[:4], pers_str + '.nc')
        if not os.path.isfile(pers_path):
            continue
        pers_ds = xr.open_dataset(pers_path)
        pers = pers_ds['dis24'].values[conus_mask].astype(np.float64)
        pers_ds.close()

        target_files = []
        for t in range(lead_time):
            date = np.datetime64(init_date) + np.timedelta64(t + 1, 'D')
            date_str = str(date).replace('-', '')
            target_path = os.path.join(config.root_glofas_reanalysis, date_str[:4], date_str + '.nc')
            if os.path.isfile(target_path):
                target_files.append(target_path)
            else:
                break

        if len(target_files) != lead_time:
            continue

        target = []
        for tf in target_files:
            ds = xr.open_dataset(tf)
            target.append(ds['dis24'].values)
        target = np.stack(target, axis=0)
        target = target[:, conus_mask]

        for t in range(lead_time):
            p_all = pred[t].astype(np.float64)
            o_all = target[t].astype(np.float64)

            for sub_name, sub_info in subsets.items():
                sm = sub_info['mask']
                thr = sub_info['thresholds']
                accumulate(acc[(sub_name, 'model')], t, p_all[sm], o_all[sm], thr)
                accumulate(acc[(sub_name, 'persistence')], t, pers[sm], o_all[sm], thr)

            # --- per-cell accumulation for --write_nc ---
            if pc is not None:
                for s, p_vec in [('model', p_all), ('persistence', pers)]:
                    d = p_vec - o_all
                    a = pc[s]
                    a['sum_g'][t] += o_all
                    a['sum_gg'][t] += o_all * o_all
                    a['sum_d2'][t] += d * d
                    a['n_valid'][t] += 1
                    pred_ev = p_vec[np.newaxis, :] >= thresholds_conus  # (n_rp, n_conus)
                    tgt_ev = o_all[np.newaxis, :] >= thresholds_conus
                    a['tp'][t] += (pred_ev & tgt_ev).astype(np.int64)
                    a['fp'][t] += (pred_ev & ~tgt_ev).astype(np.int64)
                    a['fn'][t] += (~pred_ev & tgt_ev).astype(np.int64)

        # daily metrics for lead time 1 (on all CONUS points, not per subset)
        p0 = pred[0].astype(np.float64)
        o0 = target[0].astype(np.float64)
        d0 = p0 - o0
        d_rmse = np.sqrt((d0 * d0).mean())
        d_mae = np.abs(d0).mean()
        d_ss_tot = ((o0 - o0.mean()) ** 2).sum()
        d_nse = 1.0 - (d0 * d0).sum() / max(d_ss_tot, 1e-12)
        d_f1s = []
        d_pred_ev = p0[np.newaxis, :] >= thresholds_conus
        d_tgt_ev = o0[np.newaxis, :] >= thresholds_conus
        for k in range(n_rp):
            d_tp = float((d_pred_ev[k] & d_tgt_ev[k]).sum())
            d_fp = float((d_pred_ev[k] & ~d_tgt_ev[k]).sum())
            d_fn = float((~d_pred_ev[k] & d_tgt_ev[k]).sum())
            d_prec = d_tp / max(d_tp + d_fp, 1)
            d_rec = d_tp / max(d_tp + d_fn, 1)
            d_f1s.append(2 * d_prec * d_rec / max(d_prec + d_rec, 1e-12))
        daily_rows.append({
            'date': init_date,
            'rmse': d_rmse, 'mae': d_mae, 'nse': d_nse,
            **{'f1_rp%.1f' % rp: d_f1s[k] for k, rp in enumerate(return_periods)}
        })

        if (f_idx + 1) % 100 == 0:
            log_string(logger, "processed %d / %d files" % (f_idx + 1, len(pred_files)))

    # --- compute final metrics ---
    all_metrics = {}
    n_points_per_subset = {}
    for sub_name in subset_names:
        n_points_per_subset[sub_name] = int(subsets[sub_name]['mask'].sum())
        for s in sources:
            all_metrics[(sub_name, s)] = compute_metrics(acc[(sub_name, s)])

    msg = format_results(all_metrics, subset_names, sources, lead_time, n_points_per_subset)
    log_string(logger, msg)

    # --- save daily metrics to CSV ---
    daily_csv = os.path.join(output_path, 'daily_metrics_conus.csv')
    if daily_rows:
        fieldnames = list(daily_rows[0].keys())
        with open(daily_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(daily_rows)
        log_string(logger, "daily metrics saved to %s (%d rows)" % (daily_csv, len(daily_rows)))

    # --- write per-cell subset metric NCs for --write_nc ---
    if pc is not None and nc_subsets is not None:
        log_string(logger, "computing per-cell metrics and writing NCs ...")
        lead_coords = np.arange(1, lead_time + 1, dtype=np.int64)
        rps = list(return_periods)
        source_names = {'model': 'rivermamba', 'persistence': 'persistence'}

        for s in sources:
            a = pc[s]
            n_v = a['n_valid'].astype(np.float64)
            mean_g = np.divide(a['sum_g'], n_v, out=np.zeros_like(a['sum_g']),
                               where=n_v > 0)
            var_g = a['sum_gg'] - n_v * mean_g * mean_g

            nse_full = np.full((lead_time, n_conus), np.nan, dtype=np.float64)
            ok = (a['n_valid'] >= 30) & (var_g > 1e-9)
            nse_full[ok] = 1.0 - a['sum_d2'][ok] / var_g[ok]

            denom = 2 * a['tp'] + a['fp'] + a['fn']
            f1_full = np.full((lead_time, n_rp, n_conus), np.nan, dtype=np.float64)
            ok_f1 = denom > 0
            f1_full[ok_f1] = (2.0 * a['tp'][ok_f1]) / denom[ok_f1]

            nse_by_subset = {}
            f1_by_subset = {}
            for csv_name, sub_mask in nc_subsets.items():
                nse_by_subset[csv_name] = nse_full[:, sub_mask]
                f1_by_subset[csv_name] = f1_full[:, :, sub_mask]

            out_path = os.path.join(output_path,
                                    f"{source_names[s]}_subset_metrics.nc")
            write_source_nc(out_path, source_names[s], rps, lead_coords,
                            nse_by_subset, f1_by_subset)
            log_string(logger, "wrote %s" % out_path)


if __name__ == '__main__':
    evaluate()
