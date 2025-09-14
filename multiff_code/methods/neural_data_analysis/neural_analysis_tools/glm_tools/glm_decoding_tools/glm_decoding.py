import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from neural_data_analysis.neural_analysis_tools.glm_tools.glm_decoding_tools import glm_decoding_llr
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.stop_glm.glm_fit import stop_glm_fit, cv_stop_glm
from neural_data_analysis.design_kits.design_by_segment import create_design_df
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_aligned_by_event
from data_wrangling import general_utils
from planning_analysis.show_planning.cur_vs_nxt_ff import cvn_from_ref_class

def init_decoding_data(raw_data_folder_path):
    planning_data_by_point_exists_ok = True

    pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(raw_data_folder_path=raw_data_folder_path)
    pn.prep_data_to_analyze_planning(planning_data_by_point_exists_ok=planning_data_by_point_exists_ok)

    pn.rebin_data_in_new_segments(cur_or_nxt='cur', first_or_last='first', time_limit_to_count_sighting=2,
                                    pre_event_window=0, post_event_window=1.5, rebinned_max_x_lag_number=2)

    for col in ['cur_vis', 'nxt_vis', 'cur_in_memory', 'nxt_in_memory']:
        pn.rebinned_y_var[col] = (pn.rebinned_y_var[col] > 0).astype(int)
    return pn


def get_data_for_decoding_vis(rebinned_x_var, rebinned_y_var, dt):
    data = rebinned_y_var.copy()
    trial_ids = data['new_segment']
    design_df, meta0, meta = create_design_df.get_initial_design_df(data, dt, trial_ids)

    # design_df, meta = create_design_df.add_spike_history(
    #     design_df, y, meta0['trial_ids'], dt,
    #     n_basis=4, t_max=0.20, edge='zero',
    #     prefix='spk_hist', style='bjk',
    #     meta=meta
    # )

    df_X = design_df[['speed_z', 'time_since_last_capture',
        'ang_accel_mag_spline:s0', 'ang_accel_mag_spline:s1',
        'ang_accel_mag_spline:s2', 'ang_accel_mag_spline:s3', 'cur_vis', 'nxt_vis']].copy()

    df_X['random_0_or_1'] = np.random.randint(0, 2, len(df_X))

    cluster_cols = [col for col in rebinned_x_var.columns if col.startswith('cluster_')]
    df_Y = rebinned_x_var[cluster_cols]
    df_Y.columns = df_Y.columns.str.replace('cluster_', '').astype(int)
    return df_X, df_Y
        

def glm_decoding_from_fit(cols_to_decode, df_X, df_Y, offset_log, report):

    for decoding_col in cols_to_decode:
        y_vars = df_X[decoding_col].to_numpy()

        # 1) Build params from the report and align to your spike matrix:
        params_df = glm_decoding_llr.params_df_from_coefs_df(report['coefs_df'])     # long → wide
        params_df = params_df.reindex(columns=df_X.columns, fill_value=0.0)
        params_df = glm_decoding_llr.align_params_to_Y(params_df, df_Y)              # row order = df_Y columns

        # 2) Decode on all rows (no CV)
        #    vis_col must match the column name you used for visibility in df_X/params_df.
        llr, p_vis = glm_decoding_llr.decode_from_fitted_glm(
            df_X,
            df_Y,
            offset_log,          # log(dt) per row OR scalar 0.0 if uniform bins
            params_df=params_df,
            vis_col=decoding_col                   # <-- change if your term is named 'visible', etc.
        )

        print('-'*100)
        print(f"=== Decoding from fit: {decoding_col} ===")
        print(f'{decoding_col}: mean =', df_X[decoding_col].mean())

        # If you have ground-truth per-bin labels:
        auc = roc_auc_score(y_vars, llr)               # using raw LLR is fine
        aupr = average_precision_score(y_vars, p_vis)  # PR-AUC often more informative with class imbalance
        print(f"AUC={auc:.3f}, PR-AUC={aupr:.3f}")

        # Pick an operating point (threshold)
        fpr, tpr, thr = roc_curve(y_vars, llr)
        # Example choices:
        # - Youden J (maximizes tpr - fpr)
        j_idx = np.argmax(tpr - fpr)
        thr_llr = thr[j_idx]

        # Hard labels
        y_hat = (llr >= thr_llr).astype(int)



def glm_decoding_cv(cols_to_decode, df_X, df_Y, groups, offset_log):
    for decoding_col in cols_to_decode:
        y_vars = df_X[decoding_col].to_numpy()

        res = glm_decoding_llr.cv_decode_with_glm_report(
            df_X=df_X,                 # design (must include 'any_ff_visible')
            df_Y=df_Y,                 # spikes (T x N), columns are unit IDs
            y=y_vars,               # (T,) 0/1
            groups=groups,             # (T,) session/episode IDs for GroupKFold
            offset_log=offset_log,     # scalar or (T,)
            fit_fn=stop_glm_fit.glm_mini_report,
            fit_kwargs=dict(cov_type='HC1', fast_mle=True, do_inference=False, make_plots=False, show_plots=False),
            bins_2d=None,           # to enable guard-band evaluation
            vis_col=decoding_col,
            n_splits=5,
            standardize=False,         # set True if you want z-scoring of features on train only
            guard=0.05                 # None to disable; else ignores test bins near edges
        )

        print('-'*100)
        print(f"=== CV decoding: {decoding_col} ===")
        print(f"GroupKFold AUC: {res['auc_mean']:.3f} ± {res['auc_std']:.3f} | "
            f"PR-AUC: {res['pr_mean']:.3f} ± {res['pr_std']:.3f} | folds={res['n_splits']}")
        for m in res['fold_metrics']:
            print(f"fold {m['fold']}: AUC={m['auc']:.3f}, PR-AUC={m['pr_auc']:.3f}, n_test={m['n_test']}, kept={m['n_kept']}")



def glm_decoding_permutation_test(cols_to_decode, df_X, df_Y, groups, offset_log, report, print_progress=True):
    for decoding_col in cols_to_decode:
        y_vars = df_X[decoding_col].to_numpy()

        # 1) Build params from the report and align to your spike matrix:
        params_df = glm_decoding_llr.params_df_from_coefs_df(
            report['coefs_df'])     # long → wide
        params_df = params_df.reindex(columns=df_X.columns, fill_value=0.0)
        params_df = glm_decoding_llr.align_params_to_Y(
            params_df, df_Y)              # row order = df_Y columns

        # 2) Decode on all rows (no CV)
        #    vis_col must match the column name you used for visibility in df_X/params_df.
        llr, p_vis = glm_decoding_llr.decode_from_fitted_glm(
            df_X,
            df_Y,
            offset_log,
            params_df=params_df,
            # <-- change if your term is named 'visible', etc.
            vis_col=decoding_col
        )

        print('-'*100)
        

        auc_obs, pval, null = glm_decoding_llr.auc_permutation_test(
            y_vars, p_vis, groups=groups, n_perm=2000,
            progress=print_progress, desc="Permutations"
        )

        # mean_auc, lo, hi, aucs = glm_decoding_llr.auc_block_bootstrap_ci(
        #     y_vars, p_vis, groups=groups, n_boot=5000, rng=1,
        #     progress=True, desc="Bootstraps"
        # )

        print(f"=== Permutation test: {decoding_col} ===")
        # print(f'{decoding_col}: mean =', df_X[decoding_col].mean())
        print(f"Observed AUC        : {auc_obs:.3f}")
        print(f"Permutation p-value : {pval:.4g}  (n={len(null)})")
        # print(f"Bootstrap mean AUC  : {mean_auc:.3f}")
        # print(f"Bootstrap 95% CI    : [{lo:.3f}, {hi:.3f}]  (n={len(aucs)})")

        exceed = int(pval*(1+len(null)) - 1)  # how many null >= obs
        print(f"{decoding_col}: AUC={auc_obs:.3f}, p={pval:.3g}, exceed={exceed}/{len(null)}, "
              f"null μ={null.mean():.3f}±{null.std(ddof=1):.3f}")

        print(null[:10])
