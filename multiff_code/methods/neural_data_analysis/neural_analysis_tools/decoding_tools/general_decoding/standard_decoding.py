import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, average_precision_score

from neural_data_analysis.design_kits.design_by_segment import create_design_df
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_aligned_by_event
from data_wrangling import general_utils


# ---------------------------------------------------------------------
# Data initialization (UNCHANGED from your original script)
# ---------------------------------------------------------------------

def init_decoding_data(raw_data_folder_path):
    planning_data_by_point_exists_ok = True

    pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
        raw_data_folder_path=raw_data_folder_path)

    pn.prep_data_to_analyze_planning(
        planning_data_by_point_exists_ok=planning_data_by_point_exists_ok)

    pn.rebin_data_in_new_segments(
        cur_or_nxt='cur',
        first_or_last='first',
        time_limit_to_count_sighting=2,
        pre_event_window=0,
        post_event_window=1.5,
        rebinned_max_x_lag_number=2
    )

    for col in ['cur_vis', 'nxt_vis', 'cur_in_memory', 'nxt_in_memory']:
        pn.rebinned_y_var[col] = (pn.rebinned_y_var[col] > 0).astype(int)

    return pn


# ---------------------------------------------------------------------
# Design matrix + neural features (UNCHANGED)
# ---------------------------------------------------------------------

def get_data_for_decoding_vis(rebinned_x_var, rebinned_y_var, dt):

    data = rebinned_y_var.copy()
    trial_ids = data['new_segment']

    design_df, meta0, meta = create_design_df.get_initial_design_df(
        data, dt, trial_ids)

    df_X = design_df[
        [
            'speed_z',
            'time_since_last_capture',
            'ang_accel_mag_spline:s0',
            'ang_accel_mag_spline:s1',
            'ang_accel_mag_spline:s2',
            'ang_accel_mag_spline:s3',
            'cur_vis',
            'nxt_vis',
        ]
    ].copy()

    # neural matrix
    cluster_cols = [c for c in rebinned_x_var.columns if c.startswith('cluster_')]
    df_Y = rebinned_x_var[cluster_cols]
    df_Y.columns = df_Y.columns.str.replace('cluster_', '').astype(int)

    return df_X, df_Y


# ---------------------------------------------------------------------
# Core CV decoding routine
# ---------------------------------------------------------------------

def run_population_decoder(
    X_neural,
    y,
    groups,
    decoder='logreg',
    n_splits=5
):
    """
    decoder: 'logreg' | 'lda'
    """

    gkf = GroupKFold(n_splits=n_splits)

    aucs = []
    pr_aucs = []

    for fold, (tr, te) in enumerate(gkf.split(X_neural, y, groups)):

        X_tr, X_te = X_neural[tr], X_neural[te]
        y_tr, y_te = y[tr], y[te]

        # --------------------------------------------------
        # Decoder selection
        # --------------------------------------------------
        if decoder == 'logreg':
            scaler = StandardScaler(with_mean=True, with_std=True)
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

            clf = LogisticRegression(
                penalty='l2',
                solver='lbfgs',
                max_iter=2000,
                class_weight='balanced'
            )

        elif decoder == 'lda':
            clf = LinearDiscriminantAnalysis(solver='svd')

        else:
            raise ValueError(f'Unknown decoder: {decoder}')

        clf.fit(X_tr, y_tr)

        p_te = clf.predict_proba(X_te)[:, 1]

        aucs.append(roc_auc_score(y_te, p_te))
        pr_aucs.append(average_precision_score(y_te, p_te))

    return {
        'auc_mean': np.mean(aucs),
        'auc_std': np.std(aucs),
        'pr_mean': np.mean(pr_aucs),
        'pr_std': np.std(pr_aucs),
        'n_splits': n_splits
    }


# ---------------------------------------------------------------------
# Wrapper: decode multiple task variables
# ---------------------------------------------------------------------

def population_decoding_cv(
    cols_to_decode,
    df_X,
    df_Y,
    groups,
    decoders=('logreg', 'lda'),
    n_splits=5
):

    X_neural = df_Y.to_numpy()

    for decoding_col in cols_to_decode:
        y = df_X[decoding_col].to_numpy()

        print('-' * 100)
        print(f'=== Decoding {decoding_col} from population activity ===')
        print(f'Mean label = {y.mean():.3f}')

        for dec in decoders:
            res = run_population_decoder(
                X_neural=X_neural,
                y=y,
                groups=groups,
                decoder=dec,
                n_splits=n_splits
            )

            print(
                f'{dec:12s} | '
                f'AUC = {res["auc_mean"]:.3f} ± {res["auc_std"]:.3f} | '
                f'PR-AUC = {res["pr_mean"]:.3f} ± {res["pr_std"]:.3f}'
            )


# ---------------------------------------------------------------------
# Permutation test (group-respecting)
# ---------------------------------------------------------------------

def permutation_test_auc(y, p, groups, n_perm=2000, rng=0):

    rng = np.random.default_rng(rng)
    auc_obs = roc_auc_score(y, p)

    null = np.zeros(n_perm)
    for i in range(n_perm):
        y_perm = general_utils.shuffle_within_groups(y, groups, rng)
        null[i] = roc_auc_score(y_perm, p)

    pval = (1 + np.sum(null >= auc_obs)) / (1 + n_perm)
    return auc_obs, pval, null

